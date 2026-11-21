#pragma once
// Qwen3.5-4B text-only inference -- COMPLETE ordinary-DSL executor.
//
// QwenDslEngine is the QwenEngine implementation for the dsl-bf16 / dsl-fp8 /
// dsl-q8 variants. The weight format is derived from the uploaded tensors:
// bf16_packed, fp8_packed or q8_0_packed (native Q8_0 block storage, phase 8a).
// A Q8_0 checkpoint is never re-quantized and never routed through FP8. It drives the frozen 32-layer schedule (8 x [GDN, GDN, GDN, full
// attention]; full-attention blocks 3, 7, ..., 31) entirely through the
// device-free kernel factories in qwen/dsl_ops.h. No Tile, no native shaders.
//
// Per decoder block:
// input RMSNorm (direct pre-added GGUF weight; the HF zero-centered 1+w
// conversion happened offline in the converter)
//   -> full attention (q/k/v projections, fused Q/gate split + QK norm +
//      partial NeoX rotate-half RoPE, sparsely backed per-layer FP32 KV cache, causal
//      GQA prefill / block-cooperative decode with sigmoid output gate,
//      output projection with fused residual epilogue into hidden)
//      OR GDN (packed QKV + z + alpha + beta projections, depthwise conv4 +
//      SiLU with per-layer ring history, QK L2 norm + Q scale, beta/decay
//      gates, FP32 delta-rule recurrent state per layer, gated RMS output
//      norm with direct weight, output projection with fused residual
//      epilogue into hidden)
// -> post-attention RMSNorm (direct pre-added weight)
// -> decode: fused ffn_gate/ffn_up projections + SwiGLU -> ffn_down
//    (prefill retains separate projections and SwiGLU)
//   -> residual add.
// Final: output_norm (direct pre-added weight) on the LAST position only, then the tied
// token_embd LM head for that single position -> FP32 logits [vocab].
//
// State: one sparse virtual allocation with an on-demand FP32 K and V region
// per full-attention layer ([max_context, kv_heads, head_dim] each), one FP32 recurrent state
// ([v_heads, head_dim, head_dim]) and one conv ring ([qkv_channels,
// kernel-1]) per GDN layer. reset() clears logical state, releases KV physical
// residency, and zeroes the fixed GDN state/ring buffers.
// Prefill processes the prompt in chunks of options.prefill_chunk rows. GDN
// uses the FP32 WY representation + UT transform over serial 64-token chunks
// by default; options.dsl_gdn_wy_ut=false selects the exact per-token recurrence
// for differential debugging. K/V are appended at the proper absolute positions.
// decode() consumes one token through the four-way cooperative recurrent operator.
//
// Weight storage is validated to be UNIFORMLY bf16_packed, fp8_packed or
// q8_0_packed for every packed matrix (dense F32 tensors stay F32 in all
// variants); a mixed or unsupported storage is a recoverable failure, never a
// substitution. For q8_0 the per-row scale group is the GGUF block (32) instead
// of FP8's 64, and every projection path uses the same kernels as fp8 with the
// int8 decode substituted in-kernel (embedding, decode GEMV with the
// 4-accumulator specialization and the 128-bit uint4 realization, reference and
// MT prefill GEMM, fused FFN decode, fused GDN gate GEMV). The CUDA tensor-core
// native GEMM does not implement Q8_0 and is disabled for that format (the MT
// DSL GEMM stays in charge), and the static decode graph is format-agnostic:
// its captured body only contains these kernels, so dsl-q8 uses it unchanged.
//
// All kernels are compiled ONCE at construction (shapes are static); compute
// and transfer commands for each prefill/decode step use ONE QwenCommandBatch
// (a CommandList wrapper) per public call and synchronize only when logits are
// downloaded. Sparse tile maps are separate ordered stream events issued before
// dependent kernels; normal growth adds no host synchronization. With options.benchmark=true the batch instead
// commits + synchronizes after EVERY annotated command and records host-wall
// samples into the engine-owned QwenBenchmarkReport exposed through
// benchmark_report(). Greedy CUDA also supports bounded in-flight decode: device
// argmax feeds the next token and only chunk downloads synchronize. BF16 and FP8
// CUDA capture the device-only decode step as a STATIC graph (opt_plan/03
// section 2.3): every per-token scalar (absolute position, KV length, greedy
// history slot) lives in a device state buffer, so the body is recorded once and
// later tokens are a bare launch with zero graph updates. Token uploads and
// sparse KV page maps remain ordered pre-graph events, position ramps are derived
// from the device position without host staging, and no map command is ever
// captured. MTP, benchmark mode, the host-logits (temperature) decode path and
// non-CUDA devices retain ordinary command lists.
// Recoverable Status everywhere; no exceptions, no RTTI.

#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

#include "qwen/benchmark.h"
#include "qwen/cuda_graph_while.h"
#include "qwen/cuda_tile_gemm.h"
#include "qwen/dsl_ops.h"
#include "qwen/native_attn.h"
#include "qwen/native_gemm.h"
#include "qwen/engine.h"
#include "qwen/kv_cache_allocator.h"
#include "qwen/model.h"
#include "qwen/pinned_ring.h"
#include "qwen/weights.h"

namespace luisa::qwen {

class QwenDslEngine final : public QwenEngine {
    // The opt-in NextN executor reuses the exact target forward/state schedule
    // for verify, snapshot and accepted-prefix replay. It never mutates weight
    // ownership or substitutes another kernel family.
    friend class QwenMtpExecutor;

public:
    // Allocator-matched deleter for the factory handle. The engine is
    // allocated with Luisa's allocator (mimalloc-backed); luisa::unique_ptr's
    // default eastl::default_delete frees through EASTL's default allocator,
    // and mixing the two heaps corrupts memory (segfault on destruction).
    // This deleter destroys virtually and frees with the SAME allocator.
    using Deleter = QwenEngineDeleter;

    // Factory used by QwenEngine::create for QwenKernelKind::dsl. Validates
    // options/weights, allocates state + workspace, compiles every kernel
    // variant once. `weights` is borrowed and must outlive the engine.
    [[nodiscard]] static Result<luisa::unique_ptr<QwenEngine, Deleter>> create(
        compute::Device &device, compute::Stream &stream,
        const Qwen35Model &model, const QwenDeviceWeights &weights,
        const QwenEngineOptions &options);

    // Drain before ANY GPU child member is destroyed. In particular this
    // covers bounded in-flight device-greedy decode at engine teardown.
    ~QwenDslEngine() noexcept override;

    [[nodiscard]] Status reset(compute::Stream &stream) override;
    [[nodiscard]] Status prefill(compute::Stream &stream,
                                 luisa::span<const uint32_t> prompt_tokens,
                                 luisa::vector<float> &logits_out) override;
    [[nodiscard]] Status decode(compute::Stream &stream, uint32_t input_token,
                                luisa::vector<float> &logits_out) override;
    [[nodiscard]] bool supports_device_greedy_decode() const noexcept override { return true; }
    [[nodiscard]] Status decode_device_greedy(compute::Stream &stream,
                                              const uint32_t *input_token,
                                              uint32_t history_slot) override;
    [[nodiscard]] Status download_device_greedy(compute::Stream &stream,
                                                uint32_t first_slot,
                                                luisa::span<uint32_t> tokens_out) override;
    // Chunked in-flight decode (opt_plan/09 N4.2). ONE conditional WHILE graph
    // launch executes 1..count tokens and stops on the device (EOS / invalid
    // selection / context limit / budget); `result.executed` is read back from the
    // device, so an early exit can never look like unexecuted tokens ran. When the
    // WHILE is not selected (not requested, capability or probe refusal, capture
    // failure, non-CUDA) the SAME chunk is served as per-token steps and reported
    // with `result.fallback`, so the produced ids are identical either way.
    [[nodiscard]] Status decode_device_greedy_chunk(compute::Stream &stream,
                                                    const uint32_t *input_token,
                                                    uint32_t history_slot,
                                                    uint32_t count,
                                                    QwenChunkResult &result) override;
    // Calibration-only mode (opt_plan/05 section 3 item 2). Requires
    // QwenEngineOptions::calibration_only at create(): the fused decode kernels
    // hide the ffn_down input activation, so the reference route is used instead.
    [[nodiscard]] Status begin_activation_calibration(compute::Stream &stream) override;
    [[nodiscard]] Result<QwenCalibrationReport> finish_activation_calibration(
        compute::Stream &stream) override;
    [[nodiscard]] QwenEngineInfo info() const noexcept override;
    // Nonnull exactly when options.benchmark was true; latest prefill/decode
    // report (reset samples of a prefill are included when recorded).
    [[nodiscard]] const QwenBenchmarkReport *benchmark_report() const noexcept override;

private:
    QwenDslEngine() noexcept = default;

      // ---- offline merged input projections (opt_plan/03 section 3) -----------
    // `_merged_decode` is the SELECTED state: QwenEngineOptions::dsl_merged_decode
    // was requested, the uploaded weight set really carries merged matrices for every
    // layer (a pack-only weight set), the descriptors matched the frozen geometry at
    // create AND the merged kernels compiled. `_merged_resolved` says the weight set
    // carried a pack at all, so info() can distinguish "no pack" from "pack but the
    // merged route was refused".
    bool _merged_decode{false};
    bool _merged_available{false};
    luisa::string _merged_pack_name;
    luisa::string _merged_pack_format;
    uint64_t _merged_matrices{0};
    uint64_t _merged_segments{0};
    uint32_t _merged_launches_saved{0};
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<uint32_t>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                    float> _gemv_merged_gdn;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<uint32_t>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, uint32_t, uint32_t, uint32_t, uint32_t> _gemv_merged_attn;
    // ---- static geometry (frozen at construction) -------------------------
      dsl::QwenDslConfig _cfg{};
      uint32_t _layers{0};
      uint32_t _max_context{0};
      // Workspace row capacity: the fixed prefill chunk, or prefill_chunk_max
      // when adaptive chunking is enabled (the selected chunk is always <= it).
      uint32_t _prefill_chunk{0};
      // Adaptive outer chunking (opt_plan/02 section 4).
      bool _prefill_chunk_adaptive{false};
      uint32_t _prefill_chunk_max{0};
      uint32_t _prefill_chunk_backend_limit{0};
      uint32_t _prefill_chunk_selected{0};// chunk the last prefill used
      uint32_t _prefill_chunks{0};        // chunks the last prefill split into
      // KV sparse allocation unit requested at create (0 = allocator default).
      // Geometry, not session state: alloc_state() freezes it into the allocator.
      uint64_t _kv_page_bytes{0};
    // Resolved KV dtype/layout contract (qwen/kv_cache_layout.h). Set once at
    // create from QwenEngineOptions::kv_dtype and read by every attention
    // dispatch; `_kv_packed_append` says the packed append kernel is in the path.
    dsl::QwenKvStorage _kv_storage{};
    bool _kv_packed_append{false};
    // fp8_e4m3: the ONLINE-scale append kernel (one CTA per (64-position block,
    // kv_head)) owns the cache writes instead of the packed-word one. Mutually
    // exclusive with it; both are "append outside the prepare kernel".
    bool _kv_fp8_append{false};
    dsl::QwenWeightFormat _format{dsl::QwenWeightFormat::bf16};
    WeightStorage _storage{WeightStorage::bf16_packed};
    uint32_t _warp_size{0};
    // Scale group along K of the DECODE weights (fp8 64, q8_0 32, w4 128 from the
    // pack manifest). Every packed kernel takes `group` as a runtime argument, so
    // a per-format value is all the dispatch needs.
    uint32_t _decode_group{0};
    // Prefill/emit second copy (opt_plan/05 Q2 plan option (c)): W4 decode +
    // FP8 prefill and FP8 LM head. `_dual_prefill` is true only when the uploaded
    // weights actually carry an FP8 partner ("<name>.fp8") for EVERY matrix the
    // prefill route touches; it is resolved once at create and never mid-request.
    bool _dual_prefill{false};
    dsl::QwenWeightFormat _prefill_format{dsl::QwenWeightFormat::fp8_e4m3};
    WeightStorage _prefill_storage{WeightStorage::fp8_packed};
    uint32_t _prefill_group{0};
    // Explicit opt-in for the measured A/B arm (plan option (b)): run prefill
    // through the DECODE (W4) tensors with in-kernel dequantization when no FP8
    // partner exists. Never the default: without it a W4 session without an FP8
    // copy fails closed at create.
    bool _w4_prefill_in_kernel{false};
    bool _w4_lm_head_fp8{true};

    // ---- opt-in command profiling ------------------------------------------
    bool _benchmark{false};
    QwenBenchmarkReport _report;
    // Cached annotation strings (QwenCommandBatch::annotate borrows the view).
    luisa::vector<luisa::string> _reset_names;

    // ---- borrowed weights / retained device --------------------------------
    const QwenDeviceWeights *_weights{nullptr};
    // Declared before every buffer/shader so reverse destruction releases all
    // GPU children first. Retaining the handle also makes device-first caller
    // teardown safe, matching QwenTileEngine's lifetime contract.
    compute::Device _device;
    compute::Stream _stream;

    // ---- per-layer resolved weight pointers (into *_weights) --------------
    struct LayerWeights {
        // Plan option (c) second copy: the FP8 partners of every matrix the PREFILL
        // route touches (rows > 1). All null unless _dual_prefill.
        struct Prefill {
            const QwenDeviceTensor *ffn_gate{nullptr};
            const QwenDeviceTensor *ffn_up{nullptr};
            const QwenDeviceTensor *ffn_down{nullptr};
            const QwenDeviceTensor *attn_q{nullptr};
            const QwenDeviceTensor *attn_k{nullptr};
            const QwenDeviceTensor *attn_v{nullptr};
            const QwenDeviceTensor *attn_output{nullptr};
            const QwenDeviceTensor *gdn_qkv{nullptr};
            const QwenDeviceTensor *gdn_gate{nullptr};
            const QwenDeviceTensor *gdn_alpha{nullptr};
            const QwenDeviceTensor *gdn_beta{nullptr};
            const QwenDeviceTensor *gdn_out{nullptr};
        };
        Prefill pf;
        // shared
        const QwenDeviceTensor *attn_norm{nullptr};
        const QwenDeviceTensor *post_norm{nullptr};
        const QwenDeviceTensor *ffn_gate{nullptr};
        const QwenDeviceTensor *ffn_up{nullptr};
        const QwenDeviceTensor *ffn_down{nullptr};
        // full attention
        const QwenDeviceTensor *attn_q{nullptr};
        const QwenDeviceTensor *attn_k{nullptr};
        const QwenDeviceTensor *attn_v{nullptr};
        const QwenDeviceTensor *attn_output{nullptr};
        const QwenDeviceTensor *attn_q_norm{nullptr};
        const QwenDeviceTensor *attn_k_norm{nullptr};
        // GDN
        const QwenDeviceTensor *gdn_qkv{nullptr};
        const QwenDeviceTensor *gdn_gate{nullptr};
        const QwenDeviceTensor *gdn_alpha{nullptr};
        const QwenDeviceTensor *gdn_beta{nullptr};
        const QwenDeviceTensor *gdn_out{nullptr};
        const QwenDeviceTensor *gdn_conv1d{nullptr};
        const QwenDeviceTensor *gdn_a{nullptr};
        const QwenDeviceTensor *gdn_dt_bias{nullptr};
        const QwenDeviceTensor *gdn_norm{nullptr};
      bool full_attention{false};
      // OFFLINE merged input projections (opt_plan/03 section 3 + D5). Nonnull
      // exactly when this layer's input-projection weights came from a merged pack:
      // `merged_gdn` covers [qkv | z | alpha | beta] and `merged_attn` covers
      // [q | k | v]. The pointers are the merge PARENTS; the per-role pointers above
      // already point at the segment aliases (sub-views), so every other consumer is
      // unchanged. The row boundaries are validated against the frozen geometry once,
      // at create, and are the constants the merged decode kernel routes on.
      const QwenDeviceTensor *merged_gdn{nullptr};
      const QwenDeviceTensor *merged_attn{nullptr};
      uint32_t merged_qkv_rows{0};
      uint32_t merged_z_rows{0};
      uint32_t merged_gate_heads{0};
      uint32_t merged_q_rows{0};
      uint32_t merged_kv_rows{0};
      // Stable per-layer command annotations (owned; views stay valid).
      luisa::string p_input_norm, p_post_norm, p_swiglu;
        luisa::string p_attn_prepare, p_attn_core, p_attn_split, p_attn_combine;
              luisa::string p_conv, p_gates, p_qk_norm, p_recurrent, p_chunk, p_gated_norm;
        // Fused one-CTA recurrent + gated RMSNorm launch (opt_plan/03 section 5).
        luisa::string p_recurrent_norm;
      luisa::string p_merged_proj;
  };
    luisa::vector<LayerWeights> _layer_weights;
    const QwenDeviceTensor *_token_embd{nullptr};
    // The LM head matrix: `_token_embd` unless a dual-copy session provides the
    // 8-bit partner ("<token_embd>.fp8"), which is the quality-preserving choice
    // the plan requires for the most sensitive matrix.
    const QwenDeviceTensor *_lm_head{nullptr};
    const QwenDeviceTensor *_output_norm{nullptr};

    // ---- persistent device state ------------------------------------------
    // Full attention: one sparse virtual buffer, with stable dense-addressing
    // K/V views per full layer and physical tiles committed on demand.
    QwenKvCacheAllocator _kv_cache;
    // GDN: one recurrent state and one conv ring per GDN layer.
    luisa::vector<compute::Buffer<float>> _gdn_states;// [gdn_layers]
    luisa::vector<compute::Buffer<float>> _gdn_rings;
    // Map layer index -> cache/state slot (UINT32_MAX when not applicable).
    luisa::vector<uint32_t> _full_slot;
    luisa::vector<uint32_t> _gdn_slot;

    // ---- activation workspace (reused across layers/chunks) ---------------
    compute::Buffer<uint32_t> _ids;       // [chunk] token ids
    compute::Buffer<uint32_t> _ids_host;  // [chunk] device-side staging for ids
    // Persistent host staging for scalar uploads. copy_from(span) borrows the
    // host pointer until the command encoder commits, so this storage must
    // outlive queued commands; ids are then copied with a device-side kernel.
    // Sized to max_context: a prefill records ALL chunks into ONE command
    // list, so every chunk keeps a DISTINCT span [_staged, _staged+rows) that
    // is never rewritten before the single commit. Positions need no staging:
    // the engine specialization derives its contiguous ramp from pos0 + row.
    luisa::vector<uint32_t> _host_ids;
    uint32_t _staged{0};// next unused staging slot within the current call
    compute::Buffer<float> _hidden;       // [chunk, hidden] current activation
    compute::Buffer<float> _normed;       // [chunk, hidden] norm output
    compute::Buffer<float> _residual;     // [chunk, hidden] residual stream
    compute::Buffer<float> _proj_a;       // [chunk, max_proj_n] projection scratch
    compute::Buffer<float> _proj_b;       // [chunk, max_proj_n] projection scratch
    compute::Buffer<float> _proj_c;       // [chunk, max_proj_n] projection scratch
    compute::Buffer<float> _ffn_gate;     // [chunk, ffn]
    compute::Buffer<float> _ffn_up;       // [chunk, ffn]
    compute::Buffer<float> _ffn_act;      // [chunk, ffn] swiglu output
    // attention scratch
    compute::Buffer<float> _attn_q;       // [chunk, q_heads, hd]
    compute::Buffer<float> _attn_gate;    // [chunk, q_heads, hd]
    compute::Buffer<float> _attn_k;       // [chunk, kv_heads, hd]
    compute::Buffer<float> _attn_v;       // [chunk, kv_heads, hd]
    compute::Buffer<float> _attn_out;     // [chunk, q_heads, hd]
    compute::Buffer<float> _attn_proj_q;  // dedicated q proj scratch
    compute::Buffer<float> _attn_proj_k;  // dedicated k proj scratch
    compute::Buffer<float> _attn_proj_v;  // dedicated v proj scratch
    // GDN scratch
    compute::Buffer<float> _gdn_conv;     // [chunk, qkv_channels]
    compute::Buffer<float> _gdn_q;        // [chunk, qk_heads, hd]
    compute::Buffer<float> _gdn_k;        // [chunk, qk_heads, hd]
    compute::Buffer<float> _gdn_beta;     // [chunk, v_heads]
    compute::Buffer<float> _gdn_g;        // [chunk, v_heads]
    compute::Buffer<float> _gdn_alpha;    // [chunk, v_heads] raw a projection
    compute::Buffer<float> _gdn_out;      // [chunk, v_heads, hd]
    compute::Buffer<float> _gdn_w;        // WY scratch [chunk, v_heads, hd]
    compute::Buffer<float> _gdn_vnew;     // unscaled WY v_new [chunk, v_heads, hd]
    compute::Buffer<float> _gdn_normed;   // [chunk, v_heads, hd]
    // LM head
    compute::Buffer<float> _last_normed;  // [hidden] last-position norm
    compute::Buffer<float> _logits;       // [vocab]
    compute::Buffer<uint32_t> _greedy_history;// [max_context] device-selected tokens
    compute::Buffer<float> _greedy_values;   // [256] partial argmax candidates
    compute::Buffer<uint32_t> _greedy_indices;// [256]
    compute::Buffer<uint32_t> _greedy_invalid;// [256] NaN flags
    bool _greedy_next_valid{false};   // previous in-flight step produced _ids[0]
    // Device-resident decode state (opt_plan/03 section 2.3): [position,
    // kv_len, history_slot, eos]. See qwen/dsl_ops.h for the layout contract.
    // The static CUDA graph reads it instead of host kernel scalars, which is
    // what makes one captured decode body valid for every token.
    compute::Buffer<uint32_t> _dev_state;
    // Persistent host staging for the state uploads. copy_from(span) borrows the
    // pointer until the encoder commits, so the upload is recorded in its own
    // one-shot batch and flushed before this storage can be reused.
    luisa::vector<uint32_t> _host_state;
    // Host mirror of _dev_state. UINT32_MAX/_dev_state_synced == false means the
    // device copy is stale (reset / prefill / first capture / caller slot drift)
    // and must be re-uploaded before the next graph launch.
    uint32_t _dev_state_position{UINT32_MAX};
    uint32_t _dev_state_kv_len{0};
    uint32_t _dev_state_slot{0};
    bool _dev_state_synced{false};
    compute::Buffer<float> _dummy_f;      // 1-element dummy for unused args
    compute::Buffer<uint32_t> _dummy_u;

    // ---- compiled shaders (one per variant; shapes are static) ------------
    // Kernel signatures come from dsl_ops.h; the Shader args mirror them.
    compute::Shader<2, compute::Buffer<uint32_t>, compute::Buffer<uint32_t>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _embedding;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, float> _rmsnorm_plain;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, float> _rmsnorm_snapshot;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<uint32_t>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t> _gemv;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<uint32_t>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t> _gemv_residual;
    compute::Shader<2, compute::Buffer<float>, compute::Buffer<uint32_t>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _gemm;
    compute::Shader<2, compute::Buffer<float>, compute::Buffer<uint32_t>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _gemm_residual;
    // Weight-reusing prefill GEMM (opt_plan/02 section 3): same ten arguments
    // and the same 2D dispatch shape as _gemm/_gemm_residual, one warp per
    // output column and an MT-row m tile. Only used for rows > 1.
    compute::Shader<2, compute::Buffer<float>, compute::Buffer<uint32_t>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _gemm_mt;
    compute::Shader<2, compute::Buffer<float>, compute::Buffer<uint32_t>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _gemm_mt_residual;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, uint32_t> _swiglu;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<uint32_t>,
                    compute::Buffer<float>, compute::Buffer<uint32_t>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t> _ffn_decode;
    compute::Shader<1, compute::Buffer<float>, uint32_t> _zero_fill;
    compute::Shader<1, compute::Buffer<uint32_t>, compute::Buffer<uint32_t>, uint32_t> _copy_u;
    // Activation second-moment accumulator used ONLY while the calibration mode is
    // active (qwen/w4_importance.h); compiled lazily by begin_activation_calibration.
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, uint32_t,
                    uint32_t> _calibration_acc;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<uint32_t>, compute::Buffer<uint32_t>,
                    uint32_t> _greedy_argmax_partial;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<uint32_t>,
                    compute::Buffer<uint32_t>, compute::Buffer<uint32_t>,
                    compute::Buffer<uint32_t>, uint32_t, uint32_t> _greedy_argmax_finish;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<uint32_t>,
                    compute::Buffer<uint32_t>, compute::Buffer<uint32_t>,
                    compute::Buffer<uint32_t>, compute::Buffer<uint32_t>,
                    uint32_t, uint32_t> _greedy_argmax_finish_dev;
    // Conditional-WHILE realization of the same finish kernel (opt_plan/09 N4.2).
    // Compiled LAZILY, at the first chunk capture, so a session that never asks for
    // the chunked graph never pays for the extra shader.
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<uint32_t>,
                    compute::Buffer<uint32_t>, compute::Buffer<uint32_t>,
                    compute::Buffer<uint32_t>, compute::Buffer<uint32_t>,
                    uint32_t, uint32_t, uint64_t, uint32_t> _greedy_argmax_finish_while;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<int32_t>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _attn_prepare;
    // Device-state realization (static graph body): same op, position from
    // _dev_state. Compiled unconditionally so the graph is a pure dispatch
    // decision; the reference kernel above stays for prefill / graph-off / MTP.
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<int32_t>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<uint32_t>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float,
                    uint32_t, uint32_t, uint32_t, uint32_t> _attn_prepare_dev;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                  compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t, uint32_t,
                  uint32_t> _attn_prefill;
    // Packed (BF16) KV append (opt_plan/05 Q3): one thread per 32-bit word, both
    // codes in the same thread. Compiled only for a packed KV dtype; the FP32 arm
    // keeps its previous launch list exactly. `_dev` reads the position from
    // _dev_state instead of a host scalar (static CUDA graph body).
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<uint32_t>, compute::Buffer<uint32_t>, compute::Buffer<uint32_t>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _attn_kv_append;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<uint32_t>, compute::Buffer<uint32_t>, compute::Buffer<uint32_t>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _attn_kv_append_dev;
    // FP8 E4M3 KV append with ONLINE per-(head, 64-position block) scales
    // (opt_plan/05 Q3). One block per (touched 64-position block, kv_head); it
    // owns the block maximum, the block scale and any rescale of earlier codes,
    // so this kernel -- not the attention prepare -- writes an fp8 cache.
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<uint32_t>, compute::Buffer<uint32_t>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<uint32_t>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _attn_kv_append_fp8;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<uint32_t>, compute::Buffer<uint32_t>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<uint32_t>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _attn_kv_append_fp8_dev;
    // Model-path tiled prefill attention (opt_plan/02 section 2).
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, float, uint32_t, uint32_t, uint32_t> _attn_prefill_tiled;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, float, uint32_t, uint32_t> _attn_decode;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<uint32_t>,
                    uint32_t, uint32_t, uint32_t, float, uint32_t> _attn_decode_dev;

    // Split-KV decode attention (opt_plan/03 section 4.2): partial kernel (one
    // block per (kv_head, key_split)) + combine kernel (one block per q_head,
    // applies the sigmoid gate). Host-scalar and device-state realizations, both
    // compiled with the SAME compile-time threshold as the legacy pair above.
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, float, uint32_t> _attn_decode_split;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<uint32_t>,
                    float, uint32_t> _attn_decode_split_dev;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t> _attn_decode_combine;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<uint32_t>,
                    uint32_t> _attn_decode_combine_dev;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, uint32_t, uint32_t, uint32_t, uint32_t> _gdn_conv_sh;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<uint32_t>, uint32_t, uint32_t, uint32_t> _gdn_conv_dev;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, float> _gdn_gates_sh;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<uint32_t>,
                    compute::Buffer<float>, compute::Buffer<uint32_t>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, uint32_t, uint32_t, uint32_t, float> _gdn_decode_gates_sh;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, float, float, uint32_t> _gdn_qk_norm_sh;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _gdn_recurrent_sh;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _gdn_decode4_sh;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _gdn_chunk_sh;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, uint32_t, uint32_t, uint32_t, float> _gdn_gated_norm_sh;
 // Fused one-CTA-per-head recurrence + gated output RMSNorm (opt_plan/03 section 5):
 // q,k,conv_out,beta,g,state,z,norm_w,out + tokens,v_heads,head_dim,qk_heads,v_offset,eps.
 compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
 compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
 compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
 uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float> _gdn_recurrent_norm_sh;

    // ---- session state -----------------------------------------------------
    uint32_t _position{0};// number of tokens consumed (next absolute position)
    bool _has_session{false};
    bool _gdn_wy_ut{true};
    // GDN state-column kernels compiled with the compile-time 128-channel head
    // specialization (opt_plan/09 V4); see QwenEngineInfo.gdn_state_specialized.
    // Register-cached RMSNorm specialization (opt_plan/09 N3.1).
 bool _rmsnorm_reg{false};
 bool _gdn_state_specialized{false};
 // FUSED one-CTA-per-head GDN recurrence + gated output RMSNorm (opt_plan/03
 // section 5): the SELECTED state (option AND dsl_decode_specialized AND the model
 // geometry AND a successful compile). `_gdn_recurrent_norm_reason` records why it is
 // not in force, so "not requested" stays distinguishable from "requested but
 // refused" in a log and in QwenEngineInfo.
 bool _gdn_recurrent_norm{false};
 luisa::string _gdn_recurrent_norm_reason;
// Fused decode conv+SiLU+QK-norm (opt_plan/09 N3.3): _gdn_decode_fused is the
// SELECTED state (option AND model geometry AND successful compile); the
// compiled shaders exist only when the geometry matched.
bool _gdn_decode_fused{false};
compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                uint32_t, uint32_t, uint32_t, uint32_t, float, float, uint32_t> _gdn_decode_fused_sh;
compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                compute::Buffer<uint32_t>, uint32_t, uint32_t, uint32_t, uint32_t, float, float> _gdn_decode_fused_dev;
        // Vectorized 128-bit decode realizations (opt_plan/09 N2). `_decode_packed`
    // is the SELECTED state (option AND geometry AND successful compile); the
    // extra shaders exist only when it was compiled.
    bool _decode_packed{false};
    bool _decode_packed_fp8{false};
    compute::Shader<1, compute::Buffer<float4>, compute::Buffer<uint4>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t> _gemv_packed;
    compute::Shader<1, compute::Buffer<float4>, compute::Buffer<uint4>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t> _gemv_packed_residual;
    compute::Shader<1, compute::Buffer<float4>, compute::Buffer<uint4>, compute::Buffer<float>,
                    compute::Buffer<uint4>, compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t> _ffn_decode_packed;
    compute::Shader<1, compute::Buffer<float4>, compute::Buffer<uint4>, compute::Buffer<float>,
                    compute::Buffer<uint4>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, float, uint32_t> _gdn_decode_gates_packed;
    // Plan option (c) SECOND shader set: the same prefill GEMM family compiled for
    // the 8-bit PREFILL format (FP8), plus the 8-bit LM head GEMV. Compiled only
    // when the uploaded weights carry the dual copy, so a single-copy session pays
    // nothing. Shapes are static and `group` is a runtime argument, so one set per
    // format is enough.
    compute::Shader<2, compute::Buffer<float>, compute::Buffer<uint32_t>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _gemm_pf;
    compute::Shader<2, compute::Buffer<float>, compute::Buffer<uint32_t>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _gemm_pf_residual;
    compute::Shader<2, compute::Buffer<float>, compute::Buffer<uint32_t>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _gemm_mt_pf;
    compute::Shader<2, compute::Buffer<float>, compute::Buffer<uint32_t>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _gemm_mt_pf_residual;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<uint32_t>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t> _gemv_lm;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<uint32_t>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t> _gemv_lm_residual;
    bool _prefill_gemm_mt_pf{false};
    // Model-specialized decode attention prepare (opt_plan/09 N3.2): head_dim 256,
    // rope_dim 64, 16 q heads and 4 kv heads as compile-time constants.
    bool _attn_prepare_model{false};
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, float, float, uint32_t, uint32_t> _attn_prepare_model_sh;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<uint32_t>,
                    uint32_t, float, float, uint32_t> _attn_prepare_model_dev_sh;
    // Per-prompt GDN prefill selection (opt_plan/09 V4 follow-up).
    bool _gdn_auto{false};
    // Selection actually used for the CURRENT prefill (one kernel for every chunk of a
    // prompt, so the state stays on one numerical path).
    bool _gdn_wy_ut_active{true};
    uint32_t _gdn_auto_threshold{0};
    bool _decode_specialized{true};
    // Weight-reusing prefill GEMM selection (opt_plan/02 section 3). Compiled
    // only when the device pins a 32-lane warp; rows == 1 (decode) never uses it.
    bool _prefill_gemm_mt{false};
    dsl::QwenGemmMtConfig _gemm_mt_cfg{};
    // CUDA tensor-core prefill GEMM (opt_plan/09 N1a): a NativeShaderExt
    // CUDA_NVRTC `mma.sync` kernel, JIT-compiled at create, dispatched as a native
    // command for prefill rows >= dsl::qwen_native_gemm_min_rows. `valid()` false
    // means the whole path is off and the DSL GEMM handles every row count.
    dsl::QwenNativeGemm _native_gemm;
    uint64_t _native_gemm_launches{0u};
    // EXPERIMENTAL CUDA native-Tile typed-FP8 prefill GEMM (QwenEngineOptions::
    // cuda_tile_fp8_prefill, strictly opt-in and never auto-selected). Owns the lazily
    // compiled per-static-shape `tile::Shader` instances, so it must be reset before
    // the Device dies (see ~QwenDslEngine). `valid()` false means the whole path is
    // off - either it was not requested (then nothing was probed or compiled) or
    // create() failed, which FAILS engine creation instead of falling back.
    // `_cuda_tile_gemm_launches` counts dispatches so a report can prove the Tile
    // realization engaged instead of assuming it.
    dsl::QwenCudaTileGemm _cuda_tile_gemm;
    uint64_t _cuda_tile_gemm_launches{0u};
    // CUDA tensor-core FlashAttention-2 prefill attention (opt_plan/09 N1b): a
    // NativeShaderExt CUDA_NVRTC kernel, JIT-compiled once at create, dispatched as a
    // native command for every prefill chunk while it is valid. `valid()` false means
    // the DSL tiled (or reference) kernel owns prefill attention for the session.
    dsl::QwenNativeAttn _native_attn;
    uint64_t _native_attn_launches{0u};
    // Tiled prefill attention selection (compiled only when the device can host
    // it and options.dsl_tiled_prefill_attn is set; see select_... in dsl_ops.h).
    bool _attn_tiled{false};
    dsl::QwenAttnPrefillTiledConfig _attn_tiled_cfg{};

    // Split-KV decode attention with GQA dedup (opt_plan/03 section 4.2).
    // `_attn_split` is the SELECTED state (option AND device-capable AND the model
    // geometry matches); `_attn_split_cfg.threshold` is baked into every guarded
    // kernel of the family so the legacy/split boundary is a compile-time constant on
    // both sides and can never leave a kv_len served by neither or both.
    bool _attn_split{false};
    // Threshold the legacy pair is compiled with (0 = unguarded). Set from the option
    // BEFORE compile_all, because _attn_decode/_attn_decode_dev are compiled there.
    uint32_t _attn_split_threshold{0};
    dsl::QwenAttnDecodeSplitConfig _attn_split_cfg{};
    // Fixed partial workspace [max_splits, q_heads, head_dim] + [max_splits, q_heads, 2].
    compute::Buffer<float> _attn_partial_acc;
    compute::Buffer<float> _attn_partial_ml;
    compute::CudaGraphExt *_cuda_graph_ext{nullptr};// non-owning; device extension
    uint64_t _cuda_graph_steps{0};
    uint64_t _cuda_graph_updates{0};
    // The static decode graph is captured ONCE (_cuda_graph_steps == 1 after the
    // first launch) and every later token is a bare launch: no whole-graph
    // update, no per-node update, no DAG rebuild. A capture/instantiate failure
    // is reported once and never falls back to executing a consumed list.
    bool _cuda_graph_capture_failed{false};
    // ---- conditional WHILE decode chunk (opt_plan/09 N4.2) -----------------
    // `_while_requested` is the option; `_while_selected` is the resolved state
    // (option AND conditional-node capability AND a passed instantiate/execute
    // probe AND a successful capture). `_while_reason` carries the exact
    // selection/refusal text so a log never has to guess, and `_while_probe` the
    // measured probe numbers. The per-token static graph above remains the default
    // and the fallback: `_while_fallbacks` counts chunk calls it served.
    bool _while_requested{false};
    uint32_t _while_chunk{0};
    uint32_t _while_eos{0};
    bool _while_selected{false};
    bool _while_capture_failed{false};
    luisa::string _while_reason;
    QwenCudaWhileProbe _while_probe;
    uint64_t _while_launches{0};
    uint64_t _while_tokens{0};
    uint64_t _while_early_exits{0};
    uint64_t _while_fallbacks{0};
    // ---- pinned download staging (opt_plan/03 section 6, opt_plan/09 N4.3) --
    compute::PinnedMemoryExt *_pinned_ext{nullptr};// non-owning; device extension
    QwenPinnedSelection _pinned_selection;
    QwenPinnedStats _pinned_stats;
    QwenPinnedRing _pinned;
    // ---- PDL report (opt_plan/03 section 2.5) -------------------------------
    QwenPdlStatus _pdl;
    // Device-state readback of the last WHILE launch (position/kv_len/slot/eos/
    // remaining). The host mirror is derived from it, so an early device exit is
    // visible instead of assumed.
    struct DevStateReadback {
        uint32_t position{0};
        uint32_t kv_len{0};
        uint32_t slot{0};
        uint32_t eos{0};
        uint32_t remaining{0};
    };
      uint64_t _cache_bytes{0};
      uint64_t _workspace_bytes{0};
      // Chunk-scaled part of _workspace_bytes (aliased pool + permanent) and the
      // pool's per-row float count, reported so a log can attribute the
      // footprint to the chunk rule instead of the LM head/logits buffers.
      uint64_t _workspace_chunk_bytes{0};
      uint64_t _workspace_pool_floats_per_row{0};
      uint64_t _shader_count{0};
      // Device capability record captured once at create (see qwen/caps.h).
      QwenBackendCaps _caps{};

    // ---- internal helpers --------------------------------------------------
    // ---- calibration-only state (opt_plan/05 section 3 item 2) --------------
    // One accumulator per projection weight the executor can consume. `acc` holds
    // the running sum of x[j]^2 for input channel j and `tokens` the number of
    // activation rows that reached it; finish_activation_calibration() divides once.
    struct CalibrationEntry {
        const QwenDeviceTensor *weight{nullptr};
        compute::Buffer<float> acc;
        uint64_t k{0};
        uint64_t tokens{0};
    };
    luisa::vector<CalibrationEntry> _calibration_entries;
    bool _calibration_active{false};
    // QwenEngineOptions::calibration_only was requested at create() time (the
    // reference, non-fused decode route is active). Calibration refuses without it.
    bool _calibration_only{false};
    uint64_t _calibration_projections{0};
    // Record one squared-activation accumulation for `w` into `batch` (no-op when
    // the calibration mode is off). Fails closed when `w` is not a calibrated
    // projection: an unobserved projection must never look calibrated.
    [[nodiscard]] Status record_calibration(QwenCommandBatch &batch,
                                            const compute::Buffer<float> &x,
                                            const QwenDeviceTensor &w, uint32_t rows, uint32_t k);
    [[nodiscard]] int calibration_index(const QwenDeviceTensor &w) const noexcept;

    [[nodiscard]] Status compile_all(compute::Device &device);
    [[nodiscard]] Status alloc_state(compute::Device &device);
    [[nodiscard]] Status alloc_workspace(compute::Device &device);
    [[nodiscard]] Status resolve_weights(const Qwen35Model &model, const QwenDeviceWeights &weights);
    // Run the full 32-layer stack over `rows` tokens starting at absolute
    // position `pos0`, leaving the last-position hidden state in _hidden.
    // Records compute into `batch`; sparse map updates may be submitted on
    // `stream` before the new tiles are used, but ordinary compute stays batched.
    // `device_state` selects the device-state realization of the three per-token
    // position consumers (attention prepare, decode attention, GDN conv) and is
    // only ever true for the single-row body that the static CUDA graph records:
    // prefill, graph-off decode and the MTP verify/replay paths keep the host
    // scalar kernels (and `pos0` is then their real argument, not a hint).
    [[nodiscard]] Status forward_chunk(compute::Stream &stream, QwenCommandBatch &batch,
                                       uint32_t rows, uint32_t pos0, bool device_state = false);
    [[nodiscard]] Status full_attention_block(QwenCommandBatch &batch, uint32_t layer,
                                              uint32_t rows, uint32_t pos0, bool device_state = false);
    [[nodiscard]] Status gdn_block(QwenCommandBatch &batch, uint32_t layer, uint32_t rows,
                                   uint32_t pos0, bool device_state = false);
    [[nodiscard]] Status ffn_block(QwenCommandBatch &batch, uint32_t layer, uint32_t rows);
    // LM head on the last position of _hidden -> _logits. Records into batch.
    [[nodiscard]] Status lm_head(QwenCommandBatch &batch, uint32_t rows);
    // Projection helper: y[rows,n] = x[rows,k] @ W (packed), optional residual
    // epilogue (y may alias the residual buffer; only idx is read per write).
    // `label` is a short projection name; the annotation is label + w.name.
    [[nodiscard]] Status project(QwenCommandBatch &batch,
                                 const compute::Buffer<float> &x, const QwenDeviceTensor &w,
                                 const compute::Buffer<float> &y, uint32_t rows,
                                 const compute::Buffer<float> *residual,
                                 luisa::string_view label);
    // Stage token ids into a distinct _host_ids span and copy device-side into
    // _ids (see the staging-lifetime note on _host_ids).
    [[nodiscard]] Status stage_ids(QwenCommandBatch &batch, luisa::span<const uint32_t> ids);
    // Chunk rows for a prompt of `rows` tokens: the fixed _prefill_chunk, or the
    // adaptive rule (opt_plan/02 section 4) capped by the configured max and the
    // per-backend conservative limit. Always in [1, _prefill_chunk].
    [[nodiscard]] uint32_t prefill_chunk_for(uint64_t rows) const noexcept;
    // ---- static decode graph (opt_plan/03 section 2.3) ---------------------
    // Upload {position, kv_len, history_slot, eos} into _dev_state through a
    // one-shot stream-ordered batch that is committed before returning (so the
    // borrowed host span cannot dangle). This is a plain pre-graph command: the
    // graph itself is never updated.
    // `remaining` is the conditional-WHILE token budget of the next launch (0 =
    // "no chunk in flight": the per-token kernels neither read nor write it).
    [[nodiscard]] Status upload_dev_state(compute::Stream &stream, uint32_t position,
                                          uint32_t kv_len, uint32_t history_slot,
                                          uint32_t remaining = 0u);
    // True when the device copy already holds exactly these values.
    [[nodiscard]] bool dev_state_matches(uint32_t position, uint32_t kv_len,
                                         uint32_t history_slot) const noexcept {
        return _dev_state_synced && _dev_state_position == position &&
               _dev_state_kv_len == kv_len && _dev_state_slot == history_slot;
    }
    // Graph-external sparse KV growth for the static graph. The allocator maps
    // only pages that are not resident yet and never moves a virtual address, so
    // growth needs no graph update -- but the map must stay OUTSIDE the captured
    // body. Called with page granularity, i.e. it only submits work when
    // `tokens` crosses the committed page watermark.
    [[nodiscard]] Status ensure_kv_for_decode(compute::Stream &stream, uint32_t tokens);
    // Record one device-greedy decode step (forward, LM head, two-stage argmax)
    // into `batch`. `device_state` selects the device-state realizations of the
    // four per-token scalar consumers -- the body the static graph captures.
    // `while_handle` is 0 for the per-token body; a non-zero conditional handle
    // selects the WHILE finish kernel that re-arms that condition from the device.
    [[nodiscard]] Status record_greedy_step(compute::Stream &stream, QwenCommandBatch &batch,
                                            uint32_t pos0, uint32_t history_slot,
                                            bool device_state, uint64_t while_handle = 0ull);
    // Capture the recorded body into a CUDA graph (first step only) and launch.
    // The command list is consumed by the capture; a capture/instantiate failure
    // is reported and never executed as a list.
    [[nodiscard]] Status capture_static_decode_graph(compute::Stream &stream, QwenCommandBatch &batch);
    // Bare launch of the already-captured static decode graph: no recording, no
    // whole-graph update, no per-node update.
    [[nodiscard]] Status launch_static_decode_graph(compute::Stream &stream);
    // ---- conditional WHILE decode graph (opt_plan/09 N4.2) ------------------
    // Build the parent graph (one 1-thread zero-fill node, because CUDA refuses a
    // node-less graph) + the WHILE container whose body is the already-recorded
    // argument-invariant step, instantiate it, and launch it once. The body is
    // consumed by the build; a failure is reported and never executed as a list.
    [[nodiscard]] Status capture_while_decode_graph(compute::Stream &stream, uint32_t pos0,
                                                    uint32_t history_slot);
    // Bare launch of the captured WHILE graph: one cuGraphLaunch executes 1..chunk
    // tokens, decided on the device.
    [[nodiscard]] Status launch_while_decode_graph(compute::Stream &stream);
    // Read the device decode state back (pinned staging when selected) and
    // synchronize. This is what makes `executed` a measurement instead of an
    // assumption.
    [[nodiscard]] Status read_back_dev_state(compute::Stream &stream, DevStateReadback &state);
    // One logits download with the selected realization (pinned when available,
    // otherwise the existing pageable copy). Resizes `logits_out` to the vocabulary
    // and synchronizes before returning.
    [[nodiscard]] Status download_logits(compute::Stream &stream, QwenCommandBatch &batch,
                                         luisa::vector<float> &logits_out);
    // FP8 prefill partner of a decode tensor (nullptr in a single-copy session).
    [[nodiscard]] const QwenDeviceTensor *prefill_partner(const QwenDeviceTensor &w) const noexcept;
    // Annotation for a projection: <label>.<weight tensor name>.
    [[nodiscard]] static luisa::string proj_name(luisa::string_view label, const QwenDeviceTensor &w);

    // Declared last: drained by the destructor body, then destroyed before all
    // shaders, buffers, the retained stream, and the device extension owner.
    luisa::optional<compute::CudaGraphInstance> _device_greedy_graph;
    luisa::optional<compute::CudaGraphExecInstance> _device_greedy_graph_exec;
    luisa::optional<compute::CudaGraphInstance> _while_graph;
    luisa::optional<compute::CudaGraphExecInstance> _while_graph_exec;
};

}// namespace luisa::qwen
