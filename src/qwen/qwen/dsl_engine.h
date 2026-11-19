#pragma once
// Qwen3.5-4B text-only inference -- COMPLETE ordinary-DSL executor.
//
// QwenDslEngine is the QwenEngine implementation for the dsl-bf16 / dsl-fp8
// variants. It drives the frozen 32-layer schedule (8 x [GDN, GDN, GDN, full
// attention]; full-attention blocks 3, 7, ..., 31) entirely through the
// device-free kernel factories in qwen/dsl_ops.h. No Tile, no native shaders.
//
// Per decoder block:
// input RMSNorm (direct pre-added GGUF weight; the HF zero-centered 1+w
// conversion happened offline in the converter)
//   -> full attention (q/k/v projections, fused Q/gate split + QK norm +
//      partial NeoX rotate-half RoPE, contiguous per-layer FP32 KV cache, causal
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
// State: one contiguous FP32 K and V cache per full-attention layer
// ([max_context, kv_heads, head_dim] each), one FP32 recurrent state
// ([v_heads, head_dim, head_dim]) and one conv ring ([qkv_channels,
// kernel-1]) per GDN layer. reset() clears all of them and the position.
// Prefill processes the prompt in chunks of options.prefill_chunk rows. GDN
// uses the FP32 WY representation + UT transform over serial 64-token chunks
// by default; options.dsl_gdn_wy_ut=false selects the exact per-token recurrence
// for differential debugging. K/V are appended at the proper absolute positions.
// decode() consumes one token through the four-way cooperative recurrent operator.
//
// Weight storage is validated to be UNIFORMLY bf16_packed or fp8_packed for
// every packed matrix (dense F32 tensors stay F32 in both variants); a mixed
// or unsupported storage is a recoverable failure, never a substitution.
//
// All kernels are compiled ONCE at construction (shapes are static); every
// prefill/decode step is recorded into ONE QwenCommandBatch (a CommandList
// wrapper) per public call and submitted with a single synchronize only when
// the logits are downloaded. With options.benchmark=true the batch instead
// commits + synchronizes after EVERY annotated command and records host-wall
// samples into the engine-owned QwenBenchmarkReport exposed through
// benchmark_report(). Greedy CUDA also supports bounded in-flight decode: device
// argmax feeds the next token and only chunk downloads synchronize. Recoverable
// Status everywhere; no exceptions, no RTTI.

#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

#include "qwen/benchmark.h"
#include "qwen/dsl_ops.h"
#include "qwen/engine.h"
#include "qwen/model.h"
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

    ~QwenDslEngine() noexcept override = default;

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
    [[nodiscard]] QwenEngineInfo info() const noexcept override;
    // Nonnull exactly when options.benchmark was true; latest prefill/decode
    // report (reset samples of a prefill are included when recorded).
    [[nodiscard]] const QwenBenchmarkReport *benchmark_report() const noexcept override;

private:
    QwenDslEngine() noexcept = default;

    // ---- static geometry (frozen at construction) -------------------------
    dsl::QwenDslConfig _cfg{};
    uint32_t _layers{0};
    uint32_t _max_context{0};
    uint32_t _prefill_chunk{0};
    dsl::QwenWeightFormat _format{dsl::QwenWeightFormat::bf16};
    WeightStorage _storage{WeightStorage::bf16_packed};
    uint32_t _warp_size{0};

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

    // ---- per-layer resolved weight pointers (into *_weights) --------------
    struct LayerWeights {
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
        // Stable per-layer command annotations (owned; views stay valid).
        luisa::string p_input_norm, p_post_norm, p_swiglu;
        luisa::string p_attn_prepare, p_attn_core;
        luisa::string p_conv, p_gates, p_qk_norm, p_recurrent, p_chunk, p_gated_norm;
    };
    luisa::vector<LayerWeights> _layer_weights;
    const QwenDeviceTensor *_token_embd{nullptr};
    const QwenDeviceTensor *_output_norm{nullptr};

    // ---- persistent device state ------------------------------------------
    // Full attention: one contiguous K and V cache per full layer.
    luisa::vector<compute::Buffer<float>> _k_caches;// [full_layers]
    luisa::vector<compute::Buffer<float>> _v_caches;
    // GDN: one recurrent state and one conv ring per GDN layer.
    luisa::vector<compute::Buffer<float>> _gdn_states;// [gdn_layers]
    luisa::vector<compute::Buffer<float>> _gdn_rings;
    // Map layer index -> cache/state slot (UINT32_MAX when not applicable).
    luisa::vector<uint32_t> _full_slot;
    luisa::vector<uint32_t> _gdn_slot;

    // ---- activation workspace (reused across layers/chunks) ---------------
    compute::Buffer<uint32_t> _ids;       // [chunk] token ids
    compute::Buffer<uint32_t> _ids_host;  // [chunk] device-side staging for ids
    compute::Buffer<int32_t> _positions;  // [chunk] absolute positions
    compute::Buffer<int32_t> _positions_host;// [chunk] device-side staging for positions
    // Persistent host staging for scalar uploads. copy_from(span) borrows the
    // host pointer until the command encoder commits, so the storage must
    // outlive every queued CommandList (never a function-local vector).
    // Sized to max_context: a prefill records ALL chunks into ONE command
    // list, so every chunk keeps a DISTINCT span [_staged, _staged+rows) that
    // is never rewritten before the single commit.
    luisa::vector<uint32_t> _host_ids;
    luisa::vector<int32_t> _host_pos;
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
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, uint32_t> _swiglu;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<uint32_t>,
                    compute::Buffer<float>, compute::Buffer<uint32_t>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t> _ffn_decode;
    compute::Shader<1, compute::Buffer<float>, uint32_t> _zero_fill;
    compute::Shader<1, compute::Buffer<uint32_t>, compute::Buffer<uint32_t>, uint32_t> _copy_u;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<uint32_t>, compute::Buffer<uint32_t>,
                    uint32_t> _greedy_argmax_partial;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<uint32_t>,
                    compute::Buffer<uint32_t>, compute::Buffer<uint32_t>,
                    compute::Buffer<uint32_t>, uint32_t, uint32_t> _greedy_argmax_finish;
    compute::Shader<1, compute::Buffer<int32_t>, compute::Buffer<int32_t>, uint32_t> _copy_i;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<int32_t>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _attn_prepare;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t, uint32_t> _attn_prefill;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, float, uint32_t, uint32_t> _attn_decode;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, uint32_t, uint32_t, uint32_t, uint32_t> _gdn_conv_sh;
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

    // ---- session state -----------------------------------------------------
    uint32_t _position{0};// number of tokens consumed (next absolute position)
    bool _has_session{false};
    bool _gdn_wy_ut{true};
    bool _decode_specialized{true};
    uint64_t _cache_bytes{0};
    uint64_t _workspace_bytes{0};
    uint64_t _shader_count{0};

    // ---- internal helpers --------------------------------------------------
    [[nodiscard]] Status compile_all(compute::Device &device);
    [[nodiscard]] Status alloc_state(compute::Device &device);
    [[nodiscard]] Status alloc_workspace(compute::Device &device);
    [[nodiscard]] Status resolve_weights(const Qwen35Model &model, const QwenDeviceWeights &weights);
    // Run the full 32-layer stack over `rows` tokens starting at absolute
    // position `pos0`, leaving the last-position hidden state in _hidden.
    // Records into `batch`; in normal mode nothing is submitted here.
    [[nodiscard]] Status forward_chunk(QwenCommandBatch &batch, uint32_t rows, uint32_t pos0);
    [[nodiscard]] Status full_attention_block(QwenCommandBatch &batch, uint32_t layer,
                                              uint32_t rows, uint32_t pos0);
    [[nodiscard]] Status gdn_block(QwenCommandBatch &batch, uint32_t layer, uint32_t rows, uint32_t pos0);
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
    // Upload the position ramp ONCE per forward_chunk (not per attention layer).
    [[nodiscard]] Status upload_positions(QwenCommandBatch &batch, uint32_t rows, uint32_t pos0);
    // Stage token ids into a distinct _host_ids span and copy device-side into
    // _ids (see the staging-lifetime note on _host_ids).
    [[nodiscard]] Status stage_ids(QwenCommandBatch &batch, luisa::span<const uint32_t> ids);
    // Annotation for a projection: <label>.<weight tensor name>.
    [[nodiscard]] static luisa::string proj_name(luisa::string_view label, const QwenDeviceTensor &w);
};

}// namespace luisa::qwen
