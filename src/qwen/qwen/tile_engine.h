#pragma once
// Qwen3.5-4B text-only inference -- COMPLETE Tile executor.
//
// QwenTileEngine is the QwenEngine implementation for the tile-bf16 /
// tile-fp8 variants. It drives the frozen 32-layer schedule (8 x [GDN, GDN,
// GDN, full attention]; full-attention blocks 3, 7, ..., 31) entirely through
// the device-free Tile capture factories in qwen/tile_ops.h. Every compute
// operation is a Tile kernel compiled with tile::compile (default
// Lowering::NATIVE; on backends without a native Tile path the backend
// honestly reports the XIR realization or fails -- this engine NEVER falls
// back to the ordinary DSL, native shaders, or another weight format).
//
// Per decoder block (identical schedule to QwenDslEngine):
// input RMSNorm (DIRECT pre-added GGUF weight; HF's zero-centered 1+w
// was folded offline by the converter)
//   -> full attention (q/k/v packed projections -> prepare (Q/gate split +
// direct pre-added per-head QK norm) -> separate NeoX rotate-half RoPE
//      on Q and K -> KV append into the per-layer contiguous FP32 caches ->
//      causal GQA prefill / single-query decode attention with sigmoid
//      output gate -> output projection)
//      OR GDN (packed QKV + z + alpha + beta projections, depthwise conv4 +
//      SiLU with per-layer ring history, QK L2 norm + Q scale, beta/decay
//      gates, FP32 delta-rule recurrent state per layer, gated RMS output
//      norm with direct weight, output projection)
//   -> residual add
// -> post-attention RMSNorm (direct pre-added weight)
//   -> separate ffn_gate / ffn_up projections -> SwiGLU -> ffn_down
//   -> residual add.
// Final: output_norm (direct pre-added weight) on the LAST position only, then the tied
// token_embd LM head for that single position -> FP32 logits [vocab].
//
// State: one contiguous FP32 K and V cache per full-attention layer
// ([max_context, kv_heads*head_dim] each), one FP32 recurrent state
// ([v_heads, head_dim, head_dim]) and TWO ping-pong conv rings
// ([qkv_channels, kernel-1] each) per GDN layer. reset() clears all of them
// and the position. The conv runs ONE TOKEN AT A TIME through
// capture_tile_gdn_conv_silu_step with separate ring_in/ring_out buffers:
// token at absolute position pos reads ring[pos%2] and writes ring[(pos+1)%2]
// (a full history copy with only slot pos%(kernel-1) overwritten). This
// trades one launch per token for the absence of any in-place ring
// read/write hazard; exactly three capture variants (pos_mod) are compiled.
// Prefill processes the prompt in chunks of options.prefill_chunk rows using
// the SAME recurrent GDN operator (exact per-token recurrence, NOT WY/UT);
// K/V are appended at the proper absolute positions. decode() consumes one
// token at the current position.
//
// Attention lengths: the static tile_ops attention captures specialize the
// serial key loop to one exact kv_len (and prefill pos0). To avoid a JIT per
// token this engine uses the DEVICE-VISIBLE-LENGTH variants
// (capture_tile_attn_prefill_dynamic / capture_tile_attn_decode_dynamic):
// one compiled shader per (chunk_rows, capacity) streams the full capacity
// and gates keys with a limit read from a small device buffer. The scalar
// online-softmax carry pattern is the verified-safe serial-select form.
// A bounded, position-keyed cache for the STATIC variants is also available
// (QwenTileEngineOptions::static_attention) for backends/routes where the
// dynamic loop is undesirable; it fails closed on compile errors and evicts
// nothing (bounded by max_static_attention_shaders).
//
// Shader cache: every captured/compiled shader is cached by ALL semantic
// static parameters (op kind, weight format, epilogue, rows, dims, pos0,
// capacity, eps, ...). info().compiled_shader_count reports the number of
// DISTINCT shaders compiled over the engine lifetime (cache hits do not
// increment it).
//
// Weight storage is validated to be UNIFORMLY bf16_packed or fp8_packed for
// every packed matrix (dense F32 tensors stay F32 in both variants); a mixed
// or unsupported storage is a recoverable failure, never a substitution.
// FP8 GEMM requires K % 4 == 0 (word-aligned packed rows; every model K is a
// multiple of 64). FP8 embedding requires hidden % 64 == 0.
//
// LIFETIME RULE (learned from the DSL integration): BufferView::copy_from
// (span) borrows the host pointer until command encoding; function-local
// host vectors cause nondeterministic uploads. ALL host staging lives in
// persistent engine members that outlive every queued CommandList, and the
// stream is synchronized before any staging reuse/destruction and before
// prefill/decode return logits.
//
// Recoverable Status everywhere; no exceptions, no RTTI. Capture/verifier/
// shader-compile errors are propagated with their diagnostics.
//
// Benchmark: with QwenEngineOptions::benchmark=true every recorded command
// (Tile kernel or host/device transfer) is committed + synchronized
// individually through QwenCommandBatch on the SAME stream and appended to
// the engine-owned QwenBenchmarkReport with a unique operator name and
// phase/position/rows/layer attribution. The report is cleared at the start
// of prefill (reset samples are part of the prefill trace) and decode, and
// exposed via benchmark_report(). Normal-mode flush policy is selected once
// at create from device.backend_name() (_stage_sync): on DX the block-internal
// stage boundaries do NOT flush and the every-N-layer bounds flush(false)
// (commit only -- no host wait until the final logits download); on VK and
// every other backend both flush(true) exactly as before. All flushes route
// through QwenCommandBatch::flush, which is a no-op in benchmark mode (every
// command was already committed + synchronized individually).
//
// Prefill records ALL chunks into ONE QwenCommandBatch (no per-chunk sync):
// the host id/position staging vectors are max_context-sized and each chunk
// writes/uploads a DISTINCT absolute-offset span that is never rewritten
// before the batch commits; the dynamic attention lengths live in a
// 2*max_context host vector, two fresh slots per chunk (pos0 addresses the
// pair). RoPE positions and attention lengths are uploaded ONCE per
// forward_chunk and shared by every full-attention layer.

#include <cstdint>
#include <memory>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/tile/runtime.h>

#include "qwen/benchmark.h"
#include "qwen/engine.h"
#include "qwen/model.h"
#include "qwen/tile_ops.h"
#include "qwen/weights.h"

namespace luisa::qwen {

struct QwenTileEngineOptions {
    QwenEngineOptions base{};
    // Tile lowering requested from tile::compile. NATIVE is the default; a
    // backend that cannot honor it fails closed (no silent substitution).
    compute::tile::Lowering lowering{compute::tile::Lowering::NATIVE};
    // Use the device-visible-length attention shaders (one compile per
    // chunk/capacity). When false, position-specialized STATIC attention
    // shaders are compiled on demand and cached (bounded below).
    bool dynamic_attention{true};
    // Bound on the number of cached STATIC position-specialized attention
    // shaders (prefill + decode). Exceeding it is a recoverable failure.
    uint32_t max_static_attention_shaders{64};
    // GEMM tile tuning (defaults match tile_ops: 16x16 keeps the FP8 decoded
    // snapshot within the DX/VK per-worker budget).
    uint32_t gemm_tile_m{16};
    uint32_t gemm_tile_n{16};
    // Command-list flush policy, applied per backend (see _stage_sync).
    // Nonzero: on VK and other backends commit + synchronize at every
    // block-internal stage boundary (projections / conv / gates / recurrence /
    // FFN) and additionally every N transformer layers -- VK requires the
    // stage-boundary flushes (VK_ERROR_DEVICE_LOST otherwise). On DX the stage
    // boundaries never flush; only the every-N-layer bounds remain, as
    // commit-only flush(false): DX tolerates at most ~8 layers per submit
    // (DXGI_ERROR_DEVICE_REMOVED) but needs no host waits before the final
    // logits download. Verified stable on both with the default. 0 disables
    // all intermediate flushes for backends that tolerate long recordings.
    uint32_t sync_every_layers{8};
};

class QwenTileEngine final : public QwenEngine {
public:
    // Allocator-matched deleter for the factory handle (same pattern as
    // QwenDslEngine::Deleter). The engine is allocated with Luisa's allocator
    // (mimalloc-backed); luisa::unique_ptr's default eastl::default_delete
    // frees through EASTL's default allocator, and mixing the two heaps
    // corrupts memory (segfault on destruction). This deleter destroys
    // virtually and frees with the SAME allocator.
    using Deleter = QwenEngineDeleter;
    // Factory used by QwenEngine::create for QwenKernelKind::tile (and
    // directly by tests). Validates options/weights, allocates state +
    // workspace, precompiles the shape-static shaders. `weights` is borrowed
    // and must outlive the engine.
    [[nodiscard]] static Result<luisa::unique_ptr<QwenEngine, Deleter>> create(
        compute::Device &device, compute::Stream &stream,
        const Qwen35Model &model, const QwenDeviceWeights &weights,
        const QwenEngineOptions &options);
    [[nodiscard]] static Result<luisa::unique_ptr<QwenEngine, Deleter>> create(
        compute::Device &device, compute::Stream &stream,
        const Qwen35Model &model, const QwenDeviceWeights &weights,
        const QwenTileEngineOptions &options);

    ~QwenTileEngine() noexcept override = default;

    [[nodiscard]] Status reset(compute::Stream &stream) override;
    [[nodiscard]] Status prefill(compute::Stream &stream,
                                 luisa::span<const uint32_t> prompt_tokens,
                                 luisa::vector<float> &logits_out) override;
    [[nodiscard]] Status decode(compute::Stream &stream, uint32_t input_token,
                                luisa::vector<float> &logits_out) override;
    [[nodiscard]] QwenEngineInfo info() const noexcept override;
    // Null unless options.base.benchmark was true. Otherwise the latest
    // prefill/decode report (prefill includes the reset trace), owned by the
    // engine and valid until the next prefill/decode/reset call.
    [[nodiscard]] const QwenBenchmarkReport *benchmark_report() const noexcept override;

private:
    QwenTileEngine() noexcept = default;

    // ---- static geometry (frozen at construction) -------------------------
    tile_ops::QwenTileConfig _cfg{};
    uint32_t _layers{0};
    uint32_t _max_context{0};
    uint32_t _prefill_chunk{0};
    tile_ops::QwenTileWeightFormat _format{tile_ops::QwenTileWeightFormat::bf16};
    WeightStorage _storage{WeightStorage::bf16_packed};
    QwenTileEngineOptions _opt{};

    // ---- borrowed weights / device ----------------------------------------
    const QwenDeviceWeights *_weights{nullptr};
    compute::Device _device;// shared handle to the creating device
    // Stage-boundary sync policy, detected once at create from
    // device.backend_name(). true (VK and every non-DX backend): QWEN_TS
    // stage flushes and every-N-layer flushes commit + synchronize, exactly
    // the historical policy. false (DX): stage boundaries never flush and the
    // every-N-layer bounds flush(false) (commit only, no host wait until the
    // final logits download). Benchmark mode commits + syncs every command
    // individually, so flushes are no-ops under either policy.
    bool _stage_sync{true};

    struct LayerWeights {
        const QwenDeviceTensor *attn_norm{nullptr};
        const QwenDeviceTensor *post_norm{nullptr};
        const QwenDeviceTensor *ffn_gate{nullptr};
        const QwenDeviceTensor *ffn_up{nullptr};
        const QwenDeviceTensor *ffn_down{nullptr};
        const QwenDeviceTensor *attn_q{nullptr};
        const QwenDeviceTensor *attn_k{nullptr};
        const QwenDeviceTensor *attn_v{nullptr};
        const QwenDeviceTensor *attn_output{nullptr};
        const QwenDeviceTensor *attn_q_norm{nullptr};
        const QwenDeviceTensor *attn_k_norm{nullptr};
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
    };
    luisa::vector<LayerWeights> _layer_weights;
    const QwenDeviceTensor *_token_embd{nullptr};
    const QwenDeviceTensor *_output_norm{nullptr};

    // ---- persistent device state ------------------------------------------
    luisa::vector<compute::Buffer<float>> _k_caches;// [full_layers]
    luisa::vector<compute::Buffer<float>> _v_caches;
    luisa::vector<compute::Buffer<float>> _gdn_states;// [gdn_layers]
    // Two ping-pong rings per GDN layer: _gdn_rings[2*slot + parity].
    luisa::vector<compute::Buffer<float>> _gdn_rings;
    luisa::vector<uint32_t> _full_slot;
    luisa::vector<uint32_t> _gdn_slot;

    // ---- activation workspace (reused across layers/chunks) ---------------
    compute::Buffer<int32_t> _ids;         // [chunk] token ids (int32 for Tile)
    compute::Buffer<int32_t> _positions;   // [chunk] absolute positions
    compute::Buffer<int32_t> _attn_len; // [2] {pos0, kv_len} device-visible
    // kv_len alone, padded to the view alignment: the decode attention binds
    // it as a whole buffer (offset 0) instead of an unaligned _attn_len[1]
    // subview (Vulkan HLSL descriptor alignment).
    compute::Buffer<int32_t> _kv_len; // [kv_len_stride]
    uint32_t _kv_len_stride{1};
    // Host-precomputed FP32 RoPE tables [max_context, rope_dim/2]: exact
    // libm cos/sin(pos * inv_freq[pair]) rounded once. No device sin/cos.
    compute::Buffer<float> _rope_cos;
    compute::Buffer<float> _rope_sin;
    compute::Buffer<float> _hidden;        // [chunk, hidden]
    compute::Buffer<float> _normed;        // [chunk, hidden]
    compute::Buffer<float> _residual;      // [chunk, hidden]
    compute::Buffer<float> _zero;          // [chunk, hidden] all-zero copy operand
    compute::Buffer<float> _proj_a;        // [chunk, max_proj] scratch
    compute::Buffer<float> _proj_b;        // [chunk, max_proj] scratch
    compute::Buffer<float> _proj_c;        // [chunk, max_proj] scratch
    compute::Buffer<float> _kseg_acc; // [chunk, max_proj] K-split accumulator
    compute::Buffer<float> _ffn_gate;      // [chunk, ffn]
    compute::Buffer<float> _ffn_up;        // [chunk, ffn]
    compute::Buffer<float> _ffn_act;       // [chunk, ffn]
    compute::Buffer<float> _attn_proj_q;   // [chunk, 2*q_heads*hd]
    compute::Buffer<float> _attn_proj_k;   // [chunk, kv_heads*hd]
    compute::Buffer<float> _attn_proj_v;   // [chunk, kv_heads*hd]
    compute::Buffer<float> _attn_q;        // [chunk, q_heads, hd] normed
    compute::Buffer<float> _attn_gate;     // [chunk, q_heads, hd] raw
    compute::Buffer<float> _attn_k;        // [chunk, kv_heads, hd] normed
    compute::Buffer<float> _attn_v;        // [chunk, kv_heads, hd] raw
    compute::Buffer<float> _attn_qr;       // [chunk, q_heads, hd] roped
    compute::Buffer<float> _attn_kr;       // [chunk, kv_heads, hd] roped
    compute::Buffer<float> _attn_out;      // [chunk, q_heads, hd]
    compute::Buffer<float> _gdn_conv;      // [chunk, qkv_channels]
    // Per-token conv-step scratch: the one-token ping-pong conv binds these
    // as WHOLE buffers (offset 0) so every backend's storage-buffer
    // descriptor alignment rule is satisfied (Vulkan HLSL rejects unaligned
    // subview offsets). _conv_src rows are padded to _view_align_bytes.
    compute::Buffer<float> _conv_src; // [chunk * conv_src_stride]
    compute::Buffer<float> _conv_out; // [chunk * qkv_channels]
    // Per-token absolute positions, one int32 padded to the view alignment
    // (whole-buffer binding per step; avoids unaligned int32 subviews on VK).
    compute::Buffer<int32_t> _conv_pos; // [chunk * conv_pos_stride]
    uint32_t _conv_pos_stride{1};
    uint32_t _conv_src_stride{0}; // qkv_n rounded up to the view alignment
    uint64_t _view_align_bytes{4}; // storage-buffer view offset alignment
    compute::Buffer<float> _gdn_q;         // [chunk, qk_heads, hd]
    compute::Buffer<float> _gdn_k;         // [chunk, qk_heads, hd]
    compute::Buffer<float> _gdn_beta;      // [chunk, v_heads]
    compute::Buffer<float> _gdn_g; // [chunk, v_heads]
    compute::Buffer<float> _gdn_braw; // [chunk, v_heads] raw beta projection (gates input; separate from g output)
    compute::Buffer<float> _gdn_out; // [chunk, v_heads, hd]
    compute::Buffer<float> _gdn_normed;    // [chunk, v_heads, hd]
    compute::Buffer<float> _last_normed;   // [hidden]
    compute::Buffer<float> _logits;        // [vocab]
    compute::Buffer<float> _dummy_f;       // 1-element dummy
    compute::Buffer<float> _dummy_f2;      // [1,1] dummy for rank-2 unused args
    compute::Buffer<uint32_t> _dummy_u;

    // Persistent HOST staging (outlives every queued CommandList -- see the
    // lifetime rule). Prefill records all chunks into ONE batch without any
    // intermediate sync, so every chunk gets a DISTINCT absolute-offset span
    // that no later chunk rewrites before the batch commits:
    //   _host_ids / _host_pos: [max_context]; chunk at pos0 uses
    //     [pos0, pos0+rows). The absolute offset doubles as the staged cursor.
    //   _host_len: [2*max_context]; chunk at pos0 uses [2*pos0, 2*pos0+2) =
    //     {pos0, kv_len}. Decode (one chunk per batch) uses slot 0 of each.
    // reset() synchronizes the stream before any of these spans is rewritten.
    luisa::vector<int32_t> _host_ids;
    luisa::vector<int32_t> _host_pos;
    luisa::vector<int32_t> _host_len;

    // ---- shader cache ------------------------------------------------------
    // Key: a canonical string of ALL semantic static parameters. Value: the
    // compiled Tile shader (move-only; wrapped in unique_ptr so the map can
    // rehash). The map owns every shader for the engine lifetime.
    luisa::unordered_map<luisa::string, luisa::unique_ptr<compute::tile::Shader>> _shaders;
    uint64_t _shader_count{0};// distinct compiles over the lifetime
    uint32_t _static_attention_count{0};

    // ---- command benchmark (opt-in; see qwen/benchmark.h) ------------------
    bool _benchmark{false};// options.base.benchmark at construction
    QwenBenchmarkReport _report;

    // ---- session state -----------------------------------------------------
    uint32_t _position{0};
    bool _has_session{false};
    uint64_t _cache_bytes{0};
    uint64_t _workspace_bytes{0};

    // ---- internal helpers --------------------------------------------------
    [[nodiscard]] Status resolve_weights(const Qwen35Model &model, const QwenDeviceWeights &weights);
    [[nodiscard]] Status alloc_state(compute::Device &device);
    [[nodiscard]] Status alloc_workspace(compute::Device &device);
    [[nodiscard]] Status precompile(compute::Device &device, compute::Stream &stream);

    // Capture+compile through the cache. `key` must encode every semantic
    // static parameter of the capture. On any capture/verifier/compile error
    // the diagnostics are appended to `detail` and a recoverable Status is
    // returned; nothing is cached on failure.
    [[nodiscard]] Status shader_for(compute::Device &device, const luisa::string &key,
                                    const tile_ops::TileCapture &capture,
                                    const compute::tile::Shader **out,
                                    luisa::string &detail);
    // Convenience: build the capture lazily only on a cache miss.
    template<typename F>
    [[nodiscard]] Status shader_lazy(compute::Device &device, const luisa::string &key,
                                     F &&make_capture, const compute::tile::Shader **out);

    // All recording helpers take a QwenCommandBatch (plain CommandList
    // wrapper in normal mode; per-command commit+sync recorder when
    // _benchmark). `name` labels every command the helper records.
    [[nodiscard]] Status zero_buffer(QwenCommandBatch &cl, compute::Device &device,
                                     compute::Buffer<float> &buf, luisa::string_view name);
    // Each staging helper writes ONLY its chunk's absolute-offset host span
    // (ids/pos: offset pos0; len: offset 2*pos0) and uploads exactly that span
    // once; no span is ever rewritten before the owning batch commits.
    [[nodiscard]] Status stage_ids(QwenCommandBatch &cl, luisa::span<const uint32_t> ids, uint32_t pos0);
    [[nodiscard]] Status stage_positions(QwenCommandBatch &cl, uint32_t rows, uint32_t pos0);
    [[nodiscard]] Status stage_attn_len(QwenCommandBatch &cl, uint32_t pos0, uint32_t kv_len);

    [[nodiscard]] Status project(compute::Device &device, compute::Stream &stream, QwenCommandBatch &cl,
                                 const compute::Buffer<float> &x, const QwenDeviceTensor &w,
                                 const compute::Buffer<float> &y, uint32_t rows,
                                 const compute::Buffer<float> *residual, luisa::string_view name);
    [[nodiscard]] Status rmsnorm(compute::Device &device, QwenCommandBatch &cl,
                                 const compute::BufferView<float> &x, const compute::Buffer<float> &w,
                                 const compute::BufferView<float> &y, uint32_t rows, uint32_t dim,
                                 luisa::string_view name);
    [[nodiscard]] Status full_attention_block(compute::Device &device, compute::Stream &stream, QwenCommandBatch &cl,
                                              uint32_t layer, uint32_t rows, uint32_t pos0);
    [[nodiscard]] Status gdn_block(compute::Device &device, compute::Stream &stream, QwenCommandBatch &cl,
                                   uint32_t layer, uint32_t rows, uint32_t pos0);
    [[nodiscard]] Status ffn_block(compute::Device &device, compute::Stream &stream, QwenCommandBatch &cl,
                                   uint32_t layer, uint32_t rows);
    [[nodiscard]] Status forward_chunk(compute::Device &device, compute::Stream &stream, QwenCommandBatch &cl,
                                       uint32_t rows, uint32_t pos0);
    [[nodiscard]] Status lm_head(compute::Device &device, compute::Stream &stream, QwenCommandBatch &cl, uint32_t rows);
};

}// namespace luisa::qwen
