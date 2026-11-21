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
// State: one sparse virtual allocation with on-demand FP32 K/V regions per
// full-attention layer ([max_context, kv_heads*head_dim] each), one FP32 recurrent state
// ([v_heads, head_dim, head_dim]) and TWO ping-pong conv rings
// ([qkv_channels, kernel-1] each) per GDN layer. reset() clears logical state,
// releases KV physical residency, and zeroes the fixed GDN buffers.
//
// Conv+SiLU schedule (options.chunk_conv, default true): ONE
// capture_tile_gdn_conv_silu_chunk launch per chunk/layer for the WHOLE chunk
// (all `rows` tokens plus the ring image), with the causal window carried in
// registers and pos0 supplied device-visibly (one int32 upload per chunk). The
// per-token path (capture_tile_gdn_conv_silu_step: one launch + one position
// upload + one source-row staging copy + one output gather copy PER TOKEN) stays
// available as the differential / A-B arm (chunk_conv=false). The ring pair is
// ping-ponged per INVOCATION, not per position parity, so ring_in and ring_out
// are distinct buffers for any row count (the Tile runtime rejects a buffer
// bound to two arguments when either is writable).
// Prefill processes the prompt in chunks of options.prefill_chunk rows using
// the SAME recurrent GDN operator (exact per-token recurrence, NOT WY/UT);
// K/V are appended at the proper absolute positions. decode() consumes one
// token at the current position.
//
// Attention lengths: the static tile_ops attention captures specialize the
// serial key loop to one exact kv_len (and prefill pos0). To avoid a JIT per
// token this engine uses the DEVICE-VISIBLE-LENGTH variants
// (capture_tile_attn_prefill_dynamic / capture_tile_attn_decode_dynamic):
// one compiled shader per (chunk_rows, capacity) streams its STATIC key range
// and gates keys with a limit read from a small device buffer. The scalar
// online-softmax carry pattern is the verified-safe serial-select form. The
// static range is the LENGTH BUCKET ceil(kv_len/attention_bucket_keys)*bucket
// clamped to max_context (attention_bucket_keys=0 disables bucketing and
// streams the whole capacity), so a 20-token session scans 512 keys instead of
// 4096 while the length stays device-visible. The bucket list is bounded by
// ceil(max_context/bucket)+1 shaders: precompile() compiles the decode set and
// the full-chunk prefill set at create, and the remaining (ragged tail) pairs
// use a bounded, logged lazy cache (max_attention_bucket_lazy).
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
#include "qwen/kv_cache_allocator.h"
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
    // Attention length bucket in KEYS (0 disables bucketing). The
    // device-visible-length attention shader streams a STATIC key range, so a
    // single full-capacity specialization always costs `max_context` keys per
    // query even for a 20-token session. With a nonzero bucket the same
    // device-visible-length shader is compiled per length bucket and the scan
    // is ceil(kv_len/bucket)*bucket (clamped to max_context), i.e.
    // ceil(kv_len/512)*512 at the default: a 20-token context scans 512 keys
    // instead of 4096. The bucket count is bounded by
    // ceil(max_context/bucket)+1 shaders; the precompile policy in precompile()
    // plus the bounded lazy fallback below keep a request-path compile from
    // being a surprise (every lazy compile is logged and counted in
    // info().tile_attention_lazy_compiles).
    uint32_t attention_bucket_keys{512};
    // Bound on attention bucket shaders compiled on the REQUEST path (not
    // precompiled at create). Only ragged prefill tails and unusual
    // (rows, bucket) pairs can need one. Exceeding the bound is a recoverable
    // failure, never a silent fallback to the full capacity.
    uint32_t max_attention_bucket_lazy{32};
    // Multi-token GDN Conv4+SiLU: for a prefill chunk of MORE THAN ONE row, one
    // capture_tile_gdn_conv_silu_chunk launch per chunk/layer (correct causal
    // ring history + device-visible position addressing) instead of one
    // capture_tile_gdn_conv_silu_step launch plus a per-token position upload,
    // source-row staging copy and output gather copy. A one-row chunk (decode)
    // always uses the step kernel: it is one launch anyway and measured ~2%
    // faster than the generalized chunk kernel at T=1. false keeps the
    // per-token step path for every row count as the differential / A-B arm.
    bool chunk_conv{true};
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

    ~QwenTileEngine() noexcept override;

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
    compute::Stream _stream;// retained for ordered engine teardown
    // Stage-boundary sync policy, detected once at create from
    // device.backend_name(). true (VK and every non-DX backend): QWEN_TS
    // stage flushes and every-N-layer flushes commit + synchronize, exactly
    // the historical policy. false (DX): stage boundaries never flush and the
    // every-N-layer bounds flush(false) (commit only, no host wait until the
    // final logits download). Benchmark mode commits + syncs every command
    // individually, so flushes are no-ops under either policy.
    bool _stage_sync{true};
    // Device capability record captured once at create (see qwen/caps.h).
    QwenBackendCaps _caps{};

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
    // Stable dense-addressing K/V views backed by on-demand sparse tiles.
    QwenKvCacheAllocator _kv_cache;
    luisa::vector<compute::Buffer<float>> _gdn_states;// [gdn_layers]
    // Two ping-pong rings per GDN layer: _gdn_rings[2*slot + parity].
    luisa::vector<compute::Buffer<float>> _gdn_rings;
    // Which PHYSICAL ring of the pair currently holds this layer's history.
    // The pair alternates per conv INVOCATION (not per position parity): a
    // chunk kernel reads ring[p] and writes ring[1-p] for ANY number of rows,
    // so ring_in and ring_out are always distinct arguments (the Tile runtime
    // rejects one buffer bound to two arguments when either is writable).
    // reset() zeroes both rings and clears every parity to 0.
    luisa::vector<uint32_t> _gdn_ring_parity;
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
    // Packed QKV projection output, DENSE [chunk, qkv_channels]: the multi-token
    // conv chunk reads its rows directly (row stride == qkv_channels), so no
    // per-token staging copy is needed. The per-token step path stages rows out
    // of this buffer into _conv_src (whole-buffer descriptors only).
    compute::Buffer<float> _gdn_qkv; // [chunk * qkv_channels]
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
    // Attention bucket bookkeeping (reported through info()). _attention_caps is
    // the canonical, bounded bucket list derived from max_context and
    // attention_bucket_keys; the two counters separate the shaders compiled at
    // create (no request-path surprise) from the ones compiled on first use.
    luisa::vector<uint32_t> _attention_caps;
    uint32_t _attention_bucket_precompiled{0};
    uint32_t _attention_bucket_lazy{0};

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

    // Attention capacity (static key-range of the device-visible-length
    // shader) for a given kv_len: the bucket ceiling, clamped to max_context.
    // kv_len <= the returned capacity always holds, which is what the captures
    // require; it is exactly max_context when attention_bucket_keys == 0.
    [[nodiscard]] uint32_t attention_capacity(uint32_t kv_len) const noexcept;
    // Request-path attention shader: same cache as shader_lazy, plus the bounded
    // lazy-compile accounting and the log line that makes a request-path compile
    // visible instead of a silent stall. `cap`/`rows` are only for that report.
    template<typename F>
    [[nodiscard]] Status attention_shader(compute::Device &device, const luisa::string &key,
                                          uint32_t rows, uint32_t cap, F &&make_capture,
                                          const compute::tile::Shader **out);
    // Physical ring pair for the NEXT conv invocation of `slot` (0 = first GDN
    // layer): returns the READ buffer and sets `out` to the (always distinct)
    // WRITE buffer, then advances the pair for the following invocation.
    [[nodiscard]] const compute::Buffer<float> *take_ring_pair(
        uint32_t slot, const compute::Buffer<float> *&out) noexcept;

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
