#pragma once
// Qwen3.5-4B text-only inference -- LuisaCompute Tile DSL operators.
// Self-contained, device-free capture factories + host validation + CPU
// oracles + a device-free self-test. Namespace luisa::qwen::tile_ops.
// NO ordinary Kernel1D/Kernel2D, NO native shaders: every operator is a
// tile::tile_kernel definition specialized through .capture(tensor_shape...)
// into verified TileIR. Compiling the returned tile::Kernel is the caller's
// (executor's) job through tile::compile(device, kernel, options); this header
// never touches a Device and never silently falls back to another lowering,
// precision or weight format.
//
// ---------------------------------------------------------------------------
// WEIGHT STORAGE ABI (matches .kimix_cache/qwen_implementation_contract.md)
// ---------------------------------------------------------------------------
//   bf16     : Buffer<uint32_t>, K*N/2 words. Two BF16 code units per
//              little-endian word, LOW element first (raw GGUF BF16 bytes).
//   fp8_e4m3 : Buffer<uint32_t>, ceil(K*N/4) words. Four E4M3FN codes per
//              little-endian word, LOW element first, plus FP32 scales: one
//              scale per contiguous group of 64 K elements per output row
//              (group is fixed at 64; capture rejects other groups).
//              Dequantized weight = fp8(code) * scale[n*ceil(K/64) + k/64].
// Matrices are ROW-PER-OUTPUT with K contiguous: W[n*K + k] (GGUF [K,N]
// payload row stride K). y[m,n] = sum_k x[m,k] * W[n,k]. All activations,
// accumulations, norms, softmax, gates and GDN state are FP32; bf16/fp8
// denotes WEIGHT STORAGE only. Both formats are ACTUALLY DECODED inside the
// Tile kernels from the uint32 words (no FP32 upload copies).
//
// ---------------------------------------------------------------------------
// TILE / BACKEND CONSTRAINTS (verified against the pinned LuisaCompute)
// ---------------------------------------------------------------------------
// * Every kernel has exactly ONE root parallel nest (rank <= 3), static
//   extents, automatic execution scopes, no explicit Memory: this is the
//   common subset of the CUDA Tile-IR native route, the CUDA/Metal TIRx
//   route and the DX/VK TileIR->XIR->AST route.
// * Ragged tiles use the default bounds::zero (zero-filled loads, suppressed
//   stores). Dynamic per-element accesses (embedding gather, attention key
//   scans) use predicated scalar loads instead.
// * Packed decode uses uint32 division/remainder by constants and the
//   float<->uint32 bitcast (Tile has no integer shifts and no uint32->uint16
//   narrowing bitcast). BF16: bitcast<float>(lo16 << 16). FP8 E4M3FN:
//   exponent/mantissa arithmetic incl. exact subnormals and the 0x7f/0xff
//   NaN codes.
// * No narrow (BF16/FP8) tile arithmetic is used anywhere: decoded values are
//   cast to FP32 before any math (CUDA native rejects narrow arithmetic).
// * No cos/sin opcodes exist in TileIR: RoPE evaluates sin/cos with a
//   3-term Cody-Waite reduction of pi/2 plus a bounded Taylor series, all
//   in FP32. The CPU oracle uses the identical polynomial. Absolute error
//   is <= ~2e-4 for |angle| <= ~1e4 and grows to ~3e-3 at 262144 (FP32
//   reduction limit); this is documented, not silently tightened.
// * GEMM/GEMV decode the packed weight tile then contract with tile::mma in
//   FP32 (reassociation allowed; the default unordered policy). Small-M
//   (decode) uses the same specialization with M=1..4; there is no separate
//   warp-GEMV in Tile because the planner owns the physical mapping.
// * GDN recurrence and attention softmax use ordered serial nests with
//   scalar carries (deterministic left-to-right FP32 order). The chunk-64
//   WY/UT GDN prefill is NOT implemented; the exact per-token recurrence is
//   the prefill path (see tile_ops_chunk_prefill_status()).
// * The DX/VK XIR route materializes per-program snapshots; keep tile
//   extents modest (the factories below use <= 16K elements per tile and
//   serially stream K in blocks of 64/128). CUDA native caps tiles at 2^24.
//   The FP8 GEMM decodes all four byte lanes of a K-block into ONE (n,k)
//   FP32 tile and issues a single mma per 64-K block (the 4-lane-mma form
//   exceeds the 65536-byte/worker snapshot budget); tile_n defaults to 16.
// * FP8 GEMM/GEMV require K % 4 == 0: the packed ABI is element-contiguous
//   (4 codes/word) and the Tile weight load is a 2D (n, K/4) word tile, so
//   each output row must start on a word boundary. The model's K values are
//   all multiples of 64; ragged K not divisible by 4 is rejected fail-closed.
//   FP8 embedding requires hidden % 64 == 0 (whole words + scale groups/row).
// * The DX/VK XIR route has exhibited uniform-ite store-select miscompiles.
//   The model path avoids that pattern structurally: attention Q/K/V and GDN
//   Q/K preparation use separate host-selected captures, and every store is
//   in bounds by construction. No out-of-bounds-store workaround is used.
//   FP8 decode assembles the sign bit with INTEGER arithmetic
//   ((code/128)*0x80000000u) because cast<uint32>(2^31 as float) overflows
//   the signed conversion on this route, and bitcasts (not truncating
//   casts) the subnormal value m*2^-9 to its FP32 bits.
// * All capture helpers return a TileCapture { status, kernel, diagnostics };
// invalid_argument / unsupported are recoverable and carry the capture or
// verifier diagnostics. Nothing throws.
//
// ---------------------------------------------------------------------------
// OPERATOR INVENTORY (one explicitly named factory each)
// ---------------------------------------------------------------------------
//  1. embedding gather from packed rows (bf16 / fp8+scales)
//  2. RMSNorm with DIRECT pre-added GGUF weight, optional residual epilogue
//  3. packed GEMM/GEMV (M >= 1) with optional residual epilogue
//  4. SwiGLU (split buffers or fused [M,2F] projection)
//  5. residual add / zero fill
//  6. attention: projection epilogue (Q/gate split + per-head QK norm with
// direct pre-added weight, gate passthrough), NeoX rotate-half partial RoPE,
//     KV append; causal GQA prefill (online FP32 softmax + sigmoid output
//     gate); single-query decode attention. Executor chain:
//     prepare -> rope(q) -> rope(k) -> kv_append -> prefill/decode.
//  7. GDN: depthwise conv4+SiLU with ring history, beta/decay gates, Q/K
//     FP32 L2 norm (+Q scale), FP32 delta-rule recurrence (decode & exact
//     recurrent prefill), gated output RMSNorm (direct weight, norm before
//     SiLU(z) gate)
//  8. CPU oracles for every operator + tile_ops_self_test() (device-free).
//
// NOT implemented here (reported, never silently faked):
//   * chunk-64 WY/UT GDN prefill, split-KV long-context decode reduction,
//     FP8/FP16 KV cache storage, sampling, fused gate/up GEMM epilogue
//     (SwiGLU is a separate pass), vision tower, MTP.

#include <cstdint>

#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/tile.h>

namespace luisa::qwen::tile_ops {

namespace t = luisa::compute::tile;

// ---------------------------------------------------------------------------
// Recoverable status / result (self-contained; mirrors dsl_ops conventions).
// ---------------------------------------------------------------------------
enum class TileOpStatus : uint32_t {
    ok = 0,
    invalid_argument,
    unsupported,
    capture_failed,
    overflow,
};
[[nodiscard]] luisa::string_view tile_op_status_name(TileOpStatus code) noexcept;

struct TileOpResult {
    TileOpStatus code{TileOpStatus::ok};
    luisa::string message;

    [[nodiscard]] bool ok() const noexcept { return code == TileOpStatus::ok; }
    [[nodiscard]] static TileOpResult success() noexcept { return {}; }
    [[nodiscard]] static TileOpResult error(TileOpStatus code, luisa::string message) {
        return TileOpResult{code, std::move(message)};
    }
};

// A captured, statically specialized Tile kernel plus its diagnostics.
// `kernel.valid()` is guaranteed when status.ok(); the caller compiles it
// with tile::compile and must check shader.metadata().error there.
struct TileCapture {
    TileOpResult status;
    luisa::optional<t::Kernel> kernel;
    luisa::vector<luisa::string> diagnostics;

    [[nodiscard]] bool ok() const noexcept { return status.ok() && kernel.has_value() && kernel->valid(); }
};

// ---------------------------------------------------------------------------
// Model geometry (Qwen3.5-4B text trunk; host constants, same as dsl_ops).
// ---------------------------------------------------------------------------
struct QwenTileConfig {
    uint32_t hidden{2560};
    uint32_t ffn{9216};
    uint32_t vocab{248320};
    // full attention
    uint32_t q_heads{16};
    uint32_t kv_heads{4};
    uint32_t attn_head_dim{256};
    uint32_t rope_dim{64}; // partial NeoX rotate-half rotary prefix

    float rope_theta{1.0e7f};
    // gated delta net
    uint32_t gdn_qk_heads{16};
    uint32_t gdn_v_heads{32};   // one QK head serves two consecutive V heads
    uint32_t gdn_head_dim{128};
    uint32_t conv_kernel{4};
    // numerics
    float norm_eps{1.0e-6f};
    float l2_eps{1.0e-6f};
    float softplus_threshold{20.0f};

    [[nodiscard]] uint32_t attn_q_proj_n() const noexcept { return 2u * q_heads * attn_head_dim; }
    [[nodiscard]] uint32_t attn_kv_n() const noexcept { return kv_heads * attn_head_dim; }
    [[nodiscard]] uint32_t attn_o_k() const noexcept { return q_heads * attn_head_dim; }
    [[nodiscard]] uint32_t gdn_qk_n() const noexcept { return gdn_qk_heads * gdn_head_dim; }
    [[nodiscard]] uint32_t gdn_v_n() const noexcept { return gdn_v_heads * gdn_head_dim; }
    [[nodiscard]] uint32_t gdn_qkv_n() const noexcept { return 2u * gdn_qk_n() + gdn_v_n(); }
    [[nodiscard]] float attn_scale() const noexcept;  // 1/sqrt(attn_head_dim)
    [[nodiscard]] float gdn_q_scale() const noexcept; // 1/sqrt(gdn_head_dim)
};
[[nodiscard]] TileOpResult validate_qwen_tile_config(const QwenTileConfig &cfg) noexcept;

// Weight format selection (explicit; never substituted silently).
enum class QwenTileWeightFormat : uint32_t {
    bf16 = 0,
    fp8_e4m3 = 1,
};
[[nodiscard]] luisa::string_view qwen_tile_weight_format_name(QwenTileWeightFormat f) noexcept;

// The only supported FP8 scale group (contract: FP32 scale per 64 K elements).
inline constexpr uint32_t kTileFp8Group = 64u;
// K-blocking of the packed GEMM contraction (one 64-wide FP8 scale group per
// step keeps scale application exact; also bounds per-program snapshots).
inline constexpr uint32_t kTileGemmKBlock = 64u;

// Packed word / scale counts (overflow-checked, fail-closed).
[[nodiscard]] uint64_t tile_packed_matrix_words(QwenTileWeightFormat f, uint64_t k, uint64_t n) noexcept;
[[nodiscard]] uint64_t tile_fp8_scale_elements(uint64_t k, uint64_t n) noexcept;

// ---------------------------------------------------------------------------
// Host codecs / packing (bit-exact, independent of the device decode paths;
// useful for converters and tests). Same ABI as dsl_ops/common.
// ---------------------------------------------------------------------------
[[nodiscard]] uint16_t tile_host_f32_to_bf16_bits(float v) noexcept;// RNE
[[nodiscard]] float tile_host_bf16_bits_to_f32(uint16_t bits) noexcept;
[[nodiscard]] uint8_t tile_host_f32_to_fp8_e4m3_bits(float v) noexcept;// RNE, saturating
[[nodiscard]] float tile_host_fp8_e4m3_bits_to_f32(uint8_t code) noexcept;
// Pack a dequantized FP32 [n,k] row-per-output matrix into the storage ABI.
// fp8 also fills scales (n*ceil(k/64)); scale = max_abs/448, zero groups get
// scale 0. Callers must report the FP8 conversion (it is lossy).
[[nodiscard]] TileOpResult tile_host_pack_matrix(QwenTileWeightFormat format,
                                                 luisa::span<const float> w,
                                                 uint64_t k, uint64_t n,
                                                 luisa::vector<uint32_t> &words,
                                                 luisa::vector<float> &scales);
// Decode one stored element (index = n*k + kk) exactly like the kernels do.
[[nodiscard]] float tile_host_decode_packed(QwenTileWeightFormat format,
                                            luisa::span<const uint32_t> words,
                                            luisa::span<const float> scales,
                                            uint64_t k, uint64_t index) noexcept;

// ---------------------------------------------------------------------------
// 1. Embedding gather. out[t,c] = decode(table[id[t], c]).
//    args (in order): ids Buffer<int32_t> [tokens],
//                     table Buffer<uint32_t> packed [vocab, hidden],
//                     scales Buffer<float> (fp8: vocab*ceil(hidden/64);
//                             bf16: bind a 1-element dummy),
//                     out Buffer<float> [tokens, hidden].
//    ids out of [0,vocab) are guarded: the row is zero-filled.
// ---------------------------------------------------------------------------
[[nodiscard]] TileCapture capture_tile_embedding(QwenTileWeightFormat format,
                                                 uint32_t tokens, uint32_t hidden, uint32_t vocab);

// ---------------------------------------------------------------------------
// 2. RMSNorm: y = (x[+residual]) * rsqrt(mean(s^2)+eps) * w.
// GGUF norm weights are PRE-ADDED (the converter folds HF's zero-centered
// 1+w offline): multiply DIRECTLY, never re-add 1.
//    plain    args: x [rows,dim], w [dim], y [rows,dim]
//    residual args: x [rows,dim], w [dim], residual [rows,dim],
//                   y [rows,dim], sum_out [rows,dim]  (sum_out = x+residual)
//    rows/dim are static; dim may be ragged w.r.t. the internal 128-wide
//    reduction tile (bounds::zero). QK norm reuses this with rows=T*heads.
// ---------------------------------------------------------------------------
enum class TileNormEpilogue : uint32_t { plain = 0, residual = 1 };
[[nodiscard]] TileCapture capture_tile_rmsnorm(TileNormEpilogue epilogue,
                                               uint32_t rows, uint32_t dim, float eps);

// ---------------------------------------------------------------------------
// 3. Packed GEMM/GEMV: y[m,n] = sum_k x[m,k]*decode(W[n,k]) (+residual[m,n]).
//    args: x Buffer<float> [m,k], W Buffer<uint32_t> packed [n,k],
//          scales Buffer<float> (fp8: n*ceil(k/64); bf16: 1-element dummy),
//          residual Buffer<float> [m,n] (only when epilogue=residual),
//          y Buffer<float> [m,n].
//    Static m,n,k; ragged M/N/K handled by bounds::zero tiles. Decode path:
//    one (tile_m x tile_n) program grid; K streamed serially in 64-blocks,
//    decoded to FP32 tiles and contracted with tile::mma (FP32 accumulate).
//    tile_m/tile_n are tuning knobs (defaults 16/16; tile_n=16 keeps the FP8
//    decoded-tile snapshot within the DX/VK 65536-byte/worker budget); m<=4
//    selects the same specialization (Tile has no separate GEMV; the planner
//    maps the grid). FP8 requires k % 4 == 0 (word-aligned packed rows).
// ---------------------------------------------------------------------------
enum class TileMatmulEpilogue : uint32_t { none = 0, residual = 1 };
struct TileGemmShape {
    uint32_t m{1}, n{0}, k{0};
    uint32_t tile_m{16}, tile_n{16};
    // M=1 realization selected by the executor. DX is fastest with one-row
    // MMA; Vulkan uses the reduction realization because the M=1 MMA route
    // has reproduced ERROR_DEVICE_LOST. Ignored for M>1.
    bool gemv_mma{false};
};
[[nodiscard]] TileCapture capture_tile_gemm(QwenTileWeightFormat format,
                                            TileMatmulEpilogue epilogue,
                                            const TileGemmShape &shape);
// K-segment partial GEMM (executor integration for large-K projections).
// Computes the partial product over K in [k_offset, k_offset+k_seg) and
// writes it to y [m, n] (NO residual epilogue). k_offset and k_seg must be
// multiples of kTileGemmKBlock (64) and k_offset+k_seg <= K. The caller
// accumulates segments (y += partial) and applies any residual after the
// final segment. This bounds the serial K loop per dispatch: the full-K
// single-dispatch GEMM at K=9216 (ffn_down) hangs / loses the device on the
// VK backend, while K<=4096 segments execute reliably. Same packed weight
// ABI as capture_tile_gemm. args: x [m, K], w (packed full [n, K]),
// scales (fp8 full [n, groups]; bf16 dummy), y [m, n].
[[nodiscard]] TileCapture capture_tile_gemm_kseg(QwenTileWeightFormat format,
                                                 const TileGemmShape &gemm_shape,
                                                 uint32_t k_offset, uint32_t k_seg);

// ---------------------------------------------------------------------------
// 4. SwiGLU: out = silu(gate) * up.
//    split args: gate [count], up [count], out [count]
//    fused args: proj [m, 2*f] (row = [gate f | up f]), out [m, f]
// ---------------------------------------------------------------------------
[[nodiscard]] TileCapture capture_tile_swiglu_split(uint32_t count);
[[nodiscard]] TileCapture capture_tile_swiglu_fused(uint32_t m, uint32_t f);

// ---------------------------------------------------------------------------
// 5. Elementwise utilities: out = a + b; out = 0. args [count] each.
// ---------------------------------------------------------------------------
[[nodiscard]] TileCapture capture_tile_residual_add(uint32_t count);
[[nodiscard]] TileCapture capture_tile_zero_fill(uint32_t count);

// ---------------------------------------------------------------------------
// 6. Full attention.
//
// q_proj layout per token: q_heads * [Q(hd) | gate(hd)] interleaved PER HEAD
// (N = 2*q_heads*hd). k/v_proj: [kv_heads*hd]. QK norm multiplies the DIRECT
// pre-added GGUF weight per head over hd, BEFORE RoPE. RoPE rotates only the
// first rope_dim elements as NeoX rotate-half pairs (i, i+rope_dim/2);
// inv_freq[i] = theta^(-2i/rope_dim); the tail passes through. Text mode:
// positions[t] is the absolute position (three mRoPE axes coincide).
// ---------------------------------------------------------------------------

// Projection epilogue, split into THREE host-selected kernels so every
// emitted kernel writes ONLY its own outputs with in-bounds stores (no
// out-of-bounds store tricks, no uniform-lane store selects). RoPE and KV
// append are SEPARATE kernels below; the chain is exact and keeps every
// kernel a single static grid.
// Q+gate: q_proj [tokens, 2*q_heads*hd] interleaved [Q(hd)|gate(hd)] per
// head, q_norm_w [hd] -> q_out [tokens, q_heads, hd] (direct pre-added
// weight normed, NOT roped), gate_out [tokens, q_heads, hd] (raw pass-through).
[[nodiscard]] TileCapture capture_tile_attn_prepare_q(const QwenTileConfig &cfg, uint32_t tokens);
// K: k_proj [tokens, kv_heads*hd], k_norm_w [hd] ->
// k_out [tokens, kv_heads, hd] (normed, NOT roped).
[[nodiscard]] TileCapture capture_tile_attn_prepare_k(const QwenTileConfig &cfg, uint32_t tokens);
// V: v_proj [tokens, kv_heads*hd] -> v_out [tokens, kv_heads, hd] (raw copy).
[[nodiscard]] TileCapture capture_tile_attn_prepare_v(const QwenTileConfig &cfg, uint32_t tokens);

// Standalone NeoX rotate-half partial RoPE over x [tokens, heads, hd].
// TEST-ORACLE variant: evaluates sin/cos on device with the shared bounded
// polynomial (scalar_sin_cos / tile_host_rope_sin_cos). Model inference uses
// the exact host-table variant below instead.
// args: x, positions Buffer<int32_t> [tokens],
// inv_freq Buffer<float> [rope_dim/2] (host-precomputed
// theta^(-2i/rope_dim); see tile_host_rope_inv_freq),
// out [tokens, heads, hd].
// In-place (x == out) is NOT allowed through the runtime's disjoint-writes
// check on some routes; use separate buffers.
[[nodiscard]] TileCapture capture_tile_rope_partial(uint32_t tokens, uint32_t heads,
                                                    uint32_t head_dim, uint32_t rope_dim);
// Executor variant: NeoX rotate-half partial RoPE from a HOST-precomputed FP32
// sin/cos table indexed by ABSOLUTE position. No device sin/cos evaluation.
// cos_tab/sin_tab: [max_context, rope_dim/2] FP32, entries
// cos/sin(position * inv_freq[pair]) computed on the host in double and
// rounded once (see tile_host_rope_tables). positions[t] must be in
// [0, max_context). The tail (dd >= rope_dim) passes through unchanged.
// args: x [tokens, heads, hd], positions Buffer<int32_t> [tokens],
// cos_tab, sin_tab, out [tokens, heads, hd]. x == out is not allowed.
[[nodiscard]] TileCapture capture_tile_rope_partial_table(uint32_t tokens, uint32_t heads,
                                                          uint32_t head_dim, uint32_t rope_dim,
                                                          uint32_t max_context);
// Host helper: inv_freq[i] = theta^(-2i/rope_dim), i in [0, rope_dim/2).
[[nodiscard]] TileOpResult tile_host_rope_inv_freq(uint32_t rope_dim, float theta,
                                                   luisa::vector<float> &inv_freq);
// Host helper: fill cos_tab/sin_tab [max_context, rope_dim/2] with
// cos/sin(pos * inv_freq[i]) (double-precision angle, single FP32 rounding).
[[nodiscard]] TileOpResult tile_host_rope_tables(uint32_t rope_dim, float theta,
                                                 uint32_t max_context,
                                                 luisa::vector<float> &cos_tab,
                                                 luisa::vector<float> &sin_tab);

// KV append: copy k/v [tokens, channels] into caches [capacity, channels] at
// rows pos0+t. args: k, v, k_cache, v_cache.
[[nodiscard]] TileCapture capture_tile_kv_append(uint32_t tokens, uint32_t channels,
                                                 uint32_t pos0, uint32_t capacity);

// Causal GQA prefill attention with sigmoid output gate. Query t (absolute
// position pos0+t) attends cache keys [0, pos0+t]; kv_len valid cache rows
// (kv_len >= pos0+tokens after append). One program per (token, q_head);
// keys streamed serially with an online FP32 softmax (running max/sum/acc).
// args: q [tokens, q_heads, hd], gate [tokens, q_heads, hd],
//       k_cache [capacity, kv_heads, hd], v_cache [capacity, kv_heads, hd],
//       out [tokens, q_heads, hd].
[[nodiscard]] TileCapture capture_tile_attn_prefill(const QwenTileConfig &cfg, uint32_t tokens,
                                                    uint32_t pos0, uint32_t kv_len,
                                                    uint32_t capacity);

// Single-query decode attention: one program per q_head, serial online
// softmax over kv_len cache rows, sigmoid gate applied at the end.
// args: q [q_heads, hd], gate [q_heads, hd], k_cache, v_cache
// [capacity, kv_heads, hd], out [q_heads, hd]. kv_len >= 1.
[[nodiscard]] TileCapture capture_tile_attn_decode(const QwenTileConfig &cfg,
                                                   uint32_t kv_len, uint32_t capacity);

// Device-visible-length variants (executor integration): ONE compiled shader
// serves every position. The serial key loop streams the FULL capacity and
// gates each key with a limit read from a device buffer, so no per-position
// JIT is needed. Same math as the static captures above (scalar online
// softmax carries; the serial ite(active,...) selects are the verified-safe
// scalar pattern, not the miscompiled uniform tile-lane select).
// prefill args: q, gate [tokens, q_heads, hd], k_cache, v_cache
// [capacity, kv_heads, hd], pos0_len Buffer<int32_t> [2] = {pos0, kv_len},
// out [tokens, q_heads, hd]. Caller guarantees pos0+tokens <= kv_len <=
// capacity (the kernel clamps the causal diagonal to kv_len defensively).
[[nodiscard]] TileCapture capture_tile_attn_prefill_dynamic(const QwenTileConfig &cfg,
                                                            uint32_t tokens, uint32_t capacity);
// decode args: q, gate [q_heads, hd], k_cache, v_cache, kv_len_buf
// Buffer<int32_t> [1] = {kv_len}, out [q_heads, hd]. 1 <= kv_len <= capacity.
[[nodiscard]] TileCapture capture_tile_attn_decode_dynamic(const QwenTileConfig &cfg,
                                                           uint32_t capacity);

// LENGTH-BUCKET capacity of the two captures above: the STATIC key range their
// serial loop streams. Every capture requires pos0+tokens <= kv_len <= the
// capacity it was captured with, and its cost is that capacity -- so a
// full-capacity specialization charges max_context keys per query even for a
// 20-token session. This helper is the ONE policy both the executor and the
// tests use: `bucket_keys == 0` means "no bucketing" and returns `capacity`;
// otherwise the result is ceil(kv_len/bucket_keys)*bucket_keys clamped to
// `capacity` (the clamp only applies to a partially filled last bucket, and the
// result is never below kv_len when kv_len <= capacity). The number of distinct
// results is bounded by ceil(capacity/bucket_keys)+1, which is what bounds the
// compiled shader family.
[[nodiscard]] uint32_t tile_attention_bucket_capacity(uint32_t kv_len, uint32_t bucket_keys,
                                                      uint32_t capacity) noexcept;

// ---------------------------------------------------------------------------
// 7. Gated DeltaNet.
//
// Packed QKV layout per token: [Q(2048) | K(2048) | V(4096)] (16 QK heads *
// 128, 32 V heads * 128). Conv weight is the GGUF ssm_conv1d tensor
// [kernel=4, channels] whose payload is CHANNEL-major: weight[c*kernel + j]
// is tap j of channel c. Ring history holds the last kernel-1
// PRE-activation values per channel: ring[c, s], s = pos % (kernel-1);
// taps for positions < pos0 come from the ring, positions >= pos0 from src;
// src is pushed into the ring after the output (causal).
// Gates: beta = sigmoid(b); g = ssm_a[h] * softplus(a + dt_bias[h]) -- GGUF
// ssm_a is ALREADY -exp(A_log) (negative, folded offline), no device exp.
// Q/K FP32 L2 norm x*rsqrt(sum(x*x)+eps) per QK head (FLA/HF: eps INSIDE the
// rsqrt); Q scaled by 1/sqrt(128).
// V head h uses QK head h % qk_heads (ggml tiled broadcast, NOT h/(vh/qk)).
// State S[k,v] [v_heads, hd, hd] FP32. Per token:
//   S <- exp(g)*S; err = v - S^T k; S += beta * k err^T; o = S^T q.
// Output norm: y = o*rsqrt(mean(o^2)+eps) * norm_w * silu(z) (DIRECT weight,
// norm BEFORE gate).
// ---------------------------------------------------------------------------

// One-token Conv4+SiLU step with SEPARATE ring buffers (ping-pong). There is
// NO in-place ring read/write: ring_in is read-only history, ring_out is a
// full copy of ring_in with ONLY slot pos_mod overwritten by the current
// PRE-activation source row. The caller owns two rings per layer, alternates
// them per token, and zero-fills both on reset. Exactly THREE capture
// variants exist per (channels, kernel_size), keyed by pos_mod = pos %
// (kernel_size-1) -- no unbounded position-specialized compiles.
// Tap semantics for absolute position pos (R = kernel-1, pos_mod = pos % R):
// tap j in [0,R) reads position pos-R+j from ring_in slot (pos_mod+j)%R
// (always non-negative), masked to 0 when pos-R+j < 0; tap j==R reads the
// current source row. pos is supplied device-visible so the negative-position
// mask is exact even though pos_mod alone cannot distinguish pos=0 from pos=R.
// args: src [1, channels] (current pre-activation row),
// weight [channels, kernel] CHANNEL-major (weight[c*kernel+j]; the GGUF
// ssm_conv1d [kernel, channels] payload), ring_in [channels, R] (read-only),
// pos_buf Buffer<int32_t> [1] = {pos},
// ring_out [channels, R] (write-only full copy), out [1, channels] (SiLU).
[[nodiscard]] TileCapture capture_tile_gdn_conv_silu_step(uint32_t channels,
                                                          uint32_t kernel_size,
                                                          uint32_t pos_mod);

// MULTI-TOKEN Conv4+SiLU chunk (prefill integration). ONE kernel produces the
// SiLU output of `tokens` consecutive rows AND the ring image after them, so a
// prefill chunk costs one launch instead of one per token and needs no
// per-token source/output staging copy. Same math, same weight ABI, same ring
// ABI as the one-token step above (this capture is its generalization; the step
// remains the per-token differential/decode path).
//
// Causal addressing for row t at ABSOLUTE position pos0+t, taps j in
// [0, kernel_size), source position s = pos0+t-(kernel_size-1)+j:
//   s >= pos0      -> src row (s - pos0)  (already inside this chunk)
//   0 <= s < pos0  -> ring_in[c, s % (kernel_size-1)] (history before the chunk)
//   s < 0          -> 0
// The kernel carries the (kernel_size-1)-wide window in SSA registers, so the
// distinction above needs no per-row bound test beyond the initial window: the
// window starts as history positions [pos0-R, pos0) and each row shifts in the
// row it just consumed. Position dependence beyond that is only the initial
// window's ring-slot rotation (pos0 % R) and the negative-position mask, both
// from a DEVICE-VISIBLE pos0 (one int32 upload per chunk, not per token).
// ring_out is a full image of ring_in with the last min(tokens, R) slots
// overwritten (in absolute-position order, so the newest row wins); it is a
// separate buffer from ring_in (write-only), so no invocation reads and writes
// one buffer.
// args: src [tokens, channels] (packed pre-activation rows, contiguous),
//       weight [channels, kernel] CHANNEL-major (weight[c*kernel+j] = tap j of
//       channel c, the GGUF ssm_conv1d [kernel, channels] payload),
//       ring_in [channels, R] (read-only), pos_buf Buffer<int32_t> [1] = {pos0},
//       ring_out [channels, R] (write-only), out [tokens, channels] (SiLU).
// pos0 must be the absolute position of row 0 and non-negative.
[[nodiscard]] TileCapture capture_tile_gdn_conv_silu_chunk(uint32_t channels,
                                                           uint32_t kernel_size,
                                                           uint32_t tokens);

// beta/g gates. args: a [tokens, heads], b [tokens, heads],
// ssm_a [heads], dt_bias [heads], beta_out [tokens, heads],
// g_out [tokens, heads].
[[nodiscard]] TileCapture capture_tile_gdn_gates(uint32_t tokens, uint32_t heads,
                                                 float softplus_threshold);

// Q/K L2 norm + Q scale over the conv output, split into TWO host-selected
// kernels with in-bounds stores only (no out-of-bounds store tricks).
// Q: conv_out [tokens, 2*qk_heads*hd + v_channels] (Q section at column 0),
// q_out [tokens, qk_heads, hd] = x * rsqrt(sum(x*x)+eps) * q_scale.
[[nodiscard]] TileCapture capture_tile_gdn_q_l2norm(uint32_t tokens, uint32_t qk_heads,
                                                    uint32_t head_dim, uint32_t v_channels,
                                                    float eps, float q_scale);
// K: same conv_out (K section at column qk_heads*hd),
// k_out [tokens, qk_heads, hd] = x * rsqrt(sum(x*x)+eps), no extra scale.
// V is untouched (recurrence reads it from the packed conv output at
// v_offset + h*hd).
[[nodiscard]] TileCapture capture_tile_gdn_k_l2norm(uint32_t tokens, uint32_t qk_heads,
                                                    uint32_t head_dim, uint32_t v_channels,
                                                    float eps);

// FP32 delta-rule recurrence over `tokens` (1 = decode, >1 exact recurrent
// prefill). One program per V head; the [hd,hd] state tile is carried in SSA
// through the serial token loop (loaded once, written back once).
// args: q [tokens, qk_heads, hd], k [tokens, qk_heads, hd],
//       conv_out [tokens, qkv_channels] (V source),
//       beta [tokens, v_heads], g [tokens, v_heads],
//       state [v_heads, hd, hd] (in/out; MUST be a separate buffer from all
//       read-only args on routes with disjoint-writes checks -- in-place
//       read+write of the SAME argument is fine),
//       out [tokens, v_heads, hd].
[[nodiscard]] TileCapture capture_tile_gdn_recurrent(uint32_t tokens, uint32_t v_heads,
                                                     uint32_t qk_heads, uint32_t head_dim);

// Column-parallel delta-rule recurrence (executor integration). Same math and
// argument layout as capture_tile_gdn_recurrent, but the [hd,hd] state stays
// in the caller's buffer and one program owns a single (v_head, v_column)
// state column with SCALAR carries only. This avoids the DX/VK XIR per-worker
// snapshot budget that the [hd,hd] SSA tile carry exceeds at head_dim=128.
// The state buffer is read+write in place (columns are disjoint across
// programs). Prefer this variant on the XIR route at model head_dim.
[[nodiscard]] TileCapture capture_tile_gdn_recurrent_columns(uint32_t tokens, uint32_t v_heads,
                                                             uint32_t qk_heads, uint32_t head_dim);

// Chunk-64 WY/UT prefill is NOT implemented; reports the gap explicitly.
[[nodiscard]] TileOpResult tile_ops_chunk_prefill_status() noexcept;

// Gated output RMSNorm. args: o [tokens, heads, hd], z [tokens, heads*hd],
// norm_w [hd], out [tokens, heads, hd].
[[nodiscard]] TileCapture capture_tile_gdn_gated_norm(uint32_t tokens, uint32_t heads,
                                                      uint32_t head_dim, float eps);

// ---------------------------------------------------------------------------
// 8. Independent CPU oracles (double accumulation where reduction order is
//    unspecified; scalar reference order otherwise). No device/DSL calls.
//    Layouts and arguments mirror the kernels exactly.
// ---------------------------------------------------------------------------
[[nodiscard]] TileOpResult ref_tile_embedding(QwenTileWeightFormat format,
                                              luisa::span<const int32_t> ids,
                                              luisa::span<const uint32_t> table,
                                              luisa::span<const float> scales,
                                              uint32_t hidden, uint32_t vocab,
                                              luisa::vector<float> &out);
[[nodiscard]] TileOpResult ref_tile_rmsnorm(luisa::span<const float> x, luisa::span<const float> w,
                                            luisa::span<const float> residual, bool has_residual,
                                            uint32_t rows, uint32_t dim, float eps,
                                            luisa::vector<float> &y, luisa::vector<float> &sum_out);
[[nodiscard]] TileOpResult ref_tile_matmul(QwenTileWeightFormat format, luisa::span<const float> x,
                                           luisa::span<const uint32_t> w, luisa::span<const float> scales,
                                           luisa::span<const float> residual, bool has_residual,
                                           uint32_t m, uint32_t n, uint32_t k,
                                           luisa::vector<float> &y);
[[nodiscard]] TileOpResult ref_tile_swiglu(luisa::span<const float> gate, luisa::span<const float> up,
                                           luisa::vector<float> &out);
[[nodiscard]] TileOpResult ref_tile_swiglu_fused(luisa::span<const float> proj, uint32_t m, uint32_t f,
                                                 luisa::vector<float> &out);
// RoPE reference uses the SAME range reduction + Taylor polynomial as the
// kernels (see tile_host_rope_sin_cos) so comparisons are tight.
[[nodiscard]] void tile_host_rope_sin_cos(float angle, float &sin_out, float &cos_out) noexcept;
[[nodiscard]] TileOpResult ref_tile_attn_prepare(luisa::span<const float> q_proj,
                                                 luisa::span<const float> k_proj,
                                                 luisa::span<const float> v_proj,
                                                 luisa::span<const float> q_norm_w,
                                                 luisa::span<const float> k_norm_w,
                                                 uint32_t tokens, const QwenTileConfig &cfg,
                                                 luisa::vector<float> &q, luisa::vector<float> &gate,
                                                 luisa::vector<float> &k, luisa::vector<float> &v);
[[nodiscard]] TileOpResult ref_tile_kv_append(luisa::span<const float> k, luisa::span<const float> v,
                                              uint32_t tokens, uint32_t channels, uint32_t pos0,
                                              luisa::span<float> k_cache, luisa::span<float> v_cache);
[[nodiscard]] TileOpResult ref_tile_rope_partial(luisa::span<const float> x,
                                                 luisa::span<const int32_t> positions,
                                                 uint32_t tokens, uint32_t heads, uint32_t head_dim,
                                                 uint32_t rope_dim, float theta,
                                                 luisa::vector<float> &out);
[[nodiscard]] TileOpResult ref_tile_attn_prefill(luisa::span<const float> q, luisa::span<const float> gate,
                                                 luisa::span<const float> k_cache,
                                                 luisa::span<const float> v_cache,
                                                 uint32_t tokens, uint32_t q_heads, uint32_t kv_heads,
                                                 uint32_t head_dim, float scale, uint32_t pos0,
                                                 uint32_t kv_len, luisa::vector<float> &out);
[[nodiscard]] TileOpResult ref_tile_attn_decode(luisa::span<const float> q, luisa::span<const float> gate,
                                                luisa::span<const float> k_cache,
                                                luisa::span<const float> v_cache,
                                                uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                                                float scale, uint32_t kv_len, luisa::vector<float> &out);
[[nodiscard]] TileOpResult ref_tile_gdn_conv(luisa::span<const float> src, luisa::span<const float> weight,
                                             luisa::span<float> ring, uint32_t tokens, uint32_t channels,
                                             uint32_t kernel_size, uint32_t pos0,
                                             luisa::vector<float> &out);
[[nodiscard]] TileOpResult ref_tile_gdn_gates(luisa::span<const float> a, luisa::span<const float> b,
                                              luisa::span<const float> ssm_a,
                                              luisa::span<const float> dt_bias,
                                              uint32_t tokens, uint32_t heads, float softplus_threshold,
                                              luisa::vector<float> &beta, luisa::vector<float> &g);
[[nodiscard]] TileOpResult ref_tile_gdn_qk_norm(luisa::span<const float> conv_out, uint32_t tokens,
                                                uint32_t qk_heads, uint32_t head_dim, uint32_t v_channels,
                                                float eps, float q_scale,
                                                luisa::vector<float> &q, luisa::vector<float> &k);
[[nodiscard]] TileOpResult ref_tile_gdn_recurrent(luisa::span<const float> q, luisa::span<const float> k,
                                                  luisa::span<const float> conv_out,
                                                  luisa::span<const float> beta, luisa::span<const float> g,
                                                  luisa::span<float> state, uint32_t tokens,
                                                  uint32_t v_heads, uint32_t head_dim, uint32_t qk_heads,
                                                  luisa::vector<float> &out);
[[nodiscard]] TileOpResult ref_tile_gdn_gated_norm(luisa::span<const float> o, luisa::span<const float> z,
                                                   luisa::span<const float> norm_w, uint32_t tokens,
                                                   uint32_t heads, uint32_t head_dim, float eps,
                                                   luisa::vector<float> &out);

// ---------------------------------------------------------------------------
// 9. Device-free self-test: host codec round-trips, CPU oracle consistency
//    (ragged shapes, partial FP8 groups, zero groups, conv history across
//    chunk boundaries, GQA mapping, partial RoPE tails, delta-rule
//    recurrence) AND capture validity of every factory at small static
//    shapes (TileIR verifier only; no Device). `report` gets one line per
//    check. Returns ok() when everything passes.
// ---------------------------------------------------------------------------
[[nodiscard]] TileOpResult tile_ops_self_test(luisa::vector<luisa::string> *report = nullptr);

}// namespace luisa::qwen::tile_ops
