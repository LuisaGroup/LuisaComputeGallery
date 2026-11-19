// Inference benchmark — category "inference".
//
// Four benchmarks that model the hot paths of a transformer/attention inference
// stack with ordinary LuisaCompute DSL compute kernels (no lc-tile, no
// cooperative-vector / native-shader extension): every kernel is plain DSL, so
// it lowers through the normal AST -> XIR -> HLSL/SPIR-V pipeline and behaves
// the same on dx/vk/cuda.
//
//   1. infer_gemm_f32      Register-blocked shared-memory FP32 GEMM (M = N = K
//                          = 1024) with a fused bias + GELU-tanh epilogue.
//   2. infer_mha_flash     Multi-head online-softmax ("flash") attention,
//                          batch 16, heads 8, seq 512, head_dim 64, with the
//                          K/V tile broadcast through shared memory.
//   3. infer_mla_attention Multi-Head Latent Attention pipeline: Q projection
//                          with fused RoPE, KV compression + decoupled RoPE
//                          keys, then absorbed latent-space online attention
//                          with a single W_uv up-projection per query row.
//   4. infer_norm_mlp      Transformer FFN block: two-pass shared-memory
//                          LayerNorm -> tiled GEMM (1024 -> 4096, bias + GELU)
//                          -> tiled GEMM (4096 -> 1024, bias + residual add).
//
// Each benchmark compiles its kernels uncached through bench::compile_uncached()
// (so compile time is reported but nothing is reused between runs), verifies a
// spot-checked subset of the output against an exact CPU reference before the
// timed loop, and reports FLOP/s through bench::run_timed_bench(). The
// verification dispatches happen *before* timing, so the reported latency
// contains only the measured work.
//
// Conventions (AGENTS.md): C++23, no exceptions and no error codes (all geometry
// contracts are static_asserts, and a numeric mismatch is reported through
// BenchResult instead of failing the run), `luisa::span` for copy_from/copy_to,
// `&`/`|` for DSL Bool logic, and all problem dimensions at namespace scope so
// they stay constant expressions inside the DSL lambdas (they are never
// captured, hence usable as compile-time indices such as in $array<float, N>).
//
// The kernels are modelled on the verified reference implementations in
// src/ml/attention_kernels.cpp (create_mha_online_attention_kernel /
// create_online_attention_kernel, incl. the shared-memory broadcast of the
// latent/positional K tiles), src/embedding/embed/kernels/kernels_gemm.cpp
// (register-blocked tile + +1 padded shared tiles) and
// src/embedding/embed/kernels/kernels_norm.cpp (shared-memory halving-tree
// LayerNorm, deliberately not warp collectives so every backend can run it).

#include "bench_common.h"

#include <luisa/dsl/sugar.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace luisa::bench {

namespace {

// The DSL builtins (set_block_size, sync_block, block_id, shared, ...) are found
// by ADL on Var<>, but the bare calls in the kernels below follow the engine
// convention of a using-directive in the kernel-building scope.
using namespace luisa::compute;

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

// Deterministic host-side fill in [lo, hi) (the benchmark PRNG lives in
// bench_common.h so every category draws the same streams).
[[nodiscard]] std::vector<float> make_random_vector(size_t count, uint64_t &state,
                                                    float lo, float hi) {
    std::vector<float> data(count);
    for (auto &v : data) { v = prng_f32(state, lo, hi); }
    return data;
}

// Same distribution as make_random_vector(), scaled by `scale`, for weight
// matrices: a dot product of length K over U(-1, 1) * U(-1, 1) inputs then has
// unit-ish magnitude instead of growing like sqrt(K).
[[nodiscard]] std::vector<float> make_random_matrix(size_t count, uint64_t &state,
                                                    float scale) {
    std::vector<float> data(count);
    for (auto &v : data) { v = prng_f32(state, -1.0f, 1.0f) * scale; }
    return data;
}

// Exact GELU-tanh (the same closed form the kernels evaluate on device):
//   0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
[[nodiscard]] inline double gelu_tanh_ref(double x) noexcept {
    constexpr double kAlpha = 0.79788456080286535588;// sqrt(2 / pi)
    constexpr double kBeta = 0.044715;
    return 0.5 * x * (1.0 + std::tanh(kAlpha * (x + kBeta * x * x * x)));
}

// Spot-check bookkeeping shared by the four benchmarks: how many output
// elements were compared, the largest absolute deviation from the CPU
// reference, and whether every visited value was finite.
struct CheckSummary {
    uint32_t elements{0u};
    double max_abs_diff{0.0};
    bool finite{true};
    bool within_tolerance{true};
};

[[nodiscard]] inline bool close_enough(double got, double ref,
                                       double abs_tol, double rel_tol) noexcept {
    return std::abs(got - ref) <= abs_tol + rel_tol * std::abs(ref);
}

// One checked element: records finiteness, the deviation and the tolerance.
inline void accumulate_check(CheckSummary &check, double got, double ref,
                             double abs_tol, double rel_tol) noexcept {
    check.elements++;
    if (!std::isfinite(got)) {
        check.finite = false;
        check.within_tolerance = false;
        return;
    }
    auto diff = std::abs(got - ref);
    check.max_abs_diff = std::max(check.max_abs_diff, diff);
    if (!close_enough(got, ref, abs_tol, rel_tol)) {
        check.within_tolerance = false;
    }
}

[[nodiscard]] luisa::string make_check_note(const CheckSummary &check,
                                            double abs_tol, double rel_tol) {
    return luisa::format("{} spot elements, max|diff|={:.3e}, tol=abs {:.1e}/rel {:.1e}, finite={}",
                         check.elements, check.max_abs_diff, abs_tol, rel_tol, check.finite);
}

[[nodiscard]] inline bool check_passed(const CheckSummary &check) noexcept {
    return check.finite && check.within_tolerance && check.elements > 0u;
}

// ---------------------------------------------------------------------------
// Register-blocked shared-memory GEMM with a bias epilogue
// (+ GELU-tanh, or + residual add), shared by benchmarks 1 and 4.
// ---------------------------------------------------------------------------

enum struct GemmEpi : uint32_t {
    bias_gelu_tanh,// C = gelu(A * B + bias)
    bias_residual, // C = A * B + bias + aux   (aux is the residual tensor)
};

// Geometry of one register-blocked variant (same numbers as the reference
// register GEMM in src/embedding/embed/kernels/kernels_gemm.cpp::RegTile).
//   - `block` threads form a (TM / RM) x (TN / RN) grid of thread tiles; each
//     thread owns an RM x RN register accumulator.
//   - The thread's RN columns are interleaved with stride `cols`, so the global
//     stores of a warp stay coalesced.
//   - Both shared tiles get a +1 element row pad: without it the row stride is a
//     multiple of 32 floats and consecutive rows of the tile collide in the same
//     shared-memory bank for every thread of a column group.
template<uint32_t TM, uint32_t TN, uint32_t TK, uint32_t RM, uint32_t RN>
struct RegGemmTile {
    static constexpr uint32_t cols = TN / RN;          // thread tiles along N
    static constexpr uint32_t rows = TM / RM;          // thread tiles along M
    static constexpr uint32_t block = rows * cols;      // threads per block
    static constexpr uint32_t a_elems = TM * TK;        // logical A tile elements
    static constexpr uint32_t b_elems = TK * TN;        // logical B tile elements
    static constexpr uint32_t a_stride = TK + 1u;       // +1: rows hit different banks
    static constexpr uint32_t b_stride = TN + 1u;       // +1: rows hit different banks
    static constexpr uint32_t a_sh = TM * a_stride;     // padded A tile floats
    static constexpr uint32_t b_sh = TK * b_stride;     // padded B tile floats
    static_assert(TM % RM == 0u && TN % RN == 0u, "tile must split into register sub-tiles");
    static_assert(block >= 32u && block <= 1024u && block % 32u == 0u, "register GEMM block size");
    static_assert(a_elems % block == 0u && b_elems % block == 0u, "cooperative fill must divide evenly");
    static_assert((a_sh + b_sh) * 4u <= 32768u, "GEMM tile exceeds the 32 KiB DX shared floor");
};

// DSL GELU-tanh. No native GELU instruction exists, but TANH is a CallOp with
// codegen on every backend (hlsl "tanh", spv OpTanh), so the exact analytic form
// is emitted instead of an erf approximation.
[[nodiscard]] inline Float gelu_tanh_dsl(Float x) noexcept {
    constexpr float kAlpha = 0.79788456080286535588f;// sqrt(2 / pi)
    constexpr float kBeta = 0.044715f;
    Float x3 = x * x * x;
    Float inner = x + kBeta * x3;
    Float arg = kAlpha * inner;
    Float y = 0.5f * x * (1.0f + compute::tanh(arg));
    return y;
}

// Generic register-blocked GEMM kernel. M/N/K and the tile geometry are compile
// time constants (baked into the kernel), so the inner slab loop has no bounds
// checks and the accumulator/register-fragment loops are unrolled at emission
// time (there is no XIR unroll pass; see lc_optimize §10.6).
// Params: A([M][K]), B([K][N]), bias([N]), aux([M][N], residual epilogue only),
//         C([M][N]).
using RegGemmKernel = Kernel1D<void(Buffer<float>, Buffer<float>, Buffer<float>,
                                    Buffer<float>, Buffer<float>)>;

template<uint32_t M, uint32_t N, uint32_t K, uint32_t TM, uint32_t TN, uint32_t TK,
         uint32_t RM, uint32_t RN, GemmEpi EPI>
[[nodiscard]] RegGemmKernel make_reg_gemm_kernel() noexcept {
    using Tile = RegGemmTile<TM, TN, TK, RM, RN>;
    static_assert(M % TM == 0u && N % TN == 0u && K % TK == 0u,
                  "the register GEMM requires the problem to be a whole number of tiles");

    return [](BufferFloat A, BufferFloat B, BufferFloat bias,
              BufferFloat aux, BufferFloat C) noexcept {
        set_block_size(Tile::block, 1u, 1u);
        set_name(EPI == GemmEpi::bias_gelu_tanh ? "bench_reg_gemm_bias_gelu" : "bench_reg_gemm_bias_residual");

        // Padded shared tiles: [TM][TK + 1] and [TK][TN + 1].
        Shared<float> tile_a{Tile::a_sh};
        Shared<float> tile_b{Tile::b_sh};

        Var tid = thread_x();
        Var bid = block_id().x;
        // Both grid dimensions are compile time: N / TN blocks along N, then along M.
        Var m0 = (bid / (N / TN)) * TM;
        Var n0 = (bid % (N / TN)) * TN;
        Var grow = tid / Tile::cols;// row-group of this thread (0 .. rows - 1)
        Var gcol = tid % Tile::cols;// interleaved column index (0 .. cols - 1)
        Var row0 = m0 + grow * RM;
        Var nb = n0 + gcol;

        // DSL locals are zero-initialized (Var's default constructor applies the
        // default initializer), so the accumulator needs no explicit clear.
        Var<float> acc[RM][RN];

        $for (ks, K / TK) {
            Var k0 = ks * TK;

            // Cooperative fill of both tiles: every thread stages
            // a_elems / block and b_elems / block elements.
            for (auto e = 0u; e < Tile::a_elems / Tile::block; e++) {
                Var e_idx = tid + e * Tile::block;
                Var r = e_idx / TK;
                Var c = e_idx % TK;
                tile_a[r * Tile::a_stride + c] = A.read((m0 + r) * K + k0 + c);
            }
            for (auto e = 0u; e < Tile::b_elems / Tile::block; e++) {
                Var e_idx = tid + e * Tile::block;
                Var r = e_idx / TN;// K row inside the slab
                Var c = e_idx % TN;// column along N (coalesced across the warp)
                tile_b[r * Tile::b_stride + c] = B.read((k0 + r) * N + n0 + c);
            }
            sync_block();

            // Register-tile FMA. `kk`/`r`/`q` are host-known loop counters, so
            // every access below folds into a constant-index shared load.
            Var<float> bfrag[RN];
            Var<float> afrag;
            for (auto kk = 0u; kk < TK; kk++) {
                for (auto q = 0u; q < RN; q++) {
                    bfrag[q] = tile_b[kk * Tile::b_stride + gcol + q * Tile::cols];
                }
                for (auto r = 0u; r < RM; r++) {
                    afrag = tile_a[(grow * RM + r) * Tile::a_stride + kk];
                    for (auto q = 0u; q < RN; q++) {
                        acc[r][q] += afrag * bfrag[q];
                    }
                }
            }
            // Everyone must be done reading before the next slab overwrites the tiles.
            sync_block();
        };

        // Epilogue. Bias is a per-column constant: one read per thread column
        // (RM reuses per read) instead of one per output element.
        Var<float> bcol[RN];
        for (auto q = 0u; q < RN; q++) {
            bcol[q] = bias.read(nb + q * Tile::cols);
        }
        for (auto r = 0u; r < RM; r++) {
            for (auto q = 0u; q < RN; q++) {
                Var out_idx = (row0 + r) * N + nb + q * Tile::cols;
                Float v = acc[r][q] + bcol[q];
                if constexpr (EPI == GemmEpi::bias_gelu_tanh) {
                    static_cast<void>(aux);// no residual in this epilogue
                    C.write(out_idx, gelu_tanh_dsl(v));
                } else {
                    C.write(out_idx, v + aux.read(out_idx));
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Benchmark 1 dimensions: single-shot FP32 GEMM + bias + GELU-tanh.
// ---------------------------------------------------------------------------
namespace gemm_f32_dims {

constexpr uint32_t m = 1024u;
constexpr uint32_t n = 1024u;
constexpr uint32_t k = 1024u;

// Block tile 64x64 over 16-deep K slabs, 8x4 registers per thread, 128 threads.
// Shared footprint: (64 * 17 + 16 * 65) * 4 B = 8.5 KiB, well under the
// 32 KiB DX static-shared floor.
constexpr uint32_t tile_m = 64u;
constexpr uint32_t tile_n = 64u;
constexpr uint32_t tile_k = 16u;
constexpr uint32_t reg_m = 8u;
constexpr uint32_t reg_n = 4u;
constexpr uint32_t block = (tile_m / reg_m) * (tile_n / reg_n);// 128

constexpr double work_units = 2.0 * 1024.0 * 1024.0 * 1024.0;// 2 * M * N * K

}// namespace gemm_f32_dims

// ---------------------------------------------------------------------------
// Benchmark 2: multi-head online-softmax (flash) attention.
//
// Directly modelled on create_mha_online_attention_kernel() in
// src/ml/attention_kernels.cpp: one thread per (b, h, i) query row, the query
// row hoisted into registers, and the K/V tile broadcast through shared memory
// so each staged tile is amortized over the whole block instead of per thread.
// The online max/sum update keeps the softmax numerically identical to a dense
// softmax while touching every key exactly once.
// ---------------------------------------------------------------------------
namespace mha_dims {

constexpr uint32_t batch = 16u;
constexpr uint32_t heads = 8u;
constexpr uint32_t seq = 512u;
constexpr uint32_t dim = 64u;
constexpr uint32_t block = 128u;  // 4 warps per block, one block spans 128 query rows
constexpr uint32_t tile = 16u;    // keys staged per shared K/V tile
constexpr float scale = 0.125f;   // 1 / sqrt(64)
constexpr uint32_t tile_elems = tile * dim;
constexpr uint32_t tiles = seq / tile;
constexpr uint32_t rows = batch * heads * seq;

constexpr double work_units = double(16ull * 8ull * 512ull * 512ull * 64ull * 4ull);

static_assert(seq % tile == 0u, "seq must be a whole number of K/V tiles");
static_assert(seq % block == 0u, "a block must cover whole query rows of one (b, h)");
static_assert(tile_elems % block == 0u, "the cooperative tile fill must divide evenly");
static_assert((tile_elems * 2u) * 4u <= 32768u, "K/V tile exceeds the 32 KiB DX shared floor");

}// namespace mha_dims

using MhaFlashKernel = Kernel1D<void(Buffer<float>, Buffer<float>, Buffer<float>, Buffer<float>)>;

[[nodiscard]] MhaFlashKernel make_mha_flash_kernel() noexcept {
    return [](BufferFloat Q, BufferFloat K, BufferFloat V, BufferFloat O) noexcept {
        set_block_size(mha_dims::block, 1u, 1u);
        set_name("bench_mha_flash");

        Var idx = dispatch_id().x;
        Var b = idx / (mha_dims::heads * mha_dims::seq);
        Var h = (idx / mha_dims::seq) % mha_dims::heads;
        Var i = idx % mha_dims::seq;

        Var head_base = (b * mha_dims::heads + h) * (mha_dims::seq * mha_dims::dim);
        Var qi_base = head_base + i * mha_dims::dim;

        // Online softmax state (running max and running normalizer).
        Var m = def(-1.0e30f);
        Var s_norm = def(0.0f);

        // Register-resident output accumulator and query row (zero-initialized
        // DSL locals; q_local is overwritten immediately below).
        $array<float, mha_dims::dim> o_acc;
        $array<float, mha_dims::dim> q_local;
        $for (d, mha_dims::dim) {
            q_local[d] = Q.read(qi_base + d);
        };

        Shared<float> k_shared{mha_dims::tile_elems};
        Shared<float> v_shared{mha_dims::tile_elems};

        $for (jt, mha_dims::tiles) {
            // All block threads cooperatively stage the tile's K and V
            // (2 * tile * dim floats, tile * dim / block per thread).
            Var jt_base = head_base + jt * mha_dims::tile_elems;
            for (auto e = 0u; e < mha_dims::tile_elems / mha_dims::block; e++) {
                Var e_idx = thread_x() + e * mha_dims::block;
                k_shared[e_idx] = K.read(jt_base + e_idx);
                v_shared[e_idx] = V.read(jt_base + e_idx);
            }
            sync_block();

            $for (jj, mha_dims::tile) {
                Var jb = jj * mha_dims::dim;

                // score = Q[i] . K[j] (K read from shared memory)
                Var score = def(0.0f);
                $for (d, mha_dims::dim / 4u) {
                    Var d4 = d * 4u;
                    score += q_local[d4] * k_shared[jb + d4]
                           + q_local[d4 + 1u] * k_shared[jb + d4 + 1u]
                           + q_local[d4 + 2u] * k_shared[jb + d4 + 2u]
                           + q_local[d4 + 3u] * k_shared[jb + d4 + 3u];
                };
                score = score * mha_dims::scale;

                // Online softmax update: rescale the accumulator by the max
                // correction, then add the new key's contribution.
                Var m_new = max(m, score);
                Var exp_diff = exp(m - m_new);
                Var exp_score = exp(score - m_new);
                s_norm = s_norm * exp_diff + exp_score;

                $for (d, mha_dims::dim) {
                    o_acc[d] = o_acc[d] * exp_diff + v_shared[jb + d] * exp_score;
                };
                m = m_new;
            };
            // Every thread must be done with the tile before the next one
            // overwrites shared memory.
            sync_block();
        };

        // Normalize once and write the output row (single global write).
        $for (d, mha_dims::dim) {
            O.write(qi_base + d, o_acc[d] / s_norm);
        };
    };
}

// ---------------------------------------------------------------------------
// Benchmark 3: Multi-Head Latent Attention (MLA) pipeline.
//
// Modelled on the three kernels of src/ml/attention_kernels.cpp:
//   * project_q     — one block per token; H[b, i, :] is staged in shared
//                     memory and every thread produces 4 adjacent q outputs,
//                     with the decoupled RoPE rotation applied in registers
//                     before the single write.
//   * project_kv    — one block per token; threads < latent_dim produce the
//                     compressed latent key cKV, the remaining threads produce
//                     the decoupled RoPE key pairs (rotated in registers).
//   * attention     — one thread per (b, h, i) query row. The latent query is
//                     absorbed as W_uk^T * q_content (matrix absorption), so the
//                     per-key score is a latent-dim dot product plus the
//                     rope_dim positional dot product, and the value
//                     accumulator lives in latent space. W_uv is applied once
//                     per query row after the loop. The cKV/Krope slices are
//                     broadcast through shared memory, so the 128 threads of a
//                     block (one (b, h) pair) issue one global load per slice.
// ---------------------------------------------------------------------------
namespace mla_bench {

constexpr uint32_t batch = 8u;
constexpr uint32_t heads = 8u;
constexpr uint32_t seq = 256u;
constexpr uint32_t hidden = 512u;
constexpr uint32_t latent = 64u;                    // compressed KV dim d_c
constexpr uint32_t rope = 16u;                      // decoupled RoPE dim d_h^R
constexpr uint32_t head_dim = 64u;                  // rope + content
constexpr uint32_t content = head_dim - rope;       // 48
constexpr uint32_t pairs = rope / 2u;               // RoPE pairs per head
constexpr uint32_t q_out = heads * head_dim;        // 512 projected q values/token
constexpr uint32_t kv_out = heads * rope;           // 128 decoupled k values/token
constexpr float scale = 0.125f;                     // 1 / sqrt(head_dim)
constexpr float rope_theta = 10000.0f;

constexpr uint32_t proj_block = 128u;// one block per token for both projections
constexpr uint32_t q_per_thread = q_out / proj_block;// 4 adjacent q outputs
constexpr uint32_t attn_block = 128u;// one thread per query row

constexpr uint32_t tokens = batch * seq;
constexpr uint32_t rows = batch * heads * seq;

// Buffer sizes.
constexpr size_t hidden_size = size_t{batch} * seq * hidden;
constexpr size_t wq_size = size_t{q_out} * hidden;
constexpr size_t wdkv_size = size_t{latent} * hidden;
constexpr size_t wkr_size = size_t{kv_out} * hidden;
constexpr size_t wuk_size = size_t{heads} * content * latent;
constexpr size_t wuv_size = size_t{heads} * head_dim * latent;
constexpr size_t q_size = size_t{batch} * heads * seq * head_dim;
constexpr size_t ckv_size = size_t{batch} * seq * latent;
constexpr size_t krope_size = size_t{batch} * heads * seq * rope;

constexpr double work_units = double(8ull * 8ull * 256ull * 256ull * 64ull * 4ull);

static_assert(q_out % proj_block == 0u, "project_q must split evenly over the block");
static_assert(content % q_per_thread == 0u, "a thread's q outputs must not straddle the content/RoPE split");
static_assert(head_dim % q_per_thread == 0u, "a thread's q outputs must stay inside one head");
static_assert(latent + heads * pairs == proj_block, "project_kv thread split must cover the block");
static_assert(seq % attn_block == 0u, "an attention block must cover whole query rows of one (b, h)");

// Weight layouts, shared verbatim by the kernels and the CPU reference so the
// two can never drift (the helpers are templates so they accept both DSL Vars
// and host integers):
//   * projections sweep the reduction dim k and read kOut adjacent outputs ->
//     W[k][out] ("out-minor", the same layout the GEMM's B operand uses), which
//     keeps the warp loads coalesced. With the [out][k] layout every load
//     instruction of a warp touches 32 different cache lines (one distinct
//     weight row per lane) and project_q measured ~10x slower on an RTX 4060
//     (19.6 ms -> 2.0 ms) although the number of bytes read is identical;
//   * the absorbed W_uk / W_uv tables are read block-uniformly (all threads of
//     an attention block share one head), so they keep the [head][dim][latent]
//     layout and each thread walks the latent dim contiguously.
template<typename K, typename O>
[[nodiscard]] constexpr auto wq_index(K k, O o) noexcept { return k * q_out + o; }

template<typename K, typename L>
[[nodiscard]] constexpr auto wdkv_index(K k, L l) noexcept { return k * latent + l; }

template<typename K, typename O>
[[nodiscard]] constexpr auto wkr_index(K k, O o) noexcept { return k * kv_out + o; }

template<typename H, typename C, typename L>
[[nodiscard]] constexpr auto wuk_index(H h, C c, L l) noexcept {
    return (h * content + c) * latent + l;
}

template<typename H, typename D, typename L>
[[nodiscard]] constexpr auto wuv_index(H h, D d, L l) noexcept {
    return (h * head_dim + d) * latent + l;
}

}// namespace mla_bench

// Exact host-side inverse RoPE frequencies (1 / theta^(2p / rope_dim)) so the
// kernels only need a table lookup instead of a pow() per output element.
[[nodiscard]] inline const std::array<float, mla_bench::pairs> &mla_rope_freqs() noexcept {
    static const std::array<float, mla_bench::pairs> freqs = [] {
        std::array<float, mla_bench::pairs> f{};
        for (auto p = 0u; p < mla_bench::pairs; p++) {
            f[p] = 1.0f / std::pow(mla_bench::rope_theta,
                                   2.0f * static_cast<float>(p) / static_cast<float>(mla_bench::rope));
        }
        return f;
    }();
    return freqs;
}

using MlaProjectQKernel = Kernel1D<void(Buffer<float>, Buffer<float>, Buffer<float>)>;
using MlaProjectKVKernel = Kernel1D<void(Buffer<float>, Buffer<float>, Buffer<float>,
                                         Buffer<float>, Buffer<float>)>;
using MlaAttentionKernel = Kernel1D<void(Buffer<float>, Buffer<float>, Buffer<float>,
                                         Buffer<float>, Buffer<float>, Buffer<float>)>;

// Q[b, h, i, d] = Wq[:, h * head_dim + d] @ H[b, i, :], with pair-wise RoPE
// applied to the decoupled slice (d >= content) before the write.
// Params: H([tokens][hidden]), Wq([hidden][q_out]), Q([tokens][heads][head_dim]).
[[nodiscard]] MlaProjectQKernel make_mla_project_q_kernel() noexcept {
    return [](BufferFloat H, BufferFloat Wq, BufferFloat Q) noexcept {
        constexpr auto kBlock = mla_bench::proj_block;
        constexpr auto kPerThread = mla_bench::q_per_thread;
        set_block_size(kBlock, 1u, 1u);
        set_name("bench_mla_project_q");

        // Hoisted RoPE frequency table (8 exact host values).
        $constant freqs = mla_rope_freqs();

        Shared<float> h_row{mla_bench::hidden};

        Var tok = block_id().x;// one block per (b, i) token
        Var b = tok / mla_bench::seq;
        Var i = tok % mla_bench::seq;
        Var tx = thread_x();
        Var h_base = tok * mla_bench::hidden;

        // Cooperative staging of the input hidden row: afterwards every thread
        // reads the same 512 floats from shared memory (broadcast) instead of
        // re-reading the token row from global memory once per output.
        for (auto e = 0u; e < mla_bench::hidden / kBlock; e++) {
            Var e_idx = tx + e * kBlock;
            h_row[e_idx] = H.read(h_base + e_idx);
        }
        sync_block();

        Var o0 = tx * kPerThread;// first projected output of this thread
        Var<float> acc[kPerThread];
        // Hidden-dim sweep, 4-wide unrolled: kPerThread independent accumulator
        // chains (good ILP) over coalesced weight loads.
        $for (ks, mla_bench::hidden / 4u) {
            Var k4 = ks * 4u;
            Float h0 = h_row[k4];
            Float h1 = h_row[k4 + 1u];
            Float h2 = h_row[k4 + 2u];
            Float h3 = h_row[k4 + 3u];
            for (auto j = 0u; j < kPerThread; j++) {
                acc[j] += h0 * Wq.read(mla_bench::wq_index(k4, o0 + j))
                        + h1 * Wq.read(mla_bench::wq_index(k4 + 1u, o0 + j))
                        + h2 * Wq.read(mla_bench::wq_index(k4 + 2u, o0 + j))
                        + h3 * Wq.read(mla_bench::wq_index(k4 + 3u, o0 + j));
            }
        };

        Var hh = o0 / mla_bench::head_dim;
        Var dd = o0 % mla_bench::head_dim;

        // Pair-wise rotation of one (x0, x1) pair at position `pos`.
        auto rotate = [&](Float x0, Float x1, Var<uint> pair, Var<uint> pos) noexcept -> Float2 {
            Float freq = freqs.read(pair);
            Float angle = pos.cast<float>() * freq;
            Float c = cos(angle);
            Float s = sin(angle);
            return make_float2(x0 * c - x1 * s, x0 * s + x1 * c);
        };

        // dd is a multiple of kPerThread and content is a multiple of
        // kPerThread, so a thread's outputs are either all content dims or all
        // RoPE dims -- never a mix -- and the whole thread rotates as two pairs.
        $if (dd >= mla_bench::content) {
            Var p0 = (dd - mla_bench::content) / 2u;
            Float2 rot0 = rotate(acc[0], acc[1], p0, i);
            Float2 rot1 = rotate(acc[2], acc[3], p0 + 1u, i);
            acc[0] = rot0.x;
            acc[1] = rot0.y;
            acc[2] = rot1.x;
            acc[3] = rot1.y;
        };

        Var q_base = ((b * mla_bench::heads + hh) * mla_bench::seq + i) * mla_bench::head_dim + dd;
        for (auto j = 0u; j < kPerThread; j++) {
            Q.write(q_base + j, acc[j]);
        }
    };
}

// Fused KV projection: cKV[b, i, l] = Wdkv[:, l] @ H[b, i, :] for the first
// latent_dim threads, and one decoupled RoPE key pair per remaining thread
// (Krope[b, h, i, r] = Wkr[:, h * rope + r] @ H[b, i, :], rotated in registers).
// Params: H([tokens][hidden]), Wdkv([hidden][latent]), Wkr([hidden][kv_out]),
//         cKV([tokens][latent]), Krope([batch][heads][seq][rope]).
[[nodiscard]] MlaProjectKVKernel make_mla_project_kv_kernel() noexcept {
    return [](BufferFloat H, BufferFloat Wdkv, BufferFloat Wkr,
              BufferFloat cKV, BufferFloat Krope) noexcept {
        constexpr auto kBlock = mla_bench::proj_block;
        set_block_size(kBlock, 1u, 1u);
        set_name("bench_mla_project_kv");

        $constant freqs = mla_rope_freqs();

        Shared<float> h_row{mla_bench::hidden};

        Var tok = block_id().x;// one block per (b, i) token
        Var b = tok / mla_bench::seq;
        Var i = tok % mla_bench::seq;
        Var tx = thread_x();
        Var h_base = tok * mla_bench::hidden;

        for (auto e = 0u; e < mla_bench::hidden / kBlock; e++) {
            Var e_idx = tx + e * kBlock;
            h_row[e_idx] = H.read(h_base + e_idx);
        }
        sync_block();

        auto rotate = [&](Float x0, Float x1, Var<uint> pair, Var<uint> pos) noexcept -> Float2 {
            Float freq = freqs.read(pair);
            Float angle = pos.cast<float>() * freq;
            Float c = cos(angle);
            Float s = sin(angle);
            return make_float2(x0 * c - x1 * s, x0 * s + x1 * c);
        };

        $if (tx < mla_bench::latent) {
            // -- Compress KV: cKV[b, i, tx] = Wdkv[:, tx] @ H[b, i, :] --
            // One output per thread -> a warp reads 32 adjacent weight columns.
            Var sum = def(0.0f);
            $for (ks, mla_bench::hidden / 4u) {
                Var k4 = ks * 4u;
                sum += h_row[k4] * Wdkv.read(mla_bench::wdkv_index(k4, tx))
                     + h_row[k4 + 1u] * Wdkv.read(mla_bench::wdkv_index(k4 + 1u, tx))
                     + h_row[k4 + 2u] * Wdkv.read(mla_bench::wdkv_index(k4 + 2u, tx))
                     + h_row[k4 + 3u] * Wdkv.read(mla_bench::wdkv_index(k4 + 3u, tx));
            };
            cKV.write(tok * mla_bench::latent + tx, sum);
        } $else {
            // -- Decoupled RoPE keys: one (r, r + 1) pair per thread --
            Var p = tx - mla_bench::latent;     // pair slot, 0 .. heads * pairs - 1
            Var hh = p / mla_bench::pairs;      // head
            Var r = (p % mla_bench::pairs) * 2u;// first dim of the pair
            Var o = hh * mla_bench::rope + r;   // weight column of the pair
            Var sum0 = def(0.0f);
            Var sum1 = def(0.0f);
            // Both columns of the pair are adjacent, so the two loads below hit
            // the same cache line.
            $for (ks, mla_bench::hidden / 4u) {
                Var k4 = ks * 4u;
                sum0 += h_row[k4] * Wkr.read(mla_bench::wkr_index(k4, o))
                      + h_row[k4 + 1u] * Wkr.read(mla_bench::wkr_index(k4 + 1u, o))
                      + h_row[k4 + 2u] * Wkr.read(mla_bench::wkr_index(k4 + 2u, o))
                      + h_row[k4 + 3u] * Wkr.read(mla_bench::wkr_index(k4 + 3u, o));
                sum1 += h_row[k4] * Wkr.read(mla_bench::wkr_index(k4, o + 1u))
                      + h_row[k4 + 1u] * Wkr.read(mla_bench::wkr_index(k4 + 1u, o + 1u))
                      + h_row[k4 + 2u] * Wkr.read(mla_bench::wkr_index(k4 + 2u, o + 1u))
                      + h_row[k4 + 3u] * Wkr.read(mla_bench::wkr_index(k4 + 3u, o + 1u));
            };
            Float2 rot = rotate(sum0, sum1, r / 2u, i);
            Var kr_base = ((b * mla_bench::heads + hh) * mla_bench::seq + i) * mla_bench::rope + r;
            Krope.write(kr_base, rot.x);
            Krope.write(kr_base + 1u, rot.y);
        };
    };
}

// Absorbed latent-space online attention. One thread per (b, h, i) query row:
//   q_latent   = W_uk[h]^T @ q_content            (hoisted out of the j loop)
//   q_pos      = the decoupled RoPE query slice
//   score(j)   = (q_latent . cKV[b, j] + q_pos . Krope[b, h, j]) * scale
//   a_acc     += exp(score - m_new) * cKV[b, j]    (latent-space value accumulator)
//   O[i]       = W_uv[h] @ a_acc / s_norm
// The cKV/Krope slices are staged into shared memory by the first
// latent_dim/rope threads of the block and broadcast to all of them.
// Params: Q([tokens][heads][head_dim]), cKV, Krope, Wuk, Wuv, O.
[[nodiscard]] MlaAttentionKernel make_mla_attention_kernel() noexcept {
    return [](BufferFloat Q, BufferFloat cKV, BufferFloat Krope,
              BufferFloat Wuk, BufferFloat Wuv, BufferFloat O) noexcept {
        set_block_size(mla_bench::attn_block, 1u, 1u);
        set_name("bench_mla_attention");

        Shared<float> ckv_shared{mla_bench::latent};
        Shared<float> krope_shared{mla_bench::rope};

        Var idx = dispatch_id().x;
        Var b = idx / (mla_bench::heads * mla_bench::seq);
        Var h = (idx / mla_bench::seq) % mla_bench::heads;
        Var i = idx % mla_bench::seq;
        Var tx = thread_x();

        Var q_row = ((b * mla_bench::heads + h) * mla_bench::seq + i) * mla_bench::head_dim;

        // Register state. The latent query and the latent value accumulator are
        // both latent_dim wide and every access to them below uses a
        // compile-time index (the latent loops are emitted unrolled), so they
        // stay in registers: there is no dynamic indexing to spill.
        Var<float> q_latent[mla_bench::latent];
        Var<float> a_acc[mla_bench::latent];
        Var<float> q_pos[mla_bench::rope];

        for (auto r = 0u; r < mla_bench::rope; r++) {
            q_pos[r] = Q.read(q_row + mla_bench::content + r);
        }
        for (auto l = 0u; l < mla_bench::latent; l++) {
            q_latent[l] = def(0.0f);
            a_acc[l] = def(0.0f);
        }

        // Absorb the content projection into the query: q_latent = W_uk[h]^T * q_content.
        // The content loop stays a device loop (one dynamic loop, latent_dim
        // FMAs inside) so the emitted AST stays small; the weight rows it reads
        // are block-uniform (broadcast).
        $for (c, mla_bench::content) {
            Float qc = Q.read(q_row + c);
            for (auto l = 0u; l < mla_bench::latent; l++) {
                q_latent[l] += qc * Wuk.read(mla_bench::wuk_index(h, c, l));
            }
        };

        Var m = def(-1.0e30f);
        Var s_norm = def(0.0f);

        $for (j, mla_bench::seq) {
            // -- Broadcast cKV[b, j, :] and Krope[b, h, j, :] to the block --
            Var ckv_base = (b * mla_bench::seq + j) * mla_bench::latent;
            $if (tx < mla_bench::latent) {
                ckv_shared[tx] = cKV.read(ckv_base + tx);
            };
            $if (tx < mla_bench::rope) {
                Var kr_base = ((b * mla_bench::heads + h) * mla_bench::seq + j) * mla_bench::rope;
                krope_shared[tx] = Krope.read(kr_base + tx);
            };
            sync_block();

            // -- Content + positional score --
            Var content_score = def(0.0f);
            for (auto l = 0u; l < mla_bench::latent; l++) {
                content_score += q_latent[l] * ckv_shared[l];
            }
            Var pos_score = def(0.0f);
            for (auto r = 0u; r < mla_bench::rope; r++) {
                pos_score += q_pos[r] * krope_shared[r];
            }
            Var score = (content_score + pos_score) * mla_bench::scale;

            // -- Online softmax + latent-space accumulation --
            Var m_new = max(m, score);
            Var exp_diff = exp(m - m_new);
            Var exp_score = exp(score - m_new);
            s_norm = s_norm * exp_diff + exp_score;
            for (auto l = 0u; l < mla_bench::latent; l++) {
                a_acc[l] = a_acc[l] * exp_diff + ckv_shared[l] * exp_score;
            }
            // The next iteration overwrites the shared slices, so every thread
            // must be done consuming this one first.
            sync_block();
            m = m_new;
        };

        // -- Single up-projection from latent space, normalized --
        // The output-dim loop is a device loop (head_dim iterations) with the
        // latent reduction unrolled inside; a_acc stays register-resident.
        $for (d, mla_bench::head_dim) {
            Var sum = def(0.0f);
            for (auto l = 0u; l < mla_bench::latent; l++) {
                sum += Wuv.read(mla_bench::wuv_index(h, d, l)) * a_acc[l];
            }
            O.write(q_row + d, sum / s_norm);
        };
    };
}

// ---------------------------------------------------------------------------
// Benchmark 4: transformer FFN block (LayerNorm -> up GEMM -> down GEMM).
// ---------------------------------------------------------------------------
namespace mlp_dims {

constexpr uint32_t rows = 4096u;   // tokens
constexpr uint32_t d = 1024u;      // model width
constexpr uint32_t ffn = 4096u;    // feed-forward width
constexpr uint32_t norm_block = 256u;
constexpr float eps = 1e-5f;
constexpr uint32_t elems_per_thread = d / norm_block;// 4 elements per thread per pass

// Both GEMMs use the same register-blocked geometry as benchmark 1.
constexpr uint32_t tile = 64u;
constexpr uint32_t tile_k = 16u;
constexpr uint32_t reg_m = 8u;
constexpr uint32_t reg_n = 4u;
constexpr uint32_t gemm_block = (tile / reg_m) * (tile / reg_n);// 128

constexpr double work_units = double(4096ull * 1024ull * 4096ull * 4ull);

static_assert(d % norm_block == 0u, "the LayerNorm passes need a whole number of elements per thread");
static_assert(rows % tile == 0u && d % tile == 0u && ffn % tile == 0u, "GEMM tiles must divide the shapes");
static_assert(d % tile_k == 0u && ffn % tile_k == 0u, "GEMM K slabs must divide the shapes");

}// namespace mlp_dims

// Two-pass LayerNorm with a shared-memory halving-tree block reduction (the
// same math order and the same tree as src/embedding/embed/kernels/kernels_norm.cpp
// -- deliberately not warp collectives, so every backend including the CPU
// fallback runs it). One block per row, 256 threads.
// Params: x([rows][d]), gamma([d]), beta([d]), y([rows][d]).
using LayerNormKernel = Kernel1D<void(Buffer<float>, Buffer<float>, Buffer<float>, Buffer<float>)>;

[[nodiscard]] LayerNormKernel make_layernorm_kernel() noexcept {
    return [](BufferFloat x, BufferFloat gamma, BufferFloat beta, BufferFloat y) noexcept {
        set_block_size(mlp_dims::norm_block, 1u, 1u);
        set_name("bench_layernorm");

        Shared<float> red{mlp_dims::norm_block};

        Var tx = thread_x();
        Var row = block_id().x;// one block per row
        Var base = row * mlp_dims::d;
        constexpr auto kD = static_cast<float>(mlp_dims::d);

        // Block-wide sum of one per-thread partial. Every thread calls it the
        // same number of times (block-uniform control flow only) and each call
        // ends with a barrier, so `red` is never overwritten before all threads
        // have read the result.
        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = mlp_dims::norm_block / 2u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();
            return total;
        };

        // Pass 1: mean = sum(x) / D.
        Var sum1 = def(0.0f);
        for (auto e = 0u; e < mlp_dims::elems_per_thread; e++) {
            sum1 += x.read(base + tx + e * mlp_dims::norm_block);
        }
        Var mean = block_sum(sum1) / kD;

        // Pass 2: var = sum((x - mean)^2) / D  (biased, no Bessel).
        Var sum2 = def(0.0f);
        for (auto e = 0u; e < mlp_dims::elems_per_thread; e++) {
            Var diff = x.read(base + tx + e * mlp_dims::norm_block) - mean;
            sum2 += diff * diff;
        }
        Var var = block_sum(sum2) / kD;

        // Pass 3: affine normalize + write (same expression order as the CPU reference).
        Var inv = 1.0f / sqrt(var + mlp_dims::eps);
        for (auto e = 0u; e < mlp_dims::elems_per_thread; e++) {
            Var col = tx + e * mlp_dims::norm_block;
            Var xv = x.read(base + col);
            y.write(base + col, ((xv - mean) * inv) * gamma.read(col) + beta.read(col));
        }
    };
}

// ---------------------------------------------------------------------------
// CPU references (exact, double precision) used by the verification passes.
// ---------------------------------------------------------------------------

// Exact CPU GEMM + bias + GELU-tanh for one output element.
[[nodiscard]] double gemm_bias_gelu_ref(const std::vector<float> &a, const std::vector<float> &b,
                                        const std::vector<float> &bias,
                                        uint32_t n, uint32_t k,
                                        uint32_t row, uint32_t col) noexcept {
    double acc = 0.0;
    for (auto kk = 0u; kk < k; kk++) {
        acc += static_cast<double>(a[size_t{row} * k + kk]) * static_cast<double>(b[size_t{kk} * n + col]);
    }
    acc += static_cast<double>(bias[col]);
    return gelu_tanh_ref(acc);
}

// Exact CPU dense-softmax attention for one (b, h, i) query row. Mathematically
// identical to the kernel's online formulation (same max subtraction, same
// normalization), just evaluated densely.
[[nodiscard]] std::vector<double> mha_cpu_reference_row(const std::vector<float> &q,
                                                        const std::vector<float> &k,
                                                        const std::vector<float> &v,
                                                        uint32_t b, uint32_t h, uint32_t i) {
    constexpr auto kHeads = mha_dims::heads;
    constexpr auto kSeq = mha_dims::seq;
    constexpr auto kDim = mha_dims::dim;
    auto head = (b * kHeads + h) * size_t{kSeq} * kDim;
    auto q_row = head + size_t{i} * kDim;

    std::vector<double> scores(kSeq, 0.0);
    double m = -1.0e30;
    for (auto j = 0u; j < kSeq; j++) {
        auto k_row = head + size_t{j} * kDim;
        double acc = 0.0;
        for (auto dd = 0u; dd < kDim; dd++) {
            acc += static_cast<double>(q[q_row + dd]) * static_cast<double>(k[k_row + dd]);
        }
        auto score = acc * static_cast<double>(mha_dims::scale);
        scores[j] = score;
        m = std::max(m, score);
    }
    double s_norm = 0.0;
    for (auto j = 0u; j < kSeq; j++) {
        auto p = std::exp(scores[j] - m);
        scores[j] = p;
        s_norm += p;
    }
    std::vector<double> out(kDim, 0.0);
    for (auto j = 0u; j < kSeq; j++) {
        auto p = scores[j] / s_norm;
        auto v_row = head + size_t{j} * kDim;
        for (auto dd = 0u; dd < kDim; dd++) {
            out[dd] += p * static_cast<double>(v[v_row + dd]);
        }
    }
    return out;
}

// Exact CPU reference for the whole MLA pipeline on one (b, h, i) query row:
// Q projection + RoPE, KV compression + decoupled RoPE keys, absorbed latent
// attention, then the W_uv up-projection. Returns head_dim doubles.
[[nodiscard]] std::vector<double> mla_cpu_reference_row(const std::vector<float> &h_buf,
                                                        const std::vector<float> &wq,
                                                        const std::vector<float> &wdkv,
                                                        const std::vector<float> &wkr,
                                                        const std::vector<float> &wuk,
                                                        const std::vector<float> &wuv,
                                                        uint32_t b, uint32_t h, uint32_t i) {
    constexpr auto kSeq = mla_bench::seq;
    constexpr auto kHidden = mla_bench::hidden;
    constexpr auto kLatent = mla_bench::latent;
    constexpr auto kRope = mla_bench::rope;
    constexpr auto kContent = mla_bench::content;
    constexpr auto kHeadDim = mla_bench::head_dim;
    constexpr auto kPairs = mla_bench::pairs;
    const auto &freqs = mla_rope_freqs();

    // -- q row: Q[b, h, i, d] = Wq[:, h * head_dim + d] @ H[b, i, :] --
    std::vector<double> q_row(kHeadDim, 0.0);
    auto hid_row = size_t{b} * kSeq * kHidden + size_t{i} * kHidden;
    for (auto dd = 0u; dd < kHeadDim; dd++) {
        auto col = h * kHeadDim + dd;
        double acc = 0.0;
        for (auto kk = 0u; kk < kHidden; kk++) {
            acc += static_cast<double>(wq[mla_bench::wq_index(kk, col)]) * static_cast<double>(h_buf[hid_row + kk]);
        }
        q_row[dd] = acc;
    }
    // Fused pair-wise RoPE on the decoupled slice.
    for (auto p = 0u; p < kPairs; p++) {
        auto angle = static_cast<double>(i) * static_cast<double>(freqs[p]);
        auto c = std::cos(angle);
        auto s = std::sin(angle);
        auto d0 = kContent + 2u * p;
        auto x0 = q_row[d0];
        auto x1 = q_row[d0 + 1u];
        q_row[d0] = x0 * c - x1 * s;
        q_row[d0 + 1u] = x0 * s + x1 * c;
    }

    // -- Absorbed latent query: q_latent[l] = sum_c Wuk[h, c, l] * q_content[c] --
    std::vector<double> q_latent(kLatent, 0.0);
    for (auto c = 0u; c < kContent; c++) {
        auto qc = q_row[c];
        for (auto l = 0u; l < kLatent; l++) {
            q_latent[l] += static_cast<double>(wuk[mla_bench::wuk_index(h, c, l)]) * qc;
        }
    }

    // -- One pass over the keys: score + latent-space value accumulation --
    std::vector<double> scores(kSeq, 0.0);
    // cKV rows are reused by the value accumulation below, so cache them.
    std::vector<double> ckv_all(size_t{kSeq} * kLatent, 0.0);
    for (auto j = 0u; j < kSeq; j++) {
        auto src_row = size_t{b} * kSeq * kHidden + size_t{j} * kHidden;
        for (auto l = 0u; l < kLatent; l++) {
            double acc = 0.0;
            for (auto kk = 0u; kk < kHidden; kk++) {
                acc += static_cast<double>(wdkv[mla_bench::wdkv_index(kk, l)]) * static_cast<double>(h_buf[src_row + kk]);
            }
            ckv_all[size_t{j} * kLatent + l] = acc;
        }
        // Decoupled RoPE key for this (h, j).
        std::vector<double> krope(kRope, 0.0);
        for (auto r = 0u; r < kRope; r++) {
            auto col = h * kRope + r;
            double acc = 0.0;
            for (auto kk = 0u; kk < kHidden; kk++) {
                acc += static_cast<double>(wkr[mla_bench::wkr_index(kk, col)]) * static_cast<double>(h_buf[src_row + kk]);
            }
            krope[r] = acc;
        }
        for (auto p = 0u; p < kPairs; p++) {
            auto angle = static_cast<double>(j) * static_cast<double>(freqs[p]);
            auto c = std::cos(angle);
            auto s = std::sin(angle);
            auto r0 = 2u * p;
            auto x0 = krope[r0];
            auto x1 = krope[r0 + 1u];
            krope[r0] = x0 * c - x1 * s;
            krope[r0 + 1u] = x0 * s + x1 * c;
        }
        double content_score = 0.0;
        for (auto l = 0u; l < kLatent; l++) {
            content_score += q_latent[l] * ckv_all[size_t{j} * kLatent + l];
        }
        double pos_score = 0.0;
        for (auto r = 0u; r < kRope; r++) {
            pos_score += q_row[kContent + r] * krope[r];
        }
        scores[j] = (content_score + pos_score) * static_cast<double>(mla_bench::scale);
    }

    // -- Softmax (max-subtracted, same as the kernel's online form) --
    double m = -1.0e30;
    for (auto j = 0u; j < kSeq; j++) { m = std::max(m, scores[j]); }
    double s_norm = 0.0;
    for (auto j = 0u; j < kSeq; j++) {
        auto p = std::exp(scores[j] - m);
        scores[j] = p;
        s_norm += p;
    }
    std::vector<double> a_acc(kLatent, 0.0);
    for (auto j = 0u; j < kSeq; j++) {
        auto p = scores[j] / s_norm;
        for (auto l = 0u; l < kLatent; l++) {
            a_acc[l] += p * ckv_all[size_t{j} * kLatent + l];
        }
    }

    // -- Single up-projection: O[d] = sum_l Wuv[h, d, l] * a_acc[l] --
    std::vector<double> out(kHeadDim, 0.0);
    for (auto dd = 0u; dd < kHeadDim; dd++) {
        double acc = 0.0;
        for (auto l = 0u; l < kLatent; l++) {
            acc += static_cast<double>(wuv[mla_bench::wuv_index(h, dd, l)]) * a_acc[l];
        }
        // a_acc and s_norm already carry the normalization (p = exp / s_norm).
        out[dd] = acc;
    }
    return out;
}

// Exact CPU reference for the FFN block on one row: LayerNorm -> up GEMM with
// bias + GELU-tanh -> down GEMM with bias + residual add.
[[nodiscard]] std::vector<double> mlp_cpu_reference_row(const std::vector<float> &x,
                                                        const std::vector<float> &gamma,
                                                        const std::vector<float> &beta,
                                                        const std::vector<float> &wup,
                                                        const std::vector<float> &bup,
                                                        const std::vector<float> &wdown,
                                                        const std::vector<float> &bdown,
                                                        uint32_t row) {
    constexpr auto kD = mlp_dims::d;
    constexpr auto kFfn = mlp_dims::ffn;
    auto base = size_t{row} * kD;

    // -- LayerNorm (same two-pass order as the kernel) --
    double sum = 0.0;
    for (auto j = 0u; j < kD; j++) { sum += static_cast<double>(x[base + j]); }
    auto mean = sum / static_cast<double>(kD);
    double sum_sq = 0.0;
    for (auto j = 0u; j < kD; j++) {
        auto diff = static_cast<double>(x[base + j]) - mean;
        sum_sq += diff * diff;
    }
    auto var = sum_sq / static_cast<double>(kD);
    auto inv = 1.0 / std::sqrt(var + static_cast<double>(mlp_dims::eps));
    std::vector<double> norm_row(kD, 0.0);
    for (auto j = 0u; j < kD; j++) {
        norm_row[j] = ((static_cast<double>(x[base + j]) - mean) * inv) * static_cast<double>(gamma[j]) + static_cast<double>(beta[j]);
    }

    // -- Up GEMM (K = d, N = ffn) + bias + GELU --
    std::vector<double> hidden(kFfn, 0.0);
    for (auto j = 0u; j < kFfn; j++) {
        double acc = 0.0;
        for (auto kk = 0u; kk < kD; kk++) {
            acc += norm_row[kk] * static_cast<double>(wup[size_t{kk} * kFfn + j]);
        }
        hidden[j] = gelu_tanh_ref(acc + static_cast<double>(bup[j]));
    }

    // -- Down GEMM (K = ffn, N = d) + bias + residual --
    std::vector<double> out(kD, 0.0);
    for (auto j = 0u; j < kD; j++) {
        double acc = 0.0;
        for (auto kk = 0u; kk < kFfn; kk++) {
            acc += hidden[kk] * static_cast<double>(wdown[size_t{kk} * kD + j]);
        }
        out[j] = acc + static_cast<double>(bdown[j]) + static_cast<double>(x[base + j]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Benchmark drivers
// ---------------------------------------------------------------------------

void log_result(const BenchResult &result) {
    if (result.verified) {
        LUISA_INFO("[{}] {}: compile {:.2f} ms, median {:.3f} ms, {:.3f} {} ({})",
                   result.category, result.name, result.compile_ms, result.median_ms,
                   result.throughput, result.throughput_unit, result.check_note);
    } else {
        LUISA_INFO("[{}] {}: compile {:.2f} ms, median {:.3f} ms, {:.3f} {} -- CHECK FAILED ({})",
                   result.category, result.name, result.compile_ms, result.median_ms,
                   result.throughput, result.throughput_unit, result.check_note);
    }
}

// 1) FP32 GEMM (1024^3) + bias + GELU-tanh.
void run_gemm_f32_bench(Device &device, Stream &stream,
                        const BenchConfig &cfg, std::vector<BenchResult> &results) {
    constexpr auto kM = gemm_f32_dims::m;
    constexpr auto kN = gemm_f32_dims::n;
    constexpr auto kK = gemm_f32_dims::k;
    constexpr auto kTileM = gemm_f32_dims::tile_m;
    constexpr auto kTileN = gemm_f32_dims::tile_n;
    constexpr auto kTileK = gemm_f32_dims::tile_k;
    constexpr auto kBlock = gemm_f32_dims::block;

    LUISA_INFO("[inference] preparing infer_gemm_f32 (M={}, N={}, K={}, tile {}x{}x{}, {} threads)",
               kM, kN, kK, kTileM, kTileN, kTileK, kBlock);

    uint64_t rng = splitmix64(0x1111'1111'1111'1111ull);
    auto host_a = make_random_vector(size_t{kM} * kK, rng, -1.0f, 1.0f);
    auto host_b = make_random_vector(size_t{kK} * kN, rng, -1.0f, 1.0f);
    auto host_bias = make_random_vector(kN, rng, -0.1f, 0.1f);

    auto a_buf = device.create_buffer<float>(host_a.size());
    auto b_buf = device.create_buffer<float>(host_b.size());
    auto bias_buf = device.create_buffer<float>(host_bias.size());
    auto c_buf = device.create_buffer<float>(size_t{kM} * kN);

    stream << a_buf.copy_from(luisa::span{host_a})
           << b_buf.copy_from(luisa::span{host_b})
           << bias_buf.copy_from(luisa::span{host_bias})
           << synchronize();

    double compile_ms = 0.0;
    auto kernel = make_reg_gemm_kernel<kM, kN, kK, kTileM, kTileN, kTileK,
                                       gemm_f32_dims::reg_m, gemm_f32_dims::reg_n,
                                       GemmEpi::bias_gelu_tanh>();
    auto gemm_shader = compile_uncached(device, kernel, "bench_infer_gemm_f32", cfg, compile_ms);

    constexpr auto kBlocks = (kM / kTileM) * (kN / kTileN);
    constexpr auto kDispatch = kBlocks * kBlock;
    // The bias_gelu epilogue ignores the residual operand; pass the bias buffer
    // so the kernel signature stays uniform across epilogues.
    auto dispatch = [&] {
        stream << gemm_shader(a_buf, b_buf, bias_buf, bias_buf, c_buf).dispatch(kDispatch);
    };

    CheckSummary check;
    constexpr double kAbsTol = 1.0e-3;// required by the benchmark contract
    constexpr double kRelTol = 0.0;
    if (cfg.verify) {
        // Spot-check a spread of output elements against the exact CPU GEMM.
        struct Spot {
            uint32_t row;
            uint32_t col;
        };
        constexpr Spot kSpots[] = {{0u, 0u}, {0u, 1000u}, {17u, 33u}, {255u, 999u},
                                   {511u, 512u}, {700u, 64u}, {1023u, 0u}, {1023u, 1023u}};
        std::vector<float> host_c(size_t{kM} * kN);
        dispatch();
        stream << c_buf.copy_to(luisa::span{host_c}) << synchronize();
        for (auto spot : kSpots) {
            auto ref = gemm_bias_gelu_ref(host_a, host_b, host_bias, kN, kK, spot.row, spot.col);
            accumulate_check(check, static_cast<double>(host_c[size_t{spot.row} * kN + spot.col]),
                             ref, kAbsTol, kRelTol);
        }
    }

    auto result = run_timed_bench("inference", "infer_gemm_f32", stream, cfg, compile_ms,
                                  gemm_f32_dims::work_units, "FLOP/s", dispatch);
    result.verified = !cfg.verify || check_passed(check);
    result.check_note = cfg.verify ?
                            make_check_note(check, kAbsTol, kRelTol) :
                            luisa::string{"verification disabled"};
    log_result(result);
    results.push_back(std::move(result));
}

// 2) Multi-head online-softmax (flash) attention.
void run_mha_flash_bench(Device &device, Stream &stream,
                         const BenchConfig &cfg, std::vector<BenchResult> &results) {
    LUISA_INFO("[inference] preparing infer_mha_flash (batch={}, heads={}, seq={}, head_dim={}, "
               "block={}, K/V tile={})",
               mha_dims::batch, mha_dims::heads, mha_dims::seq, mha_dims::dim,
               mha_dims::block, mha_dims::tile);

    uint64_t rng = splitmix64(0x2222'2222'2222'2222ull);
    auto host_q = make_random_vector(mha_dims::rows * mha_dims::dim, rng, -1.0f, 1.0f);
    auto host_k = make_random_vector(mha_dims::rows * mha_dims::dim, rng, -1.0f, 1.0f);
    auto host_v = make_random_vector(mha_dims::rows * mha_dims::dim, rng, -1.0f, 1.0f);

    auto q_buf = device.create_buffer<float>(host_q.size());
    auto k_buf = device.create_buffer<float>(host_k.size());
    auto v_buf = device.create_buffer<float>(host_v.size());
    auto o_buf = device.create_buffer<float>(host_q.size());

    stream << q_buf.copy_from(luisa::span{host_q})
           << k_buf.copy_from(luisa::span{host_k})
           << v_buf.copy_from(luisa::span{host_v})
           << synchronize();

    double compile_ms = 0.0;
    auto kernel = make_mha_flash_kernel();
    auto attend_shader = compile_uncached(device, kernel, "bench_infer_mha_flash", cfg, compile_ms);

    // One thread per (b, h, i) query row.
    constexpr auto kDispatch = mha_dims::rows;
    auto dispatch = [&] {
        stream << attend_shader(q_buf, k_buf, v_buf, o_buf).dispatch(kDispatch);
    };

    CheckSummary check;
    constexpr double kAbsTol = 1.0e-3;
    constexpr double kRelTol = 0.0;
    if (cfg.verify) {
        // Whole query rows, spread over batch / head / position.
        struct Spot {
            uint32_t b;
            uint32_t h;
            uint32_t i;
        };
        constexpr Spot kSpots[] = {{0u, 0u, 0u}, {3u, 5u, 17u}, {8u, 2u, 256u}, {15u, 7u, 511u}};
        std::vector<float> host_o(host_q.size());
        dispatch();
        stream << o_buf.copy_to(luisa::span{host_o}) << synchronize();
        for (auto spot : kSpots) {
            auto ref = mha_cpu_reference_row(host_q, host_k, host_v, spot.b, spot.h, spot.i);
            auto base = ((spot.b * mha_dims::heads + spot.h) * size_t{mha_dims::seq} + spot.i) * mha_dims::dim;
            for (auto dd = 0u; dd < mha_dims::dim; dd++) {
                accumulate_check(check, static_cast<double>(host_o[base + dd]), ref[dd], kAbsTol, kRelTol);
            }
        }
    }

    auto result = run_timed_bench("inference", "infer_mha_flash", stream, cfg, compile_ms,
                                  mha_dims::work_units, "FLOP/s", dispatch);
    result.verified = !cfg.verify || check_passed(check);
    result.check_note = cfg.verify ?
                            make_check_note(check, kAbsTol, kRelTol) :
                            luisa::string{"verification disabled"};
    log_result(result);
    results.push_back(std::move(result));
}

// 3) MLA pipeline: project_q (+RoPE) -> project_kv (+RoPE) -> absorbed attention.
//
// The reported FLOP/s counts the attention work units only, as the benchmark
// contract specifies (2 * batch * heads * seq^2 * head_dim), so it understates
// the pipeline's arithmetic intensity: the two projections are memory-bound
// (each token block streams the whole weight matrix) and their time is included
// in the measured latency.
void run_mla_attention_bench(Device &device, Stream &stream,
                             const BenchConfig &cfg, std::vector<BenchResult> &results) {
    LUISA_INFO("[inference] preparing infer_mla_attention (batch={}, heads={}, seq={}, hidden={}, "
               "latent={}, rope={}, content={})",
               mla_bench::batch, mla_bench::heads, mla_bench::seq, mla_bench::hidden,
               mla_bench::latent, mla_bench::rope, mla_bench::content);

    // Weights are scaled so the projections stay O(1) (a dot product of length
    // K over U(-1, 1) inputs then behaves like a unit-variance sum), which keeps
    // the spot check meaningful instead of comparing large near-equal values.
    // Layouts are the ones documented next to mla_bench::wq_index/... (the
    // projections consume the transposed W[k][out] form, the absorbed W_uk/W_uv
    // stay [head][dim][latent]); the kernels and the CPU reference share those
    // helpers, so a mismatch cannot go unnoticed.
    uint64_t rng = splitmix64(0x3333'3333'3333'3333ull);
    auto host_h = make_random_vector(mla_bench::hidden_size, rng, -1.0f, 1.0f);
    auto host_wq = make_random_matrix(mla_bench::wq_size, rng, 1.0f / std::sqrt(static_cast<float>(mla_bench::hidden)));
    auto host_wdkv = make_random_matrix(mla_bench::wdkv_size, rng, 1.0f / std::sqrt(static_cast<float>(mla_bench::hidden)));
    auto host_wkr = make_random_matrix(mla_bench::wkr_size, rng, 1.0f / std::sqrt(static_cast<float>(mla_bench::hidden)));
    auto host_wuk = make_random_matrix(mla_bench::wuk_size, rng, 1.0f / std::sqrt(static_cast<float>(mla_bench::content)));
    auto host_wuv = make_random_matrix(mla_bench::wuv_size, rng, 1.0f / std::sqrt(static_cast<float>(mla_bench::latent)));

    auto h_buf = device.create_buffer<float>(host_h.size());
    auto wq_buf = device.create_buffer<float>(host_wq.size());
    auto wdkv_buf = device.create_buffer<float>(host_wdkv.size());
    auto wkr_buf = device.create_buffer<float>(host_wkr.size());
    auto wuk_buf = device.create_buffer<float>(host_wuk.size());
    auto wuv_buf = device.create_buffer<float>(host_wuv.size());
    auto q_buf = device.create_buffer<float>(mla_bench::q_size);
    auto ckv_buf = device.create_buffer<float>(mla_bench::ckv_size);
    auto krope_buf = device.create_buffer<float>(mla_bench::krope_size);
    auto o_buf = device.create_buffer<float>(mla_bench::q_size);

    stream << h_buf.copy_from(luisa::span{host_h})
           << wq_buf.copy_from(luisa::span{host_wq})
           << wdkv_buf.copy_from(luisa::span{host_wdkv})
           << wkr_buf.copy_from(luisa::span{host_wkr})
           << wuk_buf.copy_from(luisa::span{host_wuk})
           << wuv_buf.copy_from(luisa::span{host_wuv})
           << synchronize();

    double compile_ms = 0.0;
    auto project_q_kernel = make_mla_project_q_kernel();
    auto project_q_shader = compile_uncached(device, project_q_kernel, "bench_infer_mla_project_q", cfg, compile_ms);
    auto project_kv_kernel = make_mla_project_kv_kernel();
    auto project_kv_shader = compile_uncached(device, project_kv_kernel, "bench_infer_mla_project_kv", cfg, compile_ms);
    auto attention_kernel = make_mla_attention_kernel();
    auto attention_shader = compile_uncached(device, attention_kernel, "bench_infer_mla_attention", cfg, compile_ms);

    constexpr auto kProjDispatch = mla_bench::tokens * mla_bench::proj_block;
    constexpr auto kAttnDispatch = mla_bench::rows;
    auto dispatch = [&] {
        stream << project_q_shader(h_buf, wq_buf, q_buf).dispatch(kProjDispatch)
               << project_kv_shader(h_buf, wdkv_buf, wkr_buf, ckv_buf, krope_buf).dispatch(kProjDispatch)
               << attention_shader(q_buf, ckv_buf, krope_buf, wuk_buf, wuv_buf, o_buf).dispatch(kAttnDispatch);
    };

    CheckSummary check;
    constexpr double kAbsTol = 1.0e-3;
    constexpr double kRelTol = 1.0e-3;
    if (cfg.verify) {
        // Whole output rows of the end-to-end pipeline (projections included).
        struct Spot {
            uint32_t b;
            uint32_t h;
            uint32_t i;
        };
        constexpr Spot kSpots[] = {{0u, 0u, 0u}, {5u, 3u, 128u}};
        std::vector<float> host_o(mla_bench::q_size);
        dispatch();
        stream << o_buf.copy_to(luisa::span{host_o}) << synchronize();
        for (auto spot : kSpots) {
            auto ref = mla_cpu_reference_row(host_h, host_wq, host_wdkv, host_wkr, host_wuk, host_wuv,
                                            spot.b, spot.h, spot.i);
            auto base = ((spot.b * mla_bench::heads + spot.h) * size_t{mla_bench::seq} + spot.i) * mla_bench::head_dim;
            for (auto dd = 0u; dd < mla_bench::head_dim; dd++) {
                accumulate_check(check, static_cast<double>(host_o[base + dd]), ref[dd], kAbsTol, kRelTol);
            }
        }
    }

    auto result = run_timed_bench("inference", "infer_mla_attention", stream, cfg, compile_ms,
                                  mla_bench::work_units, "FLOP/s", dispatch);
    result.verified = !cfg.verify || check_passed(check);
    result.check_note = cfg.verify ?
                            make_check_note(check, kAbsTol, kRelTol) :
                            luisa::string{"verification disabled"};
    log_result(result);
    results.push_back(std::move(result));
}

// 4) Transformer FFN block: LayerNorm -> up GEMM (+bias, +GELU) -> down GEMM
//    (+bias, +residual).
void run_norm_mlp_bench(Device &device, Stream &stream,
                        const BenchConfig &cfg, std::vector<BenchResult> &results) {
    constexpr auto kRows = mlp_dims::rows;
    constexpr auto kD = mlp_dims::d;
    constexpr auto kFfn = mlp_dims::ffn;

    LUISA_INFO("[inference] preparing infer_norm_mlp (rows={}, D={}, FFN={}, norm block={}, GEMM tile {}x{})",
               kRows, kD, kFfn, mlp_dims::norm_block, mlp_dims::tile, mlp_dims::tile);

    uint64_t rng = splitmix64(0x4444'4444'4444'4444ull);
    auto host_x = make_random_vector(size_t{kRows} * kD, rng, -1.0f, 1.0f);
    auto host_gamma = make_random_vector(kD, rng, 0.5f, 1.5f);
    auto host_beta = make_random_vector(kD, rng, -0.5f, 0.5f);
    auto host_wup = make_random_matrix(size_t{kD} * kFfn, rng, 1.0f / std::sqrt(static_cast<float>(kD)));
    auto host_bup = make_random_vector(kFfn, rng, -0.05f, 0.05f);
    auto host_wdown = make_random_matrix(size_t{kFfn} * kD, rng, 1.0f / std::sqrt(static_cast<float>(kFfn)));
    auto host_bdown = make_random_vector(kD, rng, -0.05f, 0.05f);

    auto x_buf = device.create_buffer<float>(host_x.size());
    auto gamma_buf = device.create_buffer<float>(host_gamma.size());
    auto beta_buf = device.create_buffer<float>(host_beta.size());
    auto norm_buf = device.create_buffer<float>(host_x.size());// LayerNorm output
    auto wup_buf = device.create_buffer<float>(host_wup.size());
    auto bup_buf = device.create_buffer<float>(host_bup.size());
    auto hidden_buf = device.create_buffer<float>(size_t{kRows} * kFfn);// FFN activation
    auto wdown_buf = device.create_buffer<float>(host_wdown.size());
    auto bdown_buf = device.create_buffer<float>(host_bdown.size());
    auto out_buf = device.create_buffer<float>(host_x.size());

    stream << x_buf.copy_from(luisa::span{host_x})
           << gamma_buf.copy_from(luisa::span{host_gamma})
           << beta_buf.copy_from(luisa::span{host_beta})
           << wup_buf.copy_from(luisa::span{host_wup})
           << bup_buf.copy_from(luisa::span{host_bup})
           << wdown_buf.copy_from(luisa::span{host_wdown})
           << bdown_buf.copy_from(luisa::span{host_bdown})
           << synchronize();

    double compile_ms = 0.0;
    auto norm_kernel = make_layernorm_kernel();
    auto norm_shader = compile_uncached(device, norm_kernel, "bench_infer_norm_mlp_layernorm", cfg, compile_ms);
    // up: [rows x D] * [D x FFN] + bias + GELU
    auto up_kernel = make_reg_gemm_kernel<kRows, kFfn, kD, mlp_dims::tile, mlp_dims::tile,
                                          mlp_dims::tile_k, mlp_dims::reg_m, mlp_dims::reg_n,
                                          GemmEpi::bias_gelu_tanh>();
    auto up_shader = compile_uncached(device, up_kernel, "bench_infer_norm_mlp_up", cfg, compile_ms);
    // down: [rows x FFN] * [FFN x D] + bias + residual(x)
    auto down_kernel = make_reg_gemm_kernel<kRows, kD, kFfn, mlp_dims::tile, mlp_dims::tile,
                                            mlp_dims::tile_k, mlp_dims::reg_m, mlp_dims::reg_n,
                                            GemmEpi::bias_residual>();
    auto down_shader = compile_uncached(device, down_kernel, "bench_infer_norm_mlp_down", cfg, compile_ms);

    constexpr auto kNormDispatch = kRows * mlp_dims::norm_block;
    constexpr auto kUpDispatch = (kRows / mlp_dims::tile) * (kFfn / mlp_dims::tile) * mlp_dims::gemm_block;
    constexpr auto kDownDispatch = (kRows / mlp_dims::tile) * (kD / mlp_dims::tile) * mlp_dims::gemm_block;
    auto dispatch = [&] {
        stream << norm_shader(x_buf, gamma_buf, beta_buf, norm_buf).dispatch(kNormDispatch)
               << up_shader(norm_buf, wup_buf, bup_buf, bup_buf, hidden_buf).dispatch(kUpDispatch)
               << down_shader(hidden_buf, wdown_buf, bdown_buf, x_buf, out_buf).dispatch(kDownDispatch);
    };

    CheckSummary check;
    constexpr double kAbsTol = 1.0e-3;
    constexpr double kRelTol = 2.0e-3;
    if (cfg.verify) {
        // Whole output rows against the exact CPU FFN block, plus a global
        // finiteness / magnitude scan over the downloaded tensor.
        constexpr uint32_t kSpotRows[] = {0u, 1234u, 4095u};
        std::vector<float> host_out(size_t{kRows} * kD);
        dispatch();
        stream << out_buf.copy_to(luisa::span{host_out}) << synchronize();
        for (auto v : host_out) {
            if (!std::isfinite(v)) {
                check.finite = false;
                check.within_tolerance = false;
                break;
            }
        }
        for (auto row : kSpotRows) {
            auto ref = mlp_cpu_reference_row(host_x, host_gamma, host_beta,
                                             host_wup, host_bup, host_wdown, host_bdown, row);
            auto base = size_t{row} * kD;
            for (auto j = 0u; j < kD; j++) {
                accumulate_check(check, static_cast<double>(host_out[base + j]), ref[j], kAbsTol, kRelTol);
            }
        }
    }

    auto result = run_timed_bench("inference", "infer_norm_mlp", stream, cfg, compile_ms,
                                  mlp_dims::work_units, "FLOP/s", dispatch);
    result.verified = !cfg.verify || check_passed(check);
    result.check_note = cfg.verify ?
                            make_check_note(check, kAbsTol, kRelTol) :
                            luisa::string{"verification disabled"};
    log_result(result);
    results.push_back(std::move(result));
}

}// namespace

void run_inference_benchmarks(compute::Device &device, compute::Stream &stream,
                              const BenchConfig &cfg, std::vector<BenchResult> &results) {
    if (should_run(cfg, "inference", "infer_gemm_f32")) {
        run_gemm_f32_bench(device, stream, cfg, results);
    }
    if (should_run(cfg, "inference", "infer_mha_flash")) {
        run_mha_flash_bench(device, stream, cfg, results);
    }
    if (should_run(cfg, "inference", "infer_mla_attention")) {
        run_mla_attention_bench(device, stream, cfg, results);
    }
    if (should_run(cfg, "inference", "infer_norm_mlp")) {
        run_norm_mlp_bench(device, stream, cfg, results);
    }
}

}// namespace luisa::bench
