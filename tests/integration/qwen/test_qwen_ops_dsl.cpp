// Qwen DSL operator device regression test (tests/integration/qwen).
//
// Usage: test_qwen_ops_dsl <backend>       e.g. test_qwen_ops_dsl cuda
// The backend is EXPLICIT argv[1]; there is no default and no silent skip.
// Requires no checkpoint: every operator runs on deterministic synthetic
// data with ragged shapes, in BOTH weight formats (BF16 and FP8 E4M3), and
// is compared against the independent CPU oracles in qwen/dsl_ops.h.
// Dedicated cases pin down BOTH GEMV weight paths: the aligned packed-word
// fast path (fp8 K=128/256 group 64, bf16 even K) and the ragged/group-62
// strided fallback, against a second, locally decoded double oracle.
// Six external-contract fixtures compare device results against INDEPENDENT
// hand-computed expectations (not the shared oracles). Exits nonzero on any
// The KV dtype/layout differential (opt_plan/05 Q3) runs every attention
// realization three ways: the FP32 position-major cache, the BF16 cache in the
// 64-position blocked layout, and the DEQUANTIZED CPU oracle built from the very
// codes the BF16 cache stores (qwen_kv_region_to_dense + ref_qwen_attn_*), over
// pos0 0/3/61/64/640, decode kv_len 1/63/64/65/511/512/513/640/650, legacy +
// split-KV + tiled, plus a word-exact check of the packed append kernel.
// mismatch and prints located, actionable errors.
//
// Consolidated from the verified scratch harnesses (CUDA + DX variants);
// the pre-added GGUF norm-weight convention (direct multiply) is used
// throughout, per the storage contract.
#include "qwen_device_test_utils.h"

#include "qwen/common.h"// host Q8_0 ABI: pack_q8_0_blocks / fp16 / dequantize
#include "qwen/dsl_ops.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

// opt_plan/09 N4.2-N4.4: the conditional WHILE chunk (probe + the production
// device-terminated finish kernel) and the pinned download ring. Both are CUDA
// extensions, so every case reports "not available on this backend" and skips
// instead of failing when the extension is absent (dx/vk must stay green).
#include "qwen/cuda_graph_while.h"
#include "qwen/pinned_ring.h"
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <luisa/backends/ext/pinned_memory_ext.hpp>
#include <luisa/runtime/command_list.h>

#include <cstring>// std::memcpy (independent bf16 oracle decode)
#include <limits>// std::numeric_limits (independent fp8 NaN code)

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::qwen::dsl;
using luisa::qwen::pack_w4_matrix;
using luisa::qwen::dequantize_w4_matrix;
using luisa::qwen::validate_w4_geometry;
using luisa::qwen::W4ClipSearchConfig;
using luisa::qwen::W4PackStats;
using luisa::qwen::QwenCudaWhileProbe;
using luisa::qwen::qwen_probe_cuda_graph_while;
using luisa::qwen::QwenPinnedRing;
using luisa::qwen::QwenPinnedStats;
using luisa::qwen::QwenWhileRequest;
using luisa::qwen::qwen_validate_while_request;
using namespace qwen_test;

namespace {

const char *fmt_name(QwenWeightFormat f) {
    return f == QwenWeightFormat::bf16 ? "bf16" : "fp8";
}

// ---------------------------------------------------------------------------
// GEMV + GEMM, both formats, ragged K/N/M, residual and plain epilogues.
// ---------------------------------------------------------------------------
void test_matmul(Device &dev, Stream &stream, Rng &rng, Buffer<float> &dummy_f) {
    for (auto format : {QwenWeightFormat::bf16, QwenWeightFormat::fp8_e4m3}) {
        const uint32_t K = 130u, N = 37u, M = 3u;// K=130: 2 full fp8 groups + 2 tail
        luisa::vector<float> w(K * N), x(M * K), res(M * N);
        rng.fill(w, -1.f, 1.f);
        rng.fill(x, -1.f, 1.f);
        rng.fill(res, -0.5f, 0.5f);
        luisa::vector<uint32_t> words;
        luisa::vector<float> scales;
        check(host_pack_matrix(format, w, K, N, 64u, words, scales).ok(), "matmul host pack");
        auto bw = upload(dev, stream, words);
        auto bs = upload(dev, stream, scales.empty() ? luisa::vector<float>{0.f} : scales);
        auto bx = upload(dev, stream, x);
        auto br = upload(dev, stream, res);
        auto by = dev.create_buffer<float>(M * N);
        luisa::vector<float> ref;
        check(ref_qwen_matmul(format, x, words, scales, res, true, M, N, K, 64u, ref).ok(),
              "matmul oracle residual");
        // Tolerances: BF16 storage error ~2^-8 relative; FP8 E4M3 storage is
        // coarse (3 mantissa bits) -- the oracle decodes the SAME quantized
        // weights, so only accumulation-order noise remains, but keep the
        // verified scratch bounds.
        const float rel = format == QwenWeightFormat::bf16 ? 2e-2f : 0.4f;
        {
            auto sh = dev.compile(make_qwen_gemv_kernel(format, QwenMatmulEpilogue::residual));
            auto plan = plan_qwen_gemv_dispatch(M, N, K);
            check(plan.ok(), "gemv plan");
            stream << sh(bx, bw, bs, br, by, M, N, K, 64u).dispatch(qwen_dispatch_threads(*plan.value));
            auto y = download(stream, by, M * N);
            stream << synchronize();
            check_close(y, ref, 2e-3f, rel,
                        format == QwenWeightFormat::bf16 ? "gemv bf16 residual" : "gemv fp8 residual");
        }
        {
            auto sh = dev.compile(make_qwen_gemm_kernel(format, QwenMatmulEpilogue::residual));
            auto plan = plan_qwen_gemm_dispatch(M, N, K);
            check(plan.ok(), "gemm plan");
            stream << sh(bx, bw, bs, br, by, M, N, K, 64u, plan.value->padded_cols)
                          .dispatch(plan.value->total_x, plan.value->total_y);
            auto y = download(stream, by, M * N);
            stream << synchronize();
            check_close(y, ref, 2e-3f, rel,
                        format == QwenWeightFormat::bf16 ? "gemm bf16 residual" : "gemm fp8 residual");
        }
        {
            luisa::vector<float> ref0;
            check(ref_qwen_matmul(format, x, words, scales, {}, false, M, N, K, 64u, ref0).ok(),
                  "matmul oracle plain");
            auto sh = dev.compile(make_qwen_gemv_kernel(format, QwenMatmulEpilogue::none));
            auto plan = plan_qwen_gemv_dispatch(M, N, K);
            stream << sh(bx, bw, bs, dummy_f, by, M, N, K, 64u).dispatch(qwen_dispatch_threads(*plan.value));
            auto y = download(stream, by, M * N);
            stream << synchronize();
            check_close(y, ref0, 2e-3f, rel,
                        format == QwenWeightFormat::bf16 ? "gemv bf16 plain" : "gemv fp8 plain");
        }
    }
}

// ---------------------------------------------------------------------------
// GEMV packed-word fast path vs legacy strided fallback (make_qwen_gemv_kernel
// selects in-kernel via qwen_gemv_packed_ok: bf16 needs K%2==0; fp8 needs
// K%4==0 AND group%4==0 AND group>0). The committed K=130 case above NEVER
// enters the packed path, so these permanent cases pin it down:
//   fp8 K=128 & K=256, group 64  -> ALIGNED packed quads, one scale per quad
//                                   (K=256 spans 4 groups: per-word scale
//                                   index wi/(group/4) is exercised)
//   fp8 K=128 & K=256, group 62  -> FALLBACK (62%4!=0; quads would cross
//                                   scale boundaries -> legacy strided loop)
//   bf16 K=128 & K=256           -> ALIGNED packed pairs
//   fp8 K=130 group 64           -> ragged-K FALLBACK canary (K%4!=0)
// (bf16 odd-K fallback cannot be built: host_pack_matrix requires an even
// element count K*N, so odd K needs even N -- covered by neither here nor
// in the committed K=130 bf16 case, which uses odd N=37 and packs evenly.)
// M=4 (kQwenGemvMaxM) covers the multi-row loop; M=1 covers decode.
// Oracles: (1) the shared ref_qwen_matmul, (2) an INDEPENDENT local oracle
// that re-extracts each packed word's bytes/halves itself (never calls
// host_decode_packed) and accumulates in double -- it does not share the
// decode helpers with either device path. Tolerances match test_matmul
// (bf16 rel 2e-2, fp8 rel 0.4, atol 2e-3); the oracle decodes the SAME
// quantized weights so only accumulation-order noise remains. Guarded
// canary tails on y catch OOB writes; inputs are verified read-only.
// ---------------------------------------------------------------------------
void test_matmul_packed_paths(Device &dev, Stream &stream, Rng &rng, Buffer<float> &dummy_f) {
    struct Shape {
        QwenWeightFormat format;
        uint32_t k, n, group, m;
        bool packed;// expected fast-path eligibility (mirrors the documented gate)
        const char *tag;
    };
    const Shape shapes[] = {
        {QwenWeightFormat::fp8_e4m3, 128u, 37u, 64u, 4u, true, "fp8 K128 g64 packed"},
        {QwenWeightFormat::fp8_e4m3, 256u, 37u, 64u, 4u, true, "fp8 K256 g64 packed"},
        {QwenWeightFormat::fp8_e4m3, 128u, 37u, 62u, 4u, false, "fp8 K128 g62 fallback"},
        {QwenWeightFormat::fp8_e4m3, 256u, 37u, 62u, 4u, false, "fp8 K256 g62 fallback"},
        {QwenWeightFormat::fp8_e4m3, 130u, 37u, 64u, 4u, false, "fp8 K130 g64 fallback"},
        {QwenWeightFormat::fp8_e4m3, 128u, 37u, 64u, 1u, true, "fp8 K128 g64 packed M1"},
        {QwenWeightFormat::fp8_e4m3, 128u, 37u, 62u, 1u, false, "fp8 K128 g62 fallback M1"},
        {QwenWeightFormat::bf16, 128u, 37u, 64u, 4u, true, "bf16 K128 packed"},
        {QwenWeightFormat::bf16, 256u, 37u, 64u, 4u, true, "bf16 K256 packed"},
        {QwenWeightFormat::bf16, 129u, 38u, 64u, 4u, false, "bf16 K129 N38 fallback"},
    };
    constexpr float kCanary = -55555.25f;
    constexpr size_t kPad = 8u;
    // INDEPENDENT oracle: decode straight from the packed uint32 words with
    // local bit manipulation (fp8 e4m3: sign/exp3/mant3, bias 7, 0x7F is NaN;
    // bf16: high 16 bits of f32), double accumulation in ascending K order.
    auto decode_fp8_code = [](uint32_t code) -> double {
        const double sign = (code & 0x80u) ? -1.0 : 1.0;
        const uint32_t exp = (code >> 3u) & 0xFu, man = code & 0x7u;
        if (exp == 0xFu && man == 0x7u) { return std::numeric_limits<double>::quiet_NaN(); }
        if (exp == 0u) { return sign * double(man) * 0x1p-9;// subnormal: man * 2^-9
        }
        return sign * double(8u + man) * std::ldexp(1.0, int(exp) - 10);// (1+man/8)*2^(exp-7)
    };
    auto decode_bf16_half = [](uint32_t half) -> double {
        const uint32_t bits = half << 16u;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return double(f);
    };
    for (auto [format, K, N, group, M, packed, tag] : shapes) {
        const bool fp8 = format == QwenWeightFormat::fp8_e4m3;
        luisa::vector<float> w(size_t(K) * N), x(size_t(M) * K), res(size_t(M) * N);
        rng.fill(w, -1.f, 1.f);
        rng.fill(x, -1.f, 1.f);
        rng.fill(res, -0.5f, 0.5f);
        luisa::vector<uint32_t> words;
        luisa::vector<float> scales;
        check(host_pack_matrix(format, w, K, N, group, words, scales).ok(), "packed-paths host pack");
        const uint32_t pack = fp8 ? 4u : 2u;
        const uint64_t gpr = (K + group - 1u) / group;
        // Storage ABI: words are a FLAT stream of packed quads/pairs over the
        // whole [N,K] matrix (row*k + ki)/pack -- with ragged K a word can
        // hold the tail of one row and the head of the next. The oracle
        // therefore indexes globally, never per-row.
        const uint64_t total_elems = uint64_t(K) * N;
        // Independent double-precision oracle over the SAME packed words.
        auto local_oracle = [&](bool has_residual) {
            luisa::vector<float> y(size_t(M) * N);
            for (uint32_t mi = 0; mi < M; mi++) {
                for (uint32_t ni = 0; ni < N; ni++) {
                    double acc = 0.0;
                    for (uint32_t ki = 0; ki < K; ki++) {
                        const uint64_t idx = uint64_t(ni) * K + ki;
                        const uint32_t word = words[static_cast<size_t>(idx / pack)];
                        const uint32_t e = uint32_t(idx % pack);
                        {
                            double wv;
                            if (fp8) {
                                const double s = double(scales[size_t(ni) * gpr + ki / group]);
                                wv = decode_fp8_code((word >> (8u * e)) & 0xFFu) * s;
                            } else {
                                wv = decode_bf16_half((word >> (16u * e)) & 0xFFFFu);
                            }
                            acc += double(x[size_t(mi) * K + ki]) * wv;
                        }
                    }
                    if (has_residual) { acc += double(res[size_t(mi) * N + ni]); }
                    y[size_t(mi) * N + ni] = float(acc);
                }
            }
            return y;
        };
        const auto ref_ind = local_oracle(true);
        const auto ref_ind0 = local_oracle(false);
        luisa::vector<float> ref, ref0;
        check(ref_qwen_matmul(format, x, words, scales, res, true, M, N, K, group, ref).ok(),
              "packed-paths shared oracle residual");
        check(ref_qwen_matmul(format, x, words, scales, {}, false, M, N, K, group, ref0).ok(),
              "packed-paths shared oracle plain");
        // The two oracles must agree (same decode, different accumulation).
        const float rel = fp8 ? 0.4f : 2e-2f;
        check_close(ref_ind, ref, 2e-3f, rel, "packed-paths independent oracle vs shared");
        auto bw = upload(dev, stream, words);
        auto bs = upload(dev, stream, scales.empty() ? luisa::vector<float>{0.f} : scales);
        auto bx = upload(dev, stream, x);
        auto br = upload(dev, stream, res);
        const size_t yn = size_t(M) * N;
        auto by = upload(dev, stream, luisa::vector<float>(yn + kPad, kCanary));
        // Guarded residual dummy for the plain epilogue: must never be written.
        constexpr float kDummySentinel = 4242.5f;
        auto bdummy = upload(dev, stream, luisa::vector<float>{kDummySentinel});
        luisa::vector<float> tail(kPad);
        auto check_tail = [&]() {
            stream << by.view(yn, kPad).copy_to(luisa::span{tail}) << synchronize();
            bool ok = true;
            for (auto v : tail) { ok = ok && v == kCanary; }
            char name[128];
            std::snprintf(name, sizeof(name), "%s y no OOB tail write", tag);
            check(ok, name);
        };
        auto run = [&](QwenMatmulEpilogue epi) {
            auto sh = dev.compile(make_qwen_gemv_kernel(format, epi));
            auto plan = plan_qwen_gemv_dispatch(M, N, K);
            check(plan.ok(), "packed-paths gemv plan");
            const bool residual = epi == QwenMatmulEpilogue::residual;
            stream << sh(bx, bw, bs, residual ? br : bdummy, by, M, N, K, group)
                          .dispatch(qwen_dispatch_threads(*plan.value));
            return download(stream, by, yn);
        };
        char name[128];
        {
            const auto y = run(QwenMatmulEpilogue::residual);
            std::snprintf(name, sizeof(name), "gemv %s residual vs independent", tag);
            check_close(y, ref_ind, 2e-3f, rel, name);
            std::snprintf(name, sizeof(name), "gemv %s residual vs shared", tag);
            check_close(y, ref, 2e-3f, rel, name);
            check_tail();
        }
        {
            const auto y = run(QwenMatmulEpilogue::none);
            std::snprintf(name, sizeof(name), "gemv %s plain vs independent", tag);
            check_close(y, ref_ind0, 2e-3f, rel, name);
            check_tail();
            auto d = download(stream, bdummy, 1);
            std::snprintf(name, sizeof(name), "%s residual dummy untouched", tag);
            check(d.size() == 1 && d[0] == kDummySentinel, name);
        }
        // The GEMM (thread-per-output, always strided) shares the packed ABI:
        // on aligned shapes it cross-checks the fast-path GEMV reduction.
        // bf16 with ragged K AND odd N is excluded: the flat bf16 pair stream
        // then straddles row boundaries with mixed halves, which the GEMM's
        // per-element (w_base+ki)>>1 decode cannot address (pre-existing ABI
        // constraint, out of scope here -- GEMV handles it via word ownership).
        const bool gemm_ok = fp8 || (K % 2u == 0u) || (N % 2u == 0u);
        if (gemm_ok) {
            auto sh = dev.compile(make_qwen_gemm_kernel(format, QwenMatmulEpilogue::residual));
            auto plan = plan_qwen_gemm_dispatch(M, N, K);
            check(plan.ok(), "packed-paths gemm plan");
            stream << sh(bx, bw, bs, br, by, M, N, K, group, plan.value->padded_cols)
                          .dispatch(plan.value->total_x, plan.value->total_y);
            const auto y = download(stream, by, yn);
            std::snprintf(name, sizeof(name), "gemm %s residual vs independent", tag);
            check_close(y, ref_ind, 2e-3f, rel, name);
            check_tail();
        }
        // Packed and fallback paths partition K differently (documented):
        // NOT bit-identical in general, only within accumulation-order noise.
        // Inputs must be read-only on every path.
        check_exact(download(stream, bx, x.size()), x, "packed-paths x unmodified");
        auto wback = download(stream, bw, words.size());
        bool wok = wback.size() == words.size();
        for (size_t i = 0; i < words.size() && wok; i++) { wok = wback[i] == words[i]; }
        check(wok, "packed-paths packed weights unmodified");
        if (fp8) {
            check_exact(download(stream, bs, scales.size()), scales, "packed-paths scales unmodified");
        }
        (void) packed;// documents which gate branch this shape must take
    }
}

// Qwen3.5-4B M=1 decode GEMV: no-tail four-stream packed loop and, for FP8,
// the exact shared E4M3 LUT. Exercise every real K with both epilogues.
void test_qwen35_decode_gemv(Device &dev, Stream &stream, Rng &rng, Buffer<float> &dummy_f) {
    constexpr uint32_t N = 37u, M = 1u;
    for (auto format : {QwenWeightFormat::bf16, QwenWeightFormat::fp8_e4m3}) {
        for (uint32_t K : {2560u, 4096u, 9216u}) {
            luisa::vector<float> w(size_t(K) * N), x(K), residual(N);
            rng.fill(w, -1.f, 1.f);
            rng.fill(x, -1.f, 1.f);
            rng.fill(residual, -0.5f, 0.5f);
            luisa::vector<uint32_t> words;
            luisa::vector<float> scales;
            check(host_pack_matrix(format, w, K, N, 64u, words, scales).ok(),
                  "qwen35 decode gemv host pack");
            auto bw = upload(dev, stream, words);
            auto bs = upload(dev, stream, scales.empty() ? luisa::vector<float>{0.f} : scales);
            auto bx = upload(dev, stream, x);
            auto br = upload(dev, stream, residual);
            auto by = dev.create_buffer<float>(N);
            auto sh = dev.compile(make_qwen35_decode_gemv_kernel(format, QwenMatmulEpilogue::residual));
            auto plan = plan_qwen35_decode_gemv_dispatch(N, K, format);
            check(plan.ok(), "qwen35 decode gemv plan");
            stream << sh(bx, bw, bs, br, by, M, N, K, 64u)
                          .dispatch(qwen_dispatch_threads(*plan.value));
            auto y = download(stream, by, N);
            luisa::vector<float> ref;
            check(ref_qwen_matmul(format, x, words, scales, residual, true,
                                  M, N, K, 64u, ref).ok(), "qwen35 decode gemv oracle");
            check_close(y, ref, 3e-3f, format == QwenWeightFormat::bf16 ? 2e-2f : 0.4f,
                        "qwen35 decode gemv residual");
            auto plain = dev.compile(make_qwen35_decode_gemv_kernel(format, QwenMatmulEpilogue::none));
            stream << plain(bx, bw, bs, dummy_f, by, M, N, K, 64u)
                          .dispatch(qwen_dispatch_threads(*plan.value));
            y = download(stream, by, N);
            check(ref_qwen_matmul(format, x, words, scales, {}, false,
                                  M, N, K, 64u, ref).ok(), "qwen35 decode gemv plain oracle");
            check_close(y, ref, 3e-3f, format == QwenWeightFormat::bf16 ? 2e-2f : 0.4f,
                        "qwen35 decode gemv plain");

        }
    }
}

// ---------------------------------------------------------------------------
// Weight-reusing prefill GEMM (make_qwen35_gemm_mt_kernel, opt_plan/02 s3).
//
// Three independent links per case:
//  1. the shared CPU oracle `ref_qwen_matmul` (double accumulation) for the
//     small shapes, i.e. the same reference the legacy GEMV/GEMM use;
//  2. the LEGACY device GEMM (one thread per (m,n)) on the SAME packed buffers,
//     compared over the whole [M,N] output -- this is the differential the
//     executor's kernel switch relies on;
//  3. for the model shapes a sampled host oracle: 16 random (row, col) pairs
//     recomputed elementwise in double from the SAME packed words, which anchors
//     the device-vs-device comparison to the real math without a full O(M*N*K)
//     host loop.
// Shapes: m = 1,2,3,4,5,16,17,63,64,65 (decode/MTP/mt-tile boundaries and
// non-multiples of MT), k = 130/132 (ragged) and 2560/4096/9216 (model),
// n = 32/37 (ragged, 8/10 column blocks) and 1024/8192/9216 (model),
// both weight formats and both epilogues.
// ---------------------------------------------------------------------------
void gemm_mt_check_shape(Device &dev, Stream &stream, Rng &rng, Buffer<float> &dummy_f,
                         QwenWeightFormat format, uint32_t M, uint32_t N, uint32_t K,
                         bool check_legacy, bool sampled_oracle_only) {
    const uint32_t pack = format == QwenWeightFormat::bf16 ? 2u : 4u;
    const auto tag = [&](const char *what) {
        return luisa::string{what} + " " + (format == QwenWeightFormat::bf16 ? "bf16" : "fp8") +
               " m" + std::to_string(M).c_str() + " n" + std::to_string(N).c_str() +
               " k" + std::to_string(K).c_str();
    };
    // `check`/`check_close` take a bare `const char *`; keep the labelled name
    // alive for the duration of the call.
    const auto check_tagged = [&](bool ok, const char *what, const luisa::string &extra = {}) {
        const auto name = tag(what);
        check(ok, name.c_str(), extra);
    };
    const auto check_close_tagged = [&](luisa::span<const float> got, luisa::span<const float> want,
                                        float atol, float rtol, const char *what) {
        const auto name = tag(what);
        check_close(got, want, atol, rtol, name.c_str());
    };
    check_tagged(K % pack == 0u, "gemm mt shape is packed-word aligned");
    luisa::vector<float> w(size_t(K) * N), x(size_t(M) * K), residual(size_t(M) * N);
    rng.fill(w, -1.f, 1.f);
    rng.fill(x, -1.f, 1.f);
    rng.fill(residual, -0.5f, 0.5f);
    luisa::vector<uint32_t> words;
    luisa::vector<float> scales;
    check_tagged(host_pack_matrix(format, w, K, N, 64u, words, scales).ok(), "gemm mt host pack");
    auto bw = upload(dev, stream, words);
    auto bs = upload(dev, stream, scales.empty() ? luisa::vector<float>{0.f} : scales);
    auto bx = upload(dev, stream, x);
    auto br = upload(dev, stream, residual);
    auto by = dev.create_buffer<float>(size_t(M) * N);
    const float rel = format == QwenWeightFormat::bf16 ? 2e-2f : 0.4f;

    QwenGemmMtConfig cfg{};
    auto residual_plan = plan_qwen35_gemm_mt_dispatch(M, N, K, format, 64u, cfg);
    check_tagged(residual_plan.ok() && !residual_plan.value->empty(), "gemm mt plan");
    if (!residual_plan.ok()) { return; }
    const auto &plan = *residual_plan.value;
    auto sh = dev.compile(make_qwen35_gemm_mt_kernel(cfg, format, QwenMatmulEpilogue::residual));
    stream << sh(bx, bw, bs, br, by, M, N, K, 64u, plan.column_blocks)
                  .dispatch(plan.total_x, plan.total_y);
    auto y = download(stream, by, size_t(M) * N);

    if (!sampled_oracle_only) {
        luisa::vector<float> ref;
        check_tagged(ref_qwen_matmul(format, x, words, scales, residual, true, M, N, K, 64u, ref).ok(),
                     "gemm mt oracle");
        check_close_tagged(y, ref, 3e-3f, rel, "gemm mt residual vs cpu oracle");
    }
    if (check_legacy) {
        // Legacy GEMM on the same buffers: the whole output must agree to FP32
        // reassociation (K is split across lanes here, summed serially there).
        auto by_legacy = dev.create_buffer<float>(size_t(M) * N);
        auto legacy = dev.compile(make_qwen_gemm_kernel(format, QwenMatmulEpilogue::residual));
        auto legacy_plan = plan_qwen_gemm_dispatch(M, N, K);
        check_tagged(legacy_plan.ok(), "legacy gemm plan");
        stream << legacy(bx, bw, bs, br, by_legacy, M, N, K, 64u, legacy_plan.value->padded_cols)
                      .dispatch(legacy_plan.value->total_x, legacy_plan.value->total_y);
        auto y_legacy = download(stream, by_legacy, size_t(M) * N);
        double max_diff = 0.0, max_val = 0.0;
        bool finite = true;
        for (size_t i = 0; i < y.size(); i++) {
            if (!std::isfinite(y[i]) || !std::isfinite(y_legacy[i])) { finite = false; }
            max_diff = std::max(max_diff, std::fabs(double(y[i]) - double(y_legacy[i])));
            max_val = std::max(max_val, std::fabs(double(y_legacy[i])));
        }
        char extra[160];
        std::snprintf(extra, sizeof(extra), "max_abs=%.3g value_scale=%.3g", max_diff, max_val);
        check_tagged(finite && max_diff <= 2e-3 * max_val + 1e-4, "gemm mt vs legacy gemm", extra);
    }
    // Sampled host oracle: a handful of elements recomputed from the packed words.
    if (sampled_oracle_only) {
        bool ok = true;
        double worst = 0.0;
        for (uint32_t s = 0u; s < 16u && ok; s++) {
            const uint32_t row = (rng.next() % M);
            const uint32_t col = (rng.next() % N);
            double acc = 0.0;
            for (uint32_t ki = 0u; ki < K; ki++) {
                acc += double(x[size_t(row) * K + ki]) *
                       double(host_decode_packed(format, words, scales, K, 64u,
                                                 size_t(col) * K + ki));
            }
            const double want = acc + double(residual[size_t(row) * N + col]);
            const double got = double(y[size_t(row) * N + col]);
            const double diff = std::fabs(got - want);
            const double tol = 3e-3 * std::fabs(want) + (format == QwenWeightFormat::bf16 ? 5e-3 : 5e-2);
            worst = std::max(worst, diff / (std::fabs(want) + 1e-3));
            if (!(diff <= tol)) { ok = false; }
        }
        char extra[128];
        std::snprintf(extra, sizeof(extra), "worst_rel=%.3g", worst);
        check_tagged(ok, "gemm mt sampled host oracle", extra);
    }
    // Plain epilogue (no residual) with the same tiling.
    auto plain = dev.compile(make_qwen35_gemm_mt_kernel(cfg, format, QwenMatmulEpilogue::none));
    stream << plain(bx, bw, bs, dummy_f, by, M, N, K, 64u, plan.column_blocks)
                  .dispatch(plan.total_x, plan.total_y);
    auto y_plain = download(stream, by, size_t(M) * N);
    luisa::vector<float> ref_plain;
    check_tagged(ref_qwen_matmul(format, x, words, scales, {}, false, M, N, K, 64u, ref_plain).ok(),
                 "gemm mt plain oracle");
    if (!sampled_oracle_only) {
        check_close_tagged(y_plain, ref_plain, 3e-3f, rel, "gemm mt plain vs cpu oracle");
    } else {
        // Sampled comparison keeps the model shapes affordable.
        bool ok = true;
        for (uint32_t s = 0u; s < 8u && ok; s++) {
            const uint32_t row = (rng.next() % M);
            const uint32_t col = (rng.next() % N);
            double acc = 0.0;
            for (uint32_t ki = 0u; ki < K; ki++) {
                acc += double(x[size_t(row) * K + ki]) *
                       double(host_decode_packed(format, words, scales, K, 64u,
                                                 size_t(col) * K + ki));
            }
            const double got = double(y_plain[size_t(row) * N + col]);
            ok = std::fabs(got - acc) <= 3e-3 * std::fabs(acc) +
                                             (format == QwenWeightFormat::bf16 ? 5e-3 : 5e-2);
        }
        check_tagged(ok, "gemm mt plain sampled host oracle");
    }
}

void test_gemm_mt(Device &dev, Stream &stream, Rng &rng, Buffer<float> &dummy_f) {
    for (auto format : {QwenWeightFormat::bf16, QwenWeightFormat::fp8_e4m3}) {
        // (a) Full CPU-oracle differential over every m of interest on a ragged
        // K/N (130/132 x 37): 65 rows is the last full stripe of MT=16 plus a
        // tail, 63/17/5/3 are in-tile tails, 1..4 are decode/MTP verify sizes.
        const uint32_t k_small = format == QwenWeightFormat::bf16 ? 130u : 132u;
        for (uint32_t m : {1u, 2u, 3u, 4u, 5u, 16u, 17u, 63u, 64u, 65u}) {
            gemm_mt_check_shape(dev, stream, rng, dummy_f, format, m, 37u, k_small, false, false);
        }
        // (b) Model shapes: legacy-GEMM differential over the whole output plus a
        // sampled host oracle. n = 32 is the ragged 8-column-block case.
        gemm_mt_check_shape(dev, stream, rng, dummy_f, format, 64u, 9216u, 2560u, true, true);
        gemm_mt_check_shape(dev, stream, rng, dummy_f, format, 65u, 9216u, 2560u, true, true);
        gemm_mt_check_shape(dev, stream, rng, dummy_f, format, 17u, 32u, 2560u, true, false);
        gemm_mt_check_shape(dev, stream, rng, dummy_f, format, 16u, 1024u, 4096u, true, true);
        // (c) m = 1 at a model shape: the decode path never selects this kernel,
        // but the family must stay correct there (MTP verify row count 1..4).
        gemm_mt_check_shape(dev, stream, rng, dummy_f, format, 1u, 8192u, 2560u, true, true);
    }
    // (d) The planner's contract failures are the reason the executor keeps the
    // reference GEMM for those shapes (device-free checks live in the host
    // self-test; these document the boundary at the device API level).
    QwenGemmMtConfig cfg{};
    check(!plan_qwen35_gemm_mt_dispatch(8u, 32u, 129u, QwenWeightFormat::bf16, 64u, cfg).ok(),
          "gemm mt rejects odd bf16 K");
    check(!plan_qwen35_gemm_mt_dispatch(8u, 32u, 130u, QwenWeightFormat::fp8_e4m3, 64u, cfg).ok(),
          "gemm mt rejects fp8 K % 4 != 0");
    check(!plan_qwen35_gemm_mt_dispatch(8u, 32u, 2560u, QwenWeightFormat::fp8_e4m3, 62u, cfg).ok(),
          "gemm mt rejects a group that is not a multiple of 4");
    check(plan_qwen35_gemm_mt_dispatch(8u, 32u, 2560u, QwenWeightFormat::fp8_e4m3, 64u, cfg).ok(),
          "gemm mt accepts the model fp8 group");
}

// ---------------------------------------------------------------------------
// Embedding gather, both formats, ragged hidden; exact decode parity.
// ---------------------------------------------------------------------------
void test_embedding(Device &dev, Stream &stream, Rng &rng) {
    for (auto format : {QwenWeightFormat::bf16, QwenWeightFormat::fp8_e4m3}) {
        const uint64_t hidden = 66u, vocab = 9u;
        luisa::vector<float> table(hidden * vocab);
        rng.fill(table, -1.f, 1.f);
        luisa::vector<uint32_t> words;
        luisa::vector<float> scales;
        check(host_pack_matrix(format, table, hidden, vocab, 64u, words, scales).ok(), "embedding host pack");
        luisa::vector<uint32_t> ids{8u, 0u, 5u, 8u, 2u};// repeated + boundary ids
        auto bt = upload(dev, stream, words);
        auto bs = upload(dev, stream, scales.empty() ? luisa::vector<float>{0.f} : scales);
        auto bi = upload(dev, stream, ids);
        auto bo = dev.create_buffer<float>(ids.size() * hidden);
        auto sh = dev.compile(make_qwen_embedding_kernel(format));
        auto plan = plan_qwen_embedding_dispatch(ids.size(), hidden);
        check(plan.ok(), "embedding plan");
        stream << sh(bi, bt, bs, bo, uint32_t(ids.size()), uint32_t(hidden), uint32_t(vocab), 64u,
                     plan.value->padded_cols).dispatch(plan.value->total_x, plan.value->total_y);
        auto out = download(stream, bo, ids.size() * hidden);
        stream << synchronize();
        luisa::vector<float> ref;
        check(ref_qwen_embedding(format, ids, words, scales, hidden, vocab, 64u, ref).ok(),
              "embedding oracle");
        // Decode parity is EXACT: both paths decode the same packed words.
        check_exact(out, ref, format == QwenWeightFormat::bf16 ? "embedding bf16 exact" : "embedding fp8 exact");
    }
}

// ---------------------------------------------------------------------------
// RMSNorm plain + residual + snapshot, ragged dim; direct pre-added GGUF
// weights. The snapshot epilogue must copy x BIT-EXACTLY into sum_out during
// the sum-of-squares pass and produce y identical to plain; the residual
// buffer is a guarded 1-element canary (OOB writes there are detected).
// ---------------------------------------------------------------------------
void test_rmsnorm(Device &dev, Stream &stream, Rng &rng, Buffer<float> &dummy_f) {
    const uint32_t rows = 5u, dim = 37u;
    const size_t n = size_t(rows) * dim;
    luisa::vector<float> x(n), w(dim), res(n);
    rng.fill(x, -2.f, 2.f);
    rng.fill(w, 0.5f, 1.5f);// pre-added GGUF weights (direct multiply)
    rng.fill(res, -1.f, 1.f);
    auto bx = upload(dev, stream, x);
    auto bw = upload(dev, stream, w);
    auto br = upload(dev, stream, res);
    // Guarded outputs: canaries beyond rows*dim catch OOB writes.
    constexpr float kCanary = -12345.25f;
    constexpr size_t kPad = 8u;
    const luisa::vector<float> guard(n + kPad, kCanary);
    auto by = upload(dev, stream, guard);
    auto bsum = upload(dev, stream, guard);
    // Guarded residual dummy for plain/snapshot: must never be written.
    const float kDummySentinel = 777.5f;
    auto bdummy = upload(dev, stream, luisa::vector<float>{kDummySentinel});
    luisa::vector<float> tail_scratch(kPad);
    auto check_tail = [&](Buffer<float> &buf, const char *what) {
        stream << buf.view(n, kPad).copy_to(luisa::span{tail_scratch});
        stream << synchronize();
        bool ok = true;
        for (auto v : tail_scratch) { ok = ok && v == kCanary; }
        check(ok, what);
    };
    for (auto epi : {QwenNormEpilogue::plain, QwenNormEpilogue::residual, QwenNormEpilogue::snapshot}) {
        const bool residual = epi == QwenNormEpilogue::residual;
        const bool snapshot = epi == QwenNormEpilogue::snapshot;
        auto sh = dev.compile(make_qwen_rmsnorm_kernel(epi));
        auto plan = plan_qwen_rmsnorm_dispatch(rows, dim);
        check(plan.ok(), "rmsnorm plan");
        stream << sh(bx, bw, residual ? br : bdummy, by, (residual || snapshot) ? bsum : dummy_f,
                     rows, dim, 1e-6f)
                      .dispatch(qwen_dispatch_threads(*plan.value));
        auto y = download(stream, by, n);
        luisa::vector<float> sum;
        if (residual || snapshot) { sum = download(stream, bsum, n); }
        stream << synchronize();
        luisa::vector<float> ry, rsum;
        check(ref_qwen_rmsnorm(x, w, res, residual, rows, dim, 1e-6f, ry, rsum).ok(), "rmsnorm oracle");
        const char *label = residual ? "rmsnorm residual" : snapshot ? "rmsnorm snapshot" : "rmsnorm plain";
        check_close(y, ry, 1e-5f, 1e-4f, label);
        if (residual) { check_close(sum, rsum, 1e-6f, 1e-5f, "rmsnorm residual sum"); }
        if (snapshot) {
            // sum_out must be the ORIGINAL x, bit-exactly (residual snapshot).
            check_exact(sum, x, "rmsnorm snapshot sum == x exact");
            luisa::vector<float> sy, ssum;
            check(ref_qwen_rmsnorm_snapshot(x, w, rows, dim, 1e-6f, sy, ssum).ok(),
                  "rmsnorm snapshot oracle");
            check_close(y, sy, 1e-5f, 1e-4f, "rmsnorm snapshot vs snapshot oracle");
        }
        check_tail(by, "rmsnorm y no OOB tail write");
        if (residual || snapshot) { check_tail(bsum, "rmsnorm sum_out no OOB tail write"); }
        if (!residual) {
            auto d = download(stream, bdummy, 1);
            check(d.size() == 1 && d[0] == kDummySentinel, "rmsnorm residual dummy untouched");
        }
        // x and w must be read-only for every epilogue.
        auto xback = download(stream, bx, n);
        check_exact(xback, x, "rmsnorm x unmodified");
        auto wback = download(stream, bw, dim);
        check_exact(wback, w, "rmsnorm w unmodified");
    }
    // Snapshot must produce EXACTLY the same y bits as plain (same FP32 math,
    // normalization reads the bit-exact copy instead of x).
    {
        auto run = [&](QwenNormEpilogue epi) {
            auto sh = dev.compile(make_qwen_rmsnorm_kernel(epi));
            auto plan = plan_qwen_rmsnorm_dispatch(rows, dim);
            stream << sh(bx, bw, bdummy, by, bsum, rows, dim, 1e-6f)
                          .dispatch(qwen_dispatch_threads(*plan.value));
            auto y = download(stream, by, n);
            stream << synchronize();
            return y;
        };
        const auto yp = run(QwenNormEpilogue::plain);
        const auto ys = run(QwenNormEpilogue::snapshot);
        check_exact(ys, yp, "rmsnorm snapshot y == plain y bitwise");
    }
}

// ---------------------------------------------------------------------------
// RMSNorm snapshot on ragged row counts / dims (dim < warp, dim = warp,
// dim > warp non-multiple, rows > warps-per-block), plus zero-row and
// zero-dim empty-plan contracts. Both y and sum_out checked; guarded tails.
// ---------------------------------------------------------------------------
void test_rmsnorm_snapshot_ragged(Device &dev, Stream &stream, Rng &rng, Buffer<float> &dummy_f) {
    struct Shape { uint32_t rows, dim; };
    const Shape shapes[] = {{1u, 1u}, {3u, 31u}, {2u, 32u}, {7u, 33u}, {9u, 130u}, {5u, 2560u}};
    constexpr float kCanary = -98765.5f;
    constexpr size_t kPad = 8u;
    auto sh = dev.compile(make_qwen_rmsnorm_kernel(QwenNormEpilogue::snapshot));
    for (auto [rows, dim] : shapes) {
        const size_t n = size_t(rows) * dim;
        luisa::vector<float> x(n), w(dim);
        rng.fill(x, -2.f, 2.f);
        rng.fill(w, 0.5f, 1.5f);
        auto bx = upload(dev, stream, x);
        auto bw = upload(dev, stream, w);
        const luisa::vector<float> guard(n + kPad, kCanary);
        auto by = upload(dev, stream, guard);
        auto bsum = upload(dev, stream, guard);
        auto plan = plan_qwen_rmsnorm_dispatch(rows, dim);
        check(plan.ok(), "rmsnorm snapshot ragged plan");
        stream << sh(bx, bw, dummy_f, by, bsum, rows, dim, 1e-6f)
                      .dispatch(qwen_dispatch_threads(*plan.value));
        auto y = download(stream, by, n);
        auto sum = download(stream, bsum, n);
        luisa::vector<float> ytail(kPad), stail(kPad);
        stream << by.view(n, kPad).copy_to(luisa::span{ytail})
               << bsum.view(n, kPad).copy_to(luisa::span{stail}) << synchronize();
        check_exact(sum, x, "snapshot ragged sum == x exact");
        luisa::vector<float> ry, rsum;
        check(ref_qwen_rmsnorm_snapshot(x, w, rows, dim, 1e-6f, ry, rsum).ok(),
              "snapshot ragged oracle");
        check_close(y, ry, 1e-5f, 1e-4f, "snapshot ragged y");
        bool ok = true;
        for (size_t i = 0; i < kPad; i++) { ok = ok && ytail[i] == kCanary && stail[i] == kCanary; }
        check(ok, "snapshot ragged no OOB tail write");
    }
    // Zero-row plan is empty: callers must skip the dispatch.
    check(plan_qwen_rmsnorm_dispatch(0u, 37u).value->empty(), "snapshot zero rows empty plan");
    // dim == 0: the shared warp-row planner reports rows warps (NOT empty),
    // but the kernel's block-uniform `dim > 0` guard makes the launch a safe
    // no-op. Verify no buffer is touched (canaries everywhere).
    {
        auto plan = plan_qwen_rmsnorm_dispatch(2u, 0u);
        check(plan.ok(), "snapshot zero dim plan");
        luisa::vector<float> w1{1.f};
        auto bw = upload(dev, stream, w1);
        const luisa::vector<float> guard(kPad, kCanary);
        auto by = upload(dev, stream, guard);
        auto bsum = upload(dev, stream, guard);
        stream << sh(dummy_f, bw, dummy_f, by, bsum, 2u, 0u, 1e-6f)
                      .dispatch(qwen_dispatch_threads(*plan.value));
        luisa::vector<float> ytail(kPad), stail(kPad);
        stream << by.view(0, kPad).copy_to(luisa::span{ytail})
               << bsum.view(0, kPad).copy_to(luisa::span{stail}) << synchronize();
        bool ok = true;
        for (size_t i = 0; i < kPad; i++) { ok = ok && ytail[i] == kCanary && stail[i] == kCanary; }
        check(ok, "snapshot zero dim launch is guarded no-op");
    }
}

void test_qwen35_wide_rmsnorm(Device &dev, Stream &stream, Rng &rng, Buffer<float> &dummy_f) {
    for (uint32_t dim : {257u, 2560u}) {
        const uint32_t rows = 3u;
        const size_t count = size_t(rows) * dim;
        luisa::vector<float> x(count), w(dim), residual(count);
        rng.fill(x, -2.f, 2.f);
        rng.fill(w, 0.5f, 1.5f);
        rng.fill(residual, -1.f, 1.f);
        auto bx = upload(dev, stream, x);
        auto bw = upload(dev, stream, w);
        auto br = upload(dev, stream, residual);
        auto by = dev.create_buffer<float>(count);
        auto bs = dev.create_buffer<float>(count);
        for (uint32_t block : {128u, 256u}) {
        for (auto epi : {QwenNormEpilogue::plain, QwenNormEpilogue::residual,
                         QwenNormEpilogue::snapshot}) {
            const bool has_residual = epi == QwenNormEpilogue::residual;
            const bool has_sum = epi != QwenNormEpilogue::plain;
            auto sh = dev.compile(make_qwen35_rmsnorm_kernel(epi, block));
            auto plan = plan_qwen35_rmsnorm_dispatch(rows, dim, block);
            check(plan.ok(), "qwen35 wide norm plan");
            stream << sh(bx, bw, has_residual ? br : dummy_f, by,
                         has_sum ? bs : dummy_f, rows, dim, 1e-6f)
                          .dispatch(qwen_dispatch_threads(*plan.value));
            auto y = download(stream, by, count);
            luisa::vector<float> ref, sum;
            if (epi == QwenNormEpilogue::snapshot) {
                check(ref_qwen_rmsnorm_snapshot(x, w, rows, dim, 1e-6f, ref, sum).ok(),
                      "qwen35 wide snapshot oracle");
                check_exact(download(stream, bs, count), x, "qwen35 wide snapshot exact copy");
            } else {
                check(ref_qwen_rmsnorm(x, w, residual, has_residual, rows, dim,
                                       1e-6f, ref, sum).ok(), "qwen35 wide norm oracle");
                if (has_residual) {
                    check_close(download(stream, bs, count), sum, 1e-6f, 1e-5f,
                                "qwen35 wide residual sum");
                }
            }
            check_close(y, ref, 2e-5f, 2e-4f, "qwen35 wide rmsnorm");
        }
        }
    }
}

// ---------------------------------------------------------------------------
// SwiGLU split + fused, ragged F.
// ---------------------------------------------------------------------------
void test_swiglu(Device &dev, Stream &stream, Rng &rng) {
    const uint32_t m = 3u, f = 17u;
    luisa::vector<float> proj(m * 2u * f);
    rng.fill(proj, -3.f, 3.f);
    luisa::vector<float> gate, up;
    for (uint32_t mi = 0; mi < m; mi++) {
        for (uint32_t i = 0; i < f; i++) {
            gate.push_back(proj[mi * 2u * f + i]);
            up.push_back(proj[mi * 2u * f + f + i]);
        }
    }
    auto bg = upload(dev, stream, gate);
    auto bu = upload(dev, stream, up);
    auto bp = upload(dev, stream, proj);
    auto bo = dev.create_buffer<float>(m * f);
    {
        auto sh = dev.compile(make_qwen_swiglu_split_kernel());
        auto plan = plan_qwen_swiglu_dispatch(m * f);
        stream << sh(bg, bu, bo, m * f).dispatch(qwen_dispatch_threads(*plan.value));
        auto y = download(stream, bo, m * f);
        stream << synchronize();
        luisa::vector<float> ref;
        check(ref_qwen_swiglu(gate, up, ref).ok(), "swiglu split oracle");
        check_close(y, ref, 1e-6f, 1e-5f, "swiglu split");
    }
    {
        auto sh = dev.compile(make_qwen_swiglu_fused_kernel());
        auto plan = plan_qwen_swiglu_fused_dispatch(m, f);
        stream << sh(bp, bo, m, f, plan.value->padded_cols).dispatch(plan.value->total_x, plan.value->total_y);
        auto y = download(stream, bo, m * f);
        stream << synchronize();
        luisa::vector<float> ref;
        check(ref_qwen_swiglu_fused(proj, m, f, ref).ok(), "swiglu fused oracle");
        check_close(y, ref, 1e-6f, 1e-5f, "swiglu fused");
    }
}

void test_ffn_decode_fusion(Device &dev, Stream &stream, Rng &rng) {
    constexpr uint32_t F = 37u, K = 2560u;
    luisa::vector<float> x(K), gate(size_t(F) * K), up(gate.size());
    rng.fill(x, -1.f, 1.f);
    rng.fill(gate, -1.f, 1.f);
    rng.fill(up, -1.f, 1.f);
    for (auto format : {QwenWeightFormat::bf16, QwenWeightFormat::fp8_e4m3}) {
        luisa::vector<uint32_t> gw, uw;
        luisa::vector<float> gs, us;
        check(host_pack_matrix(format, gate, K, F, 64u, gw, gs).ok(), "fused FFN gate pack");
        check(host_pack_matrix(format, up, K, F, 64u, uw, us).ok(), "fused FFN up pack");
        auto bx = upload(dev, stream, x);
        auto bg = upload(dev, stream, gw);
        auto bu = upload(dev, stream, uw);
        auto bgs = upload(dev, stream, gs.empty() ? luisa::vector<float>{0.f} : gs);
        auto bus = upload(dev, stream, us.empty() ? luisa::vector<float>{0.f} : us);
        auto out = dev.create_buffer<float>(F);
        auto sh = dev.compile(make_qwen_ffn_decode_kernel(format));
        auto plan = plan_qwen_ffn_decode_dispatch(F, K, format);
        check(plan.ok(), "fused FFN decode plan");
        stream << sh(bx, bg, bgs, bu, bus, out, F, K, 64u)
                      .dispatch(qwen_dispatch_threads(*plan.value));
        auto y = download(stream, out, F);
        luisa::vector<float> gate_ref, up_ref, ref;
        check(ref_qwen_matmul(format, x, gw, gs, {}, false, 1u, F, K, 64u, gate_ref).ok(),
              "fused FFN gate oracle");
        check(ref_qwen_matmul(format, x, uw, us, {}, false, 1u, F, K, 64u, up_ref).ok(),
              "fused FFN up oracle");
        check(ref_qwen_swiglu(gate_ref, up_ref, ref).ok(), "fused FFN swiglu oracle");
        check_close(y, ref, 3e-3f, format == QwenWeightFormat::bf16 ? 2e-2f : 0.4f,
                    "fused FFN decode");
    }
}

void test_greedy_argmax(Device &dev, Stream &stream) {
    constexpr uint32_t vocab = 248320u;
    luisa::vector<float> logits(vocab, -std::numeric_limits<float>::infinity());
    auto ids = dev.create_buffer<uint32_t>(1u);
    auto history = dev.create_buffer<uint32_t>(4u);
    auto sh = dev.compile(make_qwen_greedy_argmax_kernel());
    auto plan = plan_qwen_greedy_argmax_dispatch(vocab);
    check(plan.ok(), "greedy argmax plan");
    auto partial = dev.compile(make_qwen_greedy_argmax_partial_kernel());
    auto finish = dev.compile(make_qwen_greedy_argmax_finish_kernel());
    auto partial_plan = plan_qwen_greedy_argmax_partial_dispatch(vocab);
    auto finish_plan = plan_qwen_greedy_argmax_finish_dispatch(vocab);
    auto values = dev.create_buffer<float>(qwen_greedy_argmax_blocks);
    auto indices = dev.create_buffer<uint32_t>(qwen_greedy_argmax_blocks);
    auto invalid = dev.create_buffer<uint32_t>(qwen_greedy_argmax_blocks);
    auto parallel_ids = dev.create_buffer<uint32_t>(1u);
    auto parallel_history = dev.create_buffer<uint32_t>(4u);
    const auto run = [&](uint32_t slot) {
        auto input = upload(dev, stream, logits);
        stream << sh(input, ids, history, vocab, slot)
                      .dispatch(qwen_dispatch_threads(*plan.value));
        auto selected = download(stream, ids, 1u);
        auto saved = download(stream, history, 4u);
        check(selected.size() == 1u && saved.size() == 4u && selected[0] == saved[slot],
              "greedy argmax next-id/history parity");
        stream << partial(input, values, indices, invalid, vocab)
                      .dispatch(qwen_dispatch_threads(*partial_plan.value))
               << finish(values, indices, invalid, parallel_ids, parallel_history, vocab, slot)
                      .dispatch(qwen_dispatch_threads(*finish_plan.value));
        auto pselected = download(stream, parallel_ids, 1u);
        auto psaved = download(stream, parallel_history, 4u);
        check(pselected.size() == 1u && psaved.size() == 4u &&
              pselected[0] == selected[0] && psaved[slot] == selected[0],
              "greedy parallel argmax parity");
        return selected[0];
    };
    check(run(0u) == 0u, "greedy argmax all -inf chooses lowest index");
    logits[17] = 42.f;
    logits[65536] = 42.f;
    logits[248319] = 42.f;
    check(run(1u) == 17u, "greedy argmax finite lowest-index tie");
    logits[100] = std::numeric_limits<float>::infinity();
    logits[200] = std::numeric_limits<float>::infinity();
    check(run(2u) == 100u, "greedy argmax +inf lowest-index tie");
    logits[999] = std::numeric_limits<float>::quiet_NaN();
    check(run(3u) == vocab, "greedy argmax NaN fail-closed sentinel");
}

// ---------------------------------------------------------------------------
// Full attention chain: prepare(+append) -> prefill -> decode -> standalone
// RoPE / KV append. GQA 4q/2kv, hd=64, rope 16, history pos0=3, T=5, cap=11.
// ---------------------------------------------------------------------------
void test_attention(Device &dev, Stream &stream, Rng &rng) {
    // The f32/bf16 attention arms take the fp8 scale arguments but never read them
    // (the scale load folds away), so one 1-element buffer serves every unscaled
    // dispatch of this test.
    auto bscale_dummy = dev.create_buffer<float>(1);
    QwenDslConfig cfg;
    cfg.q_heads = 4u;
    cfg.kv_heads = 2u;
    cfg.attn_head_dim = 64u;
    cfg.rope_dim = 16u;
    const uint32_t hd = cfg.attn_head_dim, T = 5u, pos0 = 3u, capacity = 11u;
    luisa::vector<float> q_proj(T * cfg.attn_q_proj_n()), k_proj(T * cfg.attn_kv_n()), v_proj(T * cfg.attn_kv_n());
    luisa::vector<float> qnw(hd), knw(hd);
    luisa::vector<int32_t> positions(T);
    rng.fill(q_proj, -1.f, 1.f);
    rng.fill(k_proj, -1.f, 1.f);
    rng.fill(v_proj, -1.f, 1.f);
    rng.fill(qnw, 0.7f, 1.3f);// pre-added GGUF weights
    rng.fill(knw, 0.7f, 1.3f);
    for (uint32_t t = 0; t < T; t++) { positions[t] = int32_t(pos0 + t); }
    luisa::vector<float> k_cache(capacity * cfg.attn_kv_n()), v_cache(capacity * cfg.attn_kv_n());
    rng.fill(k_cache, -1.f, 1.f);
    rng.fill(v_cache, -1.f, 1.f);
    luisa::vector<float> rq, rgate, rk, rv;
    check(ref_qwen_attn_prepare(q_proj, k_proj, v_proj, qnw, knw, positions, T, cfg, pos0, true,
                                k_cache, v_cache, rq, rgate, rk, rv).ok(),
          "attn prepare oracle");
    auto bqp = upload(dev, stream, q_proj);
    auto bkp = upload(dev, stream, k_proj);
    auto bvp = upload(dev, stream, v_proj);
    auto bqnw = upload(dev, stream, qnw);
    auto bknw = upload(dev, stream, knw);
    auto bpos = upload(dev, stream, positions);
    auto bq = dev.create_buffer<float>(rq.size());
    auto bgate = dev.create_buffer<float>(rgate.size());
    auto bk = dev.create_buffer<float>(rk.size());
    auto bv = dev.create_buffer<float>(rv.size());
    auto bkc = upload(dev, stream, k_cache);
    auto bvc = upload(dev, stream, v_cache);
    {
        auto sh = dev.compile(make_qwen_attn_prepare_kernel(true));
        auto plan = plan_qwen_attn_prepare_dispatch(T, cfg);
        check(plan.ok(), "attn prepare plan");
        const uint32_t q_rows = T * cfg.q_heads, k_rows = T * cfg.kv_heads, v_rows = k_rows;
        stream << sh(bqp, bkp, bvp, bqnw, bknw, bpos, bq, bgate, bk, bv, bkc, bvc,
                     T, cfg.q_heads, cfg.kv_heads, hd, cfg.rope_dim, cfg.rope_theta, cfg.norm_eps,
                     pos0, capacity, q_rows, k_rows, v_rows)
                      .dispatch(qwen_dispatch_threads(*plan.value));
        auto dq = download(stream, bq, rq.size());
        auto dgate = download(stream, bgate, rgate.size());
        auto dk = download(stream, bk, rk.size());
        auto dv = download(stream, bv, rv.size());
        auto dkc = download(stream, bkc, k_cache.size());
        auto dvc = download(stream, bvc, v_cache.size());
        stream << synchronize();
        check_close(dq, rq, 2e-5f, 2e-4f, "attn prepare q (norm+rope)");
        check_exact(dgate, rgate, "attn prepare gate raw");
        check_close(dk, rk, 2e-5f, 2e-4f, "attn prepare k (norm+rope)");
        check_exact(dv, rv, "attn prepare v raw");
        check_close(dkc, k_cache, 2e-5f, 2e-4f, "attn prepare k cache append");
        check_exact(dvc, v_cache, "attn prepare v cache append");
        auto implicit = dev.compile(make_qwen_attn_prepare_kernel(true, true));
        stream << implicit(bqp, bkp, bvp, bqnw, bknw, bpos, bq, bgate, bk, bv, bkc, bvc,
                           T, cfg.q_heads, cfg.kv_heads, hd, cfg.rope_dim, cfg.rope_theta,
                           cfg.norm_eps, pos0, capacity, q_rows, k_rows, v_rows)
                      .dispatch(qwen_dispatch_threads(*plan.value));
        check_close(download(stream, bq, rq.size()), rq, 2e-5f, 2e-4f, "attn prepare implicit-position q");
        check_exact(download(stream, bgate, rgate.size()), rgate, "attn prepare implicit-position gate");
        check_close(download(stream, bk, rk.size()), rk, 2e-5f, 2e-4f, "attn prepare implicit-position k");
        check_exact(download(stream, bv, rv.size()), rv, "attn prepare implicit-position v");
        check_close(download(stream, bkc, k_cache.size()), k_cache, 2e-5f, 2e-4f, "attn prepare implicit-position k cache");
        check_exact(download(stream, bvc, v_cache.size()), v_cache, "attn prepare implicit-position v cache");
    }
    const uint32_t kv_len = pos0 + T;
    luisa::vector<float> rpref;
    check(ref_qwen_attn_prefill(rq, rgate, k_cache, v_cache, T, cfg.q_heads, cfg.kv_heads, hd,
                                cfg.attn_scale(), pos0, kv_len, rpref).ok(),
          "attn prefill oracle");
    auto bpref = dev.create_buffer<float>(rpref.size());
    {
        auto sh = dev.compile(make_qwen_attn_prefill_kernel());
        auto plan = plan_qwen_attn_prefill_dispatch(T, cfg.q_heads);
        stream << sh(bq, bgate, bkc, bvc, bscale_dummy.view(), bscale_dummy.view(), bpref, T, cfg.q_heads, cfg.kv_heads, hd,
                     cfg.attn_scale(), pos0, kv_len, capacity).dispatch(qwen_dispatch_threads(*plan.value));
        auto dpref = download(stream, bpref, rpref.size());
        stream << synchronize();
        check_close(dpref, rpref, 2e-5f, 2e-4f, "attn prefill online softmax + gate");
    }
    luisa::vector<float> q_last(rq.end() - cfg.q_heads * hd, rq.end());
    luisa::vector<float> gate_last(rgate.end() - cfg.q_heads * hd, rgate.end());
    auto bql = upload(dev, stream, q_last);
    auto bgl = upload(dev, stream, gate_last);
    auto bdec = dev.create_buffer<float>(cfg.q_heads * hd);
    luisa::vector<float> rdec;
    check(ref_qwen_attn_decode(q_last, gate_last, k_cache, v_cache, cfg.q_heads, cfg.kv_heads, hd,
                               cfg.attn_scale(), kv_len, rdec).ok(),
          "attn decode oracle");
    {
        auto sh = dev.compile(make_qwen_attn_decode_kernel());
        auto plan = plan_qwen_attn_decode_dispatch(cfg.q_heads);
        stream << sh(bql, bgl, bkc, bvc, bscale_dummy.view(), bscale_dummy.view(), bdec, cfg.q_heads, cfg.kv_heads, hd,
                     cfg.attn_scale(), kv_len, capacity).dispatch(qwen_dispatch_threads(*plan.value));
        auto ddec = download(stream, bdec, rdec.size());
        stream << synchronize();
        check_close(ddec, rdec, 2e-5f, 2e-4f, "attn decode block-cooperative + gate");
    }
    // Standalone partial RoPE on the normed q (double rotation parity).
    {
        luisa::vector<float> rrope;
        check(ref_qwen_rope_partial(rq, positions, T, cfg.q_heads, hd, cfg.rope_dim, cfg.rope_theta, rrope).ok(),
              "rope oracle");
        auto brope = dev.create_buffer<float>(rrope.size());
        auto sh = dev.compile(make_qwen_rope_partial_neox_kernel());
        auto plan = plan_qwen_rope_dispatch(T, cfg.q_heads);
        stream << sh(bq, brope, bpos, T, cfg.q_heads, hd, cfg.rope_dim, cfg.rope_theta, T * cfg.q_heads)
                      .dispatch(qwen_dispatch_threads(*plan.value));
        auto drope = download(stream, brope, rrope.size());
        stream << synchronize();
        check_close(drope, rrope, 2e-5f, 2e-4f, "rope partial neox");
    }
    // KV append standalone (append device k/v at pos0 into fresh caches).
    {
        auto bkc2 = dev.create_buffer<float>(capacity * cfg.attn_kv_n());
        auto bvc2 = dev.create_buffer<float>(capacity * cfg.attn_kv_n());
        auto sh = dev.compile(make_qwen_kv_append_kernel());
        auto plan = plan_qwen_kv_append_dispatch(T, cfg.attn_kv_n());
        stream << sh(bk, bv, bkc2, bvc2, T, cfg.attn_kv_n(), pos0, capacity, plan.value->padded_cols)
                      .dispatch(plan.value->total_x, plan.value->total_y);
        auto dkc2 = download(stream, bkc2, capacity * cfg.attn_kv_n());
        auto dvc2 = download(stream, bvc2, capacity * cfg.attn_kv_n());
        auto dk_src = download(stream, bk, rk.size());
        auto dv_src = download(stream, bv, rv.size());
        stream << synchronize();
        bool ok = true;
        size_t bad = 0;
        for (uint32_t t = 0; t < T && ok; t++) {
            for (uint32_t c = 0; c < cfg.attn_kv_n(); c++) {
                const size_t idx = size_t(pos0 + t) * cfg.attn_kv_n() + c;
                const size_t src = size_t(t) * cfg.attn_kv_n() + c;
                if (dkc2[idx] != dk_src[src] || dvc2[idx] != dv_src[src]) { ok = false; bad = idx; }
            }
        }
        check(ok, "kv append standalone", ok ? luisa::string{} : luisa::string{"cache index "} + std::to_string(bad).c_str());
    }
}

// ---------------------------------------------------------------------------
// Decode edge cases: kv_len 1..capacity (empty warp subsets in the 4-warp
// block) and kv_len not a multiple of 4.
// ---------------------------------------------------------------------------
void test_decode_edges(Device &dev, Stream &stream, Rng &rng) {
    // The f32/bf16 attention arms take the fp8 scale arguments but never read them
    // (the scale load folds away), so one 1-element buffer serves every unscaled
    // dispatch of this test.
    auto bscale_dummy = dev.create_buffer<float>(1);
    QwenDslConfig cfg;
    cfg.q_heads = 4u;
    cfg.kv_heads = 2u;
    cfg.attn_head_dim = 64u;
    cfg.rope_dim = 16u;
    const uint32_t hd = cfg.attn_head_dim, capacity = 7u;
    luisa::vector<float> k_cache(capacity * cfg.attn_kv_n()), v_cache(capacity * cfg.attn_kv_n());
    rng.fill(k_cache, -1.f, 1.f);
    rng.fill(v_cache, -1.f, 1.f);
    luisa::vector<float> q(cfg.q_heads * hd), gate(cfg.q_heads * hd);
    rng.fill(q, -1.f, 1.f);
    rng.fill(gate, -2.f, 2.f);
    auto bqc = upload(dev, stream, q);
    auto bgc = upload(dev, stream, gate);
    auto bkc = upload(dev, stream, k_cache);
    auto bvc = upload(dev, stream, v_cache);
    auto sh = dev.compile(make_qwen_attn_decode_kernel());
    auto plan = plan_qwen_attn_decode_dispatch(cfg.q_heads);
    for (uint32_t kv_len : {1u, 2u, 3u, 5u, 7u}) {
        auto bdec = dev.create_buffer<float>(cfg.q_heads * hd);
        stream << sh(bqc, bgc, bkc, bvc, bscale_dummy.view(), bscale_dummy.view(), bdec, cfg.q_heads, cfg.kv_heads, hd,
                     cfg.attn_scale(), kv_len, capacity).dispatch(qwen_dispatch_threads(*plan.value));
        auto ddec = download(stream, bdec, cfg.q_heads * hd);
        stream << synchronize();
        luisa::vector<float> rdec;
        check(ref_qwen_attn_decode(q, gate, k_cache, v_cache, cfg.q_heads, cfg.kv_heads, hd,
                                   cfg.attn_scale(), kv_len, rdec).ok(),
              "decode edge oracle");
        char name[64];
        std::snprintf(name, sizeof(name), "attn decode kv_len=%u edge", kv_len);
        check_close(ddec, rdec, 2e-5f, 2e-4f, name);
    }
}

void test_gdn_decode_gates(Device &dev, Stream &stream, Rng &rng) {
    constexpr uint32_t H = 4u, K = 2560u;
    luisa::vector<float> x(K), aw(size_t(H) * K), bw(aw.size()), ssm_a(H), dt(H);
    rng.fill(x, -1.f, 1.f);
    rng.fill(aw, -1.f, 1.f);
    rng.fill(bw, -1.f, 1.f);
    rng.fill(ssm_a, -2.f, -0.1f);
    rng.fill(dt, -0.5f, 0.5f);
    for (auto format : {QwenWeightFormat::bf16, QwenWeightFormat::fp8_e4m3}) {
        luisa::vector<uint32_t> ap, bp;
        luisa::vector<float> as, bs;
        check(host_pack_matrix(format, aw, K, H, 64u, ap, as).ok(), "gdn decode alpha pack");
        check(host_pack_matrix(format, bw, K, H, 64u, bp, bs).ok(), "gdn decode beta pack");
        auto bx = upload(dev, stream, x);
        auto ba = upload(dev, stream, ap);
        auto bb = upload(dev, stream, bp);
        auto bas = upload(dev, stream, as.empty() ? luisa::vector<float>{0.f} : as);
        auto bbs = upload(dev, stream, bs.empty() ? luisa::vector<float>{0.f} : bs);
        auto bsa = upload(dev, stream, ssm_a);
        auto bdt = upload(dev, stream, dt);
        auto beta = dev.create_buffer<float>(H);
        auto g = dev.create_buffer<float>(H);
        auto sh = dev.compile(make_qwen_gdn_decode_gates_kernel(format));
        auto plan = plan_qwen_gdn_decode_gates_dispatch(H, K, format);
        check(plan.ok(), "gdn decode gates plan");
        stream << sh(bx, ba, bas, bb, bbs, bsa, bdt, beta, g, K, H, 64u, 20.f)
                      .dispatch(qwen_dispatch_threads(*plan.value));
        luisa::vector<float> a_proj, b_proj, beta_ref, g_ref;
        check(ref_qwen_matmul(format, x, ap, as, {}, false, 1u, H, K, 64u, a_proj).ok(),
              "gdn decode alpha oracle");
        check(ref_qwen_matmul(format, x, bp, bs, {}, false, 1u, H, K, 64u, b_proj).ok(),
              "gdn decode beta oracle");
        check(ref_qwen_gdn_gates(a_proj, b_proj, ssm_a, dt, 1u, H, 20.f, beta_ref, g_ref).ok(),
              "gdn decode gates oracle");
        auto beta_dev = download(stream, beta, H);
        auto g_dev = download(stream, g, H);
        check_close(beta_dev, beta_ref, 3e-3f,
                    format == QwenWeightFormat::bf16 ? 2e-2f : 0.4f, "gdn decode beta fusion");
        check_close(g_dev, g_ref, 3e-3f,
                    format == QwenWeightFormat::bf16 ? 2e-2f : 0.4f, "gdn decode alpha fusion");
    }
}

// ---------------------------------------------------------------------------
// GDN chain: conv (two chunks, carried ring) -> gates -> qk norm -> recurrent
// (prefill + step-by-step decode parity) -> gated norm.
// ---------------------------------------------------------------------------
void test_gdn(Device &dev, Stream &stream, Rng &rng) {
    QwenDslConfig cfg;
    cfg.gdn_qk_heads = 2u;
    cfg.gdn_v_heads = 4u;
    cfg.gdn_head_dim = 32u;
    const uint32_t hd = cfg.gdn_head_dim, C = cfg.gdn_qkv_n(), K = cfg.conv_kernel;
    const uint32_t T1 = 3u, T2 = 4u, T = T1 + T2, H = cfg.gdn_v_heads;
    luisa::vector<float> src1(T1 * C), src2(T2 * C), weight(K * C);
    luisa::vector<float> ring(C * (K - 1u), 0.f);
    rng.fill(src1, -1.f, 1.f);
    rng.fill(src2, -1.f, 1.f);
    rng.fill(weight, -0.5f, 0.5f);
    luisa::vector<float> rconv1, rconv2;
    check(ref_qwen_gdn_conv(src1, weight, ring, T1, C, K, 0u, rconv1).ok(), "gdn conv oracle 1");
    check(ref_qwen_gdn_conv(src2, weight, ring, T2, C, K, T1, rconv2).ok(), "gdn conv oracle 2");
    luisa::vector<float> rconv = rconv1;
    rconv.insert(rconv.end(), rconv2.begin(), rconv2.end());
    auto bsrc1 = upload(dev, stream, src1);
    auto bsrc2 = upload(dev, stream, src2);
    auto bw = upload(dev, stream, weight);
    auto bring = dev.create_buffer<float>(C * (K - 1u));
    auto bconv1 = dev.create_buffer<float>(T1 * C);
    auto bconv2 = dev.create_buffer<float>(T2 * C);
    auto zsh = dev.compile(make_qwen_zero_fill_kernel());
    {
        auto zplan = plan_qwen_threads(C * (K - 1u));
        stream << zsh(bring, C * (K - 1u)).dispatch(qwen_dispatch_threads(*zplan.value));
        auto sh = dev.compile(make_qwen_gdn_conv_silu_kernel());
        auto plan = plan_qwen_gdn_conv_dispatch(C);
        stream << sh(bsrc1, bw, bring, bconv1, T1, C, K, 0u).dispatch(qwen_dispatch_threads(*plan.value));
        stream << sh(bsrc2, bw, bring, bconv2, T2, C, K, T1).dispatch(qwen_dispatch_threads(*plan.value));
        auto dconv1 = download(stream, bconv1, T1 * C);
        auto dconv2 = download(stream, bconv2, T2 * C);
        auto dring = download(stream, bring, C * (K - 1u));
        stream << synchronize();
        check_close(dconv1, rconv1, 1e-5f, 1e-4f, "gdn conv chunk 1");
        check_close(dconv2, rconv2, 1e-5f, 1e-4f, "gdn conv chunk 2 (history)");
        check_exact(dring, ring, "gdn conv ring final");
    }
    luisa::vector<float> dconv = download(stream, bconv1, T1 * C);
    {
        auto tmp = download(stream, bconv2, T2 * C);
        dconv.insert(dconv.end(), tmp.begin(), tmp.end());
        stream << synchronize();
    }
    auto bconv = upload(dev, stream, dconv);
    // Gates (ssm_a is the pre-folded GGUF -exp(A_log): always negative).
    luisa::vector<float> a(T * H), b(T * H), ssm_a(H), dt_bias(H);
    rng.fill(a, -4.f, 4.f);
    rng.fill(b, -4.f, 4.f);
    for (auto &v : ssm_a) { v = -std::exp(rng.u(-4.6f, 2.77f)); }
    rng.fill(dt_bias, -1.f, 1.f);
    luisa::vector<float> rbeta, rg;
    check(ref_qwen_gdn_gates(a, b, ssm_a, dt_bias, T, H, cfg.softplus_threshold, rbeta, rg).ok(),
          "gdn gates oracle");
    auto ba = upload(dev, stream, a);
    auto bb = upload(dev, stream, b);
    auto bsa = upload(dev, stream, ssm_a);
    auto bdt = upload(dev, stream, dt_bias);
    auto bbeta = dev.create_buffer<float>(T * H);
    auto bg = dev.create_buffer<float>(T * H);
    {
        auto sh = dev.compile(make_qwen_gdn_gates_kernel());
        auto plan = plan_qwen_gdn_gates_dispatch(T, H);
        stream << sh(ba, bb, bsa, bdt, bbeta, bg, T, H, cfg.softplus_threshold)
                      .dispatch(qwen_dispatch_threads(*plan.value));
        auto dbeta = download(stream, bbeta, T * H);
        auto dg = download(stream, bg, T * H);
        stream << synchronize();
        check_close(dbeta, rbeta, 1e-6f, 1e-5f, "gdn gates beta");
        check_close(dg, rg, 1e-5f, 1e-4f, "gdn gates g");
    }
    // QK L2 norm over the packed conv output.
    luisa::vector<float> rq, rk;
    check(ref_qwen_gdn_qk_norm(rconv, T, cfg.gdn_qk_heads, hd, cfg.gdn_v_n(), cfg.l2_eps,
                               cfg.gdn_q_scale(), rq, rk).ok(),
          "gdn qk oracle");
    auto bq = dev.create_buffer<float>(rq.size());
    auto bk = dev.create_buffer<float>(rk.size());
    {
        auto sh = dev.compile(make_qwen_gdn_qk_l2norm_kernel());
        auto plan = plan_qwen_gdn_qk_norm_dispatch(T, cfg.gdn_qk_heads);
        stream << sh(bconv, bq, bk, T, cfg.gdn_qk_heads, hd, cfg.gdn_v_n(), cfg.l2_eps, cfg.gdn_q_scale(),
                     T * cfg.gdn_qk_heads).dispatch(qwen_dispatch_threads(*plan.value));
        auto dq = download(stream, bq, rq.size());
        auto dk = download(stream, bk, rk.size());
        stream << synchronize();
        check_close(dq, rq, 1e-5f, 1e-4f, "gdn qk l2 norm q");
        check_close(dk, rk, 1e-5f, 1e-4f, "gdn qk l2 norm k");
    }
    // Recurrent prefill vs oracle; then step-by-step decode parity.
    luisa::vector<float> state(size_t(H) * hd * hd);
    rng.fill(state, -0.1f, 0.1f);
    luisa::vector<float> state_ref = state;
    luisa::vector<float> rout;
    check(ref_qwen_gdn_recurrent(rq, rk, rconv, rbeta, rg, state_ref, T, H, hd, cfg.gdn_qk_heads, rout).ok(),
          "gdn recurrent oracle");
    auto bstate = upload(dev, stream, state);
    auto bout = dev.create_buffer<float>(size_t(T) * H * hd);
    {
        auto sh = dev.compile(make_qwen_gdn_recurrent_kernel());
        auto plan = plan_qwen_gdn_recurrent_dispatch(H);
        stream << sh(bq, bk, bconv, bbeta, bg, bstate, bout, T, H, hd, cfg.gdn_qk_heads,
                     2u * cfg.gdn_qk_n()).dispatch(qwen_dispatch_threads(*plan.value));
        auto dout = download(stream, bout, size_t(T) * H * hd);
        auto dstate = download(stream, bstate, state.size());
        stream << synchronize();
        check_close(dout, rout, 3e-5f, 3e-4f, "gdn recurrent prefill out");
        check_close(dstate, state_ref, 3e-5f, 3e-4f, "gdn recurrent prefill state");
    }
    {
        auto bstate2 = upload(dev, stream, state);
        auto sh = dev.compile(make_qwen_gdn_recurrent_kernel());
        auto plan = plan_qwen_gdn_recurrent_dispatch(H);
        const uint64_t qk_stride = uint64_t(cfg.gdn_qk_heads) * hd;
        luisa::vector<float> step_out;
        for (uint32_t t = 0; t < T; t++) {
            luisa::vector<float> qt(rq.begin() + t * qk_stride, rq.begin() + (t + 1) * qk_stride);
            luisa::vector<float> kt(rk.begin() + t * qk_stride, rk.begin() + (t + 1) * qk_stride);
            luisa::vector<float> ct(rconv.begin() + t * C, rconv.begin() + (t + 1) * C);
            luisa::vector<float> bt(rbeta.begin() + t * H, rbeta.begin() + (t + 1) * H);
            luisa::vector<float> gt(rg.begin() + t * H, rg.begin() + (t + 1) * H);
            auto bqt = upload(dev, stream, qt);
            auto bkt = upload(dev, stream, kt);
            auto bct = upload(dev, stream, ct);
            auto bbt = upload(dev, stream, bt);
            auto bgt = upload(dev, stream, gt);
            auto bot = dev.create_buffer<float>(H * hd);
            stream << sh(bqt, bkt, bct, bbt, bgt, bstate2, bot, 1u, H, hd, cfg.gdn_qk_heads,
                         2u * cfg.gdn_qk_n()).dispatch(qwen_dispatch_threads(*plan.value));
            auto ot = download(stream, bot, H * hd);
            step_out.insert(step_out.end(), ot.begin(), ot.end());
        }
        auto dstate2 = download(stream, bstate2, state.size());
        stream << synchronize();
        check_close(step_out, rout, 3e-5f, 3e-4f, "gdn decode step-by-step == prefill out");
        check_close(dstate2, state_ref, 3e-5f, 3e-4f, "gdn decode step-by-step == prefill state");
    }
    // Gated output norm (direct weight, norm BEFORE the SiLU(z) gate).
    luisa::vector<float> z(size_t(T) * H * hd), nw(hd);
    rng.fill(z, -2.f, 2.f);
    rng.fill(nw, 0.5f, 1.5f);
    luisa::vector<float> rnorm;
    check(ref_qwen_gdn_gated_norm(rout, z, nw, T, H, hd, cfg.norm_eps, rnorm).ok(),
          "gdn gated norm oracle");
    auto bz = upload(dev, stream, z);
    auto bnw = upload(dev, stream, nw);
    auto bnorm = dev.create_buffer<float>(rnorm.size());
    {
        auto sh = dev.compile(make_qwen_gdn_gated_norm_kernel());
        auto plan = plan_qwen_gdn_gated_norm_dispatch(T, H);
        stream << sh(bout, bz, bnw, bnorm, T, H, hd, cfg.norm_eps)
                      .dispatch(qwen_dispatch_threads(*plan.value));
        auto dnorm = download(stream, bnorm, rnorm.size());
        stream << synchronize();
        check_close(dnorm, rnorm, 1e-5f, 1e-4f, "gdn gated norm (direct weight, norm before gate)");
    }
}

void test_gdn_decode4_model_shape(Device &dev, Stream &stream, Rng &rng) {
    constexpr uint32_t QK = 16u, VH = 32u, HD = 128u;
    constexpr uint32_t C = 2u * QK * HD + VH * HD;
    luisa::vector<float> q(QK * HD), k(q.size()), conv(C), beta(VH), g(VH);
    luisa::vector<float> state(size_t(VH) * HD * HD);
    rng.fill(q, -1.f, 1.f);
    rng.fill(k, -1.f, 1.f);
    rng.fill(conv, -1.f, 1.f);
    rng.fill(beta, 0.1f, 0.9f);
    rng.fill(g, -0.2f, -0.001f);
    rng.fill(state, -0.1f, 0.1f);
    auto state_ref = state;
    luisa::vector<float> ref;
    check(ref_qwen_gdn_recurrent(q, k, conv, beta, g, state_ref, 1u, VH, HD, QK, ref).ok(),
          "gdn decode4 model-shape oracle");
    auto bq = upload(dev, stream, q);
    auto bk = upload(dev, stream, k);
    auto bc = upload(dev, stream, conv);
    auto bb = upload(dev, stream, beta);
    auto bg = upload(dev, stream, g);
    auto bs = upload(dev, stream, state);
    auto bo = dev.create_buffer<float>(VH * HD);
    auto sh = dev.compile(make_qwen35_gdn_decode4_kernel());
    auto plan = plan_qwen35_gdn_decode4_dispatch(VH);
    check(plan.ok(), "gdn decode4 model-shape plan");
    stream << sh(bq, bk, bc, bb, bg, bs, bo, 1u, VH, HD, QK, 2u * QK * HD)
                  .dispatch(qwen_dispatch_threads(*plan.value));
    auto out = download(stream, bo, ref.size());
    auto updated = download(stream, bs, state.size());
    check_close(out, ref, 5e-5f, 5e-4f, "gdn decode4 model-shape output");
    check_close(updated, state_ref, 5e-5f, 5e-4f, "gdn decode4 model-shape state");
}

// ---------------------------------------------------------------------------
// WY/UT chunk-64 vs the independent per-token FP64 oracle. Boundary lengths
// cover a partial chunk, exact chunks, and multi-chunk state carry.
// FUSED one-CTA-per-head GDN recurrence + gated output RMSNorm (opt_plan/03 section
// 5). One decode step is realized FOUR ways on the model geometry:
//   pair_one_cta : make_qwen35_gdn_recurrent_kernel(128) then
//                  make_qwen_gdn_gated_norm_kernel                 (2 launches)
//   fused        : make_qwen35_gdn_recurrent_norm_kernel()         (1 launch)
//   pair_decode4 : make_qwen35_gdn_decode4_kernel() then the gated norm kernel (2)
//   cpu64        : ref_qwen_gdn_recurrent -> ref_qwen_gdn_gated_norm (FP64 oracle)
// The state is carried across SEVERAL tokens (one launch per token, exactly like a
// decode loop) and then RESET, so the state update, the carry and the reset all see
// more than one step. Numerics:
// * the fused kernel runs the SAME recurrence as the one-CTA pair and re-associates
// only the head reduction (per-lane o*o -> 32-lane warp trees -> a fixed 4-term
// fold of the warp partials, one barrier), so the two agree to the FP32 last-bit
// envelope (measured <= 3.8e-06 absolute over six carried steps) -- not merely within a
// loose tolerance;
//   * the FP64 oracle is the numerical gate for both (2e-4 / 2e-3 over six steps);
//   * decode4 changes the key-axis reduction order, so its deviation from the fused
//     result is measured and reported (documented envelope 2e-3 / 2e-2), never
//     assumed;
//   * every realization must stay FINITE at every step (a decayed-to-zero state, an
//     all-zero gate and an extreme SiLU(z) argument included): rsqrt(mean(o^2)+eps)
//     is the only place a NaN could appear and it must not.
void test_gdn_recurrent_norm(Device &dev, Stream &stream, Rng &rng) {
    constexpr uint32_t QK = 16u, VH = 32u, HD = qwen35_gdn_model_head_dim;
    constexpr uint32_t QKN = QK * HD, VN = VH * HD, C = 2u * QKN + VN, T = 6u;
    const float eps = 1e-6f;
    luisa::vector<float> q(size_t(T) * QKN), k(q.size()), conv(size_t(T) * C);
    luisa::vector<float> beta(size_t(T) * VH), g(beta.size());
    luisa::vector<float> z(size_t(T) * VN), nw(HD);
    rng.fill(q, -1.f, 1.f);
    rng.fill(k, -1.f, 1.f);
    rng.fill(conv, -1.f, 1.f);
    rng.fill(beta, 0.05f, 0.9f);
    rng.fill(g, -0.2f, -0.001f);
    rng.fill(z, -6.f, 6.f);
    rng.fill(nw, 0.4f, 1.6f);
    // Match the engine's normalized q/k contract (q also carries 1/sqrt(hd)).
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t h = 0; h < QK; h++) {
            const size_t base = (size_t(t) * QK + h) * HD;
            double qs = 0.0, ks = 0.0;
            for (uint32_t d = 0; d < HD; d++) {
                qs += double(q[base + d]) * q[base + d];
                ks += double(k[base + d]) * k[base + d];
            }
            const float qi = float(1.0 / std::sqrt(qs + 1e-6) / std::sqrt(double(HD)));
            const float ki = float(1.0 / std::sqrt(ks + 1e-6));
            for (uint32_t d = 0; d < HD; d++) { q[base + d] *= qi; k[base + d] *= ki; }
        }
    }
    // Token 3 is the degenerate case: an all-zero conv row (o == 0 -> the RMS
    // reduction is exactly eps), an all-zero gate (silu(0) == 0) and a decay that
    // wipes the carried state; token 4 keeps the extreme SiLU arguments.
    for (uint32_t c = 0; c < C; c++) { conv[size_t(3) * C + c] = 0.0f; }
    for (uint32_t i = 0; i < VN; i++) { z[size_t(3) * VN + i] = 0.0f; }
    for (uint32_t h = 0; h < VH; h++) { g[size_t(3) * VH + h] = -30.0f; }
    luisa::vector<float> initial_state(size_t(VH) * HD * HD);
    rng.fill(initial_state, -0.1f, 0.1f);

    // ---- host oracle trajectory (FP64 accumulation, independent code path) ----
    luisa::vector<float> oracle_state = initial_state, oracle_out;
    luisa::vector<luisa::vector<float>> oracle_normed(T);
    luisa::vector<luisa::vector<float>> oracle_states(T);
    for (uint32_t t = 0; t < T; t++) {
        luisa::vector<float> qt(q.begin() + size_t(t) * QKN, q.begin() + size_t(t + 1) * QKN);
        luisa::vector<float> kt(k.begin() + size_t(t) * QKN, k.begin() + size_t(t + 1) * QKN);
        luisa::vector<float> ct(conv.begin() + size_t(t) * C, conv.begin() + size_t(t + 1) * C);
        luisa::vector<float> bt(beta.begin() + size_t(t) * VH, beta.begin() + size_t(t + 1) * VH);
        luisa::vector<float> gt(g.begin() + size_t(t) * VH, g.begin() + size_t(t + 1) * VH);
        luisa::vector<float> zt(z.begin() + size_t(t) * VN, z.begin() + size_t(t + 1) * VN);
        luisa::vector<float> raw;
        check(ref_qwen_gdn_recurrent(qt, kt, ct, bt, gt, oracle_state, 1u, VH, HD, QK, raw).ok(),
              "recurrent+norm oracle: recurrence");
        check(ref_qwen_gdn_gated_norm(raw, zt, nw, 1u, VH, HD, eps, oracle_normed[t]).ok(),
              "recurrent+norm oracle: gated norm");
        oracle_states[t] = oracle_state;
    }

    // ---- device buffers -------------------------------------------------------
    auto bq = upload(dev, stream, q);
    auto bk = upload(dev, stream, k);
    auto bc = upload(dev, stream, conv);
    auto bb = upload(dev, stream, beta);
    auto bg = upload(dev, stream, g);
    auto bz = upload(dev, stream, z);
    auto bnw = upload(dev, stream, nw);
    auto bs_pair = upload(dev, stream, initial_state);
    auto bs_fused = upload(dev, stream, initial_state);
    auto bs_d4 = upload(dev, stream, initial_state);
    auto bout_pair = dev.create_buffer<float>(VN);
    auto bout_fused = dev.create_buffer<float>(VN);
    auto bout_d4 = dev.create_buffer<float>(VN);
    auto braw = dev.create_buffer<float>(VN);

    auto sh_rec = dev.compile(make_qwen35_gdn_recurrent_kernel(HD));
    auto sh_norm = dev.compile(make_qwen_gdn_gated_norm_kernel());
    auto sh_fused = dev.compile(make_qwen35_gdn_recurrent_norm_kernel());
    auto sh_d4 = dev.compile(make_qwen35_gdn_decode4_kernel());
    auto plan_rec = plan_qwen_gdn_recurrent_dispatch(VH);
    auto plan_norm = plan_qwen_gdn_gated_norm_dispatch(1u, VH);
    auto plan_fused = plan_qwen35_gdn_recurrent_norm_dispatch(VH);
    auto plan_d4 = plan_qwen35_gdn_decode4_dispatch(VH);
    check(plan_rec.ok() && plan_norm.ok() && plan_fused.ok() && plan_d4.ok(),
          "recurrent+norm: dispatch plans");
    if (!plan_rec.ok() || !plan_norm.ok() || !plan_fused.ok() || !plan_d4.ok()) { return; }
    // One block per V head: the whole head lives in one block, which is what makes
    // the in-block norm reduction legal.
    check(plan_fused.value->work_items == VH, "fused recurrent+norm: one block per V head");
    const auto v_offset = 2u * QKN;

    // ---- fail-closed host contract -------------------------------------------
    check(!validate_qwen35_gdn_recurrent_norm_inputs(2u, VH, QK, HD).ok(),
          "fused recurrent+norm refuses T > 1");
    check(!validate_qwen35_gdn_recurrent_norm_inputs(1u, VH, QK, 64u).ok(),
          "fused recurrent+norm refuses a non-model head_dim");
    check(!validate_qwen35_gdn_recurrent_norm_inputs(1u, VH, QK, 256u).ok(),
          "fused recurrent+norm refuses a wider head_dim");
    check(!validate_qwen35_gdn_recurrent_norm_inputs(1u, 0u, QK, HD).ok(),
          "fused recurrent+norm refuses an empty head count");
    check(!validate_qwen35_gdn_recurrent_norm_inputs(1u, 8u, 16u, HD).ok(),
          "fused recurrent+norm refuses v_heads % qk_heads != 0");
    check(validate_qwen35_gdn_recurrent_norm_inputs(1u, VH, QK, HD).ok(),
          "fused recurrent+norm accepts the model geometry");
    check(qwen_gdn_recurrent_norm_shared_bytes(HD) == 3u * HD * sizeof(float),
          "fused recurrent+norm reports its 3-row shared budget");

    // Token-0 views used by the guarded-no-op and the reset arms below.
    auto q0 = bq.view(0u, QKN);
    auto k0 = bk.view(0u, QKN);
    auto c0 = bc.view(0u, C);
    auto b0 = bb.view(0u, VH);
    auto g0 = bg.view(0u, VH);
    auto z0 = bz.view(0u, VN);

    // A non-model head_dim is a GUARDED NO-OP, never a misread: the output buffer
    // must come back untouched.
    {
        luisa::vector<float> sentinel(VN, -7.5f);
        auto bsent = upload(dev, stream, sentinel);
        auto bsent_state = upload(dev, stream, initial_state);
        stream << sh_fused(q0, k0, c0, b0, g0, bsent_state, z0, bnw, bsent,
                           1u, VH, 64u, QK, v_offset, eps)
                      .dispatch(qwen_dispatch_threads(*plan_fused.value));
        check_exact(download(stream, bsent, VN), sentinel,
                    "fused recurrent+norm: non-model head_dim writes nothing");
        stream << sh_fused(q0, k0, c0, b0, g0, bsent_state, z0, bnw, bsent,
                           2u, VH, HD, QK, v_offset, eps)
                      .dispatch(qwen_dispatch_threads(*plan_fused.value));
        check_exact(download(stream, bsent, VN), sentinel,
                    "fused recurrent+norm: tokens != 1 writes nothing");
    }

    // ---- six decode steps with the state carried, then a reset ---------------
    double max_fused_vs_pair = 0.0, max_fused_vs_d4 = 0.0;
    bool all_finite = true;
    for (uint32_t t = 0; t < T; t++) {
        // The kernels are single-token realizations, so each step binds THIS token's
        // slice as a view -- exactly what the engine does when the projections of a
        // decode step write row 0 of the per-step buffers.
        auto qv = bq.view(size_t(t) * QKN, QKN);
        auto kv = bk.view(size_t(t) * QKN, QKN);
        auto cv = bc.view(size_t(t) * C, C);
        auto bv = bb.view(size_t(t) * VH, VH);
        auto gv = bg.view(size_t(t) * VH, VH);
        auto zv = bz.view(size_t(t) * VN, VN);
        stream << sh_rec(qv, kv, cv, bv, gv, bs_pair, braw, 1u, VH, HD, QK, v_offset)
                      .dispatch(qwen_dispatch_threads(*plan_rec.value))
               << sh_norm(braw, zv, bnw, bout_pair, 1u, VH, HD, eps)
                      .dispatch(qwen_dispatch_threads(*plan_norm.value))
               << sh_fused(qv, kv, cv, bv, gv, bs_fused, zv, bnw, bout_fused,
                           1u, VH, HD, QK, v_offset, eps)
                      .dispatch(qwen_dispatch_threads(*plan_fused.value))
               << sh_d4(qv, kv, cv, bv, gv, bs_d4, braw, 1u, VH, HD, QK, v_offset)
                      .dispatch(qwen_dispatch_threads(*plan_d4.value))
               << sh_norm(braw, zv, bnw, bout_d4, 1u, VH, HD, eps)
                      .dispatch(qwen_dispatch_threads(*plan_norm.value));
        auto out_pair = download(stream, bout_pair, VN);
        auto out_fused = download(stream, bout_fused, VN);
        auto out_d4 = download(stream, bout_d4, VN);
        auto st_pair = download(stream, bs_pair, initial_state.size());
        auto st_fused = download(stream, bs_fused, initial_state.size());
        auto st_d4 = download(stream, bs_d4, initial_state.size());
        char name[128];
        std::snprintf(name, sizeof(name), "fused recurrent+norm T=%u output == one-CTA pair", t);
        check_close(out_fused, out_pair, 1e-6f, 1e-6f, name);
        std::snprintf(name, sizeof(name), "fused recurrent+norm T=%u state == one-CTA pair", t);
        check_close(st_fused, st_pair, 1e-6f, 1e-6f, name);
        std::snprintf(name, sizeof(name), "fused recurrent+norm T=%u output == FP64 oracle", t);
        check_close(out_fused, oracle_normed[t], 2e-4f, 2e-3f, name);
        std::snprintf(name, sizeof(name), "fused recurrent+norm T=%u state == FP64 oracle", t);
        check_close(st_fused, oracle_states[t], 2e-4f, 2e-3f, name);
        std::snprintf(name, sizeof(name), "decode4+norm T=%u output == FP64 oracle", t);
        check_close(out_d4, oracle_normed[t], 5e-5f, 5e-4f, name);
        for (uint32_t i = 0; i < VN; i++) {
            if (!std::isfinite(out_fused[i])) { all_finite = false; }
            max_fused_vs_pair = std::max(max_fused_vs_pair,
                                         std::fabs(double(out_fused[i]) - double(out_pair[i])));
            max_fused_vs_d4 = std::max(max_fused_vs_d4,
                                       std::fabs(double(out_fused[i]) - double(out_d4[i])));
        }
        for (float v : st_fused) {
            if (!std::isfinite(v)) { all_finite = false; }
        }
    }
    check(all_finite, "fused recurrent+norm: every output and state element is finite");
    // decode4 splits the key axis across four blocks, so its FP32 reduction order
    // differs from both the one-CTA recurrence and the fused kernel. The measured
    // deviation is printed and bounded here; it is the documented tolerance the
    // report quotes, not a silent assumption.
    std::printf("fused recurrent+norm: max|fused - pair| = %.3e, max|fused - decode4+norm| = %.3e\n",
                max_fused_vs_pair, max_fused_vs_d4);
    // Envelope, not a bit claim. The fused kernel re-associates ONLY the head
    // reduction (per-lane o*o -> 32-lane warp trees -> a fixed 4-term fold of the
    // warp partials), so its sum differs from the pair's four strided per-lane sums
    // in the last bits, and the HLSL/SPIR-V compilers then contract the
    // `row * inv * w * silu(z)` chain differently from the buffer-operand form.
    // Measured over six carried steps: cuda 2.9e-06, dx 3.8e-06, vk 1.9e-06 absolute,
    // i.e. 1-4 ulp on values of order 1-8. The 1e-5 rail below is a guard rail against a
    // real regression; the gate is the elementwise 1e-6 rel / 1e-6 abs check above.
    // rail is 1e-5 while the real gate is the elementwise 1e-6 rel / 1e-6 abs check above.
    check(max_fused_vs_pair <= 1e-5, "fused == one-CTA pair within the FP32 last-bit envelope");
    check(max_fused_vs_d4 <= 2e-3, "fused vs decode4+norm within the documented envelope");

    // Reset: re-seeding the initial state must reproduce the first step exactly.
    stream << bs_fused.copy_from(luisa::span<const float>{initial_state}) << synchronize();
    stream << sh_fused(q0, k0, c0, b0, g0, bs_fused, z0, bnw, bout_fused,
                       1u, VH, HD, QK, v_offset, eps)
                  .dispatch(qwen_dispatch_threads(*plan_fused.value));
    auto after_reset = download(stream, bout_fused, VN);
    luisa::vector<float> first_step;
    {
        auto bs_check = upload(dev, stream, initial_state);
        stream << sh_fused(q0, k0, c0, b0, g0, bs_check, z0, bnw, bout_fused,
                           1u, VH, HD, QK, v_offset, eps)
                      .dispatch(qwen_dispatch_threads(*plan_fused.value));
        first_step = download(stream, bout_fused, VN);
    }
    check_exact(after_reset, first_step, "fused recurrent+norm: reset reproduces step 0 exactly");
    check_close(after_reset, oracle_normed[0], 2e-4f, 2e-3f,
                "fused recurrent+norm: reset output == FP64 oracle step 0");
}

// ---------------------------------------------------------------------------
void test_gdn_chunk_boundaries(Device &dev, Stream &stream, Rng &rng) {
    constexpr uint32_t QK = 2u, H = 4u, HD = 32u, MAX_T = 129u;
    constexpr uint32_t QKN = QK * HD, VN = H * HD, C = 2u * QKN + VN;
    luisa::vector<float> q(size_t(MAX_T) * QKN), k(q.size()), conv(size_t(MAX_T) * C);
    rng.fill(q, -1.0f, 1.0f);
    rng.fill(k, -1.0f, 1.0f);
    rng.fill(conv, -0.5f, 0.5f);
    // Match the engine's normalized q/k contract, including q's 1/sqrt(HD).
    for (uint32_t t = 0; t < MAX_T; t++) {
        for (uint32_t h = 0; h < QK; h++) {
            const size_t base = (size_t(t) * QK + h) * HD;
            double qs = 0.0, ks = 0.0;
            for (uint32_t d = 0; d < HD; d++) {
                qs += double(q[base + d]) * q[base + d];
                ks += double(k[base + d]) * k[base + d];
            }
            const float qi = float(1.0 / std::sqrt(qs + 1e-6) / std::sqrt(double(HD)));
            const float ki = float(1.0 / std::sqrt(ks + 1e-6));
            for (uint32_t d = 0; d < HD; d++) { q[base + d] *= qi; k[base + d] *= ki; }
        }
    }
    luisa::vector<float> beta(size_t(MAX_T) * H), g(beta.size());
    rng.fill(beta, 0.05f, 0.4f);
    rng.fill(g, -0.08f, -0.001f);
    luisa::vector<float> initial_state(size_t(H) * HD * HD);
    rng.fill(initial_state, -0.05f, 0.05f);
    auto shader = dev.compile(make_qwen_gdn_chunk_kernel());
    auto plan = plan_qwen_gdn_chunk_dispatch(H);
    check(plan.ok(), "chunk-64 dispatch plan");
    if (!plan.ok()) { return; }
    for (uint32_t tokens : {1u, 63u, 64u, 65u, 128u, 129u}) {
        const size_t qk_count = size_t(tokens) * QKN;
        const size_t conv_count = size_t(tokens) * C;
        const size_t gate_count = size_t(tokens) * H;
        const size_t out_count = gate_count * HD;
        luisa::vector<float> qt(q.begin(), q.begin() + qk_count);
        luisa::vector<float> kt(k.begin(), k.begin() + qk_count);
        luisa::vector<float> ct(conv.begin(), conv.begin() + conv_count);
        luisa::vector<float> bt(beta.begin(), beta.begin() + gate_count);
        luisa::vector<float> gt(g.begin(), g.begin() + gate_count);
        if (tokens == 64u) { std::fill(gt.begin(), gt.end(), 0.0f); }
        if (tokens == 129u) { std::fill(gt.begin(), gt.end(), -20.0f); }
        luisa::vector<float> reference_state = initial_state, reference_out;
        check(ref_qwen_gdn_recurrent(qt, kt, ct, bt, gt, reference_state, tokens,
                                     H, HD, QK, reference_out).ok(),
              "chunk-64 recurrent oracle");
        auto bq = upload(dev, stream, qt);
        auto bk = upload(dev, stream, kt);
        auto bc = upload(dev, stream, ct);
        auto bb = upload(dev, stream, bt);
        auto bg = upload(dev, stream, gt);
        auto bs = upload(dev, stream, initial_state);
        auto bw = dev.create_buffer<float>(out_count);
        auto bvn = dev.create_buffer<float>(out_count);
        auto bo = dev.create_buffer<float>(out_count);
        stream << shader(bq, bk, bc, bb, bg, bs, bw, bvn, bo, tokens, H, HD, QK, 2u * QKN)
                      .dispatch(qwen_dispatch_threads(*plan.value));
        auto actual_out = download(stream, bo, out_count);
        auto actual_state = download(stream, bs, initial_state.size());
        stream << synchronize();
        char name[96];
        std::snprintf(name, sizeof(name), "WY/UT T=%u output == recurrent", tokens);
        check_close(actual_out, reference_out, 2e-4f, 2e-3f, name);
        std::snprintf(name, sizeof(name), "WY/UT T=%u state == recurrent", tokens);
        check_close(actual_state, reference_state, 2e-4f, 2e-3f, name);
    }
}

// ---------------------------------------------------------------------------
// Elementwise utilities: residual add / zero fill / copy f32+u32+i32 on a
// ragged count (1003 = 3 full 256-blocks + 235 tail).
// ---------------------------------------------------------------------------
void test_utilities(Device &dev, Stream &stream, Rng &rng) {
    const uint32_t n = 1003u;
    luisa::vector<float> a(n), b(n);
    rng.fill(a, -1.f, 1.f);
    rng.fill(b, -1.f, 1.f);
    auto ba = upload(dev, stream, a);
    auto bb = upload(dev, stream, b);
    auto bo = dev.create_buffer<float>(n);
    auto plan = plan_qwen_threads(n);
    {
        auto sh = dev.compile(make_qwen_residual_add_kernel());
        stream << sh(ba, bb, bo, n).dispatch(qwen_dispatch_threads(*plan.value));
        auto o = download(stream, bo, n);
        stream << synchronize();
        luisa::vector<float> want(n);
        for (uint32_t i = 0; i < n; i++) { want[i] = a[i] + b[i]; }
        check_exact(o, want, "residual add ragged");
    }
    {
        auto sh = dev.compile(make_qwen_zero_fill_kernel());
        stream << sh(bo, n).dispatch(qwen_dispatch_threads(*plan.value));
        auto z = download(stream, bo, n);
        stream << synchronize();
        luisa::vector<float> want(n, 0.f);
        check_exact(z, want, "zero fill ragged");
    }
    {
        auto sh = dev.compile(make_qwen_copy_kernel());
        stream << sh(ba, bo, n).dispatch(qwen_dispatch_threads(*plan.value));
        auto o = download(stream, bo, n);
        stream << synchronize();
        check_exact(o, a, "copy f32 ragged");
    }
    {
        luisa::vector<uint32_t> u(n);
        for (uint32_t i = 0; i < n; i++) { u[i] = rng.next(); }
        auto bu = upload(dev, stream, u);
        auto buo = dev.create_buffer<uint32_t>(n);
        auto sh = dev.compile(make_qwen_copy_u_kernel());
        stream << sh(bu, buo, n).dispatch(qwen_dispatch_threads(*plan.value));
        auto o = download(stream, buo, n);
        stream << synchronize();
        bool ok = o.size() == u.size();
        for (uint32_t i = 0; i < n && ok; i++) { ok = o[i] == u[i]; }
        check(ok, "copy u32 ragged");
    }
    {
        luisa::vector<int32_t> v(n);
        for (uint32_t i = 0; i < n; i++) { v[i] = int32_t(rng.next()) - 1000000; }
        auto bv = upload(dev, stream, v);
        auto bvo = dev.create_buffer<int32_t>(n);
        auto sh = dev.compile(make_qwen_copy_i_kernel());
        stream << sh(bv, bvo, n).dispatch(qwen_dispatch_threads(*plan.value));
        auto o = download(stream, bvo, n);
        stream << synchronize();
        bool ok = o.size() == v.size();
        for (uint32_t i = 0; i < n && ok; i++) { ok = o[i] == v[i]; }
        check(ok, "copy i32 ragged");
    }
}

// ---------------------------------------------------------------------------
// EXTERNAL-CONTRACT DEVICE FIXTURES: device kernels vs INDEPENDENT
// hand-computed GGUF expectations (not the shared oracles).
// ---------------------------------------------------------------------------
void test_fixtures(Device &dev, Stream &stream, Buffer<float> &dummy_f) {
    // (a) rmsnorm pre-added direct weight: x=[3,4,0,-1], w=[2,0.5,1,1.5].
    {
        const luisa::vector<float> x{3.f, 4.f, 0.f, -1.f};
        const luisa::vector<float> w{2.f, 0.5f, 1.f, 1.5f};
        auto bx = upload(dev, stream, x);
        auto bw = upload(dev, stream, w);
        auto by = dev.create_buffer<float>(4);
        auto sh = dev.compile(make_qwen_rmsnorm_kernel(QwenNormEpilogue::plain));
        auto plan = plan_qwen_rmsnorm_dispatch(1, 4);
        stream << sh(bx, bw, dummy_f, by, dummy_f, 1u, 4u, 1e-6f).dispatch(qwen_dispatch_threads(*plan.value));
        auto y = download(stream, by, 4);
        stream << synchronize();
        const double inv = 1.0 / std::sqrt(26.0 / 4.0 + 1e-6);
        const luisa::vector<float> expect{float(3 * inv * 2), float(4 * inv * 0.5), 0.f, float(-1 * inv * 1.5)};
        check_close(y, expect, 1e-5f, 1e-6f, "fixture device rmsnorm pre-added");
    }
    // (b) gates with negative pre-folded ssm_a: g = ssm_a * softplus(a+dt).
    {
        const luisa::vector<float> a{1.f, 30.f}, b{0.f, 2.f}, ssm_a{-2.f, -0.5f}, dt{0.25f, -1.f};
        auto ba = upload(dev, stream, a);
        auto bb = upload(dev, stream, b);
        auto bsa = upload(dev, stream, ssm_a);
        auto bdt = upload(dev, stream, dt);
        auto bbeta = dev.create_buffer<float>(2);
        auto bg = dev.create_buffer<float>(2);
        auto sh = dev.compile(make_qwen_gdn_gates_kernel());
        auto plan = plan_qwen_gdn_gates_dispatch(1, 2);
        stream << sh(ba, bb, bsa, bdt, bbeta, bg, 1u, 2u, 20.f).dispatch(qwen_dispatch_threads(*plan.value));
        auto g = download(stream, bg, 2);
        auto beta = download(stream, bbeta, 2);
        stream << synchronize();
        const luisa::vector<float> gw{float(-2.0 * std::log1p(std::exp(1.25))), -0.5f * 29.f};
        const luisa::vector<float> bw{0.5f, float(1.0 / (1.0 + std::exp(-2.0)))};
        check_close(g, gw, 1e-5f, 1e-6f, "fixture device gates negative ssm_a g");
        check_close(beta, bw, 1e-6f, 1e-6f, "fixture device gates negative ssm_a beta");
    }
    // (c) channel-major impulse conv: impulse at position 0 is seen by
    //     output t through tap K-1-t: out[t,0] = silu(w[K-1-t]).
    {
        const uint32_t C = 2u, K = 4u, T = 4u;
        luisa::vector<float> w(C * K), src(T * C, 0.f);
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t j = 0; j < K; j++) { w[c * K + j] = float(c * 10 + j + 1); }
        }
        src[0u] = 1.f;
        auto bw = upload(dev, stream, w);
        auto bsrc = upload(dev, stream, src);
        auto bring = dev.create_buffer<float>(C * (K - 1u));
        auto zsh = dev.compile(make_qwen_zero_fill_kernel());
        auto zplan = plan_qwen_threads(C * (K - 1u));
        stream << zsh(bring, C * (K - 1u)).dispatch(qwen_dispatch_threads(*zplan.value));
        auto bout = dev.create_buffer<float>(T * C);
        auto sh = dev.compile(make_qwen_gdn_conv_silu_kernel());
        auto plan = plan_qwen_gdn_conv_dispatch(C);
        stream << sh(bsrc, bw, bring, bout, T, C, K, 0u).dispatch(qwen_dispatch_threads(*plan.value));
        auto out = download(stream, bout, T * C);
        stream << synchronize();
        luisa::vector<float> expect(T * C);
        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t c = 0; c < C; c++) {
                const double acc = (c == 0u && t < K) ? double(w[K - 1u - t]) : 0.0;
                expect[t * C + c] = float(acc / (1.0 + std::exp(-acc)));
            }
        }
        check_close(out, expect, 1e-5f, 1e-6f, "fixture device conv channel-major impulse");
    }
    // (d) tiled GGUF V-head pairing: V head h uses QK head h % qk_heads.
    {
        const uint32_t QK = 2u, VH = 4u, HD = 32u;
        luisa::vector<float> q(QK * HD, 0.f), k(QK * HD, 0.f);
        q[0] = 1.f;
        q[HD + 1] = 1.f;
        k[0] = 1.f;
        k[HD + 1] = 1.f;
        const uint32_t C = 2u * QK * HD + VH * HD;
        luisa::vector<float> conv(C, 0.f);
        for (uint32_t vh = 0; vh < VH; vh++) { conv[2u * QK * HD + vh * HD] = float(vh + 1); }
        luisa::vector<float> beta(VH, 1.f), g(VH, 0.f), state(VH * HD * HD, 0.f);
        auto bq = upload(dev, stream, q);
        auto bk = upload(dev, stream, k);
        auto bconv = upload(dev, stream, conv);
        auto bbeta = upload(dev, stream, beta);
        auto bg = upload(dev, stream, g);
        auto bstate = upload(dev, stream, state);
        auto bout = dev.create_buffer<float>(VH * HD);
        auto sh = dev.compile(make_qwen_gdn_recurrent_kernel());
        auto plan = plan_qwen_gdn_recurrent_dispatch(VH);
        stream << sh(bq, bk, bconv, bbeta, bg, bstate, bout, 1u, VH, HD, QK, 2u * QK * HD)
                      .dispatch(qwen_dispatch_threads(*plan.value));
        auto out = download(stream, bout, VH * HD);
        stream << synchronize();
        luisa::vector<float> expect(VH * HD, 0.f);
        for (uint32_t vh = 0; vh < VH; vh++) { expect[vh * HD] = float(vh + 1); }
        check_close(out, expect, 1e-5f, 1e-6f, "fixture device gdn tiled pairing");
    }
    // (e) NeoX rotate-half partial RoPE: hd=8, rope_dim=4, theta=1e4, pos=1.
    {
        const luisa::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
        const luisa::vector<int32_t> pos{1};
        auto bx = upload(dev, stream, x);
        auto bpos = upload(dev, stream, pos);
        auto bout = dev.create_buffer<float>(8);
        auto sh = dev.compile(make_qwen_rope_partial_neox_kernel());
        auto plan = plan_qwen_rope_dispatch(1, 1);
        stream << sh(bx, bout, bpos, 1u, 1u, 8u, 4u, 10000.f, 1u).dispatch(qwen_dispatch_threads(*plan.value));
        auto out = download(stream, bout, 8);
        stream << synchronize();
        const double f1 = std::pow(10000.0, -0.5);
        const double c0 = std::cos(1.0), s0 = std::sin(1.0);
        const double c1 = std::cos(f1), s1 = std::sin(f1);
        const luisa::vector<float> expect{float(c0 - 3 * s0), float(2 * c1 - 4 * s1),
                                          float(s0 + 3 * c0), float(2 * s1 + 4 * c1), 5, 6, 7, 8};
        check_close(out, expect, 1e-4f, 1e-6f, "fixture device rope NeoX");
    }
    // (f) near-zero L2 rows: eps INSIDE the rsqrt keeps tiny rows finite.
    {
        const uint32_t QK = 1u, HD = 32u, VN = 32u;
        const uint32_t C = 2u * QK * HD + VN;
        luisa::vector<float> conv(C, 0.f);
        conv[0] = 1e-4f;
        auto bconv = upload(dev, stream, conv);
        auto bq = dev.create_buffer<float>(HD);
        auto bk = dev.create_buffer<float>(HD);
        auto sh = dev.compile(make_qwen_gdn_qk_l2norm_kernel());
        auto plan = plan_qwen_gdn_qk_norm_dispatch(1, QK);
        const float qs = 1.f / std::sqrt(32.f);
        stream << sh(bconv, bq, bk, 1u, QK, HD, VN, 1e-6f, qs, 1u).dispatch(qwen_dispatch_threads(*plan.value));
        auto q = download(stream, bq, HD);
        auto k = download(stream, bk, HD);
        stream << synchronize();
        luisa::vector<float> qw(HD, 0.f), kw(HD, 0.f);
        qw[0] = float(1e-4 / std::sqrt(1e-8 + 1e-6) * qs);
        check_close(q, qw, 1e-8f, 1e-6f, "fixture device l2norm near-zero q");
        check_exact(k, kw, "fixture device l2norm near-zero k");
    }
}

  // ---------------------------------------------------------------------------
  // Tiled prefill attention, host contract (opt_plan/02 section 2 / M-b).
  // Device-free: tiling selection from the reported limits, launch planning and
  // the causal/capacity input contract that the engine relies on.
  // ---------------------------------------------------------------------------
  void test_attn_prefill_tiled_plan() {
      QwenAttnPrefillTiledConfig cfg{};
      check(cfg.head_dim == 256u && cfg.q_heads == 16u && cfg.kv_heads == 4u &&
                cfg.q_tile == 2u && cfg.k_tile == 2u,
            "tiled attn default config is the model geometry with the measured tiling");
      check(qwen35_attn_prefill_tiled_shared_bytes(cfg) == 2u * 2u * 256u * 4u,
            "tiled attn shared bytes = 2 K/V tiles of 2 KiB");
      check(validate_qwen35_attn_prefill_tiled_config(cfg).ok(), "tiled attn config valid");
      // ceil(tokens/q_tile) x kv_heads blocks of 128 threads, warp 32 pinned.
      for (uint32_t tokens : {1u, 3u, 4u, 5u, 63u, 64u, 65u}) {
          auto plan = plan_qwen35_attn_prefill_tiled_dispatch(tokens, cfg);
          const uint64_t tiles = (tokens + cfg.q_tile - 1u) / cfg.q_tile;
          const uint64_t blocks = tiles * cfg.kv_heads;
          check(plan.ok() && plan.value->work_items == blocks && plan.value->block == 128u &&
                    plan.value->warp_size == 32u && qwen_dispatch_threads(*plan.value) == blocks * 128u,
                "tiled attn plan: one 128-thread block per (query tile, kv head)");
      }
      auto empty_plan = plan_qwen35_attn_prefill_tiled_dispatch(0u, cfg);
      check(empty_plan.ok() && empty_plan.value->empty(), "tiled attn plan empty for zero tokens");
      // Tiling selection fails closed instead of launching a shader that cannot run.
      QwenAttnPrefillTiledConfig chosen{};
      check(!select_qwen35_attn_prefill_tiled_config(64u, 0u, chosen).ok(),
            "tiled attn selection rejects warp size 64");
      check(!select_qwen35_attn_prefill_tiled_config(32u, 2048u, chosen).ok(),
            "tiled attn selection rejects a 2 KiB shared budget");
      check(select_qwen35_attn_prefill_tiled_config(32u, 0u, chosen).ok() &&
                chosen.q_tile == 2u && chosen.k_tile == 2u,
            "tiled attn selection: unreported limit -> measured 4 KiB tile");
      check(select_qwen35_attn_prefill_tiled_config(32u, 32768u, chosen).ok() &&
                chosen.q_tile == 2u && chosen.k_tile == 2u,
            "tiled attn selection: 32 KiB limit keeps the measured tile");
      check(select_qwen35_attn_prefill_tiled_config(32u, 49152u, chosen).ok() &&
                chosen.q_tile == 2u && chosen.k_tile == 2u,
            "tiled attn selection: 48 KiB limit still uses the measured tile");
      // Geometry the kernel cannot pin must be rejected, never approximated.
      auto bad = cfg;
      bad.head_dim = 48u;
      check(!validate_qwen35_attn_prefill_tiled_config(bad).ok(), "tiled attn rejects head_dim 48");
      bad = cfg;
      bad.kv_heads = 8u;
      check(!validate_qwen35_attn_prefill_tiled_config(bad).ok(), "tiled attn rejects 2 q heads per kv head");
      bad = cfg;
      bad.q_tile = 9u;
      check(!validate_qwen35_attn_prefill_tiled_config(bad).ok(), "tiled attn rejects q_tile 9 (register bound)");
      bad = cfg;
      bad.k_tile = 0u;
      check(!validate_qwen35_attn_prefill_tiled_config(bad).ok(), "tiled attn rejects k_tile 0");
      // The input contract is the reference kernel's contract.
      check(validate_qwen35_attn_prefill_tiled_inputs(4u, 3u, 7u, 8u, cfg).ok(), "tiled attn inputs accepted");
      check(!validate_qwen35_attn_prefill_tiled_inputs(4u, 3u, 7u, 6u, cfg).ok(),
            "tiled attn rejects kv_len > capacity");
      check(!validate_qwen35_attn_prefill_tiled_inputs(5u, 3u, 7u, 8u, cfg).ok(),
            "tiled attn rejects a causal window past kv_len");
      check(!validate_qwen35_attn_prefill_tiled_inputs(1u, 9u, 8u, 8u, cfg).ok(),
            "tiled attn rejects pos0 > kv_len");
      check(!validate_qwen35_attn_prefill_tiled_inputs(1u, 0u, 0u, 0u, cfg).ok(),
            "tiled attn rejects a zero-capacity cache");
  }

  // ---------------------------------------------------------------------------
  // Tiled prefill attention, device sweep (opt_plan/02 section 5 item 1).
  //
  // Three-way comparison: CPU oracle (double accumulation, independent code
  // path) vs the untiled reference kernel (one thread per (token, q_head),
  // runtime head_dim local accumulator) vs the tiled kernel, over
  //   S = 1,2,3,4,31,32,33,63,64,65,127,128,129,1024  x  pos0 = 0,63,64,512
  // with ragged tails (capacity is never a multiple of the tile extents) and the
  // two extreme cache boundaries (capacity == pos0+S, and a kv_len beyond the
  // cache that only the kernel's defensive clamp keeps in bounds). Both the
  // device-selected tiling and a deliberately narrow q_tile=2/k_tile=4 tiling run
  // the same sweep, so tile-tail masking is covered for every prompt length.
  // Tolerance: the tiled kernel reassociates the softmax denominator and the P.V
  // accumulation (per-tile rescale, warp-tree dot product), so it is compared at
  // the documented FP32 reassociation envelope (atol 2e-5, rtol 2e-4) used by the
  // rest of this suite -- not bit equality.
  // ---------------------------------------------------------------------------
  using QwenTiledAttnShader = compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>,
                                              compute::Buffer<float>, compute::Buffer<float>,
                                              compute::Buffer<float>, compute::Buffer<float>,
                                              compute::Buffer<float>, uint32_t, float, uint32_t,
                                              uint32_t, uint32_t>;

  void test_attn_prefill_tiled(Device &dev, Stream &stream, Rng &rng) {
    // The f32/bf16 attention arms take the fp8 scale arguments but never read them
    // (the scale load folds away), so one 1-element buffer serves every unscaled
    // dispatch of this test.
    auto bscale_dummy = dev.create_buffer<float>(1);
      constexpr uint32_t QH = 16u, KVH = 4u, HD = 256u;
      const auto scale = 1.0f / std::sqrt(static_cast<float>(HD));
      QwenAttnPrefillTiledConfig selected{};
      const auto warp_size = dev.compute_warp_size();
      const auto shared_bytes = static_cast<uint32_t>(dev.compute_max_shared_memory_size());
      const auto selected_status = select_qwen35_attn_prefill_tiled_config(warp_size, shared_bytes, selected);
      check(selected_status.ok(), "tiled attn device tiling selection",
            luisa::string{"warp "} + std::to_string(warp_size).c_str() + " shared " +
                std::to_string(shared_bytes).c_str() + " B -> q_tile " +
                std::to_string(selected.q_tile).c_str() + " k_tile " +
                std::to_string(selected.k_tile).c_str());
      // A second, deliberately ragged tiling: odd q_tile/k_tile exercise the
      // tile-tail masking beyond the (even) selected tiling.
      QwenAttnPrefillTiledConfig narrow{};
      narrow.q_tile = 3u;
      narrow.k_tile = 5u;
      check(validate_qwen35_attn_prefill_tiled_config(narrow).ok(), "tiled attn ragged tiling valid");
      auto shader_ref = dev.compile(make_qwen_attn_prefill_kernel());
      auto shader_selected = dev.compile(make_qwen35_attn_prefill_tiled_kernel(selected));
      auto shader_narrow = dev.compile(make_qwen35_attn_prefill_tiled_kernel(narrow));
      check(shader_ref.compile_ok() && shader_selected.compile_ok() && shader_narrow.compile_ok(),
            "tiled attn shaders compile");

      const auto run_tiled = [&](QwenTiledAttnShader &shader, const QwenAttnPrefillTiledConfig &cfg,
                                 Buffer<float> &bq, Buffer<float> &bg, Buffer<float> &bk,
                                 Buffer<float> &bv, Buffer<float> &bo, uint32_t tokens,
                                 uint32_t pos0, uint32_t kv_len, uint32_t capacity) {
          auto plan = plan_qwen35_attn_prefill_tiled_dispatch(tokens, cfg);
          if (!plan.ok()) { return; }
          if (plan.value->empty()) { return; }
          stream << shader(bq, bg, bk, bv, bscale_dummy.view(), bscale_dummy.view(), bo, tokens, scale, pos0, kv_len, capacity)
                        .dispatch(qwen_dispatch_threads(*plan.value));
      };
      const auto run_ref = [&](Buffer<float> &bq, Buffer<float> &bg, Buffer<float> &bk,
                          Buffer<float> &bv, Buffer<float> &bo, uint32_t tokens,
                          uint32_t pos0, uint32_t kv_len, uint32_t capacity) {
          auto plan = plan_qwen_attn_prefill_dispatch(tokens, QH);
          if (!plan.ok() || plan.value->empty()) { return; }
          stream << shader_ref(bq, bg, bk, bv, bscale_dummy.view(), bscale_dummy.view(), bo, tokens, QH, KVH, HD, scale, pos0, kv_len, capacity)
                        .dispatch(qwen_dispatch_threads(*plan.value));
      };
      const auto finite = [](luisa::span<const float> v) {
          for (auto x : v) {
              if (!std::isfinite(x)) { return false; }
          }
          return true;
      };

      const auto run_case = [&](const char *variant, QwenTiledAttnShader &shader,
                                const QwenAttnPrefillTiledConfig &cfg, uint32_t tokens,
                                uint32_t pos0, uint32_t capacity, bool with_oracle) {
          const uint32_t kv_len = pos0 + tokens;
          const size_t qn = static_cast<size_t>(tokens) * QH * HD;
          const size_t kn = static_cast<size_t>(capacity) * KVH * HD;
          luisa::vector<float> q(qn), gate(qn), kc(kn), vc(kn);
          rng.fill(q, -1.f, 1.f);
          rng.fill(gate, -2.f, 2.f);
          rng.fill(kc, -1.f, 1.f);
          rng.fill(vc, -1.f, 1.f);
          auto bq = upload(dev, stream, q);
          auto bg = upload(dev, stream, gate);
          auto bk = upload(dev, stream, kc);
          auto bv = upload(dev, stream, vc);
          auto bo_ref = dev.create_buffer<float>(qn);
          auto bo_tiled = dev.create_buffer<float>(qn);
          run_ref(bq, bg, bk, bv, bo_ref, tokens, pos0, kv_len, capacity);
          run_tiled(shader, cfg, bq, bg, bk, bv, bo_tiled, tokens, pos0, kv_len, capacity);
          auto got_ref = download(stream, bo_ref, qn);
          auto got_tiled = download(stream, bo_tiled, qn);
          const auto name = [&](const char *what) {
              return luisa::string{variant} + " S=" + std::to_string(tokens).c_str() +
                     " pos0=" + std::to_string(pos0).c_str() +
                     " cap=" + std::to_string(capacity).c_str() + " " + what;
          };
          check(finite(got_ref) && finite(got_tiled), name("finite outputs").c_str());
          check_close(got_tiled, got_ref, 2e-5f, 2e-4f, name("tiled vs reference kernel").c_str());
          if (!with_oracle) { return; }
          luisa::vector<float> want;
          check(ref_qwen_attn_prefill(q, gate, kc, vc, tokens, QH, KVH, HD, scale, pos0, kv_len, want).ok(),
                name("cpu oracle").c_str());
          check_close(got_ref, want, 2e-5f, 2e-4f, name("reference kernel vs cpu oracle").c_str());
          check_close(got_tiled, want, 2e-5f, 2e-4f, name("tiled vs cpu oracle").c_str());
      };

      const uint32_t lengths[] = {1u, 2u, 3u, 4u, 31u, 32u, 33u, 63u, 64u, 65u, 127u, 128u, 129u, 1024u};
      const uint32_t offsets[] = {0u, 63u, 64u, 512u};
      for (auto tokens : lengths) {
          for (auto pos0 : offsets) {
              // Ragged tail: the cache never lands on a tile/key boundary.
              const uint32_t capacity = pos0 + tokens + 3u;
              // The double-accumulation oracle is O(S^2*hd*q_heads) on the host:
              // keep it for the whole sweep except the two off-boundary positions
              // of the longest prompt, which still get the two device paths.
              const bool with_oracle = tokens <= 129u || pos0 == 0u || pos0 == 512u;
              run_case("tiled(selected)", shader_selected, selected, tokens, pos0, capacity, with_oracle);
              run_case("tiled(narrow)", shader_narrow, narrow, tokens, pos0, capacity, with_oracle);
          }
      }
      // Full cache: capacity exactly pos0 + S (the last valid key is the last slot).
      run_case("tiled(selected)", shader_selected, selected, 33u, 64u, 64u + 33u, true);
      run_case("tiled(narrow)", shader_narrow, narrow, 33u, 64u, 64u + 33u, true);
      run_case("tiled(selected)", shader_selected, selected, 1u, 0u, 1u, true);
      // Token tail on a Q_TILE boundary with an empty trailing tile.
      run_case("tiled(selected)", shader_selected, selected, 8u, 0u, 12u, true);
      run_case("tiled(selected)", shader_selected, selected, 5u, 3u, 9u, true);

      // Defensive kv_len clamp: kv_len beyond the cache capacity. The host
      // validator rejects this (checked above), so dispatch the kernel directly:
      // it clamps keys to min(kv_len, capacity), and therefore must produce the
      // same finite result as the in-range call (the reference kernel ignores
      // kv_len and is the cross-check).
      {
          const uint32_t tokens = 64u, pos0 = 0u, capacity = 64u;
          const size_t qn = static_cast<size_t>(tokens) * QH * HD;
          const size_t kn = static_cast<size_t>(capacity) * KVH * HD;
          luisa::vector<float> q(qn), gate(qn), kc(kn), vc(kn);
          rng.fill(q, -1.f, 1.f);
          rng.fill(gate, -2.f, 2.f);
          rng.fill(kc, -1.f, 1.f);
          rng.fill(vc, -1.f, 1.f);
          auto bq = upload(dev, stream, q);
          auto bg = upload(dev, stream, gate);
          auto bk = upload(dev, stream, kc);
          auto bv = upload(dev, stream, vc);
          auto bo_lo = dev.create_buffer<float>(qn);
          auto bo_hi = dev.create_buffer<float>(qn);
          auto bo_ref = dev.create_buffer<float>(qn);
          run_tiled(shader_selected, selected, bq, bg, bk, bv, bo_lo, tokens, pos0, capacity, capacity);
          run_tiled(shader_selected, selected, bq, bg, bk, bv, bo_hi, tokens, pos0,
                    capacity + 4096u, capacity);
          run_ref(bq, bg, bk, bv, bo_ref, tokens, pos0, capacity, capacity);
          auto got_lo = download(stream, bo_lo, qn);
          auto got_hi = download(stream, bo_hi, qn);
          auto got_ref = download(stream, bo_ref, qn);
          check(finite(got_hi), "tiled attn defensive clamp: finite");
          check_close(got_hi, got_lo, 0.f, 0.f, "tiled attn defensive kv_len clamp == capacity");
          check_close(got_hi, got_ref, 2e-5f, 2e-4f, "tiled attn defensive clamp vs reference kernel");
      }

      // The sigmoid output gate must be the same function of `gate` as in the
      // reference kernel: sigmoid(+50) ~ 1 and sigmoid(-50) ~ 0, so relative to
      // gate == 0 (sigmoid 0.5) the outputs must double and vanish.
      {
          const uint32_t tokens = 64u, pos0 = 7u, kv_len = pos0 + tokens, capacity = kv_len + 5u;
          const size_t qn = static_cast<size_t>(tokens) * QH * HD;
          const size_t kn = static_cast<size_t>(capacity) * KVH * HD;
          luisa::vector<float> q(qn), kc(kn), vc(kn);
          rng.fill(q, -1.f, 1.f);
          rng.fill(kc, -1.f, 1.f);
          rng.fill(vc, -1.f, 1.f);
          luisa::vector<float> gate_zero(qn, 0.0f), gate_pos(qn, 50.0f), gate_neg(qn, -50.0f);
          auto bq = upload(dev, stream, q);
          auto bk = upload(dev, stream, kc);
          auto bv = upload(dev, stream, vc);
          auto bo = dev.create_buffer<float>(qn);
          const auto run_gate = [&](luisa::vector<float> &host_gate) {
              auto bg = upload(dev, stream, host_gate);
              run_tiled(shader_selected, selected, bq, bg, bk, bv, bo, tokens, pos0, kv_len, capacity);
              return download(stream, bo, qn);
          };
          auto out_zero = run_gate(gate_zero);
          auto out_pos = run_gate(gate_pos);
          auto out_neg = run_gate(gate_neg);
          luisa::vector<float> want_pos(qn);
          for (size_t i = 0; i < qn; i++) { want_pos[i] = out_zero[i] * 2.0f; }
          check_close(out_pos, want_pos, 2e-5f, 2e-4f, "tiled attn gate +50 doubles the ungated output");
          bool vanished = true;
          for (size_t i = 0; i < qn; i++) {
              if (!std::isfinite(out_neg[i]) || std::fabs(out_neg[i]) > 1e-6f) { vanished = false; break; }
          }
          check(vanished, "tiled attn gate -50 suppresses the output");
      }
  }

}// namespace

// ---------------------------------------------------------------------------
// Device-resident decode state (opt_plan/03 section 2.3, milestone M-c).
//
// The static CUDA decode graph is only possible because the four per-token
// scalar consumers of a decode step read their argument from ONE small device
// buffer instead of a host kernel argument. These checks pin the differential
// contract: every device-state realization must reproduce the reference
// host-scalar realization element for element at the same position, and the
// in-kernel advance must happen exactly once per step (a stale position would
// corrupt the GDN conv ring, a stale KV length would truncate attention).
// ---------------------------------------------------------------------------
// Conditional WHILE decode chunk (opt_plan/09 N4.2).
//
// This is the ONLY place the device-terminated greedy step is exercised without a
// checkpoint, and it uses the PRODUCTION kernel (make_qwen_greedy_argmax_finish_
// while_kernel) inside a real WHILE container, so what is verified here is what the
// engine runs: the trip count of one graph launch is decided on the device by the
// token budget, the EOS id, the NaN sentinel and the context limit, and the host
// learns the real count from the state readback.
//
// Every case re-uploads the four candidate buffers and the state OUTSIDE the graph
// (exactly like the engine's pre-launch commands) and then issues one bare launch,
// so a replay after a state change is the same code path as a fresh chunk.
// ---------------------------------------------------------------------------
// Conditional WHILE decode chunk (opt_plan/09 N4.2).
//
// This is the only place the device-terminated greedy step runs without a
// checkpoint, and it uses the PRODUCTION kernel
// (make_qwen_greedy_argmax_finish_while_kernel) inside a real WHILE container, so
// what is verified here is what the engine runs: one graph launch executes 1..N
// tokens and the trip count is decided ON THE DEVICE by the token budget, the EOS
// id, the NaN sentinel and the context limit. The host learns the real count from
// the state readback (remaining), which is exactly the engine's contract.
//
// Every case re-uploads the four candidate buffers and the state OUTSIDE the graph
// (like the engine's pre-launch commands) and then issues a bare launch, so a
// replay after a state change is the same code path as a fresh chunk.
// ---------------------------------------------------------------------------
void test_greedy_while_conditional(Device &dev, Stream &stream) {
    auto *ext = dev.extension<CudaGraphExt>();
    if (ext == nullptr) {
        std::printf("[skip] conditional WHILE: no CudaGraphExt on backend '%s'\n",
                    luisa::string(dev.backend_name()).c_str());
        return;
    }
    const auto caps = ext->capabilities();
    check(caps.conditional_nodes, "conditional WHILE: CudaGraphExt reports conditional nodes");
    if (!caps.conditional_nodes) { return; }
    // The DEDICATED probe is the gate: the capability flag is a toolkit check, so a
    // real instantiate + execute + replay + zero-trip run has to pass first.
    const auto probe = qwen_probe_cuda_graph_while(dev, stream);
    check(probe.supported, "conditional WHILE probe: instantiate/execute/replay/zero-trip",
          probe.summary());
    if (!probe.supported) { return; }

    constexpr uint32_t kBlocks = qwen_greedy_argmax_blocks;// 256 candidates
    constexpr uint32_t kVocab = 1024u;
    constexpr uint32_t kHistoryCapacity = 16u;
    constexpr uint32_t kStep = 42u;  // token the candidate block selects
    constexpr uint32_t kOther = 7u;/* an EOS id the selection never produces */

    auto bvalues = dev.create_buffer<float>(kBlocks);
    auto bindices = dev.create_buffer<uint32_t>(kBlocks);
    auto binvalid = dev.create_buffer<uint32_t>(kBlocks);
    auto bnext = dev.create_buffer<uint32_t>(1u);
    auto bhistory = dev.create_buffer<uint32_t>(kHistoryCapacity);
    auto bstate = dev.create_buffer<uint32_t>(qwen_dec_state_count);
    // Parent-graph node 0: CUDA refuses a node-less graph, so the parent owns one
    // upload of a value nothing reads. Its host storage must outlive the launch.
    auto bseed = dev.create_buffer<uint32_t>(1u);
    luisa::vector<uint32_t> seed{0u};

    auto cmdlist = CommandList::create();
    cmdlist << bseed.view().copy_from(luisa::span{seed});
    auto graph = ext->create_graph(cmdlist.commit().command_list());
    check(graph.handle().handle != CudaGraphExt::invalid_handle,
          "conditional WHILE: parent graph captured");
    if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
    const auto graph_handle = graph.handle().handle;
    // Do-while semantics: the launch default is 1 and is re-assigned on every launch,
    // so a chunk always makes progress and the body alone decides when to stop.
    const auto handle = ext->create_conditional_handle(graph_handle, 1u, true);
    check(handle != 0u, "conditional WHILE: conditional handle created");
    if (handle == 0u) { return; }
    luisa::vector<CudaGraphExt::GraphHandle> bodies(1u, CudaGraphExt::invalid_handle);
    size_t conditional_index = ~size_t{0u};
    check(ext->add_conditional_node(graph_handle, CudaGraphExt::ConditionalType::WHILE,
                                    handle, 1u, luisa::span{bodies}, &conditional_index),
          "conditional WHILE: WHILE node added");
    check(bodies[0] != CudaGraphExt::invalid_handle,
          "conditional WHILE: CUDA handed back a body graph");
    check(ext->add_dependency(graph_handle, 0u, conditional_index),
          "conditional WHILE: seed -> WHILE dependency wired");
    // The body embeds the handle (created above) as a scalar argument, which is the
    // production pattern: the body is only ever populated after the handle exists, and
    // the handle is a session constant, so the body stays argument-invariant.
    auto finish = dev.compile(make_qwen_greedy_argmax_finish_while_kernel());
    auto finish_plan = plan_qwen_greedy_argmax_finish_dispatch(kVocab);
    check(finish_plan.ok(), "conditional WHILE: finish dispatch plan");
    if (!finish_plan.ok()) { return; }
    // The finish kernel is a 256-thread reduction: `.dispatch(n)` is a THREAD count in
    // this DSL, so the block count of the plan is what must be dispatched (a bare
    // `dispatch(1u)` would run one thread and leave 255 shared slots uninitialized).
    auto body_cmdlist = CommandList::create();
    body_cmdlist << finish(bvalues, bindices, binvalid, bnext, bhistory, bstate,
                           kVocab, kHistoryCapacity, handle, kOther)
                        .dispatch(qwen_dispatch_threads(*finish_plan.value));
    check(ext->build_body_graph(bodies[0], body_cmdlist.commit().command_list()),
          "conditional WHILE: body accepted (single device-terminated finish kernel)");
    auto exec = ext->instantiate(graph_handle);
    check(exec.handle().handle != CudaGraphExt::invalid_handle,
          "conditional WHILE: instantiated");
    if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }

    // Candidate block: index i has value -1 except kStep, which wins. `invalid` all
    // zero means "no NaN" (case E flips one to test the fail-closed sentinel).
    luisa::vector<float> values(kBlocks, -1.0f);
    luisa::vector<uint32_t> indices(kBlocks, 0u), invalid(kBlocks, 0u);
    for (uint32_t i = 0; i < kBlocks; i++) { indices[i] = i; }
    values[kStep] = 1.0f;
    luisa::vector<uint32_t> zero_history(kHistoryCapacity, 0u);

    // The EOS id is a BODY constant (kOther, like the engine's session constant), so a
    // case that must stop on EOS selects that id rather than changing a kernel argument.
    struct Expect {
        const char *name;
        uint32_t budget;
        uint32_t position;
        uint32_t selected;// candidate index the reduction picks
        bool nan;
        uint32_t executed;
        uint32_t eos;
        uint32_t token;
    };
    const Expect cases[] = {
        // (A) plain budget: 4 tokens, none of them EOS
        {"conditional WHILE: budget 4 runs 4 tokens", 4u, 0u, kStep, false, 4u, 0u, kStep},
        // (B) EOS on the FIRST token: exactly one token, eos flag set, 3 left
        {"conditional WHILE: EOS stops after 1 token", 4u, 0u, kOther, false, 1u, 1u, kOther},
        // (C) context limit: position 15 of a 16-token history, so one more token is
        //     legal but a second is not -> 1 token, eos flag CLEAR (the selection was
        //     fine, the loop stopped), 3 left
        {"conditional WHILE: context limit stops after 1 token", 4u, 15u, kStep, false, 1u, 0u, kStep},
        // (D) budget of 1: exactly one token, nothing left
        {"conditional WHILE: budget 1 runs 1 token", 1u, 0u, kStep, false, 1u, 0u, kStep},
        // (E) NaN in the candidate block: the sentinel (vocab) is selected, written to
        //     next_ids/history and reported as eos, so the loop can never survive it
        {"conditional WHILE: NaN sentinel stops the loop", 4u, 0u, kStep, true, 1u, 1u, kVocab},
    };
    for (const auto &c : cases) {
        auto v = values;
        auto inv = invalid;
        if (c.selected != kStep) {
            v[kStep] = -1.0f;
            v[c.selected] = 1.0f;
        }
        if (c.nan) { inv[3] = 1u; }
        luisa::vector<uint32_t> state(qwen_dec_state_count, 0u);
        state[qwen_dec_state_position] = c.position;
        state[qwen_dec_state_kv_len] = c.position + 1u;
        state[qwen_dec_state_remaining] = c.budget;
        // Uploads OUTSIDE the graph, exactly like the engine's pre-launch commands.
        stream << bvalues.view().copy_from(luisa::span{v})
               << bindices.view().copy_from(luisa::span{indices})
               << binvalid.view().copy_from(luisa::span{inv})
               << bstate.view().copy_from(luisa::span{state})
               << bhistory.view().copy_from(luisa::span{zero_history})
               << synchronize();
        ext->launch(exec.handle().handle, stream.handle());
        stream << synchronize();
        const auto after = download(stream, bstate, qwen_dec_state_count);
        const auto next = download(stream, bnext, 1u);
        const auto hist = download(stream, bhistory, kHistoryCapacity);
        // `executed` comes from the DEVICE budget, which is what the engine reads back.
        const auto remaining = after[qwen_dec_state_remaining];
        const auto executed = c.budget >= remaining ? c.budget - remaining : 0xFFFFFFFFu;
        check(executed == c.executed, c.name,
              luisa::string{"executed "} + std::to_string(executed).c_str() +
                  " (want " + std::to_string(c.executed).c_str() + "), remaining " +
                  std::to_string(remaining).c_str());
        check(after[qwen_dec_state_eos] == c.eos, c.name,
              luisa::string{"eos flag "} + std::to_string(after[qwen_dec_state_eos]).c_str() +
                  " (want " + std::to_string(c.eos).c_str() + ")");
        check(after[qwen_dec_state_position] == c.position + c.executed, c.name,
              luisa::string{"position "} +
                  std::to_string(after[qwen_dec_state_position]).c_str() +
                  " (want " + std::to_string(c.position + c.executed).c_str() + ")");
        check(after[qwen_dec_state_kv_len] == after[qwen_dec_state_position] + 1u, c.name,
              "kv_len == position + 1 after the chunk");
        check(next[0] == c.token, c.name,
              luisa::string{"next token "} + std::to_string(next[0]).c_str() +
                  " (want " + std::to_string(c.token).c_str() + ")");
        // Every executed iteration wrote its own slot and no UNEXECUTED slot was
        // touched (the history was zeroed above), so an early exit can never look like
        // unexecuted tokens ran.
        // The history slot comes from the state (0 in every case here), not from the
        // position, so the written range is [slot, slot + executed).
        bool slots_ok = true;
        for (uint32_t i = 0; i < c.executed; i++) {
            slots_ok = slots_ok && hist[i] == c.token;
        }
        for (uint32_t i = c.executed; i < kHistoryCapacity; i++) {
            slots_ok = slots_ok && hist[i] == 0u;
        }
        check(slots_ok, c.name,
              "exactly the executed slots were written (no unexecuted token ran)");
    }
    // (F) Reset/reuse: the SAME instantiated graph runs fresh chunks after the state
    // was re-uploaded, so nothing about the WHILE is one-shot.
    {
        luisa::vector<uint32_t> state(qwen_dec_state_count, 0u);
        state[qwen_dec_state_remaining] = 3u;
        auto inv = invalid;
        stream << bvalues.view().copy_from(luisa::span{values})
               << bindices.view().copy_from(luisa::span{indices})
               << binvalid.view().copy_from(luisa::span{inv})
               << bstate.view().copy_from(luisa::span{state})
               << bhistory.view().copy_from(luisa::span{zero_history})
               << synchronize();
        ext->launch(exec.handle().handle, stream.handle());
        // A replayed launch re-applies the assigned default 1 but NOT the budget: the
        // previous launch consumed it (remaining == 0), so the second launch runs its
        // one do-while iteration and then stops. This is the documented behaviour of a
        // do-while launch default and why the engine re-uploads the budget per chunk.
        ext->launch(exec.handle().handle, stream.handle());
        stream << synchronize();
        const auto after = download(stream, bstate, qwen_dec_state_count);
        const auto hist = download(stream, bhistory, kHistoryCapacity);
        // The replay re-applies the assigned default 1 (do-while) but NOT the budget,
        // which the previous launch consumed -- so it runs its one mandatory iteration
        // and stops. That is why the engine re-uploads the budget before every chunk.
        check(after[qwen_dec_state_position] == 4u,
              "conditional WHILE: replay without a new budget executes 3 + 1 tokens",
              luisa::string{"position "} + std::to_string(after[qwen_dec_state_position]).c_str());
        check(hist[3] == kStep && hist[4] == 0u,
              "conditional WHILE: replay fills exactly four history slots");
    }

    // (G) Host-side validation is device-free and must fail closed on an unusable
    // request BEFORE any graph work (the engine calls the same function).
    {
        QwenWhileRequest request;
        request.requested = true;
        request.chunk = 4u;
        request.eos_token = kStep;
        request.vocab = kVocab;
        request.max_context = 8u;
        check(qwen_validate_while_request(request).ok(),
              "while request validation accepts a bounded, EOS-carrying request");
        auto no_eos = request;
        no_eos.eos_token = 0xFFFFFFFFu;
        check(!qwen_validate_while_request(no_eos).ok(),
              "while request validation refuses a request without an EOS id");
        auto no_budget = request;
        no_budget.chunk = 0u;
        check(!qwen_validate_while_request(no_budget).ok(),
              "while request validation refuses a zero chunk");
        auto past_end = request;
        past_end.position = 8u;
        check(!qwen_validate_while_request(past_end).ok(),
              "while request validation refuses a position at max_context");
        auto too_long = request;
        too_long.position = 6u;
        check(!qwen_validate_while_request(too_long).ok(),
              "while request validation refuses a chunk past the remaining context");
        auto not_requested = request;
        not_requested.requested = false;
        not_requested.chunk = 0u;
        check(qwen_validate_while_request(not_requested).ok(),
              "while request validation is a no-op when the chunked path is not requested");
    }
}

// ---------------------------------------------------------------------------
// Pinned download ring (opt_plan/03 section 6, opt_plan/09 N4.3).
//
// The ring is the piece the engine's logits/token downloads use, so it is tested
// directly: a persistent page-locked buffer whose device->device copy is
// synchronized before the host memcpy. The cases pin the parts that can silently
// corrupt a caller's span: an ARBITRARY range (not ring-sized), a range larger than
// the ring (sliced), the empty range, an out-of-range request (refused, never an OOB
// read) and reuse of the same ring for a second pattern.
// ---------------------------------------------------------------------------
void test_pinned_download_ring(Device &dev, Stream &stream) {
    auto *ext = dev.extension<PinnedMemoryExt>();
    if (ext == nullptr) {
        std::printf("[skip] pinned downloads: no PinnedMemoryExt on backend '%s'\n",
                    luisa::string(dev.backend_name()).c_str());
        return;
    }
    constexpr uint32_t N = 1000u;
    constexpr uint32_t kRing = 7u;// tiny on purpose: forces many slices
    luisa::vector<uint32_t> pattern(N);
    for (uint32_t i = 0; i < N; i++) { pattern[i] = i * 2654435761u + 17u; }
    auto bsrc = upload(dev, stream, pattern);
    QwenPinnedRing ring = QwenPinnedRing::create(dev, ext, kRing, 16u);
    check(ring.valid(), "pinned ring: allocated through PinnedMemoryExt");
    if (!ring.valid()) { return; }
    check(ring.capacity() == kRing, "pinned ring: capacity is the requested one");
    check(ring.host_tokens() != nullptr && ring.host_logits() != nullptr,
          "pinned ring: both staging buffers expose their page-locked host address");
    QwenPinnedStats stats;
    // got[i] must equal want[want_offset + i] for every i (sizes may differ).
    const auto same = [](luisa::span<const uint32_t> got, luisa::span<const uint32_t> want,
                         size_t want_offset) {
        if (want_offset + got.size() > want.size()) { return false; }
        for (size_t i = 0; i < got.size(); i++) {
            if (got[i] != want[want_offset + i]) { return false; }
        }
        return true;
    };
    // (a) the FULL buffer through a 7-element ring: 143 slices, arbitrary length
    {
        luisa::vector<uint32_t> out(N, 0u);
        check(ring.download_tokens(stream, bsrc.view(), 0u, out, stats).ok(),
              "pinned ring: full-range download succeeds");
        check(same(out, pattern, 0u), "pinned ring: full-range content is byte-identical",
              luisa::string{"expected 143 slices, used "} + std::to_string(stats.slices).c_str());
    }
    // (b) an arbitrary interior range, not aligned to the ring size
    {
        luisa::vector<uint32_t> out(250u, 0u);
        check(ring.download_tokens(stream, bsrc.view(), 3u, out, stats).ok(),
              "pinned ring: interior range download succeeds");
        check(same(out, pattern, 3u), "pinned ring: interior range content is byte-identical");
    }
    // (c) the last element
    {
        luisa::vector<uint32_t> out(1u, 0u);
        check(ring.download_tokens(stream, bsrc.view(), N - 1u, out, stats).ok(),
              "pinned ring: single tail element download succeeds");
        check(same(out, pattern, N - 1u), "pinned ring: tail element is byte-identical");
    }
    // (d) the empty range moves nothing
    {
        luisa::vector<uint32_t> out;
        const auto before = stats.downloads;
        check(ring.download_tokens(stream, bsrc.view(), 5u, out, stats).ok(),
              "pinned ring: empty range succeeds without moving bytes");
        check(stats.downloads == before, "pinned ring: empty range records no download");
    }
    // (e) out of range: refused, never an out-of-bounds copy
    {
        luisa::vector<uint32_t> out(200u, 0u);
        check(!ring.download_tokens(stream, bsrc.view(), 900u, out, stats).ok(),
              "pinned ring: a range past the source is refused");
        luisa::vector<float> big(64u, 0.0f);
        check(!ring.download_floats(stream, ring.logits_buffer().view(), luisa::span{big}, stats).ok(),
              "pinned ring: a logits range past the staging buffer is refused");
    }
    // (f) reuse for a DIFFERENT pattern (the engine keeps the ring for the session)
    {
        luisa::vector<uint32_t> second(N);
        for (uint32_t i = 0; i < N; i++) { second[i] = 0x9E3779B9u ^ (i * 31u); }
        stream << bsrc.view().copy_from(luisa::span{second}) << synchronize();
        luisa::vector<uint32_t> out(N, 0u);
        check(ring.download_tokens(stream, bsrc.view(), 0u, out, stats).ok(),
              "pinned ring: reuse for a second pattern succeeds");
        check(same(out, second, 0u), "pinned ring: reused ring returns the new contents");
    }
    // (g) FP32 logits path (the prefill/decode download), inside and outside the
    // staging size
    {
        luisa::vector<float> logits(20u);
        for (uint32_t i = 0; i < logits.size(); i++) { logits[i] = 0.25f * float(i) - 2.0f; }
        auto bsrc_f = upload(dev, stream, logits);
        luisa::vector<float> out16(16u, 0.0f);
        check(ring.download_floats(stream, bsrc_f.view().subview(0u, 16u), out16, stats).ok(),
              "pinned ring: float logits download succeeds inside the staging size");
        bool ok = true;
        for (uint32_t i = 0; i < 16u; i++) { ok = ok && out16[i] == logits[i]; }
        check(ok, "pinned ring: float logits are bit-identical");
    }
    check(stats.host_copies == stats.slices,
          "pinned ring: one host memcpy per slice is accounted (NOT eliminated)",
          luisa::string{"copies "} + std::to_string(stats.host_copies).c_str() +
              ", slices " + std::to_string(stats.slices).c_str());
}

// ---------------------------------------------------------------------------
void test_decode_device_state(Device &dev, Stream &stream, Rng &rng) {
    // The f32/bf16 attention arms take the fp8 scale arguments but never read them
    // (the scale load folds away), so one 1-element buffer serves every unscaled
    // dispatch of this test.
    auto bscale_dummy = dev.create_buffer<float>(1);
    const auto upload_state = [&](uint32_t position, uint32_t kv_len, uint32_t slot) {
        luisa::vector<uint32_t> words(qwen_dec_state_count, 0u);
        words[qwen_dec_state_position] = position;
        words[qwen_dec_state_kv_len] = kv_len;
        words[qwen_dec_state_history_slot] = slot;
        return upload(dev, stream, words);
    };
    // (a) GDN conv ring over SUCCESSIVE positions (opt_plan/09 V3): six tokens,
    // one launch each, host pos0 = t versus state[position] = t. This is the
    // case the plan originally missed -- 24 layers of the decode body used to
    // change their pos0 argument every token.
    {
        constexpr uint32_t C = 40u, K = 4u, T = 6u, rw = K - 1u;
        luisa::vector<float> src(size_t(T) * C), weight(size_t(K) * C);
        rng.fill(src, -1.f, 1.f);
        rng.fill(weight, -0.5f, 0.5f);
        auto oracle_ring = luisa::vector<float>(size_t(C) * rw, 0.f);
        luisa::vector<float> oracle;
        check(ref_qwen_gdn_conv(src, weight, oracle_ring, T, C, K, 0u, oracle).ok(),
              "gdn conv dev oracle");
        auto bw = upload(dev, stream, weight);
        auto bring_ref = dev.create_buffer<float>(C * rw);
        auto bring_dev = dev.create_buffer<float>(C * rw);
        auto zsh = dev.compile(make_qwen_zero_fill_kernel());
        auto zplan = plan_qwen_threads(C * rw);
        check(zplan.ok(), "gdn conv dev zero plan");
        stream << zsh(bring_ref, C * rw).dispatch(qwen_dispatch_threads(*zplan.value))
               << zsh(bring_dev, C * rw).dispatch(qwen_dispatch_threads(*zplan.value));
        auto sh_ref = dev.compile(make_qwen_gdn_conv_silu_kernel());
        auto sh_dev = dev.compile(make_qwen_gdn_conv_silu_dev_kernel());
        auto plan = plan_qwen_gdn_conv_dispatch(C);
        check(plan.ok(), "gdn conv dev plan");
        auto out_ref = dev.create_buffer<float>(C);
        auto out_dev = dev.create_buffer<float>(C);
        luisa::vector<float> got_dev, got_ref;
        for (uint32_t t = 0; t < T; t++) {
            luisa::vector<float> chunk(src.begin() + size_t(t) * C, src.begin() + size_t(t + 1) * C);
            auto bsrc = upload(dev, stream, chunk);
            auto bstate = upload_state(t, t + 1u, 0u);
            stream << sh_ref(bsrc, bw, bring_ref, out_ref, 1u, C, K, t)
                          .dispatch(qwen_dispatch_threads(*plan.value))
                   << sh_dev(bsrc, bw, bring_dev, out_dev, bstate, 1u, C, K)
                          .dispatch(qwen_dispatch_threads(*plan.value));
            auto r = download(stream, out_ref, C);
            auto d = download(stream, out_dev, C);
            stream << synchronize();
            char name[96];
            std::snprintf(name, sizeof(name), "gdn conv dev == host at position %u", t);
            check_exact(d, r, name);
            got_ref.insert(got_ref.end(), r.begin(), r.end());
            got_dev.insert(got_dev.end(), d.begin(), d.end());
        }
        check_close(got_dev, oracle, 1e-5f, 1e-4f, "gdn conv dev six successive positions");
        check_close(got_ref, oracle, 1e-5f, 1e-4f, "gdn conv host six successive positions");
        check_exact(download(stream, bring_dev, C * rw), oracle_ring,
                    "gdn conv dev ring history matches the recurrence");
        check_exact(download(stream, bring_ref, C * rw), oracle_ring,
                    "gdn conv host ring history matches the recurrence");
    }
    // (b) Attention prepare (device position, implicit ramp, KV append) and the
    // decode attention (device KV length) at three absolute positions.
    {
        QwenDslConfig cfg;
        cfg.q_heads = 4u;
        cfg.kv_heads = 2u;
        cfg.attn_head_dim = 64u;
        cfg.rope_dim = 16u;
        const uint32_t hd = cfg.attn_head_dim;
        const uint32_t capacity = 11u;
        const uint32_t kv_n = cfg.attn_kv_n();
        const uint32_t q_n = cfg.q_heads * hd;
        luisa::vector<float> q_proj(cfg.attn_q_proj_n()), k_proj(kv_n), v_proj(kv_n);
        luisa::vector<float> qnw(hd), knw(hd);
        luisa::vector<int32_t> positions(1u, 0);
        rng.fill(q_proj, -1.f, 1.f);
        rng.fill(k_proj, -1.f, 1.f);
        rng.fill(v_proj, -1.f, 1.f);
        rng.fill(qnw, 0.7f, 1.3f);
        rng.fill(knw, 0.7f, 1.3f);
        luisa::vector<float> k_seed(size_t(capacity) * kv_n), v_seed(size_t(capacity) * kv_n);
        rng.fill(k_seed, -1.f, 1.f);
        rng.fill(v_seed, -1.f, 1.f);
        auto bqp = upload(dev, stream, q_proj);
        auto bkp = upload(dev, stream, k_proj);
        auto bvp = upload(dev, stream, v_proj);
        auto bqnw = upload(dev, stream, qnw);
        auto bknw = upload(dev, stream, knw);
        auto bpos = upload(dev, stream, positions);
        const uint32_t q_rows = cfg.q_heads, kv_rows = cfg.kv_heads;
        auto sh_host = dev.compile(make_qwen_attn_prepare_kernel(true, true));
        auto sh_dev = dev.compile(make_qwen_attn_prepare_dev_kernel());
        auto prep_plan = plan_qwen_attn_prepare_dispatch(1u, cfg);
        check(prep_plan.ok(), "attn prepare dev plan");
        auto dec_host = dev.compile(make_qwen_attn_decode_kernel());
        auto dec_dev = dev.compile(make_qwen_attn_decode_dev_kernel());
        auto dec_plan = plan_qwen_attn_decode_dispatch(cfg.q_heads);
        check(dec_plan.ok(), "attn decode dev plan");
        for (uint32_t pos0 : {0u, 3u, 7u}) {
            // Both realizations start from the same seeded cache, so they must
            // write the same slot and read the same history.
            auto bkc_h = upload(dev, stream, k_seed);
            auto bvc_h = upload(dev, stream, v_seed);
            auto bkc_d = upload(dev, stream, k_seed);
            auto bvc_d = upload(dev, stream, v_seed);
            auto bstate = upload_state(pos0, pos0 + 1u, 0u);
            auto bq = dev.create_buffer<float>(q_n);
            auto bgate = dev.create_buffer<float>(q_n);
            auto bk = dev.create_buffer<float>(kv_rows * hd);
            auto bv = dev.create_buffer<float>(kv_rows * hd);
            stream << sh_host(bqp, bkp, bvp, bqnw, bknw, bpos, bq, bgate, bk, bv,
                              bkc_h, bvc_h, 1u, cfg.q_heads, cfg.kv_heads, hd, cfg.rope_dim,
                              cfg.rope_theta, cfg.norm_eps, pos0, capacity, q_rows, kv_rows, kv_rows)
                          .dispatch(qwen_dispatch_threads(*prep_plan.value))
                   << sh_dev(bqp, bkp, bvp, bqnw, bknw, bpos, bq, bgate, bk, bv,
                             bkc_d, bvc_d, bstate, 1u, cfg.q_heads, cfg.kv_heads, hd, cfg.rope_dim,
                             cfg.rope_theta, cfg.norm_eps, capacity, q_rows, kv_rows, kv_rows)
                          .dispatch(qwen_dispatch_threads(*prep_plan.value));
            auto dkc = download(stream, bkc_d, k_seed.size());
            auto dvc = download(stream, bvc_d, v_seed.size());
            auto dq = download(stream, bq, q_n);
            auto dgate = download(stream, bgate, q_n);
            auto dk = download(stream, bk, kv_rows * hd);
            auto dv = download(stream, bv, kv_rows * hd);
            stream << synchronize();
            auto rkc = download(stream, bkc_h, k_seed.size());
            auto rvc = download(stream, bvc_h, v_seed.size());
            char name[96];
            std::snprintf(name, sizeof(name), "attn prepare dev k cache append pos0=%u", pos0);
            check(rkc == dkc && rvc == dvc, name);
            // The seeded cache must have changed: proves the device position
            // really indexed the cache instead of a constant zero.
            check(rkc != k_seed, "attn prepare appended at the device position");
            std::snprintf(name, sizeof(name), "attn prepare dev q/gate/k/v pos0=%u", pos0);
            auto rq = download(stream, bq, q_n);
            auto rgate = download(stream, bgate, q_n);
            auto rk = download(stream, bk, kv_rows * hd);
            auto rv = download(stream, bv, kv_rows * hd);
            stream << synchronize();
            check(rq == dq && rgate == dgate && rk == dk && rv == dv, name);
            // Decode attention over the appended cache: host kv_len == pos0 + 1.
            auto bdec_h = dev.create_buffer<float>(q_n);
            auto bdec_d = dev.create_buffer<float>(q_n);
            stream << dec_host(bq, bgate, bkc_h, bvc_h, bscale_dummy.view(), bscale_dummy.view(), bdec_h, cfg.q_heads, cfg.kv_heads, hd,
                               cfg.attn_scale(), pos0 + 1u, capacity)
                          .dispatch(qwen_dispatch_threads(*dec_plan.value))
                   << dec_dev(bq, bgate, bkc_h, bvc_h, bscale_dummy.view(), bscale_dummy.view(), bdec_d, bstate, cfg.q_heads, cfg.kv_heads,
                              hd, cfg.attn_scale(), capacity)
                          .dispatch(qwen_dispatch_threads(*dec_plan.value));
            auto ddec = download(stream, bdec_d, q_n);
            stream << synchronize();
            auto rdec = download(stream, bdec_h, q_n);
            stream << synchronize();
            std::snprintf(name, sizeof(name), "attn decode dev kv_len=%u", pos0 + 1u);
            check_exact(ddec, rdec, name);
        }
    }
    // (c) Greedy argmax finish on the device state: same selection as the
    // host-slot reference, the token stored in the DEVICE history slot, and the
    // state advanced exactly once (position, kv_len, slot).
    {
        constexpr uint32_t vocab = 4096u, history_capacity = 8u;
        luisa::vector<float> logits(vocab, -1.0f);
        logits[17] = 12.5f;
        logits[2047] = 12.5f;// tie -> the lowest index wins
        auto bvalues = dev.create_buffer<float>(qwen_greedy_argmax_blocks);
        auto bindices = dev.create_buffer<uint32_t>(qwen_greedy_argmax_blocks);
        auto binvalid = dev.create_buffer<uint32_t>(qwen_greedy_argmax_blocks);
        auto partial = dev.compile(make_qwen_greedy_argmax_partial_kernel());
        auto finish_host = dev.compile(make_qwen_greedy_argmax_finish_kernel());
        auto finish_dev = dev.compile(make_qwen_greedy_argmax_finish_dev_kernel());
        auto pp = plan_qwen_greedy_argmax_partial_dispatch(vocab);
        auto fp = plan_qwen_greedy_argmax_finish_dispatch(vocab);
        check(pp.ok() && fp.ok(), "greedy finish dev plans");
        const luisa::vector<uint32_t> empty_history(history_capacity, 0xFFFFFFFFu);
        struct FinishRun {
            uint32_t host_token{0u};
            uint32_t dev_token{0u};
            luisa::vector<uint32_t> host_history;
            luisa::vector<uint32_t> dev_history;
            luisa::vector<uint32_t> state;
        };
        const auto run = [&](uint32_t position, uint32_t slot, bool nan) {
            FinishRun out;
            auto entry = logits;
            if (nan) { entry[999] = std::numeric_limits<float>::quiet_NaN(); }
            auto blogits = upload(dev, stream, entry);
            auto bnext_h = dev.create_buffer<uint32_t>(1u);
            auto bnext_d = dev.create_buffer<uint32_t>(1u);
            auto bhist_h = upload(dev, stream, empty_history);
            auto bhist_d = upload(dev, stream, empty_history);
            auto bstate = upload_state(position, position + 1u, slot);
            stream << partial(blogits, bvalues, bindices, binvalid, vocab)
                          .dispatch(qwen_dispatch_threads(*pp.value))
                   << finish_host(bvalues, bindices, binvalid, bnext_h, bhist_h, vocab, slot)
                          .dispatch(qwen_dispatch_threads(*fp.value))
                   << partial(blogits, bvalues, bindices, binvalid, vocab)
                          .dispatch(qwen_dispatch_threads(*pp.value))
                   << finish_dev(bvalues, bindices, binvalid, bnext_d, bhist_d, bstate, vocab,
                                 history_capacity)
                          .dispatch(qwen_dispatch_threads(*fp.value));
            out.host_token = download(stream, bnext_h, 1u)[0];
            out.dev_token = download(stream, bnext_d, 1u)[0];
            out.host_history = download(stream, bhist_h, history_capacity);
            out.dev_history = download(stream, bhist_d, history_capacity);
            out.state = download(stream, bstate, qwen_dec_state_count);
            stream << synchronize();
            return out;
        };
        {
            const auto r = run(5u, 2u, false);
            check(r.host_token == 17u && r.dev_token == 17u,
                  "greedy finish dev selection == host selection");
            check(r.host_history == r.dev_history, "greedy finish dev history == host history");
            check(r.dev_history[2u] == 17u, "greedy finish dev wrote the device slot");
            check(r.state[qwen_dec_state_position] == 6u, "greedy finish dev advanced position");
            check(r.state[qwen_dec_state_kv_len] == 7u, "greedy finish dev re-derived kv_len");
            check(r.state[qwen_dec_state_history_slot] == 3u, "greedy finish dev advanced the slot");
            check(r.state[qwen_dec_state_eos] == 0u, "greedy finish dev left the reserved word alone");
        }
        {
            // NaN logits: the fail-closed vocabulary sentinel is stored (never a
            // plausible token id) and the state still advances.
            const auto r = run(9u, 0u, true);
            check(r.host_token == vocab && r.dev_token == vocab,
                  "greedy finish dev NaN sentinel == host sentinel");
            check(r.dev_history[0u] == vocab, "greedy finish dev stored the NaN sentinel");
            check(r.state[qwen_dec_state_position] == 10u && r.state[qwen_dec_state_kv_len] == 11u,
                  "greedy finish dev advanced the state after a NaN");
        }
        {
            // Last valid slot; and an out-of-range slot must not write anything
            // (kernel-level guard, i.e. no OOB store even if a caller lies).
            const auto last = run(0u, history_capacity - 1u, false);
            check(last.dev_token == 17u && last.dev_history[history_capacity - 1u] == 17u,
                  "greedy finish dev last valid slot");
            const auto oob = run(0u, history_capacity, false);
            check(oob.dev_token == 17u && oob.dev_history == empty_history,
                  "greedy finish dev refuses an out-of-range slot");
        }
    }
}

  // ---------------------------------------------------------------------------
  // Split-KV decode attention with GQA dedup (opt_plan/03 section 4.2).
  //
  // Three-way differential: CPU oracle (double accumulation) / legacy per-q_head
  // kernel / split+combine family, at the plan's kv_len boundary set (1, 511, 512,
  // 513, capacity) plus the exact split boundaries of the in-kernel split rule.
  // The split path reorders the FP32 softmax reduction, so it is compared with the
  // documented reassociation envelope (the same one the tiled prefill attention
  // uses), never with the legacy kernel's tighter bound.
  // ---------------------------------------------------------------------------
  void test_decode_split(Device &dev, Stream &stream, Rng &rng) {
    // The f32/bf16 attention arms take the fp8 scale arguments but never read them
    // (the scale load folds away), so one 1-element buffer serves every unscaled
    // dispatch of this test.
    auto bscale_dummy = dev.create_buffer<float>(1);
      QwenDslConfig dsl_cfg;// model geometry: 16 q heads / 4 kv heads / 256 dims
      const uint32_t qh = dsl_cfg.q_heads, kvh = dsl_cfg.kv_heads, hd = dsl_cfg.attn_head_dim;
      const uint32_t capacity = 1024u;
      const uint32_t qn = qh * hd;
      const uint32_t kv_n = capacity * kvh * hd;
      luisa::vector<float> q(qn), gate(qn), k_cache(kv_n), v_cache(kv_n);
      rng.fill(q, -1.f, 1.f);
      rng.fill(gate, -2.f, 2.f);
      rng.fill(k_cache, -1.f, 1.f);
      rng.fill(v_cache, -1.f, 1.f);
      auto bq = upload(dev, stream, q);
      auto bg = upload(dev, stream, gate);
      auto bk = upload(dev, stream, k_cache);
      auto bv = upload(dev, stream, v_cache);
      // ---- host-side selection / bound rules (no device needed) --------------
      {
          QwenAttnDecodeSplitConfig sel{};
          check(select_qwen_attn_decode_split_config(32u, 49152u, 0u, sel).ok(),
                "split select: 48 KiB / no SM count");
          check(sel.k_tile == 8u, "split select: the measured 16 KiB tile is selected");
          check(sel.max_splits == qwen_attn_decode_split_default_bound,
                "split select: no SM count falls back to the documented default bound");
          check(select_qwen_attn_decode_split_config(32u, 0u, 0u, sel).ok() && sel.k_tile == 8u,
                "split select: unreported shared memory picks the portable 16 KiB tile");
          check(select_qwen_attn_decode_split_config(32u, 101376u, 80u, sel).ok() &&
                    sel.k_tile == 8u &&
                    sel.max_splits == qwen_attn_decode_split_blocks_per_sm * 80u / 4u,
                "split select: a 96 KiB report keeps the measured tile and 2 blocks/SM");
          check(select_qwen_attn_decode_split_config(32u, 101376u, 170u, sel).ok() &&
                    sel.max_splits == qwen_attn_decode_split_cap,
                "split select: a 170-SM part is capped at the 40-split plan bound");
          check(!select_qwen_attn_decode_split_config(64u, 49152u, 0u, sel).ok(),
                "split select: a 64-lane warp device is rejected (kernel pins 32)");
          check(!select_qwen_attn_decode_split_config(32u, 8u, 0u, sel).ok(),
                "split select: a shared-memory budget below the tile floor is rejected");
          // Reported shared-memory budget sweep. The measured 16 KiB tile is a
          // CEILING, and the selection must halve k_tile until the K+V staging fits a
          // smaller REPORTED limit (2 KiB -> 1 key, 4 KiB -> 2, 8/12 KiB -> 4,
          // >= 16 KiB -> 8): a device whose budget only allows 4/8/12 KiB used to
          // receive a 16 KiB image, i.e. a hard compile failure instead of a measured
          // fallback.
          {
              struct LimitCase {
                  uint32_t limit;
                  uint32_t k_tile;
              };
              const LimitCase limit_cases[] = {{2048u, 1u},  {4096u, 2u},  {8192u, 4u}, {12288u, 4u},
                                               {16384u, 8u}, {49152u, 8u}, {101376u, 8u}};
              for (const auto &c : limit_cases) {
                  QwenAttnDecodeSplitConfig picked{};
                  const auto st = select_qwen_attn_decode_split_config(32u, c.limit, 0u, picked);
                  const auto shared = qwen_attn_decode_split_shared_bytes(picked);
                  const auto plan = plan_qwen_attn_decode_split_dispatch(picked);
                  char name[192];
                  std::snprintf(name, sizeof(name),
                                "split select: %u B report -> k_tile %u, %u B shared <= limit, planner valid",
                                c.limit, picked.k_tile, shared);
                  check(st.ok() && picked.k_tile == c.k_tile && shared <= c.limit && plan.ok() &&
                            shared == 2u * picked.k_tile * picked.head_dim * 4u &&
                            qwen_attn_decode_split_blocks(picked) == kvh * picked.max_splits,
                        name);
              }
              // Just below the one-key floor: no tile fits, so the selection refuses it
              // instead of compiling a 2 KiB image the driver would reject.
              for (uint32_t below : {1u, 1024u, 2047u}) {
                  QwenAttnDecodeSplitConfig picked{};
                  char name[160];
                  std::snprintf(name, sizeof(name),
                                "split select: %u B report is below the 2 KiB one-key floor and is refused",
                                below);
                  check(!select_qwen_attn_decode_split_config(32u, below, 0u, picked).ok(), name);
              }
              // The target box (48 KiB reported) and an unreported limit (DX) keep
              // exactly the measured configuration: no performance/policy change.
              QwenAttnDecodeSplitConfig target{};
              QwenAttnDecodeSplitConfig unreported{};
              check(select_qwen_attn_decode_split_config(32u, 49152u, 0u, target).ok() &&
                        select_qwen_attn_decode_split_config(32u, 0u, 0u, unreported).ok() &&
                        target.k_tile == 8u && unreported.k_tile == 8u &&
                        qwen_attn_decode_split_shared_bytes(target) == 16384u &&
                        qwen_attn_decode_split_shared_bytes(unreported) == 16384u &&
                        target.max_splits == qwen_attn_decode_split_default_bound &&
                        unreported.max_splits == qwen_attn_decode_split_default_bound,
                    "split select: 48 KiB and an unreported limit keep the measured 16 KiB tile");
          }
          // A caller override of the compile-time split bound
          // (QwenEngineOptions::dsl_split_kv_max_splits / --split-kv-max-splits) must stay
          // inside the plan cap, must not index past the fixed partial workspace, and must
          // leave the selected K/V tile inside the reported shared-memory budget.
          {
              QwenAttnDecodeSplitConfig over{};
              check(select_qwen_attn_decode_split_config(32u, 49152u, 0u, over).ok() &&
                        over.k_tile == 8u && over.max_splits == qwen_attn_decode_split_default_bound,
                    "split override: baseline is the measured tile and the default bound");
              check(qwen_attn_decode_split_apply_max_splits_override(over, 24u, 49152u).ok() &&
                        over.max_splits == 24u && over.k_tile == 8u &&
                        qwen_attn_decode_split_shared_bytes(over) <= 49152u,
                    "split override: 24 replaces the bound and keeps the 16 KiB tile");
              check(qwen_attn_decode_split_apply_max_splits_override(
                        over, qwen_attn_decode_split_cap, 49152u).ok() &&
                        over.max_splits == qwen_attn_decode_split_cap &&
                        qwen_attn_decode_split_partial_bytes(over) == 660480ull,
                    "split override: the cap is accepted and sizes the 645 KiB partials");
              check(!qwen_attn_decode_split_apply_max_splits_override(
                        over, qwen_attn_decode_split_cap + 1u, 49152u).ok() &&
                        over.max_splits == qwen_attn_decode_split_cap,
                    "split override: cap+1 is refused and the config is left untouched");
              check(!qwen_attn_decode_split_apply_max_splits_override(over, 0xFFFFFFFFu, 49152u).ok() &&
                        over.max_splits == qwen_attn_decode_split_cap,
                    "split override: an overflowing value is refused before the partials wrap");
              // The selection can never produce an over-budget tile, but a hand-built
              // config must not slip past this path either.
              QwenAttnDecodeSplitConfig wide{};
              wide.max_splits = 12u;
              check(!qwen_attn_decode_split_apply_max_splits_override(wide, 16u, 4096u).ok() &&
                        wide.max_splits == 12u,
                    "split override: a tile over the reported budget is refused");
              check(qwen_attn_decode_split_apply_max_splits_override(wide, 0u, 4096u).ok() &&
                        wide.max_splits == 12u,
                    "split override: 0 keeps the derived bound (no-op)");
              // Every reported limit: the selection fits and the override keeps it
              // fitting, with a plan the kernels can still be dispatched with.
              bool override_fits = true;
              for (uint32_t limit : {2048u, 4096u, 8192u, 12288u, 16384u, 49152u}) {
                  QwenAttnDecodeSplitConfig c{};
                  if (!select_qwen_attn_decode_split_config(32u, limit, 0u, c).ok() ||
                      !qwen_attn_decode_split_apply_max_splits_override(c, 32u, limit).ok() ||
                      c.max_splits != 32u || qwen_attn_decode_split_shared_bytes(c) > limit ||
                      !plan_qwen_attn_decode_split_dispatch(c).ok()) {
                      override_fits = false;
                  }
              }
              check(override_fits,
                    "split override: the selected tile stays within every reported budget");
          }
          check(qwen_attn_decode_split_bound(0u, 4u) == qwen_attn_decode_split_default_bound,
                "split bound: unqueryable SM count -> the documented measured default");
          check(qwen_attn_decode_split_bound(80u, 4u) == 40u &&
                    qwen_attn_decode_split_bound(170u, 4u) == qwen_attn_decode_split_cap &&
                    qwen_attn_decode_split_bound(4u, 4u) == 2u,
                "split bound: 2 blocks/SM clamped to [1, 40]");
          QwenAttnDecodeSplitConfig bad{};
          bad.max_splits = qwen_attn_decode_split_cap + 1u;
          check(!validate_qwen_attn_decode_split_config(bad).ok(),
                "split config: max_splits above the cap is rejected");
          QwenAttnDecodeSplitConfig bad2{};
          bad2.q_heads = 8u;// 8/4 = 2 q heads per kv head: no warp mapping
          check(!validate_qwen_attn_decode_split_config(bad2).ok(),
                "split config: a GQA ratio other than 4 is rejected");
          QwenAttnDecodeSplitConfig bad3{};
          bad3.head_dim = 32u;
          bad3.k_tile = 1u;// 1*32 is not a multiple of the 128-thread load
          check(!validate_qwen_attn_decode_split_config(bad3).ok(),
                "split config: a tile that does not cover the block load is rejected");
          QwenAttnDecodeSplitConfig bad4{};
          bad4.threshold = 0u;
          check(!validate_qwen_attn_decode_split_config(bad4).ok(),
                "split config: threshold 0 is rejected");
          // Split-count / span rule: the plan's "scale with kv_len, cap at max_splits".
          QwenAttnDecodeSplitConfig rule{};
          check(qwen_attn_decode_split_blocks(rule) == kvh * rule.max_splits,
                "split blocks are kv_heads * max_splits (static grid)");
          bool spans_ok = true;
          for (uint32_t kv_len : {1u, 511u, 512u, 513u, 1024u, 8192u, 32768u, 131072u}) {
              const auto splits = qwen_attn_decode_split_count(kv_len, rule);
              uint64_t total = 0u;
              for (uint32_t s = 0u; s < splits; s++) { total += qwen_attn_decode_split_span(kv_len, s, rule); }
              if (total != kv_len || splits == 0u || splits > rule.max_splits) { spans_ok = false; }
              if (kv_len % rule.split_chunk == 0u && kv_len / rule.split_chunk < rule.max_splits) {
                  if (splits != kv_len / rule.split_chunk) { spans_ok = false; }
              }
          }
          check(spans_ok, "split rule: spans partition [0, kv_len) with splits <= max_splits");
      }
      // ---- three-way differential at the boundary set ------------------------
      // Config S: the split family owns EVERY kv_len (threshold 1), so kv_len 1 and
      // 511 exercise 1 split and a full first split instead of the legacy path.
      QwenAttnDecodeSplitConfig split_cfg{};
      split_cfg.threshold = 1u;
      // 64 keys per split so kv_len 1024 really exercises a 16-way reduction
      // (the engine's 512-key chunk would only reach 2 splits here).
      split_cfg.split_chunk = 64u;
      // Config G: the plan's guarded configuration (legacy below 512, split above).
      QwenAttnDecodeSplitConfig guard_cfg{};
      guard_cfg.threshold = 512u;
      const auto partial_acc_elems = static_cast<size_t>(split_cfg.max_splits) * qh * hd;
      const auto partial_ml_elems = static_cast<size_t>(split_cfg.max_splits) * qh * 2u;
      auto sh_legacy = dev.compile(make_qwen_attn_decode_kernel());
      auto sh_legacy_guarded = dev.compile(make_qwen_attn_decode_kernel(guard_cfg.threshold));
      auto sh_split = dev.compile(make_qwen_attn_decode_split_kernel(split_cfg));
      auto sh_split_dev = dev.compile(make_qwen_attn_decode_split_dev_kernel(split_cfg));
      auto sh_combine = dev.compile(make_qwen_attn_decode_combine_kernel(split_cfg));
      auto sh_combine_dev = dev.compile(make_qwen_attn_decode_combine_dev_kernel(split_cfg));
      auto sh_split_guarded = dev.compile(make_qwen_attn_decode_split_kernel(guard_cfg));
      auto sh_combine_guarded = dev.compile(make_qwen_attn_decode_combine_kernel(guard_cfg));
      check(sh_legacy.compile_ok() && sh_legacy_guarded.compile_ok() && sh_split.compile_ok() &&
                sh_split_dev.compile_ok() && sh_combine.compile_ok() && sh_combine_dev.compile_ok() &&
                sh_split_guarded.compile_ok() && sh_combine_guarded.compile_ok(),
            "split decode: shaders compile");
      auto legacy_plan = plan_qwen_attn_decode_dispatch(qh);
      auto split_plan = plan_qwen_attn_decode_split_dispatch(split_cfg);
      auto combine_plan = plan_qwen_attn_decode_combine_dispatch(split_cfg);
      auto split_plan_g = plan_qwen_attn_decode_split_dispatch(guard_cfg);
      auto combine_plan_g = plan_qwen_attn_decode_combine_dispatch(guard_cfg);
      check(legacy_plan.ok() && split_plan.ok() && combine_plan.ok() &&
                split_plan_g.ok() && combine_plan_g.ok(),
            "split decode: dispatch plans");
      if (!legacy_plan.ok() || !split_plan.ok() || !combine_plan.ok() ||
          !split_plan_g.ok() || !combine_plan_g.ok()) {
          return;
      }
      check(split_plan.value->work_items == kvh * split_cfg.max_splits,
            "split decode: plan is the static kv_heads * max_splits grid");
      check(combine_plan.value->work_items == qh, "split combine: one block per q head");
      auto bpartial_acc = dev.create_buffer<float>(partial_acc_elems);
      auto bpartial_ml = dev.create_buffer<float>(partial_ml_elems);
      auto bout_split = dev.create_buffer<float>(qn);
      auto bout_legacy = dev.create_buffer<float>(qn);
      const float scale = dsl_cfg.attn_scale();
      // The reassociation envelope for the split reduction (same order as the
      // tiled prefill acceptance in opt_plan/02 section 5 item 1).
      constexpr float kSplitAtol = 2.0e-4f;
      constexpr float kSplitRtol = 2.0e-3f;
      luisa::vector<float> out_split_last;
      for (uint32_t kv_len : {1u, 511u, 512u, 513u, capacity}) {
          luisa::vector<float> ref;
          check(ref_qwen_attn_decode(q, gate, k_cache, v_cache, qh, kvh, hd, scale, kv_len, ref).ok(),
                "split decode oracle");
          stream << sh_legacy(bq, bg, bk, bv, bscale_dummy.view(), bscale_dummy.view(), bout_legacy, qh, kvh, hd, scale, kv_len, capacity)
                        .dispatch(qwen_dispatch_threads(*legacy_plan.value))
                 << sh_split(bq, bk, bv, bscale_dummy.view(), bscale_dummy.view(), bpartial_acc, bpartial_ml, kv_len, scale, capacity)
                        .dispatch(qwen_dispatch_threads(*split_plan.value))
                 << sh_combine(bpartial_acc, bpartial_ml, bg, bout_split, kv_len, capacity)
                        .dispatch(qwen_dispatch_threads(*combine_plan.value));
          auto d_legacy = download(stream, bout_legacy, qn);
          auto d_split = download(stream, bout_split, qn);
          char name[128];
          std::snprintf(name, sizeof(name), "attn decode legacy vs oracle kv_len=%u", kv_len);
          check_close(d_legacy, ref, 2e-5f, 2e-4f, name);
          std::snprintf(name, sizeof(name), "attn decode split+combine vs oracle kv_len=%u", kv_len);
          check_close(d_split, ref, kSplitAtol, kSplitRtol, name);
          std::snprintf(name, sizeof(name), "attn decode split finite kv_len=%u", kv_len);
          bool finite = true;
          for (auto x : d_split) { if (!std::isfinite(x)) { finite = false; } }
          check(finite, name);
          // Device-state realization: identical arithmetic, so bit-identical output.
          {
              luisa::vector<uint32_t> words(qwen_dec_state_count, 0u);
              words[qwen_dec_state_position] = kv_len - 1u;
              words[qwen_dec_state_kv_len] = kv_len;
              auto bstate = upload(dev, stream, words);
              auto bout_dev = dev.create_buffer<float>(qn);
              auto bpartial_acc_dev = dev.create_buffer<float>(partial_acc_elems);
              auto bpartial_ml_dev = dev.create_buffer<float>(partial_ml_elems);
              stream << sh_split_dev(bq, bk, bv, bscale_dummy.view(), bscale_dummy.view(), bpartial_acc_dev, bpartial_ml_dev, bstate, scale, capacity)
                            .dispatch(qwen_dispatch_threads(*split_plan.value))
                     << sh_combine_dev(bpartial_acc_dev, bpartial_ml_dev, bg, bout_dev, bstate, capacity)
                            .dispatch(qwen_dispatch_threads(*combine_plan.value));
              auto d_dev = download(stream, bout_dev, qn);
              std::snprintf(name, sizeof(name), "attn decode split dev-state == host kv_len=%u", kv_len);
              check_exact(d_dev, d_split, name);
              stream << bstate.view().copy_to(luisa::span<uint32_t>{words}) << synchronize();
              std::snprintf(name, sizeof(name), "attn decode split never writes the device state kv_len=%u", kv_len);
              check(words[qwen_dec_state_kv_len] == kv_len && words[qwen_dec_state_position] == kv_len - 1u, name);
          }
          out_split_last = std::move(d_split);
      }
      // ---- guarded pairing: exactly one realization runs per kv_len ----------
      {
          const luisa::vector<float> poison_acc(partial_acc_elems, 7.0f);
          const luisa::vector<float> poison_ml(partial_ml_elems, 7.0f);
          for (uint32_t kv_len : {511u, 512u}) {
              auto bpoison_acc = upload(dev, stream, poison_acc);
              auto bpoison_ml = upload(dev, stream, poison_ml);
              auto bout_g = dev.create_buffer<float>(qn);
              luisa::vector<float> out_poison(qn, -5.0f);
              stream << bout_g.copy_from(luisa::span<const float>{out_poison})
                     << sh_legacy_guarded(bq, bg, bk, bv, bscale_dummy.view(), bscale_dummy.view(), bout_g, qh, kvh, hd, scale, kv_len, capacity)
                            .dispatch(qwen_dispatch_threads(*legacy_plan.value))
                     << sh_split_guarded(bq, bk, bv, bscale_dummy.view(), bscale_dummy.view(), bpoison_acc, bpoison_ml, kv_len, scale, capacity)
                            .dispatch(qwen_dispatch_threads(*split_plan_g.value))
                     << sh_combine_guarded(bpoison_acc, bpoison_ml, bg, bout_g, kv_len, capacity)
                            .dispatch(qwen_dispatch_threads(*combine_plan_g.value));
              auto d_out = download(stream, bout_g, qn);
              luisa::vector<float> ref;
              check(ref_qwen_attn_decode(q, gate, k_cache, v_cache, qh, kvh, hd, scale, kv_len, ref).ok(),
                    "split guard oracle");
              char name[128];
              std::snprintf(name, sizeof(name), "attn decode guarded pair picks one path kv_len=%u", kv_len);
              check_close(d_out, ref, kSplitAtol, kSplitRtol, name);
              const auto d_acc = download(stream, bpoison_acc, partial_acc_elems);
              const auto d_ml = download(stream, bpoison_ml, partial_ml_elems);
              const bool untouched = d_acc == poison_acc && d_ml == poison_ml;
              std::snprintf(name, sizeof(name), "attn decode split is a no-op below the threshold kv_len=%u", kv_len);
              check(untouched == (kv_len < guard_cfg.threshold), name);
          }
      }
      // ---- a down-clamped K/V tile still computes the right answer -----------
      // The selection halves k_tile when a device reports a small static budget
      // (4 KiB report -> 2 keys), so prove that realization is real: compile AND run
      // it at a kv_len that exercises a multi-split reduction, against the oracle.
      {
          QwenAttnDecodeSplitConfig small = split_cfg;
          small.k_tile = 2u;// what a 4 KiB static shared-memory report selects
          check(validate_qwen_attn_decode_split_config(small).ok(),
                "split small tile: config valid");
          auto sh_small = dev.compile(make_qwen_attn_decode_split_kernel(small));
          check(sh_small.compile_ok(), "split small tile: shader compiles");
          // The combine kernel does not read k_tile, so the split-family combine
          // compiled above reduces these partials unchanged.
          const uint32_t kv_len = 777u;// 13 splits at split_chunk 64
          luisa::vector<float> ref;
          check(ref_qwen_attn_decode(q, gate, k_cache, v_cache, qh, kvh, hd, scale, kv_len, ref).ok(),
                "split small tile: oracle");
          auto bsmall_acc = dev.create_buffer<float>(partial_acc_elems);
          auto bsmall_ml = dev.create_buffer<float>(partial_ml_elems);
          auto bsmall_out = dev.create_buffer<float>(qn);
          stream << sh_small(bq, bk, bv, bscale_dummy.view(), bscale_dummy.view(), bsmall_acc, bsmall_ml, kv_len, scale, capacity)
                        .dispatch(qwen_dispatch_threads(*split_plan.value))
                 << sh_combine(bsmall_acc, bsmall_ml, bg, bsmall_out, kv_len, capacity)
                        .dispatch(qwen_dispatch_threads(*combine_plan.value));
          auto d_small = download(stream, bsmall_out, qn);
          check_close(d_small, ref, kSplitAtol, kSplitRtol,
                      "split small tile: k_tile 2 matches the oracle at kv_len 777");
      }
      // ---- empty splits: a capped split count leaves trailing splits empty ---
      {
          QwenAttnDecodeSplitConfig tight{};
          tight.threshold = 1u;
          tight.split_chunk = 8u;
          tight.max_splits = 16u;
          check(validate_qwen_attn_decode_split_config(tight).ok(), "split empty: config valid");
          auto sh_tight = dev.compile(make_qwen_attn_decode_split_kernel(tight));
          auto sh_tight_combine = dev.compile(make_qwen_attn_decode_combine_kernel(tight));
          auto plan_tight = plan_qwen_attn_decode_split_dispatch(tight);
          auto plan_tight_c = plan_qwen_attn_decode_combine_dispatch(tight);
          check(plan_tight.ok() && plan_tight_c.ok(), "split empty: plans");
          if (plan_tight.ok() && plan_tight_c.ok()) {
              const uint32_t kv_len = 100u;// splits 13, per 8 -> split 13..15 empty
              const auto splits = qwen_attn_decode_split_count(kv_len, tight);
              uint32_t empties = 0u;
              for (uint32_t s = 0u; s < tight.max_splits; s++) {
                  if (qwen_attn_decode_split_span(kv_len, s, tight) == 0u) { empties++; }
              }
              check(splits == 13u && empties == 3u, "split empty: 100 keys over 16 splits leave 3 empty");
              const auto acc_elems = static_cast<size_t>(tight.max_splits) * qh * hd;
              const auto ml_elems = static_cast<size_t>(tight.max_splits) * qh * 2u;
              auto bacc = dev.create_buffer<float>(acc_elems);
              auto bml = dev.create_buffer<float>(ml_elems);
              auto bout = dev.create_buffer<float>(qn);
              stream << sh_tight(bq, bk, bv, bscale_dummy.view(), bscale_dummy.view(), bacc, bml, kv_len, scale, capacity)
                            .dispatch(qwen_dispatch_threads(*plan_tight.value))
                     << sh_tight_combine(bacc, bml, bg, bout, kv_len, capacity)
                            .dispatch(qwen_dispatch_threads(*plan_tight_c.value));
              auto d_out = download(stream, bout, qn);
              luisa::vector<float> ref;
              check(ref_qwen_attn_decode(q, gate, k_cache, v_cache, qh, kvh, hd, scale, kv_len, ref).ok(),
                    "split empty: oracle");
              check_close(d_out, ref, kSplitAtol, kSplitRtol,
                          "split empty: combine skips the 3 empty splits of kv_len=100");
              // The empty splits must publish l == 0 (that is what the combine skips
              // on): scan the m/l buffer of the empty split range.
              const auto d_ml = download(stream, bml, ml_elems);
              bool empty_marked = true;
              for (uint32_t s = splits; s < tight.max_splits; s++) {
                  for (uint32_t h = 0u; h < qh; h++) {
                      const auto l = d_ml[(s * qh + h) * 2u + 1u];
                      if (!(l == 0.0f)) { empty_marked = false; }
                  }
              }
              check(empty_marked, "split empty: empty splits publish l == 0");
              bool live_positive = true;
              for (uint32_t s = 0u; s < splits; s++) {
                  for (uint32_t h = 0u; h < qh; h++) {
                      const auto l = d_ml[(s * qh + h) * 2u + 1u];
                      if (!(l > 0.0f) || !std::isfinite(l)) { live_positive = false; }
                  }
              }
              check(live_positive, "split empty: live splits publish a finite positive l");
          }
      }
      // ---- combine finiteness + gate ----------------------------------------
      {
          // All-empty partials (m = 0, l = 0 everywhere): the combine must write a
          // finite value instead of 0/0 = NaN.
          const luisa::vector<float> zero_acc(partial_acc_elems, 0.0f);
          const luisa::vector<float> zero_ml(partial_ml_elems, 0.0f);
          auto bz_acc = upload(dev, stream, zero_acc);
          auto bz_ml = upload(dev, stream, zero_ml);
          auto bout = dev.create_buffer<float>(qn);
          stream << sh_combine(bz_acc, bz_ml, bg, bout, 8u, capacity)
                        .dispatch(qwen_dispatch_threads(*combine_plan.value));
          auto d_out = download(stream, bout, qn);
          bool finite = true, all_zero = true;
          for (auto x : d_out) {
              if (!std::isfinite(x)) { finite = false; }
              if (x != 0.0f) { all_zero = false; }
          }
          check(finite && all_zero, "split combine: an all-empty reduction is a finite zero");
          // Gate: the same partials through two gate buffers must differ exactly by
          // the sigmoid ratio, which is what "the combine applies the gate" means.
          auto bpartial_acc = upload(dev, stream, luisa::vector<float>(partial_acc_elems, 0.25f));
          auto bpartial_ml = upload(dev, stream, luisa::vector<float>(partial_ml_elems, 1.0f));
          luisa::vector<float> gate_a(qn), gate_b(qn);
          for (uint32_t h = 0u; h < qh; h++) {
              for (uint32_t d = 0u; d < hd; d++) {
                  gate_a[h * hd + d] = (d % 2u) ? 1.5f : -1.5f;
                  gate_b[h * hd + d] = 0.0f;
              }
          }
          auto bga = upload(dev, stream, gate_a);
          auto bgb = upload(dev, stream, gate_b);
          auto bout_a = dev.create_buffer<float>(qn);
          auto bout_b = dev.create_buffer<float>(qn);
          stream << sh_combine(bpartial_acc, bpartial_ml, bga, bout_a, 8u, capacity)
                        .dispatch(qwen_dispatch_threads(*combine_plan.value))
                 << sh_combine(bpartial_acc, bpartial_ml, bgb, bout_b, 8u, capacity)
                        .dispatch(qwen_dispatch_threads(*combine_plan.value));
          auto da = download(stream, bout_a, qn);
          auto db = download(stream, bout_b, qn);
          bool gate_ok = true;
          for (uint32_t h = 0u; h < qh && gate_ok; h++) {
              for (uint32_t d = 0u; d < hd; d++) {
                  const auto sig_a = 1.0f / (1.0f + std::exp(-gate_a[h * hd + d]));
                  const auto sig_b = 1.0f / (1.0f + std::exp(-gate_b[h * hd + d]));
                  const auto want = db[h * hd + d] * sig_a / sig_b;
                  if (!close(da[h * hd + d], want, 1e-6f, 1e-5f)) {
                      gate_ok = false;
                      break;
                  }
              }
          }
          check(gate_ok, "split combine: output is scaled by sigmoid(gate) exactly");
      }
  }


// ---------------------------------------------------------------------------
// Register-cached one-pass RMSNorm model specialization (opt_plan/09 N3.1).
// hidden = 2560 at block 256 (10 elements/thread) and 128 (20), all three
// epilogues, checked against the independent CPU oracle AND against the generic
// wide kernel on the same inputs. The specialization must be numerically
// identical (same per-thread element order, same warp-partial order, same
// epilogue expression), and it must refuse a geometry it was not compiled for.
// ---------------------------------------------------------------------------
void test_rmsnorm_reg(Device &dev, Stream &stream, Rng &rng, Buffer<float> &dummy_f) {
    const uint32_t dim = qwen35_rmsnorm_model_dim;
    const uint32_t rows = 3u;
    const size_t count = size_t(rows) * dim;
    luisa::vector<float> x(count), w(dim), residual(count);
    rng.fill(x, -2.f, 2.f);
    rng.fill(w, 0.5f, 1.5f);
    rng.fill(residual, -1.f, 1.f);
    auto bx = upload(dev, stream, x);
    auto bw = upload(dev, stream, w);
    auto br = upload(dev, stream, residual);
    auto by = dev.create_buffer<float>(count);
    auto by_ref = dev.create_buffer<float>(count);
    auto bs = dev.create_buffer<float>(count);
    auto bs_ref = dev.create_buffer<float>(count);
    for (uint32_t block : {128u, 256u}) {
        for (auto epi : {QwenNormEpilogue::plain, QwenNormEpilogue::residual,
                         QwenNormEpilogue::snapshot}) {
            const bool has_residual = epi == QwenNormEpilogue::residual;
            const bool has_sum = epi != QwenNormEpilogue::plain;
            const auto bres = has_residual ? br : dummy_f;
            const auto bsum = has_sum ? bs : dummy_f;
            auto sh = dev.compile(make_qwen35_rmsnorm_reg_kernel(epi, dim, block));
            auto sh_wide = dev.compile(make_qwen35_rmsnorm_kernel(epi, block));
            auto plan = plan_qwen35_rmsnorm_reg_dispatch(rows, dim, block);
            check(plan.ok(), "register norm plan");
            if (!plan.ok()) { return; }
            auto plan_wide = plan_qwen35_rmsnorm_dispatch(rows, dim, block);
            check(plan_wide.ok(), "wide norm plan (reference)");
            if (!plan_wide.ok()) { return; }
            stream << sh(bx, bw, bres, by, bsum, rows, dim, 1e-6f)
                          .dispatch(qwen_dispatch_threads(*plan.value));
            stream << sh_wide(bx, bw, bres, by_ref,
                              has_sum ? bs_ref : dummy_f, rows, dim, 1e-6f)
                          .dispatch(qwen_dispatch_threads(*plan_wide.value));
            auto y = download(stream, by, count);
            auto y_wide = download(stream, by_ref, count);
            luisa::vector<float> ref, sum;
            if (epi == QwenNormEpilogue::snapshot) {
                check(ref_qwen_rmsnorm_snapshot(x, w, rows, dim, 1e-6f, ref, sum).ok(),
                      "register norm snapshot oracle");
                check_exact(download(stream, bs, count), x, "register norm snapshot exact copy");
            } else {
                check(ref_qwen_rmsnorm(x, w, residual, has_residual, rows, dim,
                                       1e-6f, ref, sum).ok(),
                      "register norm oracle");
                if (has_residual) {
                    check_close(download(stream, bs, count), sum, 1e-6f, 1e-5f,
                                "register norm residual sum");
                }
            }
            char name[96];
            std::snprintf(name, sizeof(name), "register rmsnorm plain/echo block %u", block);
            check_close(y, ref, 2e-5f, 2e-4f, name);
            // Same per-thread order and the same 4-partial block reduction. CUDA
            // reproduces the wide kernel bit for bit; DX/VK are allowed to contract
            // the epilogue multiply differently in the two code shapes (the
            // register form has the value and the weight in registers), so the
            // cross-kernel check uses the FP32 last-bit envelope while the CPU
            // oracle above stays the numerical gate.
            std::snprintf(name, sizeof(name), "register rmsnorm == wide kernel block %u", block);
            check_close(y, y_wide, 1e-6f, 1e-6f, name);
            if (has_sum) {
                std::snprintf(name, sizeof(name), "register norm sum_out == wide block %u", block);
                check_close(download(stream, bs, count), download(stream, bs_ref, count),
                            0.0f, 0.0f, name);
            }
        }
    }
    // A geometry the kernel was not compiled for is a guarded no-op.
    {
        auto sh = dev.compile(make_qwen35_rmsnorm_reg_kernel(QwenNormEpilogue::plain, dim, 256u));
        luisa::vector<float> sentinel(count, 7.25f);
        stream << by.copy_from(luisa::span<const float>{sentinel}) << synchronize();
        auto plan = plan_qwen35_rmsnorm_reg_dispatch(rows, dim, 256u);
        check(plan.ok(), "register norm guard plan");
        if (plan.ok()) {
            stream << sh(bx, bw, dummy_f, by, dummy_f, rows, dim + 256u, 1e-6f)
                          .dispatch(qwen_dispatch_threads(*plan.value));
            check_exact(download(stream, by, count), sentinel,
                        "register norm refuses a mismatched dim");
        }
        // The planner rejects a dim that does not divide the block.
        check(!plan_qwen35_rmsnorm_reg_dispatch(rows, 2559u, 256u).ok(),
              "register norm plan rejects ragged dim");
    }
}

// ---------------------------------------------------------------------------
// GDN state-column model specialization (opt_plan/09 V4): compile-time
// head_dim = 128 removes every runtime index of the lane-owned state column in
// BOTH the WY/UT chunk-64 prefill kernel and the per-token recurrent fallback.
// Checked on the model geometry (qk_heads 16, v_heads 32, head_dim 128) at every
// chunk boundary the plan lists, against the independent per-token FP64 oracle
// and against the generic realizations (which must agree bit for bit: the
// specialization only makes extents constant).
// ---------------------------------------------------------------------------
void test_gdn_model_shape(Device &dev, Stream &stream, Rng &rng) {
    constexpr uint32_t QK = 16u, H = 32u, HD = qwen35_gdn_model_head_dim, MAX_T = 129u;
    constexpr uint32_t QKN = QK * HD, VN = H * HD, C = 2u * QKN + VN;
    luisa::vector<float> q(size_t(MAX_T) * QKN), k(q.size()), conv(size_t(MAX_T) * C);
    rng.fill(q, -1.0f, 1.0f);
    rng.fill(k, -1.0f, 1.0f);
    rng.fill(conv, -0.5f, 0.5f);
    // Match the engine's normalized q/k contract, including q's 1/sqrt(HD).
    for (uint32_t t = 0; t < MAX_T; t++) {
        for (uint32_t h = 0; h < QK; h++) {
            const size_t base = (size_t(t) * QK + h) * HD;
            double qs = 0.0, ks = 0.0;
            for (uint32_t d = 0; d < HD; d++) {
                qs += double(q[base + d]) * q[base + d];
                ks += double(k[base + d]) * k[base + d];
            }
            const float qi = float(1.0 / std::sqrt(qs + 1e-6) / std::sqrt(double(HD)));
            const float ki = float(1.0 / std::sqrt(ks + 1e-6));
            for (uint32_t d = 0; d < HD; d++) { q[base + d] *= qi; k[base + d] *= ki; }
        }
    }
    luisa::vector<float> beta(size_t(MAX_T) * H), g(beta.size());
    rng.fill(beta, 0.05f, 0.4f);
    rng.fill(g, -0.08f, -0.001f);
    luisa::vector<float> initial_state(size_t(H) * HD * HD);
    rng.fill(initial_state, -0.05f, 0.05f);
    auto sh_generic = dev.compile(make_qwen_gdn_chunk_kernel());
    auto sh_model = dev.compile(make_qwen35_gdn_chunk_kernel(HD));
    auto plan = plan_qwen_gdn_chunk_dispatch(H);
    check(plan.ok(), "model-shape chunk dispatch plan");
    if (!plan.ok()) { return; }
    for (uint32_t tokens : {1u, 63u, 64u, 65u, 127u, 128u, 129u}) {
        const size_t qk_count = size_t(tokens) * QKN;
        const size_t conv_count = size_t(tokens) * C;
        const size_t gate_count = size_t(tokens) * H;
        const size_t out_count = gate_count * HD;
        luisa::vector<float> qt(q.begin(), q.begin() + qk_count);
        luisa::vector<float> kt(k.begin(), k.begin() + qk_count);
        luisa::vector<float> ct(conv.begin(), conv.begin() + conv_count);
        luisa::vector<float> bt(beta.begin(), beta.begin() + gate_count);
        luisa::vector<float> gt(g.begin(), g.begin() + gate_count);
        if (tokens == 64u) { std::fill(gt.begin(), gt.end(), 0.0f); }
        if (tokens == 129u) { std::fill(gt.begin(), gt.end(), -20.0f); }
        luisa::vector<float> reference_state = initial_state, reference_out;
        check(ref_qwen_gdn_recurrent(qt, kt, ct, bt, gt, reference_state, tokens,
                                     H, HD, QK, reference_out).ok(),
              "model-shape recurrent oracle");
        auto bq = upload(dev, stream, qt);
        auto bk = upload(dev, stream, kt);
        auto bc = upload(dev, stream, ct);
        auto bb = upload(dev, stream, bt);
        auto bg = upload(dev, stream, gt);
        auto bs = upload(dev, stream, initial_state);
        auto bs_ref = upload(dev, stream, initial_state);
        auto bw = dev.create_buffer<float>(out_count);
        auto bvn = dev.create_buffer<float>(out_count);
        auto bo = dev.create_buffer<float>(out_count);
        auto bo_ref = dev.create_buffer<float>(out_count);
        stream << sh_model(bq, bk, bc, bb, bg, bs, bw, bvn, bo, tokens, H, HD, QK, 2u * QKN)
                          .dispatch(qwen_dispatch_threads(*plan.value));
        stream << sh_generic(bq, bk, bc, bb, bg, bs_ref, bw, bvn, bo_ref, tokens, H, HD, QK, 2u * QKN)
                          .dispatch(qwen_dispatch_threads(*plan.value));
        auto actual_out = download(stream, bo, out_count);
        auto actual_state = download(stream, bs, initial_state.size());
        auto generic_out = download(stream, bo_ref, out_count);
        auto generic_state = download(stream, bs_ref, initial_state.size());
        char name[112];
        std::snprintf(name, sizeof(name), "model WY/UT T=%u output == recurrent", tokens);
        check_close(actual_out, reference_out, 2e-4f, 2e-3f, name);
        std::snprintf(name, sizeof(name), "model WY/UT T=%u state == recurrent", tokens);
        check_close(actual_state, reference_state, 2e-4f, 2e-3f, name);
        // Compile-time extents must not change the arithmetic: CUDA agrees bit for
        // bit, DX/VK may contract the same expression differently once the loop is
        // unrolled, so the cross-kernel check is the FP32 last-bit envelope and the
        // per-token FP64 oracle above is the numerical gate.
        std::snprintf(name, sizeof(name), "model WY/UT T=%u output == generic chunk", tokens);
        check_close(actual_out, generic_out, 1e-6f, 1e-6f, name);
        std::snprintf(name, sizeof(name), "model WY/UT T=%u state == generic chunk", tokens);
        check_close(actual_state, generic_state, 1e-6f, 1e-6f, name);
    }
    // Per-token recurrent fallback: the same specialization on the exact
    // recurrence (used by --gdn-recurrent, MTP verify/replay and T>1 prefill when
    // the WY/UT path is disabled).
    {
        auto sh_gen = dev.compile(make_qwen_gdn_recurrent_kernel());
        auto sh_mod = dev.compile(make_qwen35_gdn_recurrent_kernel(HD));
        auto rplan = plan_qwen_gdn_recurrent_dispatch(H);
        check(rplan.ok(), "model-shape recurrent plan");
        if (rplan.ok()) {
            for (uint32_t tokens : {1u, 2u, 3u}) {
                const size_t qk_count = size_t(tokens) * QKN;
                const size_t conv_count = size_t(tokens) * C;
                const size_t gate_count = size_t(tokens) * H;
                const size_t out_count = gate_count * HD;
                luisa::vector<float> qt(q.begin(), q.begin() + qk_count);
                luisa::vector<float> kt(k.begin(), k.begin() + qk_count);
                luisa::vector<float> ct(conv.begin(), conv.begin() + conv_count);
                luisa::vector<float> bt(beta.begin(), beta.begin() + gate_count);
                luisa::vector<float> gt(g.begin(), g.begin() + gate_count);
                luisa::vector<float> reference_state = initial_state, reference_out;
                check(ref_qwen_gdn_recurrent(qt, kt, ct, bt, gt, reference_state, tokens,
                                             H, HD, QK, reference_out).ok(),
                      "model-shape recurrent oracle (exact path)");
                auto bq = upload(dev, stream, qt);
                auto bk = upload(dev, stream, kt);
                auto bc = upload(dev, stream, ct);
                auto bb = upload(dev, stream, bt);
                auto bg = upload(dev, stream, gt);
                auto bs = upload(dev, stream, initial_state);
                auto bs_ref = upload(dev, stream, initial_state);
                auto bo = dev.create_buffer<float>(out_count);
                auto bo_ref = dev.create_buffer<float>(out_count);
                stream << sh_mod(bq, bk, bc, bb, bg, bs, bo, tokens, H, HD, QK, 2u * QKN)
                              .dispatch(qwen_dispatch_threads(*rplan.value));
                stream << sh_gen(bq, bk, bc, bb, bg, bs_ref, bo_ref, tokens, H, HD, QK, 2u * QKN)
                              .dispatch(qwen_dispatch_threads(*rplan.value));
                auto actual_out = download(stream, bo, out_count);
                auto actual_state = download(stream, bs, initial_state.size());
                char name[112];
                std::snprintf(name, sizeof(name), "model recurrent T=%u output == oracle", tokens);
                check_close(actual_out, reference_out, 2e-4f, 2e-3f, name);
                std::snprintf(name, sizeof(name), "model recurrent T=%u state == oracle", tokens);
                check_close(actual_state, reference_state, 2e-4f, 2e-3f, name);
                std::snprintf(name, sizeof(name), "model recurrent T=%u output == generic", tokens);
                check_close(actual_out, download(stream, bo_ref, out_count), 1e-6f, 1e-6f, name);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused decode GDN conv+SiLU+ring update+QK L2 norm (opt_plan/09 N3.3).
// The single fused launch must reproduce the reference pair (conv_silu followed by
// qk_l2norm) element for element: the normalized Q (with its 1/sqrt(hd) scale) and
// K heads, the V channels of the conv output that the recurrent kernel consumes,
// and the whole conv ring -- at several absolute positions, so both ring history
// and slot rotation are exercised. Both realizations are checked: the host-scalar
// one and the device-state one (which must agree bit for bit, the same contract as
// every other per-token consumer).
// ---------------------------------------------------------------------------
void test_gdn_decode_fused(Device &dev, Stream &stream, Rng &rng) {
    constexpr uint32_t QKH = 16u, HD = qwen35_gdn_model_head_dim, VH = 32u;
    constexpr uint32_t QKN = QKH * HD, VN = VH * HD, C = 2u * QKN + VN, K = 4u, rw = K - 1u;
    const float eps = 1e-6f;
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(HD));
    luisa::vector<float> src(C), weight(size_t(K) * C), ring0(size_t(C) * rw);
    rng.fill(src, -1.f, 1.f);
    rng.fill(weight, -0.5f, 0.5f);
    rng.fill(ring0, -0.5f, 0.5f);
    auto bsrc = upload(dev, stream, src);
    auto bw = upload(dev, stream, weight);
    auto bq_ref = dev.create_buffer<float>(QKN);
    auto bk_ref = dev.create_buffer<float>(QKN);
    auto bconv_ref = dev.create_buffer<float>(C);
    auto bring_ref = dev.create_buffer<float>(C * rw);
    auto bq_f = dev.create_buffer<float>(QKN);
    auto bk_f = dev.create_buffer<float>(QKN);
    auto bconv_f = dev.create_buffer<float>(C);
    auto bring_f = dev.create_buffer<float>(C * rw);
    auto bq_d = dev.create_buffer<float>(QKN);
    auto bk_d = dev.create_buffer<float>(QKN);
    auto bconv_d = dev.create_buffer<float>(C);
    auto bring_d = dev.create_buffer<float>(C * rw);
    auto sh_conv = dev.compile(make_qwen_gdn_conv_silu_kernel());
    auto sh_norm = dev.compile(make_qwen_gdn_qk_l2norm_kernel());
    auto sh_fused = dev.compile(make_qwen35_gdn_decode_fused_kernel());
    auto sh_fused_dev = dev.compile(make_qwen35_gdn_decode_fused_dev_kernel());
    auto plan_conv = plan_qwen_gdn_conv_dispatch(C);
    auto plan_norm = plan_qwen_gdn_qk_norm_dispatch(1u, QKH);
    auto plan_fused = plan_qwen35_gdn_decode_fused_dispatch(QKH, HD, VN);
    check(plan_conv.ok() && plan_norm.ok(), "fused gdn decode reference plans");
    check(plan_fused.ok(), "fused gdn decode plan");
    if (!plan_conv.ok() || !plan_norm.ok() || !plan_fused.ok()) { return; }
    // One 128-thread block covers 4 channel groups (4 warps of 128 channels).
    check(plan_fused.value->work_items == C / (128u * 4u),
          "fused gdn decode covers every channel group");
    check(!plan_qwen35_gdn_decode_fused_dispatch(QKH, 64u, VN).ok(),
          "fused gdn decode rejects a non-model head_dim");
    check(!plan_qwen35_gdn_decode_fused_dispatch(QKH, HD, 0u).ok(),
          "fused gdn decode rejects an empty value channel count");
    check(!validate_qwen35_gdn_decode_fused_inputs(2u, QKH, HD, VN, K, 0u).ok(),
          "fused gdn decode is single-token only");
    const auto upload_state = [&](uint32_t position) {
        luisa::vector<uint32_t> words(qwen_dec_state_count, 0u);
        words[qwen_dec_state_position] = position;
        words[qwen_dec_state_kv_len] = position + 1u;
        return upload(dev, stream, words);
    };
    for (uint32_t pos : {0u, 1u, 2u, 3u, 5u}) {
        stream << bring_ref.copy_from(luisa::span<const float>{ring0})
               << bring_f.copy_from(luisa::span<const float>{ring0})
               << bring_d.copy_from(luisa::span<const float>{ring0}) << synchronize();
        auto bstate = upload_state(pos);
        stream << sh_conv(bsrc, bw, bring_ref, bconv_ref, 1u, C, K, pos)
                      .dispatch(qwen_dispatch_threads(*plan_conv.value))
               << sh_norm(bconv_ref, bq_ref, bk_ref, 1u, QKH, HD, VN, eps, q_scale, QKH)
                      .dispatch(qwen_dispatch_threads(*plan_norm.value))
               << sh_fused(bsrc, bw, bring_f, bq_f, bk_f, bconv_f, QKH, HD, VN, K, eps, q_scale, pos)
                      .dispatch(qwen_dispatch_threads(*plan_fused.value))
               << sh_fused_dev(bsrc, bw, bring_d, bq_d, bk_d, bconv_d, bstate,
                               QKH, HD, VN, K, eps, q_scale)
                      .dispatch(qwen_dispatch_threads(*plan_fused.value));
        auto q_ref = download(stream, bq_ref, QKN);
        auto k_ref = download(stream, bk_ref, QKN);
        auto conv_ref = download(stream, bconv_ref, C);
        auto ring_ref = download(stream, bring_ref, C * rw);
        stream << synchronize();
        char name[112];
        // The fused launch computes `conv * inv * q_scale` from registers while the
        // reference pair reads the stored conv value back: CUDA agrees bit for bit,
        // DX/VK may contract the two multiplies differently (FP32 last-bit envelope;
        // the CPU oracle checks are the numerical gate).
        std::snprintf(name, sizeof(name), "fused gdn decode q pos=%u", pos);
        check_close(download(stream, bq_f, QKN), q_ref, 1e-6f, 1e-6f, name);
        std::snprintf(name, sizeof(name), "fused gdn decode k pos=%u", pos);
        check_close(download(stream, bk_f, QKN), k_ref, 1e-6f, 1e-6f, name);
        // Only the V channel range is produced by the fused launch; the Q/K range of
        // the conv output is precisely the round trip it removes.
        luisa::vector<float> v_ref(conv_ref.begin() + 2 * QKN, conv_ref.end());
        std::snprintf(name, sizeof(name), "fused gdn decode v channels pos=%u", pos);
        check_exact([&] {
            auto all = download(stream, bconv_f, C);
            return luisa::vector<float>(all.begin() + 2 * QKN, all.end());
        }(), v_ref, name);
        std::snprintf(name, sizeof(name), "fused gdn decode ring pos=%u", pos);
        check_exact(download(stream, bring_f, C * rw), ring_ref, name);
        // Device-state realization: same numerics, position read from _dev_state.
        std::snprintf(name, sizeof(name), "fused gdn decode dev q pos=%u", pos);
        check_close(download(stream, bq_d, QKN), q_ref, 1e-6f, 1e-6f, name);
        std::snprintf(name, sizeof(name), "fused gdn decode dev k pos=%u", pos);
        check_close(download(stream, bk_d, QKN), k_ref, 1e-6f, 1e-6f, name);
        std::snprintf(name, sizeof(name), "fused gdn decode dev ring pos=%u", pos);
        check_exact(download(stream, bring_d, C * rw), ring_ref, name);
        std::snprintf(name, sizeof(name), "fused gdn decode dev v channels pos=%u", pos);
        check_exact([&] {
            auto all = download(stream, bconv_d, C);
            return luisa::vector<float>(all.begin() + 2 * QKN, all.end());
        }(), v_ref, name);
    }
}

// ---------------------------------------------------------------------------
// Vectorized 128-bit decode realizations (opt_plan/09 N2).
//
// Four independent links:
//  1. BYTE-ACCURATE FP8 DECODE. N = 256 rows, K = 512, every element of row n is
//     code n and every scale is 1.0; x is e_0, so the dot of row n is exactly
//     decode(n) whatever the reduction order. The packed (LUT-free, branchless)
//     kernel is compared against an INDEPENDENT host E4M3 oracle written in
//     double precision, and against the 32-bit LUT realization on the same
//     buffers. All 256 codes: subnormals, signed zero, +/-normal, and the two
//     NaN codes (0x7F/0xFF) which must both be NaN.
//  2. The shared double-accumulation CPU oracle `ref_qwen_matmul` on a small N
//     (37) at the model K, both formats and both epilogues.
//  3. Model shapes (8192x2560, 2560x4096, 9216x2560, 2560x9216) against the
//     32-bit decode kernel on the SAME packed buffers -- the differential the
//     engine's realization switch relies on. The accumulation partition differs,
//     so the gate is the documented FP32 relative envelope.
//  4. TAIL REJECTION: every geometry the packed contract cannot serve is a
//     planning FAILURE, never a silent fallback.
// ---------------------------------------------------------------------------
// Finite, non-NaN packed weight word (the model-shape differentials must not be
// swamped by NaN codes: a single 0x7F/0xFF byte would poison the whole row sum).
uint32_t finite_packed_word(QwenWeightFormat format, Rng &rng) {
    const auto byte = [&]() {
        // exponent in [4,7] and any mantissa -> |value| in [2^-3, 1.875], plus sign.
        return (0x20u | (rng.next() & 0x1Fu)) | (((rng.next() >> 7u) & 1u) << 7u);
    };
    if (format == QwenWeightFormat::fp8_e4m3) {
        return byte() | (byte() << 8u) | (byte() << 16u) | (byte() << 24u);
    }
    const auto half = [&]() {
        return 0x3800u | (rng.next() & 0xFFu) | (((rng.next() >> 8u) & 1u) << 15u);
    };
    return half() | (half() << 16u);
}

// Independent E4M3 -> double oracle (shares no code with dsl_ops).
double host_e4m3(uint32_t code) noexcept {
    const uint32_t sign = (code >> 7u) & 1u;
    const uint32_t e = (code >> 3u) & 0xFu;
    const uint32_t m = code & 0x7u;
    double v = 0.0;
    if (e == 0u) {
        v = std::ldexp(static_cast<double>(m), -9);// m * 2^-9
    } else {
        v = (1.0 + static_cast<double>(m) / 8.0) * std::ldexp(1.0, static_cast<int>(e) - 7);
    }
    return sign != 0u ? -v : v;
}

void test_decode_packed_e4m3_codes(Device &dev, Stream &stream, Rng &rng, Buffer<float> &dummy_f) {
    (void)rng;
    constexpr uint32_t N = 256u, K = 512u, group = 64u;
    const uint32_t words_per_row = K / 4u;
    luisa::vector<uint32_t> words(size_t(N) * words_per_row);
    for (uint32_t n = 0u; n < N; n++) {
        const uint32_t w = n | (n << 8u) | (n << 16u) | (n << 24u);
        for (uint32_t i = 0; i < words_per_row; i++) { words[size_t(n) * words_per_row + i] = w; }
    }
    luisa::vector<float> scales(size_t(N) * (K / group), 1.0f);
    luisa::vector<float> x(K, 0.0f);
    x[0] = 1.0f;
    auto bw = upload(dev, stream, words);
    auto bs = upload(dev, stream, scales);
    auto bx = upload(dev, stream, x);
    auto by = dev.create_buffer<float>(N);
    const auto elems = qwen_packed_vec_elements(QwenWeightFormat::fp8_e4m3);
    auto packed_plan = plan_qwen35_decode_gemv_packed_dispatch(N, K, QwenWeightFormat::fp8_e4m3, group);
    check(packed_plan.ok(), "packed decode e4m3 plan");
    if (!packed_plan.ok()) { return; }
    auto packed = dev.compile(make_qwen35_decode_gemv_packed_kernel(
        QwenWeightFormat::fp8_e4m3, QwenMatmulEpilogue::none));
    stream << packed(bx.view().as<float4>(), bw.view().as<uint4>(), bs, dummy_f, by,
                     N, K, group, K / elems)
                  .dispatch(qwen_dispatch_threads(*packed_plan.value));
    auto got_packed = download(stream, by, N);
    auto ref_plan = plan_qwen35_decode_gemv_dispatch(N, K, QwenWeightFormat::fp8_e4m3);
    check(ref_plan.ok(), "reference decode e4m3 plan");
    auto ref = dev.compile(make_qwen35_decode_gemv_kernel(
        QwenWeightFormat::fp8_e4m3, QwenMatmulEpilogue::none));
    stream << ref(bx, bw, bs, dummy_f, by, 1u, N, K, group)
                  .dispatch(qwen_dispatch_threads(*ref_plan.value));
    auto got_ref = download(stream, by, N);
    auto exact = true;
    auto ref_ok = true;
    for (uint32_t n = 0u; n < N; n++) {
        const bool nan = (n & 0x7Fu) == 0x7Fu;
        const auto want = static_cast<float>(host_e4m3(n));
        if (nan) {
            if (!std::isnan(got_packed[n])) { exact = false; }
            if (!std::isnan(got_ref[n])) { ref_ok = false; }
            continue;
        }
        // Exact: one product with a scale of exactly 1.0, every other term an
        // exact zero, so no reduction order can change the result.
        if (!(got_packed[n] == want)) { exact = false; }
        if (!(got_ref[n] == want)) { ref_ok = false; }
    }
    check(exact, "packed 128-bit E4M3 decode is byte-accurate for all 256 codes");
    check(ref_ok, "32-bit LUT decode agrees with the independent E4M3 oracle");
    // Spot anchors so a systematically wrong oracle cannot hide.
    const auto at = [&](uint32_t code) { return got_packed[code]; };
    // Signed zero: the DECODE preserves the sign (it returns -0.0 for 0x80), but a
    // GEMV cannot observe it -- (+0.0) + (-0.0) is +0.0, and a0 starts at +0.0. The
    // probe therefore pins (a) both zero codes decode to zero and (b) the sign of a
    // nonzero negative code survives the whole warp reduction.
    check(at(0x00u) == 0.0f && at(0x80u) == 0.0f && std::signbit(at(0xB8u)) && at(0xB8u) == -1.0f,
          "E4M3 decode maps 0x00/0x80 to zero and keeps the sign of a negative code");
    check(at(0x30u) == 0.5f && at(0x38u) == 1.0f && at(0x40u) == 2.0f,
          "E4M3 decode normal anchors");
    check(at(0x01u) == std::ldexp(1.0f, -9) && at(0x07u) == 7.0f * std::ldexp(1.0f, -9),
          "E4M3 decode subnormal anchors");
    check(at(0x7Eu) == 448.0f, "E4M3 decode maximum finite value");
    // A non-unit power-of-two scale must rescale exactly.
    luisa::vector<float> half_scales(size_t(N) * (K / group), 0.5f);
    auto bs_half = upload(dev, stream, half_scales);
    stream << packed(bx.view().as<float4>(), bw.view().as<uint4>(), bs_half, dummy_f, by,
                     N, K, group, K / elems)
                  .dispatch(qwen_dispatch_threads(*packed_plan.value));
    auto half = download(stream, by, N);
    check(half[0x38u] == 0.5f && half[0x40u] == 1.0f,
          "packed decode applies the per-uint4 FP8 scale");
}

void test_decode_packed_shapes(Device &dev, Stream &stream, Rng &rng, Buffer<float> &dummy_f) {
    for (auto format : {QwenWeightFormat::bf16, QwenWeightFormat::fp8_e4m3}) {
        const auto elems = qwen_packed_vec_elements(format);
        const auto rel = format == QwenWeightFormat::bf16 ? 2e-2f : 0.4f;
        const auto tag = [&](const char *what) {
            return luisa::string{what} + (format == QwenWeightFormat::bf16 ? " bf16" : " fp8");
        };
        // (1) shared CPU oracle at a small N, both epilogues.
        {
            constexpr uint32_t N = 37u, K = 2560u;
            luisa::vector<float> w(size_t(K) * N), x(K), residual(N);
            rng.fill(w, -1.f, 1.f);
            rng.fill(x, -1.f, 1.f);
            rng.fill(residual, -0.5f, 0.5f);
            luisa::vector<uint32_t> words;
            luisa::vector<float> scales;
            check(host_pack_matrix(format, w, K, N, 64u, words, scales).ok(),
                  tag("packed decode host pack").c_str());
            auto bw = upload(dev, stream, words);
            auto bs = upload(dev, stream, scales.empty() ? luisa::vector<float>{0.f} : scales);
            auto bx = upload(dev, stream, x);
            auto br = upload(dev, stream, residual);
            auto by = dev.create_buffer<float>(N);
            auto plan = plan_qwen35_decode_gemv_packed_dispatch(N, K, format, 64u);
            check(plan.ok(), tag("packed decode plan (small N)").c_str());
            if (!plan.ok()) { continue; }
            for (bool with_residual : {false, true}) {
                auto sh = dev.compile(make_qwen35_decode_gemv_packed_kernel(
                    format, with_residual ? QwenMatmulEpilogue::residual : QwenMatmulEpilogue::none));
                stream << sh(bx.view().as<float4>(), bw.view().as<uint4>(), bs,
                             with_residual ? br : dummy_f, by, N, K, 64u, K / elems)
                              .dispatch(qwen_dispatch_threads(*plan.value));
                auto y = download(stream, by, N);
                luisa::vector<float> ref;
                check(ref_qwen_matmul(format, x, words, scales,
                                      with_residual ? residual : luisa::vector<float>{},
                                      with_residual, 1u, N, K, 64u, ref).ok(),
                      tag("packed decode oracle").c_str());
                check_close(y, ref, 3e-3f, rel,
                            with_residual ? tag("packed 128-bit decode GEMV residual").c_str()
                                          : tag("packed 128-bit decode GEMV plain").c_str());
            }
        }
        // (2) model shapes against the 32-bit decode kernel on the same buffers.
        for (uint32_t idx = 0u; idx < 4u; idx++) {
            const uint32_t n = idx == 0u ? 8192u : (idx == 1u ? 2560u : (idx == 2u ? 9216u : 2560u));
            const uint32_t k = idx == 0u ? 2560u : (idx == 1u ? 4096u : (idx == 2u ? 2560u : 9216u));
            const auto label = luisa::format("packed/decode {} n{} k{}",
                                             format == QwenWeightFormat::bf16 ? "bf16" : "fp8", n, k);
            const auto words_count = size_t(n) * (k / (format == QwenWeightFormat::bf16 ? 2u : 4u));
            luisa::vector<uint32_t> words(words_count);
            for (size_t i = 0; i < words.size(); i++) { words[i] = finite_packed_word(format, rng); }
            luisa::vector<float> scales;
            if (format == QwenWeightFormat::fp8_e4m3) {
                scales.assign(size_t(n) * (k / 64u), 0.03125f);
            }
            luisa::vector<float> x(k);
            rng.fill(x, -1.f, 1.f);
            auto bw = upload(dev, stream, words);
            auto bs = upload(dev, stream, scales.empty() ? luisa::vector<float>{0.f} : scales);
            auto bx = upload(dev, stream, x);
            auto by = dev.create_buffer<float>(n);
            auto packed_plan = plan_qwen35_decode_gemv_packed_dispatch(n, k, format, 64u);
            auto ref_plan = plan_qwen35_decode_gemv_dispatch(n, k, format);
            check(packed_plan.ok() && ref_plan.ok(), "packed decode model-shape plans", label);
            if (!packed_plan.ok() || !ref_plan.ok()) { continue; }
            auto packed = dev.compile(make_qwen35_decode_gemv_packed_kernel(
                format, QwenMatmulEpilogue::none));
            auto ref = dev.compile(make_qwen35_decode_gemv_kernel(
                format, QwenMatmulEpilogue::none));
            stream << packed(bx.view().as<float4>(), bw.view().as<uint4>(), bs, dummy_f, by,
                             n, k, 64u, k / elems)
                          .dispatch(qwen_dispatch_threads(*packed_plan.value));
            auto got = download(stream, by, n);
            stream << ref(bx, bw, bs, dummy_f, by, 1u, n, k, 64u)
                          .dispatch(qwen_dispatch_threads(*ref_plan.value));
            auto want = download(stream, by, n);
            check_close(got, want, 2e-3f, rel, label.c_str());
        }
    }
    // (3) tail rejection: every unsupported geometry is a planning failure.
    check(!plan_qwen35_decode_gemv_packed_dispatch(64u, 768u, QwenWeightFormat::fp8_e4m3, 64u).ok(),
          "packed decode rejects fp8 K % 512 != 0");
    check(!plan_qwen35_decode_gemv_packed_dispatch(64u, 384u, QwenWeightFormat::bf16, 64u).ok(),
          "packed decode rejects bf16 K % 256 != 0");
    check(!plan_qwen35_decode_gemv_packed_dispatch(64u, 512u, QwenWeightFormat::fp8_e4m3, 8u).ok(),
          "packed decode rejects an FP8 group that is not a multiple of 16");
    check(!plan_qwen35_decode_gemv_packed_dispatch(64u, 512u, QwenWeightFormat::fp8_e4m3, 96u).ok(),
          "packed decode rejects an FP8 group that does not divide K");
    check(plan_qwen35_decode_gemv_packed_dispatch(64u, 512u, QwenWeightFormat::fp8_e4m3, 32u).ok(),
          "packed decode accepts a 32-element FP8 group aligned to a uint4");
    check(plan_qwen35_decode_gemv_packed_dispatch(64u, 256u, QwenWeightFormat::bf16, 64u).ok(),
          "packed decode accepts bf16 K % 256 == 0");
    check(!plan_qwen_ffn_decode_packed_dispatch(64u, 768u, QwenWeightFormat::fp8_e4m3, 64u).ok(),
          "packed fused FFN shares the same tail rejection");
    check(!plan_qwen_gdn_decode_gates_packed_dispatch(32u, 768u, QwenWeightFormat::fp8_e4m3, 64u).ok(),
          "packed GDN gates share the same tail rejection");
}

void test_ffn_decode_packed(Device &dev, Stream &stream, Rng &rng) {
    for (auto format : {QwenWeightFormat::bf16, QwenWeightFormat::fp8_e4m3}) {
        const auto elems = qwen_packed_vec_elements(format);
        const auto rel = format == QwenWeightFormat::bf16 ? 2e-2f : 0.4f;
        // Small F against the shared oracle.
        {
            constexpr uint32_t F = 26u, K = 2560u;
            luisa::vector<float> gw(size_t(K) * F), uw(size_t(K) * F), x(K);
            rng.fill(gw, -1.f, 1.f);
            rng.fill(uw, -1.f, 1.f);
            rng.fill(x, -1.f, 1.f);
            luisa::vector<uint32_t> gwords, uwords;
            luisa::vector<float> gscales, uscales;
            check(host_pack_matrix(format, gw, K, F, 64u, gwords, gscales).ok(),
                  "packed ffn gate pack");
            check(host_pack_matrix(format, uw, K, F, 64u, uwords, uscales).ok(),
                  "packed ffn up pack");
            auto bgw = upload(dev, stream, gwords);
            auto buw = upload(dev, stream, uwords);
            auto bgs = upload(dev, stream, gscales.empty() ? luisa::vector<float>{0.f} : gscales);
            auto bus = upload(dev, stream, uscales.empty() ? luisa::vector<float>{0.f} : uscales);
            auto bx = upload(dev, stream, x);
            auto by = dev.create_buffer<float>(F);
            auto plan = plan_qwen_ffn_decode_packed_dispatch(F, K, format, 64u);
            check(plan.ok(), "packed ffn plan");
            if (!plan.ok()) { continue; }
            auto sh = dev.compile(make_qwen_ffn_decode_packed_kernel(format));
            stream << sh(bx.view().as<float4>(), bgw.view().as<uint4>(), bgs,
                         buw.view().as<uint4>(), bus, by, F, K, 64u, K / elems)
                          .dispatch(qwen_dispatch_threads(*plan.value));
            auto y = download(stream, by, F);
            luisa::vector<float> gref, uref, want(F);
            check(ref_qwen_matmul(format, x, gwords, gscales, {}, false, 1u, F, K, 64u, gref).ok(),
                  "packed ffn gate oracle");
            check(ref_qwen_matmul(format, x, uwords, uscales, {}, false, 1u, F, K, 64u, uref).ok(),
                  "packed ffn up oracle");
            for (uint32_t i = 0; i < F; i++) {
                want[i] = gref[i] / (1.0f + std::exp(-gref[i])) * uref[i];
            }
            check_close(y, want, 3e-3f, rel, "packed fused FFN decode against the CPU oracle");
        }
        // Model shape against the 32-bit fused FFN.
        {
            constexpr uint32_t F = 9216u, K = 2560u;
            const auto words_count = size_t(F) * (K / (format == QwenWeightFormat::bf16 ? 2u : 4u));
            luisa::vector<uint32_t> gwords(words_count), uwords(words_count);
            for (size_t i = 0; i < words_count; i++) {
                gwords[i] = finite_packed_word(format, rng);
                uwords[i] = finite_packed_word(format, rng);
            }
            luisa::vector<float> scales;
            if (format == QwenWeightFormat::fp8_e4m3) { scales.assign(size_t(F) * (K / 64u), 0.03125f); }
            luisa::vector<float> x(K);
            rng.fill(x, -1.f, 1.f);
            auto bgw = upload(dev, stream, gwords);
            auto buw = upload(dev, stream, uwords);
            auto bs = upload(dev, stream, scales.empty() ? luisa::vector<float>{0.f} : scales);
            auto bx = upload(dev, stream, x);
            auto by = dev.create_buffer<float>(F);
            auto packed_plan = plan_qwen_ffn_decode_packed_dispatch(F, K, format, 64u);
            auto ref_plan = plan_qwen_ffn_decode_dispatch(F, K, format, 64u);
            check(packed_plan.ok() && ref_plan.ok(), "packed ffn model plans");
            if (!packed_plan.ok() || !ref_plan.ok()) { continue; }
            auto packed = dev.compile(make_qwen_ffn_decode_packed_kernel(format));
            auto ref = dev.compile(make_qwen_ffn_decode_kernel(format));
            stream << packed(bx.view().as<float4>(), bgw.view().as<uint4>(), bs,
                             buw.view().as<uint4>(), bs, by, F, K, 64u, K / elems)
                          .dispatch(qwen_dispatch_threads(*packed_plan.value));
            auto got = download(stream, by, F);
            stream << ref(bx, bgw, bs, buw, bs, by, F, K, 64u)
                          .dispatch(qwen_dispatch_threads(*ref_plan.value));
            auto want = download(stream, by, F);
            check_close(got, want, 2e-3f, rel, "packed fused FFN decode equals the 32-bit kernel");
        }
    }
}

void test_gdn_decode_gates_packed(Device &dev, Stream &stream, Rng &rng) {
    for (auto format : {QwenWeightFormat::bf16, QwenWeightFormat::fp8_e4m3}) {
        constexpr uint32_t H = 32u, K = 2560u;
        const auto elems = qwen_packed_vec_elements(format);
        const auto rel = format == QwenWeightFormat::bf16 ? 2e-2f : 0.4f;
        luisa::vector<float> aw(size_t(K) * H), bwv(size_t(K) * H), x(K), ssm_a(H), dt(H);
        rng.fill(aw, -1.f, 1.f);
        rng.fill(bwv, -1.f, 1.f);
        rng.fill(x, -1.f, 1.f);
        rng.fill(ssm_a, -1.f, -0.01f);
        rng.fill(dt, -1.f, 1.f);
        luisa::vector<uint32_t> awords, bwords;
        luisa::vector<float> ascales, bscales;
        check(host_pack_matrix(format, aw, K, H, 64u, awords, ascales).ok(), "gdn packed gate pack");
        check(host_pack_matrix(format, bwv, K, H, 64u, bwords, bscales).ok(), "gdn packed gate pack b");
        auto baw = upload(dev, stream, awords);
        auto bbw = upload(dev, stream, bwords);
        auto bas = upload(dev, stream, ascales.empty() ? luisa::vector<float>{0.f} : ascales);
        auto bbs = upload(dev, stream, bscales.empty() ? luisa::vector<float>{0.f} : bscales);
        auto bx = upload(dev, stream, x);
        auto bssm = upload(dev, stream, ssm_a);
        auto bdt = upload(dev, stream, dt);
        auto beta_out = dev.create_buffer<float>(H);
        auto g_out = dev.create_buffer<float>(H);
        auto packed_plan = plan_qwen_gdn_decode_gates_packed_dispatch(H, K, format, 64u);
        auto ref_plan = plan_qwen_gdn_decode_gates_dispatch(H, K, format, 64u);
        check(packed_plan.ok() && ref_plan.ok(), "gdn packed gate plans");
        if (!packed_plan.ok() || !ref_plan.ok()) { continue; }
        auto packed = dev.compile(make_qwen_gdn_decode_gates_packed_kernel(format));
        auto ref = dev.compile(make_qwen_gdn_decode_gates_kernel(format));
        stream << packed(bx.view().as<float4>(), baw.view().as<uint4>(), bas,
                         bbw.view().as<uint4>(), bbs, bssm, bdt, beta_out, g_out,
                         K, H, 64u, 20.0f, K / elems)
                      .dispatch(qwen_dispatch_threads(*packed_plan.value));
        auto g_beta = download(stream, beta_out, H);
        auto g_g = download(stream, g_out, H);
        stream << ref(bx, baw, bas, bbw, bbs, bssm, bdt, beta_out, g_out, K, H, 64u, 20.0f)
                      .dispatch(qwen_dispatch_threads(*ref_plan.value));
        auto w_beta = download(stream, beta_out, H);
        auto w_g = download(stream, g_out, H);
        check_close(g_beta, w_beta, 2e-3f, rel, "packed GDN gate beta equals the 32-bit kernel");
        check_close(g_g, w_g, 2e-3f, rel, "packed GDN gate g equals the 32-bit kernel");
        luisa::vector<float> aref;
        check(ref_qwen_matmul(format, x, awords, ascales, {}, false, 1u, H, K, 64u, aref).ok(),
              "gdn packed gate oracle");
        luisa::vector<float> want_g(H);
        for (uint32_t h = 0; h < H; h++) {
            const auto av = aref[h] + dt[h];
            const auto sp = av > 20.0f ? av : std::log1p(std::exp(av));
            want_g[h] = ssm_a[h] * sp;
        }
        check_close(g_g, want_g, 3e-3f, rel, "packed GDN gate g against the CPU oracle");
    }
}

// ---------------------------------------------------------------------------
// Model-specialized decode attention prepare (opt_plan/09 N3.2).
// Differential against the GENERIC realization at several absolute positions, in
// both the host-scalar and the device-state form, for a single decode token and
// for a 3-token chunk. Compares every output buffer AND the KV cache bytes.
// ---------------------------------------------------------------------------
void test_attn_prepare_model(Device &dev, Stream &stream, Rng &rng) {
    QwenDslConfig cfg{};
    cfg.hidden = qwen35_rmsnorm_model_dim;
    cfg.ffn = 9216u;
    cfg.vocab = 151936u;
    cfg.q_heads = qwen35_attn_model_q_heads;
    cfg.kv_heads = qwen35_attn_model_kv_heads;
    cfg.attn_head_dim = qwen35_attn_model_head_dim;
    cfg.rope_dim = qwen35_attn_model_rope_dim;
    cfg.rope_theta = 10000000.0f;
    cfg.norm_eps = 1e-6f;
    const uint32_t qh = cfg.q_heads, kvh = cfg.kv_heads, hd = cfg.attn_head_dim;
    constexpr uint32_t capacity = 64u;
    for (uint32_t tokens : {1u, 3u}) {
        const auto qn = size_t(tokens) * qh * 2u * hd;
        const auto kn = size_t(tokens) * kvh * hd;
        luisa::vector<float> qp(qn), kp(kn), vp(kn), qnw(hd), knw(hd);
        rng.fill(qp, -1.f, 1.f);
        rng.fill(kp, -1.f, 1.f);
        rng.fill(vp, -1.f, 1.f);
        rng.fill(qnw, -1.f, 1.f);
        rng.fill(knw, -1.f, 1.f);
        auto bqp = upload(dev, stream, qp);
        auto bkp = upload(dev, stream, kp);
        auto bvp = upload(dev, stream, vp);
        auto bqnw = upload(dev, stream, qnw);
        auto bknw = upload(dev, stream, knw);
        auto q_out = dev.create_buffer<float>(size_t(tokens) * qh * hd);
        auto gate_out = dev.create_buffer<float>(size_t(tokens) * qh * hd);
        auto k_out = dev.create_buffer<float>(kn);
        auto v_out = dev.create_buffer<float>(kn);
        auto k_cache = dev.create_buffer<float>(size_t(capacity) * kvh * hd);
        auto v_cache = dev.create_buffer<float>(size_t(capacity) * kvh * hd);
        auto positions = dev.create_buffer<int32_t>(tokens);
        auto state = dev.create_buffer<uint32_t>(4u);
        auto generic_plan = plan_qwen_attn_prepare_dispatch(tokens, cfg);
        auto model_plan = plan_qwen35_attn_prepare_model_dispatch(tokens);
        check(generic_plan.ok() && model_plan.ok(), "attn prepare model plans");
        if (!generic_plan.ok() || !model_plan.ok()) { continue; }
        auto generic = dev.compile(make_qwen_attn_prepare_kernel(true, true));
        auto generic_dev = dev.compile(make_qwen_attn_prepare_dev_kernel());
        auto model = dev.compile(make_qwen35_attn_prepare_model_kernel());
        auto model_dev = dev.compile(make_qwen35_attn_prepare_model_dev_kernel());
        for (uint32_t pos0 : {0u, 1u, 5u, 37u}) {
            const auto label = luisa::format("attn prepare model tokens={} pos0={}", tokens, pos0);
            for (bool device_state : {false, true}) {
                const auto run = [&](bool use_model) {
                    luisa::vector<int32_t> pos(tokens);
                    for (uint32_t t = 0; t < tokens; t++) {
                        pos[t] = static_cast<int32_t>(pos0 + t);
                    }
                    luisa::vector<uint32_t> st(4u);
                    st[qwen_dec_state_position] = pos0;
                    stream << positions.copy_from(luisa::span<const int32_t>{pos})
                           << state.copy_from(luisa::span<const uint32_t>{st}) << synchronize();
                    luisa::vector<float> zeros(size_t(capacity) * kvh * hd, 0.0f);
                    stream << k_cache.copy_from(luisa::span<const float>{zeros})
                           << v_cache.copy_from(luisa::span<const float>{zeros}) << synchronize();
                    if (!use_model) {
                        if (device_state) {
                            stream << generic_dev(bqp, bkp, bvp, bqnw, bknw, positions,
                                                  q_out, gate_out, k_out, v_out, k_cache, v_cache,
                                                  state, tokens, qh, kvh, hd, cfg.rope_dim,
                                                  cfg.rope_theta, cfg.norm_eps, capacity,
                                                  tokens * qh, tokens * kvh, tokens * kvh)
                                          .dispatch(qwen_dispatch_threads(*generic_plan.value));
                        } else {
                            stream << generic(bqp, bkp, bvp, bqnw, bknw, positions,
                                              q_out, gate_out, k_out, v_out, k_cache, v_cache,
                                              tokens, qh, kvh, hd, cfg.rope_dim,
                                              cfg.rope_theta, cfg.norm_eps, pos0, capacity,
                                              tokens * qh, tokens * kvh, tokens * kvh)
                                          .dispatch(qwen_dispatch_threads(*generic_plan.value));
                        }
                    } else {
                        if (device_state) {
                            stream << model_dev(bqp, bkp, bvp, bqnw, bknw,
                                                q_out, gate_out, k_out, v_out, k_cache, v_cache,
                                                state, tokens, cfg.rope_theta, cfg.norm_eps, capacity)
                                          .dispatch(qwen_dispatch_threads(*model_plan.value));
                        } else {
                            stream << model(bqp, bkp, bvp, bqnw, bknw,
                                            q_out, gate_out, k_out, v_out, k_cache, v_cache,
                                            tokens, cfg.rope_theta, cfg.norm_eps, pos0, capacity)
                                          .dispatch(qwen_dispatch_threads(*model_plan.value));
                        }
                    }
                };
                run(false);
                auto want_q = download(stream, q_out, size_t(tokens) * qh * hd);
                auto want_gate = download(stream, gate_out, size_t(tokens) * qh * hd);
                auto want_k = download(stream, k_out, kn);
                auto want_v = download(stream, v_out, kn);
                auto want_kc = download(stream, k_cache, size_t(capacity) * kvh * hd);
                auto want_vc = download(stream, v_cache, size_t(capacity) * kvh * hd);
                run(true);
                auto got_q = download(stream, q_out, size_t(tokens) * qh * hd);
                auto got_gate = download(stream, gate_out, size_t(tokens) * qh * hd);
                auto got_k = download(stream, k_out, kn);
                auto got_v = download(stream, v_out, kn);
                auto got_kc = download(stream, k_cache, size_t(capacity) * kvh * hd);
                auto got_vc = download(stream, v_cache, size_t(capacity) * kvh * hd);
                const auto name = [&](const char *what) {
                    return label + (device_state ? " dev " : " host ") + what;
                };
                check_close(got_q, want_q, 4e-6f, 4e-6f, name("q").c_str());
                check_close(got_gate, want_gate, 0.0f, 0.0f, name("gate").c_str());
                check_close(got_k, want_k, 4e-6f, 4e-6f, name("k").c_str());
                check_close(got_v, want_v, 0.0f, 0.0f, name("v").c_str());
                check_close(got_kc, want_kc, 4e-6f, 4e-6f, name("k cache").c_str());
                check_close(got_vc, want_vc, 0.0f, 0.0f, name("v cache").c_str());
            }
        }
    }
    // Geometry gates.
    auto bad = cfg;
    bad.attn_head_dim = 128u;
    check(!validate_qwen35_attn_prepare_model_inputs(1u, bad, 0u, capacity).ok(),
          "model attn prepare refuses a non-256 head_dim");
    auto bad_rope = cfg;
    bad_rope.rope_dim = 128u;
    check(!validate_qwen35_attn_prepare_model_inputs(1u, bad_rope, 0u, capacity).ok(),
          "model attn prepare refuses a non-64 rope_dim");
    check(!validate_qwen35_attn_prepare_model_inputs(1u, cfg, capacity, capacity).ok(),
          "model attn prepare refuses an append past the cache capacity");
}

// ---------------------------------------------------------------------------
// Native Q8_0 weight storage (opt_plan/05 Q1, phase 8a).
//
// A synthetic Q8_0 matrix is built the way the loader builds a real one: raw
// signed int8 block codes + one IEEE FP16 scale per 32-element block, encoded
// through host_pack_q8_0_codes (the same primitive the GGUF uploader uses).
// Every matrix path the model needs is then compared against
// ref_qwen_matmul(q8_0, ...), which decodes the SAME codes, so only FP32
// accumulation-order noise remains:
//   * generic packed GEMV with M = 1..4 and the residual epilogue,
//   * reference GEMM (one thread per (m,n)) at M = 4,
//   * the weight-reusing mt prefill GEMM at M = 16 and a ragged M = 20,
//   * the 4-accumulator M=1 decode specialization (K % 512 == 0),
//   * the 128-bit uint4 decode realization (K % 512 == 0, group 32),
//   * the fused FFN decode (SiLU(gate)*up) and the fused GDN alpha/beta gates,
//   * the embedding gather (exact decode parity).
// K covers the model's real dimensions (2560 / 4096 / 9216) and one N is ragged.
// ---------------------------------------------------------------------------
void test_qwen_q8_native(Device &dev, Stream &stream, Rng &rng, Buffer<float> &dummy_f) {
    constexpr uint32_t group = kQwenQ8BlockSize;// 32 (the GGUF block)
    // Synthetic Q8_0 matrix: full-range signed int8 codes and positive FP16
    // block scales in [2^-10, 2^-3]. `bits` is what a GGUF block stores.
    const auto make_q8 = [&](uint32_t K, uint32_t N, luisa::vector<uint32_t> &words,
                             luisa::vector<float> &scales, luisa::vector<uint8_t> &codes) {
        codes.resize(size_t(K) * N);
        luisa::vector<uint16_t> bits(size_t(K) * N / group);
        for (auto &c : codes) { c = static_cast<uint8_t>(rng.next() & 0xFFu); }
        for (auto &b : bits) {
            // Exactly representable FP16 values: k * 2^-12 for k in [1, 4095].
            b = luisa::qwen::f32_to_fp16(static_cast<float>(1u + (rng.next() & 0xFFFu)) / 4096.0f);
        }
        if (auto st = host_pack_q8_0_codes(codes, bits, K, N, words, scales); !st.ok()) {
            check(false, "q8 host_pack_q8_0_codes", st.message);
        }
    };
    // Sum |x*w| grows with K, so the absolute bound scales with it (FP32
    // reassociation only; the oracle decodes the same codes).
    const auto tol = [](uint32_t K) { return 2e-3f * static_cast<float>(K) / 2560.0f; };

    // --- host-side ABI facts (no device): zero re-quantization ---------------
    {
        // One hand-built 34-byte block: scale fp16 0.125 (0x3000), codes -128..-97.
        luisa::vector<std::byte> payload;
        payload.emplace_back(std::byte{0x00});
        payload.emplace_back(std::byte{0x30});
        for (uint32_t i = 0; i < 32u; i++) {
            payload.emplace_back(static_cast<std::byte>(static_cast<uint8_t>(-128 + static_cast<int>(i))));
        }
        luisa::vector<uint32_t> words(8u, 0u);
        luisa::vector<float> scales(1u, 0.0f);
        check(luisa::qwen::pack_q8_0_blocks(payload, 32u, words, scales).ok(), "q8 block split");
        // Code bytes must be a byte copy (4 codes per word, LOW element first).
        bool bytes_ok = true;
        for (uint32_t i = 0; i < 32u; i++) {
            const uint32_t want = static_cast<uint32_t>(static_cast<uint8_t>(-128 + static_cast<int>(i)));
            const uint32_t got = (words[i / 4u] >> (8u * (i % 4u))) & 0xFFu;
            bytes_ok = bytes_ok && got == want;
        }
        check(bytes_ok, "q8 block codes are a byte copy");
        check(scales[0] == 0.125f, "q8 block scale is the exact fp16 widening");
        luisa::vector<float> dq(32u);
        check(luisa::qwen::dequantize_q8_0_matrix(words, scales, 32u, 1u, dq).ok(), "q8 dequantize");
        bool dq_ok = true;
        for (uint32_t i = 0; i < 32u; i++) {
            dq_ok = dq_ok && dq[i] == 0.125f * static_cast<float>(-128 + static_cast<int>(i));
        }
        check(dq_ok, "q8 dequantize = int8 * scale");
        // Geometry fail-closed: a payload that is not whole blocks, a wrong
        // scale count, and K % 32 != 0 in the matrix form.
        check(!luisa::qwen::pack_q8_0_blocks(luisa::span<const std::byte>{payload.data(), 33u},
                                             32u, words, scales).ok(),
              "q8 pack rejects a partial block payload");
        check(!luisa::qwen::pack_q8_0_blocks(payload, 48u, words, scales).ok(),
              "q8 pack rejects a non-multiple-of-32 element count");
        check(!packed_matrix_words(QwenWeightFormat::q8_0, 2560u - 8u, 4u).ok(),
              "q8 packed words reject K % 32 != 0");
        check(!q8_0_scale_elements(2540u, 4u).ok(), "q8 scale elements reject K % 32 != 0");
        check(!validate_qwen_q8_0_geometry(2560u, 64u).ok(), "q8 geometry rejects group 64");
        // Host re-quantization is refused outright.
        luisa::vector<float> some_w(2560u * 2u, 0.5f), some_words_f;
        luisa::vector<uint32_t> some_words;
        luisa::vector<float> some_scales;
        check(!host_pack_matrix(QwenWeightFormat::q8_0, some_w, 2560u, 2u, group,
                                some_words, some_scales).ok(),
              "host_pack_matrix refuses to quantize to q8_0");
    }

    // --- generic GEMV (M rows) + reference GEMM, real K, ragged N -----------
    for (const auto spec : {std::pair<uint32_t, uint32_t>{2560u, 64u},
                            std::pair<uint32_t, uint32_t>{4096u, 40u},
                            std::pair<uint32_t, uint32_t>{9216u, 32u}}) {
        const uint32_t K = spec.first, N = spec.second, M = 3u;
        luisa::vector<uint32_t> words;
        luisa::vector<float> scales;
        luisa::vector<uint8_t> codes;
        make_q8(K, N, words, scales, codes);
        luisa::vector<float> x(size_t(M) * K), res(size_t(M) * N);
        rng.fill(x, -1.f, 1.f);
        rng.fill(res, -0.5f, 0.5f);
        auto bw = upload(dev, stream, words);
        auto bs = upload(dev, stream, scales);
        auto bx = upload(dev, stream, x);
        auto br = upload(dev, stream, res);
        auto by = dev.create_buffer<float>(size_t(M) * N);
        luisa::vector<float> ref;
        check(ref_qwen_matmul(QwenWeightFormat::q8_0, x, words, scales, res, true, M, N, K,
                              group, ref).ok(),
              "q8 gemv oracle");
        const auto label = [&](const char *what) {
            return luisa::string{"q8 "} + what + " K" + std::to_string(K).c_str() +
                   " N" + std::to_string(N).c_str();
        };
        {
            auto sh = dev.compile(make_qwen_gemv_kernel(QwenWeightFormat::q8_0,
                                                       QwenMatmulEpilogue::residual));
            auto plan = plan_qwen_gemv_dispatch(M, N, K);
            check(plan.ok(), "q8 gemv plan");
            stream << sh(bx, bw, bs, br, by, M, N, K, group)
                          .dispatch(qwen_dispatch_threads(*plan.value));
            auto y = download(stream, by, size_t(M) * N);
            check_close(y, ref, tol(K), 2e-2f, label("generic gemv residual").c_str());
        }
        {
            auto sh = dev.compile(make_qwen_gemm_kernel(QwenWeightFormat::q8_0,
                                                       QwenMatmulEpilogue::residual));
            auto plan = plan_qwen_gemm_dispatch(M, N, K);
            check(plan.ok(), "q8 gemm plan");
            stream << sh(bx, bw, bs, br, by, M, N, K, group, plan.value->padded_cols)
                          .dispatch(plan.value->total_x, plan.value->total_y);
            auto y = download(stream, by, size_t(M) * N);
            check_close(y, ref, tol(K), 2e-2f, label("reference gemm residual").c_str());
        }
        // --- weight-reusing mt prefill GEMM (M = 16 exact, 20 ragged stripes) --
        for (const uint32_t m : {16u, 20u}) {
            luisa::vector<float> xm(size_t(m) * K), resm(size_t(m) * N);
            rng.fill(xm, -1.f, 1.f);
            rng.fill(resm, -0.5f, 0.5f);
            auto bxm = upload(dev, stream, xm);
            auto brm = upload(dev, stream, resm);
            auto bym = dev.create_buffer<float>(size_t(m) * N);
            luisa::vector<float> refm;
            check(ref_qwen_matmul(QwenWeightFormat::q8_0, xm, words, scales, resm, true, m, N, K,
                                  group, refm).ok(),
                  "q8 mt gemm oracle");
            QwenGemmMtConfig cfg{};
            auto plan = plan_qwen35_gemm_mt_dispatch(m, N, K, QwenWeightFormat::q8_0, group, cfg);
            check(plan.ok(), "q8 mt gemm plan");
            auto sh = dev.compile(make_qwen35_gemm_mt_kernel(cfg, QwenWeightFormat::q8_0,
                                                            QwenMatmulEpilogue::residual));
            stream << sh(bxm, bw, bs, brm, bym, m, N, K, group, plan.value->column_blocks)
                          .dispatch(plan.value->total_x, plan.value->total_y);
            auto ym = download(stream, bym, size_t(m) * N);
            check_close(ym, refm, tol(K), 2e-2f, (label("mt gemm residual") +
                                                 " M" + std::to_string(m).c_str()).c_str());
        }
        // --- M = 1 decode specializations (model path) -------------------------
        {
            luisa::vector<float> x1(x.begin(), x.begin() + K), res1(res.begin(), res.begin() + N);
            auto bx1 = upload(dev, stream, x1);
            auto br1 = upload(dev, stream, res1);
            auto by1 = dev.create_buffer<float>(N);
            luisa::vector<float> ref1;
            check(ref_qwen_matmul(QwenWeightFormat::q8_0, x1, words, scales, res1, true, 1u, N, K,
                                  group, ref1).ok(),
                  "q8 decode gemv oracle");
            {
                auto plan = plan_qwen35_decode_gemv_dispatch(N, K, QwenWeightFormat::q8_0);
                check(plan.ok(), "q8 decode gemv plan");
                auto sh = dev.compile(make_qwen35_decode_gemv_kernel(
                    QwenWeightFormat::q8_0, QwenMatmulEpilogue::residual));
                stream << sh(bx1, bw, bs, br1, by1, 1u, N, K, group)
                              .dispatch(qwen_dispatch_threads(*plan.value));
                auto y1 = download(stream, by1, N);
                check_close(y1, ref1, tol(K), 2e-2f, label("decode gemv residual").c_str());
            }
            {
                auto plan = plan_qwen35_decode_gemv_packed_dispatch(N, K, QwenWeightFormat::q8_0, group);
                check(plan.ok(), "q8 packed decode gemv plan");
                auto sh = dev.compile(make_qwen35_decode_gemv_packed_kernel(
                    QwenWeightFormat::q8_0, QwenMatmulEpilogue::residual));
                const auto vecs = uint32_t(K / qwen_packed_vec_elements(QwenWeightFormat::q8_0));
                stream << sh(bx1.view().as<float4>(), bw.view().as<uint4>(), bs, br1, by1,
                             N, K, group, vecs)
                              .dispatch(qwen_dispatch_threads(*plan.value));
                auto y1 = download(stream, by1, N);
                check_close(y1, ref1, tol(K), 3e-2f, label("packed decode gemv residual").c_str());
            }
        }
    }

    // --- fused FFN decode (gate/up + SiLU) at the model shape ---------------
    {
        const uint32_t K = 2560u, F = 64u;
        luisa::vector<uint32_t> gw, uw;
        luisa::vector<float> gs, us;
        luisa::vector<uint8_t> gc, uc;
        make_q8(K, F, gw, gs, gc);
        make_q8(K, F, uw, us, uc);
        luisa::vector<float> x(K);
        rng.fill(x, -1.f, 1.f);
        luisa::vector<float> gate_ref, up_ref, want;
        check(ref_qwen_matmul(QwenWeightFormat::q8_0, x, gw, gs, {}, false, 1u, F, K, group,
                              gate_ref).ok(),
              "q8 ffn gate oracle");
        check(ref_qwen_matmul(QwenWeightFormat::q8_0, x, uw, us, {}, false, 1u, F, K, group,
                              up_ref).ok(),
              "q8 ffn up oracle");
        check(ref_qwen_swiglu(gate_ref, up_ref, want).ok(), "q8 ffn swiglu oracle");
        auto bx = upload(dev, stream, x);
        auto bgw = upload(dev, stream, gw);
        auto buw = upload(dev, stream, uw);
        auto bgs = upload(dev, stream, gs);
        auto bus = upload(dev, stream, us);
        auto bo = dev.create_buffer<float>(F);
        auto plan = plan_qwen_ffn_decode_dispatch(F, K, QwenWeightFormat::q8_0, group);
        check(plan.ok(), "q8 ffn decode plan");
        auto sh = dev.compile(make_qwen_ffn_decode_kernel(QwenWeightFormat::q8_0));
        stream << sh(bx, bgw, bgs, buw, bus, bo, F, K, group)
                      .dispatch(qwen_dispatch_threads(*plan.value));
        auto out = download(stream, bo, F);
        check_close(out, want, tol(K), 3e-2f, "q8 ffn decode fusion");
        // 128-bit realization of the same fusion.
        auto shp = dev.compile(make_qwen_ffn_decode_packed_kernel(QwenWeightFormat::q8_0));
        auto planp = plan_qwen_ffn_decode_packed_dispatch(F, K, QwenWeightFormat::q8_0, group);
        check(planp.ok(), "q8 ffn decode packed plan");
        const auto vecs = uint32_t(K / qwen_packed_vec_elements(QwenWeightFormat::q8_0));
        stream << shp(bx.view().as<float4>(), bgw.view().as<uint4>(), bgs, buw.view().as<uint4>(),
                      bus, bo, F, K, group, vecs)
                      .dispatch(qwen_dispatch_threads(*planp.value));
        auto outp = download(stream, bo, F);
        check_close(outp, want, tol(K), 3e-2f, "q8 ffn decode packed fusion");
    }

    // --- fused GDN alpha/beta gates at the model shape ----------------------
    {
        const uint32_t K = 2560u, H = 8u;
        luisa::vector<uint32_t> aw, bw;
        luisa::vector<float> as, bs;
        luisa::vector<uint8_t> ac, bc;
        make_q8(K, H, aw, as, ac);
        make_q8(K, H, bw, bs, bc);
        luisa::vector<float> x(K), ssm_a(H), dt(H);
        rng.fill(x, -1.f, 1.f);
        rng.fill(ssm_a, -2.f, -0.1f);
        rng.fill(dt, -0.5f, 0.5f);
        luisa::vector<float> a_proj, b_proj, beta_ref, g_ref;
        check(ref_qwen_matmul(QwenWeightFormat::q8_0, x, aw, as, {}, false, 1u, H, K, group,
                              a_proj).ok(),
              "q8 gdn alpha oracle");
        check(ref_qwen_matmul(QwenWeightFormat::q8_0, x, bw, bs, {}, false, 1u, H, K, group,
                              b_proj).ok(),
              "q8 gdn beta oracle");
        check(ref_qwen_gdn_gates(a_proj, b_proj, ssm_a, dt, 1u, H, 20.f, beta_ref, g_ref).ok(),
              "q8 gdn gates oracle");
        auto bx = upload(dev, stream, x);
        auto baw = upload(dev, stream, aw);
        auto bbw = upload(dev, stream, bw);
        auto bas = upload(dev, stream, as);
        auto bbs = upload(dev, stream, bs);
        auto bsa = upload(dev, stream, ssm_a);
        auto bdt = upload(dev, stream, dt);
        auto beta = dev.create_buffer<float>(H);
        auto g = dev.create_buffer<float>(H);
        {
            auto plan = plan_qwen_gdn_decode_gates_dispatch(H, K, QwenWeightFormat::q8_0, group);
            check(plan.ok(), "q8 gdn gates plan");
            auto sh = dev.compile(make_qwen_gdn_decode_gates_kernel(QwenWeightFormat::q8_0));
            stream << sh(bx, baw, bas, bbw, bbs, bsa, bdt, beta, g, K, H, group, 20.f)
                          .dispatch(qwen_dispatch_threads(*plan.value));
            check_close(download(stream, beta, H), beta_ref, 3e-3f, 3e-2f,
                        "q8 gdn decode beta fusion");
            check_close(download(stream, g, H), g_ref, 6e-3f, 3e-2f,
                        "q8 gdn decode alpha fusion");
        }
        {
            auto plan = plan_qwen_gdn_decode_gates_packed_dispatch(H, K, QwenWeightFormat::q8_0, group);
            check(plan.ok(), "q8 gdn gates packed plan");
            auto sh = dev.compile(make_qwen_gdn_decode_gates_packed_kernel(QwenWeightFormat::q8_0));
            const auto vecs = uint32_t(K / qwen_packed_vec_elements(QwenWeightFormat::q8_0));
            stream << sh(bx.view().as<float4>(), baw.view().as<uint4>(), bas,
                         bbw.view().as<uint4>(), bbs, bsa, bdt, beta, g, K, H, group, 20.f, vecs)
                          .dispatch(qwen_dispatch_threads(*plan.value));
            check_close(download(stream, beta, H), beta_ref, 3e-3f, 3e-2f,
                        "q8 gdn decode beta packed fusion");
            check_close(download(stream, g, H), g_ref, 6e-3f, 3e-2f,
                        "q8 gdn decode alpha packed fusion");
        }
    }

    // --- embedding gather: exact decode parity with the oracle --------------
    {
        const uint32_t hidden = 2560u, vocab = 12u;
        luisa::vector<uint32_t> words;
        luisa::vector<float> scales;
        luisa::vector<uint8_t> codes;
        make_q8(hidden, vocab, words, scales, codes);
        luisa::vector<uint32_t> ids{11u, 0u, 7u, 11u};
        auto bt = upload(dev, stream, words);
        auto bs = upload(dev, stream, scales);
        auto bi = upload(dev, stream, ids);
        auto bo = dev.create_buffer<float>(ids.size() * hidden);
        auto sh = dev.compile(make_qwen_embedding_kernel(QwenWeightFormat::q8_0));
        auto plan = plan_qwen_embedding_dispatch(ids.size(), hidden);
        check(plan.ok(), "q8 embedding plan");
        stream << sh(bi, bt, bs, bo, uint32_t(ids.size()), hidden, vocab, group,
                     plan.value->padded_cols)
                      .dispatch(plan.value->total_x, plan.value->total_y);
        auto out = download(stream, bo, ids.size() * hidden);
        luisa::vector<float> ref;
        check(ref_qwen_embedding(QwenWeightFormat::q8_0, ids, words, scales, hidden, vocab,
                                 group, ref).ok(),
              "q8 embedding oracle");
        check_exact(out, ref, "q8 embedding exact decode");
    }

    // --- fail-closed planners for the q8 geometry --------------------------
    // 2304 = 72 * 32 (valid q8 geometry) but not a whole 128-word stride.
    check(!plan_qwen35_decode_gemv_dispatch(64u, 2304u, QwenWeightFormat::q8_0).ok(),
          "q8 decode gemv rejects K % 512 != 0");
    check(!plan_qwen35_decode_gemv_packed_dispatch(64u, 2560u, QwenWeightFormat::q8_0, 64u).ok(),
          "q8 packed decode rejects a non-block group");
    check(!plan_qwen_ffn_decode_dispatch(64u, 2500u, QwenWeightFormat::q8_0, group).ok(),
          "q8 ffn decode rejects K % 32 != 0");
    check(!plan_qwen_gdn_decode_gates_dispatch(32u, 2560u, QwenWeightFormat::q8_0, 64u).ok(),
          "q8 gdn gates reject a non-block group");
    check(!plan_qwen35_gemm_mt_dispatch(16u, 32u, 2540u, QwenWeightFormat::q8_0, group,
                                        QwenGemmMtConfig{}).ok(),
          "q8 mt gemm rejects K % 32 != 0");
    check(plan_qwen35_gemm_mt_dispatch(16u, 32u, 2560u, QwenWeightFormat::q8_0, group,
                                       QwenGemmMtConfig{}).ok(),
          "q8 mt gemm accepts the model geometry");
}



// ---------------------------------------------------------------------------
// W4A16 group-wise weights (opt_plan/05 Q2, phase 8b).
//
// A synthetic matrix is quantized through host_pack_matrix(w4, ...), i.e. the
// deterministic clipping search of common.h, and every device path the model
// needs is compared against ref_qwen_matmul(w4, ...), which decodes the SAME
// words and scales (only FP32 accumulation-order noise remains):
// * generic packed GEMV (M = 1..4) with the residual epilogue,
// * reference GEMM (one thread per (m,n)),
// * weight-reusing mt prefill GEMM at M = 16 and a ragged M = 20,
// * the 4-accumulator M = 1 decode specialization (one word = 8 codes),
// * the fused FFN decode and the fused GDN alpha/beta gate GEMV,
// * the embedding gather (exact decode parity),
// * the host ABI facts: geometry fail-closed, RTN vs clipping MSE, exact
//   host/device decode parity and reproducibility.
// K covers the model's real dimensions (2560 / 4096 / 9216) and one N is ragged;
// N = 32 is the narrow alpha/beta gate width. There is NO 128-bit W4 realization:
// plan_qwen35_decode_gemv_packed_dispatch must refuse it.
// ---------------------------------------------------------------------------
void test_qwen_w4(Device &dev, Stream &stream, Rng &rng, Buffer<float> &dummy_f) {
    constexpr uint32_t group = 128u;
    const auto make_w4 = [&](uint32_t K, uint32_t N, luisa::vector<uint32_t> &words,
                             luisa::vector<float> &scales, luisa::vector<float> &w) {
        w.resize(size_t(K) * N);
        // Deterministic, non-uniform weights with per-group outliers: exactly the
        // distribution the clipping search exists for.
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t k = 0; k < K; ++k) {
                const float u = static_cast<float>((rng.next() & 0xFFFFu)) / 65536.0f;
                const float base = (k / group) % 3u == 0u ? 0.05f : 0.012f;
                w[size_t(n) * K + k] = (u - 0.5f) * 2.0f * base;
                if ((k % group) == 5u) { w[size_t(n) * K + k] *= 4.0f; }
            }
        }
        if (auto st = host_pack_matrix(QwenWeightFormat::w4, w, K, N, group, words, scales); !st.ok()) {
            check(false, "w4 host_pack_matrix", st.message);
        }
    };
    const auto tol = [](uint32_t K) { return 1e-2f * static_cast<float>(K) / 2560.0f; };

    // --- host ABI facts (no device) ----------------------------------------
    {
        check(!validate_qwen_w4_geometry(2562u, 128u).ok(), "w4 geometry rejects K % 8 != 0");
        check(!validate_qwen_w4_geometry(2600u, 128u).ok(), "w4 geometry rejects ragged groups");
        check(!validate_qwen_w4_geometry(2560u, 96u).ok(), "w4 geometry rejects group 96");
        check(validate_qwen_w4_geometry(2560u, 128u).ok(), "w4 geometry accepts K=2560 g128");
        check(validate_qwen_w4_geometry(1024u, 64u).ok(), "w4 geometry accepts K=1024 g64");
        check(!packed_matrix_words(QwenWeightFormat::w4, 2540u, 4u).ok(),
              "w4 packed words reject K % 8 != 0");
        check(!weight_scale_elements(QwenWeightFormat::w4, 2560u, 4u, 96u).ok(),
              "w4 scale elements reject group 96");
        check(qwen_weight_pack(QwenWeightFormat::w4) == 8u, "w4 packs 8 codes per word");
        // No 128-bit W4 realization: the planner refuses instead of mis-dispatching.
        check(!plan_qwen35_decode_gemv_packed_dispatch(32u, 2560u, QwenWeightFormat::w4, group).ok(),
              "w4 has no 128-bit decode realization");
        // Deterministic clipping search beats plain RTN on the same data.
        luisa::vector<float> w, scales;
        luisa::vector<uint32_t> words;
        make_w4(256u, 2u, words, scales, w);
        W4PackStats stats{};
        W4ClipSearchConfig cfg{};
        luisa::vector<uint32_t> w2(size_t(256u) * 2u / 8u, 0u);
        luisa::vector<float> s2(2u * 2u, 0.0f);
        check(pack_w4_matrix(w, 256u, 2u, cfg, w2, s2, &stats).ok(), "w4 common packer");
        check(stats.mse <= stats.rtn_mse, "w4 clipping search is not worse than RTN");
        check(std::memcmp(words.data(), w2.data(), words.size() * 4u) == 0,
              "dsl and common W4 packers agree bit for bit");
        check(std::memcmp(scales.data(), s2.data(), scales.size() * 4u) == 0,
              "dsl and common W4 scales agree bit for bit");
        // Exact host decode parity with the common ABI decoder.
        luisa::vector<float> dq0(size_t(256u) * 2u);
        check(dequantize_w4_matrix(words, scales, 256u, 2u, group, dq0).ok(), "w4 dequantize");
        bool parity = true;
        for (uint64_t i = 0; i < dq0.size(); ++i) {
            parity = parity && (host_decode_packed(QwenWeightFormat::w4, words, scales, 256u,
                                                   group, i) == dq0[i]);
        }
        check(parity, "w4 host decode parity (dsl vs common)");
    }

    // --- real shapes: generic GEMV / GEMM / mt GEMM / decode GEMV ----------
    for (const auto spec : {std::pair<uint32_t, uint32_t>{2560u, 64u},
                            std::pair<uint32_t, uint32_t>{4096u, 40u},
                            std::pair<uint32_t, uint32_t>{9216u, 32u}}) {
        const uint32_t K = spec.first, N = spec.second, M = 3u;
        luisa::vector<uint32_t> words;
        luisa::vector<float> scales, w;
        make_w4(K, N, words, scales, w);
        luisa::vector<float> x(size_t(M) * K), res(size_t(M) * N);
        rng.fill(x, -1.f, 1.f);
        rng.fill(res, -0.5f, 0.5f);
        auto bw = upload(dev, stream, words);
        auto bs = upload(dev, stream, scales);
        auto bx = upload(dev, stream, x);
        auto br = upload(dev, stream, res);
        auto by = dev.create_buffer<float>(size_t(M) * N);
        const auto label = [&](const char *what) {
            return luisa::string{"w4 "} + what + " K" + std::to_string(K).c_str() +
                   " N" + std::to_string(N).c_str();
        };
        luisa::vector<float> ref;
        check(ref_qwen_matmul(QwenWeightFormat::w4, x, words, scales, res, true, M, N, K,
                              group, ref).ok(),
              "w4 gemv oracle");
        {
            auto sh = dev.compile(make_qwen_gemv_kernel(QwenWeightFormat::w4,
                                                       QwenMatmulEpilogue::residual));
            auto plan = plan_qwen_gemv_dispatch(M, N, K);
            check(plan.ok(), "w4 gemv plan");
            stream << sh(bx, bw, bs, br, by, M, N, K, group)
                          .dispatch(qwen_dispatch_threads(*plan.value));
            auto y = download(stream, by, size_t(M) * N);
            check_close(y, ref, tol(K), 2e-2f, label("generic gemv residual").c_str());
        }
        {
            auto sh = dev.compile(make_qwen_gemm_kernel(QwenWeightFormat::w4,
                                                       QwenMatmulEpilogue::residual));
            auto plan = plan_qwen_gemm_dispatch(M, N, K);
            check(plan.ok(), "w4 gemm plan");
            stream << sh(bx, bw, bs, br, by, M, N, K, group, plan.value->padded_cols)
                          .dispatch(plan.value->total_x, plan.value->total_y);
            auto y = download(stream, by, size_t(M) * N);
            check_close(y, ref, tol(K), 2e-2f, label("reference gemm residual").c_str());
        }
        for (const uint32_t m : {16u, 20u}) {
            luisa::vector<float> xm(size_t(m) * K), resm(size_t(m) * N);
            rng.fill(xm, -1.f, 1.f);
            rng.fill(resm, -0.5f, 0.5f);
            auto bxm = upload(dev, stream, xm);
            auto brm = upload(dev, stream, resm);
            auto bym = dev.create_buffer<float>(size_t(m) * N);
            luisa::vector<float> refm;
            check(ref_qwen_matmul(QwenWeightFormat::w4, xm, words, scales, resm, true, m, N, K,
                                  group, refm).ok(),
                  "w4 mt gemm oracle");
            QwenGemmMtConfig cfg{};
            auto plan = plan_qwen35_gemm_mt_dispatch(m, N, K, QwenWeightFormat::w4, group, cfg);
            check(plan.ok(), "w4 mt gemm plan");
            auto sh = dev.compile(make_qwen35_gemm_mt_kernel(cfg, QwenWeightFormat::w4,
                                                            QwenMatmulEpilogue::residual));
            stream << sh(bxm, bw, bs, brm, bym, m, N, K, group, plan.value->column_blocks)
                          .dispatch(plan.value->total_x, plan.value->total_y);
            auto ym = download(stream, bym, size_t(m) * N);
            check_close(ym, refm, tol(K), 2e-2f,
                        (label("mt gemm residual") + " M" + std::to_string(m).c_str()).c_str());
        }
        {
            luisa::vector<float> x1(x.begin(), x.begin() + K), res1(res.begin(), res.begin() + N);
            auto bx1 = upload(dev, stream, x1);
            auto br1 = upload(dev, stream, res1);
            auto by1 = dev.create_buffer<float>(N);
            luisa::vector<float> ref1;
            check(ref_qwen_matmul(QwenWeightFormat::w4, x1, words, scales, res1, true, 1u, N, K,
                                  group, ref1).ok(),
                  "w4 decode gemv oracle");
            auto plan = plan_qwen35_decode_gemv_dispatch(N, K, QwenWeightFormat::w4, group);
            check(plan.ok(), "w4 decode gemv plan");
            auto sh = dev.compile(make_qwen35_decode_gemv_kernel(
                QwenWeightFormat::w4, QwenMatmulEpilogue::residual));
            stream << sh(bx1, bw, bs, br1, by1, 1u, N, K, group)
                          .dispatch(qwen_dispatch_threads(*plan.value));
            auto y1 = download(stream, by1, N);
            check_close(y1, ref1, tol(K), 2e-2f, label("decode gemv residual").c_str());
        }
    }

    // --- fused FFN decode (gate/up + SiLU) at the model shape, N = 32 tail --
    for (const auto spec : {std::pair<uint32_t, uint32_t>{2560u, 9216u},
                            std::pair<uint32_t, uint32_t>{2560u, 32u}}) {
        const uint32_t K = spec.first, F = spec.second;
        luisa::vector<uint32_t> gw, uw;
        luisa::vector<float> gs, us, gwm, uwm;
        make_w4(K, F, gw, gs, gwm);
        make_w4(K, F, uw, us, uwm);
        luisa::vector<float> x1(K);
        rng.fill(x1, -1.f, 1.f);
        auto bx = upload(dev, stream, x1);
        auto bgw = upload(dev, stream, gw);
        auto bgs = upload(dev, stream, gs);
        auto buw = upload(dev, stream, uw);
        auto bus = upload(dev, stream, us);
        auto by = dev.create_buffer<float>(F);
        luisa::vector<float> g, u;
        check(ref_qwen_matmul(QwenWeightFormat::w4, x1, gw, gs, {}, false, 1u, F, K, group, g).ok(),
              "w4 ffn gate oracle");
        check(ref_qwen_matmul(QwenWeightFormat::w4, x1, uw, us, {}, false, 1u, F, K, group, u).ok(),
              "w4 ffn up oracle");
        luisa::vector<float> ref(F);
        for (uint32_t i = 0; i < F; ++i) {
            const float silu = g[i] / (1.0f + std::exp(-g[i]));
            ref[i] = silu * u[i];
        }
        auto plan = plan_qwen_ffn_decode_dispatch(F, K, QwenWeightFormat::w4, group);
        check(plan.ok(), "w4 ffn decode plan");
        auto sh = dev.compile(make_qwen_ffn_decode_kernel(QwenWeightFormat::w4));
        stream << sh(bx, bgw, bgs, buw, bus, by, F, K, group)
                      .dispatch(qwen_dispatch_threads(*plan.value));
        auto y = download(stream, by, F);
        check_close(y, ref, tol(K), 3e-2f,
                    (luisa::string{"w4 fused ffn decode F"} + std::to_string(F).c_str()).c_str());
    }

    // --- fused GDN alpha/beta gates (N = heads = 32) ------------------------
    {
        const uint32_t K = 2560u, H = 32u;
        luisa::vector<uint32_t> aw, bw4;
        luisa::vector<float> as, bs, awm, bwm;
        make_w4(K, H, aw, as, awm);
        make_w4(K, H, bw4, bs, bwm);
        luisa::vector<float> x1(K), ssm_a(H), dt_bias(H);
        rng.fill(x1, -1.f, 1.f);
        rng.fill(ssm_a, -0.5f, 0.5f);
        rng.fill(dt_bias, -0.5f, 0.5f);
        auto bx = upload(dev, stream, x1);
        auto baw = upload(dev, stream, aw);
        auto bas = upload(dev, stream, as);
        auto bbw = upload(dev, stream, bw4);
        auto bbs = upload(dev, stream, bs);
        auto ba = upload(dev, stream, ssm_a);
        auto bdt = upload(dev, stream, dt_bias);
        auto bbeta = dev.create_buffer<float>(H);
        auto bg = dev.create_buffer<float>(H);
        luisa::vector<float> alpha, beta;
        check(ref_qwen_matmul(QwenWeightFormat::w4, x1, aw, as, {}, false, 1u, H, K, group, alpha).ok(),
              "w4 gdn alpha oracle");
        check(ref_qwen_matmul(QwenWeightFormat::w4, x1, bw4, bs, {}, false, 1u, H, K, group, beta).ok(),
              "w4 gdn beta oracle");
        luisa::vector<float> ref_beta(H), ref_g(H);
        for (uint32_t h = 0; h < H; ++h) {
            ref_beta[h] = 1.0f / (1.0f + std::exp(-beta[h]));
            const float av = alpha[h] + dt_bias[h];
            const float sp = av > 20.0f ? av : std::log1p(std::exp(av));
            ref_g[h] = ssm_a[h] * sp;
        }
        auto plan = plan_qwen_gdn_decode_gates_dispatch(H, K, QwenWeightFormat::w4, group);
        check(plan.ok(), "w4 gdn gates plan");
        auto sh = dev.compile(make_qwen_gdn_decode_gates_kernel(QwenWeightFormat::w4));
        stream << sh(bx, baw, bas, bbw, bbs, ba, bdt, bbeta, bg, K, H, group, 20.0f)
                      .dispatch(qwen_dispatch_threads(*plan.value));
        auto got_beta = download(stream, bbeta, H);
        auto got_g = download(stream, bg, H);
        check_close(got_beta, ref_beta, tol(K), 1e-3f, "w4 gdn beta gate");
        check_close(got_g, ref_g, tol(K) * 4.0f, 1e-3f, "w4 gdn g gate");
    }

    // --- embedding gather (exact decode parity) ----------------------------
    {
        const uint32_t V = 64u, hidden = 2560u;
        luisa::vector<uint32_t> words;
        luisa::vector<float> scales, wm;
        make_w4(hidden, V, words, scales, wm);
        luisa::vector<uint32_t> ids{3u, 17u, 63u};
        auto bid = upload(dev, stream, ids);
        auto bw = upload(dev, stream, words);
        auto bs = upload(dev, stream, scales);
        auto by = dev.create_buffer<float>(size_t(ids.size()) * hidden);
        auto plan = plan_qwen_embedding_dispatch(ids.size(), hidden);
        check(plan.ok(), "w4 embedding plan");
        auto sh = dev.compile(make_qwen_embedding_kernel(QwenWeightFormat::w4));
        stream << sh(bid, bw, bs, by, uint32_t(ids.size()), hidden, V, group,
                     plan.value->padded_cols)
                      .dispatch(plan.value->total_x, plan.value->total_y);
        auto y = download(stream, by, size_t(ids.size()) * hidden);
        bool exact = true;
        for (size_t t = 0; t < ids.size(); ++t) {
            for (uint32_t c = 0; c < hidden; ++c) {
                const float want = host_decode_packed(QwenWeightFormat::w4, words, scales,
                                                      hidden, group,
                                                      uint64_t(ids[t]) * hidden + c);
                exact = exact && (y[t * hidden + c] == want);
            }
        }
        check(exact, "w4 embedding gather is exact decode parity");
    }
}

// ---------------------------------------------------------------------------
// KV dtype/layout three-way differential (opt_plan/05-quantization.md Q3).
//
//   arm A  FP32 cache, legacy position-major layout
//   arm B  BF16 cache, 64-position blocked layout, two RNE codes per word
//   arm C  the DEQUANTIZED CPU ORACLE: the same BF16 codes widened back to FP32
//          and fed to ref_qwen_attn_* -- i.e. exactly what the cache stores
//
// Arm A must agree with the dense oracle, arm B with the dequantized oracle, and
// A vs B only within the documented BF16 storage drift. Every attention
// realization is covered: prefill legacy + tiled, decode legacy + split-KV (both
// dtypes), the combined kernel, and the packed append word-for-word. Positions
// deliberately cross 64-position block boundaries (61 -> 64, 640 -> 650) and the
// capacity is ragged (650 = 10 full blocks + 10 positions), so a layout that
// silently truncates, wraps or pads wrongly fails here.
// ---------------------------------------------------------------------------
void test_attn_kv_dtype(Device &dev, Stream &stream, Rng &rng) {
    // The f32/bf16 attention arms take the fp8 scale arguments but never read them
    // (the scale load folds away), so one 1-element buffer serves every unscaled
    // dispatch of this test.
    auto bscale_dummy = dev.create_buffer<float>(1);
    // The dtype/layout contract lives in luisa::qwen (the allocator's namespace),
    // the kernels in luisa::qwen::dsl (imported at file scope).
    using luisa::qwen::QwenKvDType;
    using luisa::qwen::QwenKvLayoutDesc;
    using luisa::qwen::qwen_kv_layout_desc;
    using luisa::qwen::qwen_kv_region_from_dense;
    using luisa::qwen::qwen_kv_region_to_dense;
    using luisa::qwen::f32_to_bf16;
    using luisa::qwen::bf16_to_f32;
    QwenDslConfig cfg;// model geometry: 16 q heads / 4 kv heads / 256 dims
    const uint32_t qh = cfg.q_heads, kvh = cfg.kv_heads, hd = cfg.attn_head_dim;
    const uint32_t kv_n = cfg.attn_kv_n();
    const uint32_t capacity = 650u;// ragged: 11 x 64-position blocks, 10 spare
    const float scale = cfg.attn_scale();
    const QwenKvStorage f32{};
    const QwenKvStorage bf16 = qwen_kv_storage_native(QwenKvDType::bf16);

    // ---- contract (host, no device) ----------------------------------------
    QwenKvLayoutDesc fd{}, bd{};
    check(qwen_kv_layout_desc(f32.dtype, f32.layout, capacity, kvh, hd, fd).ok(), "kv dtype: f32 desc");
    check(qwen_kv_layout_desc(bf16.dtype, bf16.layout, capacity, kvh, hd, bd).ok(), "kv dtype: bf16 desc");
    check(bd.elements_per_word() == 2u && bd.position_blocks == 11u,
          "kv dtype: bf16 packs two codes per word in ceil(capacity/64) blocks");
    check(fd.region_words == static_cast<uint64_t>(capacity) * kv_n &&
              bd.region_words * 2u == static_cast<uint64_t>(bd.position_blocks) * 64u * kv_n,
          "kv dtype: region words follow the contract");
    check(bd.region_bytes() * 20u <= fd.region_bytes() * 11u,
          "kv dtype: the blocked bf16 region is ~half the fp32 bytes (plus <64-position padding)");

    // ---- fixture: one dense FP32 cache and its BF16 encoding ---------------
    const uint64_t dense_n = fd.region_words;
    luisa::vector<float> kc(dense_n), vc(dense_n);
    rng.fill(kc, -1.f, 1.f);
    rng.fill(vc, -1.f, 1.f);
    luisa::vector<uint32_t> kw(bd.region_words), vw(bd.region_words);
    check(qwen_kv_region_from_dense(bd, kc, capacity, kw).ok(), "kv dtype: encode bf16 K");
    check(qwen_kv_region_from_dense(bd, vc, capacity, vw).ok(), "kv dtype: encode bf16 V");
    luisa::vector<float> kq(dense_n), vq(dense_n);// the dequantized oracle image
    check(qwen_kv_region_to_dense(bd, kw, capacity, kq).ok(), "kv dtype: decode bf16 K");
    check(qwen_kv_region_to_dense(bd, vw, capacity, vq).ok(), "kv dtype: decode bf16 V");
    {
        bool rne_ok = true, negative_ok = false;
        for (uint64_t i = 0; i < dense_n && rne_ok; i++) {
            rne_ok = kq[i] == bf16_to_f32(f32_to_bf16(kc[i]));
        }
        for (uint64_t i = 0; i < dense_n && !negative_ok; i++) { negative_ok = kc[i] < 0.f; }
        check(rne_ok, "kv dtype: bf16 storage is exactly RNE(f32) for every stored value (no truncation)");
        check(negative_ok, "kv dtype: the fixture exercises negative codes");
        // RNE ties, both parities: 1+1/256 keeps an even mantissa and must round
        // DOWN, 1+3/256 keeps an odd one and must round UP.
        check(f32_to_bf16(1.00390625f) == 0x3F80u && f32_to_bf16(1.01171875f) == 0x3F82u,
              "kv dtype: bf16 ties round to even (both parities)");
        // Subnormal range: 2^-131 is BF16's smallest mantissa step times four.
        const float tiny = std::ldexp(1.0f, -131);
        check(f32_to_bf16(tiny) == 0x0004u && f32_to_bf16(-tiny) == 0x8004u &&
                  bf16_to_f32(0x0004u) == tiny,
              "kv dtype: bf16 subnormal codes round and widen exactly (sign included)");
    }

    // ---- device buffers ----------------------------------------------------
    auto bkc = upload(dev, stream, kc);// arm A: dense FP32 cache
    auto bvc = upload(dev, stream, vc);
    auto bkw = upload(dev, stream, kw);// arm B: packed BF16 words
    auto bvw = upload(dev, stream, vw);
    auto bkc_w = bkc.view().as<uint32_t>();// the FP32 A/B word view (same memory)
    auto kwf = bkw.view().as<float>();     // the packed cache as the kernels bind it
    auto vwf = bvw.view().as<float>();
    check(bkc_w.size() == dense_n &&
              bkc_w.size_bytes() == bkc.view().size_bytes(),
          "kv dtype: the fp32 cache exposes a uint32 A/B view of the same bytes");

    // ---- prefill: legacy + tiled, arms A and B -----------------------------
    {
        auto sh_pref_f32 = dev.compile(make_qwen_attn_prefill_kernel(f32));
        auto sh_pref_bf16 = dev.compile(make_qwen_attn_prefill_kernel(bf16));
        QwenAttnPrefillTiledConfig tcfg{};
        check(select_qwen35_attn_prefill_tiled_config(32u, 49152u, tcfg).ok(),
              "kv dtype: tiled prefill config");
        auto sh_tiled_f32 = dev.compile(make_qwen35_attn_prefill_tiled_kernel(tcfg, f32));
        auto sh_tiled_bf16 = dev.compile(make_qwen35_attn_prefill_tiled_kernel(tcfg, bf16));
        const struct {
            uint32_t tokens, pos0;
        } cases[] = {{1u, 0u}, {2u, 3u}, {5u, 61u}, {7u, 64u}, {3u, 640u}};
        for (const auto &c : cases) {
            const uint32_t rows = c.tokens, pos0 = c.pos0;
            const uint32_t kv_len = pos0 + rows;
            luisa::vector<float> q(static_cast<size_t>(rows) * qh * hd), gate(q.size());
            rng.fill(q, -1.f, 1.f);
            rng.fill(gate, -2.f, 2.f);
            auto bq = upload(dev, stream, q);
            auto bg = upload(dev, stream, gate);
            auto bo = dev.create_buffer<float>(q.size());
            luisa::vector<float> ref_dense, ref_deq;
            check(ref_qwen_attn_prefill(q, gate, kc, vc, rows, qh, kvh, hd, scale, pos0, kv_len, ref_dense).ok(),
                  "kv dtype: prefill oracle (dense)");
            check(ref_qwen_attn_prefill(q, gate, kq, vq, rows, qh, kvh, hd, scale, pos0, kv_len, ref_deq).ok(),
                  "kv dtype: prefill oracle (dequantized)");
            auto p_legacy = plan_qwen_attn_prefill_dispatch(rows, qh);
            auto p_tiled = plan_qwen35_attn_prefill_tiled_dispatch(rows, tcfg);
            check(p_legacy.ok() && p_tiled.ok(), "kv dtype: prefill plans");
            const auto tag = luisa::string{"t="} + luisa::string{std::to_string(rows).c_str()} +
                             luisa::string{"/pos="} + luisa::string{std::to_string(pos0).c_str()};
            const auto run_legacy = [&](auto &sh, auto kcv, auto vcv) {
                stream << sh(bq, bg, kcv, vcv, bscale_dummy.view(), bscale_dummy.view(), bo.view(), rows, qh, kvh, hd, scale, pos0, kv_len, capacity)
                              .dispatch(qwen_dispatch_threads(*p_legacy.value));
                return download(stream, bo, q.size());
            };
            const auto run_tiled = [&](auto &sh, auto kcv, auto vcv) {
                stream << sh(bq, bg, kcv, vcv, bscale_dummy.view(), bscale_dummy.view(), bo.view(), rows, scale, pos0, kv_len, capacity)
                              .dispatch(qwen_dispatch_threads(*p_tiled.value));
                return download(stream, bo, q.size());
            };
            const auto a_legacy = run_legacy(sh_pref_f32, bkc.view(), bvc.view());
            const auto a_tiled = run_tiled(sh_tiled_f32, bkc.view(), bvc.view());
            const auto b_legacy = run_legacy(sh_pref_bf16, kwf, vwf);
            const auto b_tiled = run_tiled(sh_tiled_bf16, kwf, vwf);
            stream << synchronize();
            check_close(a_legacy, ref_dense, 2e-5f, 2e-4f, (luisa::string{"kv f32 prefill legacy "} + tag).c_str());
            check_close(a_tiled, ref_dense, 2e-5f, 2e-4f, (luisa::string{"kv f32 prefill tiled "} + tag).c_str());
            check_close(b_legacy, ref_deq, 2e-5f, 2e-4f, (luisa::string{"kv bf16 prefill legacy "} + tag).c_str());
            check_close(b_tiled, ref_deq, 2e-5f, 2e-4f, (luisa::string{"kv bf16 prefill tiled "} + tag).c_str());
            // A vs B: BF16 STORAGE drift only (documented, never bit equality).
            check_close(b_legacy, a_legacy, 5e-2f, 5e-2f, (luisa::string{"kv f32-vs-bf16 prefill legacy "} + tag).c_str());
            check_close(b_tiled, a_tiled, 5e-2f, 5e-2f, (luisa::string{"kv f32-vs-bf16 prefill tiled "} + tag).c_str());
        }
    }

    // ---- decode: legacy + split-KV, arms A and B ---------------------------
    {
        QwenAttnDecodeSplitConfig scfg{};
        scfg.threshold = 8u;
        scfg.split_chunk = 16u;
        scfg.max_splits = 4u;
        scfg.k_tile = 4u;
        check(validate_qwen_attn_decode_split_config(scfg).ok(), "kv dtype: split config");
        auto sh_legacy_f32 = dev.compile(make_qwen_attn_decode_kernel(0u, f32));
        auto sh_legacy_bf16 = dev.compile(make_qwen_attn_decode_kernel(0u, bf16));
        auto sh_split_f32 = dev.compile(make_qwen_attn_decode_split_kernel(scfg, f32));
        auto sh_split_bf16 = dev.compile(make_qwen_attn_decode_split_kernel(scfg, bf16));
        auto sh_comb_f32 = dev.compile(make_qwen_attn_decode_combine_kernel(scfg));
        const uint64_t acc_n = static_cast<uint64_t>(scfg.max_splits) * qh * hd;
        const uint64_t ml_n = static_cast<uint64_t>(scfg.max_splits) * qh * 2u;
        auto bacc = dev.create_buffer<float>(static_cast<size_t>(acc_n));
        auto bml = dev.create_buffer<float>(static_cast<size_t>(ml_n));
        luisa::vector<float> q(qh * hd), gate(qh * hd);
        rng.fill(q, -1.f, 1.f);
        rng.fill(gate, -2.f, 2.f);
        auto bq = upload(dev, stream, q);
        auto bg = upload(dev, stream, gate);
        auto bo = dev.create_buffer<float>(qh * hd);
        auto decode_plan = plan_qwen_attn_decode_dispatch(qh);
        auto split_plan = plan_qwen_attn_decode_split_dispatch(scfg);
        auto comb_plan = plan_qwen_attn_decode_combine_dispatch(scfg);
        check(decode_plan.ok() && split_plan.ok() && comb_plan.ok(), "kv dtype: decode plans");
        for (const uint32_t kv_len : {1u, 63u, 64u, 65u, 511u, 512u, 513u, 640u, 650u}) {
            luisa::vector<float> ref_dense, ref_deq;
            check(ref_qwen_attn_decode(q, gate, kc, vc, qh, kvh, hd, scale, kv_len, ref_dense).ok(),
                  "kv dtype: decode oracle (dense)");
            check(ref_qwen_attn_decode(q, gate, kq, vq, qh, kvh, hd, scale, kv_len, ref_deq).ok(),
                  "kv dtype: decode oracle (dequantized)");
            const auto tag = luisa::string{"kv_len="} + luisa::string{std::to_string(kv_len).c_str()};
            const auto decode_legacy = [&](auto &sh, auto kcv, auto vcv) {
                stream << sh(bq, bg, kcv, vcv, bscale_dummy.view(), bscale_dummy.view(), bo.view(), qh, kvh, hd, scale, kv_len, capacity)
                              .dispatch(qwen_dispatch_threads(*decode_plan.value));
                return download(stream, bo, qh * hd);
            };
            const auto a_legacy = decode_legacy(sh_legacy_f32, bkc.view(), bvc.view());
            const auto b_legacy = decode_legacy(sh_legacy_bf16, kwf, vwf);
            stream << synchronize();
            check_close(a_legacy, ref_dense, 2e-5f, 2e-4f, (luisa::string{"kv f32 decode legacy "} + tag).c_str());
            check_close(b_legacy, ref_deq, 2e-5f, 2e-4f, (luisa::string{"kv bf16 decode legacy "} + tag).c_str());
            check_close(b_legacy, a_legacy, 5e-2f, 5e-2f, (luisa::string{"kv f32-vs-bf16 decode legacy "} + tag).c_str());
            if (kv_len >= scfg.threshold) {
                const auto decode_split = [&](auto &shs, auto kcv, auto vcv) {
                    stream << shs(bq, kcv, vcv, bscale_dummy.view(), bscale_dummy.view(), bacc.view(), bml.view(), kv_len, scale, capacity)
                                  .dispatch(qwen_dispatch_threads(*split_plan.value));
                    stream << sh_comb_f32(bacc.view(), bml.view(), bg, bo.view(), kv_len, capacity)
                                  .dispatch(qwen_dispatch_threads(*comb_plan.value));
                    return download(stream, bo, qh * hd);
                };
                const auto a_split = decode_split(sh_split_f32, bkc.view(), bvc.view());
                const auto b_split = decode_split(sh_split_bf16, kwf, vwf);
                stream << synchronize();
                check_close(a_split, ref_dense, 2e-5f, 2e-4f, (luisa::string{"kv f32 decode split "} + tag).c_str());
                check_close(b_split, ref_deq, 2e-5f, 2e-4f, (luisa::string{"kv bf16 decode split "} + tag).c_str());
                check_close(b_split, a_split, 5e-2f, 5e-2f, (luisa::string{"kv f32-vs-bf16 decode split "} + tag).c_str());
            }
        }
    }

    // ---- packed append: word-exact against the host encoder ----------------
    {
        const uint32_t rows = 3u, pos0 = 61u;// straddles the 64-position boundary
        luisa::vector<float> krows(static_cast<size_t>(rows) * kv_n), vrows(krows.size());
        rng.fill(krows, -1.f, 1.f);
        rng.fill(vrows, -1.f, 1.f);
        luisa::vector<float> kexp = kc, vexp = vc;
        for (uint32_t t = 0; t < rows; t++) {
            for (uint32_t i = 0; i < kv_n; i++) {
                kexp[static_cast<size_t>(pos0 + t) * kv_n + i] = krows[static_cast<size_t>(t) * kv_n + i];
                vexp[static_cast<size_t>(pos0 + t) * kv_n + i] = vrows[static_cast<size_t>(t) * kv_n + i];
            }
        }
        luisa::vector<uint32_t> kexp_w(bd.region_words), vexp_w(bd.region_words);
        check(qwen_kv_region_from_dense(bd, kexp, capacity, kexp_w).ok(), "kv dtype: append expectation K");
        check(qwen_kv_region_from_dense(bd, vexp, capacity, vexp_w).ok(), "kv dtype: append expectation V");
        luisa::vector<uint32_t> kstart = kw, vstart = vw;// host copies of the initial caches
        auto bkrow = upload(dev, stream, krows);
        auto bvrow = upload(dev, stream, vrows);
        auto bkapp = upload(dev, stream, kstart);
        auto bvapp = upload(dev, stream, vstart);
        auto dummy_u = dev.create_buffer<uint32_t>(1);
        auto append_plan = plan_qwen_kv_append_packed_dispatch(rows, kvh, hd, bf16);
    check(append_plan.ok(), "kv dtype: append plan");
    check(append_plan.ok() &&
              append_plan.value->work_items == rows * kvh * (hd / 2u),
          "kv dtype: append plan counts words, not 128-thread blocks");
    auto sh_append = dev.compile(make_qwen_kv_append_packed_kernel(bf16, false));
        stream << sh_append(bkrow, bvrow, bkapp.view(), bvapp.view(), dummy_u.view(),
                            rows, kvh, hd, pos0, capacity)
                      .dispatch(qwen_dispatch_threads(*append_plan.value));
        const auto got_k = download(stream, bkapp, bd.region_words);
        const auto got_v = download(stream, bvapp, bd.region_words);
        stream << synchronize();
        check(got_k == kexp_w, "kv dtype: packed bf16 append writes exactly the host-encoded K words");
        check(got_v == vexp_w, "kv dtype: packed bf16 append writes exactly the host-encoded V words");
        // Device-state arm used inside the static CUDA graph: pos0 comes from
        // state[position], so a deliberately different host scalar is ignored.
        auto bkapp_dev = upload(dev, stream, kstart);
        auto bvapp_dev = upload(dev, stream, vstart);
        luisa::vector<uint32_t> state(qwen_dec_state_count, 0u);
        state[qwen_dec_state_position] = pos0;
        state[qwen_dec_state_kv_len] = pos0 + rows;
        auto bstate = upload(dev, stream, state);
        auto sh_append_dev = dev.compile(make_qwen_kv_append_packed_kernel(bf16, true));
        stream << sh_append_dev(bkrow, bvrow, bkapp_dev.view(), bvapp_dev.view(),
                                bstate.view(), rows, kvh, hd, 999u, capacity)
                      .dispatch(qwen_dispatch_threads(*append_plan.value));
        const auto got_kd = download(stream, bkapp_dev, bd.region_words);
        const auto got_vd = download(stream, bvapp_dev, bd.region_words);
        check(got_kd == kexp_w && got_vd == vexp_w,
              "kv dtype: device-state packed append ignores host pos0 and matches the graph-free words");
        // The SAME kernel at f32 must be the identity dense append.
        auto bkf = upload(dev, stream, kc);
        auto bvf = upload(dev, stream, vc);
        auto f32_plan = plan_qwen_kv_append_packed_dispatch(rows, kvh, hd, f32);
        check(f32_plan.ok(), "kv dtype: f32 append plan");
        auto sh_append_f32 = dev.compile(make_qwen_kv_append_packed_kernel(f32, false));
        stream << sh_append_f32(bkrow, bvrow, bkf.view().as<uint32_t>(),
                                bvf.view().as<uint32_t>(), dummy_u.view(),
                                rows, kvh, hd, pos0, capacity)
                      .dispatch(qwen_dispatch_threads(*f32_plan.value));
        const auto got_kf = download(stream, bkf, fd.region_words);
        const auto got_vf = download(stream, bvf, fd.region_words);
        stream << synchronize();
        check(got_kf == kexp, "kv dtype: f32 append through the packed kernel is the dense append (K)");
        check(got_vf == vexp, "kv dtype: f32 append through the packed kernel is the dense append (V)");
    }
}


// ---------------------------------------------------------------------------
// fp8_e4m3 KV: codec differential, ONLINE-SCALE append differential and the
// three-way attention comparison (opt_plan/05-quantization.md Q3, FP8 step).
//
// What makes this test the contract rather than a smoke test:
//  * the device E4M3 codec is compared BIT-FOR-BIT against qwen::f32_to_fp8_e4m3 /
//    fp8_e4m3_to_f32 over all 256 codes, their neighbours, the ties, the
//    subnormal/saturation/NaN edges and a random sweep -- the append kernel and
//    the weight path share that codec, so a drift here would silently move every
//    stored value;
//  * the append kernel is compared WORD-FOR-WORD (codes and scales) against
//    qwen_kv_append_reference over a sequence that forces the block maximum to
//    GROW (old codes must be rescaled) and to stay put (old bytes untouched), over
//    a 64-position block boundary, over a ragged tail and in the device-state arm
//    a static graph uses;
//  * attention is compared three ways at the block and split boundaries: FP32 vs
//    the dense oracle, BF16 vs its dequantized image, FP8 vs ITS dequantized image
//    (tight, i.e. the kernels read the right block scale) and FP32 vs FP8 within
//    the documented E4M3 storage drift (loose, never equality).
// ---------------------------------------------------------------------------
namespace {
// vk's SPIR-V may canonicalize a -0.0f produced by FNegate(+0), so an E4M3 -0 code
// (0x80) can decode as +0 there. The two are numerically IDENTICAL (zero K/V
// contributes nothing to attention), so the word comparison canonicalizes the
// zero codes and every non-zero code is still compared bit for bit.
[[nodiscard]] uint32_t canonical_fp8_word(uint32_t w) noexcept {
    uint32_t out = 0u;
    for (uint32_t j = 0u; j < 4u; ++j) {
        uint32_t byte = (w >> (8u * j)) & 0xFFu;
        if (byte == 0x80u) { byte = 0u; }
        out |= byte << (8u * j);
    }
    return out;
}
[[nodiscard]] luisa::vector<uint32_t> canonical_fp8_words(luisa::span<const uint32_t> words) {
    luisa::vector<uint32_t> out(words.size());
    for (size_t i = 0u; i < words.size(); ++i) { out[i] = canonical_fp8_word(words[i]); }
    return out;
}
}// namespace

void test_attn_kv_fp8(Device &dev, Stream &stream, Rng &rng) {
    using luisa::qwen::QwenKvDType;
    using luisa::qwen::QwenKvLayout;
    using luisa::qwen::QwenKvLayoutDesc;
    using luisa::qwen::qwen_kv_append_reference;
    using luisa::qwen::qwen_kv_fp8_code_from_f32;
    using luisa::qwen::qwen_kv_fp8_code_to_f32;
    using luisa::qwen::qwen_kv_layout_desc;
    using luisa::qwen::qwen_kv_region_to_dense;
    QwenDslConfig cfg;
    const uint32_t qh = cfg.q_heads, kvh = cfg.kv_heads, hd = cfg.attn_head_dim;
    const uint32_t kv_n = cfg.attn_kv_n();
    const uint32_t capacity = 650u;
    const float attn_scale = cfg.attn_scale();
    const QwenKvStorage f32{kv_n == 0u ? QwenKvDType::f32 : QwenKvDType::f32,
                            QwenKvLayout::position_major};
    const QwenKvStorage bf16 = qwen_kv_storage_native(QwenKvDType::bf16);
    const QwenKvStorage fp8 = qwen_kv_storage_native(QwenKvDType::fp8_e4m3);
    QwenKvLayoutDesc fd{}, bd{}, d8{};
    check(qwen_kv_layout_desc(f32.dtype, f32.layout, capacity, kvh, hd, fd).ok() &&
              qwen_kv_layout_desc(bf16.dtype, bf16.layout, capacity, kvh, hd, bd).ok() &&
              qwen_kv_layout_desc(fp8.dtype, fp8.layout, capacity, kvh, hd, d8).ok(),
          "kv fp8: the three storage descs build");
    check(d8.elements_per_word() == 4u && d8.scale_count == kvh * d8.position_blocks &&
              d8.scale_bytes() == d8.scale_count * sizeof(float) &&
              d8.payload_bytes() * 4u == fd.payload_bytes(),
          "kv fp8: contract geometry (quads, scales, a quarter of the fp32 payload)");

    // ---- A. device codec vs the host codec, bit for bit --------------------
    {
        auto sh_enc = dev.compile(make_qwen_kv_fp8_encode_kernel());
        auto sh_dec = dev.compile(make_qwen_kv_fp8_decode_kernel());
        luisa::vector<float> values;
        values.reserve(256u * 4u + 4096u);
        for (uint32_t code = 0u; code < 256u; ++code) {
            const auto v = qwen_kv_fp8_code_to_f32(static_cast<uint8_t>(code));
            values.emplace_back(v);
            values.emplace_back(std::nextafter(v, -std::numeric_limits<float>::infinity()));
            values.emplace_back(std::nextafter(v, std::numeric_limits<float>::infinity()));
            values.emplace_back(-v);
        }
        const float fixed[] = {0.0f, -0.0f, std::ldexp(1.0f, -9), std::ldexp(1.0f, -10),
                              3.0f * std::ldexp(1.0f, -10), std::ldexp(1.0f, -6),
                              1.0f + 1.0f / 16.0f, 1.0f + 3.0f / 16.0f, 448.0f, 448.0001f,
                              1.0e30f, -1.0e30f, std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()};
        values.insert(values.end(), std::begin(fixed), std::end(fixed));
        luisa::vector<float> random(4096u);
        rng.fill(random, -1000.0f, 1000.0f);
        values.insert(values.end(), random.begin(), random.end());
        auto bv = upload(dev, stream, values);
        auto bc = dev.create_buffer<uint32_t>(values.size());
        auto plan = plan_qwen_kv_fp8_codec_dispatch(values.size());
        check(plan.ok(), "kv fp8: codec plan");
        stream << sh_enc(bv, bc, static_cast<uint32_t>(values.size()))
                      .dispatch(qwen_dispatch_threads(*plan.value));
        const auto codes = download(stream, bc, values.size());
        bool enc_exact = true;
        uint32_t first_bad = 0u;
        for (size_t k = 0u; k < values.size(); ++k) {
            const auto want = static_cast<uint32_t>(qwen_kv_fp8_code_from_f32(values[k]));
            if (codes[k] != want) {
                enc_exact = false;
                first_bad = static_cast<uint32_t>(k);
                break;
            }
        }
        check(enc_exact, "kv fp8: device encode is bit-identical to qwen::f32_to_fp8_e4m3",
              luisa::string{"first mismatch at "} + std::to_string(first_bad).c_str());
        luisa::vector<uint32_t> all_codes(256u);
        for (uint32_t code = 0u; code < 256u; ++code) { all_codes[code] = code; }
        auto bca = upload(dev, stream, all_codes);
        auto bfv = dev.create_buffer<float>(256u);
        stream << sh_dec(bca, bfv, 256u).dispatch(qwen_dispatch_threads(*plan.value));
        const auto decoded = download(stream, bfv, 256u);
        bool dec_exact = true;
        for (uint32_t code = 0u; code < 256u; ++code) {
            const auto want = qwen_kv_fp8_code_to_f32(static_cast<uint8_t>(code));
            const auto got = decoded[code];
            if (std::isnan(want) || std::isnan(got)) {
                dec_exact = dec_exact && std::isnan(want) && std::isnan(got);
            } else if (want == 0.0f) {
                // vk's SPIR-V may canonicalize the -0 of code 0x80 to +0; the two are
                // numerically identical, so only the value is compared for zeros.
                dec_exact = dec_exact && got == 0.0f;
            } else {
                dec_exact = dec_exact && want == got && std::signbit(want) == std::signbit(got);
            }
        }
        check(dec_exact, "kv fp8: device decode matches fp8_e4m3_to_f32 for all 256 codes");
        }

    // ---- B. online append vs qwen_kv_append_reference ----------------------
    const uint64_t region_words = d8.region_words;
    const uint64_t scale_count = d8.scale_count;
    luisa::vector<uint32_t> host_kw(region_words, 0u), host_vw(region_words, 0u);
    luisa::vector<float> host_ks(scale_count, 0.0f), host_vs(scale_count, 0.0f);
    auto bkw = dev.create_buffer<uint32_t>(region_words);
    auto bvw = dev.create_buffer<uint32_t>(region_words);
    auto bks = dev.create_buffer<float>(scale_count);
    auto bvs = dev.create_buffer<float>(scale_count);
    {
        luisa::vector<uint32_t> zero_w(region_words, 0u);
        luisa::vector<float> zero_s(scale_count, 0.0f);
        stream << bkw.copy_from(luisa::span<const uint32_t>{zero_w})
               << bvw.copy_from(luisa::span<const uint32_t>{zero_w})
               << bks.copy_from(luisa::span<const float>{zero_s})
               << bvs.copy_from(luisa::span<const float>{zero_s}) << synchronize();
    }
    // The append kernel's contract: fast math OFF. Its code path contains the
    // v/scale division, which a fast-math build lowers to an approximate
    // reciprocal and would move a code LSB away from the CPU oracle. The engine
    // compiles every KV kernel with the same option (dsl_engine.cpp compile_all).
    compute::ShaderOption fp8_opt;
    fp8_opt.enable_fast_math = false;
    auto sh_append = dev.compile(make_qwen_kv_append_fp8_kernel(fp8, false), fp8_opt);
    auto sh_append_dev = dev.compile(make_qwen_kv_append_fp8_kernel(fp8, true), fp8_opt);
    {
        struct Step {
            uint32_t pos0, tokens;
            float lo, hi;
        };
        // The sequence forces block 0 to GROW twice, block 1 to grow once and then
        // take a NON-GROWING append (its bytes must not move), crosses the
        // 64-position boundary at 63, appends a single token to a fresh block and
        // ends with a longer chunk that grows blocks 1-2 again.
        const Step steps[] = {{0u, 63u, -1.0f, 1.0f},
                              {63u, 2u, -4.0f, 4.0f},
                              {65u, 10u, -0.5f, 0.5f},
                              {75u, 30u, -9.0f, 9.0f},
                              {105u, 1u, -0.25f, 0.25f},
                              {106u, 40u, -1.0f, 1.0f}};
        bool all_exact = true;
        luisa::string detail;
        for (const auto &step : steps) {
            luisa::vector<float> krows(static_cast<size_t>(step.tokens) * kv_n);
            luisa::vector<float> vrows(krows.size());
            rng.fill(krows, step.lo, step.hi);
            rng.fill(vrows, step.lo, step.hi);
            auto bkrows = upload(dev, stream, krows);
            auto bvrows = upload(dev, stream, vrows);
            auto plan = plan_qwen_kv_append_fp8_dispatch(step.pos0, step.tokens, kvh, fp8);
            check(plan.ok(), "kv fp8: append plan");
            auto dummy_state = dev.create_buffer<uint32_t>(qwen_dec_state_count);
            stream << sh_append(bkrows, bvrows, bkw, bvw, bks, bvs, dummy_state,
                                step.tokens, kvh, hd, step.pos0, capacity)
                          .dispatch(qwen_dispatch_threads(*plan.value));
            check(qwen_kv_append_reference(d8, krows, step.pos0, step.tokens, host_ks, host_kw).ok() &&
                      qwen_kv_append_reference(d8, vrows, step.pos0, step.tokens, host_vs, host_vw).ok(),
                  "kv fp8: host append oracle");
            const auto got_kw = download(stream, bkw, region_words);
            const auto got_vw = download(stream, bvw, region_words);
            const auto got_ks = download(stream, bks, scale_count);
            const auto got_vs = download(stream, bvs, scale_count);
            const auto tag = luisa::string{"pos="} + std::to_string(step.pos0).c_str() +
                             "/t=" + std::to_string(step.tokens).c_str();
            const bool step_ok = canonical_fp8_words(got_kw) == canonical_fp8_words(host_kw) &&
                                 canonical_fp8_words(got_vw) == canonical_fp8_words(host_vw) &&
                                 got_ks == host_ks && got_vs == host_vs;
            if (!step_ok) {
                detail = luisa::string{"append mismatch at "} + tag;
                for (size_t k = 0u; k < scale_count; ++k) {
                    if (got_ks[k] != host_ks[k]) {
                        detail += luisa::string{"; first K scale "} + std::to_string(k).c_str();
                        break;
                    }
                }
                for (size_t k = 0u; k < region_words; ++k) {
                    if (got_kw[k] != host_kw[k]) {
                        detail += luisa::string{"; first K word "} + std::to_string(k).c_str();
                        break;
                    }
                }
                for (size_t k = 0u; k < region_words; ++k) {
                    if (got_vw[k] != host_vw[k]) {
                        detail += luisa::string{"; first V word "} + std::to_string(k).c_str();
                        break;
                    }
                }
            }
            check(step_ok, (luisa::string{"kv fp8: append words+scales match the host oracle "} + tag).c_str(),
                  detail);
            all_exact = all_exact && step_ok;
            // Every block with a valid position carries a scale, and no block beyond
            // the last valid one does: a scale read is never unmapped and never stale.
            bool scales_sane = true;
            const uint32_t valid = step.pos0 + step.tokens;
            for (uint32_t head = 0u; head < kvh; ++head) {
                for (uint32_t b = 0u; b < d8.position_blocks; ++b) {
                    const auto written = b * 64u < valid;
                    const auto s = got_ks[static_cast<size_t>(head) * d8.position_blocks + b];
                    scales_sane = scales_sane && (written ? s > 0.0f : s == 0.0f);
                }
            }
            check(scales_sane, (luisa::string{"kv fp8: only written blocks carry a scale "} + tag).c_str());
        }
        check(all_exact, "kv fp8: the whole append sequence is word-exact");
        // Device-state arm: the position comes from state[position] and the host
        // scalar is deliberately wrong (the static CUDA graph relies on this).
        {
            const uint32_t pos0 = 300u, tokens = 20u;
            luisa::vector<float> krows(static_cast<size_t>(tokens) * kv_n), vrows(krows.size());
            rng.fill(krows, -3.0f, 3.0f);
            rng.fill(vrows, -3.0f, 3.0f);
            auto bkrows = upload(dev, stream, krows);
            auto bvrows = upload(dev, stream, vrows);
            luisa::vector<uint32_t> state(qwen_dec_state_count, 0u);
            state[qwen_dec_state_position] = pos0;
            state[qwen_dec_state_kv_len] = pos0 + tokens;
            auto bstate = upload(dev, stream, state);
            luisa::vector<uint32_t> host_kw2 = host_kw, host_vw2 = host_vw;
            luisa::vector<float> host_ks2 = host_ks, host_vs2 = host_vs;
            check(qwen_kv_append_reference(d8, krows, pos0, tokens, host_ks2, host_kw2).ok() &&
                      qwen_kv_append_reference(d8, vrows, pos0, tokens, host_vs2, host_vw2).ok(),
                  "kv fp8: device-state host oracle");
            auto bkw2 = upload(dev, stream, host_kw);
            auto bvw2 = upload(dev, stream, host_vw);
            auto bks2 = upload(dev, stream, host_ks);
            auto bvs2 = upload(dev, stream, host_vs);
            auto plan = plan_qwen_kv_append_fp8_dispatch(pos0, tokens, kvh, fp8);
            stream << sh_append_dev(bkrows, bvrows, bkw2, bvw2, bks2, bvs2, bstate,
                                    tokens, kvh, hd, 9999u, capacity)
                          .dispatch(qwen_dispatch_threads(*plan.value));
            check(canonical_fp8_words(download(stream, bkw2, region_words)) ==
                          canonical_fp8_words(host_kw2) &&
                      canonical_fp8_words(download(stream, bvw2, region_words)) ==
                          canonical_fp8_words(host_vw2) &&
                      download(stream, bks2, scale_count) == host_ks2 &&
                      download(stream, bvs2, scale_count) == host_vs2,
                  "kv fp8: device-state append ignores the host position and matches the graph-free words");
            host_kw = std::move(host_kw2);
            host_vw = std::move(host_vw2);
            host_ks = std::move(host_ks2);
            host_vs = std::move(host_vs2);
        }
    }

    // Non-finite-only block: invalid model data must not disappear as zero.
    // A full-scale contribution gives the block scale 1, so inf saturates and
    // NaN keeps its E4M3 NaN code instead of producing a 0/0 scale.
    {
        luisa::vector<float> kbad(kv_n, 0.0f), vbad(kv_n, 0.0f);
        kbad[0] = std::numeric_limits<float>::infinity();
        kbad[1] = std::numeric_limits<float>::quiet_NaN();
        luisa::vector<uint32_t> refw(region_words, 0u);
        luisa::vector<float> refs(scale_count, 0.0f);
        check(qwen_kv_append_reference(d8, kbad, 0u, 1u, refs, refw).ok(),
              "kv fp8: non-finite host append oracle");
        check(refs[0] == 1.0f && (refw[0] & 0xFFu) == 0x7eu &&
                  ((refw[0] >> 8u) & 0xFFu) == 0x7fu,
              "kv fp8: non-finite-only block gets scale 1 and preserves inf/NaN codes");
        auto bki = dev.create_buffer<uint32_t>(region_words);
        auto bvi = dev.create_buffer<uint32_t>(region_words);
        auto bsi = dev.create_buffer<float>(scale_count);
        auto bsv = dev.create_buffer<float>(scale_count);
        auto bkrows = upload(dev, stream, kbad);
        auto bvrows = upload(dev, stream, vbad);
        // The append kernel folds the block scale as max(STORED, new), so the
        // stored scale is an INPUT: a fresh cache block starts at 0 and must be
        // written as such. Without this the arm reads whatever the allocator
        // returned (latent since the FP8 KV phase; it only showed up once the
        // surrounding allocation sequence changed).
        {
            luisa::vector<uint32_t> zero_i(region_words, 0u);
            luisa::vector<float> zero_is(scale_count, 0.0f);
            stream << bki.copy_from(luisa::span<const uint32_t>{zero_i})
                   << bvi.copy_from(luisa::span<const uint32_t>{zero_i})
                   << bsi.copy_from(luisa::span<const float>{zero_is})
                   << bsv.copy_from(luisa::span<const float>{zero_is}) << synchronize();
        }
        auto dummy = dev.create_buffer<uint32_t>(qwen_dec_state_count);
        auto plan = plan_qwen_kv_append_fp8_dispatch(0u, 1u, kvh, fp8);
        stream << sh_append(bkrows, bvrows, bki, bvi, bsi, bsv, dummy,
                            1u, kvh, hd, 0u, capacity)
                      .dispatch(qwen_dispatch_threads(*plan.value));
        const auto word = download(stream, bki, 1u);
        const auto scale = download(stream, bsi, 1u);
        check(word[0] == refw[0] && scale[0] == 1.0f,
              "kv fp8: device non-finite append matches the host code and scale");
    }

    // ---- C. three-way attention at the block and split boundaries ----------
    {
        const uint64_t dense_n = fd.region_words;
        luisa::vector<float> kc(dense_n), vc(dense_n);
        rng.fill(kc, -1.0f, 1.0f);
        rng.fill(vc, -1.0f, 1.0f);
        luisa::vector<uint32_t> bkw8(bd.region_words), bvw8(bd.region_words);
        luisa::vector<float> bks8(bd.scale_count), bvs8(bd.scale_count);
        check(qwen_kv_region_from_dense(bd, bks8, kc, capacity, bkw8).ok() &&
                  qwen_kv_region_from_dense(bd, bvs8, vc, capacity, bvw8).ok(),
              "kv fp8: dense -> bf16 fixture");
        luisa::vector<uint32_t> fkw(d8.region_words), fvw(d8.region_words);
        luisa::vector<float> fks(d8.scale_count), fvs(d8.scale_count);
        check(qwen_kv_region_from_dense(d8, fks, kc, capacity, fkw).ok() &&
                  qwen_kv_region_from_dense(d8, fvs, vc, capacity, fvw).ok(),
              "kv fp8: dense -> fp8 fixture");
        luisa::vector<float> kq(dense_n), vq(dense_n);
        check(qwen_kv_region_to_dense(d8, fkw, fks, capacity, kq).ok() &&
                  qwen_kv_region_to_dense(d8, fvw, fvs, capacity, vq).ok(),
              "kv fp8: fp8 -> dequantized oracle image");
        {
            bool exact = true;
            for (uint32_t head = 0u; head < kvh && exact; ++head) {
                for (uint32_t pos = 0u; pos < capacity && exact; ++pos) {
                    const auto s = fks[d8.scale_index(head, pos)];
                    for (uint32_t c = 0u; c < hd && exact; ++c) {
                        const auto code = d8.code_index(head, pos, c);
                        const auto byte = static_cast<uint8_t>(
                            (fkw[d8.word_index(code)] >> d8.word_shift(code)) & 0xFFu);
                        const auto at = static_cast<uint64_t>(pos) * kv_n +
                                        static_cast<uint64_t>(head) * hd + c;
                        exact = qwen_kv_fp8_code_to_f32(byte) * s == kq[at];
                    }
                }
            }
            check(exact, "kv fp8: the dequantized image is code * its own block scale");
        }
        auto bkcf = upload(dev, stream, kc);
        auto bvcf = upload(dev, stream, vc);
        auto bkwb = upload(dev, stream, bkw8);
        auto bvwb = upload(dev, stream, bvw8);
        auto bksb = upload(dev, stream, bks8);
        auto bvsb = upload(dev, stream, bvs8);
        auto bkwf8 = upload(dev, stream, fkw);
        auto bvwf8 = upload(dev, stream, fvw);
        auto bksf = upload(dev, stream, fks);
        auto bvsf = upload(dev, stream, fvs);
        auto bk8f = bkwf8.view().as<float>();
        auto bv8f = bvwf8.view().as<float>();
        auto bkb = bkwb.view().as<float>();
        auto bvb = bvwb.view().as<float>();
        auto bdummy = dev.create_buffer<float>(1);
        {
            auto sh_leg_f = dev.compile(make_qwen_attn_prefill_kernel(f32));
            auto sh_leg_b = dev.compile(make_qwen_attn_prefill_kernel(bf16));
            auto sh_leg_8 = dev.compile(make_qwen_attn_prefill_kernel(fp8));
            QwenAttnPrefillTiledConfig tcfg{};
            check(select_qwen35_attn_prefill_tiled_config(32u, 49152u, tcfg).ok(), "kv fp8: tiled cfg");
            auto sh_til_f = dev.compile(make_qwen35_attn_prefill_tiled_kernel(tcfg, f32));
            auto sh_til_b = dev.compile(make_qwen35_attn_prefill_tiled_kernel(tcfg, bf16));
            auto sh_til_8 = dev.compile(make_qwen35_attn_prefill_tiled_kernel(tcfg, fp8));
            const struct {
                uint32_t tokens, pos0;
            } cases[] = {{1u, 0u}, {2u, 62u}, {2u, 63u}, {2u, 64u}, {5u, 61u}, {7u, 60u}, {3u, 640u}};
            for (const auto &c : cases) {
                const uint32_t rows = c.tokens, pos0 = c.pos0, kv_len = pos0 + rows;
                luisa::vector<float> q(static_cast<size_t>(rows) * qh * hd), gate(q.size());
                rng.fill(q, -1.0f, 1.0f);
                rng.fill(gate, -2.0f, 2.0f);
                auto bq = upload(dev, stream, q);
                auto bg = upload(dev, stream, gate);
                auto bo = dev.create_buffer<float>(q.size());
                luisa::vector<float> ref_dense, ref_8;
                check(ref_qwen_attn_prefill(q, gate, kc, vc, rows, qh, kvh, hd, attn_scale, pos0, kv_len, ref_dense).ok() &&
                          ref_qwen_attn_prefill(q, gate, kq, vq, rows, qh, kvh, hd, attn_scale, pos0, kv_len, ref_8).ok(),
                      "kv fp8: prefill oracles");
                auto p_legacy = plan_qwen_attn_prefill_dispatch(rows, qh);
                auto p_tiled = plan_qwen35_attn_prefill_tiled_dispatch(rows, tcfg);
                const auto tag = luisa::string{"t="} + std::to_string(rows).c_str() +
                                 "/pos=" + std::to_string(pos0).c_str();
                const auto run_legacy = [&](auto &sh, auto kcv, auto vcv, auto ksc, auto vsc) {
                    stream << sh(bq, bg, kcv, vcv, ksc, vsc, bo.view(), rows, qh, kvh, hd,
                                 attn_scale, pos0, kv_len, capacity)
                                  .dispatch(qwen_dispatch_threads(*p_legacy.value));
                    return download(stream, bo, q.size());
                };
                const auto run_tiled = [&](auto &sh, auto kcv, auto vcv, auto ksc, auto vsc) {
                    stream << sh(bq, bg, kcv, vcv, ksc, vsc, bo.view(), rows, attn_scale, pos0,
                                 kv_len, capacity)
                                  .dispatch(qwen_dispatch_threads(*p_tiled.value));
                    return download(stream, bo, q.size());
                };
                const auto a_legacy = run_legacy(sh_leg_f, bkcf.view(), bvcf.view(), bdummy.view(), bdummy.view());
                const auto b_legacy = run_legacy(sh_leg_b, bkb, bvb, bdummy.view(), bdummy.view());
                const auto c_legacy = run_legacy(sh_leg_8, bk8f, bv8f, bksf.view(), bvsf.view());
                const auto a_tiled = run_tiled(sh_til_f, bkcf.view(), bvcf.view(), bdummy.view(), bdummy.view());
                const auto b_tiled = run_tiled(sh_til_b, bkb, bvb, bdummy.view(), bdummy.view());
                const auto c_tiled = run_tiled(sh_til_8, bk8f, bv8f, bksf.view(), bvsf.view());
                stream << synchronize();
                check_close(a_legacy, ref_dense, 2e-5f, 2e-4f, (luisa::string{"kv fp8 f32 prefill legacy "} + tag).c_str());
                check_close(a_tiled, ref_dense, 2e-5f, 2e-4f, (luisa::string{"kv fp8 f32 prefill tiled "} + tag).c_str());
                check_close(b_legacy, a_legacy, 5e-2f, 5e-2f, (luisa::string{"kv fp8 bf16-vs-f32 prefill legacy "} + tag).c_str());
                check_close(c_legacy, ref_8, 2e-5f, 2e-4f, (luisa::string{"kv fp8 prefill legacy vs dequantized oracle "} + tag).c_str());
                check_close(c_tiled, ref_8, 2e-5f, 2e-4f, (luisa::string{"kv fp8 prefill tiled vs dequantized oracle "} + tag).c_str());
                // FP32 vs FP8: the documented E4M3 storage drift (6.25 % per value),
                // never equality. The point is that it is BOUNDED.
                check_close(c_legacy, a_legacy, 2.5e-1f, 2.5e-1f, (luisa::string{"kv fp8 f32-vs-fp8 prefill legacy "} + tag).c_str());
                check_close(c_tiled, a_tiled, 2.5e-1f, 2.5e-1f, (luisa::string{"kv fp8 f32-vs-fp8 prefill tiled "} + tag).c_str());
            }
        }
        {
            QwenAttnDecodeSplitConfig scfg{};
            scfg.threshold = 8u;
            scfg.split_chunk = 16u;
            scfg.max_splits = 4u;
            scfg.k_tile = 4u;
            check(validate_qwen_attn_decode_split_config(scfg).ok(), "kv fp8: split config");
            auto sh_leg_f = dev.compile(make_qwen_attn_decode_kernel(0u, f32));
            auto sh_leg_8 = dev.compile(make_qwen_attn_decode_kernel(0u, fp8));
            auto sh_split_8 = dev.compile(make_qwen_attn_decode_split_kernel(scfg, fp8));
            auto sh_split_f = dev.compile(make_qwen_attn_decode_split_kernel(scfg, f32));
            auto sh_comb = dev.compile(make_qwen_attn_decode_combine_kernel(scfg));
            const uint64_t acc_n = static_cast<uint64_t>(scfg.max_splits) * qh * hd;
            const uint64_t ml_n = static_cast<uint64_t>(scfg.max_splits) * qh * 2u;
            auto bacc = dev.create_buffer<float>(static_cast<size_t>(acc_n));
            auto bml = dev.create_buffer<float>(static_cast<size_t>(ml_n));
            luisa::vector<float> q(qh * hd), gate(qh * hd);
            rng.fill(q, -1.0f, 1.0f);
            rng.fill(gate, -2.0f, 2.0f);
            auto bq = upload(dev, stream, q);
            auto bg = upload(dev, stream, gate);
            auto bo = dev.create_buffer<float>(qh * hd);
            auto decode_plan = plan_qwen_attn_decode_dispatch(qh);
            auto split_plan = plan_qwen_attn_decode_split_dispatch(scfg);
            auto comb_plan = plan_qwen_attn_decode_combine_dispatch(scfg);
            for (const uint32_t kv_len : {1u, 63u, 64u, 65u, 127u, 128u, 129u, 511u, 512u, 513u, 640u, 650u}) {
                luisa::vector<float> ref_dense, ref_8;
                check(ref_qwen_attn_decode(q, gate, kc, vc, qh, kvh, hd, attn_scale, kv_len, ref_dense).ok() &&
                          ref_qwen_attn_decode(q, gate, kq, vq, qh, kvh, hd, attn_scale, kv_len, ref_8).ok(),
                      "kv fp8: decode oracles");
                const auto tag = luisa::string{"kv_len="} + std::to_string(kv_len).c_str();
                const auto decode_one = [&](auto &sh, auto kcv, auto vcv, auto ksc, auto vsc) {
                    stream << sh(bq, bg, kcv, vcv, ksc, vsc, bo.view(), qh, kvh, hd, attn_scale,
                                 kv_len, capacity)
                                  .dispatch(qwen_dispatch_threads(*decode_plan.value));
                    return download(stream, bo, qh * hd);
                };
                const auto a = decode_one(sh_leg_f, bkcf.view(), bvcf.view(), bdummy.view(), bdummy.view());
                const auto c = decode_one(sh_leg_8, bk8f, bv8f, bksf.view(), bvsf.view());
                stream << synchronize();
                check_close(a, ref_dense, 2e-5f, 2e-4f, (luisa::string{"kv fp8 decode f32 "} + tag).c_str());
                check_close(c, ref_8, 2e-5f, 2e-4f, (luisa::string{"kv fp8 decode vs dequantized oracle "} + tag).c_str());
                check_close(c, a, 2.5e-1f, 2.5e-1f, (luisa::string{"kv fp8 f32-vs-fp8 decode "} + tag).c_str());
                if (kv_len >= scfg.threshold) {
                    const auto split_one = [&](auto &shs, auto kcv, auto vcv, auto ksc, auto vsc) {
                        stream << shs(bq, kcv, vcv, ksc, vsc, bacc.view(), bml.view(), kv_len,
                                      attn_scale, capacity)
                                      .dispatch(qwen_dispatch_threads(*split_plan.value));
                        stream << sh_comb(bacc.view(), bml.view(), bg, bo.view(), kv_len, capacity)
                                      .dispatch(qwen_dispatch_threads(*comb_plan.value));
                        return download(stream, bo, qh * hd);
                    };
                    const auto as = split_one(sh_split_f, bkcf.view(), bvcf.view(), bdummy.view(), bdummy.view());
                    const auto cs = split_one(sh_split_8, bk8f, bv8f, bksf.view(), bvsf.view());
                    stream << synchronize();
                    check_close(as, ref_dense, 2e-5f, 2e-4f, (luisa::string{"kv fp8 split f32 "} + tag).c_str());
                    check_close(cs, ref_8, 2e-5f, 2e-4f, (luisa::string{"kv fp8 split vs dequantized oracle "} + tag).c_str());
                    check_close(cs, as, 2.5e-1f, 2.5e-1f, (luisa::string{"kv fp8 f32-vs-fp8 split "} + tag).c_str());
                }
            }
        }
    }

    }

// ---------------------------------------------------------------------------
// OFFLINE MERGED decode input projections (opt_plan/03 section 3 + D5).
//
// The merged route must be INDISTINGUISHABLE from the unmerged one: the same
// per-output FP32 accumulation order, so the destinations are BIT-IDENTICAL.
// This test therefore compares, on the SAME packed bytes:
//   merged  : ONE launch over the row-concatenated [N, K] matrix
//             (make_qwen35_merged_{gdn,attn}_gemv_kernel),
//   unmerged: one decode GEMV launch per segment (the engine's own realization:
//             the generic packed pair loop for BF16, the four-stream decode for
//             FP8/Q8_0) plus make_qwen_gdn_decode_gates_kernel for the GDN
//             alpha/beta rows,
// and requires EXACT equality of every destination buffer. No tolerance: the
// merged kernel is contracted to preserve the reduction order, and a tolerance
// here would hide exactly the regression this test exists for.
// The segments are RAGGED on purpose (not multiples of the 4-warp block and not
// the model widths), so blocks mix segments and the routing is exercised off the
// friendly boundaries. K stays 2560 so each format's packed contract holds; both
// merged widths are therefore non-multiples of 4 (12346 and 10236).
// ---------------------------------------------------------------------------
void test_merged_decode_gemv(Device &dev, Stream &stream, Rng &rng) {
    constexpr uint32_t K = 2560u;
    constexpr uint32_t G_QKV = 8190u, G_Z = 4094u, G_H = 31u;// N = 12346 (ragged)
    constexpr uint32_t A_Q = 8192u, A_KV = 1022u;           // N = 10236 (ragged)
    const auto label = [](QwenWeightFormat f) {
        return f == QwenWeightFormat::bf16    ? "bf16"
               : f == QwenWeightFormat::q8_0 ? "q8_0"
                                             : "fp8";
    };
    for (auto format : {QwenWeightFormat::bf16, QwenWeightFormat::fp8_e4m3, QwenWeightFormat::q8_0}) {
        const uint32_t group = format == QwenWeightFormat::q8_0 ? kQwenQ8BlockSize : 64u;
        const uint32_t pack = qwen_weight_pack(format);
        const bool generic = format == QwenWeightFormat::bf16;
        // One host-side packer for the whole merged matrix: the merge is a ROW
        // concatenation, so packing the concatenated rows is exactly the pack of
        // the segments in order (that is the property the offline packer relies on).
        const auto pack_matrix = [&](luisa::span<const float> w, uint32_t n,
                                     luisa::vector<uint32_t> &words,
                                     luisa::vector<float> &scales) {
            if (format != QwenWeightFormat::q8_0) {
                return host_pack_matrix(format, w, K, n, group, words, scales);
            }
            // Native Q8_0 payload: full-range signed codes + exactly representable
            // FP16 block scales, then the block ABI split (never re-quantized).
            luisa::vector<uint8_t> codes(static_cast<size_t>(K) * n);
            luisa::vector<uint16_t> bits(static_cast<size_t>(K) * n / group);
            for (auto &c : codes) { c = static_cast<uint8_t>(rng.next() & 0xFFu); }
            // Small, exactly representable FP16 block scales: the point of this arm is
            // the merged ROUTING, not the epilogue's saturation behaviour, and a coarse
            // scale would drive sigmoid/softplus to their exact 0/1 limits (which is
            // correct, but makes a range assertion meaningless).
            for (auto &b : bits) {
                b = luisa::qwen::f32_to_fp16(static_cast<float>(1u + (rng.next() & 0xFFu)) / 65536.0f);
            }
            return host_pack_q8_0_codes(codes, bits, K, n, words, scales);
        };
        const auto tag = [&](const char *what, const char *shape) {
            return luisa::string{what} + " " + label(format) + " " + shape;
        };

        // ---- GDN merged: [attn_qkv | attn_gate | ssm_alpha | ssm_beta] ----------
        {
            const uint32_t n = G_QKV + G_Z + 2u * G_H;
            luisa::vector<float> x(K), w(static_cast<size_t>(n) * K);
            luisa::vector<float> ssm_a(G_H), dt_bias(G_H);
            rng.fill(x, -1.f, 1.f);
            rng.fill(w, -1.f, 1.f);
            rng.fill(ssm_a, -2.f, -0.1f);
            rng.fill(dt_bias, -0.5f, 0.5f);
            luisa::vector<uint32_t> words;
            luisa::vector<float> scales;
            check(pack_matrix(w, n, words, scales).ok(), tag("merged gdn host pack", "").c_str());
            const auto words_per_row = K / pack;
            const auto scales_per_row = format == QwenWeightFormat::bf16 ? 0u : K / group;
            auto bx = upload(dev, stream, x);
            auto bw = upload(dev, stream, words);
            auto bsc = upload(dev, stream, scales.empty() ? luisa::vector<float>{0.f} : scales);
            auto bssm = upload(dev, stream, ssm_a);
            auto bdt = upload(dev, stream, dt_bias);

            // merged destinations
            auto m_qkv = dev.create_buffer<float>(G_QKV);
            auto m_z = dev.create_buffer<float>(G_Z);
            auto m_beta = dev.create_buffer<float>(G_H);
            auto m_g = dev.create_buffer<float>(G_H);
            // unmerged destinations
            auto u_qkv = dev.create_buffer<float>(G_QKV);
            auto u_z = dev.create_buffer<float>(G_Z);
            auto u_beta = dev.create_buffer<float>(G_H);
            auto u_g = dev.create_buffer<float>(G_H);

            auto merged = dev.compile(make_qwen35_merged_gdn_gemv_kernel(format));
            auto merged_plan = plan_qwen35_merged_gdn_gemv_dispatch(
                G_QKV, G_Z, G_H, K, format, group);
            check(merged_plan.ok(), tag("merged gdn plan", "").c_str());
            stream << merged(bx, bw, bsc, m_qkv, m_z, bssm, bdt, m_beta, m_g,
                             K, group, G_QKV, G_Z, G_H, 20.f)
                          .dispatch(qwen_dispatch_threads(*merged_plan.value));

            // Unmerged reference: the SAME rows, one launch per segment, through the
            // realization the engine compiles for this format, then the fused gate
            // kernel on the alpha/beta rows.
            auto gemv = generic ? dev.compile(make_qwen_gemv_kernel(format, QwenMatmulEpilogue::none))
                                : dev.compile(make_qwen35_decode_gemv_kernel(
                                      format, QwenMatmulEpilogue::none));
            const auto run_gemv = [&](uint32_t row_off, uint32_t rows, Buffer<float> &dst) {
                auto planned = plan_qwen35_decode_gemv_dispatch(rows, K, format, group);
                check(planned.ok(), tag("unmerged segment plan", "").c_str());
                const auto word_off = static_cast<size_t>(row_off) * words_per_row;
                const auto word_cnt = static_cast<size_t>(rows) * words_per_row;
                const auto scale_off = static_cast<size_t>(row_off) * scales_per_row;
                const auto scale_cnt = static_cast<size_t>(rows) * scales_per_row;
                stream << gemv(bx.view(), bw.view(word_off, word_cnt),
                               scales_per_row == 0u ? bsc.view() : bsc.view(scale_off, scale_cnt),
                               bsc.view(), dst.view(), 1u, rows, K, group)
                              .dispatch(qwen_dispatch_threads(*planned.value));
            };
            run_gemv(0u, G_QKV, u_qkv);
            run_gemv(G_QKV, G_Z, u_z);
            {
                auto gates = dev.compile(make_qwen_gdn_decode_gates_kernel(format));
                auto planned = plan_qwen_gdn_decode_gates_dispatch(G_H, K, format, group);
                check(planned.ok(), tag("unmerged gates plan", "").c_str());
                const auto a_off = static_cast<size_t>(G_QKV + G_Z) * words_per_row;
                const auto a_cnt = static_cast<size_t>(G_H) * words_per_row;
                const auto b_off = a_off + a_cnt;
                const auto as_off = static_cast<size_t>(G_QKV + G_Z) * scales_per_row;
                const auto as_cnt = static_cast<size_t>(G_H) * scales_per_row;
                stream << gates(bx.view(),
                                bw.view(a_off, a_cnt),
                                scales_per_row == 0u ? bsc.view() : bsc.view(as_off, as_cnt),
                                bw.view(b_off, a_cnt),
                                scales_per_row == 0u ? bsc.view() : bsc.view(as_off + as_cnt, as_cnt),
                                bssm, bdt, u_beta, u_g, K, G_H, group, 20.f)
                              .dispatch(qwen_dispatch_threads(*planned.value));
            }

            check_exact(download(stream, m_qkv, G_QKV), download(stream, u_qkv, G_QKV),
                        tag("merged gdn QKV bit-identical to the per-segment GEMV", "").c_str());
            check_exact(download(stream, m_z, G_Z), download(stream, u_z, G_Z),
                        tag("merged gdn Z bit-identical to the per-segment GEMV", "").c_str());
            check_exact(download(stream, m_beta, G_H), download(stream, u_beta, G_H),
                        tag("merged gdn beta bit-identical to the fused gate kernel", "").c_str());
            check_exact(download(stream, m_g, G_H), download(stream, u_g, G_H),
                        tag("merged gdn g bit-identical to the fused gate kernel", "").c_str());

            // Epilogue sanity on the merged kernel's own gate outputs: the alpha
            // segment's nonlinearity must land in the same range the fused kernel
            // produces (ssm_a < 0 and softplus > 0 => g <= 0; sigmoid in (0, 1]).
            {
                const auto g_got = download(stream, m_g, G_H);
                const auto b_got = download(stream, m_beta, G_H);
                luisa::string why;
                for (uint32_t h = 0; h < G_H; h++) {
                    const auto gj = g_got[h];
                    const auto bj = b_got[h];
                    if (!(std::isfinite(gj) && gj <= 0.0f &&
                          std::isfinite(bj) && bj >= 0.0f && bj <= 1.0f)) {
                        char buf[192];
                        std::snprintf(buf, sizeof(buf), "col %u: g=%.9g beta=%.9g ssm_a=%.9g dt=%.9g",
                                      h, double(gj), double(bj), double(ssm_a[h]), double(dt_bias[h]));
                        why = buf;
                        break;
                    }
                }
                check(why.empty(), tag("merged gdn gate epilogue range", "").c_str(), why);
            }
        }

        // ---- attention merged: [attn_q | attn_k | attn_v] ----------------------
        {
            const uint32_t n = A_Q + 2u * A_KV;
            luisa::vector<float> x(K), w(static_cast<size_t>(n) * K);
            rng.fill(x, -1.f, 1.f);
            rng.fill(w, -1.f, 1.f);
            luisa::vector<uint32_t> words;
            luisa::vector<float> scales;
            check(pack_matrix(w, n, words, scales).ok(), tag("merged attn host pack", "").c_str());
            const auto words_per_row = K / pack;
            const auto scales_per_row = format == QwenWeightFormat::bf16 ? 0u : K / group;
            auto bx = upload(dev, stream, x);
            auto bw = upload(dev, stream, words);
            auto bsc = upload(dev, stream, scales.empty() ? luisa::vector<float>{0.f} : scales);
            auto m_q = dev.create_buffer<float>(A_Q);
            auto m_k = dev.create_buffer<float>(A_KV);
            auto m_v = dev.create_buffer<float>(A_KV);
            auto u_q = dev.create_buffer<float>(A_Q);
            auto u_k = dev.create_buffer<float>(A_KV);
            auto u_v = dev.create_buffer<float>(A_KV);
            auto merged = dev.compile(make_qwen35_merged_attn_gemv_kernel(format));
            auto merged_plan = plan_qwen35_merged_attn_gemv_dispatch(A_Q, A_KV, K, format, group);
            check(merged_plan.ok(), tag("merged attn plan", "").c_str());
            stream << merged(bx, bw, bsc, m_q, m_k, m_v, K, group, A_Q, A_KV)
                          .dispatch(qwen_dispatch_threads(*merged_plan.value));
            auto gemv = generic ? dev.compile(make_qwen_gemv_kernel(format, QwenMatmulEpilogue::none))
                                : dev.compile(make_qwen35_decode_gemv_kernel(
                                      format, QwenMatmulEpilogue::none));
            const auto run_gemv = [&](uint32_t row_off, uint32_t rows, Buffer<float> &dst) {
                auto planned = plan_qwen35_decode_gemv_dispatch(rows, K, format, group);
                check(planned.ok(), tag("unmerged attn segment plan", "").c_str());
                const auto word_off = static_cast<size_t>(row_off) * words_per_row;
                const auto word_cnt = static_cast<size_t>(rows) * words_per_row;
                const auto scale_off = static_cast<size_t>(row_off) * scales_per_row;
                const auto scale_cnt = static_cast<size_t>(rows) * scales_per_row;
                stream << gemv(bx.view(), bw.view(word_off, word_cnt),
                               scales_per_row == 0u ? bsc.view() : bsc.view(scale_off, scale_cnt),
                               bsc.view(), dst.view(), 1u, rows, K, group)
                              .dispatch(qwen_dispatch_threads(*planned.value));
            };
            run_gemv(0u, A_Q, u_q);
            run_gemv(A_Q, A_KV, u_k);
            run_gemv(A_Q + A_KV, A_KV, u_v);
            check_exact(download(stream, m_q, A_Q), download(stream, u_q, A_Q),
                        tag("merged attn Q bit-identical to the per-segment GEMV", "").c_str());
            check_exact(download(stream, m_k, A_KV), download(stream, u_k, A_KV),
                        tag("merged attn K bit-identical to the per-segment GEMV", "").c_str());
            check_exact(download(stream, m_v, A_KV), download(stream, u_v, A_KV),
                        tag("merged attn V bit-identical to the per-segment GEMV", "").c_str());
        }
    }

    // ---- fail-closed plans --------------------------------------------------
    check(!plan_qwen35_merged_gdn_gemv_dispatch(8192u, 4096u, 32u, K,
                                                QwenWeightFormat::w4, 128u)
               .ok(),
          "merged gdn plan refuses W4 (no 4-bit merged realization)");
    check(!plan_qwen35_merged_attn_gemv_dispatch(0u, 1024u, K,
                                                 QwenWeightFormat::bf16, 0u)
               .ok(),
          "merged attn plan refuses an empty segment");
    check(!plan_qwen35_merged_attn_gemv_dispatch(8192u, 1024u, 2304u,
                                                 QwenWeightFormat::fp8_e4m3, 64u)
               .ok(),
          "merged attn plan refuses a K that violates the packed contract (fp8 K % 512)");
}

int main(int argc, char **argv) {
    unbuffered_output();
    if (argc < 2 || argv[1][0] == '\0') {
        std::fprintf(stderr,
                     "usage: %s <backend>\n"
                     "  backend is explicit (e.g. cuda, dx, vk); no default, no weights needed.\n",
                     argv[0]);
        return 2;
    }
    Context ctx{argv[0]};
    auto dev = ctx.create_device(argv[1]);
    if (!dev) {
        std::fprintf(stderr, "failed to create device for backend '%s' (is the backend built/enabled?)\n", argv[1]);
        return 2;
    }
    auto stream = dev.create_stream();
    std::printf("backend: %s\n", luisa::string(dev.backend_name()).c_str());
    Rng rng;
    auto dummy_f = dev.create_buffer<float>(1);

    test_matmul(dev, stream, rng, dummy_f);
    test_matmul_packed_paths(dev, stream, rng, dummy_f);
    test_qwen35_decode_gemv(dev, stream, rng, dummy_f);
  test_gemm_mt(dev, stream, rng, dummy_f);
    test_embedding(dev, stream, rng);
    test_rmsnorm(dev, stream, rng, dummy_f);
    test_rmsnorm_snapshot_ragged(dev, stream, rng, dummy_f);
    test_qwen35_wide_rmsnorm(dev, stream, rng, dummy_f);
    test_rmsnorm_reg(dev, stream, rng, dummy_f);
    test_swiglu(dev, stream, rng);
    test_ffn_decode_fusion(dev, stream, rng);
    test_greedy_argmax(dev, stream);
    test_attention(dev, stream, rng);
    test_attn_prefill_tiled_plan();
    test_attn_prefill_tiled(dev, stream, rng);
    test_decode_edges(dev, stream, rng);
    test_decode_split(dev, stream, rng);
    test_gdn_decode_gates(dev, stream, rng);
    test_gdn(dev, stream, rng);
    test_gdn_decode4_model_shape(dev, stream, rng);
    test_gdn_chunk_boundaries(dev, stream, rng);
    test_gdn_model_shape(dev, stream, rng);
    test_gdn_decode_fused(dev, stream, rng);
    test_gdn_recurrent_norm(dev, stream, rng);
    test_decode_packed_e4m3_codes(dev, stream, rng, dummy_f);
    test_decode_packed_shapes(dev, stream, rng, dummy_f);
    test_ffn_decode_packed(dev, stream, rng);
    test_gdn_decode_gates_packed(dev, stream, rng);
    test_attn_prepare_model(dev, stream, rng);
      test_attn_kv_dtype(dev, stream, rng);
      test_attn_kv_fp8(dev, stream, rng);
    test_decode_device_state(dev, stream, rng);
    test_greedy_while_conditional(dev, stream);
    test_pinned_download_ring(dev, stream);
    test_qwen_q8_native(dev, stream, rng, dummy_f);
    test_qwen_w4(dev, stream, rng, dummy_f);
 test_merged_decode_gemv(dev, stream, rng);
    test_utilities(dev, stream, rng);
    test_fixtures(dev, stream, dummy_f);

    return summarize("qwen dsl device ops", argv[1]);
}
