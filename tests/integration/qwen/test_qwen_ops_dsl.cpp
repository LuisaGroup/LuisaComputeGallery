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
// mismatch and prints located, actionable errors.
//
// Consolidated from the verified scratch harnesses (CUDA + DX variants);
// the pre-added GGUF norm-weight convention (direct multiply) is used
// throughout, per the storage contract.
#include "qwen_device_test_utils.h"

#include "qwen/dsl_ops.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <cstring>// std::memcpy (independent bf16 oracle decode)
#include <limits>// std::numeric_limits (independent fp8 NaN code)

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::qwen::dsl;
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
        stream << sh(bq, bgate, bkc, bvc, bpref, T, cfg.q_heads, cfg.kv_heads, hd,
                     cfg.attn_scale(), pos0, kv_len).dispatch(qwen_dispatch_threads(*plan.value));
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
        stream << sh(bql, bgl, bkc, bvc, bdec, cfg.q_heads, cfg.kv_heads, hd,
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
        stream << sh(bqc, bgc, bkc, bvc, bdec, cfg.q_heads, cfg.kv_heads, hd,
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

}// namespace

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
    test_embedding(dev, stream, rng);
    test_rmsnorm(dev, stream, rng, dummy_f);
    test_rmsnorm_snapshot_ragged(dev, stream, rng, dummy_f);
    test_qwen35_wide_rmsnorm(dev, stream, rng, dummy_f);
    test_swiglu(dev, stream, rng);
    test_ffn_decode_fusion(dev, stream, rng);
    test_greedy_argmax(dev, stream);
    test_attention(dev, stream, rng);
    test_decode_edges(dev, stream, rng);
    test_gdn_decode_gates(dev, stream, rng);
    test_gdn(dev, stream, rng);
    test_gdn_decode4_model_shape(dev, stream, rng);
    test_gdn_chunk_boundaries(dev, stream, rng);
    test_utilities(dev, stream, rng);
    test_fixtures(dev, stream, dummy_f);

    return summarize("qwen dsl device ops", argv[1]);
}
