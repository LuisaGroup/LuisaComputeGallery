// Qwen CUDA tensor-core PREFILL GEMM device test (tests/integration/qwen).
//
// Usage: test_qwen_native_gemm <backend>      e.g. test_qwen_native_gemm cuda
//
// The backend is EXPLICIT argv[1] and must be supplied. This suite covers ONE
// accelerator: the `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` prefill
// GEMM that `NativeShaderExt`/`NativeShaderLanguage::CUDA_NVRTC` JIT-compiles
// (opt_plan/09 N1a). No other backend has that route (dx/vk reject CUDA C++
// source, which the header documents), so on any other backend - and on a cuda
// device without the extension or without NVRTC - the suite prints a SKIP line
// and returns 0 instead of failing the DX/VK test runs it is compiled into.
//
// What is checked, all against independent oracles:
//  * the device-free contract: tiling/config validation, the fail-closed shape
//    gate, the dispatch plan, and the generated CUDA source (tiling baked in, one
//    `__global__`, the sm_80 guard);
//  * ragged M/N/K (16..129 rows, 32..130 columns, 8..72 reduction, plus the model
//    K = 2560) in BOTH epilogues against `ref_qwen_matmul` (double accumulation);
//  * the exactness property the accelerator relies on: against a reference run on
//    activations that were ALREADY rounded to BF16, the kernel must agree to FP32
//    reassociation only (<1e-4 relative), which separates implementation error
//    from the documented BF16 activation rounding;
//  * model-sized shapes against the DSL MT kernel and the reference GEMM;
//  * stream ordering and lifetime: several shapes dispatched back-to-back from ONE
//    compiled instance, an overwrite of a previously written result, a deferred
//    commit, and the residual epilogue with `y` aliasing the residual buffer;
//  * the extension is destroyed only after its dispatches completed.
//
// The same file also covers the SEPARATE CUDA native Tile IR FP8 GEMM
// (qwen/cuda_tile_gemm.h): its device-free contract, its fail-closed answer on
// every backend that cannot serve it, and - only when the process opted in with
// LUISA_CUDA_TILE_IR=1 - the typed-FP8 differential itself. When the opt-in is
// absent that arm prints a SKIP line and every other arm is unchanged.
#include "qwen_device_test_utils.h"

#include "qwen/cuda_tile_gemm.h"
#include "qwen/dsl_ops.h"
#include "qwen/native_gemm.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <algorithm>
#include <cmath>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::qwen::dsl;
using namespace qwen_test;

namespace {

constexpr uint32_t kGroup = 64u;

// Tolerances. The synthetic data here is cancellation-dominated on purpose (the
// weights are ~3e-5 and the activations are O(1)), so an ELEMENT-WISE relative
// test says nothing: two different FP32 summation orders already disagree by 1e-4
// of an output whose magnitude is 1e-4 of its summands. Both gates below are
// therefore normalized by the largest reference element and its RMS, which is the
// well-conditioned part of the output - the same metric the prefill-GEMM benchmark
// reports.
//
//  * kOracleTol: agreement with the SAME activations already rounded to BF16,
//    i.e. accumulation order only. Measured max|d|/max|ref| = 1.4e-6, so 1e-4 is
//    a tight gate that fails on any decode, indexing or fragment-layout error.
//  * kFp32Tol: the documented deviation from the FP32-activation reference, which
//    is the single BF16 rounding of the activation at the shared-tile boundary.
//    Worst measured on this data is 4.7e-2 (K = 9216); 1e-1 leaves a factor 2.
//    On the real model the prefill is greedy-id identical (see
//    docs/examples/example_qwen.md).
constexpr float kOracleTol = 1e-4f;
constexpr float kFp32Tol = 1e-1f;

[[nodiscard]] float bf16_round(float v) noexcept {
    return host_bf16_bits_to_f32(host_f32_to_bf16_bits(v));
}

// Cancellation-robust agreement check that always prints its metric.
[[nodiscard]] bool check_agreement(luisa::span<const float> got, luisa::span<const float> want,
                                   float tol, const char *name) {
    if (got.size() != want.size()) {
        check(false, name, luisa::string{"size mismatch: got "} +
                               std::to_string(got.size()).c_str() + " want " +
                               std::to_string(want.size()).c_str());
        return false;
    }
    double max_ref = 0.0, max_diff = 0.0, sum_ref = 0.0, sum_diff = 0.0;
    size_t worst = 0u;
    for (size_t i = 0; i < got.size(); i++) {
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
            check(false, name, luisa::format("index {} is not finite (got {}, want {})",
                                             i, got[i], want[i]));
            return false;
        }
        const double d = double(got[i]) - double(want[i]);
        if (std::fabs(d) > max_diff) {
            max_diff = std::fabs(d);
            worst = i;
        }
        max_ref = std::max(max_ref, std::fabs(double(want[i])));
        sum_diff += d * d;
        sum_ref += double(want[i]) * double(want[i]);
    }
    const double rel_max = max_ref > 0.0 ? max_diff / max_ref : max_diff;
    const double rel_rms = std::sqrt(sum_diff / (sum_ref + 1e-30));
    check(rel_max <= tol, name,
          luisa::format("max|d|/max|ref| {:.3e} (tol {:.0e}), rms ratio {:.3e}, worst index {} "
                        "device {:.9g} reference {:.9g}",
                        rel_max, tol, rel_rms, worst, got[worst], want[worst]));
    return rel_max <= tol;
}

// ---------------------------------------------------------------------------
// 1. device-free contract
// ---------------------------------------------------------------------------
void test_contract() {
    QwenNativeGemmConfig cfg{};
    check(validate_qwen_native_gemm_config(cfg).ok(), "native gemm default tiling accepted");
    check(cfg.block() == 256u && cfg.warps == 8u, "native gemm default is 8 warps / 256 threads");

    {
        auto bad = cfg;
        bad.block_k = 16u;
        check(!validate_qwen_native_gemm_config(bad).ok(), "native gemm refuses a 16-wide k step");
    }
    {
        auto bad = cfg;
        bad.block_m = 96u;// 96 % 64 != 0 with 8 warps / 2 n-warps
        check(!validate_qwen_native_gemm_config(bad).ok(),
              "native gemm refuses a block tile the warp grid cannot cover");
    }
    {
        auto bad = cfg;
        bad.warps = 3u;
        check(!validate_qwen_native_gemm_config(bad).ok(), "native gemm refuses an odd warp count");
    }
    {
        // The 64x64/4-warp tile is a valid alternative configuration (the measured
        // small-M arm), so it must pass the same validator.
        QwenNativeGemmConfig alt{};
        alt.block_m = 64u;
        alt.block_n = 64u;
        alt.block_k = 32u;
        alt.warps = 4u;
        check(validate_qwen_native_gemm_config(alt).ok(), "native gemm accepts the 64x64x32/4-warp tile");
    }

    check(qwen_native_gemm_supported_format(QwenWeightFormat::bf16),
          "native gemm supports bf16 weights");
    check(qwen_native_gemm_supported_format(QwenWeightFormat::fp8_e4m3),
          "native gemm supports fp8 E4M3 weights (widened into the shared tile)");
    check(validate_qwen_native_gemm_shape(16u, 1024u, 2560u, QwenWeightFormat::fp8_e4m3).ok(),
          "native gemm shape gate accepts the fp8 model geometry");
    check(!validate_qwen_native_gemm_shape(16u, 1024u, 2558u, QwenWeightFormat::fp8_e4m3).ok(),
          "native gemm shape gate refuses fp8 K % 4 != 0 (a packed word would straddle a row)");
    check(!validate_qwen_native_gemm_shape(16u, 1024u, 2560u, QwenWeightFormat::fp8_e4m3, 62u).ok(),
          "native gemm shape gate refuses an fp8 scale group that is not a multiple of 4");
    check(!validate_qwen_native_gemm_shape(16u, 1024u, 2559u, QwenWeightFormat::bf16).ok(),
          "native gemm shape gate refuses an odd K (packed-word ABI)");
    check(!validate_qwen_native_gemm_shape(16u, 0u, 2560u, QwenWeightFormat::bf16).ok(),
          "native gemm shape gate refuses N = 0");
    check(validate_qwen_native_gemm_shape(16u, 1024u, 2560u, QwenWeightFormat::bf16).ok(),
          "native gemm shape gate accepts the model geometry");
    check(validate_qwen_native_gemm_shape(0u, 1024u, 2560u, QwenWeightFormat::bf16).ok(),
          "native gemm shape gate accepts M = 0 (empty dispatch)");

    {
        auto plan = plan_qwen_native_gemm_dispatch(1024u, 9216u, 2560u, cfg);
        check(plan.ok(), "native gemm plan M=1024 N=9216");
        check(plan.ok() && plan.value->blocks_x == 8u && plan.value->blocks_y == 144u &&
                  plan.value->threads_x == 8u * 256u && plan.value->blocks == 8u * 144u,
              "native gemm plan tiles M by 128 and N by 64");
    }
    {
        auto plan = plan_qwen_native_gemm_dispatch(0u, 9216u, 2560u, cfg);
        check(plan.ok() && plan.value->empty(), "native gemm plan with M = 0 is empty");
    }
    {
        auto plan = plan_qwen_native_gemm_dispatch(17u, 33u, 8u, cfg);
        check(plan.ok() && plan.value->blocks_x == 1u && plan.value->blocks_y == 1u,
              "native gemm plan covers a ragged M=17 N=33 K=8 in one block");
    }

    // The measured selection rule (narrow N keeps the DSL MT kernel).
    check(qwen_native_gemm_min_rows == 16u, "native gemm row floor is 16");
    check(!qwen_native_gemm_prefers_columns(32u),
          "native gemm leaves N=32 to the DSL MT kernel (measured)");
    check(qwen_native_gemm_prefers_columns(1024u) && qwen_native_gemm_prefers_columns(9216u),
          "native gemm takes N>=256");

    // Generated source: the tiling is baked in and there is exactly one kernel.
    {
        auto source = qwen_native_gemm_cuda_source(cfg, false);
        const auto count = [&](luisa::string_view needle) {
            size_t n = 0u;
            for (auto pos = source.find(needle); pos != luisa::string::npos;
                 pos = source.find(needle, pos + needle.size())) {
                n++;
            }
            return n;
        };
        check(count("__global__") == 1u, "generated source declares exactly one __global__ kernel");
        // The trailing space distinguishes the asm operand string from the same
        // text inside the sm_80 `#error` message.
        check(count("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ") == 1u,
              "generated source issues m16n8k16 bf16 mma");
        check(count("cp.async") >= 2u, "generated source uses cp.async staged tiles");
        check(source.find("#define QN_BM 128\n") != luisa::string::npos &&
                  source.find("#define QN_BN 64\n") != luisa::string::npos &&
                  source.find("#define QN_BK 32\n") != luisa::string::npos &&
                  source.find("#define QN_BLOCK 256\n") != luisa::string::npos,
              "generated source carries the configured tiling");
        check(source.find("@") == luisa::string::npos, "generated source has no unsubstituted token");
        auto residual_source = qwen_native_gemm_cuda_source(cfg, true);
        check(residual_source.find("#define QN_RESIDUAL 1\n") != luisa::string::npos,
              "generated residual variant compiles the epilogue in");
        check(source.find("#define QN_RESIDUAL 0\n") != luisa::string::npos,
              "generated plain variant compiles the epilogue out");
        check(source.find("__CUDA_ARCH__ < 800") != luisa::string::npos,
              "generated source fails closed below sm_80");
        check(source.find("#define QN_FP8 0\n") != luisa::string::npos,
              "the bf16 source compiles the FP8 staging out");
        auto fp8_source = qwen_native_gemm_cuda_source(cfg, false, QwenWeightFormat::fp8_e4m3);
        check(fp8_source.find("#define QN_FP8 1\n") != luisa::string::npos &&
                  fp8_source.find("#define QN_GROUP 64\n") != luisa::string::npos,
              "the fp8 source compiles the E4M3 widening and its scale group in");
        check(fp8_source.find("0x3C800000u") != luisa::string::npos,
              "the fp8 source decodes E4M3 with the branchless magic-constant identity");
    }
}

// ---------------------------------------------------------------------------
// helpers shared by the device cases
// ---------------------------------------------------------------------------
struct Sample {
    uint32_t m{0}, n{0}, k{0};
    luisa::vector<float> x;// [m, k] FP32
    luisa::vector<float> x_bf16;// the same activations, already BF16-rounded
    luisa::vector<uint32_t> words;// [n, k] packed BF16, 2 codes per word
    luisa::vector<float> scales;// unused by the BF16 realization
    luisa::vector<float> residual;// [m, n]
};

Sample make_sample(Rng &rng, uint32_t m, uint32_t n, uint32_t k) {
    Sample s;
    s.m = m;
    s.n = n;
    s.k = k;
    s.x.resize(size_t(m) * k);
    rng.fill(s.x, -1.f, 1.f);
    s.x_bf16.resize(s.x.size());
    for (size_t i = 0; i < s.x.size(); i++) { s.x_bf16[i] = bf16_round(s.x[i]); }
    s.words.resize(size_t(n) * k / 2u);
    for (auto &w : s.words) {
        // Same construction as the GEMV/GEMM suite: distinct BF16 codes per lane,
        // exponent 0x70 so every code is a normal, non-tiny value.
        const uint32_t lo = 0x3800u | (rng.next() & 0xffu) | ((rng.next() & 1u) << 15u);
        const uint32_t hi = 0x3800u | (rng.next() & 0xffu) | ((rng.next() & 1u) << 15u);
        w = lo | (hi << 16u);
    }
    s.scales.assign(1u, 0.f);
    s.residual.resize(size_t(m) * n);
    rng.fill(s.residual, -0.5f, 0.5f);
    return s;
}

// Dispatch one native GEMM into `y` (already uploaded) inside `stream`.
bool dispatch_native(Stream &stream, QwenNativeGemm &gemm, const Sample &s, bool residual,
                     Buffer<float> &bx, Buffer<uint32_t> &bw, Buffer<float> &bs,
                     Buffer<float> &br, Buffer<float> &by, const char *name) {
    auto command = gemm.build_dispatch(residual, bx.view(), bw.view(), bs.view(), br.view(),
                                       by.view(), s.m, s.n, s.k);
    if (command == nullptr) {
        check(false, name, "build_dispatch returned no command");
        return false;
    }
    stream << std::move(command) << synchronize();
    return true;
}

// Run the reference DSL GEMM (one thread per output) on the device: it is the
// FP32-activation oracle the engine's fallback path uses, and (fed the already
// BF16-rounded activations) the exactness oracle of the tensor-core kernel.
luisa::vector<float> run_dsl_gemm(Device &dev, Stream &stream, QwenWeightFormat format,
                                  const Sample &s, luisa::span<const float> x, bool residual) {
    auto bx = upload(dev, stream, luisa::vector<float>{x.begin(), x.end()});
    auto bw = upload(dev, stream, s.words);
    auto bs = upload(dev, stream, s.scales);
    auto br = upload(dev, stream, s.residual);
    auto by = dev.create_buffer<float>(size_t(s.m) * s.n);
    auto shader = dev.compile(make_qwen_gemm_kernel(
        format, residual ? QwenMatmulEpilogue::residual : QwenMatmulEpilogue::none));
    auto plan = plan_qwen_gemm_dispatch(s.m, s.n, s.k);
    if (!plan.ok()) { return {}; }
    luisa::vector<float> out;
    stream << shader(bx, bw, bs, br, by, s.m, s.n, s.k, kGroup, plan.value->padded_cols)
                  .dispatch(plan.value->total_x, plan.value->total_y)
           << synchronize();
    out = download(stream, by, size_t(s.m) * s.n);
    return out;
}

// ---------------------------------------------------------------------------
// 2. ragged differential vs the CPU oracle, both epilogues
// ---------------------------------------------------------------------------
void test_ragged(QwenNativeGemm &gemm, Device &dev, Stream &stream, Rng &rng) {
    struct Shape { uint32_t m, n, k; };
    const Shape shapes[] = {
        {16u, 256u, 2560u},// the smallest row count the accelerator takes
        {17u, 256u, 8u},   // ragged M and a K below one k step
        {31u, 257u, 40u},  // ragged N, K not a multiple of the 32-element step
        {64u, 256u, 72u},  // ragged K (8-aligned, not 32-aligned)
        {65u, 259u, 12u},  // K % 8 != 0 -> the 4-byte cp.async path
        {129u, 256u, 64u}, // two M blocks, the second ragged
        {16u, 1024u, 96u},
    };
    for (const auto &shape : shapes) {
        auto s = make_sample(rng, shape.m, shape.n, shape.k);
        for (bool residual : {false, true}) {
            auto bx = upload(dev, stream, s.x);
            auto bw = upload(dev, stream, s.words);
            auto bs = upload(dev, stream, s.scales);
            auto br = upload(dev, stream, s.residual);
            auto by = dev.create_buffer<float>(size_t(s.m) * s.n);
            luisa::string name = luisa::format("native gemm M={} N={} K={} {}", s.m, s.n, s.k,
                                               residual ? "residual" : "plain");
            if (!dispatch_native(stream, gemm, s, residual, bx, bw, bs, br, by, name.c_str())) {
                continue;
            }
            auto got = download(stream, by, size_t(s.m) * s.n);
            auto dev_ref = run_dsl_gemm(dev, stream, QwenWeightFormat::bf16, s, s.x, residual);
            // 1) implementation exactness: same activations, already BF16.
            auto bx_bf = upload(dev, stream, s.x_bf16);
            auto by_bf = dev.create_buffer<float>(size_t(s.m) * s.n);
            luisa::vector<float> got_bf;
            luisa::vector<float> oracle;
            if (dispatch_native(stream, gemm, s, residual, bx_bf, bw, bs, br, by_bf,
                                "native gemm bf16-x dispatch")) {
                got_bf = download(stream, by_bf, size_t(s.m) * s.n);
                oracle = run_dsl_gemm(dev, stream, QwenWeightFormat::bf16, s, s.x_bf16, residual);
                check_agreement(got_bf, oracle, kOracleTol, (name + " vs bf16-x oracle").c_str());
            }
            // 2) the documented tolerance against the FP32-activation reference.
            check_agreement(got, dev_ref, kFp32Tol, (name + " vs fp32 dsl gemm").c_str());
            // 3) the independent CPU double oracle, for the small shapes only.
            if (size_t(s.m) * s.n * s.k <= 8u * 1000u * 1000u) {
                const auto res_span = residual ? luisa::span<const float>{s.residual}
                                               : luisa::span<const float>{};
                luisa::vector<float> cpu_ref;
                luisa::vector<float> cpu_ref_bf;
                const bool ok_ref = ref_qwen_matmul(QwenWeightFormat::bf16, s.x, s.words, s.scales,
                                                    res_span, residual, s.m, s.n, s.k, kGroup, cpu_ref).ok();
                const bool ok_ref_bf = ref_qwen_matmul(QwenWeightFormat::bf16, s.x_bf16, s.words,
                                                       s.scales, res_span, residual, s.m, s.n, s.k,
                                                       kGroup, cpu_ref_bf).ok();
                if (ok_ref && ok_ref_bf) {
                    check_agreement(got_bf, cpu_ref_bf, kOracleTol,
                                    (name + " vs bf16-x cpu oracle").c_str());
                    check_agreement(got, cpu_ref, kFp32Tol,
                                    (name + " vs fp32 cpu oracle").c_str());
                    // The DSL GEMM on the BF16-rounded activation must be the very
                    // same function the CPU oracle computes, so this pins the
                    // reference itself, not only the accelerator.
                    check_agreement(oracle, cpu_ref_bf, kOracleTol,
                                    (name + " dsl-on-bf16-x vs cpu oracle").c_str());
                } else {
                    check(false, (name + " cpu oracle").c_str(), "ref_qwen_matmul refused the shape");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. model-sized shapes vs the DSL MT and reference GEMMs
// ---------------------------------------------------------------------------
void test_model_shapes(QwenNativeGemm &gemm, Device &dev, Stream &stream, Rng &rng) {
    struct Shape { uint32_t m, n, k; };
    const Shape shapes[] = {
        {256u, 1024u, 2560u},
        {64u, 9216u, 2560u},
    };
    for (const auto &shape : shapes) {
        auto s = make_sample(rng, shape.m, shape.n, shape.k);
        auto bx = upload(dev, stream, s.x);
        auto bx_bf = upload(dev, stream, s.x_bf16);
        auto bw = upload(dev, stream, s.words);
        auto bs = upload(dev, stream, s.scales);
        auto br = upload(dev, stream, s.residual);
        auto by = dev.create_buffer<float>(size_t(s.m) * s.n);
        auto by_bf = dev.create_buffer<float>(size_t(s.m) * s.n);
        luisa::string name = luisa::format("native gemm model M={} N={} K={}", s.m, s.n, s.k);
        if (!dispatch_native(stream, gemm, s, false, bx, bw, bs, br, by, name.c_str())) { continue; }
        dispatch_native(stream, gemm, s, false, bx_bf, bw, bs, br, by_bf,
                        "native gemm model bf16-x dispatch");
        auto got = download(stream, by, size_t(s.m) * s.n);
        auto got_bf = download(stream, by_bf, size_t(s.m) * s.n);
        const auto dsl_all = run_dsl_gemm(dev, stream, QwenWeightFormat::bf16, s, s.x, false);
        const auto dsl_bf = run_dsl_gemm(dev, stream, QwenWeightFormat::bf16, s, s.x_bf16, false);
        check_agreement(got_bf, dsl_bf, kOracleTol, (name + " vs bf16-x dsl gemm").c_str());
        check_agreement(got, dsl_all, kFp32Tol, (name + " vs fp32 dsl gemm").c_str());
        // The weight-reusing MT kernel must agree with the tensor-core kernel to the
        // same documented tolerance (both consume the same packed buffer).
        QwenGemmMtConfig mt_cfg{};
        auto mt_shader = dev.compile(make_qwen35_gemm_mt_kernel(
            mt_cfg, QwenWeightFormat::bf16, QwenMatmulEpilogue::none));
        auto mt_plan = plan_qwen35_gemm_mt_dispatch(s.m, s.n, s.k, QwenWeightFormat::bf16, kGroup, mt_cfg);
        if (mt_plan.ok()) {
            auto by_mt = dev.create_buffer<float>(size_t(s.m) * s.n);
            stream << mt_shader(bx, bw, bs, br, by_mt, s.m, s.n, s.k, kGroup,
                                mt_plan.value->column_blocks)
                          .dispatch(mt_plan.value->total_x, mt_plan.value->total_y)
                   << synchronize();
            auto got_mt = download(stream, by_mt, size_t(s.m) * s.n);
            check_agreement(got, got_mt, kFp32Tol,
                            (name + " vs dsl mt gemm").c_str());
        } else {
            check(false, (name + " mt plan").c_str(), mt_plan.status.detail);
        }
    }
}

// ---------------------------------------------------------------------------
// 4. sequencing, overwrite, deferred commit and y/residual aliasing
// ---------------------------------------------------------------------------
void test_sequence_and_lifetime(QwenNativeGemm &gemm, Device &dev, Stream &stream, Rng &rng) {
    // Several shapes dispatched back-to-back from ONE compiled instance, in ONE
    // command list, interleaved with a DSL kernel: stream order must hold and no
    // dispatch may leak into another.
    const uint32_t m1 = 64u, n1 = 1024u, k1 = 2560u;
    const uint32_t m2 = 32u, n2 = 512u, k2 = 40u;
    auto s1 = make_sample(rng, m1, n1, k1);
    auto s2 = make_sample(rng, m2, n2, k2);
    auto bw1 = upload(dev, stream, s1.words);
    auto bs1 = upload(dev, stream, s1.scales);
    auto br1 = upload(dev, stream, s1.residual);
    auto bw2 = upload(dev, stream, s2.words);
    auto bs2 = upload(dev, stream, s2.scales);
    auto br2 = upload(dev, stream, s2.residual);
    auto bx1 = upload(dev, stream, s1.x);
    auto bx2 = upload(dev, stream, s2.x);
    auto by1 = dev.create_buffer<float>(size_t(m1) * n1);
    auto by2 = dev.create_buffer<float>(size_t(m2) * n2);
    {
        auto c1 = gemm.build_dispatch(false, bx1.view(), bw1.view(), bs1.view(), br1.view(),
                                      by1.view(), m1, n1, k1);
        auto c2 = gemm.build_dispatch(true, bx2.view(), bw2.view(), bs2.view(), br2.view(),
                                      by2.view(), m2, n2, k2);
        // Deferred commit: both commands are built and held, then submitted in one
        // list, so the launcher's packed argument buffers must outlive the build call.
        check(c1 != nullptr && c2 != nullptr, "native gemm builds two commands up front");
        auto list = CommandList::create(2u);
        if (c1 != nullptr) { list << std::move(c1); }
        if (c2 != nullptr) { list << std::move(c2); }
        stream << list.commit() << synchronize();
    }
    auto got1 = download(stream, by1, size_t(m1) * n1);
    auto got2 = download(stream, by2, size_t(m2) * n2);
    check_agreement(got1, run_dsl_gemm(dev, stream, QwenWeightFormat::bf16, s1, s1.x, false),
                    kFp32Tol, "native gemm sequence arm 1 (plain)");
    check_agreement(got2, run_dsl_gemm(dev, stream, QwenWeightFormat::bf16, s2, s2.x, true),
                    kFp32Tol, "native gemm sequence arm 2 (residual)");

    // Overwrite: a second dispatch into the same y must replace, not accumulate.
    {
        auto s3 = make_sample(rng, 16u, 256u, 2560u);
        auto bw3 = upload(dev, stream, s3.words);
        auto bs3 = upload(dev, stream, s3.scales);
        auto br3 = upload(dev, stream, s3.residual);
        auto bx3 = upload(dev, stream, s3.x);
        auto by3 = dev.create_buffer<float>(size_t(s3.m) * s3.n);
        dispatch_native(stream, gemm, s3, true, bx3, bw3, bs3, br3, by3, "native gemm overwrite 1");
        auto first = download(stream, by3, size_t(s3.m) * s3.n);
        dispatch_native(stream, gemm, s3, false, bx3, bw3, bs3, br3, by3, "native gemm overwrite 2");
        auto second = download(stream, by3, size_t(s3.m) * s3.n);
        const auto plain = run_dsl_gemm(dev, stream, QwenWeightFormat::bf16, s3, s3.x, false);
        check_agreement(second, plain, kFp32Tol, "native gemm overwrite is not an accumulate");
        bool differs = false;
        for (size_t i = 0; i < first.size(); i++) { differs = differs || first[i] != second[i]; }
        check(differs, "native gemm residual and plain epilogues differ on the same buffer");
    }

    // Aliasing: `y` IS the residual buffer. Every output element is written by one
    // thread that reads its own residual slot first, so in-place must be exact.
    {
        auto s4 = make_sample(rng, 40u, 300u, 96u);
        auto bw4 = upload(dev, stream, s4.words);
        auto bs4 = upload(dev, stream, s4.scales);
        auto bx4 = upload(dev, stream, s4.x);
        auto by4 = upload(dev, stream, s4.residual);// holds the residual first
        dispatch_native(stream, gemm, s4, true, bx4, bw4, bs4, by4, by4, "native gemm alias dispatch");
        auto aliased = download(stream, by4, size_t(s4.m) * s4.n);
        check_agreement(aliased, run_dsl_gemm(dev, stream, QwenWeightFormat::bf16, s4, s4.x, true),
                        kFp32Tol, "native gemm y == residual aliasing");
    }

    // Lifetime: the shader must survive many dispatches and be destroyed only after
    // the stream that used it completed - exercised by the scope of `gemm` itself,
    // which main() destroys after the final synchronize.
    {
        auto s5 = make_sample(rng, 16u, 1024u, 32u);
        auto bw5 = upload(dev, stream, s5.words);
        auto bs5 = upload(dev, stream, s5.scales);
        auto br5 = upload(dev, stream, s5.residual);
        auto bx5 = upload(dev, stream, s5.x);
        auto by5 = dev.create_buffer<float>(size_t(s5.m) * s5.n);
        bool all_ok = true;
        for (uint32_t i = 0; i < 8u; i++) {
            all_ok = all_ok && dispatch_native(stream, gemm, s5, i % 2u == 1u, bx5, bw5, bs5, br5,
                                               by5, "native gemm repeat dispatch");
        }
        check(all_ok, "native gemm survives eight sequential dispatches from one instance");
        auto last = download(stream, by5, size_t(s5.m) * s5.n);
        // i == 7 took the residual epilogue (i % 2 == 1), so that is the arm to compare.
        check_agreement(last, run_dsl_gemm(dev, stream, QwenWeightFormat::bf16, s5, s5.x, true),
                        kFp32Tol, "native gemm eighth dispatch result (residual arm)");
    }
}


// ---------------------------------------------------------------------------
// 5. FP8 E4M3 realization: codes+scales widened into the shared tile
// ---------------------------------------------------------------------------
// The FP8 kernel decodes each packed code, multiplies by its group scale and rounds
// THAT product to BF16 on the way into shared memory. Its exact oracle is therefore
// a reference GEMM whose weights are already BF16(fp8(code) * scale) - built here on
// the host - run on the already-BF16 activations, which isolates implementation
// error (decode, scale index, fragment layout) from both roundings.
luisa::vector<uint32_t> widen_fp8_words(const Sample &s, uint32_t group) {
    const uint32_t groups_per_row = (s.k + group - 1u) / group;
    luisa::vector<uint32_t> words(size_t(s.n) * s.k / 2u, 0u);
    for (uint32_t n = 0; n < s.n; n++) {
        for (uint32_t k = 0; k < s.k; k += 2u) {
            unsigned packed[2];
            for (unsigned j = 0; j < 2u; j++) {
                const auto index = size_t(n) * s.k + k + j;
                const unsigned word = s.words[index >> 2u];
                const unsigned code = (word >> ((index & 3u) * 8u)) & 0xFFu;
                const float scale = s.scales[size_t(n) * groups_per_row + (k + j) / group];
                packed[j] = host_f32_to_bf16_bits(host_fp8_e4m3_bits_to_f32(
                                static_cast<uint8_t>(code)) * scale);
            }
            words[size_t(n) * s.k / 2u + k / 2u] = packed[0] | (packed[1] << 16u);
        }
    }
    return words;
}

void test_fp8(Device &dev, Stream &stream, Rng &rng) {
    constexpr uint32_t group = 64u;
    QwenNativeGemmConfig cfg{};
    QwenNativeGemm gemm;
    if (auto status = gemm.create(dev, QwenWeightFormat::fp8_e4m3, cfg); !status.ok()) {
        check(false, "native gemm fp8 create",
              luisa::format("{}: {}", status.message, status.detail));
        return;
    }
    check(gemm.valid() && gemm.is_fp8(), "native gemm compiles the fp8 E4M3 realization");
    check(gemm.has_residual(), "native gemm fp8 compiled the residual variant");

    struct Shape { uint32_t m, n, k; };
    const Shape shapes[] = {
        {16u, 256u, 2560u},
        {33u, 257u, 68u}, // ragged M/N, K % 32 != 0 but K % 4 == 0
        {65u, 300u, 12u}, // one partial scale group per row
        {16u, 1024u, 96u},
    };
    for (const auto &shape : shapes) {
        Sample s;
        s.m = shape.m;
        s.n = shape.n;
        s.k = shape.k;
        s.x.resize(size_t(s.m) * s.k);
        rng.fill(s.x, -1.f, 1.f);
        s.x_bf16.resize(s.x.size());
        for (size_t i = 0; i < s.x.size(); i++) { s.x_bf16[i] = bf16_round(s.x[i]); }
        luisa::vector<float> w(size_t(s.n) * s.k);
        rng.fill(w, -0.5f, 0.5f);
        if (!host_pack_matrix(QwenWeightFormat::fp8_e4m3, w, s.k, s.n, group,
                              s.words, s.scales).ok()) {
            check(false, "native gemm fp8 host pack", "host_pack_matrix refused the shape");
            continue;
        }
        s.residual.resize(size_t(s.m) * s.n);
        rng.fill(s.residual, -0.5f, 0.5f);
        const auto widened = widen_fp8_words(s, group);

        auto bx = upload(dev, stream, s.x);
        auto bx_bf = upload(dev, stream, s.x_bf16);
        auto bw = upload(dev, stream, s.words);
        auto bs = upload(dev, stream, s.scales);
        auto br = upload(dev, stream, s.residual);
        for (bool residual : {false, true}) {
            luisa::string name =
                luisa::format("native gemm fp8 M={} N={} K={} {}", s.m, s.n, s.k,
                              residual ? "residual" : "plain");
            auto by = dev.create_buffer<float>(size_t(s.m) * s.n);
            auto by_bf = dev.create_buffer<float>(size_t(s.m) * s.n);
            if (!dispatch_native(stream, gemm, s, residual, bx, bw, bs, br, by, name.c_str())) {
                continue;
            }
            dispatch_native(stream, gemm, s, residual, bx_bf, bw, bs, br, by_bf,
                            "native gemm fp8 bf16-x dispatch");
            auto got = download(stream, by, size_t(s.m) * s.n);
            auto got_bf = download(stream, by_bf, size_t(s.m) * s.n);
            // Exact oracle: BF16(fp8 * scale) weights, BF16 activations, double sum.
            const auto res_span = residual ? luisa::span<const float>{s.residual}
                                           : luisa::span<const float>{};
            luisa::vector<float> oracle;
            if (ref_qwen_matmul(QwenWeightFormat::bf16, s.x_bf16, widened, {},
                                res_span, residual, s.m, s.n, s.k, group, oracle).ok()) {
                check_agreement(got_bf, oracle, kOracleTol,
                                (name + " vs widened-bf16 cpu oracle").c_str());
            } else {
                check(false, (name + " widened oracle").c_str(), "oracle refused the shape");
            }
            // The DSL path (FP32 multiply of the decoded weight) differs only by the
            // BF16 widening of fp8*scale, so it uses the documented loose bound.
            auto dsl = run_dsl_gemm(dev, stream, QwenWeightFormat::fp8_e4m3, s, s.x, residual);
            check_agreement(got, dsl, kFp32Tol, (name + " vs fp8 dsl gemm").c_str());
        }
    }
    gemm.reset();
}


// ---------------------------------------------------------------------------
// 6. CUDA native Tile IR typed-FP8 prefill GEMM (qwen/cuda_tile_gemm.h)
// ---------------------------------------------------------------------------
// A DIFFERENT accelerator from the one above: the kernel is a Tile DSL capture
// compiled with tile::compile(Lowering::NATIVE) for the CUDA 13.4+ native route.
// It consumes the SAME packed uint32 FP8 weight buffer through
// `.as<tile::float8_e4m3fn>()` and it scales the BF16 partial dot AFTER the
// contraction, in FP32. Its exact oracle is therefore a different function from
// the native-shader kernel's (which scales each weight BEFORE the dot):
//  * `ref_tile_fp8_gemm` reproduces the kernel step by step - BF16 activations,
//    BF16(fp8) weights (an exact conversion), one double-precision partial dot
//    per 64-element K block, that partial multiplied by the FP32 group scale.
//    Agreement with it is FP32 accumulation order ONLY: the implementation-
//    exactness gate.
//  * `ref_qwen_matmul(fp8_e4m3, ...)` is the suite's FP32-activation reference,
//    so agreement with it is the documented BF16 ACTIVATION rounding - the loose
//    gate. The kernel never quantizes an activation to FP8 and never re-encodes
//    a weight.
//  * the same kernel fed the already-BF16-rounded activations must reproduce the
//    device result to FP32 reassociation, which separates the (expected)
//    activation rounding from any decode/index/layout error.
constexpr uint32_t kTileGroup = 64u;
// Measured on this suite's cancellation-dominated data: agreement with the
// decomposed oracle is FP32 reassociation (max|d|/max|ref| < 1e-6), so 1e-4 is a
// tight implementation gate; the deviation from the FP32-activation reference is
// the single BF16 activation rounding and shares kFp32Tol's envelope.
constexpr float kTileOracleTol = 1e-4f;
constexpr float kTileFp32Tol = 1e-1f;
// Host double-precision oracle budget (products); larger shapes are compared to
// the device-side DSL GEMM instead so the suite stays fast.
constexpr uint64_t kTileCpuOracleProducts = 40ull * 1000ull * 1000ull;

// One weight element exactly as the Tile kernel consumes it: BF16(fp8 code). The
// E4M3 -> BF16 conversion is exact, and the round-trip is spelled out to make
// that claim checkable rather than assumed.
[[nodiscard]] double tile_weight_at(const Sample &s, size_t n, size_t k) noexcept {
    const auto index = n * size_t(s.k) + k;
    const auto word = s.words[index >> 2u];
    const auto code = static_cast<uint8_t>((word >> ((index & 3u) * 8u)) & 0xFFu);
    return double(host_bf16_bits_to_f32(host_f32_to_bf16_bits(host_fp8_e4m3_bits_to_f32(code))));
}

// Exact oracle of the typed Tile kernel (see the note above).
[[nodiscard]] luisa::vector<float> ref_tile_fp8_gemm(const Sample &s, uint32_t group, bool residual) {
    const auto M = size_t(s.m), N = size_t(s.n), K = size_t(s.k);
    const auto groups = K / group;
    luisa::vector<float> y(M * N, 0.0f);
    luisa::vector<double> wr(K, 0.0);
    for (size_t n = 0; n < N; n++) {
        for (size_t k = 0; k < K; k++) { wr[k] = tile_weight_at(s, n, k); }
        for (size_t m = 0; m < M; m++) {
            double acc = 0.0;
            for (size_t g = 0; g < groups; g++) {
                double partial = 0.0;
                for (size_t j = 0; j < group; j++) {
                    const auto k = g * group + j;
                    partial += double(s.x_bf16[m * K + k]) * wr[k];
                }
                acc += partial * double(s.scales[n * groups + g]);
            }
            if (residual) { acc += double(s.residual[m * N + n]); }
            y[m * N + n] = static_cast<float>(acc);
        }
    }
    return y;
}

// FP8 E4M3 sample in the packed weight ABI: `words` is EXACTLY the buffer the
// existing native/DSL GEMMs read (n*k/4 little-endian words, low code first) and
// `scales` the FP32 group scales of the same allocation.
[[nodiscard]] Sample make_fp8_sample(Rng &rng, uint32_t m, uint32_t n, uint32_t k) {
    Sample s;
    s.m = m;
    s.n = n;
    s.k = k;
    s.x.resize(size_t(m) * k);
    rng.fill(s.x, -1.f, 1.f);
    s.x_bf16.resize(s.x.size());
    for (size_t i = 0; i < s.x.size(); i++) { s.x_bf16[i] = bf16_round(s.x[i]); }
    luisa::vector<float> w(size_t(n) * k);
    rng.fill(w, -0.5f, 0.5f);
    if (!host_pack_matrix(QwenWeightFormat::fp8_e4m3, w, k, n, kTileGroup, s.words, s.scales).ok()) {
        check(false, "cuda tile gemm fp8 sample", "host_pack_matrix refused the shape");
    }
    s.residual.resize(size_t(m) * n);
    rng.fill(s.residual, -0.5f, 0.5f);
    return s;
}

// ---------------------------------------------------------------------------
// 6a. device-free contract
// ---------------------------------------------------------------------------
void test_cuda_tile_contract() {
    QwenCudaTileGemmConfig cfg{};
    check(validate_qwen_cuda_tile_gemm_config(cfg).ok(), "cuda tile gemm default tiling accepted");
    check(cfg.tile_m == 64u && cfg.tile_n == 64u && cfg.tile_k == 64u && cfg.group == 64u,
          "cuda tile gemm default is 64x64x64 with the 64-element scale group");
    {
        auto bad = cfg;
        bad.tile_m = 96u;
        check(!validate_qwen_cuda_tile_gemm_config(bad).ok(),
              "cuda tile gemm refuses a non-power-of-two tile_m");
        bad = cfg;
        bad.tile_n = 256u;
        check(!validate_qwen_cuda_tile_gemm_config(bad).ok(),
              "cuda tile gemm refuses tile_n outside [16, 128]");
        bad = cfg;
        bad.tile_m = 128u;
        bad.tile_n = 128u;
        check(validate_qwen_cuda_tile_gemm_config(bad).ok(), "cuda tile gemm accepts 128x128x64");
        bad = cfg;
        bad.tile_k = 32u;
        check(!validate_qwen_cuda_tile_gemm_config(bad).ok(), "cuda tile gemm refuses a k step != 64");
        bad = cfg;
        bad.group = 32u;
        check(!validate_qwen_cuda_tile_gemm_config(bad).ok(),
              "cuda tile gemm refuses a scale group != 64");
    }

    check(qwen_cuda_tile_gemm_supported_format(QwenWeightFormat::fp8_e4m3),
          "cuda tile gemm takes fp8 E4M3 weight storage");
    check(!qwen_cuda_tile_gemm_supported_format(QwenWeightFormat::bf16) &&
              !qwen_cuda_tile_gemm_supported_format(QwenWeightFormat::q8_0) &&
              !qwen_cuda_tile_gemm_supported_format(QwenWeightFormat::w4),
          "cuda tile gemm refuses bf16/q8_0/w4 storage (never re-encodes a weight buffer)");

    check(validate_qwen_cuda_tile_gemm_shape(16u, 1024u, 2560u, QwenWeightFormat::fp8_e4m3).ok(),
          "cuda tile gemm shape gate accepts the model geometry");
    check(validate_qwen_cuda_tile_gemm_shape(17u, 33u, 64u, QwenWeightFormat::fp8_e4m3).ok(),
          "cuda tile gemm shape gate accepts ragged M/N and K = one k step");
    check(validate_qwen_cuda_tile_gemm_shape(65u, 129u, 128u, QwenWeightFormat::fp8_e4m3).ok(),
          "cuda tile gemm shape gate accepts K = 128");
    check(!validate_qwen_cuda_tile_gemm_shape(16u, 1024u, 2560u, QwenWeightFormat::bf16).ok(),
          "cuda tile gemm shape gate refuses bf16 storage");
    check(!validate_qwen_cuda_tile_gemm_shape(16u, 1024u, 2560u, QwenWeightFormat::fp8_e4m3, 32u).ok(),
          "cuda tile gemm shape gate refuses a 32-element scale group");
    check(!validate_qwen_cuda_tile_gemm_shape(16u, 1024u, 2559u, QwenWeightFormat::fp8_e4m3).ok(),
          "cuda tile gemm shape gate refuses a K tail");
    check(!validate_qwen_cuda_tile_gemm_shape(16u, 1024u, 32u, QwenWeightFormat::fp8_e4m3).ok(),
          "cuda tile gemm shape gate refuses K below one k step");
    check(!validate_qwen_cuda_tile_gemm_shape(0u, 1024u, 2560u, QwenWeightFormat::fp8_e4m3).ok(),
          "cuda tile gemm shape gate refuses M = 0 (no static Tile shape)");
    check(!validate_qwen_cuda_tile_gemm_shape(16u, 0u, 2560u, QwenWeightFormat::fp8_e4m3).ok(),
          "cuda tile gemm shape gate refuses N = 0");
    check(!validate_qwen_cuda_tile_gemm_shape(0u, 0u, 0u, QwenWeightFormat::fp8_e4m3).ok(),
          "cuda tile gemm shape gate refuses an all-zero shape");
    check(!validate_qwen_cuda_tile_gemm_shape(1u, 1024u, 1u << 21u, QwenWeightFormat::fp8_e4m3).ok(),
          "cuda tile gemm shape gate refuses an extent above the static bound");
    check(!validate_qwen_cuda_tile_gemm_shape(1u << 20u, 1u << 20u, 64u,
                                              QwenWeightFormat::fp8_e4m3)
               .ok(),
          "cuda tile gemm shape gate refuses a volume above the element bound");

    // The captured specialization is verified device-free on EVERY backend: the
    // kernel is a pure TileIR capture, so a malformed kernel would be caught here
    // instead of only at cuda compile time.
    {
        auto plain = capture_qwen_cuda_tile_gemm(cfg, 17u, 33u, 64u, false);
        check(plain.ok(), "cuda tile gemm plain capture verifies");
        check(plain.ok() && plain.kernel->function().name() == "qwen_cuda_tile_fp8_gemm",
              "cuda tile gemm plain capture is the plain specialization");
        check(plain.diagnostics.empty(), "cuda tile gemm plain capture has no diagnostics");
        auto residual_capture = capture_qwen_cuda_tile_gemm(cfg, 17u, 33u, 64u, true);
        check(residual_capture.ok(), "cuda tile gemm residual capture verifies");
        check(residual_capture.ok() &&
                  residual_capture.kernel->function().name() == "qwen_cuda_tile_fp8_gemm_residual",
              "cuda tile gemm residual capture is a separate specialization");
        auto wide = capture_qwen_cuda_tile_gemm(QwenCudaTileGemmConfig{128u, 128u, 64u, 64u}, 300u,
                                                200u, 128u, false);
        check(wide.ok(), "cuda tile gemm capture verifies for a 128x128 tiling and ragged extents");
        auto k_tail = capture_qwen_cuda_tile_gemm(cfg, 17u, 33u, 96u, false);
        check(!k_tail.ok() && !k_tail.kernel.has_value(),
              "cuda tile gemm capture refuses a K tail");
        auto bad_cfg = cfg;
        bad_cfg.group = 32u;
        auto bad_capture = capture_qwen_cuda_tile_gemm(bad_cfg, 17u, 33u, 64u, false);
        check(!bad_capture.ok() && !bad_capture.kernel.has_value(),
              "cuda tile gemm capture refuses a foreign scale group");
    }
}

// ---------------------------------------------------------------------------
// 6b. fail-closed answer when this backend cannot serve the accelerator
// ---------------------------------------------------------------------------
// Runs on every backend (dx/vk included) and touches no GPU work: `create()` must
// refuse before it reaches the probe, and the object must stay unusable.
void test_cuda_tile_fail_closed(Device &device, const char *backend, bool expected_cuda) {
    QwenCudaTileGemm gemm;
    auto status = gemm.create(device, QwenCudaTileGemmConfig{});
    if (expected_cuda) {
        // cuda: the answer depends on the run-time opt-in and the toolkit; both
        // arms are reported by the caller (see main), so only the invariant is
        // asserted here.
        check(gemm.valid() == status.ok(), "cuda tile gemm validity follows create()");
        return;
    }
    check(!status.ok(), "cuda tile gemm refuses a non-cuda backend");
    check(status.code == StatusCode::unsupported,
          "cuda tile gemm reports unsupported on a non-cuda backend");
    check(!gemm.valid(), "cuda tile gemm stays invalid on a non-cuda backend");
    check(gemm.compile_count() == 0u && gemm.shader_count() == 0u,
          "cuda tile gemm compiled nothing on a non-cuda backend");
    // build_dispatch on an invalid object must return nullptr with a diagnostic,
    // never a command and never an assertion.
    auto dummy_f = device.create_buffer<float>(1u);
    auto dummy_u = device.create_buffer<uint32_t>(1u);
    luisa::string error;
    auto command = gemm.build_dispatch(false, dummy_f.view(), dummy_u.view(), dummy_f.view(),
                                       dummy_f.view(), dummy_f.view(), 16u, 64u, 64u, &error);
    check(command == nullptr && !error.empty(),
          "cuda tile gemm builds no command on an invalid object");
    std::printf("cuda tile gemm on %s: SKIP (%s) - dx/vk reject FP8 Tile storage and keep "
                "their own GEMM paths.\n",
                backend, status.detail.empty() ? status.message.c_str() : status.detail.c_str());
}

// ---------------------------------------------------------------------------
// 6c. runtime differential (cuda + LUISA_CUDA_TILE_IR=1 only)
// ---------------------------------------------------------------------------
bool dispatch_cuda_tile(Stream &stream, QwenCudaTileGemm &gemm, bool residual, Buffer<float> &bx,
                        Buffer<uint32_t> &bw, Buffer<float> &bs, Buffer<float> &br,
                        Buffer<float> &by, uint32_t m, uint32_t n, uint32_t k, const char *name) {
    luisa::string error;
    auto command = gemm.build_dispatch(residual, bx.view(), bw.view(), bs.view(), br.view(), by.view(),
                                       m, n, k, &error);
    if (command == nullptr) {
        check(false, name, error.empty() ? luisa::string{"build_dispatch returned no command"} : error);
        return false;
    }
    stream << std::move(command) << synchronize();
    return true;
}

void run_cuda_tile_case(QwenCudaTileGemm &gemm, Device &dev, Stream &stream, const Sample &s,
                        bool residual) {
    luisa::string name = luisa::format("cuda tile gemm M={} N={} K={} {}", s.m, s.n, s.k,
                                       residual ? "residual" : "plain");
    auto bx = upload(dev, stream, s.x);
    auto bx_bf = upload(dev, stream, s.x_bf16);
    auto bw = upload(dev, stream, s.words);
    auto bs = upload(dev, stream, s.scales);
    auto br = upload(dev, stream, s.residual);
    auto by = dev.create_buffer<float>(size_t(s.m) * s.n);
    auto by_bf = dev.create_buffer<float>(size_t(s.m) * s.n);
    if (!dispatch_cuda_tile(stream, gemm, residual, bx, bw, bs, br, by, s.m, s.n, s.k,
                            name.c_str())) {
        return;
    }
    if (!dispatch_cuda_tile(stream, gemm, residual, bx_bf, bw, bs, br, by_bf, s.m, s.n, s.k,
                            "cuda tile gemm bf16-x dispatch")) {
        return;
    }
    auto got = download(stream, by, size_t(s.m) * s.n);
    auto got_bf = download(stream, by_bf, size_t(s.m) * s.n);
    // 0) The kernel's ONLY rounding is the activation's: feeding the activations
    //    already rounded to BF16 must reproduce the same result to reassociation.
    //    A decode/index/layout error would break this, not just widen it.
    check_agreement(got_bf, got, kTileOracleTol, (name + " bf16-x reproduces the result").c_str());
    // 1) Implementation exactness against the decomposed oracle.
    const auto work = uint64_t(s.m) * s.n * s.k;
    const auto res_span = residual ? luisa::span<const float>{s.residual} : luisa::span<const float>{};
    if (work <= kTileCpuOracleProducts) {
        check_agreement(got, ref_tile_fp8_gemm(s, kTileGroup, residual), kTileOracleTol,
                        (name + " vs decomposed bf16-partial oracle").c_str());
    }
    // 2) The documented deviation from the FP32-activation reference.
    luisa::vector<float> fp32_ref;
    if (ref_qwen_matmul(QwenWeightFormat::fp8_e4m3, s.x, s.words, s.scales, res_span, residual,
                        s.m, s.n, s.k, kTileGroup, fp32_ref)
            .ok()) {
        check_agreement(got, fp32_ref, kTileFp32Tol,
                        (name + " vs fp32-activation reference").c_str());
    } else {
        check(false, (name + " fp32 reference").c_str(), "ref_qwen_matmul refused the shape");
    }
    // 3) Large shapes additionally compare against the device-side DSL GEMM, the
    //    engine's own fallback, so the accelerator is never only self-consistent.
    if (work > kTileCpuOracleProducts) {
        auto dsl = run_dsl_gemm(dev, stream, QwenWeightFormat::fp8_e4m3, s, s.x, residual);
        check_agreement(got, dsl, kTileFp32Tol, (name + " vs fp8 dsl gemm").c_str());
    }
}

void test_cuda_tile_shapes(QwenCudaTileGemm &gemm, Device &dev, Stream &stream, Rng &rng) {
    struct Shape {
        uint32_t m, n, k;
    };
    // M is ALWAYS allowed to be ragged: the activation and output tiles are FP32,
    // and their partial tiles are ordinary bounds::zero masks. N is ragged only
    // where the target can compile a PARTIAL typed-FP8 tile load, which the
    // create-time probe measured (SM89 rejects it: NVRTC refuses FP8 Tile types
    // for partial tiles below SM90). Both facts are exercised below.
    const Shape shapes[] = {
        {17u, 64u, 64u},   // ragged M, one k step
        {65u, 128u, 128u}, // two M programs, two N programs, two k blocks
        {33u, 192u, 64u},  // ragged M, three N programs
        {16u, 256u, 2560u},// the model K (40 k blocks) with the host oracle
    };
    for (const auto &shape : shapes) {
        auto s = make_fp8_sample(rng, shape.m, shape.n, shape.k);
        for (bool residual : {false, true}) { run_cuda_tile_case(gemm, dev, stream, s, residual); }
    }
    // The largest model geometry: device-side oracles only (see run_cuda_tile_case).
    auto big = make_fp8_sample(rng, 64u, 1024u, 2560u);
    for (bool residual : {false, true}) { run_cuda_tile_case(gemm, dev, stream, big, residual); }

    // Ragged N: dispatched where the target supports the partial FP8 tile load,
    // and otherwise refused with a diagnostic instead of a broken shader. Both
    // arms are reported so a run on either kind of target is honest.
    const Shape ragged_columns[] = {{17u, 33u, 64u}, {65u, 129u, 128u}};
    for (const auto &shape : ragged_columns) {
        auto s = make_fp8_sample(rng, shape.m, shape.n, shape.k);
        luisa::string name = luisa::format("cuda tile gemm ragged-N M={} N={} K={}", s.m, s.n, s.k);
        if (gemm.supports_ragged_columns()) {
            for (bool residual : {false, true}) {
                run_cuda_tile_case(gemm, dev, stream, s, residual);
            }
        } else {
            auto bx = upload(dev, stream, s.x);
            auto bw = upload(dev, stream, s.words);
            auto bs = upload(dev, stream, s.scales);
            auto br = upload(dev, stream, s.residual);
            auto by = dev.create_buffer<float>(size_t(s.m) * s.n);
            luisa::string error;
            auto command = gemm.build_dispatch(false, bx.view(), bw.view(), bs.view(), br.view(),
                                               by.view(), s.m, s.n, s.k, &error);
            check(command == nullptr && error.find("ragged N") != luisa::string::npos,
                  (name + " refused on this target").c_str(), error);
        }
    }
}

void test_cuda_tile_shared_words(QwenCudaTileGemm &gemm, Device &dev, Stream &stream, Rng &rng) {
    QwenNativeGemm native;
    if (auto status = native.create(dev, QwenWeightFormat::fp8_e4m3, QwenNativeGemmConfig{});
        !status.ok()) {
        check(false, "cuda tile gemm shared-words arm needs the native fp8 gemm",
              luisa::format("{}: {}", status.message, status.detail));
        return;
    }
    auto s = make_fp8_sample(rng, 32u, 256u, 512u);// 8 k blocks, both epilogues cheap
    auto bx = upload(dev, stream, s.x);
    auto bw = upload(dev, stream, s.words);
    auto bs = upload(dev, stream, s.scales);
    auto br = upload(dev, stream, s.residual);
    auto by_tile = dev.create_buffer<float>(size_t(s.m) * s.n);
    auto by_native = dev.create_buffer<float>(size_t(s.m) * s.n);
    auto words_before = download(stream, bw, s.words.size());
    // ONE packed uint32 buffer feeds both implementations back to back. No
    // repacked copy exists: the Tile kernel only reinterprets the view.
    if (!dispatch_cuda_tile(stream, gemm, false, bx, bw, bs, br, by_tile, s.m, s.n, s.k,
                            "cuda tile gemm shared words")) {
        return;
    }
    auto command = native.build_dispatch(false, bx.view(), bw.view(), bs.view(), br.view(),
                                         by_native.view(), s.m, s.n, s.k);
    if (command == nullptr) {
        check(false, "cuda tile gemm shared-words arm", "native fp8 gemm built no command");
        return;
    }
    stream << std::move(command) << synchronize();
    auto tile_out = download(stream, by_tile, size_t(s.m) * s.n);
    auto native_out = download(stream, by_native, size_t(s.m) * s.n);
    auto words_after = download(stream, bw, s.words.size());
    check(words_before == words_after,
          "the packed FP8 words are untouched by both implementations (no transformed copy)");
    check_agreement(tile_out, native_out, kTileFp32Tol,
                    "cuda tile gemm vs native fp8 gemm on the same packed words");
    luisa::vector<float> fp32_ref;
    if (ref_qwen_matmul(QwenWeightFormat::fp8_e4m3, s.x, s.words, s.scales, {}, false, s.m, s.n, s.k,
                        kTileGroup, fp32_ref)
            .ok()) {
        check_agreement(tile_out, fp32_ref, kTileFp32Tol,
                        "cuda tile gemm shared-words arm vs fp32 reference");
        check_agreement(native_out, fp32_ref, kTileFp32Tol,
                        "native fp8 gemm shared-words arm vs fp32 reference");
    }
}

void test_cuda_tile_cache_and_alias(QwenCudaTileGemm &gemm, Device &dev, Stream &stream, Rng &rng) {
    // Shape cache: a shape that test_cuda_tile_shapes already dispatched must be
    // cached in BOTH epilogues, and a second dispatch must not compile again.
    const uint32_t m = 17u, n = 64u, k = 64u;
    check(gemm.has_shader(m, n, k, false), "cuda tile gemm cached the plain shader of its shape");
    check(gemm.has_shader(m, n, k, true), "cuda tile gemm cached the residual shader of its shape");
    auto compiled = gemm.compile_count();
    auto hits = gemm.cache_hits();
    auto s = make_fp8_sample(rng, m, n, k);
    auto bx = upload(dev, stream, s.x);
    auto bw = upload(dev, stream, s.words);
    auto bs = upload(dev, stream, s.scales);
    auto br = upload(dev, stream, s.residual);
    auto by = dev.create_buffer<float>(size_t(m) * n);
    for (uint32_t i = 0; i < 4u; i++) {
        dispatch_cuda_tile(stream, gemm, i % 2u == 1u, bx, bw, bs, br, by, m, n, k,
                           "cuda tile gemm cached dispatch");
    }
    check(gemm.compile_count() == compiled, "cuda tile gemm compiles nothing for a cached shape");
    check(gemm.cache_hits() == hits + 4u, "cuda tile gemm records four cache hits");
    check(gemm.shader_count() == compiled, "cuda tile gemm shader count equals its compile count");
    auto last = download(stream, by, size_t(m) * n);
    // i == 3 took the residual epilogue.
    check_agreement(last, ref_tile_fp8_gemm(s, kTileGroup, true), kTileOracleTol,
                    "cuda tile gemm fourth cached dispatch (residual)");

    // Residual aliasing: `y` IS the residual buffer. Each output tile is written
    // only after its own residual tile was loaded, so in-place is exact and the
    // native Tile path does not claim disjoint writes.
    {
        auto s4 = make_fp8_sample(rng, 40u, 128u, 128u);
        auto bx4 = upload(dev, stream, s4.x);
        auto bw4 = upload(dev, stream, s4.words);
        auto bs4 = upload(dev, stream, s4.scales);
        auto by4 = upload(dev, stream, s4.residual);// holds the residual first
        if (dispatch_cuda_tile(stream, gemm, true, bx4, bw4, bs4, by4, by4, s4.m, s4.n, s4.k,
                               "cuda tile gemm alias dispatch")) {
            auto aliased = download(stream, by4, size_t(s4.m) * s4.n);
            check_agreement(aliased, ref_tile_fp8_gemm(s4, kTileGroup, true), kTileOracleTol,
                            "cuda tile gemm y == residual aliasing");
        }
    }

    // Host validation never reaches a runtime assertion: a short view, a K tail
    // and a zero extent each return nullptr with a diagnostic.
    {
        luisa::string error;
        auto command = gemm.build_dispatch(false, bx.view(), bw.view(), bs.view(), br.view(), by.view(),
                                           m, n, k - 64u, &error);
        check(command == nullptr && !error.empty(), "cuda tile gemm refuses a K tail at dispatch");
        error.clear();
        auto short_view = by.view(0u, 1u);
        command = gemm.build_dispatch(false, bx.view(), bw.view(), bs.view(), br.view(), short_view,
                                      m, n, k, &error);
        check(command == nullptr && !error.empty(),
              "cuda tile gemm refuses a short output view at dispatch");
        error.clear();
        command = gemm.build_dispatch(false, bx.view(), bw.view(), bs.view(), br.view(), by.view(),
                                      0u, n, k, &error);
        check(command == nullptr && !error.empty(),
              "cuda tile gemm refuses M = 0 at dispatch");
        error.clear();
        auto empty_u = dev.create_buffer<uint32_t>(1u);
        command = gemm.build_dispatch(false, bx.view(), empty_u.view(), bs.view(), br.view(), by.view(),
                                      m, n, k, &error);
        check(command == nullptr && !error.empty(),
              "cuda tile gemm refuses a short weight view at dispatch");
    }
}
}// namespace

int main(int argc, char **argv) {
    unbuffered_output();
    if (argc < 2 || argv[1][0] == '\0') {
        std::fprintf(stderr,
                     "usage: %s <backend>\n"
                     "  backend is explicit (e.g. cuda, dx, vk); the CUDA tensor-core PREFILL GEMM\n"
                     "  needs NativeShaderExt/CUDA_NVRTC, so every other backend SKIPs.\n",
                     argv[0]);
        return 2;
    }
    // The device-free contract is checked on every backend.
    test_contract();
    test_cuda_tile_contract();

    Context ctx{argv[0]};
    auto device = ctx.create_device(argv[1]);
    if (!device) {
        std::fprintf(stderr, "failed to create device for backend '%s' (is the backend built/enabled?)\n",
                     argv[1]);
        return 2;
    }
    auto stream = device.create_stream();
    const auto backend = luisa::string{device.backend_name()};
    std::printf("backend: %s\n", backend.c_str());

    if (backend != "cuda" || device.extension<NativeShaderExt>() == nullptr) {
        // The CUDA native Tile IR FP8 GEMM must ALSO answer fail-closed here, and
        // it must do so before any device work: this is the arm that keeps the
        // DX/VK suites green.
        test_cuda_tile_fail_closed(device, backend.c_str(), backend == "cuda");
        std::printf("SKIP: native tensor GEMM requires the cuda backend with NativeShaderExt "
                    "(CUDA_NVRTC); backend '%s' %s. The DSL MT/legacy GEMM covers this "
                    "backend and is tested by test_qwen_ops_dsl.\n",
                    backend.c_str(),
                    backend == "cuda" ? "has no NativeShaderExt" : "has no native CUDA route");
        return summarize("qwen native tensor gemm", argv[1]);
    }

    QwenNativeGemm gemm;
    QwenNativeGemmConfig cfg{};
    if (auto status = gemm.create(device, QwenWeightFormat::bf16, cfg); !status.ok()) {
        std::printf("SKIP: the CUDA tensor-core prefill GEMM could not be compiled at run time "
                    "(%s%s%s) - no NVRTC or sm_80+ hardware. The engine falls back to the DSL "
                    "MT kernel, which is what this SKIP documents.\n",
                    status.message.c_str(), status.detail.empty() ? "" : ": ",
                    status.detail.c_str());
        return summarize("qwen native tensor gemm", argv[1]);
    }
    check(gemm.valid(), "native gemm compiled through NativeShaderExt/CUDA_NVRTC");
    check(gemm.has_residual(), "native gemm compiled the residual variant");
    check(gemm.bindings().size() == 5u, "native gemm reflects five buffer bindings");
    check(gemm.bindings().size() == 5u &&
              gemm.bindings()[0].usage == Usage::READ &&
              gemm.bindings()[1].usage == Usage::READ &&
              gemm.bindings()[2].usage == Usage::READ &&
              gemm.bindings()[3].usage == Usage::READ &&
              gemm.bindings()[4].usage == Usage::READ_WRITE,
          "native gemm binding usages are x/w/scales/residual READ and y READ_WRITE");
    std::printf("native gemm: %s tile %ux%ux%u, %u warps, cold JIT %.1f ms\n",
                qwen_weight_format_name(gemm.format()), gemm.config().block_m,
                gemm.config().block_n, gemm.config().block_k, gemm.config().warps,
                gemm.jit_ms());

    Rng rng;
    test_ragged(gemm, device, stream, rng);
    test_model_shapes(gemm, device, stream, rng);
    test_sequence_and_lifetime(gemm, device, stream, rng);
    test_fp8(device, stream, rng);

    // ---------------------------------------------------------------------
    // CUDA native Tile IR FP8 GEMM: a SEPARATE accelerator on a separate route.
    // Its absence (no opt-in, no CUDA 13.4+ toolchain, unsupported SM) must never
    // change the result of any arm above.
    // ---------------------------------------------------------------------
    {
        QwenCudaTileGemm tile_gemm;
        if (!qwen_cuda_tile_gemm_native_route_requested()) {
            auto status = tile_gemm.create(device, QwenCudaTileGemmConfig{});
            check(!status.ok() && !tile_gemm.valid(),
                  "cuda tile gemm refuses to build without LUISA_CUDA_TILE_IR=1");
            check(status.message.find("LUISA_CUDA_TILE_IR") != luisa::string::npos,
                  "cuda tile gemm names LUISA_CUDA_TILE_IR in its refusal");
            check(status.code == StatusCode::unsupported,
                  "cuda tile gemm reports unsupported when the route is not requested");
            std::printf("SKIP: the CUDA native Tile IR FP8 GEMM requires the run-time opt-in "
                        "LUISA_CUDA_TILE_IR=1 (set it to cover the typed-FP8 Tile arm); every "
                        "other arm above already ran and is unchanged.\n");
        } else if (auto status = tile_gemm.create(device, QwenCudaTileGemmConfig{}); !status.ok()) {
            check(false, "cuda tile gemm create", luisa::format("{}: {}", status.message, status.detail));
        } else {
            check(tile_gemm.valid(), "cuda tile gemm created on the CUDA native Tile IR route");
            check(tile_gemm.shader_count() == 0u,
                  "cuda tile gemm compiles lazily (the probe is not a shape shader)");
            check(tile_gemm.probe_ms() > 0.0, "cuda tile gemm timed its typed-FP8 compile probe");
            std::printf("cuda tile gemm: native Tile FP8 storage, tile %ux%ux%u, probe %.1f ms, "
                        "ragged-N %s\n",
                        tile_gemm.config().tile_m, tile_gemm.config().tile_n,
                        tile_gemm.config().tile_k, tile_gemm.probe_ms(),
                        tile_gemm.supports_ragged_columns() ?
                            "supported (partial FP8 tile load compiled)" :
                            "REFUSED (this target rejects partial FP8 tile loads; N must be a "
                            "multiple of tile_n)");
            test_cuda_tile_shapes(tile_gemm, device, stream, rng);
            test_cuda_tile_shared_words(tile_gemm, device, stream, rng);
            test_cuda_tile_cache_and_alias(tile_gemm, device, stream, rng);
            check(tile_gemm.compile_count() != 0u, "cuda tile gemm reported its lazy compiles");
            std::printf("cuda tile gemm: %u lazy compiles, %u cache hits, %.1f ms JIT\n",
                        tile_gemm.compile_count(), tile_gemm.cache_hits(), tile_gemm.jit_ms());
            stream << synchronize();
            tile_gemm.reset();
            check(!tile_gemm.valid(), "cuda tile gemm reset() destroys the cached shaders");
        }
    }

    // Destroy the native shaders explicitly (before the device) and prove the
    // object reports itself invalid afterwards.
    stream << synchronize();
    gemm.reset();
    check(!gemm.valid(), "native gemm reset() destroys the shader instances");
    return summarize("qwen native tensor gemm", argv[1]);
}
