// Checkpoint-free Qwen3.5-4B single-token packed GEMV benchmark.
//
// Unlike a repeated single-matrix launch, each timed batch walks independent
// matrix replicas (256 MiB target, 1024-replica cap; >64 MiB for every shipped
// shape) so the reported bandwidth is not merely an L2-resident best case on
// the RTX 4080 SUPER. The same packed buffers feed the generic and
// decode-specialized implementations. Recording is excluded; each measurement
// includes one batch submission and one final stream synchronization.
//
// Usage:
//   benchmark_qwen_decode <backend> [--format both|bf16|fp8]
//       [--impl both|generic|decode] [--iters N] [--lm-head]
//       [--ffn] [--attention] [--shape NAME]
#include <luisa/luisa-compute.h>
#include <luisa/core/clock.h>
#include <luisa/dsl/sugar.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

#include "qwen/dsl_ops.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::qwen::dsl;

namespace {

struct Shape {
    const char *name;
    uint32_t n;
    uint32_t k;
};

constexpr Shape kShapes[] = {
    {"gdn_gate", 32u, 2560u},
    {"attn_kv", 1024u, 2560u},
    {"gdn_out_attn_out", 2560u, 4096u},
    {"gdn_z", 4096u, 2560u},
    {"attn_q_gdn_qkv", 8192u, 2560u},
    {"ffn_gate_up", 9216u, 2560u},
    {"ffn_down", 2560u, 9216u},
};
constexpr Shape kLmHead{"lm_head", 248320u, 2560u};
constexpr uint32_t kGroup = 64u;
constexpr size_t kTargetWeightBytes = 256u * 1024u * 1024u;
constexpr size_t kMaxReplicas = 1024u;

struct Options {
    bool bf16{true};
    bool fp8{true};
    bool generic{true};
    bool decode{true};
    bool attention{false};
    bool lm_head{false};
    bool ffn{false};
    std::string shape;
    uint32_t iters{7};
};

bool parse_options(int argc, char **argv, Options &o) {
    for (int i = 2; i < argc; ++i) {
        const std::string_view a{argv[i]};
        auto value = [&]() -> std::string_view {
            return i + 1 < argc ? std::string_view{argv[++i]} : std::string_view{};
        };
        if (a == "--format") {
            auto v = value();
            o.bf16 = v == "both" || v == "bf16";
            o.fp8 = v == "both" || v == "fp8";
            if (!o.bf16 && !o.fp8) { return false; }
        } else if (a == "--impl") {
            auto v = value();
            o.generic = v == "both" || v == "generic";
            o.decode = v == "both" || v == "decode";
            if (!o.generic && !o.decode) { return false; }
        } else if (a == "--iters") {
            auto v = value();
            if (v.empty()) { return false; }
            o.iters = static_cast<uint32_t>(std::strtoul(std::string{v}.c_str(), nullptr, 10));
            if (o.iters == 0u || o.iters > 1000u) { return false; }
        } else if (a == "--lm-head") {
            o.lm_head = true;
        } else if (a == "--attention") {
            o.attention = true;
        } else if (a == "--ffn") {
            o.ffn = true;
        } else if (a == "--shape") {
            auto v = value();
            if (v.empty()) { return false; }
            o.shape = std::string{v};
        } else {
            return false;
        }
    }
    // The separate FFN arm needs the decode-specialized shader even when the
    // caller otherwise requested only generic rows.
    if (o.ffn) { o.decode = true; }
    return true;
}

// Finite, nonzero packed values. Performance data need not come from the host
// quantizer, but must avoid FP8 NaNs and overflow so every timed dispatch has
// the same instruction behavior as real weights.
uint32_t packed_word(QwenWeightFormat format, size_t i) {
    uint32_t h = static_cast<uint32_t>(i) * 1664525u + 1013904223u;
    auto byte = [&](uint32_t shift) {
        const uint32_t code = 0x20u | ((h >> shift) & 0x1fu);
        const uint32_t sign = ((h >> (shift + 5u)) & 1u) << 7u;
        return code | sign;
    };
    if (format == QwenWeightFormat::fp8_e4m3) {
        return byte(0u) | (byte(3u) << 8u) | (byte(6u) << 16u) | (byte(9u) << 24u);
    }
    const uint32_t lo = 0x3800u | (h & 0xffu) | (((h >> 8u) & 1u) << 15u);
    const uint32_t hi = 0x3800u | ((h >> 11u) & 0xffu) | (((h >> 19u) & 1u) << 15u);
    return lo | (hi << 16u);
}

double median(luisa::vector<double> v) {
    std::sort(v.begin(), v.end());
    const auto n = v.size();
    return n % 2u ? v[n / 2u] : (v[n / 2u - 1u] + v[n / 2u]) * 0.5;
}

void run_attention(Device &device, Stream &stream, uint32_t iters) {
    constexpr uint32_t qh = 16u, kvh = 4u, hd = 256u, capacity = 4096u;
    constexpr size_t qn = qh * hd, kvn = static_cast<size_t>(capacity) * kvh * hd;
    auto q_host = luisa::vector<float>(qn);
    auto gate_host = luisa::vector<float>(qn);
    auto k_host = luisa::vector<float>(kvn);
    auto v_host = luisa::vector<float>(kvn);
    for (size_t i = 0; i < qn; ++i) {
        q_host[i] = 0.1f * std::sin(0.017f * static_cast<float>(i));
        gate_host[i] = 0.2f * std::cos(0.011f * static_cast<float>(i));
    }
    for (size_t i = 0; i < kvn; ++i) {
        k_host[i] = 0.1f * std::sin(0.003f * static_cast<float>(i));
        v_host[i] = 0.2f * std::cos(0.005f * static_cast<float>(i));
    }
    auto q = device.create_buffer<float>(qn);
    auto gate = device.create_buffer<float>(qn);
    auto k = device.create_buffer<float>(kvn);
    auto v = device.create_buffer<float>(kvn);
    auto out = device.create_buffer<float>(qn);
    stream << q.copy_from(luisa::span<const float>{q_host})
           << gate.copy_from(luisa::span<const float>{gate_host})
           << k.copy_from(luisa::span<const float>{k_host})
           << v.copy_from(luisa::span<const float>{v_host}) << synchronize();
    q_host.clear(); gate_host.clear(); k_host.clear(); v_host.clear();
    q_host.shrink_to_fit(); gate_host.shrink_to_fit(); k_host.shrink_to_fit(); v_host.shrink_to_fit();
    auto shader = device.compile(make_qwen_attn_decode_kernel());
    auto plan = plan_qwen_attn_decode_dispatch(qh);
    if (!plan.ok()) { std::exit(2); }
    for (uint32_t kv_len : {12u, 256u, 4096u}) {
        const auto record = [&] {
            auto list = CommandList::create(128u);
            for (uint32_t r = 0; r < 128u; ++r) {
                list << shader(q.view(), gate.view(), k.view(), v.view(), out.view(),
                               qh, kvh, hd, 1.0f / std::sqrt(static_cast<float>(hd)), kv_len, capacity)
                            .dispatch(qwen_dispatch_threads(*plan.value));
            }
            return list;
        };
        auto warmup = record();
        stream << warmup.commit() << synchronize();
        luisa::vector<double> times;
        for (uint32_t it = 0; it < iters; ++it) {
            auto list = record();
            Clock clock;
            clock.tic();
            stream << list.commit() << synchronize();
            times.emplace_back(clock.toc() / 128.0);
        }
        const auto min_ms = *std::min_element(times.begin(), times.end());
        const auto med_ms = median(times);
        std::printf("RESULT,attention,kv%u,decode,%u,%u,128,%.9f,%.9f,0,0\n",
                    kv_len, qh, hd, min_ms, med_ms);
    }
}

void run_shape(Device &device, Stream &stream, const Shape &shape,
               QwenWeightFormat format, const Options &options) {
    const bool fp8 = format == QwenWeightFormat::fp8_e4m3;
    const uint64_t elements = static_cast<uint64_t>(shape.n) * shape.k;
    const size_t words_per_matrix = static_cast<size_t>((elements + (fp8 ? 3u : 1u)) / (fp8 ? 4u : 2u));
    const size_t scales_per_matrix = fp8 ? static_cast<size_t>(shape.n) * (shape.k / kGroup) : 1u;
    const size_t weight_bytes = words_per_matrix * sizeof(uint32_t);
    const size_t replicas = std::max<size_t>(1u, std::min(kMaxReplicas, kTargetWeightBytes / weight_bytes));
    const size_t total_words = words_per_matrix * replicas;
    const size_t total_scales = scales_per_matrix * replicas;

    std::printf("prepare format=%s shape=%s N=%u K=%u replicas=%zu weight-working-set=%.1f MiB\n",
                fp8 ? "fp8" : "bf16", shape.name, shape.n, shape.k, replicas,
                static_cast<double>(weight_bytes * replicas) / (1024.0 * 1024.0));
    std::fflush(stdout);
    auto x_host = luisa::vector<float>(shape.k);
    for (uint32_t i = 0; i < shape.k; ++i) { x_host[i] = 0.01f * std::sin(0.013f * static_cast<float>(i)); }
    auto x = device.create_buffer<float>(shape.k);
    stream << x.copy_from(luisa::span<const float>{x_host}) << synchronize();
    x_host.clear();
    x_host.shrink_to_fit();

    auto w_host = luisa::vector<uint32_t>(total_words);
    for (size_t i = 0; i < total_words; ++i) { w_host[i] = packed_word(format, i); }
    auto w = device.create_buffer<uint32_t>(total_words);
    stream << w.copy_from(luisa::span<const uint32_t>{w_host}) << synchronize();
    w_host.clear();
    w_host.shrink_to_fit();

    auto s_host = luisa::vector<float>(total_scales, fp8 ? 0.03125f : 0.0f);
    auto scales = device.create_buffer<float>(total_scales);
    stream << scales.copy_from(luisa::span<const float>{s_host}) << synchronize();
    s_host.clear();
    s_host.shrink_to_fit();
    auto y = device.create_buffer<float>(shape.n);
    auto residual = device.create_buffer<float>(1u);

    auto generic = options.generic
                       ? device.compile(make_qwen_gemv_kernel(format, QwenMatmulEpilogue::none))
                       : Shader<1, Buffer<float>, Buffer<uint32_t>, Buffer<float>, Buffer<float>, Buffer<float>,
                                uint32_t, uint32_t, uint32_t, uint32_t>{};
    auto decode = options.decode
                      ? device.compile(make_qwen35_decode_gemv_kernel(format, QwenMatmulEpilogue::none))
                      : Shader<1, Buffer<float>, Buffer<uint32_t>, Buffer<float>, Buffer<float>, Buffer<float>,
                               uint32_t, uint32_t, uint32_t, uint32_t>{};
    auto generic_plan = plan_qwen_gemv_dispatch(1u, shape.n, shape.k);
    auto decode_plan = plan_qwen35_decode_gemv_dispatch(shape.n, shape.k, format);
    if ((options.generic && !generic_plan.ok()) || (options.decode && !decode_plan.ok())) {
        std::fprintf(stderr, "planning failed for %s\n", shape.name);
        std::exit(2);
    }

    const double effective_bytes = static_cast<double>(weight_bytes) +
                                   (fp8 ? static_cast<double>(scales_per_matrix) * 4.0 : 0.0) +
                                   static_cast<double>(shape.k + shape.n) * 4.0;
    const auto measure = [&](auto &shader, const QwenLaunch1D &plan, const char *impl) {
        const auto record = [&] {
            auto list = CommandList::create(static_cast<size_t>(replicas));
            for (size_t r = 0; r < replicas; ++r) {
                list << shader(x.view(), w.view(r * words_per_matrix, words_per_matrix),
                               scales.view(r * scales_per_matrix, scales_per_matrix),
                               residual.view(), y.view(), 1u, shape.n, shape.k, kGroup)
                            .dispatch(qwen_dispatch_threads(plan));
            }
            return list;
        };
        auto warmup = record();
        stream << warmup.commit() << synchronize();
        luisa::vector<double> samples;
        samples.reserve(options.iters);
        for (uint32_t it = 0; it < options.iters; ++it) {
            auto list = record();
            Clock clock;
            clock.tic();
            stream << list.commit() << synchronize();
            samples.emplace_back(clock.toc() / static_cast<double>(replicas));
        }
        const auto min_ms = *std::min_element(samples.begin(), samples.end());
        const auto med_ms = median(samples);
        std::printf("RESULT,%s,%s,%s,%u,%u,%zu,%.9f,%.9f,%.3f,%.3f\n",
                    fp8 ? "fp8" : "bf16", shape.name, impl, shape.n, shape.k, replicas,
                    min_ms, med_ms, effective_bytes / min_ms * 1.0e-6,
                    effective_bytes / med_ms * 1.0e-6);
        std::fflush(stdout);
    };
    if (options.generic) { measure(generic, *generic_plan.value, "generic"); }
    if (options.decode) { measure(decode, *decode_plan.value, "decode"); }
    if (options.ffn && shape.n == 9216u && shape.k == 2560u) {
        // Use distinct allocations for gate and up. Binding the same allocation
        // twice would understate real traffic and could let a compiler common
        // the two loads in the fused kernel.
        auto up_w = device.create_buffer<uint32_t>(total_words);
        auto up_scales = device.create_buffer<float>(total_scales);
        stream << up_w.copy_from(w.view()) << up_scales.copy_from(scales.view()) << synchronize();
        auto gate_y = device.create_buffer<float>(shape.n);
        auto up_y = device.create_buffer<float>(shape.n);
        auto swiglu = device.compile(make_qwen_swiglu_split_kernel());
        auto fused = device.compile(make_qwen_ffn_decode_kernel(format));
        auto swiglu_plan = plan_qwen_swiglu_dispatch(shape.n);
        auto fused_plan = plan_qwen_ffn_decode_dispatch(shape.n, shape.k, format, kGroup);
        if (!fused_plan.ok() || !swiglu_plan.ok()) { std::exit(2); }
        const auto run_ffn = [&](bool use_fused, const char *name) {
            const auto record_ffn = [&] {
                auto list = CommandList::create(static_cast<size_t>(replicas) * (use_fused ? 1u : 3u));
                for (size_t r = 0; r < replicas; ++r) {
                    const auto wv = w.view(r * words_per_matrix, words_per_matrix);
                    const auto sv = scales.view(r * scales_per_matrix, scales_per_matrix);
                    if (use_fused) {
                        const auto uv = up_w.view(r * words_per_matrix, words_per_matrix);
                        const auto usv = up_scales.view(r * scales_per_matrix, scales_per_matrix);
                        list << fused(x.view(), wv, sv, uv, usv, y.view(), shape.n, shape.k, kGroup)
                                    .dispatch(qwen_dispatch_threads(*fused_plan.value));
                    } else {
                        const auto uv = up_w.view(r * words_per_matrix, words_per_matrix);
                        const auto usv = up_scales.view(r * scales_per_matrix, scales_per_matrix);
                        list << decode(x.view(), wv, sv, residual.view(), gate_y.view(),
                                       1u, shape.n, shape.k, kGroup)
                                    .dispatch(qwen_dispatch_threads(*decode_plan.value));
                        list << decode(x.view(), uv, usv, residual.view(), up_y.view(),
                                       1u, shape.n, shape.k, kGroup)
                                    .dispatch(qwen_dispatch_threads(*decode_plan.value));
                        list << swiglu(gate_y.view(), up_y.view(), y.view(), shape.n)
                                    .dispatch(qwen_dispatch_threads(*swiglu_plan.value));
                    }
                }
                return list;
            };
            auto warmup_ffn = record_ffn();
            stream << warmup_ffn.commit() << synchronize();
            luisa::vector<double> times;
            for (uint32_t it = 0; it < options.iters; ++it) {
                auto list = record_ffn();
                Clock clock;
                clock.tic();
                stream << list.commit() << synchronize();
                times.emplace_back(clock.toc() / static_cast<double>(replicas));
            }
            const auto min_ms = *std::min_element(times.begin(), times.end());
            const auto med_ms = median(times);
            const double bytes = 2.0 * (static_cast<double>(weight_bytes) +
                                        (fp8 ? scales_per_matrix * 4.0 : 0.0)) +
                                 static_cast<double>(shape.k + 3u * shape.n) * 4.0;
            std::printf("RESULT,%s,%s,%s,%u,%u,%zu,%.9f,%.9f,%.3f,%.3f\n",
                        fp8 ? "fp8" : "bf16", shape.name, name, shape.n, shape.k, replicas,
                        min_ms, med_ms, bytes / min_ms * 1.0e-6, bytes / med_ms * 1.0e-6);
        };
        run_ffn(false, "separate_ffn");
        run_ffn(true, "fused_ffn");
    }
}

}// namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Options options;
    if (argc < 2 || !parse_options(argc, argv, options)) {
        std::fprintf(stderr,
                     "usage: %s <backend> [--format both|bf16|fp8] [--impl both|generic|decode] "
                     "[--iters N] [--lm-head] [--ffn] [--attention] [--shape NAME]\n",
                     argv[0]);
        return 2;
    }
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    if (!device) {
        std::fprintf(stderr, "failed to create backend '%s'\n", argv[1]);
        return 2;
    }
    auto stream = device.create_stream();
    std::printf("RESULT columns: format,shape,impl,N,K,replicas,min_ms,median_ms,min_effective_GBps,median_effective_GBps\n");
    if (options.attention) {
        run_attention(device, stream, options.iters);
        return 0;
    }
    for (auto format : {QwenWeightFormat::bf16, QwenWeightFormat::fp8_e4m3}) {
        if ((format == QwenWeightFormat::bf16 && !options.bf16) ||
            (format == QwenWeightFormat::fp8_e4m3 && !options.fp8)) { continue; }
        for (const auto &shape : kShapes) {
            if (options.shape.empty() || options.shape == shape.name) {
                run_shape(device, stream, shape, format, options);
            }
        }
        if ((options.lm_head || options.shape == "lm_head") &&
            (options.shape.empty() || options.shape == "lm_head")) {
            run_shape(device, stream, kLmHead, format, options);
        }
    }
    return 0;
}
