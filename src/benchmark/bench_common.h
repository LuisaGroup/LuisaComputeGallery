#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

namespace luisa::bench {

struct BenchConfig {
    luisa::string backend{"vk"};
    luisa::string suite{"all"};
    luisa::string filter;
    luisa::string json_path;
    uint32_t warmup_iters{3u};
    uint32_t timed_iters{15u};
    bool verify{true};
    bool fast_math{true};
};

struct BenchResult {
    luisa::string category;
    luisa::string name;
    double compile_ms{0.0};
    double min_ms{0.0};
    double median_ms{0.0};
    double mean_ms{0.0};
    double p95_ms{0.0};
    double max_ms{0.0};
    double throughput{0.0};
    luisa::string throughput_unit;
    bool verified{true};
    luisa::string check_note;
};

[[nodiscard]] inline uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30u)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27u)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31u);
}

[[nodiscard]] inline uint32_t prng_u32(uint64_t &state) noexcept {
    state = splitmix64(state);
    return static_cast<uint32_t>(state >> 32u);
}

[[nodiscard]] inline float prng_f32(uint64_t &state, float lo = 0.0f, float hi = 1.0f) noexcept {
    uint32_t r = prng_u32(state) >> 8u;
    float u = static_cast<float>(r) * (1.0f / 16777216.0f);
    return lo + (hi - lo) * u;
}

template<typename KernelT>
[[nodiscard]] inline auto compile_uncached(
    compute::Device &device,
    KernelT &&kernel,
    const char *name,
    const BenchConfig &cfg,
    double &compile_ms_accum) {
    compute::ShaderOption opt{};
    opt.enable_cache = false;
    opt.enable_fast_math = cfg.fast_math;
    opt.enable_debug_info = false;
    opt.name = name;
    Clock clk;
    auto shader = device.compile(std::forward<KernelT>(kernel), opt);
    compile_ms_accum += clk.toc();
    return shader;
}

template<typename Fn>
[[nodiscard]] inline BenchResult run_timed_bench(
    luisa::string_view category,
    luisa::string_view name,
    compute::Stream &stream,
    const BenchConfig &cfg,
    double compile_ms,
    double work_units_per_iter,
    luisa::string_view throughput_unit,
    Fn &&dispatch_fn) {
    for (uint32_t w = 0u; w < cfg.warmup_iters; ++w) {
        dispatch_fn();
        stream << compute::synchronize();
    }
    std::vector<double> samples;
    samples.reserve(cfg.timed_iters);
    for (uint32_t i = 0u; i < cfg.timed_iters; ++i) {
        Clock clk;
        dispatch_fn();
        stream << compute::synchronize();
        samples.push_back(clk.toc());
    }
    std::sort(samples.begin(), samples.end());
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    size_t n = samples.size();
    double min_ms = samples.front();
    double max_ms = samples.back();
    double mean_ms = sum / static_cast<double>(n);
    double median_ms = (n % 2u == 1u) ?
                           samples[n / 2u] :
                           0.5 * (samples[n / 2u - 1u] + samples[n / 2u]);
    size_t p95_idx = std::min(n - 1u, static_cast<size_t>(std::ceil(0.95 * static_cast<double>(n))) - 1u);
    double p95_ms = samples[p95_idx];

    double throughput = (median_ms > 0.0) ? (work_units_per_iter / (median_ms * 1e-3)) : 0.0;
    return BenchResult{
        .category = luisa::string{category},
        .name = luisa::string{name},
        .compile_ms = compile_ms,
        .min_ms = min_ms,
        .median_ms = median_ms,
        .mean_ms = mean_ms,
        .p95_ms = p95_ms,
        .max_ms = max_ms,
        .throughput = throughput,
        .throughput_unit = luisa::string{throughput_unit},
        .verified = true,
        .check_note = {},
    };
}

[[nodiscard]] inline bool should_run(
    const BenchConfig &cfg,
    luisa::string_view category,
    luisa::string_view name) noexcept {
    if (cfg.suite != "all" && cfg.suite != category) {
        return false;
    }
    if (!cfg.filter.empty() && name.find(cfg.filter) == luisa::string_view::npos) {
        return false;
    }
    return true;
}

void run_compute_benchmarks(
    compute::Device &device,
    compute::Stream &stream,
    const BenchConfig &cfg,
    std::vector<BenchResult> &results);

void run_raytracing_benchmarks(
    compute::Device &device,
    compute::Stream &stream,
    const BenchConfig &cfg,
    std::vector<BenchResult> &results);

void run_physics_benchmarks(
    compute::Device &device,
    compute::Stream &stream,
    const BenchConfig &cfg,
    std::vector<BenchResult> &results);

void run_inference_benchmarks(
    compute::Device &device,
    compute::Stream &stream,
    const BenchConfig &cfg,
    std::vector<BenchResult> &results);

}// namespace luisa::bench
