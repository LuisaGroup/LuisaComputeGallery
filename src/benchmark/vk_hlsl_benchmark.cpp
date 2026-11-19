#include "bench_common.h"

#include <luisa/core/logging.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

void print_usage(const char *prog) {
    std::printf(
        "Usage: %s [backend=vk] [options]\n"
        "Options:\n"
        "  --suite <all|compute|raytracing|physics|inference>\n"
        "  --filter <substr>\n"
        "  --warmup <N>        (default: 3)\n"
        "  --iters <N>         (default: 15)\n"
        "  --no-verify         Skip verification checks\n"
        "  --no-fast-math      Disable ShaderOption::enable_fast_math\n"
        "  --json <path>       Write results JSON to <path>\n"
        "  -h, --help          Show this help\n",
        prog);
}

[[nodiscard]] luisa::bench::BenchConfig parse_args(int argc, char *argv[]) {
    luisa::bench::BenchConfig cfg;
    int i = 1;
    if (i < argc && argv[i][0] != '-') {
        cfg.backend = argv[i++];
    }
    for (; i < argc; ++i) {
        luisa::string_view arg{argv[i]};
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--suite" && i + 1 < argc) {
            cfg.suite = argv[++i];
        } else if (arg == "--filter" && i + 1 < argc) {
            cfg.filter = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            cfg.warmup_iters = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--iters" && i + 1 < argc) {
            cfg.timed_iters = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--no-verify") {
            cfg.verify = false;
        } else if (arg == "--no-fast-math") {
            cfg.fast_math = false;
        } else if (arg == "--json" && i + 1 < argc) {
            cfg.json_path = argv[++i];
        } else {
            LUISA_WARNING("Unknown argument '{}', ignoring.", arg);
        }
    }
    LUISA_ASSERT(cfg.timed_iters >= 1u, "--iters must be >= 1");
    return cfg;
}

void write_json(const luisa::string &path,
                const luisa::bench::BenchConfig &cfg,
                const std::vector<luisa::bench::BenchResult> &results,
                double geomean_ms,
                double total_median_ms,
                double total_compile_ms) {
    std::ofstream out{path.c_str(), std::ios::binary};
    if (!out) {
        LUISA_WARNING("Failed to open JSON output file '{}'", path);
        return;
    }
    out << "{\n";
    out << "  \"backend\": \"" << cfg.backend.c_str() << "\",\n";
    out << "  \"suite\": \"" << cfg.suite.c_str() << "\",\n";
    out << "  \"warmup_iters\": " << cfg.warmup_iters << ",\n";
    out << "  \"timed_iters\": " << cfg.timed_iters << ",\n";
    out << "  \"fast_math\": " << (cfg.fast_math ? "true" : "false") << ",\n";
    out << "  \"total_compile_ms\": " << total_compile_ms << ",\n";
    out << "  \"total_median_ms\": " << total_median_ms << ",\n";
    out << "  \"geomean_median_ms\": " << geomean_ms << ",\n";
    out << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto &r = results[i];
        out << "    {\n"
            << "      \"category\": \"" << r.category.c_str() << "\",\n"
            << "      \"name\": \"" << r.name.c_str() << "\",\n"
            << "      \"compile_ms\": " << r.compile_ms << ",\n"
            << "      \"min_ms\": " << r.min_ms << ",\n"
            << "      \"median_ms\": " << r.median_ms << ",\n"
            << "      \"mean_ms\": " << r.mean_ms << ",\n"
            << "      \"p95_ms\": " << r.p95_ms << ",\n"
            << "      \"max_ms\": " << r.max_ms << ",\n"
            << "      \"throughput\": " << r.throughput << ",\n"
            << "      \"throughput_unit\": \"" << r.throughput_unit.c_str() << "\",\n"
            << "      \"verified\": " << (r.verified ? "true" : "false") << "\n"
            << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

}// namespace

int main(int argc, char *argv[]) {
    luisa::log_level_info();
    auto cfg = parse_args(argc, argv);

    luisa::compute::Context context{argv[0]};
    luisa::compute::DeviceConfig dev_cfg{};
    dev_cfg.headless = true;
    auto device = context.create_device(cfg.backend, &dev_cfg);
    auto stream = device.create_stream(luisa::compute::StreamTag::COMPUTE);

    std::printf("=== LuisaCompute VK-HLSL Benchmark ===\n");
    std::printf("backend=%s suite=%s warmup=%u iters=%u fast_math=%s verify=%s\n",
                cfg.backend.c_str(), cfg.suite.c_str(),
                cfg.warmup_iters, cfg.timed_iters,
                cfg.fast_math ? "on" : "off",
                cfg.verify ? "on" : "off");
    std::fflush(stdout);

    std::vector<luisa::bench::BenchResult> results;
    luisa::bench::run_compute_benchmarks(device, stream, cfg, results);
    luisa::bench::run_raytracing_benchmarks(device, stream, cfg, results);
    luisa::bench::run_physics_benchmarks(device, stream, cfg, results);
    luisa::bench::run_inference_benchmarks(device, stream, cfg, results);

    std::printf("\n%-12s %-25s %9s %9s %9s %9s %9s %14s %-11s %s\n",
                "category", "benchmark", "comp_ms", "min_ms", "med_ms", "mean_ms", "p95_ms", "throughput", "unit", "ok");
    std::printf("%-12s %-25s %9s %9s %9s %9s %9s %14s %-11s %s\n",
                "--------", "---------", "-------", "------", "------", "-------", "------", "----------", "----", "--");

    double total_compile_ms = 0.0;
    double total_median_ms = 0.0;
    double log_sum = 0.0;
    size_t log_cnt = 0u;
    size_t pass_cnt = 0u;

    for (const auto &r : results) {
        total_compile_ms += r.compile_ms;
        total_median_ms += r.median_ms;
        if (r.median_ms > 0.0) {
            log_sum += std::log(r.median_ms);
            ++log_cnt;
        }
        if (r.verified) {
            ++pass_cnt;
        }
        std::printf("%-12s %-25s %9.2f %9.4f %9.4f %9.4f %9.4f %14.4e %-11s %s\n",
                    r.category.c_str(),
                    r.name.c_str(),
                    r.compile_ms,
                    r.min_ms,
                    r.median_ms,
                    r.mean_ms,
                    r.p95_ms,
                    r.throughput,
                    r.throughput_unit.c_str(),
                    r.verified ? "PASS" : "FAIL");
        if (!r.verified && !r.check_note.empty()) {
            std::printf("  FAIL_NOTE[%s]: %s\n", r.name.c_str(), r.check_note.c_str());
        }
    }

    double geomean_ms = log_cnt > 0u ? std::exp(log_sum / static_cast<double>(log_cnt)) : 0.0;
    std::printf("\nSUMMARY count=%zu passed=%zu total_compile_ms=%.2f total_median_ms=%.4f geomean_median_ms=%.4f\n",
                results.size(), pass_cnt, total_compile_ms, total_median_ms, geomean_ms);
    std::fflush(stdout);

    if (!cfg.json_path.empty()) {
        write_json(cfg.json_path, cfg, results, geomean_ms, total_median_ms, total_compile_ms);
    }

    return (pass_cnt == results.size() && !results.empty()) ? 0 : 1;
}
