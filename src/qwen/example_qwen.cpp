// example_qwen -- command-line front end for the text-only Qwen3.5-4B engine.
//
// Four explicitly selectable variants (no silent fallback):
//   dsl-bf16  dsl-fp8  tile-bf16  tile-fp8
// BF16 weights are the raw GGUF BF16 payload packed 2-per-uint32; FP8 means
// true E4M3 quantized from the BF16 source AT LOAD TIME (per-row groups of 64
// K elements, FP32 scales) and the conversion is reported. Q8_0 GGUF files
// are INT8 block quantization, NOT FP8, and are rejected by the uploader.
//
// Offline entry points (no device, no model file):
//   example_qwen --list-versions   support matrix + precision semantics
//   example_qwen --self-test       DSL + Tile operator self-tests and the
//                                  tokenizer self-test (device-free)
//   example_qwen --help
//
// Generation (needs a backend plugin and data/Qwen3.5-4B-BF16.gguf):
//   example_qwen --backend cuda --version dsl-bf16 --prompt "Hello" --max-new-tokens 32
// example_qwen cuda --headless --version tile-bf16 --prompt "Hello" --chat
// Greedy (temperature 0) or reproducible temperature-based categorical
// sampling; stops on the GGUF EOS (<|im_end|>). No vision tower. MTP and
// WY/UT prefill support are reported explicitly by the CLI and engine.
//
// Exit codes: 0 success/help/list/self-test; 1 bad arguments; 2 engine or
// model refusal (reported, never fabricated output).
//
// Encoding: Windows narrow argv is the ANSI code page, NOT UTF-8. On Windows
// wmain converts every argument with WideCharToMultiByte(CP_UTF8) so prompts
// and paths reach the tokenizer as UTF-8 (same convention as example_ocr).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <string>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "qwen/dsl_ops.h"
#include "qwen/engine.h"
#include "qwen/gguf.h"
#include "qwen/model.h"
#include "qwen/mtp.h"
#include "qwen/mtp_executor.h"
#include "qwen/tile_ops.h"
#include "qwen/tokenizer.h"
#include "qwen/weights.h"

#ifdef _WIN32
#include <windows.h>
#endif

using namespace luisa;
using namespace luisa::qwen;

namespace {

// ---------------------------------------------------------------------------
// CLI surface
// ---------------------------------------------------------------------------

struct VersionSpec {
    luisa::string_view name;
    QwenKernelKind kernel;
    bool fp8;
    uint32_t default_prefill_chunk;
    luisa::string_view status;
};

// The support matrix. Status strings reflect the verified state recorded by
// the kernel/executor agents (see docs/examples/example_qwen.md); they are
// claims about prior verification runs, not live probes.
constexpr VersionSpec kVersions[] = {
    {"dsl-bf16", QwenKernelKind::dsl, false, 64,
     "real-GGUF inference verified on cuda/dx/vk; 467-check device suite passes on all three"},
    {"dsl-fp8", QwenKernelKind::dsl, true, 64,
     "real-GGUF coherent chat verified on cuda; 467-check device suite passes on cuda/dx/vk"},
    {"tile-bf16", QwenKernelKind::tile, false, 1,
     "verified on dx; vk 86-check suite and short real-model inference pass; cuda Tile path unavailable"},
    {"tile-fp8", QwenKernelKind::tile, true, 1,
     "verified on dx; vk 86-check suite and short real-model inference pass; cuda Tile path unavailable"},
};

struct CliArgs {
    bool help{false};
    bool list_versions{false};
    bool self_test{false};
    bool headless{false};
    bool chat{false};
    bool gdn_recurrent{false};
    bool benchmark{false};
    bool mtp_test{false};
    bool mtp{false};
    uint32_t draft_depth{2};
    bool have_benchmark_csv{false};
    luisa::string benchmark_csv;
    luisa::string backend;
    luisa::string version{"dsl-bf16"};
    luisa::string model{"data/Qwen3.5-4B-BF16.gguf"};
    luisa::string prompt;
    bool have_prompt{false};
    luisa::vector<uint32_t> tokens;
    bool have_tokens{false};
    uint32_t max_new_tokens{32};
    uint32_t decode_flight{4};// greedy CUDA: forwards per bounded host sync; 0 disables
    float temperature{0.0f};
    uint64_t seed{0x51f15eED1234abcdull};
    uint32_t max_context{0};// 0 = per-kernel-family safe default
    uint32_t prefill_chunk{0};// 0 = per-version default
};

void print_usage() {
    std::fputs(
        "example_qwen -- text-only Qwen3.5-4B inference (LuisaCompute)\n"
        "\n"
        "usage:\n"
        "  example_qwen --help | --list-versions | --self-test     (offline, no device)\n"
        "  example_qwen [<backend>] [--headless] --backend <name> [options]\n"
        "\n"
        "options:\n"
        "  --backend <name>        backend plugin: cuda | dx | vk | ...\n"
        "  --version <name>        dsl-bf16 (default) | dsl-fp8 | tile-bf16 | tile-fp8\n"
        "  --model <file.gguf>     default data/Qwen3.5-4B-BF16.gguf (Q8_0 is NOT FP8; rejected)\n"
        "  --prompt <text>         UTF-8 prompt text (mutually exclusive with --tokens)\n"
        "  --tokens <id,id,...>    explicit prompt token ids (alternative to --prompt)\n"
        "  --chat                  wrap the prompt in the Qwen im_start user/assistant template\n"
        "                          (non-thinking default; --prompt only)\n"
        "  --max-new-tokens <n> generated-token budget (default 32)\n"
        "  --decode-flight <n> greedy CUDA DSL forwards per sync (default 4; 0 disables)\n"
        "  --temperature <t> 0 = greedy argmax (default); >0 = categorical sampling\n"
        "  --seed <n>              sampling seed; reproducible by generated-token index\n"
        "  --max-context <n>       KV capacity (default 4096 dsl / 64 tile, <= 262144)\n"
        "  --prefill-chunk <n> prompt rows per engine prefill chunk (default 64 dsl / 1 tile)\n"
        "  --gdn-recurrent        DSL: use per-token recurrence instead of WY/UT chunk-64\n"
        "  --benchmark synchronize after every kernel/transfer and print raw wall timings\n"
        "  --benchmark-csv <file> benchmark mode; append every raw sample to CSV\n"
        "  --mtp-test probe GGUF NextN/MTP weights and measured-default policy; no GPU needed\n"
        "  --mtp                   enable real DSL NextN draft+verify (requires blk.32.* GGUF)\n"
        "  --draft-depth <n>       drafted tokens per MTP round (default 2; prefill chunk >= n+1)\n"
        "  --self-test device-free DSL + Tile operator and tokenizer self-tests\n"
        "  --list-versions         offline support matrix + precision semantics\n"
        "  --headless              accepted for gallery convention; this tool never opens a window\n"
        "\n"
        "precision semantics:\n"
        "  bf16: raw GGUF BF16 payload, packed two codes per uint32 (no conversion)\n"
        "  fp8 : E4M3 quantized from BF16 AT LOAD TIME, FP32 scale per 64-element K group;\n"
        "        the conversion is reported per matrix. Q8_0 GGUF is INT8, never FP8.\n"
        "  all accumulations/norms/GDN state/softmax are FP32; host sampling is FP64.\n",
        stdout);
}

void print_versions() {
    std::fputs("example_qwen support matrix (text-only Qwen3.5-4B, single sequence)\n\n", stdout);
    for (const auto &v : kVersions) {
        std::printf("  %-9s kernel=%-4s weights=%-3s default-prefill-chunk=%u\n",
                    luisa::string{v.name}.c_str(),
                    luisa::string{qwen_kernel_kind_name(v.kernel)}.c_str(),
                    v.fp8 ? "fp8" : "bf16", v.default_prefill_chunk);
        std::printf("            status: %s\n", luisa::string{v.status}.c_str());
    }
    std::fputs(
        "\n"
        "precision semantics:\n"
        "  bf16 = raw GGUF BF16 bytes packed 2-per-uint32 (little-endian, low element first)\n"
        "  fp8  = true E4M3 (max 448) quantized from the BF16 source AT LOAD TIME,\n"
        "         FP32 scale per contiguous group of 64 K elements per output row;\n"
        "         every converted matrix is reported (fp8_quantized diagnostic).\n"
        "  Q8_0 GGUF files are INT8 block quantization, NOT FP8, and are rejected;\n"
        "  no silent dequantize/requantize path is enabled here.\n"
        "  FP32 everywhere else: accumulations, norms, softmax, GDN recurrent state.\n"
        "\n"
        "scope: text-only backbone (24 GDN + 8 full-attention layers), tied LM head,\n"
        "  greedy or temperature sampling with GGUF EOS stop; DSL WY/UT chunk-64 prefill;\n"
        "  opt-in DSL NextN/MTP draft+verify with complete blk.32.* weights.\n"
        "  NOT implemented: vision tower, top-k/top-p, Tile WY/UT or Tile MTP.\n",
        stdout);
}

[[nodiscard]] bool parse_u32(luisa::string_view s, uint32_t &out) noexcept {
    if (s.empty()) { return false; }
    uint64_t v = 0;
    for (auto c : s) {
        if (c < '0' || c > '9') { return false; }
        v = v * 10u + static_cast<uint64_t>(c - '0');
        if (v > 0xffffffffull) { return false; }
    }
    out = static_cast<uint32_t>(v);
    return true;
}

[[nodiscard]] bool parse_u64(luisa::string_view s, uint64_t &out) noexcept {
    if (s.empty()) { return false; }
    uint64_t v = 0;
    for (auto c : s) {
        if (c < '0' || c > '9') { return false; }
        const auto digit = static_cast<uint64_t>(c - '0');
        if (v > (UINT64_MAX - digit) / 10u) { return false; }
        v = v * 10u + digit;
    }
    out = v;
    return true;
}

[[nodiscard]] bool parse_temperature(luisa::string_view s, float &out) noexcept {
    if (s.empty()) { return false; }
    luisa::string owned{s};
    char *end = nullptr;
    const auto value = std::strtof(owned.c_str(), &end);
    if (end != owned.c_str() + owned.size() || !std::isfinite(value) || value < 0.0f) { return false; }
    out = value;
    return true;
}

[[nodiscard]] bool parse_token_list(luisa::string_view s, luisa::vector<uint32_t> &out) {
    out.clear();
    size_t pos = 0;
    while (pos <= s.size()) {
        const auto comma = s.find(',', pos);
        const auto end = comma == luisa::string_view::npos ? s.size() : comma;
        const auto item = s.substr(pos, end - pos);
        uint32_t id = 0;
        if (!parse_u32(item, id)) { return false; }
        out.emplace_back(id);
        if (comma == luisa::string_view::npos) { break; }
        pos = comma + 1;
        if (pos == s.size()) { return false; }// trailing comma
    }
    return !out.empty();
}

// Returns false on a bad argument (message already printed).
[[nodiscard]] bool parse_args(int argc, const char *const *argv, CliArgs &args) {
    bool positional_backend = false;
    for (int i = 1; i < argc; i++) {
        const luisa::string_view a{argv[i]};
        const auto next = [&](luisa::string_view &out) -> bool {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "example_qwen: missing value for %s\n", luisa::string{a}.c_str());
                return false;
            }
            out = luisa::string_view{argv[++i]};
            return true;
        };
        if (a == "--help" || a == "-h") {
            args.help = true;
        } else if (a == "--list-versions") {
            args.list_versions = true;
        } else if (a == "--self-test") {
            args.self_test = true;
        } else if (a == "--headless" || a == "--offline") {
            args.headless = true;
        } else if (a == "--chat") {
            args.chat = true;
        } else if (a == "--gdn-recurrent") {
            args.gdn_recurrent = true;
        } else if (a == "--benchmark") {
            args.benchmark = true;
        } else if (a == "--benchmark-csv") {
            luisa::string_view v;
            if (!next(v)) { return false; }
            args.benchmark_csv = luisa::string{v};
            args.have_benchmark_csv = true;
        } else if (a == "--mtp-test") {
            args.mtp_test = true;
        } else if (a == "--mtp") {
            args.mtp = true;
        } else if (a == "--draft-depth") {
            luisa::string_view v;
            if (!next(v) || !parse_u32(v, args.draft_depth) || args.draft_depth == 0u) {
                std::fprintf(stderr, "example_qwen: --draft-depth expects an integer >= 1\n");
                return false;
            }
        } else if (a == "--backend") {
            luisa::string_view v;
            if (!next(v)) { return false; }
            args.backend = luisa::string{v};
        } else if (a == "--version") {
            luisa::string_view v;
            if (!next(v)) { return false; }
            args.version = luisa::string{v};
        } else if (a == "--model") {
            luisa::string_view v;
            if (!next(v)) { return false; }
            args.model = luisa::string{v};
        } else if (a == "--prompt") {
            luisa::string_view v;
            if (!next(v)) { return false; }
            args.prompt = luisa::string{v};
            args.have_prompt = true;
        } else if (a == "--tokens") {
            luisa::string_view v;
            if (!next(v)) { return false; }
            if (!parse_token_list(v, args.tokens)) {
                std::fprintf(stderr, "example_qwen: --tokens expects a comma-separated id list\n");
                return false;
            }
            args.have_tokens = true;
        } else if (a == "--max-new-tokens") {
            luisa::string_view v;
            if (!next(v) || !parse_u32(v, args.max_new_tokens)) { return false; }
} else if (a == "--decode-flight") {
 luisa::string_view v;
 if (!next(v) || !parse_u32(v, args.decode_flight)) { return false; }
 } else if (a == "--temperature") {
 luisa::string_view v;
            if (!next(v) || !parse_temperature(v, args.temperature)) {
                std::fprintf(stderr, "example_qwen: --temperature expects a finite value >= 0\n");
                return false;
            }
        } else if (a == "--seed") {
            luisa::string_view v;
            if (!next(v) || !parse_u64(v, args.seed)) {
                std::fprintf(stderr, "example_qwen: --seed expects an unsigned 64-bit integer\n");
                return false;
            }
        } else if (a == "--max-context") {
            luisa::string_view v;
            if (!next(v) || !parse_u32(v, args.max_context)) { return false; }
        } else if (a == "--prefill-chunk") {
            luisa::string_view v;
            if (!next(v) || !parse_u32(v, args.prefill_chunk)) { return false; }
        } else if (!a.empty() && a[0] != '-' && !positional_backend && args.backend.empty()) {
            args.backend = luisa::string{a};// gallery convention: positional backend
            positional_backend = true;
        } else {
            std::fprintf(stderr, "example_qwen: unknown argument '%s' (see --help)\n",
                         luisa::string{a}.c_str());
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Self-test (device-free)
// ---------------------------------------------------------------------------

[[nodiscard]] int run_self_test() {
    int failures = 0;
    const auto run = [&](const char *name, Status s, const luisa::vector<luisa::string> &report) {
        for (const auto &line : report) { std::printf("  [%s] %s\n", name, line.c_str()); }
        if (s.ok()) {
            std::printf("[%s] PASS (%zu report lines)\n", name, report.size());
        } else {
            std::printf("[%s] FAIL: %s (%s)\n", name, s.message.c_str(),
                        luisa::string{status_code_name(s.code)}.c_str());
            failures += 1;
        }
    };
    {
        luisa::vector<luisa::string> report;
        const auto s = dsl::dsl_ops_self_test(&report);
        run("dsl-ops", s, report);
    }
    {
        luisa::vector<luisa::string> report;
        const auto r = tile_ops::tile_ops_self_test(&report);
        Status s = r.ok() ? Status::success()
                          : Status::error(StatusCode::internal_error, r.message);
        run("tile-ops", s, report);
    }
    {
        luisa::vector<luisa::string> report;
        const auto s = QwenTokenizer::self_test(&report);
        run("tokenizer", s, report);
    }
    {
        luisa::vector<luisa::string> report;
        const auto s = mtp_self_test(&report);
        run("mtp-policy", s, report);
    }
    // Shared sampler contract (host, deterministic).
    {
        const float tied[] = {1.0f, 3.0f, 3.0f, 2.0f};
        auto a = qwen_argmax(tied);
        const float nan[] = {1.0f, NAN};
        auto b = qwen_argmax(nan);
        auto c = qwen_argmax(luisa::span<const float>{});
        if (a.ok() && *a.value == 1u && !b.ok() && !c.ok()) {
            std::printf("[argmax] PASS (lowest-index ties, NaN rejection, empty rejection)\n");
        } else {
            std::printf("[argmax] FAIL\n");
            failures += 1;
        }
    }
    std::printf("self-test: %s\n", failures == 0 ? "ALL PASS" : "FAILURES");
    return failures == 0 ? 0 : 2;
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

// argv[0] for the Context (set by run_cli before run_generation).
const char *&argv0_holder() {
    static const char *value = "example_qwen";
    return value;
}

[[nodiscard]] double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

[[nodiscard]] int run_mtp_test(const CliArgs &args) {
    luisa::filesystem::path path;
    if (!luisa::path_from_utf8(args.model, path)) {
        std::fprintf(stderr, "example_qwen: --model path is not valid UTF-8\n");
        return 1;
    }
    auto reader = GgufReader::open(path);
    if (!reader.ok()) {
        std::fprintf(stderr, "example_qwen: cannot open model: %s (%s)\n",
                     reader.status.message.c_str(), reader.status.detail.c_str());
        return 2;
    }
    const auto report = probe_mtp_eligibility(**reader.value);
    for (const auto &line : report.report_lines()) { std::printf("%s\n", line.c_str()); }
    std::printf("mtp policy: available=%s benchmarked=%s accelerates=%s default=%s\n",
                report.available ? "true" : "false",
                report.benchmarked ? "true" : "false",
                report.accelerates ? "true" : "false",
                report.default_enabled ? "enabled" : "disabled");
    return 0;
}

[[nodiscard]] Status emit_benchmark(const CliArgs &args, QwenEngine &engine,
                                    uint64_t iteration, bool &csv_started) {
    const auto *report = engine.benchmark_report();
    if (report == nullptr) {
        return Status::error(StatusCode::internal_error,
                             "benchmark mode requested but engine returned no report");
    }
    std::printf("benchmark: commands=%llu submits=%llu syncs=%llu record=%.3f ms "
                "submit+sync=%.3f ms sampled-total=%.3f ms "
                "(synchronized HOST WALL, not pure GPU time)\n",
                static_cast<unsigned long long>(report->command_count),
                static_cast<unsigned long long>(report->submit_count),
                static_cast<unsigned long long>(report->synchronize_count),
                report->record_ms, report->submit_sync_ms, report->total_ms);
    if (args.benchmark) {
        for (const auto &s : report->samples) {
            std::printf("  [%04llu] %s pos=%u rows=%u layer=", static_cast<unsigned long long>(s.ordinal),
                        luisa::string{qwen_benchmark_phase_name(s.phase)}.c_str(), s.position, s.rows);
            if (s.layer == UINT32_MAX) { std::printf("global"); } else { std::printf("%u", s.layer); }
            std::printf(" %-52s record=%8.4f ms submit+sync=%9.4f ms total=%9.4f ms\n",
                        s.name.c_str(), s.record_ms, s.submit_sync_ms, s.total_ms);
        }
    }
    if (args.have_benchmark_csv) {
        luisa::filesystem::path path;
        if (!luisa::path_from_utf8(args.benchmark_csv, path)) {
            return Status::error(StatusCode::invalid_argument, "benchmark CSV path is not valid UTF-8");
        }
        auto status = qwen_benchmark_append_csv(path, *report, csv_started,
                                                args.version, args.backend, iteration);
        if (!status.ok()) { return status; }
        csv_started = true;
    }
    return Status::success();
}

[[nodiscard]] int run_generation(const CliArgs &args) {
    // --- resolve the variant -------------------------------------------------
    const VersionSpec *spec = nullptr;
    for (const auto &v : kVersions) {
        if (v.name == args.version) { spec = &v; }
    }
    if (spec == nullptr) {
        std::fprintf(stderr, "example_qwen: unknown --version '%s' (see --list-versions)\n",
                     args.version.c_str());
        return 1;
    }
    if (args.backend.empty()) {
        std::fprintf(stderr, "example_qwen: --backend is required for generation (cuda|dx|vk)\n");
        return 1;
    }
    if (args.have_prompt == args.have_tokens) {
        std::fprintf(stderr, "example_qwen: exactly one of --prompt / --tokens is required\n");
        return 1;
    }
    if (args.chat && !args.have_prompt) {
        std::fprintf(stderr, "example_qwen: --chat requires --prompt\n");
        return 1;
    }
    if (args.max_new_tokens == 0) {
        std::fprintf(stderr, "example_qwen: --max-new-tokens must be >= 1\n");
        return 1;
    }
    if (args.mtp && spec->kernel != QwenKernelKind::dsl) {
        std::fprintf(stderr, "example_qwen: --mtp currently requires a dsl-* version (no Tile draft fallback)\n");
        return 1;
    }
    uint32_t prefill_chunk = args.prefill_chunk != 0 ? args.prefill_chunk
                                                     : spec->default_prefill_chunk;
    if (args.mtp) { prefill_chunk = std::max(prefill_chunk, args.draft_depth + 1u); }
    const uint32_t max_context = args.max_context != 0 ? args.max_context
                                                       : (spec->kernel == QwenKernelKind::tile ? 64u : 4096u);

    std::printf("example_qwen: version=%s backend=%s model=%s\n", args.version.c_str(),
                args.backend.c_str(), args.model.c_str());
    std::printf("precision: %s\n",
                spec->fp8 ? "FP8 E4M3 -- quantized from the BF16 GGUF source AT LOAD TIME "
                            "(FP32 scale per 64-element K group per row; conversion reported below). "
                            "Q8_0 is INT8, NOT FP8, and is rejected."
                          : "BF16 -- raw GGUF BF16 payload packed 2-per-uint32 (no conversion). "
                            "Q8_0 is INT8, NOT FP8, and is rejected.");
    std::printf("prefill: %s in chunks of %u rows%s\n",
                spec->kernel == QwenKernelKind::dsl && !args.gdn_recurrent ?
                    "FP32 WY/UT chunk-64 GDN" : "exact per-token GDN recurrence",
                prefill_chunk, args.mtp ? "; MTP draft+verify enabled" : "");

    // --- GGUF + model + tokenizer (host) --------------------------------------
    const auto t_open = std::chrono::steady_clock::now();
    luisa::filesystem::path model_path;
    if (!luisa::path_from_utf8(args.model, model_path)) {
        std::fprintf(stderr, "example_qwen: --model path is not valid UTF-8\n");
        return 1;
    }
    auto reader = GgufReader::open(model_path);
    if (!reader.ok()) {
        std::fprintf(stderr, "example_qwen: cannot open model: %s (%s)\n",
                     reader.status.message.c_str(), reader.status.detail.c_str());
        return 2;
    }
    // Explicit Q8_0 refusal with a clear message (the uploader also fails
    // closed; this makes the reason visible before any device work).
    for (const auto &t : (*reader.value)->tensors()) {
        if (t.type == static_cast<uint32_t>(GgufTensorType::q8_0)) {
            std::fprintf(stderr,
                         "example_qwen: '%s' is Q8_0 (INT8 block quantization). Q8_0 is NOT FP8 "
                         "and is not supported; use the BF16 GGUF (fp8 variants quantize it at "
                         "load time).\n",
                         t.name.c_str());
            return 2;
        }
    }
    if (args.mtp) {
        const auto eligibility = probe_mtp_eligibility(**reader.value);
        if (!eligibility.benchmark_feasible) {
            std::fprintf(stderr, "example_qwen: --mtp refused: %s\n", eligibility.reason.c_str());
            return 2;
        }
    }
    auto model = args.mtp ? Qwen35Model::load_with_nextn(**reader.value)
                          : Qwen35Model::load(**reader.value);
    if (!model.ok()) {
        std::fprintf(stderr, "example_qwen: model validation failed: %s (%s)\n",
                     model.status.message.c_str(), model.status.detail.c_str());
        return 2;
    }
    auto tokenizer = QwenTokenizer::load(**reader.value, (*model.value).config().geometry.vocab);
    if (!tokenizer.ok()) {
        std::fprintf(stderr, "example_qwen: tokenizer load failed: %s (%s)\n",
                     tokenizer.status.message.c_str(), tokenizer.status.detail.c_str());
        return 2;
    }
    const auto &tok = **tokenizer.value;
    std::printf("tokenizer: vocab=%llu merges=%llu specials=%llu eos=%u (%s) pre=%s [exact qwen35 BPE]\n",
                static_cast<unsigned long long>(tok.vocab_size()),
                static_cast<unsigned long long>(tok.info().merge_count),
                static_cast<unsigned long long>(tok.info().special_token_count),
                tok.eos_id(), tok.id_to_token(tok.eos_id()).value_or("?").c_str(),
                tok.info().pre.c_str());

    // --- prompt ids -------------------------------------------------------------
    luisa::vector<uint32_t> prompt_ids;
    if (args.have_tokens) {
        prompt_ids = args.tokens;
        for (auto id : prompt_ids) {
            if (id >= tok.vocab_size()) {
                std::fprintf(stderr, "example_qwen: token id %u outside vocab\n", id);
                return 1;
            }
        }
    } else if (args.chat) {
        auto ids = tok.chat_prompt_ids(args.prompt);
        if (!ids.ok()) {
            std::fprintf(stderr, "example_qwen: chat prompt rejected: %s\n", ids.status.message.c_str());
            return 1;
        }
        prompt_ids = std::move(*ids.value);
    } else {
        auto ids = tok.encode(args.prompt, false);
        if (!ids.ok()) {
            std::fprintf(stderr, "example_qwen: prompt encode failed: %s\n", ids.status.message.c_str());
            return 1;
        }
        prompt_ids = std::move(*ids.value);
    }
    if (prompt_ids.empty()) {
        std::fprintf(stderr, "example_qwen: empty prompt\n");
        return 1;
    }
    if (prompt_ids.size() + args.max_new_tokens > max_context) {
        std::fprintf(stderr,
                     "example_qwen: prompt (%zu) + max-new-tokens (%u) exceeds max-context (%u)\n",
                     prompt_ids.size(), args.max_new_tokens, max_context);
        return 1;
    }
    {
        auto prompt_text = tok.decode(prompt_ids, false, true);
        std::printf("prompt: %zu tokens%s\n", prompt_ids.size(), args.chat ? " [chat wrapper]" : "");
        if (prompt_text.ok()) { std::printf("prompt text: %s\n", (*prompt_text.value).c_str()); }
        std::printf("prompt ids:");
        for (auto id : prompt_ids) { std::printf(" %u", id); }
        std::printf("\n");
    }
    const double host_load_ms = ms_since(t_open);

    // --- device / stream / weights / engine --------------------------------------
    // Destruction order is explicit below: engine -> weights -> stream -> device
    // -> context. Every engine/weight buffer must be released while its device
    // is still alive (backend teardown crashes otherwise).
    compute::Context context{argv0_holder()};
    auto device = context.create_device(args.backend.c_str());
    if (!device) {
        std::fprintf(stderr, "example_qwen: backend '%s' not available (installed:", args.backend.c_str());
        for (const auto &b : context.installed_backends()) { std::fprintf(stderr, " %s", b.c_str()); }
        std::fprintf(stderr, ")\n");
        return 2;
    }
    auto stream = device.create_stream();
    if (!stream) {
        std::fprintf(stderr, "example_qwen: cannot create stream\n");
        return 2;
    }

    int exit_code = 0;
    {
        const auto t_upload = std::chrono::steady_clock::now();
        QwenWeightUploadOptions up;
        up.fp8 = spec->fp8;
        QwenWeightUploadReport report;
        auto weights = QwenDeviceWeights::upload(device, stream, **reader.value, *model.value, up, &report);
        if (!weights.ok()) {
            std::fprintf(stderr, "example_qwen: weight upload failed: %s (%s)\n",
                         weights.status.message.c_str(), weights.status.detail.c_str());
            return 2;
        }
        luisa::optional<QwenDeviceWeights> draft_weights;
        QwenWeightUploadReport draft_report;
        if (args.mtp) {
            auto uploaded = QwenDeviceWeights::upload_nextn(device, stream, **reader.value, up, &draft_report);
            if (!uploaded.ok()) {
                std::fprintf(stderr, "example_qwen: NextN weight upload failed: %s (%s)\n",
                             uploaded.status.message.c_str(), uploaded.status.detail.c_str());
                return 2;
            }
            draft_weights.emplace(std::move(*uploaded.value));
        }
        stream.synchronize();
        const double upload_ms = ms_since(t_upload);
        std::printf("weights: %llu tensors (f32=%llu bf16=%llu fp8=%llu), %.2f MiB device, upload %.1f ms\n",
                    static_cast<unsigned long long>(report.tensor_count),
                    static_cast<unsigned long long>(report.f32_count),
                    static_cast<unsigned long long>(report.bf16_count),
                    static_cast<unsigned long long>(report.fp8_count),
                    static_cast<double>(report.device_bytes) / (1024.0 * 1024.0), upload_ms);
        if (spec->fp8) {
            std::printf("fp8 conversion: %zu matrices quantized BF16->E4M3 at load time "
                        "(per-row groups of 64, FP32 scales)\n",
                        report.diagnostics.size());
        }

        QwenEngineOptions options;
        options.kernel = spec->kernel;
        options.max_context = max_context;
        options.prefill_chunk = prefill_chunk;
        options.dsl_gdn_wy_ut = !args.gdn_recurrent;
        options.benchmark = args.benchmark || args.have_benchmark_csv;
        const auto t_compile = std::chrono::steady_clock::now();
        QwenEnginePtr engine;
        QwenMtpExecutor::Ptr mtp;
        QwenEngine *engine_ptr = nullptr;
        if (args.mtp) {
            auto created = QwenMtpExecutor::create(device, stream, *model.value, *weights.value,
                                                   *draft_weights, options, args.draft_depth);
            if (!created.ok()) {
                std::fprintf(stderr, "example_qwen: MTP engine creation failed: %s (%s)\n",
                             created.status.message.c_str(), created.status.detail.c_str());
                return 2;
            }
            mtp = std::move(*created.value);
            engine_ptr = mtp->target();
            std::printf("mtp: draft_depth=%u history=%.2f MiB snapshots=%.2f MiB verify-logits=%.2f MiB\n",
                        args.draft_depth,
                        static_cast<double>(mtp->info().history_bytes) / (1024.0 * 1024.0),
                        static_cast<double>(mtp->info().snapshot_bytes) / (1024.0 * 1024.0),
                        static_cast<double>(mtp->info().verify_logits_bytes) / (1024.0 * 1024.0));
        } else {
            auto created = QwenEngine::create(device, stream, *model.value, *weights.value, options);
            if (!created.ok()) {
                std::fprintf(stderr, "example_qwen: engine creation failed: %s (%s)\n",
                             created.status.message.c_str(), created.status.detail.c_str());
                return 2;
            }
            engine = std::move(*created.value);
            engine_ptr = engine.get();
        }
        const double compile_ms = ms_since(t_compile);
        {
            const auto info = engine_ptr->info();
            std::printf("engine: kernel=%s storage=%s max_context=%u prefill_chunk=%u gdn=%s "
                        "cache=%.2f MiB workspace=%.2f MiB shaders=%llu (compile %.1f ms)\n",
                        luisa::string{qwen_kernel_kind_name(info.kernel)}.c_str(),
                        luisa::string{weight_storage_name(info.weight_storage)}.c_str(),
                        info.max_context, info.prefill_chunk,
                        info.kernel == QwenKernelKind::dsl ? (info.gdn_wy_ut ? "wy-ut-64" : "recurrent") : "recurrent-tile",
                        static_cast<double>(info.cache_bytes) / (1024.0 * 1024.0),
                        static_cast<double>(info.workspace_bytes) / (1024.0 * 1024.0),
                        static_cast<unsigned long long>(info.compiled_shader_count), compile_ms);
            std::printf("variant status: %s\n", luisa::string{spec->status}.c_str());
        }

        // --- generation ------------------------------------------------------------
        QwenSampleOptions sample_options;
        sample_options.temperature = args.temperature;
        sample_options.seed = args.seed;
        std::printf("sampling: temperature=%g seed=%llu\n", static_cast<double>(args.temperature),
                    static_cast<unsigned long long>(args.seed));
        auto detok = tok.make_detokenizer(true);
        luisa::vector<float> logits;
        luisa::vector<uint32_t> generated;
        const auto t_prefill = std::chrono::steady_clock::now();
        auto st = args.mtp ? mtp->prefill(stream, prompt_ids, logits)
                           : engine_ptr->prefill(stream, prompt_ids, logits);
        const double ttft_ms = ms_since(t_prefill);
        if (!st.ok()) {
            std::fprintf(stderr, "example_qwen: prefill failed: %s (%s)\n",
                         st.message.c_str(), st.detail.c_str());
            exit_code = 2;
        }
        double decode_total_ms = 0.0;
        double argmax_total_ms = 0.0;
        double detok_total_ms = 0.0;
        uint32_t decode_steps = 0;
        bool eos_hit = false;
        bool csv_started = false;
        uint64_t benchmark_iteration = 0;
        if (st.ok() && options.benchmark) {
            st = emit_benchmark(args, *engine_ptr, benchmark_iteration++, csv_started);
            if (!st.ok()) {
                std::fprintf(stderr, "example_qwen: benchmark reporting failed: %s (%s)\n",
                             st.message.c_str(), st.detail.c_str());
                exit_code = 2;
            }
        }
        // Emit one id with the historical EOS/budget ordering: EOS counts but
        // is never decoded; the budget is checked before another forward.
        const auto emit_one = [&](uint32_t id) -> bool {
            generated.emplace_back(id);
            if (id == tok.eos_id()) {
                eos_hit = true;
                return false;
            }
            const auto t_detok = std::chrono::steady_clock::now();
            auto piece = detok.push(id);
            detok_total_ms += ms_since(t_detok);
            if (!piece.ok()) {
                std::fprintf(stderr, "example_qwen: detokenize failed: %s\n", piece.status.message.c_str());
                exit_code = 2;
                return false;
            }
            if (!(*piece.value).empty()) {
                std::fputs((*piece.value).c_str(), stdout);
                std::fflush(stdout);
            }
            return generated.size() < args.max_new_tokens;
        };
        uint32_t pending = 0u;
        bool have_pending = false;
        const bool use_flight = args.decode_flight != 0u && !args.mtp &&
                                !options.benchmark && args.temperature == 0.0f &&
                                args.backend == "cuda" &&
                                engine_ptr->supports_device_greedy_decode();
        if (use_flight) {
            std::printf("decode policy: device-greedy in-flight chunk=%u\n", args.decode_flight);
        }
        if (use_flight && exit_code == 0) {
            // The prefill still uses the reference host sampler for token zero.
            // Subsequent selections stay on-device until each bounded chunk.
            const auto t_select = std::chrono::steady_clock::now();
            auto next = qwen_sample(logits, sample_options, 0u);
            argmax_total_ms += ms_since(t_select);
            if (!next.ok()) {
                std::fprintf(stderr, "example_qwen: token selection failed: %s\n",
                             next.status.message.c_str());
                exit_code = 2;
            } else {
                pending = *next.value;
            }
            bool need_host_input = true;
            if (exit_code == 0 && emit_one(pending)) {
                luisa::vector<uint32_t> flight_ids;
                while (exit_code == 0 && !eos_hit && generated.size() < args.max_new_tokens) {
                    const auto remaining = static_cast<uint32_t>(args.max_new_tokens - generated.size());
                    const uint32_t count = std::min(args.decode_flight, remaining);
                    const uint32_t first_slot = static_cast<uint32_t>(generated.size());
                    const auto t_decode = std::chrono::steady_clock::now();
                    for (uint32_t j = 0; j < count && exit_code == 0; j++) {
                        st = engine_ptr->decode_device_greedy(
                            stream, need_host_input ? &pending : nullptr, first_slot + j);
                        if (!st.ok()) {
                            std::fprintf(stderr, "\nexample_qwen: in-flight decode failed: %s (%s)\n",
                                         st.message.c_str(), st.detail.c_str());
                            exit_code = 2;
                            break;
                        }
                        need_host_input = false;
                    }
                    if (exit_code != 0) { break; }
                    flight_ids.resize(count);
                    st = engine_ptr->download_device_greedy(stream, first_slot, flight_ids);
                    decode_total_ms += ms_since(t_decode);
                    decode_steps += count;
                    if (!st.ok()) {
                        std::fprintf(stderr, "\nexample_qwen: in-flight download failed: %s (%s)\n",
                                     st.message.c_str(), st.detail.c_str());
                        exit_code = 2;
                        break;
                    }
                    for (const auto id : flight_ids) {
                        if (!emit_one(id)) { break; }
                    }
                }
            }
        }
        while (exit_code == 0 && !use_flight) {
            if (!have_pending) {
                const auto t_select = std::chrono::steady_clock::now();
                auto next = qwen_sample(logits, sample_options,
                                        static_cast<uint64_t>(generated.size()));
                argmax_total_ms += ms_since(t_select);
                if (!next.ok()) {
                    std::fprintf(stderr, "example_qwen: token selection failed: %s\n", next.status.message.c_str());
                    exit_code = 2;
                    break;
                }
                pending = *next.value;
                have_pending = true;
            }
            if (args.mtp && mtp->can_speculate()) {
                const auto t_decode = std::chrono::steady_clock::now();
                QwenMtpRoundResult round;
                st = mtp->round(stream, pending, sample_options, round);
                const double decode_ms = ms_since(t_decode);
                decode_total_ms += decode_ms;
                decode_steps += 1;
                if (st.ok() && options.benchmark) {
                    std::printf("mtp round=%llu input-token=%u wall=%.3f ms accepted=%u\n",
                                static_cast<unsigned long long>(mtp->rounds()), pending, decode_ms, round.accepted);
                    st = emit_benchmark(args, *engine_ptr, benchmark_iteration++, csv_started);
                }
                if (!st.ok()) {
                    std::fprintf(stderr, "\nexample_qwen: MTP round failed: %s (%s)\n",
                                 st.message.c_str(), st.detail.c_str());
                    exit_code = 2;
                    break;
                }
                bool continue_generation = true;
                for (const auto id : round.emitted) {
                    if (!emit_one(id)) { continue_generation = false; break; }
                }
                if (!continue_generation) { break; }
                // The bonus is pending, not emitted or consumed. The next round
                // verifies it first; near context end the plain path emits it.
                logits = std::move(round.next_logits);
                pending = round.next_pending;
                have_pending = true;
                continue;
            }

            // Plain target-only step (also the MTP context-end fallback).
            const uint32_t id = pending;
            if (!emit_one(id)) { break; }
            const auto t_decode = std::chrono::steady_clock::now();
            st = engine_ptr->decode(stream, id, logits);
            const double decode_ms = ms_since(t_decode);
            decode_total_ms += decode_ms;
            decode_steps += 1;
            if (st.ok() && options.benchmark) {
                std::printf("decode step=%u input-token=%u wall=%.3f ms\n", decode_steps, id, decode_ms);
                st = emit_benchmark(args, *engine_ptr, benchmark_iteration++, csv_started);
            }
            if (!st.ok()) {
                std::fprintf(stderr, "\nexample_qwen: decode failed: %s (%s)\n",
                             st.message.c_str(), st.detail.c_str());
                exit_code = 2;
                break;
            }
            have_pending = false;
        }
        if (exit_code == 0) {
            auto tail = detok.finish();
            if (tail.ok() && !(*tail.value).empty()) { std::fputs((*tail.value).c_str(), stdout); }
            std::printf("\n");
        }
        // --- report ------------------------------------------------------------------
        std::printf("generated ids (%zu):", generated.size());
        for (auto id : generated) { std::printf(" %u", id); }
        std::printf("\n");
        auto full_text = tok.decode(generated, true, true);
        if (full_text.ok()) { std::printf("generated text: %s\n", (*full_text.value).c_str()); }
        std::printf("stop: %s\n", eos_hit ? "EOS (<|im_end|>)" : "max-new-tokens budget");
        std::printf("timing: host-load %.1f ms | weight-upload %.1f ms | engine-compile %.1f ms | "
                    "TTFT(prefill %zu tokens) %.1f ms",
                    host_load_ms, upload_ms, compile_ms, prompt_ids.size(), ttft_ms);
        if (decode_steps > 0) {
            std::printf(args.mtp ? " | decode %.2f ms/round (%u target/draft rounds)"
                                 : " | decode %.2f ms/token (%u steps)",
                        decode_total_ms / decode_steps, decode_steps);
        }
        std::printf("\n");
        if (args.mtp) {
            std::printf("mtp final: rounds=%llu accepted=%llu/%llu accept-rate=%.3f\n",
                        static_cast<unsigned long long>(mtp->rounds()),
                        static_cast<unsigned long long>(mtp->accepted_total()),
                        static_cast<unsigned long long>(mtp->drafted_total()),
                        mtp->accept_rate());
        }
        std::printf("host: token-selection %.3f ms total | detokenize %.3f ms total%s\n",
                    argmax_total_ms, detok_total_ms,
                    options.benchmark ? " | per-kernel samples include sync overhead" : "");
        {
            const auto info = engine_ptr->info();
            std::printf("engine final: position=%u shaders=%llu\n", info.position,
                        static_cast<unsigned long long>(info.compiled_shader_count));
        }
        // Teardown: complete GPU work, then let the allocator-matched engine
        // handle leave scope before weights, stream and device are released.
        stream.synchronize();
    }
    stream.synchronize();
    stream = compute::Stream{};
    device = compute::Device{};
    return exit_code;
}

[[nodiscard]] int run_cli(int argc, const char *const *argv) {
    argv0_holder() = argv[0];
    CliArgs args;
    if (!parse_args(argc, argv, args)) { return 1; }
    if (args.help) {
        print_usage();
        return 0;
    }
    if (args.list_versions) {
        print_versions();
        return 0;
    }
    if (args.self_test) {
        return run_self_test();
    }
    if (args.mtp_test) {
        return run_mtp_test(args);
    }
    return run_generation(args);
}

}// namespace

#ifdef _WIN32
int wmain(int argc, wchar_t **wargv) {
    luisa::vector<luisa::string> utf8;
    luisa::vector<const char *> argv;
    utf8.reserve(static_cast<size_t>(argc));
    argv.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; i++) {
        luisa::string arg;
        const int length = static_cast<int>(::wcslen(wargv[i]));
        if (length > 0) {
            const int needed = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wargv[i],
                                                     length, nullptr, 0, nullptr, nullptr);
            if (needed <= 0) {
                std::fprintf(stderr, "example_qwen: argument %d is not convertible to UTF-8\n", i);
                return 1;
            }
            arg.resize(static_cast<size_t>(needed));
            ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wargv[i], length,
                                  arg.data(), needed, nullptr, nullptr);
        }
        utf8.emplace_back(std::move(arg));
    }
    for (const auto &a : utf8) { argv.emplace_back(a.c_str()); }
    return run_cli(argc, argv.data());
}
#else
int main(int argc, char **argv) {
    return run_cli(argc, argv);
}
#endif
