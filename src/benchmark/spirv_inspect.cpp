#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>
#include <spirv/unified1/spirv.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

struct SpvStats {
    size_t words{};
    size_t insts{};
    size_t funcs{};
    size_t calls{};
    size_t blocks{};
    size_t phis{};
    size_t vars_fn{};
    size_t vars_priv{};
    size_t loads{};
    size_t stores{};
    size_t access_chains{};
    size_t composites{};
    size_t selects{};
    size_t loops{};
    size_t sel_merges{};
    size_t fmas{};
    size_t ext_insts{};
    size_t atomics{};
    size_t ray_queries{};
    bool valid{true};
};

[[nodiscard]] bool read_spv(const std::filesystem::path &path, std::vector<uint32_t> &words) {
    std::ifstream file{path, std::ios::binary | std::ios::ate};
    if (!file) return false;
    auto sz = static_cast<std::streamoff>(file.tellg());
    if (sz <= 0 || sz % 4 != 0) return false;
    words.resize(static_cast<size_t>(sz) / 4);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char *>(words.data()), sz);
    return static_cast<bool>(file);
}

[[nodiscard]] SpvStats analyze_spv(const std::vector<uint32_t> &words, spvtools::SpirvTools &tools) {
    SpvStats s{};
    s.words = words.size();
    s.valid = tools.Validate(words.data(), words.size());

    // Header is 5 words: magic, version, generator, bound, schema
    size_t idx = 5u;
    while (idx < words.size()) {
        uint32_t word0 = words[idx];
        uint16_t wc = static_cast<uint16_t>(word0 >> 16u);
        uint16_t op = static_cast<uint16_t>(word0 & 0xffffu);
        if (wc == 0u || idx + wc > words.size()) break;
        ++s.insts;
        switch (static_cast<SpvOp>(op)) {
            case SpvOpFunction: ++s.funcs; break;
            case SpvOpFunctionCall: ++s.calls; break;
            case SpvOpLabel: ++s.blocks; break;
            case SpvOpPhi: ++s.phis; break;
            case SpvOpVariable:
                if (wc >= 4u) {
                    auto sc = static_cast<SpvStorageClass>(words[idx + 3u]);
                    if (sc == SpvStorageClassFunction) ++s.vars_fn;
                    else if (sc == SpvStorageClassPrivate) ++s.vars_priv;
                }
                break;
            case SpvOpLoad: ++s.loads; break;
            case SpvOpStore: ++s.stores; break;
            case SpvOpAccessChain:
            case SpvOpInBoundsAccessChain: ++s.access_chains; break;
            case SpvOpCompositeExtract:
            case SpvOpCompositeInsert:
            case SpvOpCompositeConstruct: ++s.composites; break;
            case SpvOpSelect: ++s.selects; break;
            case SpvOpLoopMerge: ++s.loops; break;
            case SpvOpSelectionMerge: ++s.sel_merges; break;
            case SpvOpFmaKHR: ++s.fmas; break;
            case SpvOpExtInst: ++s.ext_insts; break;
            case SpvOpAtomicLoad:
            case SpvOpAtomicStore:
            case SpvOpAtomicExchange:
            case SpvOpAtomicCompareExchange:
            case SpvOpAtomicIIncrement:
            case SpvOpAtomicIDecrement:
            case SpvOpAtomicIAdd:
            case SpvOpAtomicISub:
            case SpvOpAtomicSMin:
            case SpvOpAtomicUMin:
            case SpvOpAtomicSMax:
            case SpvOpAtomicUMax:
            case SpvOpAtomicAnd:
            case SpvOpAtomicOr:
            case SpvOpAtomicXor:
            case SpvOpAtomicFAddEXT:
            case SpvOpAtomicFMinEXT:
            case SpvOpAtomicFMaxEXT: ++s.atomics; break;
            case SpvOpRayQueryInitializeKHR:
            case SpvOpRayQueryProceedKHR: ++s.ray_queries; break;
            default: break;
        }
        idx += wc;
    }
    return s;
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: spirv_inspect [--disasm-dir <dir>] [--csv] <file.spv> [...]\n");
        return 2;
    }
    std::filesystem::path disasm_dir;
    bool csv = false;
    bool test_sroa = false;
    std::vector<std::filesystem::path> inputs;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--disasm-dir" && i + 1 < argc) {
            disasm_dir = argv[++i];
        } else if (arg == "--csv") {
            csv = true;
        } else if (arg == "--test-sroa") {
            test_sroa = true;
        } else {
            inputs.emplace_back(arg);
        }
    }
    if (test_sroa && !inputs.empty()) {
        std::vector<uint32_t> words;
        if (read_spv(inputs[0], words)) {
            spvtools::Optimizer opt{SPV_ENV_VULKAN_1_3};
            opt.SetMessageConsumer([](spv_message_level_t, const char *, const spv_position_t &, const char *msg) {
                std::fprintf(stderr, "[opt] %s\n", msg ? msg : "");
            });
            opt.RegisterPass(spvtools::CreateScalarReplacementPass(0));
            opt.RegisterPass(spvtools::CreateLocalMultiStoreElimPass());
            opt.RegisterPass(spvtools::CreateAggressiveDCEPass());
            std::vector<uint32_t> out;
            bool ok = opt.Run(words.data(), words.size(), &out);
            std::fprintf(stderr, "SROA run ok=%d in_words=%zu out_words=%zu\n", ok ? 1 : 0, words.size(), out.size());
        }
        return 0;
    }
    if (!disasm_dir.empty()) {
        std::filesystem::create_directories(disasm_dir);
    }

    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_3};
    tools.SetMessageConsumer([](spv_message_level_t, const char *src, const spv_position_t &pos, const char *msg) {
        std::fprintf(stderr, "[spirv-val] %s:%zu:%zu: %s\n", src ? src : "", pos.line, pos.column, msg ? msg : "");
    });

    if (csv) {
        std::printf("shader,valid,words,insts,funcs,calls,blocks,phis,vars_fn,vars_priv,loads,stores,access_chains,composites,selects,loops,sel_merges,ext_insts\n");
    } else {
        std::printf("%-34s %5s %6s %6s %4s %5s %5s %5s %5s %5s %6s %6s %6s %6s %4s %4s\n",
                    "shader", "valid", "words", "insts", "fn", "call", "blk", "phi", "varF", "varP", "load", "store", "ac", "comp", "sel", "loop");
    }

    size_t total_words = 0, total_insts = 0, total_loads = 0, total_stores = 0, total_phis = 0, total_calls = 0;
    for (const auto &p : inputs) {
        std::vector<uint32_t> words;
        if (!read_spv(p, words)) {
            std::fprintf(stderr, "Failed to read %s\n", p.string().c_str());
            continue;
        }
        auto s = analyze_spv(words, tools);
        auto name = p.stem().string();
        if (name.rfind("spv_output_", 0) == 0) name = name.substr(11);
        total_words += s.words;
        total_insts += s.insts;
        total_loads += s.loads;
        total_stores += s.stores;
        total_phis += s.phis;
        total_calls += s.calls;

        if (csv) {
            std::printf("%s,%d,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu\n",
                        name.c_str(), s.valid ? 1 : 0, s.words, s.insts, s.funcs, s.calls,
                        s.blocks, s.phis, s.vars_fn, s.vars_priv, s.loads, s.stores,
                        s.access_chains, s.composites, s.selects, s.loops, s.sel_merges, s.ext_insts);
        } else {
            std::printf("%-34s %5s %6zu %6zu %4zu %5zu %5zu %5zu %5zu %5zu %6zu %6zu %6zu %6zu %4zu %4zu\n",
                        name.c_str(), s.valid ? "ok" : "FAIL", s.words, s.insts, s.funcs, s.calls,
                        s.blocks, s.phis, s.vars_fn, s.vars_priv, s.loads, s.stores,
                        s.access_chains, s.composites, s.selects, s.loops);
        }

        if (!disasm_dir.empty()) {
            std::string disasm;
            if (tools.Disassemble(words, &disasm,
                                  SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES |
                                      SPV_BINARY_TO_TEXT_OPTION_INDENT |
                                      SPV_BINARY_TO_TEXT_OPTION_COMMENT)) {
                auto out_path = disasm_dir / (name + ".spvasm");
                std::ofstream out{out_path, std::ios::binary};
                out.write(disasm.data(), static_cast<std::streamsize>(disasm.size()));
            }
        }
    }
    if (!csv && inputs.size() > 1u) {
        std::printf("%-34s %5s %6zu %6zu %4s %5zu %5s %5zu %5s %5s %6zu %6zu\n",
                    "TOTAL", "", total_words, total_insts, "", total_calls, "", total_phis, "", "", total_loads, total_stores);
    }
    return 0;
}
