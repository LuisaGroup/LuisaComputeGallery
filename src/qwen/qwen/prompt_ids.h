#pragma once
// Deterministic synthetic prompt token ids (host-only).
//
// opt_plan/09-plan-verification-and-extensions.md N4.1: a prompt sweep with
// P >= 8192 cannot be expressed as `--tokens <id,id,...>` on Windows, because a
// whole CreateProcess command line is capped at 32767 characters and an id list
// costs ~5-6 bytes per token. example_qwen therefore accepts
// `--synthetic-prompt-len <N>` (and `--tokens-file <path>`); both must produce
// the same ids as `scripts/benchmark_qwen.py`'s `synthetic_prompt_ids()`, so the
// CLI and the harness can never disagree about what a "prompt of length N" is.
//
// The generator is one 64-bit LCG with explicit integer arithmetic -- no floats,
// no hashing, no dependence on a standard-library random engine -- so the same
// length yields byte-identical ids on every machine, every compiler and every
// Python version. scripts/benchmark_qwen.py documents the same rule in its
// module docstring, in `synthetic_prompt_ids()` and in --prompt-lengths' help,
// and scripts/test_benchmark_qwen.py checks the constants against this file.
//
//   state = (PROMPT_SEED ^ (length * PROMPT_LENGTH_MIX)) & 2**64-1
//   repeat `length` times:
//       state = (state * PROMPT_LCG_MUL + PROMPT_LCG_INC) & 2**64-1
//       id    = PROMPT_ID_MIN + ((state >> 32) % (PROMPT_ID_MAX - PROMPT_ID_MIN))
//
// Ids stay inside [PROMPT_ID_MIN, PROMPT_ID_MAX) = [1000, 200000), which avoids
// the low control ids and the high special-token block of Qwen3.5-4B (vocab
// 248320); the CLI still validates every id against the loaded vocab and fails
// closed, so a checkpoint with a different vocabulary cannot benchmark silently.

#include <cstdint>

#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::qwen {

// Rule constants. They are part of the CLI/harness contract: change one and
// every stored synthetic arm becomes a different prompt.
inline constexpr uint64_t kSyntheticPromptSeed = 0x0C0FFEE5EEDBEEF01ull;
inline constexpr uint64_t kSyntheticPromptLengthMix = 0xD1B54A32D192ED03ull;
inline constexpr uint64_t kSyntheticPromptLcgMul = 6364136223846793005ull;
inline constexpr uint64_t kSyntheticPromptLcgInc = 1442695040888963407ull;
inline constexpr uint32_t kSyntheticPromptIdMin = 1000u;
inline constexpr uint32_t kSyntheticPromptIdMax = 200000u;
// Qwen3.5-4B vocabulary (docs/examples/example_qwen.md). The synthetic ID range
// must stay BELOW it: kSyntheticPromptIdMax (200000) <= kSyntheticPromptVocabSize
// (248320). This constant is a sanity bound only -- the CLI validates every id
// against the vocabulary of the checkpoint it actually loaded, so a different
// vocabulary fails closed there instead of silently benchmarking.
inline constexpr uint32_t kSyntheticPromptVocabSize = 248320u;

// Exactly `length` deterministic ids. `length == 0` yields an empty vector; the
// caller reports that as a bad argument (an empty prompt is never useful).
[[nodiscard]] inline luisa::vector<uint32_t>
qwen_synthetic_prompt_ids(uint32_t length) {
    static_assert(kSyntheticPromptIdMax > kSyntheticPromptIdMin);
    luisa::vector<uint32_t> ids;
    if (length == 0u) { return ids; }
    constexpr uint64_t span =
        static_cast<uint64_t>(kSyntheticPromptIdMax) - kSyntheticPromptIdMin;
    const uint64_t mask = 0xFFFFFFFFFFFFFFFFull;
    uint64_t state = (kSyntheticPromptSeed ^
                      ((static_cast<uint64_t>(length) * kSyntheticPromptLengthMix) & mask)) &
                     mask;
    ids.reserve(length);
    for (uint32_t i = 0u; i < length; ++i) {
        state = state * kSyntheticPromptLcgMul + kSyntheticPromptLcgInc;
        ids.emplace_back(kSyntheticPromptIdMin +
                         static_cast<uint32_t>((state >> 32u) % span));
    }
    return ids;
}

// Parse a `--tokens-file` payload: decimal token ids separated by commas,
// whitespace (spaces, tabs, CR, LF) or any combination. Blank input is an error,
// because an empty prompt is never useful and a silently empty list would turn
// into a generation with no prefill.
//
// Returns false and fills `error` with the 1-based line and the offending text
// when a chunk is not a decimal id; `ids` is cleared on failure so a caller can
// never use a half-parsed prompt. Vocabulary and context bounds are NOT checked
// here: they depend on the loaded GGUF and the engine options, and example_qwen
// reports them with the token index.
[[nodiscard]] inline bool qwen_parse_prompt_ids(luisa::string_view text,
                                                luisa::vector<uint32_t> &ids,
                                                luisa::string &error) {
    ids.clear();
    uint64_t line = 1u;
    size_t pos = 0u;
    while (pos <= text.size()) {
        // Skip separators (and count newlines so the diagnostic has a line).
        while (pos < text.size() &&
               (text[pos] == ',' || text[pos] == ' ' || text[pos] == '\t' ||
                text[pos] == '\r' || text[pos] == '\n')) {
            if (text[pos] == '\n') { line++; }
            pos++;
        }
        if (pos >= text.size()) { break; }
        const auto begin = pos;
        while (pos < text.size() && text[pos] != ',' && text[pos] != ' ' &&
               text[pos] != '\t' && text[pos] != '\r' && text[pos] != '\n') {
            pos++;
        }
        const auto item = text.substr(begin, pos - begin);
        uint32_t id = 0u;
        bool ok = !item.empty();
        for (auto c : item) {
            if (c < '0' || c > '9') {
                ok = false;
                break;
            }
            const auto digit = static_cast<uint32_t>(c - '0');
            if (id > (0xFFFFFFFFu - digit) / 10u) {
                ok = false;
                break;
            }
            id = id * 10u + digit;
        }
        if (!ok) {
            error = luisa::format("line {}: '{}' is not a decimal token id",
                                  line, luisa::string{item});
            ids.clear();
            return false;
        }
        ids.emplace_back(id);
    }
    if (ids.empty()) {
        error = luisa::string{"no token ids found"};
        return false;
    }
    return true;
}

}// namespace luisa::qwen
