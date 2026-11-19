#pragma once
// Qwen3.5-4B tokenizer -- GGUF metadata -> exact GPT-2 byte-level BPE.
// Host-only: no DSL/runtime device includes.
//
// The GGUF stores the whole tokenizer (tokenizer.ggml.tokens / token_type /
// merges / eos_token_id / padding_token_id). This loader VALIDATES the tables
// (count agreement, ID range, duplicate pieces, GPT-2 byte alphabet, the 256
// single-byte base tokens, special-token classification, merge well-formedness
// and UTF-8), then constructs the exact HF-style tokenizer.json document that
// src/ocr/ocr/tokenizer.h supports and REUSES that implementation for
// encode/decode/detokenization (ocr::Tokenizer / ocr::IncrementalDetokenizer).
// Nothing is approximated silently:
//
// * The reference checkpoint declares tokenizer.ggml.pre == "qwen35". Its
//   pre-tokenizer is NOT the GPT-2 pattern the OCR loader implements; the
//   exact regex is
//     (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//     |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
//     | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//   (llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35). QwenTokenizer therefore performs
//   its OWN exact pre-tokenization (unicode_data_qwen35.h tables + a
//   hand-written matcher mirroring llama.cpp unicode_regex_split_custom_qwen35)
//   and reuses the OCR BPE merge engine only on the resulting pre-token
//   pieces, where the two implementations provably agree.
// * Added/special-token splitting (longest match first, never split),
//   byte-level encoding, merge ranking, byte decoding and the incremental
//   UTF-8-safe detokenizer are exactly the OCR code paths.
// * Any other tokenizer.ggml.model / pre identity, malformed tables or
//   unsupported special-token layout is a recoverable failure.
//
// Qwen3.5 has no BOS token; encode(add_special_tokens=true) adds nothing
// (post-processor ByteLevel). EOS is <|im_end|> (GGUF eos_token_id), NOT
// <|endoftext|>; stop handling must use eos_id() from this class.

#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

#include "ocr/detokenizer.h"
#include "ocr/tokenizer.h"
#include "qwen/common.h"
#include "qwen/gguf.h"

namespace luisa::qwen {

struct QwenTokenizerInfo {
    uint64_t vocab_size{0};
    uint64_t merge_count{0};
    uint64_t special_token_count{0};// CONTROL + USER_DEFINED entries
    uint64_t unused_pad_count{0};   // UNUSED "[PADnnnn]" entries
    uint32_t eos_id{0};             // GGUF tokenizer.ggml.eos_token_id (<|im_end|>)
    uint32_t pad_id{0};             // GGUF tokenizer.ggml.padding_token_id
    luisa::string model;            // tokenizer.ggml.model ("gpt2")
    luisa::string pre;              // tokenizer.ggml.pre ("qwen35")
    luisa::vector<Diagnostic> diagnostics;
};

class QwenTokenizer;

// Deleter that frees through Luisa's allocator (the same one every container
// member uses). eastl::default_delete would call eastl::GetDefaultAllocator(),
// whose EASTL_MIMALLOC_ENABLED / EASTL_CUSTOM_MALLOC_ENABLED configuration is
// NOT reliably propagated to gallery translation units -- a mismatch frees
// mimalloc/aligned memory with std::free and corrupts the heap. Allocating with
// luisa::allocate_with_allocator and freeing with this deleter keeps both sides
// on the same allocator regardless of the EASTL define propagation.
struct QwenTokenizerDeleter {
    void operator()(QwenTokenizer *p) const noexcept {
        luisa::delete_with_allocator(p);
    }
};

class QwenTokenizer {
public:
    using Ptr = luisa::unique_ptr<QwenTokenizer, QwenTokenizerDeleter>;
    // Load + validate from an open GGUF reader. `expected_vocab` (when > 0,
    // e.g. Qwen35Geometry::vocab) must equal the token-table length.
    [[nodiscard]] static Result<Ptr> load(
        const GgufReader &reader, uint64_t expected_vocab = 0);

    ~QwenTokenizer() noexcept;
    QwenTokenizer(const QwenTokenizer &) = delete;
    QwenTokenizer &operator=(const QwenTokenizer &) = delete;

    [[nodiscard]] const QwenTokenizerInfo &info() const noexcept { return _info; }
    [[nodiscard]] uint64_t vocab_size() const noexcept { return _info.vocab_size; }
    [[nodiscard]] uint32_t eos_id() const noexcept { return _info.eos_id; }
    [[nodiscard]] uint32_t pad_id() const noexcept { return _info.pad_id; }
    [[nodiscard]] luisa::optional<uint32_t> token_to_id(luisa::string_view piece) const noexcept;
    [[nodiscard]] luisa::optional<luisa::string> id_to_token(uint32_t id) const;
    [[nodiscard]] bool is_special(uint32_t id) const noexcept;

    // Encode UTF-8 text to exact token IDs. Invalid UTF-8 is rejected (no
    // lossy replacement). Special/added tokens in the text are matched
    // longest-first and never split. Qwen3.5 adds no BOS/EOS automatically:
    // `add_special_tokens` is accepted for interface parity and adds nothing.
    [[nodiscard]] Result<luisa::vector<uint32_t>> encode(
        luisa::string_view text, bool add_special_tokens = false) const;

    // Decode token IDs to text (GPT-2 byte decoding). Special tokens are
    // skipped when `skip_special_tokens`; unknown IDs are rejected. The
    // result must be complete UTF-8 unless `allow_partial_utf8` is set.
    [[nodiscard]] Result<luisa::string> decode(
        luisa::span<const uint32_t> ids, bool skip_special_tokens = true,
        bool allow_partial_utf8 = false) const;

    // Incremental detokenizer (reuses ocr::IncrementalDetokenizer): buffers
    // incomplete multi-byte UTF-8 across token boundaries and stops on the
    // GGUF EOS id. The tokenizer must outlive the returned stream.
    [[nodiscard]] ocr::IncrementalDetokenizer make_detokenizer(bool skip_special_tokens = true) const;

    // Single-user-turn Qwen chat prompt (the documented non-thinking default
    // of the checkpoint's Jinja template, reduced to one user message):
    //   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
    // This is a FIXED wrapper, not a template interpreter. The special
    // markers are encoded as their exact single token IDs. Invalid UTF-8 or
    // a prompt containing special-token markers is rejected (the wrapper
    // must not be spoofable by prompt text).
    [[nodiscard]] Result<luisa::vector<uint32_t>> chat_prompt_ids(luisa::string_view user_prompt) const;
    [[nodiscard]] Result<luisa::string> chat_prompt_text(luisa::string_view user_prompt) const;

    // The constructed HF tokenizer.json document (diagnostics / tooling).
    [[nodiscard]] const luisa::string &hf_tokenizer_json() const noexcept { return _json; }

    // Exact Qwen3.5 pre-tokenization of one NORMAL (special-token-free) text
    // chunk into byte spans. Exposed for tests; mirrors llama.cpp
    // unicode_regex_split_custom_qwen35 over the unicode_data_qwen35 tables.
    struct ByteSpan {
        size_t begin;
        size_t end;
    };
    [[nodiscard]] static luisa::vector<ByteSpan> qwen35_pretokenize(luisa::string_view text);

    // Device-free self-test: synthetic GGUF tokenizer fixture (small exact
    // vocab/merges/specials), encode/decode/detokenize round-trips including
    // multi-byte UTF-8 split across tokens, rejection paths, the Qwen3.5
    // pre-tokenizer vs an independent reference over ASCII/digits/CJK/
    // whitespace cases, and the chat wrapper. `report` gets one line per
    // check. No device, no model file needed.
    [[nodiscard]] static Status self_test(luisa::vector<luisa::string> *report = nullptr);


private:
    QwenTokenizer() noexcept = default;

    QwenTokenizerInfo _info;
    luisa::string _json;
    luisa::unique_ptr<ocr::Tokenizer, ocr::TokenizerDeleter> _base;
    // Exact vocab tables (indexed by id) for the qwen35 encode path.
    luisa::vector<luisa::string> _pieces;
    luisa::unordered_map<luisa::string, uint32_t> _piece_to_id;
    // Merge ranks keyed "a\0b" (same convention as the OCR BPE engine).
    luisa::unordered_map<luisa::string, uint32_t> _ranks;
    // Single-byte base token ids, indexed by raw byte value.
    uint32_t _byte_tokens[256]{};
    struct SpecialEntry {
        uint32_t id;
        luisa::string content;
    };
    luisa::vector<SpecialEntry> _specials;// longest-first, stable by id
    luisa::vector<bool> _is_special;      // indexed by id (CONTROL/USER_DEFINED)
};

}// namespace luisa::qwen
