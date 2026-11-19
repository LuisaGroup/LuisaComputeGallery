// Qwen3.5-4B tokenizer -- implementation. See tokenizer.h for the contract.
//
// Reuse map (no silent approximation):
// * ocr::Tokenizer::parse validates and owns the constructed HF tokenizer.json
//   (byte codec, vocab, merge ranks, added tokens, decode paths).
// * ocr::IncrementalDetokenizer drives streaming decode (UTF-8 hold-back).
// * The Qwen3.5 pre-tokenizer is implemented HERE exactly (the OCR GPT-2
//   matcher is a different regex); BPE merging reuses the OCR rank convention
//   and is cross-checked against ocr::Tokenizer::encode on single pre-tokens.
#include "qwen/tokenizer.h"




#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>

#include <yyjson.h>

#include "ocr/status.h"
#include "qwen/unicode_data_qwen35.h"

namespace luisa::qwen {

using ocr::ocr_is_valid_utf8;
using ocr::ocr_utf8_sequence_length;

namespace {

namespace ud = luisa::qwen::unicode_data;


// std::to_string returns std::string; the qwen contract uses luisa::string.
[[nodiscard]] luisa::string num_str(uint64_t v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(v));
    return luisa::string{buf};
}

[[nodiscard]] Status invalid(luisa::string msg, luisa::string detail = {}) {
    return Status::error(StatusCode::invalid_argument, std::move(msg), std::move(detail));
}

// ---------------------------------------------------------------------------
// UTF-8 / GPT-2 byte codec helpers (same conventions as the OCR tokenizer).
// ---------------------------------------------------------------------------

// Decode one code point at `pos`; returns the sequence length or 0 on error.
[[nodiscard]] uint32_t decode_cp(luisa::string_view s, size_t pos, uint32_t &cp) noexcept {
    if (pos >= s.size()) { return 0; }
    const auto lead = static_cast<uint8_t>(s[pos]);
    const auto len = ocr_utf8_sequence_length(lead);
    if (len == 0 || pos + len > s.size()) { return 0; }
    uint32_t value = 0;
    switch (len) {
        case 1: value = lead; break;
        case 2: value = lead & 0x1fu; break;
        case 3: value = lead & 0x0fu; break;
        case 4: value = lead & 0x07u; break;
        default: return 0;
    }
    for (uint32_t i = 1; i < len; i++) {
        const auto b = static_cast<uint8_t>(s[pos + i]);
        if ((b & 0xc0u) != 0x80u) { return 0; }
        value = (value << 6u) | (b & 0x3fu);
    }
    // Reject overlong forms, surrogates and out-of-range code points.
    static constexpr uint32_t min_value[5] = {0, 0, 0x80, 0x800, 0x10000};
    if (value < min_value[len] || value > 0x10ffffu ||
        (value >= 0xd800u && value <= 0xdfffu)) {
        return 0;
    }
    cp = value;
    return len;
}

void append_cp(luisa::string &out, uint32_t cp) {
    if (cp < 0x80u) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800u) {
        out.push_back(static_cast<char>(0xc0u | (cp >> 6u)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3fu)));
    } else if (cp < 0x10000u) {
        out.push_back(static_cast<char>(0xe0u | (cp >> 12u)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6u) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3fu)));
    } else {
        out.push_back(static_cast<char>(0xf0u | (cp >> 18u)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12u) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6u) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3fu)));
    }
}

// GPT-2 bytes_to_unicode: byte -> vocab code point (identical to the OCR map).
[[nodiscard]] uint32_t byte_to_cp(uint8_t b) noexcept {
    const bool printable = (b >= 0x21u && b <= 0x7eu) || (b >= 0xa1u && b <= 0xacu) ||
                           (b >= 0xaeu && b <= 0xffu);
    if (printable) { return b; }
    uint32_t offset = 0;
    for (uint32_t k = 0; k < b; k++) {
        const auto kb = static_cast<uint8_t>(k);
        const bool kp = (kb >= 0x21u && kb <= 0x7eu) || (kb >= 0xa1u && kb <= 0xacu) ||
                        (kb >= 0xaeu && kb <= 0xffu);
        if (!kp) { offset += 1; }
    }
    return 256u + offset;
}

// Inverse: vocab code point -> byte, or -1 when outside the GPT-2 alphabet.
[[nodiscard]] int cp_to_byte(uint32_t cp) noexcept {
    for (uint32_t b = 0; b < 256u; b++) {
        if (byte_to_cp(static_cast<uint8_t>(b)) == cp) { return static_cast<int>(b); }
    }
    return -1;
}

// Piece text -> raw bytes through the GPT-2 alphabet. Fails on any code point
// outside the alphabet (a malformed byte-level vocab).
[[nodiscard]] bool piece_to_bytes(luisa::string_view piece, luisa::string &out) {
    size_t pos = 0;
    while (pos < piece.size()) {
        uint32_t cp = 0;
        const auto len = decode_cp(piece, pos, cp);
        if (len == 0) { return false; }
        const auto b = cp_to_byte(cp);
        if (b < 0) { return false; }
        out.push_back(static_cast<char>(b));
        pos += len;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Exact Unicode classification (llama.cpp unicode_cpt_flags semantics).
// ---------------------------------------------------------------------------

[[nodiscard]] uint32_t cp_flags(uint32_t cp) noexcept {
    if (cp > 0x10ffffu) { return ud::kFlagUndefined; }
    // Binary search: last range with start <= cp.
    size_t lo = 0, hi = std::size(ud::kUnicodeFlagRanges) - 1;
    while (lo < hi) {
        const size_t mid = (lo + hi + 1) / 2;
        if (ud::kUnicodeFlagRanges[mid].start <= cp) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return ud::kUnicodeFlagRanges[lo].flags;
}

[[nodiscard]] bool cp_is(uint32_t cp, uint32_t mask) noexcept {
    return (cp_flags(cp) & mask) != 0u;
}

[[nodiscard]] uint32_t cp_tolower(uint32_t cp) noexcept {
    size_t lo = 0, hi = std::size(ud::kUnicodeToLower);
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (ud::kUnicodeToLower[mid].from < cp) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < std::size(ud::kUnicodeToLower) && ud::kUnicodeToLower[lo].from == cp) {
        return ud::kUnicodeToLower[lo].to;
    }
    return cp;
}

}// namespace

// ---------------------------------------------------------------------------
// Exact Qwen3.5 pre-tokenizer.
// Mirrors llama.cpp unicode_regex_split_custom_qwen35 (the GGUF
// tokenizer.ggml.pre == "qwen35" path) over code points with byte offsets.
// Regex (first matching alternative wins, leftmost):
//   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//   |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//   |\p{N}
//   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//   |\s*[\r\n]+
//   |\s+(?!\S)
//   |\s+
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::vector<QwenTokenizer::ByteSpan>
QwenTokenizer::qwen35_pretokenize(luisa::string_view text) {
    struct Cp {
        uint32_t value;
        size_t begin;
        size_t end;
    };
    luisa::vector<Cp> cps;
    {
        size_t pos = 0;
        while (pos < text.size()) {
            uint32_t cp = 0;
            const auto len = decode_cp(text, pos, cp);
            if (len == 0) {
                // Callers validate UTF-8 first; keep a stray byte as its own
                // symbol so the span cover is still total (defensive).
                cps.emplace_back(Cp{static_cast<uint8_t>(text[pos]), pos, pos + 1});
                pos += 1;
            } else {
                cps.emplace_back(Cp{cp, pos, pos + len});
                pos += len;
            }
        }
    }
    const size_t n = cps.size();
    static constexpr uint32_t kOutOfRange = 0xffffffffu;
    const auto cpt = [&](size_t i) -> uint32_t { return i < n ? cps[i].value : kOutOfRange; };
    const auto flags = [&](size_t i) -> uint32_t {
        return i < n ? cp_flags(cps[i].value) : 0u;// out-of-range: no flags (llama.cpp {})
    };
    const auto is_l = [&](size_t i) { return (flags(i) & ud::kFlagLetter) != 0u; };
    const auto is_m = [&](size_t i) { return (flags(i) & ud::kFlagAccentMark) != 0u; };
    const auto is_n = [&](size_t i) { return (flags(i) & ud::kFlagNumber) != 0u; };
    const auto is_s = [&](size_t i) { return (flags(i) & ud::kFlagWhitespace) != 0u; };

    luisa::vector<ByteSpan> spans;
    size_t prev_end = 0;// code-point index
    const auto emit = [&](size_t end) {
        if (end > prev_end) {
            spans.emplace_back(ByteSpan{cps[prev_end].begin, cps[end - 1].end});
        }
        prev_end = end;
    };
    size_t pos = 0;
    while (pos < n) {
        const auto c = cpt(pos);
        const auto f = flags(pos);
        // (?i:'s|'t|'re|'ve|'m|'ll|'d) -- case-insensitive contractions.
        if (c == '\'' && pos + 1 < n) {
            const auto c1 = cp_tolower(cpt(pos + 1));
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                pos += 2;
                emit(pos);
                continue;
            }
            if (pos + 2 < n) {
                const auto c2 = cp_tolower(cpt(pos + 2));
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
                    (c1 == 'l' && c2 == 'l')) {
                    pos += 3;
                    emit(pos);
                    continue;
                }
            }
        }
        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ -- the optional prefix may be any
        // non-CR/LF non-letter non-number code point (incl. space/punct).
        if (!(c == '\r' || c == '\n' || (f & ud::kFlagNumber) != 0u)) {
            if (is_l(pos) || is_m(pos) || is_l(pos + 1) || is_m(pos + 1)) {
                pos += 1;
                while (is_l(pos) || is_m(pos)) { pos += 1; }
                emit(pos);
                continue;
            }
        }
        // \p{N} -- one number code point per pre-token.
        if ((f & ud::kFlagNumber) != 0u) {
            pos += 1;
            emit(pos);
            continue;
        }
        //  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* -- symbol run; a single leading
        // space joins the run. The class test uses the SECOND code point
        // when the first is a space (llama.cpp flags2 semantics).
        {
            auto f2 = c == ' ' ? flags(pos + 1) : f;
            const auto in_class = [&](uint32_t fl) {
                return (fl & (ud::kFlagWhitespace | ud::kFlagLetter |
                              ud::kFlagAccentMark | ud::kFlagNumber)) == 0u &&
                       fl != 0u;// undefined/out-of-range never matches
            };
            if (in_class(f2)) {
                pos += (c == ' ') ? 1u : 0u;
                while (in_class(f2)) {
                    pos += 1;
                    f2 = flags(pos);
                }
                while (cpt(pos) == '\r' || cpt(pos) == '\n') { pos += 1; }
                emit(pos);
                continue;
            }
        }
        // \s*[\r\n]+ | \s+(?!\S) | \s+ -- whitespace runs. Scan the run and
        // the position of its LAST CR/LF; a run containing CR/LF ends after
        // the last one, otherwise all-but-last when followed by non-space.
        {
            size_t count = 0;
            size_t last_crlf_end = 0;
            while (is_s(pos + count)) {
                const auto c2 = cpt(pos + count);
                if (c2 == '\r' || c2 == '\n') { last_crlf_end = pos + count + 1; }
                count += 1;
            }
            if (last_crlf_end > 0) {
                pos = last_crlf_end;
                emit(pos);
                continue;
            }
            if (count > 1 && cpt(pos + count) != kOutOfRange) {
                pos += count - 1;
                emit(pos);
                continue;
            }
            if (count > 0) {
                pos += count;
                emit(pos);
                continue;
            }
        }
        // No alternative matched (e.g. an unassigned code point): emit it
        // alone, exactly like llama.cpp's fallback.
        pos += 1;
        emit(pos);
    }
    return spans;
}

// ---------------------------------------------------------------------------
// Loading / validation
// ---------------------------------------------------------------------------

QwenTokenizer::~QwenTokenizer() noexcept = default;


namespace {

// GGUF token_type values (llama.cpp llama_token_type).
constexpr int64_t kTokenTypeNormal = 1;
constexpr int64_t kTokenTypeControl = 3;
constexpr int64_t kTokenTypeUserDefined = 4;
constexpr int64_t kTokenTypeUnused = 5;

struct MutJson {
    yyjson_mut_doc *doc{nullptr};
    MutJson() noexcept : doc{yyjson_mut_doc_new(nullptr)} {}
    ~MutJson() noexcept {
        if (doc != nullptr) { yyjson_mut_doc_free(doc); }
    }
    MutJson(const MutJson &) = delete;
    MutJson &operator=(const MutJson &) = delete;
};

}// namespace

[[nodiscard]] Result<QwenTokenizer::Ptr> QwenTokenizer::load(
    const GgufReader &reader, uint64_t expected_vocab) {
    using R = Result<QwenTokenizer::Ptr>;
    // --- identity: only the exact supported byte-level BPE flavor ---------
    const auto model = reader.meta_string("tokenizer.ggml.model");
    const auto pre = reader.meta_string("tokenizer.ggml.pre");
    if (!model.has_value() || *model != "gpt2") {
        return R::failure(StatusCode::unsupported,
                          "tokenizer.ggml.model is not 'gpt2' (only byte-level BPE is supported)",
                          "tokenizer.ggml.model");
    }
    if (!pre.has_value() || (*pre != "qwen35" && *pre != "qwen2")) {
        return R::failure(StatusCode::unsupported,
                          "tokenizer.ggml.pre '" + luisa::string{pre.value_or("")} +
                              "' has no exact pre-tokenizer implementation (supported: qwen35)",
                          "tokenizer.ggml.pre");
    }
    if (*pre == "qwen2") {
        // qwen2 differs from qwen35 (\p{M} handling and \p{N} grouping); the
        // exact matcher here implements qwen35 only. Fail closed.
        return R::failure(StatusCode::unsupported,
                          "tokenizer.ggml.pre 'qwen2' is not implemented exactly (qwen35 only)",
                          "tokenizer.ggml.pre");
    }

    // --- tables ------------------------------------------------------------
    auto tokens_r = reader.meta_string_array("tokenizer.ggml.tokens");
    if (!tokens_r.ok()) { return R::failure(std::move(tokens_r.status)); }
    auto merges_r = reader.meta_string_array("tokenizer.ggml.merges");
    if (!merges_r.ok()) { return R::failure(std::move(merges_r.status)); }
    const auto types_arr = reader.meta_array("tokenizer.ggml.token_type");
    if (!types_arr.has_value()) {
        return R::failure(StatusCode::not_found, "tokenizer.ggml.token_type missing", "tokenizer.ggml");
    }
    auto &tokens = *tokens_r.value;
    auto &merges = *merges_r.value;
    const uint64_t vocab = tokens.size();
    if (vocab == 0 || vocab > 1u << 20u) {
        return R::failure(invalid("tokenizer vocab size out of range", "tokenizer.ggml.tokens"));
    }
    if (expected_vocab != 0 && vocab != expected_vocab) {
        return R::failure(invalid("tokenizer vocab size does not match the model geometry",
                                  "tokenizer.ggml.tokens"));
    }
    if (types_arr->size() != vocab) {
        return R::failure(invalid("token_type count does not match token count", "tokenizer.ggml.token_type"));
    }
    if (merges.size() > vocab) {
        return R::failure(invalid("merge count exceeds vocab size", "tokenizer.ggml.merges"));
    }

    // Allocate through Luisa's allocator; the private ctor runs inside this
    // member function. The custom deleter frees through the same allocator.
    auto tok = Ptr{new (luisa::allocate_with_allocator<QwenTokenizer>()) QwenTokenizer{}};
    tok->_pieces = std::move(tokens);
    tok->_is_special.resize(vocab, false);
    tok->_piece_to_id.reserve(vocab);

    uint64_t special_count = 0, unused_count = 0;
    for (uint64_t id = 0; id < vocab; id++) {
        const auto &piece = tok->_pieces[id];
        if (piece.empty()) {
            return R::failure(invalid("empty token piece", "tokenizer.ggml.tokens"));
        }
        if (!tok->_piece_to_id.emplace(piece, static_cast<uint32_t>(id)).second) {
            return R::failure(invalid("duplicate token piece", "tokenizer.ggml.tokens"));
        }
        const auto &tv = (*types_arr)[id];
        int64_t type = 0;
        switch (tv.type) {
            case GgufMetaType::u8:
            case GgufMetaType::u16:
            case GgufMetaType::u32:
            case GgufMetaType::u64: type = static_cast<int64_t>(tv.u64); break;
            case GgufMetaType::i8:
            case GgufMetaType::i16:
            case GgufMetaType::i32:
            case GgufMetaType::i64: type = tv.i64; break;
            default:
                return R::failure(invalid("token_type entries must be integers", "tokenizer.ggml.token_type"));
        }
        switch (type) {
            case kTokenTypeNormal: break;
            case kTokenTypeControl:
            case kTokenTypeUserDefined:
                tok->_is_special[id] = true;
                tok->_specials.emplace_back(SpecialEntry{static_cast<uint32_t>(id), piece});
                special_count += 1;
                break;
            case kTokenTypeUnused:
                unused_count += 1;
                // Unused padding slots ("[PADnnnn]") stay in the vocab (IDs
                // are preserved exactly) but never match input text.
                break;
            default:
                return R::failure(StatusCode::unsupported,
                                  "unsupported GGUF token_type " + num_str(static_cast<uint64_t>(type)),
                                  "tokenizer.ggml.token_type");
        }
    }

    // --- byte alphabet: the 256 single-byte base tokens --------------------
    // The GGUF lists the 256 GPT-2 single-byte pieces as ids 0..255, but in
    // CODEPOINT-sorted order, NOT raw-byte order (id 0 = '!' = byte 33,
    // id 188 = the byte-0 piece, id 220 = the space piece). Build the
    // byte -> id map by looking up each GPT-2 byte piece; require all 256 to
    // be present, single-codepoint, and within the first 256 ids.
    {
        bool seen_byte[256] = {};
        for (uint32_t b = 0; b < 256u; b++) {
            luisa::string piece;
            append_cp(piece, byte_to_cp(static_cast<uint8_t>(b)));
            const auto it = tok->_piece_to_id.find(piece);
            if (it == tok->_piece_to_id.end() || it->second >= 256u) {
                return R::failure(invalid("missing GPT-2 single-byte base token for byte " +
                                              num_str(static_cast<uint64_t>(b)),
                                          "tokenizer.ggml.tokens"));
            }
            if (seen_byte[b]) {
                return R::failure(invalid("duplicate GPT-2 byte base token", "tokenizer.ggml.tokens"));
            }
            seen_byte[b] = true;
            tok->_byte_tokens[b] = it->second;
        }
    }
    // Every NORMAL piece must decode through the GPT-2 alphabet.
    for (uint64_t id = 256; id < vocab; id++) {
        if (tok->_is_special[id]) { continue; }
        luisa::string bytes;
        if (!piece_to_bytes(tok->_pieces[id], bytes) || bytes.empty()) {
            return R::failure(invalid("normal token piece is not byte-level GPT-2 text",
                                      "tokenizer.ggml.tokens"));
        }
    }

    // --- merges: "a b", rank = list index, both sides known pieces ---------
    tok->_ranks.reserve(merges.size());
    for (uint64_t r = 0; r < merges.size(); r++) {
        const auto &m = merges[r];
        const auto space = m.find(' ');
        if (space == luisa::string_view::npos || space == 0 || space + 1 >= m.size()) {
            return R::failure(invalid("merge string must be 'a b'", "tokenizer.ggml.merges"));
        }
        const auto a = m.substr(0, space);
        const auto b = m.substr(space + 1);
        if (a.find(' ') != luisa::string_view::npos || b.find(' ') != luisa::string_view::npos) {
            return R::failure(invalid("merge pieces must be space-free", "tokenizer.ggml.merges"));
        }
        const auto ia = tok->_piece_to_id.find(luisa::string{a});
        const auto ib = tok->_piece_to_id.find(luisa::string{b});
        if (ia == tok->_piece_to_id.end() || ib == tok->_piece_to_id.end()) {
            return R::failure(invalid("merge references an unknown piece", "tokenizer.ggml.merges"));
        }
        const auto joined = luisa::string{a} + luisa::string{b};
        const auto ij = tok->_piece_to_id.find(joined);
        if (ij == tok->_piece_to_id.end()) {
            return R::failure(invalid("merge result piece is not in the vocab", "tokenizer.ggml.merges"));
        }
        if (ij->second < 256u) {
            return R::failure(invalid("merge result collides with a base byte token", "tokenizer.ggml.merges"));
        }
        luisa::string key{a};
        key.push_back('\0');
        key.append(b);
        if (!tok->_ranks.emplace(std::move(key), static_cast<uint32_t>(r)).second) {
            return R::failure(invalid("duplicate merge pair", "tokenizer.ggml.merges"));
        }
    }

    // --- special ids --------------------------------------------------------
    const auto eos = reader.meta_u64("tokenizer.ggml.eos_token_id");
    const auto pad = reader.meta_u64("tokenizer.ggml.padding_token_id");
    if (!eos.has_value() || !pad.has_value() || *eos >= vocab || *pad >= vocab) {
        return R::failure(invalid("eos/padding token ids missing or outside the vocab", "tokenizer.ggml"));
    }
    if (!tok->_is_special[*eos]) {
        return R::failure(invalid("eos token id is not a CONTROL/USER_DEFINED token", "tokenizer.ggml.eos_token_id"));
    }
    tok->_info.eos_id = static_cast<uint32_t>(*eos);
    tok->_info.pad_id = static_cast<uint32_t>(*pad);

    // Longest-first special matching, stable by id (OCR convention).
    std::sort(tok->_specials.begin(), tok->_specials.end(),
              [](const SpecialEntry &a, const SpecialEntry &b) {
                  if (a.content.size() != b.content.size()) { return a.content.size() > b.content.size(); }
                  return a.id < b.id;
              });

    // --- construct the HF tokenizer.json the OCR loader supports -----------
    MutJson mj;
    if (mj.doc == nullptr) {
        return R::failure(StatusCode::out_of_memory, "cannot allocate tokenizer JSON document");
    }
    auto *doc = mj.doc;
    auto *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "version", "1.0");
    yyjson_mut_obj_add_null(doc, root, "truncation");
    yyjson_mut_obj_add_null(doc, root, "padding");
    yyjson_mut_obj_add_null(doc, root, "normalizer");// llama.cpp applies none for gpt2
    // added_tokens: CONTROL + USER_DEFINED entries (special=true).
    auto *added = yyjson_mut_arr(doc);
    for (const auto &s : tok->_specials) {
        auto *e = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_uint(doc, e, "id", s.id);
        yyjson_mut_obj_add_strncpy(doc, e, "content", s.content.data(), s.content.size());
        yyjson_mut_obj_add_bool(doc, e, "single_word", false);
        yyjson_mut_obj_add_bool(doc, e, "lstrip", false);
        yyjson_mut_obj_add_bool(doc, e, "rstrip", false);
        yyjson_mut_obj_add_bool(doc, e, "normalized", false);
        yyjson_mut_obj_add_bool(doc, e, "special", true);
        yyjson_mut_arr_append(added, e);
    }
    yyjson_mut_obj_add(root, yyjson_mut_str(doc, "added_tokens"), added);
    // post_processor: ByteLevel (no automatic BOS/EOS insertion).
    auto *post = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_str(doc, post, "type", "ByteLevel");
    yyjson_mut_obj_add_bool(doc, post, "add_prefix_space", false);
    yyjson_mut_obj_add_bool(doc, post, "trim_offsets", false);
    yyjson_mut_obj_add_bool(doc, post, "use_regex", false);
    yyjson_mut_obj_add(root, yyjson_mut_str(doc, "post_processor"), post);
    // pre_tokenizer: the exact qwen35 regex is applied by THIS class; the
    // constructed document declares a plain ByteLevel so the OCR loader can
    // validate/own the vocab. Encoding never routes whole text through the
    // OCR pre-tokenizer (see encode()).
    auto *pre_obj = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_str(doc, pre_obj, "type", "ByteLevel");
    yyjson_mut_obj_add_bool(doc, pre_obj, "add_prefix_space", false);
    yyjson_mut_obj_add_bool(doc, pre_obj, "trim_offsets", false);
    yyjson_mut_obj_add_bool(doc, pre_obj, "use_regex", true);
    yyjson_mut_obj_add(root, yyjson_mut_str(doc, "pre_tokenizer"), pre_obj);
    // model: BPE with the GPT-2 byte codec, the exact vocab and merges.
    auto *model_obj = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_str(doc, model_obj, "type", "BPE");
    yyjson_mut_obj_add_null(doc, model_obj, "dropout");
    yyjson_mut_obj_add_null(doc, model_obj, "unk_token");
    yyjson_mut_obj_add_null(doc, model_obj, "continuing_subword_prefix");
    yyjson_mut_obj_add_null(doc, model_obj, "end_of_word_suffix");
    yyjson_mut_obj_add_bool(doc, model_obj, "fuse_unk", false);
    yyjson_mut_obj_add_bool(doc, model_obj, "ignore_merges", false);
    auto *byte_encoder = yyjson_mut_obj(doc);
    auto *byte_decoder = yyjson_mut_obj(doc);
    for (uint32_t b = 0; b < 256u; b++) {
        luisa::string piece;
        append_cp(piece, byte_to_cp(static_cast<uint8_t>(b)));
        char key[8];
        std::snprintf(key, sizeof(key), "%u", b);
        // Keys are NOT copied by yyjson: the byte-decoder key must be a
        // doc-owned string value (the piece local dies each iteration).
        yyjson_mut_obj_add_strncpy(doc, byte_encoder, key, piece.data(), piece.size());
        auto *dec_key = yyjson_mut_strncpy(doc, piece.data(), piece.size());
        auto *dec_val = yyjson_mut_uint(doc, b);
        if (dec_key == nullptr || dec_val == nullptr ||
            !yyjson_mut_obj_add(byte_decoder, dec_key, dec_val)) {
            return R::failure(StatusCode::out_of_memory, "cannot build byte-decoder JSON");
        }
    }
    yyjson_mut_obj_add(model_obj, yyjson_mut_str(doc, "byte_encoder"), byte_encoder);
    yyjson_mut_obj_add(model_obj, yyjson_mut_str(doc, "byte_decoder"), byte_decoder);
    auto *vocab_obj = yyjson_mut_obj(doc);
    for (uint64_t id = 0; id < vocab; id++) {
        const auto &piece = tok->_pieces[id];
        if (!yyjson_mut_obj_add_uint(doc, vocab_obj, piece.c_str(), id)) {
                }
    }
    yyjson_mut_obj_add(model_obj, yyjson_mut_str(doc, "vocab"), vocab_obj);
    auto *merges_arr = yyjson_mut_arr(doc);
    for (const auto &m : merges) {
        yyjson_mut_arr_add_strncpy(doc, merges_arr, m.data(), m.size());
    }
    yyjson_mut_obj_add(model_obj, yyjson_mut_str(doc, "merges"), merges_arr);
    yyjson_mut_obj_add(root, yyjson_mut_str(doc, "model"), model_obj);

    size_t json_len = 0;
    char *json_text = yyjson_mut_write(doc, YYJSON_WRITE_NOFLAG, &json_len);
    if (json_text == nullptr) {
        return R::failure(StatusCode::out_of_memory, "cannot serialize tokenizer JSON");
    }
    tok->_json.assign(json_text, json_len);
    std::free(json_text);

    ocr::TokenizerConfig cfg;
    cfg.max_document_bytes = 256ull * 1024ull * 1024ull;
    cfg.max_vocab_entries = 1u << 20u;
    cfg.max_added_tokens = 4096u;
    auto base = ocr::Tokenizer::parse(tok->_json, cfg);
    if (!base.ok()) { return R::failure(std::move(base.status)); }
    if ((*base.value)->vocab_size() != vocab) {
        return R::failure(Status::error(StatusCode::internal_error,
                                        "constructed tokenizer vocab size mismatch", "tokenizer"));
    }
    tok->_base = std::move(*base.value);

    tok->_info.vocab_size = vocab;
    tok->_info.merge_count = merges.size();
    tok->_info.special_token_count = special_count;
    tok->_info.unused_pad_count = unused_count;
    tok->_info.model = luisa::string{*model};
    tok->_info.pre = luisa::string{*pre};
    tok->_info.diagnostics.emplace_back(Diagnostic{
        DiagnosticSeverity::info, "tokenizer_exact_qwen35",
        "GGUF tokenizer validated; exact qwen35 pre-tokenizer + reused OCR byte-level BPE engine"});
    return R::success(std::move(tok));
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::optional<uint32_t> QwenTokenizer::token_to_id(luisa::string_view piece) const noexcept {
    const auto it = _piece_to_id.find(luisa::string{piece});
    if (it == _piece_to_id.end()) { return luisa::nullopt; }
    return it->second;
}

[[nodiscard]] luisa::optional<luisa::string> QwenTokenizer::id_to_token(uint32_t id) const {
    if (id >= _pieces.size()) { return luisa::nullopt; }
    return _pieces[id];
}

[[nodiscard]] bool QwenTokenizer::is_special(uint32_t id) const noexcept {
    return id < _is_special.size() && _is_special[id];
}

// ---------------------------------------------------------------------------
// Encoding: exact qwen35 pre-tokenization + reused BPE merge convention.
// ---------------------------------------------------------------------------

[[nodiscard]] Result<luisa::vector<uint32_t>> QwenTokenizer::encode(
    luisa::string_view text, bool add_special_tokens) const {
    using R = Result<luisa::vector<uint32_t>>;
    (void)add_special_tokens;// ByteLevel post-processor: nothing is inserted
    if (!ocr_is_valid_utf8(text)) {
        return R::failure(invalid("encode input is not valid UTF-8 (no lossy replacement)", "text"));
    }
    luisa::vector<uint32_t> ids;
    luisa::string normal;
    const auto flush_normal = [&]() -> Status {
        if (normal.empty()) { return Status::success(); }
        const auto spans = qwen35_pretokenize(normal);
        for (const auto &span : spans) {
            // Map each raw byte to its GPT-2 symbol id (base tokens 0..255).
            luisa::vector<uint32_t> symbols;
            symbols.reserve(span.end - span.begin);
            for (size_t i = span.begin; i < span.end; i++) {
                symbols.emplace_back(_byte_tokens[static_cast<uint8_t>(normal[i])]);
            }
            // Greedy lowest-rank merge, leftmost on rank ties -- the exact
            // convention of ocr::Tokenizer::encode_bpe / llama.cpp.
            while (symbols.size() > 1) {
                uint32_t best_rank = UINT32_MAX;
                size_t best_index = 0;
                bool found = false;
                for (size_t i = 0; i + 1 < symbols.size(); i++) {
                    luisa::string key = _pieces[symbols[i]];
                    key.push_back('\0');
                    key += _pieces[symbols[i + 1]];
                    const auto it = _ranks.find(key);
                    if (it != _ranks.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_index = i;
                        found = true;
                    }
                }
                if (!found) { break; }
                // The merged piece must exist (validated at load).
                const auto joined = _pieces[symbols[best_index]] + _pieces[symbols[best_index + 1]];
                const auto jt = _piece_to_id.find(joined);
                if (jt == _piece_to_id.end()) {
                    return Status::error(StatusCode::internal_error,
                                         "BPE merge result missing from vocab", "encode");
                }
                symbols[best_index] = jt->second;
                symbols.erase(symbols.begin() + static_cast<ptrdiff_t>(best_index + 1));
            }
            for (auto s : symbols) { ids.emplace_back(s); }
        }
        normal.clear();
        return Status::success();
    };
    // Split around special tokens, longest match first (OCR convention).
    size_t pos = 0;
    while (pos < text.size()) {
        bool matched = false;
        for (const auto &s : _specials) {
            if (s.content.size() > text.size() - pos) { continue; }
            if (text.compare(pos, s.content.size(), s.content) != 0) { continue; }
            if (auto st = flush_normal(); !st.ok()) { return R::failure(std::move(st)); }
            ids.emplace_back(s.id);
            pos += s.content.size();
            matched = true;
            break;
        }
        if (!matched) {
            normal.push_back(text[pos]);
            pos += 1;
        }
    }
    if (auto st = flush_normal(); !st.ok()) { return R::failure(std::move(st)); }
    return R::success(std::move(ids));
}

[[nodiscard]] Result<luisa::string> QwenTokenizer::decode(
    luisa::span<const uint32_t> ids, bool skip_special_tokens, bool allow_partial_utf8) const {
    for (auto id : ids) {
        if (id >= _pieces.size()) {
            return Result<luisa::string>::failure(invalid("unknown token id in decode", "ids"));
        }
    }
    return _base->decode(ids, skip_special_tokens, allow_partial_utf8);
}

[[nodiscard]] ocr::IncrementalDetokenizer QwenTokenizer::make_detokenizer(bool skip_special_tokens) const {
    ocr::DetokenizerOptions options;
    options.skip_special_tokens = skip_special_tokens;
    // Stop on the GGUF EOS (<|im_end|>), not the OCR heuristic's <|endoftext|>.
    options.eos_ids.emplace_back(_info.eos_id);
    return ocr::IncrementalDetokenizer{*_base, std::move(options)};
}

// ---------------------------------------------------------------------------
// Chat prompt (fixed single-user-turn wrapper; see header).
// ---------------------------------------------------------------------------

[[nodiscard]] Result<luisa::string> QwenTokenizer::chat_prompt_text(luisa::string_view user_prompt) const {
    using R = Result<luisa::string>;
    if (!ocr_is_valid_utf8(user_prompt)) {
        return R::failure(invalid("chat prompt is not valid UTF-8", "prompt"));
    }
    // The wrapper must not be spoofable: prompt text containing any special
    // marker is rejected (encode() would otherwise split it into tokens).
    for (const auto &s : _specials) {
        if (user_prompt.find(s.content) != luisa::string_view::npos) {
            return R::failure(invalid("chat prompt contains a special-token marker", "prompt"));
        }
    }
    luisa::string out;
    out += "<|im_start|>user\n";
    out.append(user_prompt.data(), user_prompt.size());
    out += "<|im_end|>\n<|im_start|>assistant\n";
    return R::success(std::move(out));
}

[[nodiscard]] Result<luisa::vector<uint32_t>> QwenTokenizer::chat_prompt_ids(luisa::string_view user_prompt) const {
    using R = Result<luisa::vector<uint32_t>>;
    auto text = chat_prompt_text(user_prompt);
    if (!text.ok()) { return R::failure(std::move(text.status)); }
    auto ids = encode(*text.value, false);
    if (!ids.ok()) { return R::failure(std::move(ids.status)); }
    // Non-thinking default of the checkpoint template: an empty think block.
    auto think = encode("\n\n", false);
    if (!think.ok()) { return R::failure(std::move(think.status)); }
    for (auto id : *think.value) { (*ids.value).emplace_back(id); }
    return ids;
}

// ---------------------------------------------------------------------------
// Device-free self-test
// ---------------------------------------------------------------------------

namespace {

// Minimal GGUF v3 writer for a tokenizer-only fixture (no tensors).
struct FixtureGguf {
    luisa::vector<std::byte> bytes;
    void u32(uint32_t x) {
        for (uint32_t i = 0; i < 4u; i++) { bytes.emplace_back(static_cast<std::byte>((x >> (8u * i)) & 0xffu)); }
    }
    void u64(uint64_t x) {
        for (uint32_t i = 0; i < 8u; i++) { bytes.emplace_back(static_cast<std::byte>((x >> (8u * i)) & 0xffu)); }
    }
    void str(luisa::string_view s) {
        u64(s.size());
        for (auto c : s) { bytes.emplace_back(static_cast<std::byte>(c)); }
    }
    void key(luisa::string_view k, uint32_t type) {
        str(k);
        u32(type);
    }
    void str_array(luisa::string_view k, const luisa::vector<luisa::string> &v) {
        key(k, 9);
        u32(8);
        u64(v.size());
        for (const auto &s : v) { str(s); }
    }
    void i32_array(luisa::string_view k, const luisa::vector<int32_t> &v) {
        key(k, 9);
        u32(5);
        u64(v.size());
        for (auto x : v) { u32(static_cast<uint32_t>(x)); }
    }
};

// Build a small but EXACT byte-level BPE fixture: 256 base tokens + a handful
// of merges + Qwen-style specials. Vocab layout mirrors the real checkpoint
// (merges first, specials after).
struct TokFixture {
    luisa::vector<luisa::string> tokens;
    luisa::vector<int32_t> types;
    luisa::vector<luisa::string> merges;
    uint32_t eos{0};
    uint32_t pad{0};

    TokFixture() {
        for (uint32_t b = 0; b < 256u; b++) {
            luisa::string piece;
            append_cp(piece, byte_to_cp(static_cast<uint8_t>(b)));
            tokens.emplace_back(std::move(piece));
            types.emplace_back(kTokenTypeNormal);
        }
        // merges: "Ġ t" -> "Ġt"; "Ġt h" -> "Ġth"; "Ġth e" -> "Ġthe";
        //         "h e" -> "he"; "l l" -> "ll"; "Ġ w" -> "Ġw"
        const auto add_merge = [&](luisa::string_view a, luisa::string_view b) {
            merges.emplace_back(luisa::string{a} + " " + luisa::string{b});
            tokens.emplace_back(luisa::string{a} + luisa::string{b});
            types.emplace_back(kTokenTypeNormal);
        };
        add_merge("\xc4\xa0", "t");        // Ġ + t
        add_merge("\xc4\xa0t", "h");       // Ġt + h
        add_merge("\xc4\xa0th", "e");      // Ġth + e
        add_merge("h", "e");               // h + e
        add_merge("l", "l");               // l + l
        add_merge("\xc4\xa0", "w");        // Ġ + w
        add_merge("e", "l");               // e + l
        add_merge("el", "l");              // el + l
        add_merge("ell", "o");             // ell + o
        // specials (ids follow the merges)
        const auto add_special = [&](luisa::string_view content, int32_t type) -> uint32_t {
            const auto id = static_cast<uint32_t>(tokens.size());
            tokens.emplace_back(content);
            types.emplace_back(type);
            return id;
        };
        add_special("<|endoftext|>", kTokenTypeControl);
        const auto im_start = add_special("<|im_start|>", kTokenTypeControl);
        eos = add_special("<|im_end|>", kTokenTypeControl);
        pad = add_special("<|vision_pad|>", kTokenTypeControl);
        add_special("<|unused_pad|>", kTokenTypeUnused);
        (void)im_start;
    }

    [[nodiscard]] luisa::filesystem::path write(const luisa::filesystem::path &dir) const {
        FixtureGguf g;
        g.bytes.emplace_back(static_cast<std::byte>('G'));
        g.bytes.emplace_back(static_cast<std::byte>('G'));
        g.bytes.emplace_back(static_cast<std::byte>('U'));
        g.bytes.emplace_back(static_cast<std::byte>('F'));
        g.u32(3);
        // The shared GGUF reader requires >= 1 tensor; add one dummy F32
        // scalar (tokenizer-only fixtures still exercise the real parser).
        g.u64(1);// tensor count
        g.u64(7);// kv count
        g.key("tokenizer.ggml.model", 8);
        g.str("gpt2");
        g.key("tokenizer.ggml.pre", 8);
        g.str("qwen35");
        g.str_array("tokenizer.ggml.tokens", tokens);
        g.i32_array("tokenizer.ggml.token_type", types);
        g.str_array("tokenizer.ggml.merges", merges);
        g.key("tokenizer.ggml.eos_token_id", 4);
        g.u32(eos);
        g.key("tokenizer.ggml.padding_token_id", 4);
        g.u32(pad);
        // dummy tensor info: name, rank 1, dim 1, type F32, relative offset 0
        g.str("dummy.weight");
        g.u32(1);
        g.u64(1);
        g.u32(0);
        g.u64(0);
        // pad to the 32-byte data-section alignment, then the 4-byte payload
        while (g.bytes.size() % 32u != 0u) { g.bytes.emplace_back(std::byte{0}); }
        g.u32(0);
        const auto path = dir / "qwen_tok_fixture.gguf";
        std::FILE *f = std::fopen(path.string().c_str(), "wb");
        if (f != nullptr) {
            std::fwrite(g.bytes.data(), 1, g.bytes.size(), f);
            std::fclose(f);
        }
        return path;
    }
};

}// namespace

[[nodiscard]] Status QwenTokenizer::self_test(luisa::vector<luisa::string> *report) {
    const auto note = [&](luisa::string line) {
        if (report != nullptr) { report->emplace_back(std::move(line)); }
    };
    const auto fail = [&](luisa::string msg) -> Status {
        note("FAIL: " + msg);
        return Status::error(StatusCode::internal_error, "tokenizer self-test: " + msg);
    };

    // --- fixture load -------------------------------------------------------
    std::error_code ec;
    const auto dir = luisa::filesystem::temp_directory_path(ec) / "lcqwen_tok_selftest";
    if (ec) { return fail("cannot resolve temp directory"); }
    luisa::filesystem::create_directories(dir, ec);
    if (ec) { return fail("cannot create temp directory"); }
    struct Cleanup {
        luisa::filesystem::path p;
        ~Cleanup() {
            std::error_code e;
            luisa::filesystem::remove_all(p, e);
        }
    } cleanup{dir};

    TokFixture fx;
    const auto path = fx.write(dir);
    auto reader = GgufReader::open(path);
    if (!reader.ok()) { return fail("fixture GGUF open: " + reader.status.message); }
    auto tok_r = QwenTokenizer::load(**reader.value, fx.tokens.size());
    if (!tok_r.ok()) { return fail("fixture load: " + tok_r.status.message); }
    auto &tok = **tok_r.value;
    note("fixture load ok (vocab " + num_str(static_cast<uint64_t>(tok.vocab_size())) + ", merges " +
         num_str(tok.info().merge_count) + ")");

    // --- identity / specials -------------------------------------------------
    if (tok.eos_id() != fx.eos) { return fail("eos id mismatch"); }
    if (tok.pad_id() != fx.pad) { return fail("pad id mismatch"); }
    if (!tok.is_special(fx.eos) || tok.is_special(0)) { return fail("special classification"); }
    if (tok.info().unused_pad_count != 1) { return fail("unused pad count"); }
    note("special/eos/pad classification ok");

    // --- encode: merges + leading-space behavior ------------------------------
    {
        auto ids = tok.encode("the", false);
        if (!ids.ok()) { return fail("encode 'the': " + ids.status.message); }
        // 't','h','e' with no prefix space: h+e merge (rank 3) -> t, he
        const auto t_id = tok.token_to_id("t");
        const auto he_id = tok.token_to_id("he");
        if (!t_id.has_value() || !he_id.has_value()) { return fail("fixture vocab lookup"); }
        if (*ids.value != luisa::vector<uint32_t>{*t_id, *he_id}) { return fail("encode 'the' ids"); }
        auto ids2 = tok.encode(" the", false);
        if (!ids2.ok()) { return fail("encode ' the'"); }
        const auto the_id = tok.token_to_id("\xc4\xa0the");
        if (!the_id.has_value() || *ids2.value != luisa::vector<uint32_t>{*the_id}) {
            return fail("encode ' the' should merge to the single Ġthe token");
        }
        note("BPE merge ranks + space handling ok");
    }
    // --- encode/decode round-trips incl. multi-byte UTF-8 ---------------------
    {
        const luisa::string samples[] = {
            "Hello, world!", "hello world", "  leading and trailing  ",
            "line1\nline2\r\nline3", "na\xc3\xafve caf\xc3\xa9",
            "\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c", // 你好，世界
            "\xf0\x9f\x98\x80 emoji", "12345 67890", "don't stop",
            "a\tb\xc2\xa0" "c", ""};
        for (const auto &s : samples) {
            auto ids = tok.encode(s, false);
            if (!ids.ok()) { return fail("encode round-trip sample: " + ids.status.message); }
            auto text = tok.decode(*ids.value, true, false);
            if (!text.ok()) { return fail("decode round-trip: " + text.status.message); }
            if (*text.value != s) { return fail("round-trip mismatch for sample"); }
        }
        note("encode/decode round-trips ok (ASCII/CJK/emoji/whitespace)");
    }
    // --- special tokens in text are matched, never split ----------------------
    {
        auto ids = tok.encode("<|im_start|>user\nhi<|im_end|>", false);
        if (!ids.ok()) { return fail("encode special text"); }
        const auto start_id = tok.token_to_id("<|im_start|>");
        if (!start_id.has_value() || (*ids.value).front() != *start_id ||
            (*ids.value).back() != tok.eos_id()) {
            return fail("special tokens not matched as single ids");
        }
        auto text = tok.decode(*ids.value, true, false);
        if (!text.ok() || *text.value != "user\nhi") { return fail("special skip on decode"); }
        auto text2 = tok.decode(*ids.value, false, false);
        if (!text2.ok() || (*text2.value).find("<|im_start|>") == luisa::string::npos) {
            return fail("special kept on decode");
        }
        note("special-token split/skip ok");
    }
    // --- incremental detokenizer: UTF-8 split across tokens + EOS stop --------
    {
        auto detok = tok.make_detokenizer(true);
        const auto ni = tok.encode("\xe4\xbd\xa0\xe5\xa5\xbd", false);// 你好
        if (!ni.ok()) { return fail("encode CJK for detok"); }
        luisa::string streamed;
        for (auto id : *ni.value) {
            auto piece = detok.push(id);
            if (!piece.ok()) { return fail("detok push: " + piece.status.message); }
            streamed += *piece.value;
        }
        auto tail = detok.finish();
        if (!tail.ok()) { return fail("detok finish"); }
        streamed += *tail.value;
        if (streamed != "\xe4\xbd\xa0\xe5\xa5\xbd") { return fail("detok CJK mismatch"); }
        // EOS stops the stream; later pushes emit nothing.
        auto detok2 = tok.make_detokenizer(true);
        auto a = detok2.push(*tok.token_to_id("h"));
        if (!a.ok()) { return fail("detok2 push h"); }
        auto e = detok2.push(tok.eos_id());
        if (!e.ok() || !detok2.stopped()) { return fail("detok2 EOS stop"); }
        auto z = detok2.push(*tok.token_to_id("e"));
        if (!z.ok() || !(*z.value).empty()) { return fail("detok2 post-EOS leak"); }
        note("incremental detokenizer ok (UTF-8 hold-back, GGUF EOS stop)");
    }
    // --- rejection paths -------------------------------------------------------
    {
        const char bad_utf8[] = {'a', static_cast<char>(0xff), 'b', 0};
        auto bad = tok.encode(bad_utf8, false);
        if (bad.ok()) { return fail("invalid UTF-8 not rejected"); }
        const uint32_t unknown = static_cast<uint32_t>(tok.vocab_size());
        auto dec = tok.decode(luisa::span<const uint32_t>{&unknown, 1});
        if (dec.ok()) { return fail("unknown id decode not rejected"); }
        auto spoof = tok.chat_prompt_ids("hi <|im_start|>");
        if (spoof.ok()) { return fail("chat prompt spoof not rejected"); }
        note("rejection paths ok (UTF-8, unknown id, chat spoof)");
    }
    // --- chat wrapper -----------------------------------------------------------
    {
        auto ids = tok.chat_prompt_ids("hi");
        if (!ids.ok()) { return fail("chat_prompt_ids: " + ids.status.message); }
        auto text = tok.decode(*ids.value, false, false);
        if (!text.ok()) { return fail("chat decode"); }
        const luisa::string expect =
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n\n\n";
        if (*text.value != expect) { return fail("chat wrapper text mismatch: '" + *text.value + "'"); }
        note("chat wrapper ok");
    }
    // --- pre-tokenizer spot checks (exact qwen35 semantics) ----------------------
    {
        const auto spans_of = [](luisa::string_view s) {
            luisa::vector<luisa::string> out;
            for (const auto &sp : QwenTokenizer::qwen35_pretokenize(s)) {
                out.emplace_back(s.substr(sp.begin, sp.end - sp.begin));
            }
            return out;
        };
        using V = luisa::vector<luisa::string>;
        struct Case {
            luisa::string_view text;
            V expect;
        };
        const Case cases[] = {
            {"hello", {"hello"}},
            {" hello", {" hello"}},
            {"hello world", {"hello", " world"}},
            {"123", {"1", "2", "3"}},                       // \p{N} single digits
            {"don't", {"don", "'t"}},                       // contraction
            {"DON'T", {"DON", "'T"}},                       // case-insensitive
            {"a\nb", {"a", "\n", "b"}},                     // \s*[\r\n]+
            {"a \nb", {"a", " \n", "b"}},
            {"a  b", {"a", " ", " b"}},                     // \s+(?!\S)
            {"a  ", {"a", "  "}},                           // trailing \s+
            {"!!!", {"!!!"}},
            {" !!!", {" !!!"}},
            {"a!!!b", {"a", "!!!", "b"}},                   // symbol run splits words
            {"x\t y", {"x", "\t", " y"}},
            {"\xe4\xbd\xa0\xe5\xa5\xbd", {"\xe4\xbd\xa0\xe5\xa5\xbd"}}, // CJK letter run
            {"na\xc3\xafve", {"na\xc3\xafve"}},             // precomposed letter run
            {"nai\xcc\x88ve", {"nai\xcc\x88ve"}},           // combining mark joins (\p{M})
            {"", {}},
        };
        for (const auto &c : cases) {
            const auto got = spans_of(c.text);
            if (got != c.expect) {
                luisa::string got_s, exp_s;
                for (const auto &g : got) { got_s += "[" + g + "]"; }
                for (const auto &e : c.expect) { exp_s += "[" + e + "]"; }
                return fail("qwen35 pretokenize '" + luisa::string{c.text} + "': got " + got_s +
                            " expect " + exp_s);
            }
        }
        note("qwen35 pre-tokenizer spot checks ok");
    }
    // --- OCR BPE cross-check on single pre-tokens --------------------------------
    // For text that is exactly ONE qwen35 pre-token, the OCR GPT-2 matcher
    // produces the same single span, so its encode must agree id-for-id.
    {
        // Only samples whose qwen35 and GPT-2 pre-tokenizations provably
        // coincide (single letter/symbol runs) are compared.
        const luisa::string singles[] = {"hello", "the", "ll", "he", "abc", "!!!"};
        for (const auto &s : singles) {
            auto mine = tok.encode(s, false);
            auto base = tok._base->encode(s, false);
            if (!mine.ok() || !base.ok()) { return fail("cross-check encode failure"); }
            if (*mine.value != *base.value) { return fail("OCR BPE cross-check mismatch on '" + s + "'"); }
        }
        note("OCR BPE engine cross-check ok");
    }
    // --- malformed fixtures fail closed -------------------------------------------
    {
        auto bad_fx = fx;
        bad_fx.tokens[5] = bad_fx.tokens[6];// duplicate piece
        const auto p2 = bad_fx.write(dir);
        auto r2 = GgufReader::open(p2);
        if (r2.ok()) {
            auto t2 = QwenTokenizer::load(**r2.value, bad_fx.tokens.size());
            if (t2.ok()) { return fail("duplicate piece not rejected"); }
        }
        auto bad_fx2 = fx;
        bad_fx2.merges[0] = "\xc4\xa0 z";// unknown merge side 'z'? z IS a base token; use a truly unknown piece
        bad_fx2.merges[0] = "\xc4\xa0\xc4\xa0\xc4\xa0 x";
        const auto p3 = bad_fx2.write(dir);
        auto r3 = GgufReader::open(p3);
        if (r3.ok()) {
            auto t3 = QwenTokenizer::load(**r3.value, bad_fx2.tokens.size());
            if (t3.ok()) { return fail("unknown merge piece not rejected"); }
        }
        auto bad_fx3 = fx;
        bad_fx3.eos = 0;// base byte token, not special
        const auto p4 = bad_fx3.write(dir);
        auto r4 = GgufReader::open(p4);
        if (r4.ok()) {
            auto t4 = QwenTokenizer::load(**r4.value, bad_fx3.tokens.size());
            if (t4.ok()) { return fail("non-special eos not rejected"); }
        }
        note("malformed fixture rejection ok");
    }
    note("tokenizer self-test: all checks passed");
    return Status::success();
}


}// namespace luisa::qwen
