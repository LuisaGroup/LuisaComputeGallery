// Test for src/qwen/qwen/engine.h/.cpp (factory dispatch + shared sampler),
// src/qwen/qwen/tokenizer.h/.cpp (GGUF -> exact qwen35 byte-level BPE) and the
// device-free operator self-tests of both kernel families.
//
// Host-only / device-light: NO device, NO backend, NO real model weights are
// required. The tokenizer tests build a small exact GGUF fixture in a temp
// directory; when data/Qwen3.5-4B-BF16.gguf is present an additional real
// metadata test loads the full 248320-token vocabulary and round-trips text
// (skipped when the file is absent).
#include "ut/ut.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>

#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "qwen/dsl_ops.h"
#include "qwen/engine.h"
#include "qwen/gguf.h"
#include "qwen/tile_ops.h"
#include "qwen/tokenizer.h"

#include "qwen_encode_vectors.h"

using namespace luisa;
using namespace luisa::qwen;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// ---------------------------------------------------------------------------
// engine.cpp: kernel-kind names + deterministic argmax
// ---------------------------------------------------------------------------

void test_kernel_kind_names() {
    expect(qwen_kernel_kind_name(QwenKernelKind::dsl) == "dsl");
    expect(qwen_kernel_kind_name(QwenKernelKind::tile) == "tile");
    expect(qwen_kernel_kind_name(static_cast<QwenKernelKind>(7u)) == "unknown");
}

void test_temperature_sampling() {
    QwenSampleOptions options;
    options.temperature = 0.0f;
    options.seed = 12345u;
    const float tied[] = {1.0f, 3.0f, 3.0f, 2.0f};
    auto greedy = qwen_sample(tied, options, 99u);
    expect(greedy.ok() && *greedy.value == 1u);

    options.temperature = 1.0f;
    auto a = qwen_sample(tied, options, 7u);
    auto b = qwen_sample(tied, options, 7u);
    expect(a.ok() && b.ok() && *a.value == *b.value);
    QwenSampleOptions other_seed = options;
    other_seed.seed++;
    bool seed_or_draw_changes = false;
    for (uint64_t i = 0; i < 32u; i++) {
        auto x = qwen_sample(tied, options, i);
        auto y = qwen_sample(tied, other_seed, i);
        expect(x.ok() && y.ok());
        if (x.ok() && y.ok() && *x.value != *y.value) { seed_or_draw_changes = true; }
    }
    expect(seed_or_draw_changes);

    // A sharply peaked distribution must select its maximum for every draw.
    const float peaked[] = {-1000.0f, 1000.0f, -1000.0f};
    options.temperature = 0.01f;
    for (uint64_t i = 0; i < 64u; i++) {
        auto x = qwen_sample(peaked, options, i);
        expect(x.ok() && *x.value == 1u);
    }

    // -infinity masks entries; +infinity entries share the limiting mass.
    options.temperature = 0.7f;
    const float masked[] = {-INFINITY, 0.0f, -INFINITY};
    for (uint64_t i = 0; i < 16u; i++) {
        auto x = qwen_sample(masked, options, i);
        expect(x.ok() && *x.value == 1u);
    }
    const float infinities[] = {INFINITY, -100.0f, INFINITY};
    bool inf0 = false, inf2 = false;
    for (uint64_t i = 0; i < 64u; i++) {
        auto x = qwen_sample(infinities, options, i);
        expect(x.ok());
        if (x.ok()) { inf0 |= *x.value == 0u; inf2 |= *x.value == 2u; }
    }
    expect(inf0 && inf2);

    const float all_ninf[] = {-INFINITY, -INFINITY};
    const float with_nan[] = {0.0f, NAN};
    expect(!qwen_sample(all_ninf, options, 0u).ok());
    expect(!qwen_sample(with_nan, options, 0u).ok());
    expect(!qwen_sample(luisa::span<const float>{}, options, 0u).ok());
    options.temperature = -1.0f;
    expect(!qwen_sample(tied, options, 0u).ok());
    options.temperature = INFINITY;
    expect(!qwen_sample(tied, options, 0u).ok());
}

void test_argmax() {
    // lowest-index ties
    const float tied[] = {1.0f, 3.0f, 3.0f, 2.0f};
    auto a = qwen_argmax(tied);
    expect(a.ok());
    expect(eq(*a.value, 1u));
    // single element
    const float one[] = {-5.0f};
    auto b = qwen_argmax(one);
    expect(b.ok() && *b.value == 0u);
    // negatives and zero
    const float neg[] = {-2.0f, -1.0f, -1.0f, -3.0f};
    auto c = qwen_argmax(neg);
    expect(c.ok() && *c.value == 1u);
    // infinities order normally; -inf everywhere ties at index 0
    const float inf[] = {-INFINITY, INFINITY, INFINITY};
    auto d = qwen_argmax(inf);
    expect(d.ok() && *d.value == 1u);
    const float ninf[] = {-INFINITY, -INFINITY};
    auto e = qwen_argmax(ninf);
    expect(e.ok() && *e.value == 0u);
    // NaN anywhere is rejected (first, middle, last)
    const float nan0[] = {NAN, 1.0f};
    const float nan1[] = {1.0f, NAN};
    const float nan2[] = {1.0f, 2.0f, NAN};
    expect(!qwen_argmax(nan0).ok());
    expect(!qwen_argmax(nan1).ok());
    expect(!qwen_argmax(nan2).ok());
    // empty is rejected
    expect(!qwen_argmax(luisa::span<const float>{}).ok());
    // larger deterministic sweep: unique max at a known index
    luisa::vector<float> big(1000u, 0.5f);
    big[731] = 2.0f;
    auto f = qwen_argmax(big);
    expect(f.ok() && *f.value == 731u);
    // all-equal keeps index 0
    luisa::vector<float> flat(257u, 1.25f);
    auto g = qwen_argmax(flat);
    expect(g.ok() && *g.value == 0u);
}

// QwenEngine::create must fail closed on invalid handles BEFORE touching any
// backend (device-light: a default-constructed Device/Stream is invalid).
void test_create_invalid_handles() {
    compute::Device device;// invalid (no backend)
    compute::Stream stream;
    Qwen35Model model;// default-constructed; never reached
    QwenDeviceWeights weights;
    QwenEngineOptions options;
    auto r = QwenEngine::create(device, stream, model, weights, options);
    expect(!r.ok());
    expect(r.status.code == StatusCode::invalid_argument);
}

// ---------------------------------------------------------------------------
// caps.cpp: backend capability record (device-free)
// ---------------------------------------------------------------------------

// A value-initialized record is the "unknown / unavailable" answer that every
// accelerator gate in opt_plan/04 must read as false: the probe is allowed to
// fail to determine a capability, but it may never report one it did not
// confirm. This is the host-only half of the caps contract; the device half is
// exercised by the engine suites and by every example_qwen run.
void test_backend_caps_defaults() {
    QwenBackendCaps caps{};
    expect(caps.backend_name.empty());
    expect(caps.warp_size == 0u);
    expect(caps.max_shared_memory_bytes == 0u);
    expect(caps.memory_granularity_bytes == 0u);
    expect(!caps.sparse_buffer);
    expect(caps.sparse_tile_bytes == 0u);
    expect(caps.kv_page_bytes == 0u);
    expect(caps.kv_tiles_per_page == 0u);
    expect(caps.kv_pages_per_region == 0u);
    expect(caps.kv_heap_slots == 0u);
    expect(!caps.cuda_graph);
    expect(!caps.cuda_graph_conditional_nodes);
    expect(!caps.cuda_graph_device_launch);
    expect(!caps.cuda_graph_programmatic_edges);
    expect(!caps.pdl);
    expect(!caps.async_copy_native);
    expect(!caps.split_barrier);
    expect(!caps.cooperative_vector);
    expect(!caps.cooperative_matrix);
    expect(caps.notes.empty());

    // QwenEngineInfo carries the record, so a benchmark log can be read back
    // without re-probing (and without the device still being alive).
    QwenEngineInfo info{};
    expect(info.caps.kv_page_bytes == 0u);
    expect(!info.caps.sparse_buffer);
    info.caps.backend_name = "vk";
    info.caps.kv_page_bytes = 2u * 1024u * 1024u;
    info.caps.kv_tiles_per_page = 32u;
    expect(info.caps.backend_name == "vk");
    expect(info.caps.kv_page_bytes == 2097152u);
    expect(info.caps.kv_tiles_per_page == 32u);

    // The probe itself must be callable without a device and must never throw:
    // it reports the false/0 default and explains it in `notes`.
    compute::Device no_device;
    const auto probed = qwen_probe_backend_caps(no_device);
    expect(probed.backend_name.empty());
    expect(!probed.sparse_buffer);
    expect(!probed.cuda_graph);
    expect(!probed.pdl);
    expect(probed.kv_page_bytes == 0u);
    expect(!probed.notes.empty());
}
// ---------------------------------------------------------------------------
// tokenizer.cpp: device-free self-test + real-GGUF metadata test
// ---------------------------------------------------------------------------

void test_tokenizer_self_test() {
    luisa::vector<luisa::string> report;
    const auto s = QwenTokenizer::self_test(&report);
    if (!s.ok()) {
        for (const auto &line : report) { std::printf("  %s\n", line.c_str()); }
    }
    expect(s.ok());
    expect(report.size() > 5u);
}

void test_qwen35_pretokenizer_unicode() {
    const auto spans_of = [](luisa::string_view s) {
        luisa::vector<luisa::string> out;
        for (const auto &sp : QwenTokenizer::qwen35_pretokenize(s)) {
            out.emplace_back(s.substr(sp.begin, sp.end - sp.begin));
        }
        return out;
    };
    using V = luisa::vector<luisa::string>;
    // \p{N} beyond ASCII: Arabic-Indic digits are single-number pre-tokens.
    expect(spans_of("\xd9\xa1\xd9\xa2") == V{"\xd9\xa1", "\xd9\xa2"});// ١٢
    // Devanagari combining marks join the letter run (\p{M}).
    expect(spans_of("\xe0\xa4\x95\xe0\xa4\xbe") == V{"\xe0\xa4\x95\xe0\xa4\xbe"});// का
    // Emoji are symbols (not letters): one symbol run.
    expect(spans_of("\xf0\x9f\x98\x80\xf0\x9f\x98\x81") ==
           V{"\xf0\x9f\x98\x80\xf0\x9f\x98\x81"});
    // CRLF runs attach to the preceding symbol run ([\r\n]*).
    expect(spans_of("a!\r\nb") == V{"a", "!\r\n", "b"});
    // Mixed whitespace: \s*[\r\n]+ ends after the last CR/LF.
    expect(spans_of("a \t\n b") == V{"a", " \t\n", " b"});
    // NBSP is \s but not CR/LF: \s+(?!\S) leaves one for the next token.
    expect(spans_of("a\xc2\xa0\xc2\xa0" "b") == V{"a", "\xc2\xa0", "\xc2\xa0" "b"});
}

// Real GGUF tokenizer metadata (skipped when the file is absent). Loads the
// full vocabulary, validates counts/ids, round-trips text and checks the chat
// wrapper against the documented template shape.
void test_real_gguf_tokenizer() {
    const luisa::filesystem::path path = "data/Qwen3.5-4B-BF16.gguf";
    std::error_code ec;
    if (!luisa::filesystem::exists(path, ec)) {
        std::printf("  [skip] %s not present\n", path.string().c_str());
        return;
    }
    auto reader = GgufReader::open(path);
    expect(reader.ok());
    if (!reader.ok()) { return; }
    auto tok_r = QwenTokenizer::load(**reader.value, 248320u);
    expect(tok_r.ok());
    if (!tok_r.ok()) {
        std::printf("  load failed: %s (%s)\n", tok_r.status.message.c_str(),
                    tok_r.status.detail.c_str());
        return;
    }
    auto &tok = **tok_r.value;
    expect(eq(tok.vocab_size(), 248320ull));
    expect(eq(tok.info().merge_count, 247587ull));
    expect(eq(tok.eos_id(), 248046u));
    expect(eq(tok.pad_id(), 248055u));
    expect(tok.id_to_token(tok.eos_id()).value_or("") == "<|im_end|>");
    expect(tok.token_to_id("<|im_start|>").value_or(0u) == 248045u);
    expect(tok.token_to_id("<|endoftext|>").value_or(0u) == 248044u);
    // Round-trips across scripts/whitespace/emoji.
    const luisa::string samples[] = {
        "Hello, world!", "The quick brown fox jumps over the lazy dog.",
        "line1\nline2\r\n\tindented", "  spaces  everywhere  ",
        "na\xc3\xafve caf\xc3\xa9 r\xc3\xa9sum\xc3\xa9",
        "\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c\xef\xbc\x81",
        "\xf0\x9f\x98\x80 emoji \xf0\x9f\x8e\x89",
        "mix 12345 of 67890 digits and symbols!!!",
        "don't stop believing", ""};
    for (const auto &s : samples) {
        auto ids = tok.encode(s, false);
        expect(ids.ok());
        if (!ids.ok()) { continue; }
        auto text = tok.decode(*ids.value, true, false);
        expect(text.ok());
        if (text.ok()) { expect(*text.value == s); }
        // Incremental detokenization must agree with the batch decode.
        auto detok = tok.make_detokenizer(true);
        luisa::string streamed;
        for (auto id : *ids.value) {
            auto piece = detok.push(id);
            expect(piece.ok());
            if (!piece.ok()) { break; }
            streamed += *piece.value;
        }
        auto tail = detok.finish();
        expect(tail.ok());
        if (tail.ok()) { streamed += *tail.value; }
        expect(streamed == s);
    }
    // Chat wrapper: exact template shape, specials as single ids.
    auto chat = tok.chat_prompt_ids("Hello");
    expect(chat.ok());
    if (chat.ok()) {
        auto text = tok.decode(*chat.value, false, false);
        expect(text.ok());
        if (text.ok()) {
            const luisa::string expect_text =
                "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n\n\n";
            expect(*text.value == expect_text);
        }
        expect((*chat.value).front() == 248045u);// <|im_start|>
    }
    // EOS stops the incremental stream.
    auto detok = tok.make_detokenizer(true);
    auto hello = tok.encode("hi", false);
    expect(hello.ok());
    if (hello.ok()) {
        for (auto id : *hello.value) { expect(detok.push(id).ok()); }
        expect(detok.push(tok.eos_id()).ok());
        expect(detok.stopped());
        expect(detok.text() == "hi");
    }
    // Differential encode vs an independent Python reference (exact qwen35
    // regex + rank-based BPE over this same vocab). Every id must match.
    for (const auto &v : test_vectors::kEncodeVectors) {
        auto ids = tok.encode(v.text, false);
        expect(ids.ok());
        if (!ids.ok()) { continue; }
        const bool same = ids.value->size() == v.count &&
                          std::equal(ids.value->begin(), ids.value->end(), v.ids);
        if (!same) {
            std::printf("  differential mismatch for '%s': got %zu ids, expect %u\n",
                        v.text, ids.value->size(), v.count);
        }
        expect(same);
    }
}

// ---------------------------------------------------------------------------
// Device-free operator self-tests (both kernel families).
// ---------------------------------------------------------------------------

void test_dsl_ops_self_test() {
    luisa::vector<luisa::string> report;
    const auto s = dsl::dsl_ops_self_test(&report);
    if (!s.ok()) {
        std::printf("  dsl self-test failed: %s\n", s.message.c_str());
        for (const auto &line : report) { std::printf("  %s\n", line.c_str()); }
    }
    expect(s.ok());
    // The fused residual-snapshot norm oracle must be covered explicitly.
    bool snapshot_covered = false;
    for (const auto &line : report) {
        if (line.find("rmsnorm snapshot oracle") != luisa::string::npos) { snapshot_covered = true; }
    }
    expect(snapshot_covered);
}

void test_tile_ops_self_test() {
    luisa::vector<luisa::string> report;
    const auto r = tile_ops::tile_ops_self_test(&report);
    if (!r.ok()) {
        std::printf("  tile self-test failed: %s\n", r.message.c_str());
        for (const auto &line : report) { std::printf("  %s\n", line.c_str()); }
    }
    expect(r.ok());
}

}// namespace

static auto test_qwen_engine_registration = [] {
    "qwen_kernel_kind_names"_test = [] { test_kernel_kind_names(); };
    "qwen_argmax"_test = [] { test_argmax(); };
    "qwen_temperature_sampling"_test = [] { test_temperature_sampling(); };
    "qwen_create_invalid_handles"_test = [] { test_create_invalid_handles(); };
    "qwen_backend_caps_defaults"_test = [] { test_backend_caps_defaults(); };
    "qwen_tokenizer_self_test"_test = [] { test_tokenizer_self_test(); };
    "qwen35_pretokenizer_unicode"_test = [] { test_qwen35_pretokenizer_unicode(); };
    "qwen_real_gguf_tokenizer"_test = [] { test_real_gguf_tokenizer(); };
    "qwen_dsl_ops_self_test"_test = [] { test_dsl_ops_self_test(); };
    "qwen_tile_ops_self_test"_test = [] { test_tile_ops_self_test(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}
