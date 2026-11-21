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
#include "qwen/cuda_graph_while.h"
#include "qwen/cuda_tile_gemm.h"
#include "qwen/engine.h"
#include "qwen/gguf.h"
#include "qwen/prompt_ids.h"
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

// opt_plan/09 N4.2: the conditional-WHILE chunk is REFUSED before any device work
// when the request cannot be honoured. These are the host-side negative paths the
// engine and the CLI both rely on (qwen_validate_while_request is the single gate).
void test_while_request_validation() {
    QwenWhileRequest request;
    request.requested = true;
    request.chunk = 4u;
    request.eos_token = 100u;
    request.vocab = 1000u;
    request.position = 10u;
    request.max_context = 64u;
    expect(qwen_validate_while_request(request).ok()) << "a bounded request with an EOS id is valid";
    expect(!request.eos_enabled() == false) << "the EOS id is inside the vocabulary";
    // Not requested: validation is a no-op even with nonsense values.
    auto off = request;
    off.requested = false;
    off.chunk = 0u;
    off.eos_token = 0xFFFFFFFFu;
    expect(qwen_validate_while_request(off).ok()) << "a disabled request validates trivially";
    // No EOS id: refused (unsupported) -- a chunk could otherwise run past EOS.
    auto no_eos = request;
    no_eos.eos_token = 0xFFFFFFFFu;
    const auto no_eos_status = qwen_validate_while_request(no_eos);
    expect(!no_eos_status.ok()) << "a missing EOS id is refused";
    expect(no_eos_status.code == StatusCode::unsupported) << "the refusal is 'unsupported'";
    // EOS id outside the vocabulary is refused as well.
    auto bad_eos = request;
    bad_eos.eos_token = 1000u;
    expect(!qwen_validate_while_request(bad_eos).ok()) << "an out-of-vocabulary EOS id is refused";
    // Zero chunk / zero vocab / zero capacity / exhausted context.
    auto zero_chunk = request;
    zero_chunk.chunk = 0u;
    expect(!qwen_validate_while_request(zero_chunk).ok()) << "a zero chunk is refused";
    auto zero_vocab = request;
    zero_vocab.vocab = 0u;
    expect(!qwen_validate_while_request(zero_vocab).ok()) << "a zero vocabulary is refused";
    auto zero_ctx = request;
    zero_ctx.max_context = 0u;
    expect(!qwen_validate_while_request(zero_ctx).ok()) << "a zero context capacity is refused";
    auto at_end = request;
    at_end.position = 64u;
    expect(!qwen_validate_while_request(at_end).ok()) << "a position at max_context is refused";
    auto past_room = request;
    past_room.position = 62u;
    expect(!qwen_validate_while_request(past_room).ok())
        << "a chunk longer than the remaining context is refused";
    auto exact_room = request;
    exact_room.position = 60u;
    expect(qwen_validate_while_request(exact_room).ok())
        << "a chunk that exactly fills the remaining context is accepted";
    // The summary is the harness-parsable description both the engine and the CLI log.
    const auto summary = qwen_while_request_summary(request);
    expect(summary.find("requested=1") != luisa::string::npos) << "the summary states the request";
    expect(summary.find("chunk=4") != luisa::string::npos) << "the summary states the chunk";
    expect(summary.find("eos=100") != luisa::string::npos) << "the summary states the EOS id";
    // A probe that never ran reports an explicit refusal instead of a silent success.
    const QwenCudaWhileProbe never;
    expect(!never.supported) << "a probe that never ran is not 'supported'";
    expect(never.summary().find("supported=0") != luisa::string::npos)
        << "an unrun probe says so";
    // Chunk results and PDL status have harness-parsable summaries (no device needed).
    QwenChunkResult chunk;
    chunk.requested = 8u;
    chunk.executed = 5u;
    chunk.eos = true;
    expect(chunk.summary().find("executed=5") != luisa::string::npos)
        << "the chunk summary reports the MEASURED executed count";
    expect(chunk.summary().find("path=per-token-fallback") == luisa::string::npos)
        << "a WHILE-served chunk is not labelled a fallback";
    chunk.fallback = true;
    expect(chunk.summary().find("path=per-token-fallback") != luisa::string::npos)
        << "a fallback chunk says so";
    QwenPdlStatus pdl;
    pdl.requested = true;
    pdl.reason = "not available on this device";
    expect(pdl.summary().find("supported=0") != luisa::string::npos)
        << "the PDL summary reports the probe, not a claim";
    expect(pdl.summary().find("not available on this device") != luisa::string::npos)
        << "the PDL summary carries the reason";
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
// engine.h: the EXPERIMENTAL CUDA native-Tile typed-FP8 prefill GEMM is OPT-IN
// ---------------------------------------------------------------------------
// The integration contract is "default OFF, never auto-selected, no performance
// claim": with default options the engine must probe nothing, JIT nothing and keep
// the measured native NVRTC / DSL MT chain, and a default QwenEngineInfo must say
// "not engaged" instead of reporting zeros that read like a measurement. These are
// the device-free half of that contract; the helper's own gates (config validation,
// the cuda/LUISA_CUDA_TILE_IR requirement, the typed-FP8 compile probe) are covered
// by test_qwen_native_gemm and by an explicit --cuda-tile-fp8 run.
void test_cuda_tile_prefill_defaults() {
    const QwenEngineOptions options;
    expect(!options.cuda_tile_fp8_prefill);
    expect(options.cuda_tile_m == 64u);
    expect(options.cuda_tile_n == 64u);
    // The 64x64 default is a VALID request once opted in (so opting in is not a
    // guaranteed failure), and the prefill copy of a bf16 session is refused by the
    // format gate: those two answers are what create() reports.
    dsl::QwenCudaTileGemmConfig cfg{};
    cfg.tile_m = options.cuda_tile_m;
    cfg.tile_n = options.cuda_tile_n;
    expect(dsl::validate_qwen_cuda_tile_gemm_config(cfg).ok());
    expect(!dsl::qwen_cuda_tile_gemm_supported_format(dsl::QwenWeightFormat::bf16));
    expect(!dsl::qwen_cuda_tile_gemm_supported_format(dsl::QwenWeightFormat::q8_0));
    expect(!dsl::qwen_cuda_tile_gemm_supported_format(dsl::QwenWeightFormat::w4));
    expect(dsl::qwen_cuda_tile_gemm_supported_format(dsl::QwenWeightFormat::fp8_e4m3));
    const QwenEngineInfo info;
    expect(!info.cuda_tile_prefill_gemm);
    expect(info.cuda_tile_prefill_gemm_m == 0u);
    expect(info.cuda_tile_prefill_gemm_n == 0u);
    expect(info.cuda_tile_prefill_gemm_k == 0u);
    expect(info.cuda_tile_prefill_gemm_min_rows == 0u);
    expect(info.cuda_tile_prefill_gemm_launches == 0u);
    expect(info.cuda_tile_prefill_gemm_compiles == 0u);
    expect(info.cuda_tile_prefill_gemm_cache_hits == 0u);
    expect(info.cuda_tile_prefill_gemm_shaders == 0u);
    expect(info.cuda_tile_prefill_gemm_jit_ms == 0.0);
    expect(info.cuda_tile_prefill_gemm_probe_ms == 0.0);
    expect(!info.cuda_tile_prefill_gemm_ragged_n);
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
    // opt_plan/04 section 7 additions: architecture/toolkit, L2, clusters and
    // the VK allocation count. Value-initialized they are the "unknown" answer,
    // which is what an unqueryable capability must report -- never a guess.
    expect(caps.total_memory_bytes == 0ull);
    expect(caps.cuda_arch.empty());
    expect(caps.cuda_toolkit_version == 0u);
    expect(caps.l2_cache_bytes == 0u);
    expect(!caps.cluster_launch);
    expect(!caps.cluster_launch_control);
    expect(caps.vk_max_memory_allocation_count == 0u);
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
    // Phase 9c LCV-3: "enabled" (a split-barrier kernel can compile) and "used"
    // (a compiled shader already declared SPV_EXT_split_barrier) are separate.
    expect(!caps.split_barrier_enabled);
    expect(!caps.cooperative_vector);
    // Phase 9c LCV-1: "enabled" and "usable" are separate answers. A record with
    // nothing known reports neither, and the configuration count is 0 (extension
    // absent), never a guessed count.
    expect(!caps.cooperative_vector_fp32_config);
    expect(!caps.cooperative_vector_usable);
    expect(caps.cooperative_vector_config_count == 0u);
    expect(!caps.cooperative_matrix);
    expect(caps.cooperative_matrix_extension.empty());
    expect(caps.notes.empty());
    // The conjunction rule the cooperative-vector gate must read:
    // usable == enabled && fp32 config, so a record can never claim usability
    // without the extension.
    {
        QwenBackendCaps enabled_only{};
        enabled_only.cooperative_vector = true;
        expect(!enabled_only.cooperative_vector_usable);
        QwenBackendCaps full{};
        full.cooperative_vector = true;
        full.cooperative_vector_fp32_config = true;
        full.cooperative_vector_usable =
            full.cooperative_vector && full.cooperative_vector_fp32_config;
        expect(full.cooperative_vector_usable);
    }

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

    // Tile schedule record (phase 9d): a default-constructed info reports the
    // conservative state (no chunk conv, no bucket, nothing compiled), and the
    // reported lazy-compile count is what a TTFT number must be read against --
    // it is not a boolean "bucketing on".
    expect(!info.tile_chunk_conv);
    expect(info.tile_attention_bucket_keys == 0u);
    expect(info.tile_attention_buckets == 0u);
    expect(info.tile_attention_precompiled == 0u);
    expect(info.tile_attention_lazy_compiles == 0u);
    info.tile_chunk_conv = true;
    info.tile_attention_bucket_keys = 512u;
    info.tile_attention_buckets = 8u;
    info.tile_attention_precompiled = 23u;
    info.tile_attention_lazy_compiles = 2u;
    expect(info.tile_chunk_conv && info.tile_attention_bucket_keys == 512u);
    expect(info.tile_attention_precompiled + info.tile_attention_lazy_compiles <=
           info.tile_attention_buckets * 2u + 64u);
    // Fused one-CTA GDN recurrent+norm record (opt_plan/03 section 5, phase 9e): a
    // default-constructed info reports the MEASURED default (not fused, i.e. the
    // decode4 recurrence + the separate gated norm kernel) and an empty reason, so a
    // log can distinguish "not requested" from "requested but refused".
    expect(!info.gdn_recurrent_norm);
    expect(info.gdn_recurrent_norm_reason.empty());
    info.gdn_recurrent_norm = true;
    info.gdn_recurrent_norm_reason = "fused one-CTA-per-head recurrent + gated norm (1 launch)";
    expect(info.gdn_recurrent_norm && !info.gdn_recurrent_norm_reason.empty());
    info.gdn_recurrent_norm = false;
    // The option is OPT-IN: the default option set keeps the existing decode
    // schedule byte for byte (mixing the fused kernel in would change both the
    // reduction order and the launch count of every GDN layer).
    const QwenEngineOptions option_defaults{};
    expect(!option_defaults.dsl_gdn_recurrent_norm);
    expect(option_defaults.dsl_gdn_decode_fused);
    expect(option_defaults.dsl_decode_specialized);
    // The probe itself must be callable without a device and must never throw:
    // it reports the false/0 default and explains it in `notes`.
    compute::Device no_device;
    const auto probed = qwen_probe_backend_caps(no_device);
    expect(probed.backend_name.empty());
    expect(!probed.sparse_buffer);
    expect(!probed.cuda_graph);
    expect(!probed.pdl);
    expect(probed.kv_page_bytes == 0u);
    expect(probed.cuda_arch.empty());
    expect(probed.l2_cache_bytes == 0u);
    expect(!probed.cluster_launch);
    expect(probed.vk_max_memory_allocation_count == 0u);
    expect(!probed.split_barrier_enabled);
    expect(!probed.cooperative_vector_fp32_config);
    expect(!probed.cooperative_vector_usable);
    expect(probed.cooperative_vector_config_count == 0u);
    expect(probed.cooperative_matrix_extension.empty());
    expect(!probed.notes.empty());

      // Logging a fully-unknown record must not crash (it is the first artifact of
      // an onboarding log, 04 section 7).
      qwen_log_backend_caps(probed);
  }

  // ---------------------------------------------------------------------------
  // gate 11 blocks/SM arithmetic (device-free half)
  //
  // The CUDA backend answers the SM-count query (patches/luisa-cuda-multiprocessor-
  // query.patch), so QwenBackendCaps::sm_count is now a measured value on cuda and
  // still 0/"unqueryable" on dx/vk. Both cases go through ONE formula
  // (caps.h qwen_decode_attn_blocks_per_sm) that QwenEngineInfo carries, which is
  // pinned here without a device: the engine's own numbers for this box
  // (176 split blocks, 16 legacy blocks, 80 SMs -> 2.2 / 0.2) plus the fail-closed
  // 0 for a backend that cannot report a count.
  // ---------------------------------------------------------------------------
  void test_blocks_per_sm_formula() {
      QwenBackendCaps caps{};
      expect(caps.sm_count == 0u);
      expect(caps.sm_count_source.empty());
      // The measured target-box case: RTX 4080 SUPER (AD103) = 80 SMs.
      expect(std::fabs(qwen_decode_attn_blocks_per_sm(176u, 80u) - 2.2f) < 1e-6f);
      expect(std::fabs(qwen_decode_attn_blocks_per_sm(16u, 80u) - 0.2f) < 1e-6f);
      // A smaller report scales the same ratio; no count is ever inferred.
      expect(std::fabs(qwen_decode_attn_blocks_per_sm(176u, 176u) - 1.0f) < 1e-6f);
      // "not queryable" (0) and "no blocks" both keep the ratio at 0 -- the report
      // then says "occupancy evidence unavailable" instead of dividing by a guess,
      // and a 0 can never become inf/NaN.
      expect(qwen_decode_attn_blocks_per_sm(176u, 0u) == 0.0f);
      expect(qwen_decode_attn_blocks_per_sm(0u, 80u) == 0.0f);
      expect(std::isfinite(qwen_decode_attn_blocks_per_sm(176u, 0u)));
      // The split bound the SM count now feeds: 2 * 80 / 4 = 40 = the cap, so a cuda
      // run on this box selects the same grid as the documented fail-closed default
      // (dsl_ops.h). The query therefore cannot move an id or a captured graph.
      expect(dsl::qwen_attn_decode_split_bound(80u, 4u) ==
             dsl::qwen_attn_decode_split_default_bound);
      expect(dsl::qwen_attn_decode_split_bound(0u, 4u) ==
             dsl::qwen_attn_decode_split_default_bound);
      // Fewer SMs shrink the bound instead of saturating it.
      expect(dsl::qwen_attn_decode_split_bound(40u, 4u) == 20u);
  }
// ---------------------------------------------------------------------------
// prompt_ids.h: the CLI half of --synthetic-prompt-len (opt_plan/09 N4.1)
// ---------------------------------------------------------------------------
//
// scripts/benchmark_qwen.py synthesizes a sweep's ids in Python and the CLI can
// regenerate them from a length alone (Windows caps a command line at 32767
// characters, so an 8192-token id list cannot travel as --tokens). The two
// generators must be identical or a stored arm silently becomes a different
// prompt; scripts/test_benchmark_qwen.py checks the constants and the rule, and
// this test pins the CLI's actual output against ids generated by that Python
// implementation.
void test_synthetic_prompt_ids() {
    // Length 0 is the caller's bad-argument case, not a prompt.
    expect(qwen_synthetic_prompt_ids(0u).empty());

    const auto one = qwen_synthetic_prompt_ids(1u);
    expect(one.size() == 1u);
    expect(one[0] == 107095u);

    const auto eight = qwen_synthetic_prompt_ids(8u);
    const uint32_t expected8[] = {175247u, 131190u, 51569u, 111130u,
                                  169674u, 137708u, 177013u, 140025u};
    expect(eight.size() == 8u);
    for (size_t i = 0; i < 8u; i++) { expect(eight[i] == expected8[i]); }

    const auto long_ids = qwen_synthetic_prompt_ids(128u);
    const uint32_t expected128[] = {48926u, 34738u, 173974u, 104920u,
                                    181297u, 75053u, 43680u, 196828u};
    expect(long_ids.size() == 128u);
    for (size_t i = 0; i < 8u; i++) { expect(long_ids[i] == expected128[i]); }
    expect(long_ids.back() == 1827u);

    // Every id stays inside the documented range, which is inside the
    // Qwen3.5-4B vocabulary the CLI validates against.
    for (uint32_t length : {1u, 2u, 7u, 64u, 512u, 8192u}) {
        const auto ids = qwen_synthetic_prompt_ids(length);
        expect(ids.size() == length);
        bool in_range = true;
        for (auto id : ids) {
            in_range &= id >= kSyntheticPromptIdMin && id < kSyntheticPromptIdMax;
        }
        expect(in_range);
    }
    expect(kSyntheticPromptIdMax <= kSyntheticPromptVocabSize);

    // The declared rule constants are the documented ones (scripts/
    // benchmark_qwen.py has the same six values; a change on one side without
    // the other is what this pair of tests exists to catch).
    expect(kSyntheticPromptSeed == 0x0C0FFEE5EEDBEEF01ull);
    expect(kSyntheticPromptLengthMix == 0xD1B54A32D192ED03ull);
    expect(kSyntheticPromptLcgMul == 6364136223846793005ull);
    expect(kSyntheticPromptLcgInc == 1442695040888963407ull);
    expect(kSyntheticPromptIdMin == 1000u);
    expect(kSyntheticPromptIdMax == 200000u);
}

// --- --tokens-file parsing (qwen_parse_prompt_ids) --------------------------
// The other half of opt_plan/09 N4.1: the CLI must accept the file the harness
// writes, must never silently truncate a prompt, and must point at the offending
// line when it refuses. All of it is host-only.
void test_prompt_id_parsing() {
    luisa::vector<uint32_t> ids;
    luisa::string error;

    expect(qwen_parse_prompt_ids("1,2,3", ids, error));
    expect(ids.size() == 3u && ids[0] == 1u && ids[2] == 3u);

    // Newlines, CRLF, tabs, spaces and commas are all separators (what a file
    // written by the harness or by hand actually looks like).
    expect(qwen_parse_prompt_ids(" 175247 131190\n51569\r\n111130\t169674 ", ids, error));
    expect(ids.size() == 5u && ids[0] == 175247u && ids[4] == 169674u);
    expect(qwen_parse_prompt_ids("1, 2 ,3\n\n4", ids, error));
    expect(ids.size() == 4u && ids[3] == 4u);

    // Empty / separator-only input fails instead of producing an empty prompt.
    expect(!qwen_parse_prompt_ids("", ids, error));
    expect(error.find("no token ids") != luisa::string::npos);
    expect(ids.empty());
    expect(!qwen_parse_prompt_ids(" , \n\t", ids, error));
    expect(ids.empty());

    // A bad chunk names its 1-based line and the text; the list is cleared, so
    // a caller can never run a half-parsed prompt.
    expect(!qwen_parse_prompt_ids("1,x,3", ids, error));
    expect(error.find("line 1") != luisa::string::npos);
    expect(error.find("'x'") != luisa::string::npos);
    expect(ids.empty());
    expect(!qwen_parse_prompt_ids("1\n2\nbad", ids, error));
    expect(error.find("line 3") != luisa::string::npos);
    expect(ids.empty());

    // Overflow is rejected (uint32 ids only) while UINT32_MAX itself is fine.
    expect(!qwen_parse_prompt_ids("4294967296", ids, error));
    expect(ids.empty());
    expect(qwen_parse_prompt_ids("4294967295", ids, error));
    expect(ids.size() == 1u && ids[0] == 4294967295u);
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

void test_tile_attention_bucket_policy() {
    using namespace luisa::qwen::tile_ops;
    // The shared policy the executor and the device test both use. Boundaries
    // are the whole point: kv_len 0/1/511/512/513 must land in 4096/512/512/
    // 512/1024, the capacity clamp must never truncate kv_len, and bucket 0 must
    // reproduce the historical full-capacity behaviour exactly.
    expect(tile_attention_bucket_capacity(0u, 512u, 4096u) == 4096u);
    expect(tile_attention_bucket_capacity(1u, 512u, 4096u) == 512u);
    expect(tile_attention_bucket_capacity(511u, 512u, 4096u) == 512u);
    expect(tile_attention_bucket_capacity(512u, 512u, 4096u) == 512u);
    expect(tile_attention_bucket_capacity(513u, 512u, 4096u) == 1024u);
    expect(tile_attention_bucket_capacity(1024u, 512u, 4096u) == 1024u);
    expect(tile_attention_bucket_capacity(1025u, 512u, 4096u) == 1536u);
    expect(tile_attention_bucket_capacity(4095u, 512u, 4096u) == 4096u);
    expect(tile_attention_bucket_capacity(4096u, 512u, 4096u) == 4096u);
    expect(tile_attention_bucket_capacity(4000u, 0u, 4096u) == 4096u);
    expect(tile_attention_bucket_capacity(600u, 512u, 1000u) == 1000u);
    expect(tile_attention_bucket_capacity(999u, 512u, 1000u) == 1000u);
    // Bucket count bound: at most ceil(capacity/bucket)+1 distinct capacities,
    // and every capacity covers its kv_len.
    uint32_t count = 0u;
    for (uint32_t kv = 1u; kv <= 4096u; kv++) {
        const auto cap = tile_attention_bucket_capacity(kv, 512u, 4096u);
        expect(cap >= kv && cap <= 4096u);
        if (kv == 1u || cap != tile_attention_bucket_capacity(kv - 1u, 512u, 4096u)) { count++; }
    }
    expect(count == 8u);
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
    "qwen_cuda_tile_prefill_defaults"_test = [] { test_cuda_tile_prefill_defaults(); };
          "qwen_backend_caps_defaults"_test = [] { test_backend_caps_defaults(); };
      "qwen_blocks_per_sm_formula"_test = [] { test_blocks_per_sm_formula(); };
    "qwen_synthetic_prompt_ids"_test = [] { test_synthetic_prompt_ids(); };
    "qwen_prompt_id_parsing"_test = [] { test_prompt_id_parsing(); };
    "qwen_tokenizer_self_test"_test = [] { test_tokenizer_self_test(); };
    "qwen35_pretokenizer_unicode"_test = [] { test_qwen35_pretokenizer_unicode(); };
    "qwen_real_gguf_tokenizer"_test = [] { test_real_gguf_tokenizer(); };
    "qwen_dsl_ops_self_test"_test = [] { test_dsl_ops_self_test(); };
    "qwen_while_request_validation"_test = [] { test_while_request_validation(); };
    "qwen_tile_ops_self_test"_test = [] { test_tile_ops_self_test(); };
 "qwen_tile_attention_bucket_policy"_test = [] { test_tile_attention_bucket_policy(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}
