// Checkpoint-free Qwen3.5-4B single-token packed GEMV benchmark.
//
// Unlike a repeated single-matrix launch, each timed batch walks independent
// matrix replicas (256 MiB target, 1024-replica cap; >64 MiB for every shipped
// shape) so the reported bandwidth is not merely an L2-resident best case on
// the RTX 4080 SUPER. The same packed buffers feed the generic and
// decode-specialized implementations. Recording is excluded; each measurement
// includes one batch submission and one final stream synchronization.
//
// --calibrate is the achievable-DRAM-bandwidth calibration of
// opt_plan/06-benchmark-plan.md section 3 step 3 and
// opt_plan/08-target-hardware.md section 7 step 7: a real 1 GiB device-to-device
// copy and a real FP32 triad (a = b + s*c) over cache-resistant buffers, >= 30
// iterations by default, reporting median/min/p95 and the resulting GB/s. That
// median is the only denominator any "percentage of roof" may use
// (benchmark_qwen.py --bandwidth). Both kernels are validated against the host
// pattern after timing, so a compiler that elided the work fails loudly instead
// of reporting an impossible bandwidth.
//
// --prefill-attention is the isolated causal GQA PREFILL ATTENTION arm of
// opt_plan/02 section 5/6 (T16/T32 gate): it runs the untiled reference kernel
// and the tiled FlashAttention-style kernel on the same buffers at S =
// 128/512/1024 (4096 optional), validates them against each other, and reports
// min/median wall time plus the achieved FP32 GFLOP/s. The prefill GEMM arm
// min/median wall time plus the achieved FP32 GFLOP/s.
//
// --prefill-gemm is the isolated PREFILL GEMM arm of opt_plan/02 section 3 and
// 09 N5.2 + N1a: for M = 64/256/1024 (plus extra model shapes) it runs the
// reference one-thread-per-(m,n) GEMM, the weight-reusing warp-per-column GEMM
// (make_qwen35_gemm_mt_kernel) and - on the CUDA backend - the tensor-core GEMM
// that NativeShaderExt/CUDA_NVRTC JIT-compiles (QwenNativeGemm,
// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32) over the SAME packed
// buffers, validates them against each other, and reports min/median wall time,
// the weight-traffic model (weight_bytes * ceil(M/pass_rows) per implementation,
// x read once, y written once) and the resulting effective GB/s. That traffic is
// a MODEL, not a hardware counter: it states how many times each implementation
// must stream the matrix, which is exactly the quantity each family changes.
// The tensor-core arm additionally reports its own cold NVRTC cost (never
// amortized into a timing) and TWO deviation metrics against the DSL reference:
// against a run on activations already rounded to BF16 (implementation error
// only, gated at 1e-4) and against the FP32-activation reference (the documented
// BF16 activation rounding). A skipped native row means the backend or the format
// has no realization; it is reported, never silently dropped.
//
// --cuda-tile-fp8 is an OPT-IN THIRD implementation inside --prefill-gemm: the
// CUDA native Tile IR typed-FP8 GEMM of qwen/cuda_tile_gemm.h
// (tile::compile, Lowering::NATIVE; requires the cuda backend, FP8 E4M3FN weights,
// K % 64 == 0 and the run-time opt-in LUISA_CUDA_TILE_IR=1). It reads the SAME
// packed uint32 FP8 weight buffer, the SAME FP32 group scales and the SAME FP32
// activation buffer as the DSL and native rows - no repack, no dequantized copy -
// and prints one extra `cuda_tile` RESULT row per (shape, M) with the same
// weight-streaming traffic model (pass_rows = tile_m). Every (m, n, k) is a static
// Tile specialization, so the first build of each shape is a lazy compile: it is
// measured, printed on its own line and excluded from all samples, and the lazy
// compile count / JIT milliseconds are reported again in a per-run total.
// Without the flag nothing changes: no extra row, no extra compile, no extra
// buffer. With it, every row the arm cannot serve is an explicit `skipped` row
// with the verbatim reason (wrong weight format, non-CUDA backend, missing
// LUISA_CUDA_TILE_IR, K tail, or a ragged N that this target cannot compile) and
// the surrounding benchmark keeps running. Numerics are NOT gated on exactness
// against the DSL: the kernel rounds the activation to BF16 before its BF16
// tensor-core dot and applies the FP32 group scale to the partial dot AFTER the
// contraction, so the gate here is the established FP32-reference envelope
// (max|d|/max|ref| <= 1e-1, the same bound as kTileFp32Tol in
// tests/integration/qwen/test_qwen_native_gemm.cpp) plus finiteness; agreement
// with the bf16-x DSL oracle and the current native FP8 result are printed as
// information. Exactness of the decomposition is proven by the differential test,
// not by this benchmark.
// Usage:
//   benchmark_qwen_decode <backend> [--format both|bf16|fp8|q8|all]
//       [--impl both|generic|decode] [--iters N] [--lm-head]
//       [--ffn] [--attention] [--shape NAME]
//   benchmark_qwen_decode <backend> --prefill-gemm [--prefill-gemm-m 64,256,1024]
//       [--prefill-gemm-iters N] [--format ...] [--cuda-tile-fp8]
//       [--cuda-tile-m N] [--cuda-tile-n N]
//   benchmark_qwen_decode <backend> --calibrate
//       [--calibrate-iters N] [--calibrate-copy-mib N] [--calibrate-triad-mib N]
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

#include "qwen/cuda_tile_gemm.h"
#include "qwen/dsl_ops.h"
#include "qwen/native_attn.h"
#include "qwen/native_gemm.h"

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
      // OFFLINE MERGED input projection widths (opt_plan/03 section 3 + D5): the
      // same per-row decode GEMV work as 4 (GDN: attn_qkv+attn_gate+ssm_alpha+
      // ssm_beta) / 3 (attention: attn_q+attn_k+attn_v) separate launches, in ONE
      // launch over the row-concatenated matrix (qwen/projection_pack.h). Comparing
      // these rows with the sum of the segment rows isolates what the merge changes:
      // the bytes are identical, only the launch count and the per-launch scheduling
      // differ. The merged kernels' segment routing/epilogue is not part of this
      // arm (it is one branch per warp), so read this as the weight-stream floor.
      {"gdn_qkv_z_ab_merged", 12352u, 2560u},
      {"attn_qkv_merged", 10240u, 2560u},
  };
constexpr Shape kLmHead{"lm_head", 248320u, 2560u};
  // Canonical per-row scale group along K: 64 for FP8 E4M3, 32 for native Q8_0
  // (the GGUF block), 128 for W4A16 (the pack's canonical clipping group).
  // format_group(format) stays the FP8 value for every existing arm.
constexpr uint32_t kGroup = 64u;
[[nodiscard]] constexpr uint32_t format_group(QwenWeightFormat format) noexcept {
          return format == QwenWeightFormat::q8_0 ? kQwenQ8BlockSize
             : format == QwenWeightFormat::w4  ? kQwenW4DefaultGroup
                                               : kGroup;
}
  // 4 codes per word for the 8-bit formats, 8 for W4, 2 for bf16.
  [[nodiscard]] constexpr uint32_t format_pack(QwenWeightFormat format) noexcept {
      return format == QwenWeightFormat::bf16 ? 2u
             : format == QwenWeightFormat::w4  ? 8u
                                               : 4u;
  }
[[nodiscard]] constexpr bool format_scaled(QwenWeightFormat format) noexcept {
    return format != QwenWeightFormat::bf16;
}
[[nodiscard]] constexpr const char *format_label(QwenWeightFormat format) noexcept {
          return format == QwenWeightFormat::bf16 ? "bf16"
             : format == QwenWeightFormat::fp8_e4m3 ? "fp8"
             : format == QwenWeightFormat::w4 ? "w4"
                                              : "q8_0";
}
constexpr size_t kTargetWeightBytes = 256u * 1024u * 1024u;
constexpr size_t kMaxReplicas = 1024u;

struct Options {
    bool bf16{true};
    bool fp8{true};
    // Native Q8_0 weight storage (opt_plan/05 Q1): int8 block codes with one
    // FP32 scale per 32 K elements. Opt-in (--format q8 or --format all); the
          // default arms stay bf16+fp8.
      bool q8{false};
      // W4A16 group-wise weights (opt_plan/05 Q2): 8 signed int4 codes per word +
      // one FP32 scale per 128 K elements. Opt-in (--format w4 or all); the byte
      // model counts the codes AND the scales, so the reported GB/s is comparable
      // with the 8-bit arms.
      bool w4{false};
    bool generic{true};
    bool decode{true};
    // --impl packed: the 128-bit (uint4/float4) decode realizations of opt_plan/09 N2.
    bool packed{false};
    bool attention{false};
    bool lm_head{false};
    // --prefill-attention: isolated causal GQA prefill attention arm.
    // Lengths, tiling (0 = device-selected) and iteration override of that arm.
    bool prefill_attention{false};
    // --prefill-gemm: isolated prefill GEMM arm (opt_plan/02 section 3).
    bool prefill_gemm{false};
    luisa::vector<uint32_t> prefill_gemm_m{64u, 256u, 1024u};
    uint32_t prefill_gemm_iters{0u};
    // --cuda-tile-fp8: OPT-IN CUDA native Tile IR FP8 GEMM arm INSIDE the prefill
    // GEMM mode (qwen/cuda_tile_gemm.h, tile::compile(Lowering::NATIVE)). Off by
    // default, so an existing invocation runs the same kernels and prints the same
    // rows. `cuda_tile_m`/`cuda_tile_n` of 0 keep the helper's own 64x64x64 default;
    // a non-zero override is validated with validate_qwen_cuda_tile_gemm_config
    // (power of two, [16,128], tile volume budget), so a bad value is a usage error
    // instead of a silently clamped tile.
    bool cuda_tile_fp8{false};
    uint32_t cuda_tile_m{0u};
    uint32_t cuda_tile_n{0u};
    luisa::vector<uint32_t> prefill_attn_lengths{128u, 512u, 1024u, 4096u};
    // --decode-attention: isolated SINGLE-QUERY decode attention arm
    // (opt_plan/03 section 4.1/4.2): legacy per-q_head kernel vs the split-KV +
    // GQA-dedup family, at the plan's context set. Splits/context, k_tile and
    // max_splits default to the engine's selection rules (0 = device-selected).
    bool decode_attention{false};
    luisa::vector<uint32_t> decode_attn_lengths{128u, 512u, 2048u, 4096u, 8192u, 32768u};
    uint32_t decode_attn_k_tile{0u};
    uint32_t decode_attn_max_splits{0u};
    uint32_t decode_attn_iters{0u};
    // --decode-attn-impl legacy|split|both: run only one realization. Both share
    // the same CUDA kernel symbol (kernel_main), so Nsight Compute can only
    // attribute a launch series to one implementation if the OTHER one does not
    // run in the same process.
    bool decode_attn_impl_legacy{true};
    bool decode_attn_impl_split{true};
    uint32_t prefill_attn_q_tile{0u};
    uint32_t prefill_attn_k_tile{0u};
    uint32_t prefill_attn_iters{0u};
    bool ffn{false};
        // --rmsnorm: isolated zero-centered RMSNorm arm (opt_plan/09 N3.1):
        // generic warp-per-row / wide kernels vs the register-cached one-pass
        // model specialization.
        bool rmsnorm{false};
        luisa::vector<uint32_t> rmsnorm_rows{1u, 4u};
        uint32_t rmsnorm_iters{0u};
        // --gdn-chunk: isolated WY/UT chunk-64 arm (opt_plan/09 V4): generic
        // runtime-extent head dimension vs the compile-time head_dim = 128
        // model specialization.
        bool gdn_chunk{false};
        luisa::vector<uint32_t> gdn_chunk_lengths{1u, 63u, 64u, 65u, 127u, 128u, 129u, 512u};
        uint32_t gdn_chunk_iters{0u};
        // --gdn-decode: isolated DECODE GDN arm (opt_plan/09 N3.3): the reference
        // two-launch pair (conv+SiLU, then QK L2 norm) vs the fused single launch.
        bool gdn_decode{false};
        uint32_t gdn_decode_iters{0u};
        uint32_t gdn_decode_positions{64u};
        // --gdn-gates: isolated T=1 alpha/beta gate GEMV arm (opt_plan/09 N2): the
        // 32-bit LUT realization vs the 128-bit branchless-LUT-free one.
        bool gdn_gates{false};
        uint32_t gdn_gates_iters{0u};
        // --gdn-recurrent-norm: isolated T=1 GDN recurrent + gated output RMSNorm arm
        // (opt_plan/03 section 5, phase 9e). Three realizations of the SAME decode step:
        //   pair_one_cta : one-CTA-per-head recurrence + the gated norm kernel (2 launches)
        //   fused        : the fused one-CTA recurrence + norm kernel        (1 launch)
        //   pair_decode4 : 4-CTA-per-head recurrence + the gated norm kernel (2 launches)
        // The launch count and the global bytes each chain moves are printed next to the
        // time, because that -- not the kernel body -- is what the fusion changes.
        bool gdn_recurrent_norm{false};
        uint32_t gdn_recurrent_norm_iters{0u};
        uint32_t gdn_recurrent_norm_steps{24u};
    bool calibrate{false};
    std::string shape;
    uint32_t iters{7};
    // --calibrate: >= 30 iterations is the plan's requirement; the default is
    // the requirement, not the shape sweep's 7 (which is a latency probe).
    uint32_t calibrate_iters{30};
    // One copy of this many bytes moves 2x this in DRAM (read + write); the
    // 512 MiB default therefore moves exactly the plan's 1 GiB per iteration.
    uint64_t calibrate_copy_mib{512};
    // The triad's three arrays are this many MiB each (3x traffic per
    // iteration); 256 MiB each keeps the working set far above the 64 MiB L2.
    uint64_t calibrate_triad_mib{256};
};

// Comma-separated positive integer list (the arm length/row overrides).
bool parse_uint_list(std::string_view text, luisa::vector<uint32_t> &out) {
    luisa::vector<uint32_t> parsed;
    size_t begin = 0u;
    const std::string t{text};
    for (;;) {
        const auto comma = t.find(',', begin);
        const auto piece = t.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
        if (piece.empty()) { return false; }
        const auto n = std::strtoul(piece.c_str(), nullptr, 10);
        if (n == 0u || n > 65536u) { return false; }
        parsed.emplace_back(static_cast<uint32_t>(n));
        if (comma == std::string::npos) { break; }
        begin = comma + 1u;
    }
    if (parsed.empty()) { return false; }
    out = std::move(parsed);
    return true;
}

// Weight formats the run enabled, in the historical order (bf16, fp8) with the
// opt-in native Q8_0 arm last, so an existing two-format run keeps its exact
// row order.
[[nodiscard]] luisa::vector<QwenWeightFormat> enabled_formats(const Options &o) {
    luisa::vector<QwenWeightFormat> formats;
    if (o.bf16) { formats.emplace_back(QwenWeightFormat::bf16); }
    if (o.fp8) { formats.emplace_back(QwenWeightFormat::fp8_e4m3); }
          if (o.q8) { formats.emplace_back(QwenWeightFormat::q8_0); }
      if (o.w4) { formats.emplace_back(QwenWeightFormat::w4); }
      return formats;
}

bool parse_options(int argc, char **argv, Options &o) {
    for (int i = 2; i < argc; ++i) {
        const std::string_view a{argv[i]};
        auto value = [&]() -> std::string_view {
            return i + 1 < argc ? std::string_view{argv[++i]} : std::string_view{};
        };
        if (a == "--format") {
            auto v = value();
            // "both" keeps its historical meaning (bf16+fp8, the two formats that
            // predate native Q8_0). Q8_0 is EXPLICIT: --format q8, or --format all
            // for the three-way comparison. A Q8_0 row is never mixed into a
            // "both" run, so an existing baseline invocation is unchanged.
            o.bf16 = v == "both" || v == "bf16" || v == "all";
            o.fp8 = v == "both" || v == "fp8" || v == "all";
                          o.q8 = v == "q8" || v == "q8_0" || v == "all";
              o.w4 = v == "w4" || v == "all";
              if (!o.bf16 && !o.fp8 && !o.q8 && !o.w4) { return false; }
        } else if (a == "--impl") {
            auto v = value();
            o.generic = v == "both" || v == "generic";
            o.decode = v == "both" || v == "decode" || v == "all" || v == "packed";
            o.packed = v == "packed" || v == "all";
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
        } else if (a == "--prefill-attention") {
            o.prefill_attention = true;
            } else if (a == "--decode-attention") {
                o.decode_attention = true;
            } else if (a == "--decode-attn-k-tile") {
                auto v = value();
                o.decode_attn_k_tile = static_cast<uint32_t>(std::strtoul(std::string{v}.c_str(), nullptr, 10));
                if (o.decode_attn_k_tile > 64u) { return false; }
            } else if (a == "--decode-attn-max-splits") {
                auto v = value();
                o.decode_attn_max_splits = static_cast<uint32_t>(std::strtoul(std::string{v}.c_str(), nullptr, 10));
                if (o.decode_attn_max_splits > 64u) { return false; }
            } else if (a == "--decode-attn-impl") {
            auto v = value();
            if (v == "legacy") {
                o.decode_attn_impl_legacy = true;
                o.decode_attn_impl_split = false;
            } else if (v == "split") {
                o.decode_attn_impl_legacy = false;
                o.decode_attn_impl_split = true;
            } else if (v == "both") {
                o.decode_attn_impl_legacy = true;
                o.decode_attn_impl_split = true;
            } else {
                return false;
            }
        } else if (a == "--decode-attn-iters") {
                auto v = value();
                o.decode_attn_iters = static_cast<uint32_t>(std::strtoul(std::string{v}.c_str(), nullptr, 10));
                if (o.decode_attn_iters == 0u || o.decode_attn_iters > 1000u) { return false; }
            } else if (a == "--decode-attn-lengths") {
                auto v = value();
                luisa::vector<uint32_t> parsed;
                const std::string text{v};
                size_t begin = 0u;
                for (;;) {
                    const auto comma = text.find(',', begin);
                    const auto piece = text.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
                    if (piece.empty()) { return false; }
                    const auto n = std::strtoul(piece.c_str(), nullptr, 10);
                    if (n == 0u || n > 262144u) { return false; }
                    parsed.emplace_back(static_cast<uint32_t>(n));
                    if (comma == std::string::npos) { break; }
                    begin = comma + 1u;
                }
                if (parsed.empty()) { return false; }
                o.decode_attn_lengths = std::move(parsed);
        } else if (a == "--prefill-gemm") {
            o.prefill_gemm = true;
        } else if (a == "--prefill-gemm-iters") {
            auto v = value();
            if (v.empty()) { return false; }
            o.prefill_gemm_iters = static_cast<uint32_t>(std::strtoul(std::string{v}.c_str(), nullptr, 10));
            if (o.prefill_gemm_iters == 0u || o.prefill_gemm_iters > 1000u) { return false; }
        } else if (a == "--prefill-gemm-m") {
            auto v = value();
            luisa::vector<uint32_t> parsed;
            const std::string text{v};
            size_t begin = 0u;
            for (;;) {
                const auto comma = text.find(',', begin);
                const auto piece = text.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
                if (piece.empty()) { return false; }
                const auto n = std::strtoul(piece.c_str(), nullptr, 10);
                if (n == 0u || n > 4096u) { return false; }
                parsed.emplace_back(static_cast<uint32_t>(n));
                if (comma == std::string::npos) { break; }
                begin = comma + 1u;
            }
            if (parsed.empty()) { return false; }
            o.prefill_gemm_m = std::move(parsed);
        } else if (a == "--cuda-tile-fp8") {
            // The arm lives inside the prefill GEMM mode, so the flag implies it
            // instead of being silently ignored without --prefill-gemm.
            o.cuda_tile_fp8 = true;
            o.prefill_gemm = true;
        } else if (a == "--cuda-tile-m") {
            auto v = value();
            if (v.empty()) { return false; }
            o.cuda_tile_m = static_cast<uint32_t>(std::strtoul(std::string{v}.c_str(), nullptr, 10));
            if (o.cuda_tile_m == 0u) { return false; }
        } else if (a == "--cuda-tile-n") {
            auto v = value();
            if (v.empty()) { return false; }
            o.cuda_tile_n = static_cast<uint32_t>(std::strtoul(std::string{v}.c_str(), nullptr, 10));
            if (o.cuda_tile_n == 0u) { return false; }
        } else if (a == "--prefill-attn-q-tile") {
            auto v = value();
            if (v.empty()) { return false; }
            o.prefill_attn_q_tile = static_cast<uint32_t>(std::strtoul(std::string{v}.c_str(), nullptr, 10));
            if (o.prefill_attn_q_tile == 0u || o.prefill_attn_q_tile > 8u) { return false; }
        } else if (a == "--prefill-attn-k-tile") {
            auto v = value();
            if (v.empty()) { return false; }
            o.prefill_attn_k_tile = static_cast<uint32_t>(std::strtoul(std::string{v}.c_str(), nullptr, 10));
            if (o.prefill_attn_k_tile == 0u || o.prefill_attn_k_tile > 64u) { return false; }
        } else if (a == "--prefill-attn-iters") {
            auto v = value();
            if (v.empty()) { return false; }
            o.prefill_attn_iters = static_cast<uint32_t>(std::strtoul(std::string{v}.c_str(), nullptr, 10));
            if (o.prefill_attn_iters == 0u || o.prefill_attn_iters > 1000u) { return false; }
        } else if (a == "--prefill-attn-lengths") {
            auto v = value();
            if (v.empty()) { return false; }
            luisa::vector<uint32_t> parsed;
            const std::string text{v};
            size_t begin = 0u;
            for (;;) {
                const auto comma = text.find(',', begin);
                const auto piece = text.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
                if (piece.empty()) { return false; }
                const auto n = std::strtoul(piece.c_str(), nullptr, 10);
                if (n == 0u || n > 32768u) { return false; }
                parsed.emplace_back(static_cast<uint32_t>(n));
                if (comma == std::string::npos) { break; }
                begin = comma + 1u;
            }
            if (parsed.empty()) { return false; }
            o.prefill_attn_lengths = std::move(parsed);
        } else if (a == "--ffn") {
            o.ffn = true;
        } else if (a == "--rmsnorm") {
            o.rmsnorm = true;
        } else if (a == "--rmsnorm-rows") {
            if (!parse_uint_list(value(), o.rmsnorm_rows)) { return false; }
        } else if (a == "--rmsnorm-iters") {
            o.rmsnorm_iters = static_cast<uint32_t>(std::strtoul(std::string{value()}.c_str(), nullptr, 10));
        } else if (a == "--gdn-chunk") {
            o.gdn_chunk = true;
        } else if (a == "--gdn-chunk-lengths") {
            if (!parse_uint_list(value(), o.gdn_chunk_lengths)) { return false; }
        } else if (a == "--gdn-chunk-iters") {
            o.gdn_chunk_iters = static_cast<uint32_t>(std::strtoul(std::string{value()}.c_str(), nullptr, 10));
        } else if (a == "--gdn-decode") {
            o.gdn_decode = true;
        } else if (a == "--gdn-decode-iters") {
            o.gdn_decode_iters = static_cast<uint32_t>(std::strtoul(std::string{value()}.c_str(), nullptr, 10));
        } else if (a == "--gdn-gates") {
            o.gdn_gates = true;
        } else if (a == "--gdn-gates-iters") {
            auto v = value();
            o.gdn_gates_iters = static_cast<uint32_t>(std::strtoul(std::string{v}.c_str(), nullptr, 10));
            } else if (a == "--gdn-recurrent-norm") {
                o.gdn_recurrent_norm = true;
            } else if (a == "--gdn-recurrent-norm-iters") {
                const auto v = value();
                o.gdn_recurrent_norm_iters = static_cast<uint32_t>(std::strtoul(std::string{v}.c_str(), nullptr, 10));
                if (o.gdn_recurrent_norm_iters == 0u || o.gdn_recurrent_norm_iters > 10000u) { return false; }
            } else if (a == "--gdn-recurrent-norm-steps") {
                const auto v = value();
                o.gdn_recurrent_norm_steps = static_cast<uint32_t>(std::strtoul(std::string{v}.c_str(), nullptr, 10));
                if (o.gdn_recurrent_norm_steps == 0u || o.gdn_recurrent_norm_steps > 100000u) { return false; }
            if (o.gdn_gates_iters == 0u || o.gdn_gates_iters > 1000u) { return false; }
        } else if (a == "--gdn-decode-positions") {
            o.gdn_decode_positions = static_cast<uint32_t>(std::strtoul(std::string{value()}.c_str(), nullptr, 10));
        } else if (a == "--calibrate") {
            o.calibrate = true;
        } else if (a == "--calibrate-iters") {
            auto v = value();
            if (v.empty()) { return false; }
            o.calibrate_iters =
                static_cast<uint32_t>(std::strtoul(std::string{v}.c_str(), nullptr, 10));
            if (o.calibrate_iters == 0u || o.calibrate_iters > 10000u) { return false; }
        } else if (a == "--calibrate-copy-mib") {
            auto v = value();
            if (v.empty()) { return false; }
            o.calibrate_copy_mib =
                std::strtoull(std::string{v}.c_str(), nullptr, 10);
            if (o.calibrate_copy_mib < 16u || o.calibrate_copy_mib > 8192u) { return false; }
        } else if (a == "--calibrate-triad-mib") {
            auto v = value();
            if (v.empty()) { return false; }
            o.calibrate_triad_mib =
                std::strtoull(std::string{v}.c_str(), nullptr, 10);
            if (o.calibrate_triad_mib < 16u || o.calibrate_triad_mib > 8192u) { return false; }
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
    // The CUDA native Tile tiling is a device-free contract: reject a bad override
    // while parsing (usage error) instead of discovering it after the device exists.
    if (o.cuda_tile_fp8) {
        QwenCudaTileGemmConfig cfg{};
        if (o.cuda_tile_m != 0u) { cfg.tile_m = o.cuda_tile_m; }
        if (o.cuda_tile_n != 0u) { cfg.tile_n = o.cuda_tile_n; }
        if (!validate_qwen_cuda_tile_gemm_config(cfg).ok()) { return false; }
    }
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
          if (format == QwenWeightFormat::w4) {
          // Eight signed 4-bit codes per word, low nibble first. Full [-8,7] range so
          // the sign extension is exercised; the value distribution is irrelevant
          // to the decode instruction stream this arm times.
          uint32_t word = 0u;
          for (uint32_t j = 0; j < 8u; j++) {
              const uint32_t code = ((h >> (j * 3u)) & 0xFu);
              word |= code << (4u * j);
          }
          return word;
      }
      if (format == QwenWeightFormat::q8_0) {
        // Native Q8_0: four SIGNED int8 codes per word. Full signed range so the
        // sign extension is exercised; the ARM only times the decode, the value
        // distribution is irrelevant to the kernel's instruction stream.
        const auto code = [&](uint32_t shift) {
            return ((h >> shift) & 0xFFu);
        };
        return code(0u) | (code(8u) << 8u) | (code(16u) << 16u) | (code(24u) << 24u);
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

// Linear-interpolation percentile (numpy "linear" / R type 7), the same
// definition scripts/benchmark_qwen.py uses, so a p95 quoted here and a p95 in
// the harness mean the same thing.
double percentile(luisa::vector<double> v, double pct) {
    if (v.empty()) { return 0.0; }
    std::sort(v.begin(), v.end());
    if (v.size() == 1u) { return v.front(); }
    const auto pos = static_cast<double>(v.size() - 1u) * pct / 100.0;
    const auto lo = static_cast<size_t>(std::floor(pos));
    const auto hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) { return v[lo]; }
    return v[lo] + (v[hi] - v[lo]) * (pos - static_cast<double>(lo));
}

// ---------------------------------------------------------------------------
// --calibrate: achievable DRAM bandwidth (opt_plan/06 section 3, 08 section 7)
// ---------------------------------------------------------------------------

// Exactly representable, cheap to evaluate on the host, and not a constant, so
// a compiler cannot fold the kernels away. `i` is the element index.
float calibration_b(size_t i) { return static_cast<float>(i % 251u) * 0.5f; }
float calibration_c(size_t i) { return static_cast<float>(i % 97u) * 0.25f; }
constexpr float kCalibrationScale = 1.5f;

struct CalibrationResult {
    const char *name;
    uint64_t bytes_per_iter;
    uint32_t iters;
    double min_ms, median_ms, p95_ms;
    bool validated;
};

void print_calibration(const CalibrationResult &r) {
    const auto gbps = [&](double ms) {
        return r.bytes_per_iter / ms * 1.0e-6;// bytes/ms -> decimal GB/s
    };
    std::printf("CALIBRATION,%s,%llu,%u,%.9f,%.9f,%.9f,%.3f,%.3f,%.3f,%s\n",
                r.name, static_cast<unsigned long long>(r.bytes_per_iter),
                r.iters, r.min_ms, r.median_ms, r.p95_ms,
                gbps(r.min_ms), gbps(r.median_ms), gbps(r.p95_ms),
                r.validated ? "ok" : "FAILED");
    std::fflush(stdout);
}

// Real device-to-device copy: `copy_mib` MiB moved per iteration, i.e. 2x that
// many bytes of DRAM traffic (one read + one write).
CalibrationResult run_calibration_copy(Device &device, Stream &stream,
                                       const Options &options) {
    const size_t bytes = static_cast<size_t>(options.calibrate_copy_mib) * 1024u * 1024u;
    const size_t count = bytes / sizeof(float);
    auto src = device.create_buffer<float>(count);
    auto dst = device.create_buffer<float>(count);
    {
        // Host initialization happens ONCE, outside the timed region.
        auto host = luisa::vector<float>(count);
        for (size_t i = 0; i < count; ++i) { host[i] = calibration_b(i); }
        stream << src.copy_from(luisa::span<const float>{host}) << synchronize();
        host.clear();
        host.shrink_to_fit();
    }
    auto list_warm = CommandList::create(1u);
    list_warm << dst.copy_from(src);
    stream << list_warm.commit() << synchronize();

    luisa::vector<double> samples;
    samples.reserve(options.calibrate_iters);
    for (uint32_t it = 0; it < options.calibrate_iters; ++it) {
        auto list = CommandList::create(1u);
        list << dst.copy_from(src);
        Clock clock;
        clock.tic();
        stream << list.commit() << synchronize();
        samples.emplace_back(clock.toc());
    }
    // Validation: read back a strided window and compare with the source
    // pattern. A copy that silently did nothing (or a compiler that removed it)
    // cannot report a bandwidth.
    bool validated = true;
    {
        const size_t probes = 8u;
        const size_t probe_count = 64u;
        auto host = luisa::vector<float>(probe_count);
        for (size_t p = 0; p < probes; ++p) {
            const auto offset = (count - probe_count) * p / (probes - 1u);
            stream << dst.view(offset, probe_count).copy_to(luisa::span{host})
                   << synchronize();
            for (size_t j = 0; j < probe_count; ++j) {
                if (host[j] != calibration_b(offset + j)) { validated = false; }
            }
        }
    }
    std::printf("prepare calibrate=copy buffer=%.1f MiB traffic=%.1f MiB iters=%u\n",
                static_cast<double>(bytes) / (1024.0 * 1024.0),
                static_cast<double>(bytes) * 2.0 / (1024.0 * 1024.0),
                options.calibrate_iters);
    std::fflush(stdout);
    return CalibrationResult{"copy", static_cast<uint64_t>(bytes) * 2u,
                             options.calibrate_iters,
                             *std::min_element(samples.begin(), samples.end()),
                             median(samples), percentile(samples, 95.0),
                             validated};
}

// Real FP32 triad (a = b + s*c): 3x array bytes of DRAM traffic per iteration
// (read b, read c, write a). This is the access pattern a decoder's weight
// stream sees (mostly read, one accumulating store), which is why the plan
// asks for it next to the copy.
CalibrationResult run_calibration_triad(Device &device, Stream &stream,
                                        const Options &options) {
    const size_t bytes = static_cast<size_t>(options.calibrate_triad_mib) * 1024u * 1024u;
    const size_t count = bytes / sizeof(float);
    auto a = device.create_buffer<float>(count);
    auto b = device.create_buffer<float>(count);
    auto c = device.create_buffer<float>(count);
    {
        // One host allocation pair, uploaded once, outside the timed region.
        auto host_b = luisa::vector<float>(count);
        auto host_c = luisa::vector<float>(count);
        for (size_t i = 0; i < count; ++i) {
            host_b[i] = calibration_b(i);
            host_c[i] = calibration_c(i);
        }
        stream << b.copy_from(luisa::span<const float>{host_b})
               << c.copy_from(luisa::span<const float>{host_c})
               << synchronize();
        host_b.clear();
        host_b.shrink_to_fit();
        host_c.clear();
        host_c.shrink_to_fit();
    }
    // a = b + s*c, one element per thread. `count` is a whole number of
    // 256-thread blocks (calibrate_triad_mib is MiB and 1 MiB = 262144 floats),
    // so no bounds check is needed inside the timed kernel.
    Kernel1D<void(Buffer<float>, Buffer<float>, Buffer<float>, float)> kernel =
        [](BufferFloat out, BufferFloat lhs, BufferFloat rhs, Float scale) noexcept {
            set_block_size(256u, 1u, 1u);
            set_name("qwen_calibration_triad");
            const auto i = dispatch_x();
            out.write(i, lhs.read(i) + scale * rhs.read(i));
        };
    auto shader = device.compile(kernel);
    // Kernel1D::dispatch() takes the TOTAL thread count of the launch (the
    // 256-thread block size below turns it into count/256 blocks); passing the
    // block count here would run 1/256 of the work and validate as FAILED.
    const auto threads = static_cast<uint32_t>(count);
    const auto record = [&] {
        auto list = CommandList::create(1u);
        list << shader(a.view(), b.view(), c.view(), kCalibrationScale)
                    .dispatch(threads);
        return list;
    };
    {
        auto warm = record();
        stream << warm.commit() << synchronize();
    }
    luisa::vector<double> samples;
    samples.reserve(options.calibrate_iters);
    for (uint32_t it = 0; it < options.calibrate_iters; ++it) {
        auto list = record();
        Clock clock;
        clock.tic();
        stream << list.commit() << synchronize();
        samples.emplace_back(clock.toc());
    }
    bool validated = true;
    {
        const size_t probes = 8u;
        const size_t probe_count = 64u;
        auto host = luisa::vector<float>(probe_count);
        for (size_t p = 0; p < probes; ++p) {
            const auto offset = (count - probe_count) * p / (probes - 1u);
            stream << a.view(offset, probe_count).copy_to(luisa::span{host})
                   << synchronize();
            for (size_t j = 0; j < probe_count; ++j) {
                const auto expected = calibration_b(offset + j) +
                                      kCalibrationScale * calibration_c(offset + j);
                if (std::abs(host[j] - expected) > 1.0e-3f) { validated = false; }
            }
        }
    }
    std::printf("prepare calibrate=triad arrays=3x%.1f MiB traffic=%.1f MiB iters=%u\n",
                static_cast<double>(bytes) / (1024.0 * 1024.0),
                static_cast<double>(bytes) * 3.0 / (1024.0 * 1024.0),
                options.calibrate_iters);
    std::fflush(stdout);
    return CalibrationResult{"triad", static_cast<uint64_t>(bytes) * 3u,
                             options.calibrate_iters,
                             *std::min_element(samples.begin(), samples.end()),
                             median(samples), percentile(samples, 95.0),
                             validated};
}

int run_calibration(Device &device, Stream &stream, const Options &options) {
    std::printf("CALIBRATION columns: name,bytes_per_iter,iters,min_ms,median_ms,"
                "p95_ms,min_GBps,median_GBps,p95_GBps,validate\n");
    const auto copy = run_calibration_copy(device, stream, options);
    print_calibration(copy);
    const auto triad = run_calibration_triad(device, stream, options);
    print_calibration(triad);
    // The roof denominator: the best sustained median the two real kernels
    // reached. Both measure the same DRAM roof from different read/write mixes;
    // taking the larger is what makes an "achievable" number, and it is still a
    // measurement on THIS device, driver and clock state (never a vendor figure).
    const auto median_gbps = [](const CalibrationResult &r) {
        return r.median_ms > 0.0 ? r.bytes_per_iter / r.median_ms * 1.0e-6 : 0.0;
    };
    const auto copy_gbps = median_gbps(copy);
    const auto triad_gbps = median_gbps(triad);
    const auto best = std::max(copy_gbps, triad_gbps);
    std::printf("CALIBRATION summary: achievable_GBps=%.3f iters=%u "
                "copy_median_GBps=%.3f triad_median_GBps=%.3f validated=%s\n",
                best, options.calibrate_iters, copy_gbps, triad_gbps,
                (copy.validated && triad.validated) ? "ok" : "FAILED");
    std::printf("CALIBRATION note: pass the achievable_GBps value to "
                "scripts/benchmark_qwen.py --bandwidth; it is the only "
                "defensible denominator for a roof percentage, per tier, per "
                "driver and per clock state.\n");
    if (options.calibrate_iters < 30u) {
        std::printf("CALIBRATION warning: only %u iterations; the plan requires "
                    ">= 30 for a reportable number.\n", options.calibrate_iters);
    }
    std::fflush(stdout);
    return (copy.validated && triad.validated) ? 0 : 3;
}

void run_attention(Device &device, Stream &stream, uint32_t iters) {
    // fp8 KV takes per-(head,64-block) scale buffers; the FP32 arm this benchmark
    // measures binds this 1-element dummy and never reads it (the scale load
    // folds away), so the measured launch/payload stays exactly the FP32 one.
    auto bscale_dummy = device.create_buffer<float>(1);
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
                list << shader(q.view(), gate.view(), k.view(), v.view(),
                     bscale_dummy.view(), bscale_dummy.view(), out.view(),
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
    const uint64_t elements = static_cast<uint64_t>(shape.n) * shape.k;
    const size_t words_per_matrix = static_cast<size_t>((elements + (format_pack(format) - 1u)) / format_pack(format));
    const size_t scales_per_matrix = format_scaled(format) ? static_cast<size_t>(shape.n) * (shape.k / format_group(format)) : 1u;
    const size_t weight_bytes = words_per_matrix * sizeof(uint32_t);
    const size_t replicas = std::max<size_t>(1u, std::min(kMaxReplicas, kTargetWeightBytes / weight_bytes));
    const size_t total_words = words_per_matrix * replicas;
    const size_t total_scales = scales_per_matrix * replicas;

    std::printf("prepare format=%s shape=%s N=%u K=%u replicas=%zu weight-working-set=%.1f MiB\n",
                format_label(format), shape.name, shape.n, shape.k, replicas,
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

    auto s_host = luisa::vector<float>(total_scales, format_scaled(format) ? 0.03125f : 0.0f);
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
          if (options.packed && format == QwenWeightFormat::w4) {
          // There is NO 128-bit W4 realization (one uint4 would be 32 codes and a
          // 1024-element lane stride, and it has not been measured). Refuse the arm
          // instead of timing a kernel that would compute the wrong thing.
          std::fprintf(stderr, "packed (128-bit) decode has no W4 realization; "
                               "use --impl generic or --impl decode for the w4 arm\n");
          std::exit(2);
      }
      auto packed = options.packed
                        ? device.compile(make_qwen35_decode_gemv_packed_kernel(
                            format, QwenMatmulEpilogue::none))
                      : Shader<1, Buffer<float4>, Buffer<uint4>, Buffer<float>, Buffer<float>,
                               Buffer<float>, uint32_t, uint32_t, uint32_t, uint32_t>{};
    auto generic_plan = plan_qwen_gemv_dispatch(1u, shape.n, shape.k);
          auto decode_plan = plan_qwen35_decode_gemv_dispatch(shape.n, shape.k, format, format_group(format));
    auto packed_plan = plan_qwen35_decode_gemv_packed_dispatch(shape.n, shape.k, format, format_group(format));
    if (options.packed && !packed_plan.ok()) {
        std::fprintf(stderr, "packed planning failed for %s: %s\n", shape.name,
                     packed_plan.status.message.c_str());
        std::exit(2);
    }
    if ((options.generic && !generic_plan.ok()) || (options.decode && !decode_plan.ok())) {
        std::fprintf(stderr, "planning failed for %s\n", shape.name);
        std::exit(2);
    }

    const double effective_bytes = static_cast<double>(weight_bytes) +
                                   (format_scaled(format) ? static_cast<double>(scales_per_matrix) * 4.0 : 0.0) +
                                   static_cast<double>(shape.k + shape.n) * 4.0;
    const auto measure = [&](auto &shader, const QwenLaunch1D &plan, const char *impl) {
        const auto record = [&] {
            auto list = CommandList::create(static_cast<size_t>(replicas));
            for (size_t r = 0; r < replicas; ++r) {
                list << shader(x.view(), w.view(r * words_per_matrix, words_per_matrix),
                               scales.view(r * scales_per_matrix, scales_per_matrix),
                               residual.view(), y.view(), 1u, shape.n, shape.k, format_group(format))
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
                    format_label(format), shape.name, impl, shape.n, shape.k, replicas,
                    min_ms, med_ms, effective_bytes / min_ms * 1.0e-6,
                    effective_bytes / med_ms * 1.0e-6);
        std::fflush(stdout);
    };
    if (options.generic) { measure(generic, *generic_plan.value, "generic"); }
    if (options.decode) { measure(decode, *decode_plan.value, "decode"); }
    if (options.packed) {
        // Same buffers, same traffic model: only the kernel realization differs.
        const auto vecs = static_cast<uint32_t>(shape.k / qwen_packed_vec_elements(format));
        const auto record_packed = [&] {
            auto list = CommandList::create(static_cast<size_t>(replicas));
            for (size_t r = 0; r < replicas; ++r) {
                list << packed(x.view().as<float4>(),
                               w.view(r * words_per_matrix, words_per_matrix).as<uint4>(),
                               scales.view(r * scales_per_matrix, scales_per_matrix),
                               residual.view(), y.view(), shape.n, shape.k, format_group(format), vecs)
                            .dispatch(qwen_dispatch_threads(*packed_plan.value));
            }
            return list;
        };
        auto warmup = record_packed();
        stream << warmup.commit() << synchronize();
        luisa::vector<double> samples;
        samples.reserve(options.iters);
        for (uint32_t it = 0; it < options.iters; ++it) {
            auto list = record_packed();
            Clock clock;
            clock.tic();
            stream << list.commit() << synchronize();
            samples.emplace_back(clock.toc() / static_cast<double>(replicas));
        }
        const auto min_ms = *std::min_element(samples.begin(), samples.end());
        const auto med_ms = median(samples);
        std::printf("RESULT,%s,%s,%s,%u,%u,%zu,%.9f,%.9f,%.3f,%.3f\n",
                    format_label(format), shape.name, "packed", shape.n, shape.k, replicas,
                    min_ms, med_ms, effective_bytes / min_ms * 1.0e-6,
                    effective_bytes / med_ms * 1.0e-6);
        std::fflush(stdout);
    }
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
        auto fused_plan = plan_qwen_ffn_decode_dispatch(shape.n, shape.k, format, format_group(format));
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
                        list << fused(x.view(), wv, sv, uv, usv, y.view(), shape.n, shape.k, format_group(format))
                                    .dispatch(qwen_dispatch_threads(*fused_plan.value));
                    } else {
                        const auto uv = up_w.view(r * words_per_matrix, words_per_matrix);
                        const auto usv = up_scales.view(r * scales_per_matrix, scales_per_matrix);
                        list << decode(x.view(), wv, sv, residual.view(), gate_y.view(),
                                       1u, shape.n, shape.k, format_group(format))
                                    .dispatch(qwen_dispatch_threads(*decode_plan.value));
                        list << decode(x.view(), uv, usv, residual.view(), up_y.view(),
                                       1u, shape.n, shape.k, format_group(format))
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
                                        (format_scaled(format) ? scales_per_matrix * 4.0 : 0.0)) +
                                 static_cast<double>(shape.k + 3u * shape.n) * 4.0;
            std::printf("RESULT,%s,%s,%s,%u,%u,%zu,%.9f,%.9f,%.3f,%.3f\n",
                        format_label(format), shape.name, name, shape.n, shape.k, replicas,
                        min_ms, med_ms, bytes / min_ms * 1.0e-6, bytes / med_ms * 1.0e-6);
        };
        run_ffn(false, "separate_ffn");
        run_ffn(true, "fused_ffn");
        if (options.packed) {
            // 128-bit fused FFN realization (opt_plan/09 N2) on the same buffers.
            auto fused_packed_sh = device.compile(make_qwen_ffn_decode_packed_kernel(format));
            auto packed_plan = plan_qwen_ffn_decode_packed_dispatch(shape.n, shape.k, format, format_group(format));
            if (!packed_plan.ok()) {
                std::fprintf(stderr, "packed FFN planning failed for %s\n", shape.name);
                std::exit(2);
            }
            const auto vecs = static_cast<uint32_t>(shape.k / qwen_packed_vec_elements(format));
            const auto record_packed_ffn = [&] {
                auto list = CommandList::create(static_cast<size_t>(replicas));
                for (size_t r = 0; r < replicas; ++r) {
                    list << fused_packed_sh(
                                    x.view().as<float4>(),
                                    w.view(r * words_per_matrix, words_per_matrix).as<uint4>(),
                                    scales.view(r * scales_per_matrix, scales_per_matrix),
                                    up_w.view(r * words_per_matrix, words_per_matrix).as<uint4>(),
                                    up_scales.view(r * scales_per_matrix, scales_per_matrix),
                                    y.view(), shape.n, shape.k, format_group(format), vecs)
                                .dispatch(qwen_dispatch_threads(*packed_plan.value));
                }
                return list;
            };
            auto warmup_packed = record_packed_ffn();
            stream << warmup_packed.commit() << synchronize();
            luisa::vector<double> times;
            for (uint32_t it = 0; it < options.iters; ++it) {
                auto list = record_packed_ffn();
                Clock clock;
                clock.tic();
                stream << list.commit() << synchronize();
                times.emplace_back(clock.toc() / static_cast<double>(replicas));
            }
            const auto min_ms = *std::min_element(times.begin(), times.end());
            const auto med_ms = median(times);
            const double bytes = 2.0 * (static_cast<double>(weight_bytes) +
                                        (format_scaled(format) ? scales_per_matrix * 4.0 : 0.0)) +
                                 static_cast<double>(shape.k + 3u * shape.n) * 4.0;
            std::printf("RESULT,%s,%s,%s,%u,%u,%zu,%.9f,%.9f,%.3f,%.3f\n",
                        format_label(format), shape.name, "fused_ffn_packed", shape.n, shape.k,
                        replicas, min_ms, med_ms, bytes / min_ms * 1.0e-6, bytes / med_ms * 1.0e-6);
            std::fflush(stdout);
        }
    }
}

// ---------------------------------------------------------------------------
// --prefill-attention: isolated causal GQA prefill attention arm
// (opt_plan/02 section 5 item 1 / section 6 gate).
//
// Runs the UNTILED reference kernel (one thread per (token, q_head), runtime
// head_dim local accumulator) and the TILED FlashAttention-style kernel on the
// exact same buffers, positions and causal math, at the model geometry
// (q_heads 16, kv_heads 4, head_dim 256) and the plan's prompt lengths. It
// reports min/median wall time and the achieved FP32 GFLOP/s (causal
// score + P.V = 2*S*(S+1)*head_dim*q_heads FLOP), and it VALIDATES the tiled
// result against the reference result (max relative deviation + finiteness)
// before printing a speedup, so a silently wrong kernel cannot report one.
// Recording is excluded; each sample is one dispatch plus one final
// synchronization, exactly like the decode arms above.
// ---------------------------------------------------------------------------
struct PrefillAttnOptions {
    // Prompt lengths of the arm (capacity == tokens, pos0 == 0: the causal
    // window then is the full cache, the plan's "full capacity" boundary).
    luisa::vector<uint32_t> lengths{128u, 512u, 1024u};
    uint32_t q_tile{0u};// 0 = the device-selected tiling
    uint32_t k_tile{0u};
    uint32_t iters{0u};// 0 = the global --iters budget
};

double prefill_attn_flops(uint32_t tokens, uint32_t q_heads, uint32_t head_dim) {
    // Causal pairs per q head = S*(S+1)/2, 2*head_dim FLOP per pair.
    const double s = static_cast<double>(tokens);
    return s * (s + 1.0) * static_cast<double>(head_dim) * static_cast<double>(q_heads);
}

void print_prefill_attn(const char *impl, const char *label, const PrefillAttnOptions &arm,
                        uint32_t tokens, uint32_t pos0, uint32_t capacity,
                        double min_ms, double med_ms, double flops) {
    const auto gflops = [&](double ms) { return ms > 0.0 ? flops / ms * 1.0e-6 : 0.0; };
    std::printf("RESULT,prefill_attn,%s,%s,%u,%u,%u,%u,%u,%.9f,%.9f,%.3f,%.3f\n",
                impl, label, tokens, pos0, capacity, arm.q_tile, arm.k_tile,
                min_ms, med_ms, gflops(min_ms), gflops(med_ms));
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// --prefill-gemm: isolated prefill GEMM arm (opt_plan/02 section 3, 09 N5.2).
//
// Both implementations run over the SAME packed buffers, so the only difference
// is the kernel. The reported traffic is the WEIGHT-STREAMING MODEL the plan
// argues about: weight_bytes * ceil(M / rows_per_weight_pass) plus one x read
// and one y write. It is not a hardware counter (x is re-read from L2 and that
// re-read is deliberately not counted); the wall time is measured.
//
// --cuda-tile-fp8 adds a THIRD, opt-in implementation to this arm: the CUDA
// native Tile IR FP8 GEMM of qwen/cuda_tile_gemm.h, run over the SAME packed
// uint32 FP8 weights, the SAME FP32 group scales and the SAME FP32 x buffer as
// the DSL and native-shader rows. It is not a substitute for either of them: it
// is only reported when the caller asks for it, it prints an explicit `skipped`
// row for every (format, shape) it cannot serve, and it never runs on a format
// whose storage it cannot read.
// ---------------------------------------------------------------------------
struct PrefillGemmShape {
    const char *name;
    uint32_t n;
    uint32_t k;
};

// Tolerance of the CUDA native Tile FP8 row against the FP32-ACTIVATION DSL
// reference (the legacy one-thread-per-(m,n) GEMM over the same FP32 x). This is
// the same metric and the same documented bound the differential test uses
// (kTileFp32Tol in tests/integration/qwen/test_qwen_native_gemm.cpp): max|d| /
// max|ref| <= 1e-1, i.e. the DEVIATION, not an exactness gate.
//
// Why it is not the 1e-4 oracle gate: the Tile kernel rounds the activation to
// BF16 before its BF16 tensor-core dot and applies the FP32 group scale to the
// partial dot AFTER the contraction, so it is a different function from both the
// DSL GEMM (which scales each decoded weight in FP32 and accumulates in FMA
// order) and the native-shader kernel (which scales the weight to BF16 BEFORE
// the dot). Exactness against the DECOMPOSED oracle (BF16 activations, exact
// BF16(fp8) weights, double-precision partials) is what the differential test
// proves to <1e-6; a benchmark row cannot afford a double-precision CPU oracle
// per shape, so this arm gates on the documented FP32-x envelope and prints the
// bf16-x-oracle and native deviations as INFORMATION.
constexpr double kCudaTileFp32Tol = 1e-1;

// Outcome of ONE row of the opt-in CUDA native Tile FP8 arm. A row that this
// device/toolkit/format/shape cannot serve carries `measured == false` plus the
// verbatim reason, which the caller prints as an explicit `skipped` RESULT row -
// never a silent substitute, never a failure of the surrounding benchmark.
struct CudaTileGemmRow {
    bool measured{false};
    luisa::string skip_reason;
    uint32_t lazy_compiles{0u}; // shape compiles this row needed (0 if cached)
    double jit_ms{0.0};         // their wall time, EXCLUDED from min_ms/median_ms
    double min_ms{0.0};
    double med_ms{0.0};
    bool validated{false};
    size_t non_finite{0u};
    // Normalized deviations of THIS result: max|d| / max|ref| (the metric every
    // gate in this file uses) and rms(d) / rms(ref). `ref` is the FP32-activation
    // DSL reference and therefore the gate; the bf16-x DSL oracle and the current
    // native-shader FP8 result are printed as information.
    double ref_rel_max{0.0}, ref_rel_rms{0.0};
    double oracle_rel_max{0.0}, oracle_rel_rms{0.0};
    double native_rel_max{0.0}, native_rel_rms{0.0};
    bool have_oracle{false};
    bool have_native{false};
};

// Normalized deviation of an already-downloaded result against a device buffer.
// The reference is read back once and freed immediately: a model-sized row is
// m*n*4 bytes, so keeping three of them alive at once would be wasteful.
void cuda_tile_deviation(Stream &stream, luisa::span<const float> got,
                         Buffer<float> &ref_buffer, double &rel_max,
                         double &rel_rms) {
    auto host_ref = luisa::vector<float>(got.size());
    stream << ref_buffer.view().copy_to(luisa::span<float>{host_ref}) << synchronize();
    double max_ref = 0.0, max_d = 0.0, sum_d = 0.0, sum_ref = 0.0;
    for (size_t i = 0; i < host_ref.size(); ++i) {
        const double d = double(got[i]) - double(host_ref[i]);
        if (std::fabs(d) > max_d) { max_d = std::fabs(d); }
        max_ref = std::max(max_ref, std::fabs(double(host_ref[i])));
        sum_d += d * d;
        sum_ref += double(host_ref[i]) * double(host_ref[i]);
    }
    rel_max = max_ref > 0.0 ? max_d / max_ref : max_d;
    rel_rms = std::sqrt(sum_d / (sum_ref + 1e-30));
}

// Measure ONE (shape, M) of the CUDA native Tile FP8 arm over the caller's OWN
// packed uint32 FP8 weights, FP32 group scales and FP32 activations (the same
// buffers the DSL and native rows read; nothing is repacked or dequantized).
//
// JIT policy: the FIRST build_dispatch of a static (m, n, k) is the lazy compile.
// It happens BEFORE the warm-up, its cost is measured into `jit_ms` and printed on
// its own line, and the counter is re-checked after the samples: a row whose
// samples contained a compile is flagged instead of silently reporting a JIT cost
// as kernel time.
CudaTileGemmRow run_cuda_tile_gemm_row(
    Device &device, Stream &stream, QwenCudaTileGemm &gemm,
    const QwenCudaTileGemmConfig &cfg, QwenWeightFormat format,
    const PrefillGemmShape &shape, uint32_t m, uint32_t iters,
    Buffer<float> &x, Buffer<uint32_t> &w, Buffer<float> &bw_scales,
    Buffer<float> &y_legacy, Buffer<float> &y_oracle, Buffer<float> &y_native,
    bool native_ok) {
    CudaTileGemmRow row;
    const auto skip = [&row](luisa::string reason) {
        row.skip_reason = std::move(reason);
        return row;
    };
    if (!qwen_cuda_tile_gemm_supported_format(format)) {
        return skip("typed FP8 E4M3 storage only: the native Tile route has no "
                    "realization for this weight format and never re-encodes one");
    }
    if (!gemm.valid()) {
        return skip("the arm is not available on this device/toolkit (see the "
                    "PREFILL-GEMM cuda_tile line above)");
    }
    if (auto s = validate_qwen_cuda_tile_gemm_shape(m, shape.n, shape.k, format, cfg.group);
        !s.ok()) {
        luisa::string reason = s.message;
        if (!s.detail.empty()) {
            reason.append(": ");
            reason.append(s.detail);
        }
        return skip(std::move(reason));
    }
    auto y_tile = device.create_buffer<float>(size_t(m) * shape.n);
    luisa::string error;
    const auto compiles_before = gemm.compile_count();
    const auto jit_before = gemm.jit_ms();
    auto first = gemm.build_dispatch(false, x.view(), w.view(), bw_scales.view(), x.view(),
                                     y_tile.view(), m, shape.n, shape.k, &error);
    if (first == nullptr) {
        return skip(error.empty() ? luisa::string{"build_dispatch refused the shape"} :
                                    std::move(error));
    }
    row.lazy_compiles = gemm.compile_count() - compiles_before;
    row.jit_ms = gemm.jit_ms() - jit_before;
    std::printf("PREFILL-GEMM cuda_tile JIT: %s %s M=%u %ux%ux%u: %u lazy compile(s), "
                "%.1f ms (excluded from warm-up and from every sample)\n",
                format_label(format), shape.name, m, cfg.tile_m, cfg.tile_n, cfg.tile_k,
                row.lazy_compiles, row.jit_ms);
    std::fflush(stdout);
    // The static shape is cached now: this command is the WARM-UP, not a sample.
    {
        auto warm = CommandList::create(1u);
        warm << std::move(first);
        stream << warm.commit() << synchronize();
    }
    const auto record = [&](luisa::string *err) {
        auto list = CommandList::create(1u);
        auto command = gemm.build_dispatch(false, x.view(), w.view(), bw_scales.view(),
                                           x.view(), y_tile.view(), m, shape.n, shape.k, err);
        if (command != nullptr) { list << std::move(command); }
        return list;
    };
    luisa::string warm_error;
    stream << record(&warm_error).commit() << synchronize();
    if (!warm_error.empty()) {
        return skip(warm_error.empty() ? luisa::string{"warm-up build failed"} :
                                         std::move(warm_error));
    }
    luisa::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t it = 0; it < iters; ++it) {
        auto list = record(nullptr);
        Clock clock;
        clock.tic();
        stream << list.commit() << synchronize();
        samples.emplace_back(clock.toc());
    }
    if (gemm.compile_count() != compiles_before + row.lazy_compiles) {
        std::printf("PREFILL-GEMM cuda_tile warning: %u compile(s) happened INSIDE the "
                    "timed samples of %s %s M=%u (JIT is not fully excluded here)\n",
                    gemm.compile_count() - compiles_before - row.lazy_compiles,
                    format_label(format), shape.name, m);
        std::fflush(stdout);
    }
    row.min_ms = *std::min_element(samples.begin(), samples.end());
    row.med_ms = median(samples);
    row.measured = true;
    auto host_tile = luisa::vector<float>(size_t(m) * shape.n);
    stream << y_tile.view().copy_to(luisa::span<float>{host_tile}) << synchronize();
    for (auto v : host_tile) {
        if (!std::isfinite(v)) { row.non_finite++; }
    }
    cuda_tile_deviation(stream, luisa::span<const float>{host_tile}, y_legacy,
                        row.ref_rel_max, row.ref_rel_rms);
    cuda_tile_deviation(stream, luisa::span<const float>{host_tile}, y_oracle,
                        row.oracle_rel_max, row.oracle_rel_rms);
    row.have_oracle = true;
    if (native_ok) {
        cuda_tile_deviation(stream, luisa::span<const float>{host_tile}, y_native,
                            row.native_rel_max, row.native_rel_rms);
        row.have_native = true;
    }
    row.validated = row.non_finite == 0u && row.ref_rel_max <= kCudaTileFp32Tol;
    return row;
}

int run_prefill_gemm(Device &device, Stream &stream, const Options &options) {
    const PrefillGemmShape shapes[] = {
        {"ffn_gate_up", 9216u, 2560u},
        {"ffn_down", 2560u, 9216u},
        {"attn_q", 8192u, 2560u},
        // The remaining N widths the model actually dispatches (the GDN gate and
        // alpha/beta projections are N = 1024 and N = 32), so this arm covers
        // N = 32/1024/2560/8192/9216 against K = 2560/9216.
        {"gdn_gate", 1024u, 2560u},
        {"gdn_alpha", 32u, 2560u},
    };
    const uint32_t iters = options.prefill_gemm_iters != 0u ? options.prefill_gemm_iters : options.iters;
    QwenGemmMtConfig mt_cfg{};
    std::printf("PREFILL-GEMM note: weight_MB is the modelled DRAM weight traffic "
                "(weight_bytes * ceil(M/pass_rows)); x is read once and y written "
                "once and no L2 reuse is credited. iters=%u mt=%u warps=%u\n",
                iters, mt_cfg.mt, mt_cfg.warps);
    std::printf("PREFILL-GEMM columns: format,shape,impl,M,N,K,pass_rows,"
                "weight_passes,gflop,min_ms,median_ms,weight_MB,x_MB,effective_GBps,validated\n");
    // The opt-in CUDA native Tile FP8 arm: config first (device-free), then ONE
    // create per run - its probes cost ~0.35 s and its answer does not depend on
    // the shape, so it happens outside the shape loop exactly once. `tile_ready`
    // false keeps every row of this arm an explicit `skipped` row.
    QwenCudaTileGemmConfig tile_cfg{};
    if (options.cuda_tile_m != 0u) { tile_cfg.tile_m = options.cuda_tile_m; }
    if (options.cuda_tile_n != 0u) { tile_cfg.tile_n = options.cuda_tile_n; }
    QwenCudaTileGemm tile_gemm;
    bool tile_create_attempted = false;
    bool tile_ready = false;
    luisa::string tile_unavailable{"the arm was not created"};
    uint32_t tile_rows_measured = 0u;
    uint32_t tile_rows_skipped = 0u;
    if (options.cuda_tile_fp8) {
        std::printf("PREFILL-GEMM cuda_tile note: backend=%s tile=%ux%ux%u group=%u "
                    "LUISA_CUDA_TILE_IR=%s; an explicit `skipped` row means this arm "
                    "cannot serve that (format, shape) - the other rows are unaffected\n",
                    luisa::string{device.backend_name()}.c_str(), tile_cfg.tile_m,
                    tile_cfg.tile_n, tile_cfg.tile_k, tile_cfg.group,
                    qwen_cuda_tile_gemm_native_route_requested() ? "1" : "unset");
    }
    bool all_validated = true;
    for (auto format : enabled_formats(options)) {
        const uint32_t pack = format_pack(format);
        // CUDA tensor-core GEMM through NativeShaderExt/CUDA_NVRTC (opt_plan/09
        // N1a): one JIT per weight format, then one dispatch per (shape, M). Its
        // cold NVRTC cost is reported, never amortized into the timings below.
        QwenNativeGemm native_gemm;
        QwenNativeGemmConfig native_cfg{};
        if (auto s = native_gemm.create(device, format, native_cfg); !s.ok()) {
            std::printf("PREFILL-GEMM native: unavailable for %s (%s%s%s)\n",
                        format_label(format), s.message.c_str(),
                        s.detail.empty() ? "" : ": ", s.detail.c_str());
        } else {
            std::printf("PREFILL-GEMM native: %s tile %ux%ux%u (%u warps, %u threads), "
                        "cold JIT %.1f ms\n",
                        format_label(format), native_cfg.block_m, native_cfg.block_n,
                        native_cfg.block_k, native_cfg.warps, native_cfg.block(),
                        native_gemm.jit_ms());
        }
        // CUDA native Tile IR FP8 GEMM, opt-in and FP8-ONLY. Created once per run
        // (its answer is shape independent); a failure is reported here, verbatim,
        // and every row of the arm below becomes explicit `skipped`.
        if (options.cuda_tile_fp8 && !tile_create_attempted &&
            qwen_cuda_tile_gemm_supported_format(format)) {
            tile_create_attempted = true;
            if (auto s = tile_gemm.create(device, tile_cfg); !s.ok()) {
                tile_unavailable = s.message;
                if (!s.detail.empty()) {
                    tile_unavailable.append(": ");
                    tile_unavailable.append(s.detail);
                }
                std::printf("PREFILL-GEMM cuda_tile: unavailable for %s (%s)\n",
                            format_label(format), tile_unavailable.c_str());
                std::fflush(stdout);
            } else {
                tile_ready = true;
                std::printf("PREFILL-GEMM cuda_tile: %s tile %ux%ux%u, ragged_columns=%s, "
                            "create probe %.1f ms\n",
                            format_label(format), tile_cfg.tile_m, tile_cfg.tile_n,
                            tile_cfg.tile_k,
                            tile_gemm.supports_ragged_columns() ? "yes" : "no",
                            tile_gemm.probe_ms());
                std::fflush(stdout);
            }
        } else if (options.cuda_tile_fp8 && !qwen_cuda_tile_gemm_supported_format(format)) {
            std::printf("PREFILL-GEMM cuda_tile: skipped for %s (typed FP8 E4M3 storage only; "
                        "the native Tile route has no realization for this weight format)\n",
                        format_label(format));
            std::fflush(stdout);
        }
        for (const auto &shape : shapes) {
            const uint64_t elements = uint64_t(shape.n) * shape.k;
            const size_t words = static_cast<size_t>(elements / pack);
            const size_t scales = format_scaled(format) ? size_t(shape.n) * (shape.k / format_group(format)) : 1u;
            const double weight_bytes = static_cast<double>(words) * sizeof(uint32_t);
            auto w_host = luisa::vector<uint32_t>(words);
            for (size_t i = 0; i < words; ++i) { w_host[i] = packed_word(format, i); }
            auto w = device.create_buffer<uint32_t>(words);
            stream << w.copy_from(luisa::span<const uint32_t>{w_host}) << synchronize();
            w_host.clear();
            w_host.shrink_to_fit();
            auto s_host = luisa::vector<float>(scales, format_scaled(format) ? 0.03125f : 0.0f);
            auto bw_scales = device.create_buffer<float>(scales);
            stream << bw_scales.copy_from(luisa::span<const float>{s_host}) << synchronize();
            for (uint32_t m : options.prefill_gemm_m) {
                auto x_host = luisa::vector<float>(size_t(m) * shape.k);
                for (size_t i = 0; i < x_host.size(); ++i) {
                    x_host[i] = 0.01f * std::sin(0.013f * static_cast<float>(i));
                }
                // `x_bf16` is the SAME activation already rounded to BF16, i.e. the
                // exact oracle of the tensor-core kernel's numerics: it converts x
                // to BF16 at the shared-tile boundary and accumulates in FP32, so
                // the DSL GEMM run on `x_bf16` differs from it only by FP32
                // reassociation. The deviation from `y_legacy(x)` is the documented
                // BF16-activation rounding, reported separately instead of being
                // mixed into the correctness gate.
                auto x_bf16_host = luisa::vector<float>(x_host.size());
                for (size_t i = 0; i < x_host.size(); ++i) {
                    x_bf16_host[i] = host_bf16_bits_to_f32(host_f32_to_bf16_bits(x_host[i]));
                }
                auto x = device.create_buffer<float>(x_host.size());
                auto x_bf16 = device.create_buffer<float>(x_bf16_host.size());
                stream << x.copy_from(luisa::span<const float>{x_host})
                       << x_bf16.copy_from(luisa::span<const float>{x_bf16_host}) << synchronize();
                x_host.clear();
                x_host.shrink_to_fit();
                x_bf16_host.clear();
                x_bf16_host.shrink_to_fit();
                auto y_mt = device.create_buffer<float>(size_t(m) * shape.n);
                auto y_legacy = device.create_buffer<float>(size_t(m) * shape.n);
                auto y_oracle = device.create_buffer<float>(size_t(m) * shape.n);
                auto y_native = device.create_buffer<float>(size_t(m) * shape.n);
                auto mt_shader = device.compile(make_qwen35_gemm_mt_kernel(
                    mt_cfg, format, QwenMatmulEpilogue::none));
                auto legacy_shader = device.compile(make_qwen_gemm_kernel(
                    format, QwenMatmulEpilogue::none));
                auto mt_plan = plan_qwen35_gemm_mt_dispatch(m, shape.n, shape.k, format, format_group(format), mt_cfg);
                auto legacy_plan = plan_qwen_gemm_dispatch(m, shape.n, shape.k);
                if (!mt_plan.ok() || !legacy_plan.ok()) {
                    std::fprintf(stderr, "planning failed for %s M=%u\n", shape.name, m);
                    return 2;
                }
                const bool native_ok =
                    native_gemm.valid() && m >= qwen_native_gemm_min_rows &&
                    validate_qwen_native_gemm_shape(m, shape.n, shape.k, format).ok();
                luisa::string native_error;
                const auto record = [&](bool use_mt, auto &dst, auto &src) {
                    auto list = CommandList::create(1u);
                    if (use_mt) {
                        list << mt_shader(src.view(), w.view(), bw_scales.view(), x.view(), dst.view(),
                                          m, shape.n, shape.k, format_group(format),
                                          mt_plan.value->column_blocks)
                                    .dispatch(mt_plan.value->total_x, mt_plan.value->total_y);
                    } else {
                        list << legacy_shader(src.view(), w.view(), bw_scales.view(), x.view(), dst.view(),
                                              m, shape.n, shape.k, format_group(format),
                                              legacy_plan.value->padded_cols)
                                    .dispatch(legacy_plan.value->total_x, legacy_plan.value->total_y);
                    }
                    return list;
                };
                const auto record_native = [&]() {
                    auto list = CommandList::create(1u);
                    auto command = native_gemm.build_dispatch(
                        false, x.view(), w.view(), bw_scales.view(), x.view(), y_native.view(),
                        m, shape.n, shape.k, &native_error);
                    if (command == nullptr) {
                        std::fprintf(stderr, "native dispatch failed for %s M=%u: %s\n",
                                     shape.name, m, native_error.c_str());
                        return list;
                    }
                    list << std::move(command);
                    return list;
                };
                stream << record(true, y_mt, x).commit()
                       << record(false, y_legacy, x).commit()
                       << record(false, y_oracle, x_bf16).commit() << synchronize();
                auto host_mt = luisa::vector<float>(size_t(m) * shape.n);
                auto host_legacy = luisa::vector<float>(size_t(m) * shape.n);
                auto host_oracle = luisa::vector<float>(size_t(m) * shape.n);
                stream << y_mt.view().copy_to(luisa::span<float>{host_mt})
                       << y_legacy.view().copy_to(luisa::span<float>{host_legacy})
                       << y_oracle.view().copy_to(luisa::span<float>{host_oracle}) << synchronize();
                double max_diff = 0.0, max_val = 0.0;
                for (size_t i = 0; i < host_mt.size(); ++i) {
                    max_diff = std::max(max_diff, std::fabs(double(host_mt[i]) - double(host_legacy[i])));
                    max_val = std::max(max_val, std::fabs(double(host_legacy[i])));
                }
                const bool validated = max_diff <= 2e-3 * max_val + 1e-3;
                all_validated = all_validated && validated;
                host_mt.clear();
                host_mt.shrink_to_fit();
                // host_legacy stays alive: it is the FP32-activation reference the
                // tensor-core numbers below are compared against.
                bool native_validated = false;
                if (native_ok) {
                    stream << record_native().commit() << synchronize();
                    auto host_native = luisa::vector<float>(size_t(m) * shape.n);
                    stream << y_native.view().copy_to(luisa::span<float>{host_native}) << synchronize();
                    double oracle_max = 0.0, oracle_max_abs = 0.0, oracle_sum_sq = 0.0, oracle_ref_sq = 0.0;
                    double fp32_max_abs = 0.0, fp32_sum_sq = 0.0, fp32_ref_sq = 0.0;
                    for (size_t i = 0; i < host_native.size(); ++i) {
                        const double d_oracle = double(host_native[i]) - double(host_oracle[i]);
                        oracle_max_abs = std::max(oracle_max_abs, std::fabs(d_oracle));
                        oracle_sum_sq += d_oracle * d_oracle;
                        oracle_ref_sq += double(host_oracle[i]) * double(host_oracle[i]);
                        oracle_max = std::max(oracle_max, std::fabs(double(host_oracle[i])));
                        const double d_fp32 = double(host_native[i]) - double(host_legacy[i]);
                        fp32_max_abs = std::max(fp32_max_abs, std::fabs(d_fp32));
                        fp32_sum_sq += d_fp32 * d_fp32;
                        fp32_ref_sq += double(host_legacy[i]) * double(host_legacy[i]);
                    }
                    const double oracle_rel_max = oracle_max > 0.0 ? oracle_max_abs / oracle_max : oracle_max_abs;
                    const double oracle_rel_rms = std::sqrt(oracle_sum_sq / (oracle_ref_sq + 1e-30));
                    const double fp32_rel_max = max_val > 0.0 ? fp32_max_abs / max_val : fp32_max_abs;
                    const double fp32_rel_rms = std::sqrt(fp32_sum_sq / (fp32_ref_sq + 1e-30));
                    // Gate: the kernel must reproduce the BF16-activation oracle to
                    // FP32 reassociation precision. The deviation from the FP32-x
                    // reference is the documented BF16 rounding of the activation
                    // (informational: this synthetic activation is a slow sinusoid,
                    // so its dot products are cancellation-dominated and the
                    // relative figure is far above the ~1.6e-3 measured on
                    // random-sign data and on the real model).
                    native_validated = oracle_rel_max <= 1e-4;
                    all_validated = all_validated && native_validated;
                    std::printf("PREFILL-GEMM native check: %s %s M=%u | vs bf16-x oracle "
                                "rel_rms=%.3e rel_max=%.3e (tol 1e-4) %s | vs fp32-x reference "
                                "rel_rms=%.3e rel_max=%.3e\n",
                                format_label(format), shape.name, m, oracle_rel_rms,
                                oracle_rel_max, native_validated ? "ok" : "FAILED",
                                fp32_rel_rms, fp32_rel_max);
                    std::fflush(stdout);
                    host_native.clear();
                    host_native.shrink_to_fit();
                }
                host_oracle.clear();
                host_oracle.shrink_to_fit();
                host_legacy.clear();
                host_legacy.shrink_to_fit();
                const double x_bytes = static_cast<double>(m) * shape.k * sizeof(float);
                const double y_bytes = static_cast<double>(m) * shape.n * sizeof(float);
                const double gflop = 2.0 * double(m) * shape.n * shape.k * 1.0e-9;
                const auto report = [&](const char *impl, uint32_t pass_rows,
                                        double min_ms, double med_ms, const char *ok_text) {
                    const double passes = std::ceil(double(m) / double(pass_rows));
                    const double bytes = weight_bytes * passes + x_bytes + y_bytes;
                    std::printf("RESULT,prefill-gemm,%s,%s,%s,%u,%u,%u,%u,%g,%.4f,%.9f,%.9f,"
                                "%.3f,%.3f,%.3f,%s\n",
                                format_label(format), shape.name, impl, m, shape.n, shape.k,
                                pass_rows, passes, gflop, min_ms, med_ms,
                                weight_bytes * passes * 1.0e-6, x_bytes * 1.0e-6,
                                bytes / med_ms * 1.0e-6, ok_text);
                    std::fflush(stdout);
                };
                const auto measure = [&](bool use_mt, auto &dst, const char *impl, uint32_t pass_rows) {
                    auto warmup = record(use_mt, dst, x);
                    stream << warmup.commit() << synchronize();
                    luisa::vector<double> samples;
                    samples.reserve(iters);
                    for (uint32_t it = 0; it < iters; ++it) {
                        auto list = record(use_mt, dst, x);
                        Clock clock;
                        clock.tic();
                        stream << list.commit() << synchronize();
                        samples.emplace_back(clock.toc());
                    }
                    const auto min_ms = *std::min_element(samples.begin(), samples.end());
                    const auto med_ms = median(samples);
                    report(impl, pass_rows, min_ms, med_ms, validated ? "ok" : "FAILED");
                };
                measure(true, y_mt, "mt", mt_cfg.mt);
                measure(false, y_legacy, "legacy", 16u);
                if (native_ok) {
                    auto warmup = record_native();
                    stream << warmup.commit() << synchronize();
                    luisa::vector<double> samples;
                    samples.reserve(iters);
                    for (uint32_t it = 0; it < iters; ++it) {
                        auto list = record_native();
                        Clock clock;
                        clock.tic();
                        stream << list.commit() << synchronize();
                        samples.emplace_back(clock.toc());
                    }
                    const auto min_ms = *std::min_element(samples.begin(), samples.end());
                    const auto med_ms = median(samples);
                    report("native", native_cfg.block_m, min_ms, med_ms,
                           native_validated ? "ok" : "FAILED");
                } else {
                    std::printf("RESULT,prefill-gemm,%s,%s,native,%u,%u,%u,0,0,%g,-1,-1,0,0,0,skipped\n",
                                format_label(format), shape.name, m, shape.n, shape.k, gflop);
                    std::fflush(stdout);
                }
                // ---- CUDA native Tile IR FP8 arm (OPT-IN, --cuda-tile-fp8) ----
                // An ADDITIONAL impl row over the same `w`, `bw_scales` and FP32 `x`.
                // Everything above is computed and printed exactly as before; this
                // block only runs when the caller opted in and prints either one
                // measured `cuda_tile` row or one explicit `skipped` row with the
                // verbatim reason (unsupported format, unavailable device/toolkit,
                // shape gate, or a ragged-N refusal) - it never substitutes another
                // implementation and never aborts the benchmark.
                if (options.cuda_tile_fp8) {
                    auto tile_row = run_cuda_tile_gemm_row(
                        device, stream, tile_gemm, tile_cfg, format, shape, m, iters,
                        x, w, bw_scales, y_legacy, y_oracle, y_native, native_ok);
                    if (!tile_row.measured) {
                        tile_rows_skipped++;
                        std::printf("RESULT,prefill-gemm,%s,%s,cuda_tile,%u,%u,%u,%u,0,%g,"
                                    "-1,-1,0,0,0,skipped\n",
                                    format_label(format), shape.name, m, shape.n, shape.k,
                                    tile_cfg.tile_m, gflop);
                        std::printf("PREFILL-GEMM cuda_tile skip: %s %s M=%u N=%u K=%u: %s\n",
                                    format_label(format), shape.name, m, shape.n, shape.k,
                                    tile_row.skip_reason.c_str());
                        std::fflush(stdout);
                    } else {
                        tile_rows_measured++;
                        std::printf("PREFILL-GEMM cuda_tile check: %s %s M=%u %ux%ux%u | "
                                    "finite=%s (%zu non-finite) | vs fp32-x DSL reference "
                                    "rel_rms=%.3e rel_max=%.3e (tol %.0e) %s | vs bf16-x DSL "
                                    "oracle rel_rms=%.3e rel_max=%.3e (info) | vs native fp8 "
                                    "rel_rms=%.3e rel_max=%.3e (info)\n",
                                    format_label(format), shape.name, m, tile_cfg.tile_m,
                                    tile_cfg.tile_n, tile_cfg.tile_k,
                                    tile_row.non_finite == 0u ? "yes" : "NO",
                                    tile_row.non_finite, tile_row.ref_rel_rms,
                                    tile_row.ref_rel_max, kCudaTileFp32Tol,
                                    tile_row.validated ? "ok" : "FAILED",
                                    tile_row.oracle_rel_rms, tile_row.oracle_rel_max,
                                    tile_row.have_native ? tile_row.native_rel_rms : -1.0,
                                    tile_row.have_native ? tile_row.native_rel_max : -1.0);
                        std::fflush(stdout);
                        // The Tile kernel streams the weight tile once per
                        // (tile_m rows) output tile, so its pass_rows IS tile_m: the
                        // SAME weight-traffic model as every other row of this arm.
                        report("cuda_tile", tile_cfg.tile_m, tile_row.min_ms, tile_row.med_ms,
                               tile_row.validated ? "ok" : "FAILED");
                        all_validated = all_validated && tile_row.validated;
                    }
                }
            }
        }
        native_gemm.reset();
    }
    if (options.cuda_tile_fp8) {
        // Reporting, not amortization: the lazy compiles are what a prompt pays on
        // its first prefill of each static shape, so the total is stated separately
        // from every sample above.
        std::printf("PREFILL-GEMM cuda_tile total: %u row(s) measured, %u row(s) skipped, "
                    "%u lazy compile(s), %.1f ms JIT, %u cache hit(s), %u shader(s), "
                    "create probe %.1f ms, tile %ux%ux%u\n",
                    tile_rows_measured, tile_rows_skipped, tile_gemm.compile_count(),
                    tile_gemm.jit_ms(), tile_gemm.cache_hits(), tile_gemm.shader_count(),
                    tile_gemm.probe_ms(), tile_cfg.tile_m, tile_cfg.tile_n, tile_cfg.tile_k);
        std::fflush(stdout);
        tile_gemm.reset();
    }
    std::printf("PREFILL-GEMM summary: validated=%s\n", all_validated ? "ok" : "FAILED");
    return all_validated ? 0 : 3;
}


int run_prefill_attention(Device &device, Stream &stream, const Options &options) {
    // fp8 KV takes per-(head,64-block) scale buffers; the FP32 arm this benchmark
    // measures binds this 1-element dummy and never reads it (the scale load
    // folds away), so the measured launch/payload stays exactly the FP32 one.
    auto bscale_dummy = device.create_buffer<float>(1);
    constexpr uint32_t qh = 16u, kvh = 4u, hd = 256u;
    const auto scale = 1.0f / std::sqrt(static_cast<float>(hd));
    QwenAttnPrefillTiledConfig cfg{};
    const auto warp = device.compute_warp_size();
    const auto shared = static_cast<uint32_t>(device.compute_max_shared_memory_size());
    if (auto s = select_qwen35_attn_prefill_tiled_config(warp, shared, cfg); !s.ok()) {
        std::fprintf(stderr, "prefill attention: tiled config selection failed: %s (%s)\n",
                     s.message.c_str(), s.detail.c_str());
        return 2;
    }
    PrefillAttnOptions arm;
    arm.lengths = options.prefill_attn_lengths;
    if (options.prefill_attn_q_tile != 0u) { cfg.q_tile = options.prefill_attn_q_tile; }
    if (options.prefill_attn_k_tile != 0u) { cfg.k_tile = options.prefill_attn_k_tile; }
    if (auto s = validate_qwen35_attn_prefill_tiled_config(cfg); !s.ok()) {
        std::fprintf(stderr, "prefill attention: requested tiling is invalid: %s (%s)\n",
                     s.message.c_str(), s.detail.c_str());
        return 2;
    }
    arm.q_tile = cfg.q_tile;
    arm.k_tile = cfg.k_tile;
    arm.iters = options.prefill_attn_iters != 0u ? options.prefill_attn_iters : options.iters;
    std::printf("PREFILL_ATTN config: backend=%s warp=%u shared_report=%u B q_tile=%u k_tile=%u "
                "tile_bytes=%u iters=%u\n",
                luisa::string{device.backend_name()}.c_str(), warp, shared, cfg.q_tile, cfg.k_tile,
                qwen35_attn_prefill_tiled_shared_bytes(cfg), arm.iters);
    auto shader_legacy = device.compile(make_qwen_attn_prefill_kernel());
    auto shader_tiled = device.compile(make_qwen35_attn_prefill_tiled_kernel(cfg));
    // CUDA tensor-core FlashAttention-2 realization (opt_plan/09 N1b). Optional: it
    // needs the CUDA backend and a NativeShaderExt, and its cold NVRTC cost is
    // reported instead of amortized into the timings.
    QwenNativeAttn native_attn;
    bool have_native = false;
    {
        QwenNativeAttnConfig native_cfg{};
        if (luisa::string_view{device.backend_name()} != "cuda") {
            std::printf("PREFILL_ATTN native: unavailable (backend '%s' has no NativeShaderExt "
                        "CUDA_NVRTC route)\n", luisa::string{device.backend_name()}.c_str());
        } else if (auto s = native_attn.create(device, native_cfg); !s.ok()) {
            std::printf("PREFILL_ATTN native: unavailable (%s%s%s)\n", s.message.c_str(),
                        s.detail.empty() ? "" : ": ", s.detail.c_str());
        } else {
            have_native = true;
            std::printf("PREFILL_ATTN native: tile %ux%u (%u warps, %u threads, %u B static "
                        "shared), causal GQA 16 q / 4 kv heads, head_dim 256, cold JIT %.1f ms\n",
                        native_cfg.q_tile, native_cfg.kv_tile, native_cfg.warps(), native_cfg.block(),
                        qwen_native_attn_shared_bytes(native_cfg), native_attn.jit_ms());
        }
    }
    if (!shader_legacy.compile_ok() || !shader_tiled.compile_ok()) {
        std::fprintf(stderr, "prefill attention: shader compile failed\n");
        return 2;
    }
    std::printf("PREFILL_ATTN note: reference = make_qwen_attn_prefill_kernel (untiled, one thread "
                "per (token, q_head), runtime head_dim local accumulator); tiled = "
                "make_qwen35_attn_prefill_tiled_kernel. Both consume the same [capacity,kv_heads,hd] "
                "cache and produce the same gated output.\n");
    double speedup_1024 = 0.0;
    // Median wall time of each realization at S = 1024, so the native-vs-tiled ratio is
    // computed from the same run instead of from a cross-run assumption.
    double legacy_ms_1024 = 0.0;
    double tiled_ms_1024 = 0.0;
    double native_ms_1024 = 0.0;
    bool all_validated = true;
    for (auto tokens : arm.lengths) {
        const uint32_t pos0 = 0u;
        // Full cache: kv_len == capacity == tokens, the plan's capacity boundary.
        const uint32_t capacity = tokens;
        const uint32_t kv_len = pos0 + tokens;
        const size_t qn = static_cast<size_t>(tokens) * qh * hd;
        const size_t kn = static_cast<size_t>(capacity) * kvh * hd;
        auto q_host = luisa::vector<float>(qn);
        auto gate_host = luisa::vector<float>(qn);
        auto k_host = luisa::vector<float>(kn);
        auto v_host = luisa::vector<float>(kn);
        for (size_t i = 0; i < qn; ++i) {
            q_host[i] = 0.1f * std::sin(0.017f * static_cast<float>(i % 4096u));
            gate_host[i] = 0.2f * std::cos(0.011f * static_cast<float>(i % 4096u));
        }
        for (size_t i = 0; i < kn; ++i) {
            k_host[i] = 0.1f * std::sin(0.003f * static_cast<float>(i % 4096u));
            v_host[i] = 0.2f * std::cos(0.005f * static_cast<float>(i % 4096u));
        }
        auto bq = device.create_buffer<float>(qn);
        auto bg = device.create_buffer<float>(qn);
        auto bk = device.create_buffer<float>(kn);
        auto bv = device.create_buffer<float>(kn);
        auto bo_legacy = device.create_buffer<float>(qn);
        auto bo_tiled = device.create_buffer<float>(qn);
        auto bo_native = device.create_buffer<float>(qn);
        stream << bq.copy_from(luisa::span<const float>{q_host})
               << bg.copy_from(luisa::span<const float>{gate_host})
               << bk.copy_from(luisa::span<const float>{k_host})
               << bv.copy_from(luisa::span<const float>{v_host}) << synchronize();
        q_host.clear(); gate_host.clear(); k_host.clear(); v_host.clear();
        q_host.shrink_to_fit(); gate_host.shrink_to_fit();
        k_host.shrink_to_fit(); v_host.shrink_to_fit();
        auto plan_legacy = plan_qwen_attn_prefill_dispatch(tokens, qh);
        auto plan_tiled = plan_qwen35_attn_prefill_tiled_dispatch(tokens, cfg);
        if (!plan_legacy.ok() || !plan_tiled.ok() || plan_legacy.value->empty() || plan_tiled.value->empty()) {
            std::fprintf(stderr, "prefill attention: dispatch plan failed for S=%u\n", tokens);
            return 2;
        }
        const auto record = [&](bool tiled) {
            if (tiled) {
                return shader_tiled(bq.view(), bg.view(), bk.view(), bv.view(),
                              bscale_dummy.view(), bscale_dummy.view(), bo_tiled.view(),
                                    tokens, scale, pos0, kv_len, capacity)
                    .dispatch(qwen_dispatch_threads(*plan_tiled.value));
            }
            return shader_legacy(bq.view(), bg.view(), bk.view(), bv.view(),
                               bscale_dummy.view(), bscale_dummy.view(), bo_legacy.view(),
                                 tokens, qh, kvh, hd, scale, pos0, kv_len, capacity)
                .dispatch(qwen_dispatch_threads(*plan_legacy.value));
        };
        // Both paths once, then read both back: the arm is self-validating.
        stream << record(false) << record(true) << synchronize();
        const auto out_legacy = [&] {
            auto host = luisa::vector<float>(qn);
            stream << bo_legacy.view(0u, qn).copy_to(luisa::span{host}) << synchronize();
            return host;
        }();
        const auto out_tiled = [&] {
            auto host = luisa::vector<float>(qn);
            stream << bo_tiled.view(0u, qn).copy_to(luisa::span{host}) << synchronize();
            return host;
        }();
        bool finite = true;
        double max_abs = 0.0, max_rel = 0.0;
        double sum_sq = 0.0;
        for (size_t i = 0; i < qn; ++i) {
            if (!std::isfinite(out_tiled[i]) || !std::isfinite(out_legacy[i])) {
                finite = false;
                break;
            }
            const auto d = std::fabs(out_tiled[i] - out_legacy[i]);
            const auto r = d / std::max(1.0e-6f, std::fabs(out_legacy[i]));
            max_abs = std::max(max_abs, static_cast<double>(d));
            max_rel = std::max(max_rel, static_cast<double>(r));
            sum_sq += static_cast<double>(out_legacy[i]) * static_cast<double>(out_legacy[i]);
        }
        // FP32 reassociation envelope of opt_plan/02 section 5 item 1: the tiled
        // kernel reorders the softmax denominator and the P.V accumulation.
        const bool validated = finite && max_abs <= 2.0e-3 && max_rel <= 2.0e-2;
        all_validated = all_validated && validated;
        std::printf("PREFILL_ATTN validate: S=%u finite=%s max_abs=%.3e max_rel=%.3e "
                    "legacy_rms=%.6g\n",
                    tokens, finite ? "yes" : "NO", max_abs, max_rel,
                    std::sqrt(sum_sq / static_cast<double>(qn)));
        // Native (BF16 tensor-core) realization: dispatch on the same buffers, then
        // validate against the FP32 reference kernel AND against the DSL tiled kernel,
        // so the documented BF16 tolerance is measured instead of assumed.
        bool native_validated = false;
        double native_abs_vs_ref = 0.0, native_rel_vs_ref = 0.0;
        double native_abs_vs_tiled = 0.0, native_rel_vs_tiled = 0.0;
        if (have_native) {
            auto command = native_attn.build_dispatch(bq.view(), bg.view(), bk.view(), bv.view(),
                                                     bo_native.view(), tokens, scale, pos0, kv_len,
                                                     capacity);
            if (command == nullptr) {
                std::printf("PREFILL_ATTN native: dispatch build failed for S=%u\n", tokens);
                have_native = false;
            } else {
                stream << std::move(command) << synchronize();
                auto out_native = luisa::vector<float>(qn);
                stream << bo_native.view(0u, qn).copy_to(luisa::span{out_native}) << synchronize();
                const auto measure = [&](const luisa::vector<float> &want, double &abs_out,
                                         double &rel_out) {
                    double max_abs = 0.0, max_ref = 0.0;
                    for (size_t i = 0; i < qn; ++i) {
                        const auto d = std::fabs(out_native[i] - want[i]);
                        max_abs = std::max(max_abs, static_cast<double>(d));
                        max_ref = std::max(max_ref, std::fabs(static_cast<double>(want[i])));
                    }
                    abs_out = max_abs;
                    rel_out = max_ref > 0.0 ? max_abs / max_ref : max_abs;
                };
                measure(out_legacy, native_abs_vs_ref, native_rel_vs_ref);
                measure(out_tiled, native_abs_vs_tiled, native_rel_vs_tiled);
                // The documented envelope of the BF16 path (Q/K/V and P rounded to
                // BF16, FP32 accumulation): see docs/examples/example_qwen.md.
                native_validated = std::isfinite(native_rel_vs_ref) &&
                                   native_rel_vs_ref <= 1.0e-2 && native_rel_vs_tiled <= 1.0e-2;
                all_validated = all_validated && native_validated;
                std::printf("PREFILL_ATTN validate native: S=%u vs_ref max_abs=%.3e rel=%.3e | "
                            "vs_tiled max_abs=%.3e rel=%.3e -> %s\n",
                            tokens, native_abs_vs_ref, native_rel_vs_ref, native_abs_vs_tiled,
                            native_rel_vs_tiled, native_validated ? "ok" : "FAILED");
            }
        }
        // Iteration budget per realization: the untiled reference kernel is the
        // O(S^2) local-memory one (seconds at S = 4096), so its arm is sampled less.
        const uint32_t legacy_iters = tokens >= 2048u ? std::max(3u, arm.iters / 6u) : arm.iters;
        if (have_native && native_validated) {
            const auto record_native = [&] {
                return native_attn.build_dispatch(bq.view(), bg.view(), bk.view(), bv.view(),
                                                 bo_native.view(), tokens, scale, pos0, kv_len,
                                                 capacity);
            };
            stream << record_native() << synchronize();
            luisa::vector<double> samples;
            samples.reserve(arm.iters);
            for (uint32_t it = 0; it < arm.iters; ++it) {
                Clock clock;
                clock.tic();
                stream << record_native() << synchronize();
                samples.emplace_back(clock.toc());
            }
            const auto min_ms = *std::min_element(samples.begin(), samples.end());
            const auto med_ms = median(samples);
            print_prefill_attn("native", "causal_gqa", arm, tokens, pos0, capacity, min_ms, med_ms,
                               prefill_attn_flops(tokens, qh, hd));
            if (tokens == 1024u) { native_ms_1024 = med_ms; }
        }
        for (auto tiled : {false, true}) {
            // warmup
            stream << record(tiled) << synchronize();
            luisa::vector<double> samples;
            samples.reserve(arm.iters);
            const uint32_t row_iters = tiled ? arm.iters : legacy_iters;
            for (uint32_t it = 0; it < row_iters; ++it) {
                Clock clock;
                clock.tic();
                stream << record(tiled) << synchronize();
                samples.emplace_back(clock.toc());
            }
            const auto min_ms = *std::min_element(samples.begin(), samples.end());
            const auto med_ms = median(samples);
            print_prefill_attn(tiled ? "tiled" : "legacy", "causal_gqa", arm, tokens, pos0, capacity,
                               min_ms, med_ms, prefill_attn_flops(tokens, qh, hd));
            if (tokens == 1024u) {
                if (!tiled) {
                    legacy_ms_1024 = med_ms;
                } else {
                    tiled_ms_1024 = med_ms;
                }
            }
        }
    }
    speedup_1024 = (legacy_ms_1024 > 0.0 && tiled_ms_1024 > 0.0) ? legacy_ms_1024 / tiled_ms_1024 : 0.0;
    const double speedup_1024_native =
        (tiled_ms_1024 > 0.0 && native_ms_1024 > 0.0) ? tiled_ms_1024 / native_ms_1024 : 0.0;
    std::printf("PREFILL_ATTN summary: speedup_at_1024=%.3fx native_vs_tiled_at_1024=%.3fx "
                "(legacy %.4f ms, tiled %.4f ms, native %.4f ms) validated=%s "
                "(acceptance gate: >= 10x)\n",
                speedup_1024, speedup_1024_native, legacy_ms_1024, tiled_ms_1024, native_ms_1024,
                all_validated ? "ok" : "FAILED");
    std::printf("PREFILL_ATTN note: GFLOP/s is causal 2*S*(S+1)*head_dim*q_heads over the measured "
                "wall time; compare the 'legacy'/'tiled' rows with the tier's FP32 peak "
                "(opt_plan/08 section 2.3) and the 'native' row with the tier's BF16 TENSOR-CORE "
                "peak, never with a vendor number. The native row is the CUDA_NVRTC "
                "FlashAttention-2 kernel: Q/K/V and P are rounded to BF16, everything else is "
                "FP32, and its deviation from both references is printed per length above.\n");
    std::fflush(stdout);
    return all_validated ? 0 : 3;
}


  // ---------------------------------------------------------------------------
  // Isolated single-query DECODE attention arm (opt_plan/03 section 4.1 vs 4.2).
  //
  // Runs the legacy per-q_head kernel (one block per q_head, every one of the 4
  // q_heads of a KV head re-streaming the same K/V rows: 4x read amplification)
  // and the split-KV + GQA-dedup family (one block per (kv_head, key_split)
  // staging K/V once for all 4 q heads, plus a combine kernel that applies the
  // sigmoid gate) on the SAME buffers at the plan's context set, validates them
  // against each other, and reports min/median wall time with the byte model of
  // each implementation.
  //
  // Everything in the `unique_kv_bytes` / `read_kv_bytes` columns is a MODEL, not
  // a hardware counter. It states how many KV bytes each implementation MUST
  // stream for this kv_len (legacy 4x, split 1x); the achieved GB/s column is that
  // model divided by the measured time, so it is comparable to the calibration
  // roof of --calibrate but never claims to be a DRAM counter. `--decode-attn-k-tile`
  // and `--decode-attn-max-splits` override the device-selected split config, which
  // is how the tile/split-bound sweep in .kimix_cache/qwen_benchmark/phase5 was run.
  // ---------------------------------------------------------------------------
  int run_decode_attention(Device &device, Stream &stream, const Options &options) {
      constexpr uint32_t qh = 16u, kvh = 4u, hd = 256u;
      const uint32_t iters = options.decode_attn_iters != 0u ? options.decode_attn_iters : 20u;
      uint32_t capacity = 1u;
      for (auto length : options.decode_attn_lengths) { capacity = std::max(capacity, length); }
      QwenAttnDecodeSplitConfig cfg{};
      // Same selection rule the engine uses: warp size and the REPORTED shared
      // memory limit, with the documented default bound when no SM count is
      // queryable (the pinned API exposes none; see qwen/caps.h).
      auto selected = select_qwen_attn_decode_split_config(
          static_cast<uint32_t>(device.compute_warp_size()),
          static_cast<uint32_t>(device.compute_max_shared_memory_size()), 0u, cfg);
      if (!selected.ok()) {
          std::fprintf(stderr, "decode attention: split config unavailable (%s: %s)\n",
                       selected.message.c_str(), selected.detail.c_str());
          return 2;
      }
      if (options.decode_attn_k_tile != 0u) { cfg.k_tile = options.decode_attn_k_tile; }
      if (options.decode_attn_max_splits != 0u) { cfg.max_splits = options.decode_attn_max_splits; }
      // The arm isolates the KERNELS: both run at every length, so each line is a
      // measurement of that implementation at that kv_len rather than of the
      // engine's threshold policy (which the engine A/B covers).
      cfg.threshold = 1u;
      if (auto s = validate_qwen_attn_decode_split_config(cfg); !s.ok()) {
          std::fprintf(stderr, "decode attention: invalid split config (%s: %s)\n",
                       s.message.c_str(), s.detail.c_str());
          return 2;
      }
      const uint32_t sm_count = 0u;// not queryable through the pinned API
      const size_t qn = static_cast<size_t>(qh) * hd;
      const size_t kvn = static_cast<size_t>(capacity) * kvh * hd;
      const size_t acc_elems = static_cast<size_t>(cfg.max_splits) * qh * hd;
      const size_t ml_elems = static_cast<size_t>(cfg.max_splits) * qh * 2u;
      auto q_host = luisa::vector<float>(qn);
      auto gate_host = luisa::vector<float>(qn);
      auto k_host = luisa::vector<float>(kvn);
      auto v_host = luisa::vector<float>(kvn);
      for (size_t i = 0; i < qn; ++i) {
          q_host[i] = 0.1f * std::sin(0.017f * static_cast<float>(i % 4096u));
          gate_host[i] = 0.2f * std::cos(0.011f * static_cast<float>(i % 4096u));
      }
      for (size_t i = 0; i < kvn; ++i) {
          k_host[i] = 0.1f * std::sin(0.003f * static_cast<float>(i % 4096u));
          v_host[i] = 0.2f * std::cos(0.005f * static_cast<float>(i % 4096u));
      }
      auto bq = device.create_buffer<float>(qn);
      auto bg = device.create_buffer<float>(qn);
      auto bk = device.create_buffer<float>(kvn);
      auto bv = device.create_buffer<float>(kvn);
      auto out_legacy = device.create_buffer<float>(qn);
      auto out_split = device.create_buffer<float>(qn);
      auto partial_acc = device.create_buffer<float>(acc_elems);
      auto partial_ml = device.create_buffer<float>(ml_elems);
      stream << bq.copy_from(luisa::span<const float>{q_host})
             << bg.copy_from(luisa::span<const float>{gate_host})
             << bk.copy_from(luisa::span<const float>{k_host})
             << bv.copy_from(luisa::span<const float>{v_host}) << synchronize();
      q_host.clear(); gate_host.clear(); k_host.clear(); v_host.clear();
      q_host.shrink_to_fit(); gate_host.shrink_to_fit(); k_host.shrink_to_fit(); v_host.shrink_to_fit();
      // fp8 KV scale arguments: the FP32 arm binds a 1-element dummy (never read, the
      // scale load folds away), so the measured launch/payload is the FP32 one.
      auto bscale_dummy = device.create_buffer<float>(1);
      auto legacy = device.compile(make_qwen_attn_decode_kernel());
      auto split = device.compile(make_qwen_attn_decode_split_kernel(cfg));
      auto combine = device.compile(make_qwen_attn_decode_combine_kernel(cfg));
      auto legacy_plan = plan_qwen_attn_decode_dispatch(qh);
      auto split_plan = plan_qwen_attn_decode_split_dispatch(cfg);
      auto combine_plan = plan_qwen_attn_decode_combine_dispatch(cfg);
      if (!legacy.compile_ok() || !split.compile_ok() || !combine.compile_ok() ||
          !legacy_plan.ok() || !split_plan.ok() || !combine_plan.ok()) {
          std::fprintf(stderr, "decode attention: shader compile or plan failed\n");
          return 2;
      }
      const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
      std::printf("DECODE_ATTN config: q_heads=%u kv_heads=%u head_dim=%u capacity=%u k_tile=%u "
                  "max_splits=%u split_shared=%u B partial_workspace=%.1f KiB "
                  "sm_count=%u (0 = unqueryable: blocks/SM unavailable)\n",
                  qh, kvh, hd, capacity, cfg.k_tile, cfg.max_splits,
                  qwen_attn_decode_split_shared_bytes(cfg),
                  static_cast<double>(qwen_attn_decode_split_partial_bytes(cfg)) / 1024.0,
                  sm_count);
      std::printf("DECODE_ATTN columns: impl,kv_len,splits,blocks,repeats,unique_kv_bytes,"
                  "read_kv_bytes,min_ms,median_ms,model_read_GBps\n");
      std::printf("DECODE_ATTN note: unique_kv_bytes is the K+V that ONE (layer, token) must "
                  "stream at this kv_len (kv_len*kv_heads*head_dim*2*4 B); read_kv_bytes is that "
                  "times the amplification of the implementation (legacy 4x, split 1x). Both are "
                  "MODELS of the traffic, not counters; model_read_GBps is model/time.\n");
      std::fflush(stdout);
      bool all_validated = true;
      for (auto kv_len : options.decode_attn_lengths) {
          const uint32_t splits = qwen_attn_decode_split_count(kv_len, cfg);
          const uint32_t legacy_blocks = qh;
          const uint32_t split_blocks = qwen_attn_decode_split_blocks(cfg) + qh;
          // repeats: keep every sample at a measurable duration without turning the
          // long-context arms into minutes.
          const uint32_t repeats = std::max(1u, std::min(64u, 32768u / std::max(1u, kv_len)));
          // Both implementations once at THIS kv_len: the kernel argument is kv_len,
          // so each length must be measured with its own dispatch value.
          const auto record_len = [&](bool split_impl) {
              auto list = CommandList::create(static_cast<size_t>(repeats));
              if (split_impl) {
                  for (uint32_t i = 0; i < repeats; ++i) {
                      list << split(bq.view(), bk.view(), bv.view(),
                    bscale_dummy.view(), bscale_dummy.view(), partial_acc.view(),
                                    partial_ml.view(), kv_len, scale, capacity)
                                  .dispatch(qwen_dispatch_threads(*split_plan.value));
                      list << combine(partial_acc.view(), partial_ml.view(), bg.view(),
                                      out_split.view(), kv_len, capacity)
                                  .dispatch(qwen_dispatch_threads(*combine_plan.value));
                  }
              } else {
                  for (uint32_t i = 0; i < repeats; ++i) {
                      list << legacy(bq.view(), bg.view(), bk.view(), bv.view(),
                     bscale_dummy.view(), bscale_dummy.view(),
                                     out_legacy.view(), qh, kvh, hd, scale, kv_len, capacity)
                                  .dispatch(qwen_dispatch_threads(*legacy_plan.value));
                  }
              }
              return list;
          };
          {
              if (options.decode_attn_impl_legacy) {
                  auto legacy_list = record_len(false);
                  stream << legacy_list.commit();
              }
              if (options.decode_attn_impl_split) {
                  auto split_list = record_len(true);
                  stream << split_list.commit();
              }
              stream << synchronize();
          }
          const auto read_back = [&](Buffer<float> &buf) {
              auto host = luisa::vector<float>(qn);
              stream << buf.view(0u, qn).copy_to(luisa::span{host}) << synchronize();
              return host;
          };
          const auto got_legacy = read_back(out_legacy);
          const auto got_split = read_back(out_split);
          bool finite = true;
          double max_abs = 0.0, max_rel = 0.0, sum_sq = 0.0;
          for (size_t i = 0; i < qn; ++i) {
              if (!std::isfinite(got_split[i]) || !std::isfinite(got_legacy[i])) {
                  finite = false;
                  break;
              }
              const auto d = std::fabs(got_split[i] - got_legacy[i]);
              max_abs = std::max(max_abs, static_cast<double>(d));
              max_rel = std::max(max_rel, static_cast<double>(d) / std::max(1.0e-6f, std::fabs(got_legacy[i])));
              sum_sq += static_cast<double>(got_legacy[i]) * static_cast<double>(got_legacy[i]);
          }
          // The split reduction reorders the FP32 softmax summation, so the two
          // agree to an FP32 reassociation envelope rather than bit-exactly.
          // Elementwise RELATIVE error is not a criterion here (a gated output
          // element can be ~1e-6, and one ULP of the summation order is then 100%
          // "relative"): the gate uses the absolute bound, and the normalized error
          // max_abs / rms(legacy) -- the same shape as the tiled prefill acceptance
          // in opt_plan/02 section 5 -- states how large the difference is against
          // the magnitude of the result.
          const auto rms_legacy = std::sqrt(sum_sq / static_cast<double>(qn));
          const auto normalized = rms_legacy > 0.0 ? max_abs / rms_legacy : 0.0;
          // With --decode-attn-impl only ONE realization runs, so there is nothing to
          // compare against: the arm then validates the finiteness of what ran and
          // says so, instead of diffing against a buffer the other impl never wrote.
          const bool both = options.decode_attn_impl_legacy && options.decode_attn_impl_split;
          const bool validated = both ? (finite && max_abs <= 2.0e-3 && normalized <= 2.0e-2)
                                     : (finite && (options.decode_attn_impl_legacy
                                                       ? std::isfinite(got_legacy[0])
                                                       : std::isfinite(got_split[0])));
          all_validated = all_validated && validated;
          std::printf("DECODE_ATTN validate: kv_len=%u finite=%s max_abs=%.3e max_rel_elementwise=%.3e "
                      "rms_legacy=%.6g max_abs_over_rms=%.3e splits=%u cross_impl=%s\n",
                      kv_len, finite ? "yes" : "NO", max_abs, max_rel, rms_legacy, normalized, splits,
                      both ? "both" : "single-impl (--decode-attn-impl)");
          const double unique_bytes = static_cast<double>(kv_len) * kvh * hd * 2.0 * 4.0;
          for (auto split_impl : {false, true}) {
            if (split_impl ? !options.decode_attn_impl_split
                           : !options.decode_attn_impl_legacy) {
                continue;
            }
              {
                  auto warm = record_len(split_impl);
                  stream << warm.commit() << synchronize();// warmup
              }
              luisa::vector<double> samples;
              samples.reserve(iters);
              for (uint32_t it = 0; it < iters; ++it) {
                  Clock clock;
                  clock.tic();
                  auto list = record_len(split_impl);
                  stream << list.commit() << synchronize();
                  samples.emplace_back(clock.toc() / static_cast<double>(repeats));
              }
              const auto min_ms = *std::min_element(samples.begin(), samples.end());
              const auto med_ms = median(samples);
              const double read_bytes = unique_bytes * (split_impl ? 1.0 : static_cast<double>(qh / kvh));
              const double gbps = med_ms > 0.0 ? read_bytes / med_ms * 1.0e-6 : 0.0;
              std::printf("DECODE_ATTN,%s,%u,%u,%u,%u,%.0f,%.0f,%.9f,%.9f,%.3f\n",
                          split_impl ? "split" : "legacy", kv_len,
                          split_impl ? splits : 0u,
                          split_impl ? split_blocks : legacy_blocks, repeats,
                          unique_bytes, read_bytes, min_ms, med_ms, gbps);
              std::fflush(stdout);
          }
      }
      // Speedup summary at the plan's long-context points (split / legacy medians
      // are re-derived from the same buffer addresses, so this is the same
      // measurement restated).
      std::printf("DECODE_ATTN note: compare columns row by row; a split row with "
                  "read_kv_bytes 4x smaller and a shorter median_ms is the dedup win "
                  "(opt_plan/03 section 4.2). validated=%s\n",
                  all_validated ? "ok" : "FAILED");
      std::fflush(stdout);
      return all_validated ? 0 : 3;
  }

}// namespace

// ---------------------------------------------------------------------------
// --rmsnorm: isolated zero-centered RMSNorm arm (opt_plan/09 N3.1).
//
// Runs the realizations the engine can pick from on the SAME buffers at the
// model geometry (hidden = 2560) and validates every one against the device
// reference output of the generic wide kernel (bit-exact is expected for the
// register-cached variant: it keeps the per-thread element order, the warp
// partial order and the epilogue expression) plus the CPU oracle.
//
// The "global passes" column is a MODEL of the activation traffic: the generic
// kernel reads the row twice (x for the sum of squares, then x or sum_out for
// the epilogue) and the norm weight twice; the register-cached one reads each
// once. It says how many full-row global read passes each realization must
// issue, which is exactly the quantity N3.1 removes -- it is not a DRAM counter.
// ---------------------------------------------------------------------------
struct RmsNormImpl {
    const char *name;
    uint32_t block;
    bool reg;
};

int run_rmsnorm(Device &device, Stream &stream, const Options &options) {
    const uint32_t dim = qwen35_rmsnorm_model_dim;
    const uint32_t iters = options.rmsnorm_iters != 0u ? options.rmsnorm_iters : 50u;
    const RmsNormImpl impls[] = {
        {"warp32", 128u, false},
        {"wide128", 128u, false},
        {"wide256", 256u, false},
        {"reg128", 128u, true},
        {"reg256", 256u, true},
    };
    std::printf("RMSNORM columns: epilogue,impl,rows,dim,block,global_passes,min_ms,median_ms,model_GBps\n");
    for (uint32_t rows : options.rmsnorm_rows) {
        const size_t count = size_t(rows) * dim;
        luisa::vector<float> x(count), w(dim), residual(count);
        for (size_t i = 0; i < count; ++i) {
            x[i] = std::sin(0.013f * static_cast<float>(i)) * 1.5f;
            residual[i] = std::cos(0.0071f * static_cast<float>(i)) * 0.5f;
        }
        for (uint32_t i = 0; i < dim; ++i) { w[i] = 0.5f + 0.001f * static_cast<float>(i % 97u); }
        auto bx = device.create_buffer<float>(count);
        auto bw = device.create_buffer<float>(dim);
        auto br = device.create_buffer<float>(count);
        auto by = device.create_buffer<float>(count);
        auto by_ref = device.create_buffer<float>(count);
        auto bs = device.create_buffer<float>(count);
        auto bdummy = device.create_buffer<float>(1u);
        stream << bx.copy_from(luisa::span<const float>{x})
               << bw.copy_from(luisa::span<const float>{w})
               << br.copy_from(luisa::span<const float>{residual}) << synchronize();
        for (auto epi : {QwenNormEpilogue::plain, QwenNormEpilogue::snapshot}) {
            const auto bres = QwenNormEpilogue::residual == epi ? br : bdummy;
            const auto bsum = epi == QwenNormEpilogue::plain ? bdummy : bs;
            const char *epi_name = epi == QwenNormEpilogue::snapshot ? "snapshot" : "plain";
            // Reference output + CPU oracle for the epilogue.
            {
                auto sh = device.compile(make_qwen35_rmsnorm_kernel(epi, 128u));
                auto plan = plan_qwen35_rmsnorm_dispatch(rows, dim, 128u);
                if (!plan.ok()) { return 2; }
                stream << sh(bx, bw, bres, by_ref, bsum, rows, dim, 1e-6f)
                              .dispatch(qwen_dispatch_threads(*plan.value));
            }
            luisa::vector<float> ref(count), sum;
            if (epi == QwenNormEpilogue::snapshot) {
                if (!ref_qwen_rmsnorm_snapshot(x, w, rows, dim, 1e-6f, ref, sum).ok()) { return 2; }
            } else if (!ref_qwen_rmsnorm(x, w, residual, false, rows, dim, 1e-6f, ref, sum).ok()) {
                return 2;
            }
            luisa::vector<float> want(count);
            stream << by_ref.view(0u, count).copy_to(luisa::span{want}) << synchronize();
            double oracle_max = 0.0;
            for (size_t i = 0; i < count; ++i) {
                oracle_max = std::max(oracle_max, std::fabs(static_cast<double>(want[i]) - ref[i]));
            }
            std::printf("RMSNORM validate: epilogue=%s rows=%u oracle_max_abs=%.3e\n",
                        epi_name, rows, oracle_max);
            for (const auto &impl : impls) {
                if (impl.block == 0u) { continue; }
                auto kernel = impl.reg
                                  ? make_qwen35_rmsnorm_reg_kernel(epi, dim, impl.block)
                                  : (impl.block == 128u && !impl.reg && std::string_view{impl.name} == "warp32"
                                         ? make_qwen_rmsnorm_kernel(epi)
                                         : make_qwen35_rmsnorm_kernel(epi, impl.block));
                auto plan = impl.reg
                                ? plan_qwen35_rmsnorm_reg_dispatch(rows, dim, impl.block)
                                : (std::string_view{impl.name} == "warp32"
                                       ? plan_qwen_rmsnorm_dispatch(rows, dim)
                                       : plan_qwen35_rmsnorm_dispatch(rows, dim, impl.block));
                if (!plan.ok()) {
                    std::printf("RMSNORM skip: impl=%s plan failed (%s)\n", impl.name, plan.status.message);
                    continue;
                }
                auto sh = device.compile(kernel);
                const auto record = [&] {
                    return sh(bx, bw, bres, by, bsum, rows, dim, 1e-6f)
                        .dispatch(qwen_dispatch_threads(*plan.value));
                };
                stream << record() << synchronize();
                luisa::vector<float> got(count);
                stream << by.view(0u, count).copy_to(luisa::span{got}) << synchronize();
                double max_abs = 0.0;
                bool exact = true;
                for (size_t i = 0; i < count; ++i) {
                    max_abs = std::max(max_abs, std::fabs(static_cast<double>(got[i]) - want[i]));
                    if (got[i] != want[i]) { exact = false; }
                }
                // One sample = RMSNORM_BATCH independent launches of the SAME
                // kernel (a decode step issues one per layer), so the reported
                // per-launch cost is not one stream-synchronization each.
                constexpr uint32_t batch = 64u;
                const auto record_batch = [&] {
                    auto flag = record();
                    for (uint32_t r = 1u; r < batch; ++r) { flag = record(); }
                    return flag;
                };
                stream << record_batch() << synchronize();
                luisa::vector<double> samples;
                samples.reserve(iters);
                for (uint32_t it = 0; it < iters; ++it) {
                    Clock clock;
                    clock.tic();
                    stream << record_batch() << synchronize();
                    samples.emplace_back(clock.toc() / static_cast<double>(batch));
                }
                const auto min_ms = *std::min_element(samples.begin(), samples.end());
                const auto med_ms = median(samples);
                // Model: 1 write of y, `passes` reads of the row, 1 or 2 reads of w.
                const double passes = impl.reg ? 1.0 : 2.0;
                const double model_bytes = passes * static_cast<double>(count) * 4.0 +
                                           static_cast<double>(count) * 4.0 +
                                           (impl.reg ? 1.0 : 2.0) * static_cast<double>(dim) * 4.0;
                std::printf("RESULT,RMSNORM,%s,%s,%u,%u,%u,%.1f,%.9f,%.9f,%.3f  validate max_abs=%.3e bit_exact=%s\n",
                            epi_name, impl.name, rows, dim, impl.block, passes,
                            min_ms, med_ms, model_bytes / med_ms * 1.0e-6,
                            max_abs, exact ? "yes" : "no");
                std::fflush(stdout);
                if (max_abs > 1.0e-5) {
                    std::fprintf(stderr, "RMSNORM FAILED: impl=%s differs from the wide kernel\n", impl.name);
                    return 3;
                }
            }
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// --gdn-chunk: isolated WY/UT chunk-64 arm (opt_plan/09 V4).
//
// Runs the generic runtime-extent kernel and the compile-time head_dim = 128
// specialization on the same model-geometry buffers, checks that their outputs
// and end states are identical (the specialization only makes extents constant,
// so it must not perturb a bit), and reports min/median wall time per call at
// every chunk boundary the plan requires. 32 V heads = 32 blocks: the arm states
// the grid explicitly so a speedup cannot be confused with a different schedule.
// ---------------------------------------------------------------------------
int run_gdn_chunk(Device &device, Stream &stream, const Options &options) {
    constexpr uint32_t qkh = 16u, vh = 32u, hd = qwen35_gdn_model_head_dim;
    constexpr uint32_t qkn = qkh * hd, vn = vh * hd, channels = 2u * qkn + vn;
    const uint32_t iters = options.gdn_chunk_iters != 0u ? options.gdn_chunk_iters : 20u;
    uint32_t max_tokens = 1u;
    for (auto t : options.gdn_chunk_lengths) { max_tokens = std::max(max_tokens, t); }
    const size_t qk_count = size_t(max_tokens) * qkn;
    const size_t conv_count = size_t(max_tokens) * channels;
    const size_t gate_count = size_t(max_tokens) * vh;
    const size_t out_count = gate_count * hd;
    const size_t state_count = size_t(vh) * hd * hd;
    luisa::vector<float> q(qk_count), k(qk_count), conv(conv_count), beta(gate_count), g(gate_count);
    luisa::vector<float> state0(state_count);
    for (size_t i = 0; i < qk_count; ++i) {
        q[i] = std::sin(0.017f * static_cast<float>(i));
        k[i] = std::cos(0.011f * static_cast<float>(i));
    }
    // Normalized q/k like the engine (q additionally scaled by 1/sqrt(hd)).
    for (uint32_t t = 0; t < max_tokens; ++t) {
        for (uint32_t h = 0; h < qkh; ++h) {
            const size_t base = (size_t(t) * qkh + h) * hd;
            double qs = 0.0, ks = 0.0;
            for (uint32_t d = 0; d < hd; ++d) {
                qs += double(q[base + d]) * q[base + d];
                ks += double(k[base + d]) * k[base + d];
            }
            const float qi = float(1.0 / std::sqrt(qs + 1e-6) / std::sqrt(double(hd)));
            const float ki = float(1.0 / std::sqrt(ks + 1e-6));
            for (uint32_t d = 0; d < hd; ++d) { q[base + d] *= qi; k[base + d] *= ki; }
        }
    }
    for (size_t i = 0; i < conv_count; ++i) { conv[i] = 0.5f * std::sin(0.0031f * static_cast<float>(i)); }
    for (size_t i = 0; i < gate_count; ++i) {
        beta[i] = 0.05f + 0.3f * std::fabs(std::sin(0.01f * static_cast<float>(i)));
        g[i] = -0.08f * std::fabs(std::cos(0.013f * static_cast<float>(i)));
    }
    for (size_t i = 0; i < state_count; ++i) { state0[i] = 0.05f * std::sin(0.0017f * static_cast<float>(i)); }
    auto bq = device.create_buffer<float>(qk_count);
    auto bk = device.create_buffer<float>(qk_count);
    auto bc = device.create_buffer<float>(conv_count);
    auto bb = device.create_buffer<float>(gate_count);
    auto bg = device.create_buffer<float>(gate_count);
    auto bs_generic = device.create_buffer<float>(state_count);
    auto bs_model = device.create_buffer<float>(state_count);
    auto bw = device.create_buffer<float>(out_count);
    auto bvn = device.create_buffer<float>(out_count);
    auto bo_generic = device.create_buffer<float>(out_count);
    auto bo_model = device.create_buffer<float>(out_count);
    stream << bq.copy_from(luisa::span<const float>{q})
           << bk.copy_from(luisa::span<const float>{k})
           << bc.copy_from(luisa::span<const float>{conv})
           << bb.copy_from(luisa::span<const float>{beta})
           << bg.copy_from(luisa::span<const float>{g}) << synchronize();
    auto sh_generic = device.compile(make_qwen_gdn_chunk_kernel());
    auto sh_model = device.compile(make_qwen35_gdn_chunk_kernel(hd));
    auto plan = plan_qwen_gdn_chunk_dispatch(vh);
    if (!plan.ok()) {
        std::fprintf(stderr, "gdn chunk: plan failed (%s)\n", plan.status.message);
        return 2;
    }
    std::printf("GDN_CHUNK columns: tokens,impl,blocks,threads,min_ms,median_ms\n");
    bool all_equal = true;
    for (uint32_t tokens : options.gdn_chunk_lengths) {
        const size_t out_n = size_t(tokens) * vh * hd;
        stream << bs_generic.copy_from(luisa::span<const float>{state0})
               << bs_model.copy_from(luisa::span<const float>{state0}) << synchronize();
        const auto launch = [&](bool model) {
            return (model ? sh_model : sh_generic)(bq.view(), bk.view(), bc.view(), bb.view(), bg.view(),
                                                  model ? bs_model.view() : bs_generic.view(),
                                                  bw.view(), bvn.view(),
                                                  model ? bo_model.view() : bo_generic.view(),
                                                  tokens, vh, hd, qkh, 2u * qkn)
                .dispatch(qwen_dispatch_threads(*plan.value));
        };
        stream << launch(false) << launch(true) << synchronize();
        luisa::vector<float> og(out_n), om(out_n), sg(state_count), sm(state_count);
        stream << bo_generic.view(0u, out_n).copy_to(luisa::span{og})
               << bo_model.view(0u, out_n).copy_to(luisa::span{om})
               << bs_generic.view(0u, state_count).copy_to(luisa::span{sg})
               << bs_model.view(0u, state_count).copy_to(luisa::span{sm}) << synchronize();
        double max_out = 0.0, max_state = 0.0;
        for (size_t i = 0; i < out_n; ++i) {
            max_out = std::max(max_out, std::fabs(static_cast<double>(og[i]) - om[i]));
        }
        for (size_t i = 0; i < state_count; ++i) {
            max_state = std::max(max_state, std::fabs(static_cast<double>(sg[i]) - sm[i]));
        }
        // CUDA reproduces the generic kernel bit for bit. DX/VK may contract the
        // same FP32 expression differently once the key-axis loop is unrolled, so
        // the contract checked here is the FP32 last-bit envelope (1e-6 absolute),
        // not bit equality; the CPU oracle checks live in test_qwen_ops_dsl.
        const bool equal = max_out <= 1.0e-6 && max_state <= 1.0e-6;
        all_equal = all_equal && equal;
        std::printf("GDN_CHUNK validate: tokens=%u max_out_abs=%.3e max_state_abs=%.3e within_fp32_envelope=%s\n",
                    tokens, max_out, max_state, equal ? "yes" : "no");
        for (auto model : {false, true}) {
            stream << launch(model) << synchronize();// warmup
            luisa::vector<double> samples;
            samples.reserve(iters);
            for (uint32_t it = 0; it < iters; ++it) {
                Clock clock;
                clock.tic();
                stream << launch(model) << synchronize();
                samples.emplace_back(clock.toc());
            }
            const auto min_ms = *std::min_element(samples.begin(), samples.end());
            const auto med_ms = median(samples);
            std::printf("RESULT,GDN_CHUNK,%u,%s,%u,%u,%.9f,%.9f\n",
                        tokens, model ? "model" : "generic", 1u, 128u, min_ms, med_ms);
            std::fflush(stdout);
        }
    }
    std::printf("GDN_CHUNK summary: fp32_envelope=%s (the specialization must not change a value)\n",
                all_equal ? "ok" : "FAILED");
    std::fflush(stdout);
    return all_equal ? 0 : 3;
}

// ---------------------------------------------------------------------------
// --gdn-decode: isolated DECODE GDN conv+SiLU+QK-norm arm (opt_plan/09 N3.3).
//
// The reference decode path launches conv_silu (8192 channels) and then the QK
// L2 norm (16 warp rows), reading the Q/K half of the conv output back from
// global memory. The fused realization does both in one launch and never
// materializes that intermediate. Both run on the same buffers here, their
// outputs are compared bit for bit, and each is timed over a batch of launches
// (a decode step issues one per GDN layer) so the per-launch cost is what gets
// reported. 24 layers/token x (launch count delta) is the quantity the fusion
// changes, so the launch count is printed next to the time.
// ---------------------------------------------------------------------------
int run_gdn_decode(Device &device, Stream &stream, const Options &options) {
    constexpr uint32_t qkh = 16u, hd = qwen35_gdn_model_head_dim, vh = 32u;
    constexpr uint32_t qkn = qkh * hd, vn = vh * hd, channels = 2u * qkn + vn, K = 4u, rw = K - 1u;
    const uint32_t iters = options.gdn_decode_iters != 0u ? options.gdn_decode_iters : 20u;
    const float eps = 1e-6f;
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(hd));
    luisa::vector<float> src(channels), weight(size_t(K) * channels), ring0(size_t(channels) * rw);
    luisa::vector<float> qkv_src(channels);
    for (uint32_t i = 0; i < channels; ++i) {
        src[i] = std::sin(0.017f * static_cast<float>(i)) * 0.7f;
        qkv_src[i] = src[i];
    }
    for (size_t i = 0; i < weight.size(); ++i) { weight[i] = 0.5f * std::cos(0.0071f * static_cast<float>(i)); }
    for (size_t i = 0; i < ring0.size(); ++i) { ring0[i] = 0.25f * std::sin(0.0031f * static_cast<float>(i)); }
    auto bsrc = device.create_buffer<float>(channels);
    auto bw = device.create_buffer<float>(weight.size());
    auto bq_ref = device.create_buffer<float>(qkn);
    auto bk_ref = device.create_buffer<float>(qkn);
    auto bconv_ref = device.create_buffer<float>(channels);
    auto bring_ref = device.create_buffer<float>(ring0.size());
    auto bq_f = device.create_buffer<float>(qkn);
    auto bk_f = device.create_buffer<float>(qkn);
    auto bconv_f = device.create_buffer<float>(channels);
    auto bring_f = device.create_buffer<float>(ring0.size());
    stream << bsrc.copy_from(luisa::span<const float>{qkv_src})
           << bw.copy_from(luisa::span<const float>{weight}) << synchronize();
    auto sh_conv = device.compile(make_qwen_gdn_conv_silu_kernel());
    auto sh_norm = device.compile(make_qwen_gdn_qk_l2norm_kernel());
    auto sh_fused = device.compile(make_qwen35_gdn_decode_fused_kernel());
    auto plan_conv = plan_qwen_gdn_conv_dispatch(channels);
    auto plan_norm = plan_qwen_gdn_qk_norm_dispatch(1u, qkh);
    auto plan_fused = plan_qwen35_gdn_decode_fused_dispatch(qkh, hd, vn);
    if (!plan_conv.ok() || !plan_norm.ok() || !plan_fused.ok()) {
        std::fprintf(stderr, "gdn decode: dispatch plan failed\n");
        return 2;
    }
    std::printf("GDN_DECODE columns: impl,token,launches,min_ms,median_ms\n");
    bool all_equal = true;
    const uint32_t positions = std::max(1u, options.gdn_decode_positions);
    for (uint32_t pos = 0; pos < positions; ++pos) {
        // The ring must be re-seeded per position: the arm is about the launch,
        // not about a long history, and both realizations get the same start.
        stream << bring_ref.copy_from(luisa::span<const float>{ring0})
               << bring_f.copy_from(luisa::span<const float>{ring0}) << synchronize();
        const auto record_ref = [&] {
            auto list = CommandList::create(2u);
            list << sh_conv(bsrc, bw, bring_ref, bconv_ref, 1u, channels, K, pos)
                        .dispatch(qwen_dispatch_threads(*plan_conv.value))
                 << sh_norm(bconv_ref, bq_ref, bk_ref, 1u, qkh, hd, vn, eps, q_scale, qkh)
                        .dispatch(qwen_dispatch_threads(*plan_norm.value));
            return list;
        };
        const auto record_fused = [&] {
            auto list = CommandList::create(1u);
            list << sh_fused(bsrc, bw, bring_f, bq_f, bk_f, bconv_f, qkh, hd, vn, K, eps, q_scale, pos)
                        .dispatch(qwen_dispatch_threads(*plan_fused.value));
            return list;
        };
        if (pos == 0u) {
            stream << record_ref().commit() << record_fused().commit() << synchronize();
            luisa::vector<float> qr(qkn), qf(qkn), kr(qkn), kf(qkn);
            stream << bq_ref.copy_to(luisa::span{qr}) << bq_f.copy_to(luisa::span{qf})
                   << bk_ref.copy_to(luisa::span{kr}) << bk_f.copy_to(luisa::span{kf}) << synchronize();
            // Same contract as the chunk arm: bit equality on CUDA, the FP32
            // last-bit envelope (1e-6 absolute) elsewhere.
            double max_q = 0.0, max_k = 0.0;
            for (uint32_t i = 0; i < qkn; ++i) {
                max_q = std::max(max_q, std::fabs(static_cast<double>(qr[i]) - qf[i]));
                max_k = std::max(max_k, std::fabs(static_cast<double>(kr[i]) - kf[i]));
            }
            const bool equal = max_q <= 1.0e-6 && max_k <= 1.0e-6;
            all_equal = equal;
            std::printf("GDN_DECODE validate: q_bit_exact=%s k_bit_exact=%s max_q_abs=%.3e max_k_abs=%.3e\n",
                        qr == qf ? "yes" : "no", kr == kf ? "yes" : "no", max_q, max_k);
        }
        // Each sample = one launch chain; the ring buffer is advanced by the run
        // itself, exactly like a decode step.
        for (auto fused : {false, true}) {
            auto warm = fused ? record_fused() : record_ref();
            stream << warm.commit() << synchronize();
            luisa::vector<double> samples;
            samples.reserve(iters);
            for (uint32_t it = 0; it < iters; ++it) {
                Clock clock;
                clock.tic();
                auto list = fused ? record_fused() : record_ref();
                stream << list.commit() << synchronize();
                samples.emplace_back(clock.toc());
            }
            const auto min_ms = *std::min_element(samples.begin(), samples.end());
            const auto med_ms = median(samples);
            std::printf("RESULT,GDN_DECODE,%s,%u,%u,%.9f,%.9f\n",
                        fused ? "fused" : "pair", pos, fused ? 1u : 2u, min_ms, med_ms);
            std::fflush(stdout);
        }
    }
    std::printf("GDN_DECODE summary: fp32_envelope=%s\n", all_equal ? "ok" : "FAILED");
    std::fflush(stdout);
    return all_equal ? 0 : 3;
}

  // ---------------------------------------------------------------------------
  // --gdn-recurrent-norm: isolated T=1 GDN recurrent + gated output RMSNorm arm
  // (opt_plan/03 section 5, phase 9e).
  //
  // ONE decode step of ONE GDN layer, three ways:
  //   pair_one_cta : one-CTA-per-head recurrence + the gated norm kernel (2 launches)
  //   fused        : the fused one-CTA recurrence + norm kernel         (1 launch)
  //   pair_decode4 : 4-CTA-per-head recurrence + the gated norm kernel  (2 launches)
  // The fused kernel exists because the DEFAULT decode schedule cannot host the norm:
  // its four blocks per V head split the key axis, so a head's 128-channel reduction
  // needs the row in global memory plus a second launch. The one-CTA recurrence owns
  // the whole head in one block, so both fit in one launch. This arm therefore prints
  // the launch count and the global bytes of every chain next to its time, so the
  // trade is visible instead of argued:
  //   fused vs pair_one_cta: -1 launch and -2 rows (32 KiB) of global traffic per
  //     GDN layer, at an identical grid,
  //   fused vs pair_decode4: -1 launch and -32 KiB as well, but 4x fewer blocks on
  //     the recurrence -- that grid-parallelism loss is what the default pays for.
  // The state is carried across samples exactly like a decode loop, and every
  // realization starts from the same seed and is validated against the others first.
  // ---------------------------------------------------------------------------
  int run_gdn_recurrent_norm(Device &device, Stream &stream, const Options &options) {
      constexpr uint32_t qkh = 16u, hd = qwen35_gdn_model_head_dim, vh = 32u;
      constexpr uint32_t qkn = qkh * hd, vn = vh * hd, channels = 2u * qkn + vn;
      constexpr uint64_t state_bytes = uint64_t(vh) * hd * hd * sizeof(float);
      constexpr uint64_t qk_bytes = 2ull * qkn * sizeof(float);
      constexpr uint64_t conv_bytes = uint64_t(vn) * sizeof(float);
      constexpr uint64_t gate_bytes = 2ull * vh * sizeof(float);
      constexpr uint64_t row_bytes = uint64_t(vn) * sizeof(float);
      constexpr uint64_t norm_w_bytes = uint64_t(hd) * sizeof(float);
      // Recurrence: state read + write, q + k, the V slice of the conv output, beta/g
      // and the head row it writes. Both recurrent realizations move exactly this.
      const uint64_t rec_bytes = 2ull * state_bytes + qk_bytes + conv_bytes + gate_bytes + row_bytes;
      // Gated norm: the head row it reads and writes, z, and the direct norm weight.
      const uint64_t norm_bytes = 2ull * row_bytes + row_bytes + norm_w_bytes;
      const uint64_t pair_bytes = rec_bytes + norm_bytes;
      // Fused: the row is written once and never read back; z and the weight are read.
      const uint64_t fused_bytes = rec_bytes + row_bytes + norm_w_bytes;
      const uint32_t iters = options.gdn_recurrent_norm_iters != 0u
                                 ? options.gdn_recurrent_norm_iters
                                 : 50u;
      const uint32_t steps = std::max(1u, options.gdn_recurrent_norm_steps);
      const float eps = 1e-6f;
      luisa::vector<float> q(qkn), k(qkn), conv(channels), beta(vh), g(vh), z(vn), nw(hd);
      luisa::vector<float> state(size_t(state_bytes / sizeof(float)));
      for (uint32_t i = 0; i < qkn; ++i) {
          q[i] = std::sin(0.013f * static_cast<float>(i)) * 0.8f;
          k[i] = std::cos(0.011f * static_cast<float>(i)) * 0.8f;
      }
      for (uint32_t i = 0; i < channels; ++i) {
          conv[i] = std::sin(0.017f * static_cast<float>(i)) * 0.7f;
      }
      for (uint32_t i = 0; i < vh; ++i) {
          beta[i] = 0.1f + 0.4f * std::fabs(std::sin(0.5f * static_cast<float>(i)));
          g[i] = -0.02f - 0.01f * std::fabs(std::cos(0.7f * static_cast<float>(i)));
      }
      for (uint32_t i = 0; i < vn; ++i) {
          z[i] = 2.5f * std::sin(0.005f * static_cast<float>(i));
      }
      for (uint32_t i = 0; i < hd; ++i) {
          nw[i] = 0.5f + 0.5f * std::fabs(std::cos(0.3f * static_cast<float>(i)));
      }
      for (uint32_t i = 0; i < state.size(); ++i) {
          state[i] = 0.05f * std::sin(0.001f * static_cast<float>(i));
      }
      auto bq = device.create_buffer<float>(qkn);
      auto bk = device.create_buffer<float>(qkn);
      auto bc = device.create_buffer<float>(channels);
      auto bb = device.create_buffer<float>(vh);
      auto bg = device.create_buffer<float>(vh);
      auto bz = device.create_buffer<float>(vn);
      auto bnw = device.create_buffer<float>(hd);
      stream << bq.copy_from(luisa::span<const float>{q}) << bk.copy_from(luisa::span<const float>{k})
             << bc.copy_from(luisa::span<const float>{conv}) << bb.copy_from(luisa::span<const float>{beta})
             << bg.copy_from(luisa::span<const float>{g}) << bz.copy_from(luisa::span<const float>{z})
             << bnw.copy_from(luisa::span<const float>{nw}) << synchronize();
      auto sh_rec = device.compile(make_qwen35_gdn_recurrent_kernel(hd));
      auto sh_norm = device.compile(make_qwen_gdn_gated_norm_kernel());
      auto sh_fused = device.compile(make_qwen35_gdn_recurrent_norm_kernel());
      auto sh_d4 = device.compile(make_qwen35_gdn_decode4_kernel());
      auto plan_rec = plan_qwen_gdn_recurrent_dispatch(vh);
      auto plan_norm = plan_qwen_gdn_gated_norm_dispatch(1u, vh);
      auto plan_fused = plan_qwen35_gdn_recurrent_norm_dispatch(vh);
      auto plan_d4 = plan_qwen35_gdn_decode4_dispatch(vh);
      if (!plan_rec.ok() || !plan_norm.ok() || !plan_fused.ok() || !plan_d4.ok()) {
          std::fprintf(stderr, "gdn recurrent+norm: dispatch plan failed\n");
          return 2;
      }
      // One state buffer per realization, all seeded identically.
      auto mk_state = [&] {
          auto b = device.create_buffer<float>(state.size());
          stream << b.copy_from(luisa::span<const float>{state}) << synchronize();
          return b;
      };
      auto bs_rec = mk_state();
      auto bs_fused = mk_state();
      auto bs_d4 = mk_state();
      auto bout_rec = device.create_buffer<float>(vn);
      auto bout_fused = device.create_buffer<float>(vn);
      auto bout_d4 = device.create_buffer<float>(vn);
      auto braw = device.create_buffer<float>(vn);
      const auto v_offset = 2u * qkn;
      const auto pull = [&](compute::Buffer<float> &b, size_t n) {
          luisa::vector<float> out(n);
          stream << b.view(0u, n).copy_to(luisa::span{out}) << synchronize();
          return out;
      };

      // ---- validation: one step from the same seed, three ways ---------------
      stream << bs_rec.copy_from(luisa::span<const float>{state})
             << bs_fused.copy_from(luisa::span<const float>{state})
             << bs_d4.copy_from(luisa::span<const float>{state}) << synchronize();
      stream << sh_rec(bq, bk, bc, bb, bg, bs_rec, braw, 1u, vh, hd, qkh, v_offset)
                    .dispatch(qwen_dispatch_threads(*plan_rec.value))
             << sh_norm(braw, bz, bnw, bout_rec, 1u, vh, hd, eps)
                    .dispatch(qwen_dispatch_threads(*plan_norm.value))
             << sh_fused(bq, bk, bc, bb, bg, bs_fused, bz, bnw, bout_fused, 1u, vh, hd, qkh, v_offset, eps)
                    .dispatch(qwen_dispatch_threads(*plan_fused.value))
             << sh_d4(bq, bk, bc, bb, bg, bs_d4, braw, 1u, vh, hd, qkh, v_offset)
                    .dispatch(qwen_dispatch_threads(*plan_d4.value))
             << sh_norm(braw, bz, bnw, bout_d4, 1u, vh, hd, eps)
                    .dispatch(qwen_dispatch_threads(*plan_norm.value));
      auto v_rec = pull(bout_rec, vn);
      auto v_fused = pull(bout_fused, vn);
      auto v_d4 = pull(bout_d4, vn);
      auto s_rec = pull(bs_rec, state.size());
      auto s_fused = pull(bs_fused, state.size());
      auto s_d4 = pull(bs_d4, state.size());
      double max_out_fused = 0.0, max_st_fused = 0.0, max_out_d4 = 0.0;
      bool finite = true;
      for (uint32_t i = 0; i < vn; ++i) {
          max_out_fused = std::max(max_out_fused, std::fabs(double(v_fused[i]) - double(v_rec[i])));
          max_out_d4 = std::max(max_out_d4, std::fabs(double(v_fused[i]) - double(v_d4[i])));
          finite = finite && std::isfinite(v_fused[i]);
      }
      for (size_t i = 0; i < state.size(); ++i) {
          max_st_fused = std::max(max_st_fused, std::fabs(double(s_fused[i]) - double(s_rec[i])));
          finite = finite && std::isfinite(s_fused[i]);
      }
      std::printf("GDN_RECURRENT_NORM validate: max|fused-pair_one_cta|=%.3e "
                  "max|fused-pair_decode4|=%.3e max_state|fused-pair_one_cta|=%.3e finite=%s\n",
                  max_out_fused, max_out_d4, max_st_fused, finite ? "yes" : "no");
      std::fflush(stdout);

      std::printf("GDN_RECURRENT_NORM columns: impl,launches,grid_blocks,global_bytes,min_ms,median_ms,per_token_24_layers_ms\n");
      struct Arm {
          const char *name;
          uint32_t launches;
          uint32_t blocks;
          uint64_t bytes;
          bool decode4;
          bool norm;
          compute::Buffer<float> *state;
          compute::Buffer<float> *out;
      };
      Arm arms[] = {
          // `recurrent_only` and `gated_norm_only` decompose the pair: (pair_one_cta -
          // recurrent_only) is what the second launch costs, (fused - recurrent_only) is
          // what the in-block norm costs on top of the same recurrence.
          {"recurrent_only", 1u, vh, rec_bytes, false, false, &bs_rec, &bout_rec},
          {"gated_norm_only", 1u, vh, norm_bytes, false, true, &bs_rec, &bout_rec},
          {"pair_one_cta", 2u, vh, pair_bytes, false, true, &bs_rec, &bout_rec},
          {"fused", 1u, vh, fused_bytes, false, true, &bs_fused, &bout_fused},
          {"pair_decode4", 2u, vh * 4u, pair_bytes, true, true, &bs_d4, &bout_d4},
      };
      bool ok = max_out_fused <= 4e-6 && max_out_d4 <= 2e-3 && finite;
      // One REPORTED SAMPLE is a whole submit of `reps` chains, divided by `reps`:
      // the DX/VK submit+synchronize cost of a single isolated launch is ~50 us and
      // would otherwise hide the kernel entirely (measured: one DX submit costs more
      // than either launch). `reps` = 24 = one decode token worth of GDN layers, so
      // the per-token column is what the schedule really adds to a token.
      const uint32_t reps = steps;
      for (auto &arm : arms) {
          const auto record = [&] {
              auto list = CommandList::create(size_t{reps} * arm.launches);
              for (uint32_t rep = 0u; rep < reps; ++rep) {
                  if (arm.name == std::string{"recurrent_only"}) {
                      list << sh_rec(bq, bk, bc, bb, bg, *arm.state, braw, 1u, vh, hd, qkh, v_offset)
                                  .dispatch(qwen_dispatch_threads(*plan_rec.value));
                      continue;
                  }
                  if (arm.name == std::string{"gated_norm_only"}) {
                      list << sh_norm(braw, bz, bnw, *arm.out, 1u, vh, hd, eps)
                                  .dispatch(qwen_dispatch_threads(*plan_norm.value));
                      continue;
                  }
                  if (arm.launches == 1u) {
                      list << sh_fused(bq, bk, bc, bb, bg, *arm.state, bz, bnw, *arm.out,
                                       1u, vh, hd, qkh, v_offset, eps)
                                  .dispatch(qwen_dispatch_threads(*plan_fused.value));
                      continue;
                  }
                  if (arm.decode4) {
                      list << sh_d4(bq, bk, bc, bb, bg, *arm.state, braw, 1u, vh, hd, qkh, v_offset)
                                  .dispatch(qwen_dispatch_threads(*plan_d4.value));
                  } else {
                      list << sh_rec(bq, bk, bc, bb, bg, *arm.state, braw, 1u, vh, hd, qkh, v_offset)
                                  .dispatch(qwen_dispatch_threads(*plan_rec.value));
                  }
                  list << sh_norm(braw, bz, bnw, *arm.out, 1u, vh, hd, eps)
                              .dispatch(qwen_dispatch_threads(*plan_norm.value));
              }
              return list;
          };
          for (uint32_t w = 0; w < 3u; ++w) { stream << record().commit() << synchronize(); }
          luisa::vector<double> samples;
          samples.reserve(iters);
          for (uint32_t it = 0; it < iters; ++it) {
              Clock clock;
              clock.tic();
              stream << record().commit() << synchronize();
              samples.emplace_back(clock.toc() / static_cast<double>(reps));
          }
          const auto min_ms = *std::min_element(samples.begin(), samples.end());
          const auto med_ms = median(samples);
          std::printf("RESULT,GDN_RECURRENT_NORM,%s,%u,%u,%llu,%.9f,%.9f,%.9f\n",
                      arm.name, arm.launches, arm.blocks,
                      static_cast<unsigned long long>(arm.bytes), min_ms, med_ms, min_ms * 24.0);
          std::fflush(stdout);
      }
      std::printf("GDN_RECURRENT_NORM summary: fp32_envelope=%s reps_per_sample=%u\n",
                  ok ? "ok" : "FAILED", reps);
      std::fflush(stdout);
      return ok ? 0 : 3;
  }

// ---------------------------------------------------------------------------
// --gdn-gates: isolated T=1 alpha/beta gate GEMV arm (opt_plan/09 N2).
// Model geometry: 32 V heads x 2 gate rows = 64 warps over hidden = 2560, i.e. one
// 128-thread block per 4 rows and only 16 blocks in total -- the shape the plan
// flags as the worst FP8 decode case (58-89 % of the roof). Runs the 32-bit LUT
// realization and the 128-bit branchless one on the SAME packed buffers,
// validates beta and g against each other before printing, and reports
// min/median per launch plus the launch count.
// ---------------------------------------------------------------------------
int run_gdn_gates(Device &device, Stream &stream, const Options &options) {
    constexpr uint32_t heads = 32u;
    const uint32_t k = 2560u;
    const uint32_t iters = options.gdn_gates_iters != 0u ? options.gdn_gates_iters : 50u;
    std::printf("GDN_GATES note: heads=%u hidden=%u iters=%u\n", heads, k, iters);
    std::printf("GDN_GATES columns: format,impl,heads,K,min_ms,median_ms,effective_GBps\n");
    bool all_validated = true;
    for (auto format : enabled_formats(options)) {
        const uint32_t pack = format_pack(format);
        const uint32_t groups = format_scaled(format) ? k / format_group(format) : 0u;
        const size_t words = size_t(heads) * (k / pack);
        const size_t scales_n = format_scaled(format) ? size_t(heads) * groups : 1u;
        luisa::vector<uint32_t> aw(words), bw(words);
        for (size_t i = 0; i < words; ++i) {
            aw[i] = packed_word(format, i);
            bw[i] = packed_word(format, i + 7919u);
        }
        auto baw = device.create_buffer<uint32_t>(words);
        auto bbw = device.create_buffer<uint32_t>(words);
        stream << baw.copy_from(luisa::span<const uint32_t>{aw})
               << bbw.copy_from(luisa::span<const uint32_t>{bw}) << synchronize();
        aw.clear(); bw.clear(); aw.shrink_to_fit(); bw.shrink_to_fit();
        luisa::vector<float> s_host(scales_n, format_scaled(format) ? 0.03125f : 0.0f);
        auto bas = device.create_buffer<float>(scales_n);
        auto bbs = device.create_buffer<float>(scales_n);
        stream << bas.copy_from(luisa::span<const float>{s_host})
               << bbs.copy_from(luisa::span<const float>{s_host}) << synchronize();
        luisa::vector<float> x_host(k);
        for (uint32_t i = 0; i < k; ++i) { x_host[i] = 0.01f * std::sin(0.013f * static_cast<float>(i)); }
        auto bx = device.create_buffer<float>(k);
        stream << bx.copy_from(luisa::span<const float>{x_host}) << synchronize();
        x_host.clear(); x_host.shrink_to_fit();
        luisa::vector<float> ssm_a(heads), dt(heads);
        for (uint32_t h = 0; h < heads; ++h) {
            ssm_a[h] = -0.05f - 0.001f * static_cast<float>(h);
            dt[h] = 0.001f * static_cast<float>(h);
        }
        auto bssm = device.create_buffer<float>(heads);
        auto bdt = device.create_buffer<float>(heads);
        stream << bssm.copy_from(luisa::span<const float>{ssm_a})
               << bdt.copy_from(luisa::span<const float>{dt}) << synchronize();
        auto beta_a = device.create_buffer<float>(heads);
        auto g_a = device.create_buffer<float>(heads);
        auto beta_b = device.create_buffer<float>(heads);
        auto g_b = device.create_buffer<float>(heads);
        auto legacy = device.compile(make_qwen_gdn_decode_gates_kernel(format));
        auto legacy_plan = plan_qwen_gdn_decode_gates_dispatch(heads, k, format, format_group(format));
        auto packed = device.compile(make_qwen_gdn_decode_gates_packed_kernel(format));
        auto packed_plan = plan_qwen_gdn_decode_gates_packed_dispatch(heads, k, format, format_group(format));
        if (!legacy_plan.ok() || !packed_plan.ok()) {
            std::fprintf(stderr, "GDN_GATES planning failed (format %s)\n", format_label(format));
            return 2;
        }
        const auto vecs = static_cast<uint32_t>(k / qwen_packed_vec_elements(format));
        const auto record = [&](bool use_packed) {
            auto list = CommandList::create(static_cast<size_t>(iters));
            for (uint32_t it = 0; it < iters; ++it) {
                if (use_packed) {
                    list << packed(bx.view().as<float4>(), baw.view().as<uint4>(), bas,
                                   bbw.view().as<uint4>(), bbs, bssm, bdt, beta_b, g_b,
                                   k, heads, format_group(format), 20.0f, vecs)
                                .dispatch(qwen_dispatch_threads(*packed_plan.value));
                } else {
                    list << legacy(bx, baw, bas, bbw, bbs, bssm, bdt, beta_a, g_a,
                                   k, heads, format_group(format), 20.0f)
                                .dispatch(qwen_dispatch_threads(*legacy_plan.value));
                }
            }
            return list;
        };
        stream << record(false).commit() << record(true).commit() << synchronize();
        auto host_beta_a = luisa::vector<float>(heads);
        auto host_g_a = luisa::vector<float>(heads);
        auto host_beta_b = luisa::vector<float>(heads);
        auto host_g_b = luisa::vector<float>(heads);
        stream << beta_a.view().copy_to(luisa::span<float>{host_beta_a})
               << g_a.view().copy_to(luisa::span<float>{host_g_a})
               << beta_b.view().copy_to(luisa::span<float>{host_beta_b})
               << g_b.view().copy_to(luisa::span<float>{host_g_b}) << synchronize();
        double max_beta = 0.0, max_g = 0.0, scale_beta = 0.0, scale_g = 0.0;
        for (uint32_t h = 0; h < heads; ++h) {
            max_beta = std::max(max_beta, std::fabs(double(host_beta_a[h]) - host_beta_b[h]));
            max_g = std::max(max_g, std::fabs(double(host_g_a[h]) - host_g_b[h]));
            scale_beta = std::max(scale_beta, std::fabs(double(host_beta_a[h])));
            scale_g = std::max(scale_g, std::fabs(double(host_g_a[h])));
        }
        const bool validated = max_beta <= 1.0e-5 * (1.0 + scale_beta) &&
                               max_g <= 1.0e-5 * (1.0 + scale_g);
        all_validated = all_validated && validated;
        std::printf("GDN_GATES validate: format=%s max_beta_abs=%.3e max_g_abs=%.3e %s\n",
                    format_label(format), max_beta, max_g, validated ? "ok" : "FAILED");
        const double weight_bytes = static_cast<double>(words) * sizeof(uint32_t);
        const double scale_bytes = format_scaled(format) ? static_cast<double>(scales_n) * 4.0 : 0.0;
        const double bytes = 2.0 * (weight_bytes + scale_bytes) +
                             static_cast<double>(k + 4u * heads) * 4.0;
        for (auto use_packed : {false, true}) {
            auto warm = record(use_packed);
            stream << warm.commit() << synchronize();
            luisa::vector<double> samples;
            samples.reserve(iters);
            for (uint32_t it = 0; it < iters; ++it) {
                auto list = record(use_packed);
                Clock clock;
                clock.tic();
                stream << list.commit() << synchronize();
                samples.emplace_back(clock.toc() / double(iters));
            }
            const auto min_ms = *std::min_element(samples.begin(), samples.end());
            const auto med_ms = median(samples);
            std::printf("RESULT,gdn_gates,%s,%s,%u,%u,%.9f,%.9f,%.3f\n",
                        format_label(format), use_packed ? "packed" : "legacy",
                        heads, k, min_ms, med_ms, bytes / med_ms * 1.0e-6);
            std::fflush(stdout);
        }
        host_beta_a.clear(); host_g_a.clear(); host_beta_b.clear(); host_g_b.clear();
    }
    std::printf("GDN_GATES summary: validated=%s\n", all_validated ? "ok" : "FAILED");
    return all_validated ? 0 : 3;
}

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Options options;
    if (argc < 2 || !parse_options(argc, argv, options)) {
        std::fprintf(stderr,
                     "usage: %s <backend> [--format both|bf16|fp8|q8|all] [--impl both|generic|decode] "
                     "[--iters N] [--lm-head] [--ffn] [--attention] [--shape NAME]\n"
                       " %s <backend> --prefill-attention [--prefill-attn-lengths 128,512,1024[,4096]] "
                       "[--prefill-attn-q-tile N] [--prefill-attn-k-tile N] [--prefill-attn-iters N]\n"
                     "       %s <backend> --calibrate [--calibrate-iters N] "
                 "[--calibrate-copy-mib N] [--calibrate-triad-mib N]\n"
                 " %s <backend> --prefill-gemm [--prefill-gemm-m 64,256,1024] "
                 "  --impl both|generic|decode|packed|all : which decode realization to time\n"
                 "  --gdn-gates [--gdn-gates-iters N] : isolated T=1 alpha/beta gate GEMV arm\n"
                 "[--prefill-gemm-iters N] [--format both|bf16|fp8|q8|all]\n"
                 " %s <backend> --cuda-tile-fp8 [--cuda-tile-m N] [--cuda-tile-n N]\n"
                 "  --cuda-tile-fp8 : OPT-IN extra impl row of --prefill-gemm: the CUDA native\n"
                 "     Tile IR FP8 GEMM (needs backend cuda, fp8 weights, K%%64==0 and\n"
                 "     LUISA_CUDA_TILE_IR=1; unservable rows print an explicit skipped row)\n"
                 "  --cuda-tile-m/--cuda-tile-n N : tile_m/tile_n override (default 64x64)\n",
                 argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    if (!device) {
        std::fprintf(stderr, "failed to create backend '%s'\n", argv[1]);
        return 2;
    }
    auto stream = device.create_stream();
    if (options.calibrate) {
        // --calibrate is its own mode: no shape, no format, no model.
        return run_calibration(device, stream, options);
    }
    std::printf("RESULT columns: format,shape,impl,N,K,replicas,min_ms,median_ms,min_effective_GBps,median_effective_GBps\n");
    if (options.prefill_attention) {
        // Isolated prefill attention arm: own mode, no shape, no model.
        return run_prefill_attention(device, stream, options);
    }
    if (options.prefill_gemm) {
        // Isolated prefill GEMM arm (opt_plan/02 section 3): own mode.
        return run_prefill_gemm(device, stream, options);
    }
    if (options.gdn_gates) {
        // Isolated T=1 alpha/beta gate GEMV arm (opt_plan/09 N2).
        return run_gdn_gates(device, stream, options);
    }
    if (options.gdn_decode) {
        // Isolated decode GDN conv+SiLU+QK-norm arm (opt_plan/09 N3.3).
        return run_gdn_decode(device, stream, options);
    }
    if (options.gdn_recurrent_norm) {
        // Isolated GDN recurrent + gated RMSNorm arm (opt_plan/03 section 5).
        return run_gdn_recurrent_norm(device, stream, options);
    }
    if (options.rmsnorm) {
        // Isolated RMSNorm arm (opt_plan/09 N3.1): own mode, no shape.
        return run_rmsnorm(device, stream, options);
    }
    if (options.gdn_chunk) {
        // Isolated WY/UT chunk arm (opt_plan/09 V4): own mode.
        return run_gdn_chunk(device, stream, options);
    }
    if (options.decode_attention) {
        // Isolated decode attention arm (opt_plan/03 section 4.2): own mode.
        return run_decode_attention(device, stream, options);
    }
    if (options.attention) {
        run_attention(device, stream, options.iters);
        return 0;
    }
    for (auto format : enabled_formats(options)) {
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
