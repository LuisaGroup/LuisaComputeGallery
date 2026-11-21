#pragma once
// Qwen3.5-4B -- W4 IMPORTANCE (activation second moment) calibration statistics.
//
// Why this file exists (opt_plan/05 section 3 item 2): the measured 32-prompt
// quality gate of phase 8b FAILED for the clipping-only W4 pack (mean prefix
// agreement 0.3179 against the 0.90 gate, KL 0.2867). Clipping alone chooses the
// per-group scale from the WEIGHTS; it cannot know which input channels actually
// carry signal. This module is the contract for the extra tensor the packer needs
// to weight the per-group error by the measured activation energy of each input
// channel.
//
// WHAT IS MEASURED (exactly, no approximation is hidden):
//   For one projection weight W with K input channels, the calibration run drives
//   the BF16 model over a small deterministic prompt set and, at every projection
//   site, accumulates
//       S[j] = sum over every activation token x of x[j]^2      (j in [0, K))
//   in FP32 on the device. The exported statistic is the SECOND MOMENT
//       m[j] = S[j] / tokens                                    (tokens > 0)
//   i.e. the diagonal of the empirical input Gram matrix diag(H) = E[x x^T],
//   assuming independent channels -- this is the "diag(H)" / activation-scale
//   statistic AWQ and GPTQ both use. It is NOT a Hessian, NOT a per-channel
//   covariance and NOT AWQ's per-layer scale search with clipping search over
//   scales; only the *weighting* of the existing group clipping objective comes
//   from data. The pack is therefore described as "importance-aware W4", never
//   as AWQ.
//
// The statistic is collected on the BF16 (or F32) checkpoint of the SAME file the
// pack is derived from, so it is never synthesized from the weights. The file
// records the checkpoint's size AND its rolling payload digest (qwen/w4_pack.h),
// the deterministic prompt set and the token count: a stats file for a
// same-size/different-payload checkpoint, for a different K, or with non-finite
// data is STALE and is refused (fail closed, named tensor).
//
// File layout (both files are written by one call; the manifest goes LAST, so a
// directory without a manifest is an incomplete calibration):
//   <dir>/importance.txt       one line-oriented, order-stable text manifest
//   <dir>/importance.f32.bin   concatenated F32 second moments, tensor by tensor
//                              in manifest order, K values per tensor
#include <cstdint>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include "qwen/common.h"

namespace luisa::qwen {

// Bump when the manifest grammar or the statistic's meaning changes.
inline constexpr uint32_t kW4ImportanceVersion = 1u;

// Identity of the checkpoint the statistics were collected on, plus the
// reproducible prompt set. `hash` is the rolling source digest of qwen/w4_pack.h
// (FNV-1a 64 over the per-tensor payload hashes in inventory order).
struct W4ImportanceSource {
    luisa::string name;
    uint64_t size{0};
    uint64_t hash{0};
    // Deterministic synthetic prompt set (qwen/prompt_ids.h): `prompts` prompts
    // of `prompt_len` tokens each, followed by `decode_tokens` greedy tokens each.
    uint32_t prompts{0};
    uint32_t prompt_len{0};
    uint32_t decode_tokens{0};
    // Total activation tokens accumulated (prefill rows + decoded rows), summed
    // over the prompt set. Never 0 for a usable file.
    uint64_t tokens{0};
};

// One tensor's record. `mean`/`min`/`max` describe the exported m[j] values; the
// payload hash pins the exact F32 bytes so a truncated or edited payload is
// refused before it can bias a single group.
struct W4ImportanceTensorRecord {
    luisa::string name;
    uint64_t k{0};
    uint64_t tokens{0};
    float mean{0.0f};
    float min{0.0f};
    float max{0.0f};
    uint64_t bytes{0};
    uint64_t hash{0};
};

struct W4Importance {
    uint32_t version{0};
    W4ImportanceSource source;
    luisa::vector<W4ImportanceTensorRecord> tensors;
    // Concatenated per-tensor channel second moments, in `tensors` order.
    luisa::vector<float> values;
    // Lookup by tensor NAME + K (a K mismatch is a different matrix, not a near
    // miss). Empty span when the tensor is missing or its K disagrees.
    [[nodiscard]] luisa::span<const float> channels(luisa::string_view name, uint64_t k) const noexcept;
    // Same lookup, reporting WHY it returned nothing (missing vs K mismatch).
    [[nodiscard]] Status status_for(luisa::string_view name, uint64_t k) const noexcept;
};

// ---------------------------------------------------------------------------
// One tensor of a finished calibration run (the engine's export payload).
//
// `second_moment` holds m[j] = mean over tokens of x[j]^2 for every input channel
// of one projection weight, already divided by `tokens`. The engine hands the
// sums back and this module normalizes them, so the divisor is applied exactly
// once.
// ---------------------------------------------------------------------------
struct QwenCalibrationTensor {
    luisa::string name;
    uint64_t k{0};
    uint64_t tokens{0};
    luisa::vector<float> second_moment;
};

struct QwenCalibrationReport {
    uint64_t tensors{0};
    // Number of projection dispatches that actually fed an accumulator (a
    // zero-count projection is missing data and the writer refuses it).
    uint64_t projections{0};
    uint64_t tokens{0};
    luisa::vector<QwenCalibrationTensor> observed;
    luisa::vector<Diagnostic> diagnostics;
};

// Write the two files of a calibration directory. `source` must carry the
// checkpoint identity and the prompt set; `report` must cover every calibrated
// projection with tokens > 0 and finite, non-negative, non-zero-mean values.
// The manifest is written LAST (an incomplete directory has no manifest).
[[nodiscard]] Status write_w4_importance(const luisa::filesystem::path &dir,
                                         const W4ImportanceSource &source,
                                         const QwenCalibrationReport &report) noexcept;

// Strict reader: version, source identity, per-tensor K/bytes/hash, the exact
// payload size and the value domain (finite, >= 0, mean > 0) are all verified;
// any violation is a recoverable failure that names the tensor.
[[nodiscard]] Result<W4Importance> read_w4_importance(const luisa::filesystem::path &dir) noexcept;

// Convenience: the manifest file name inside a calibration directory.
inline constexpr const char *kW4ImportanceManifestName = "importance.txt";
inline constexpr const char *kW4ImportancePayloadName = "importance.f32.bin";

}// namespace luisa::qwen
