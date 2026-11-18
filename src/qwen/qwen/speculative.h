#pragma once
// Qwen3.5-4B -- PURE-HOST speculative decoding acceptance engine for NextN
// (MTP) draft+verify rounds. No device, no DSL, no engine state: the module
// consumes rows of already-downloaded FP32 logits plus the drafted token ids
// and decides how many drafts are accepted and which bonus token is emitted.
//
// Contract and indexing (0-based throughout):
//
//   k = draft_tokens.size() >= 1 are the drafted ids d_1..d_k.
//   draft_logits.row(j)  is the draft distribution q_j(.) that d_{j+1} was
//                        DRAWN from (sampling mode only; greedy ignores it).
//   target_logits.row(j) is p_j(.) = target distribution after consuming
//                        tokens through d_j (d_0 = last verified token), for
//                        j = 0..k; row k is the EXTRA row feeding the
//                        all-accept bonus. Target needs exactly >= k+1 rows.
//
//   Greedy (temperature == 0): accept d_{j+1} iff it equals
//   qwen_argmax(target_logits.row(j)) (lowest-index ties, NaN rejection --
//   identical to plain decoding). First mismatch at j emits bonus =
//   that argmax; accepting all k emits bonus = qwen_argmax(row k).
//   Draft logits are NOT examined (rows may be absent/garbage).
//
//   Sampling (temperature > 0): with p = softmax(target_row/T) and
//   q = softmax(draft_row/T) -- the SAME temperature on both sides -- draw
//   r_j and accept d_{j+1} iff r_j * q(d) < p(d) (division-free
//   min(1, p(d)/q(d)) accept test; q(d) == 0 accepts iff p(d) > 0).
//   First rejection at j emits a bonus drawn from the normalized residual
//   max(p - q, 0)/sum; accepting all k emits a bonus drawn from p_k = the
//   softmax of the extra target row k. A bonus token is ALWAYS emitted:
//   emitted == accepted + 1.
//
// Reproducible counter-based randomness (no mutable RNG state):
//
//   every uniform is qwen_speculative_uniform(seed, round, index, tag), the
//   same SplitMix64 flavor as qwen_sample (engine.cpp) over a composed
//   (round, index, tag) draw identity. Draw namespaces per round:
//     * acceptance draw for draft_tokens[j] ....... index = j,       tag = Accept
//     * residual bonus after rejecting token j .... index = j,       tag = Bonus
//     * extra-row bonus after accepting all k ..... index = k,       tag = Bonus
//   `round` is the caller's speculation-round counter; it must be unique per
//   round within a session if rounds are to be statistically independent.
//
// Non-finite logit policy (fail closed, recoverable Status):
//   * NaN anywhere in an EXAMINED row -> invalid_argument (greedy inherits
//     qwen_argmax's NaN rejection; sampling checks explicitly).
//   * +inf entries in SAMPLING rows share the softmax mass equally among
//     them (the exact limiting distribution, same rule as qwen_sample);
//     greedy leaves ordering to qwen_argmax ("infinities order normally").
//   * a row of all -inf has zero mass -> invalid_argument. A residual with
//     zero total mass (p == q) is mathematically unreachable on rejection
//     but is still rejected fail-closed.
//
// The engine is EOS-agnostic: token ids are checked only against `vocab`;
// no id is ever treated as a stop. Callers own stopping (EOS / budget).
//
// `QwenSpeculativeLogits` is a non-owning STRIDED matrix view: rows are read
// in place, never copied (the multi-MB verify logits are downloaded once by
// the caller); the only host scratch is O(vocab) per bonus draw.

#include <cstddef>
#include <cstdint>

#include <luisa/core/stl/vector.h>

#include "qwen/common.h"// Status/Result (host-only)
#include "qwen/engine.h"// QwenSampleOptions, qwen_argmax

namespace luisa::qwen {

// Non-owning view of a [rows, vocab] FP32 logits matrix with an arbitrary row
// stride, so device-side [k(+1), vocab] buffers can be consumed in place via
// a downloaded pointer without per-row copies. `stride == 0` means tightly
// packed (effective stride = vocab). Padding between rows is never read.
struct QwenSpeculativeLogits {
    const float *base{nullptr};
    uint64_t rows{0};
    uint64_t vocab{0}; // valid elements per row
    uint64_t stride{0};// element distance between consecutive rows

    [[nodiscard]] uint64_t effective_stride() const noexcept {
        return stride == 0u ? vocab : stride;
    }
    // Row as a vocab-sized span. Callers must have validated the view first
    // (qwen_speculative_accept / validate do); row(r) itself does not re-check.
    [[nodiscard]] luisa::span<const float> row(uint64_t r) const noexcept {
        return luisa::span<const float>{base + r * effective_stride(),
                                        static_cast<size_t>(vocab)};
    }
    // Structural validation only (base/rows/vocab/stride relationships);
    // semantic requirements (row counts) are checked by the accept engine.
    [[nodiscard]] Status validate() const noexcept;
};

// Draw namespaces for qwen_speculative_uniform (see file contract above).
inline constexpr uint32_t kQwenSpecTagAccept = 0u;
inline constexpr uint32_t kQwenSpecTagBonus = 1u;

struct QwenSpeculativeInput {
    // d_1..d_k, k >= 1; every id must be < vocab (checked fail-closed).
    luisa::span<const uint32_t> draft_tokens;
    // Row j: distribution d_{j+1} was drawn from. Only examined in sampling
    // mode (temperature > 0); then rows >= k and vocab == target vocab.
    QwenSpeculativeLogits draft_logits;
    // Row j: p(. | consumed through d_j); must have >= k+1 rows (row k is
    // the all-accept bonus distribution).
    QwenSpeculativeLogits target_logits;
    // seed/temperature shared with qwen_sample (engine.h). temperature == 0
    // selects the greedy path; positive finite values select sampling.
    QwenSampleOptions options;
    // Caller-owned speculation-round counter (draw identity component).
    uint64_t round{0};
};

struct QwenSpeculativeResult {
    uint32_t accepted{0};// accepted draft tokens, in [0, k]
    uint32_t emitted{0}; // accepted + 1 -- a bonus token is ALWAYS emitted
    uint32_t bonus{0};   // the bonus token id; the emitted sequence is
                         // draft_tokens[0..accepted) followed by `bonus`
};

// ---------------------------------------------------------------------------
// Reproducible counter-based uniforms (no mutable RNG state).
// ---------------------------------------------------------------------------
// SplitMix64 finalizer. MUST bit-match the anonymous splitmix64 in engine.cpp:
// qwen_speculative_uniform reuses qwen_sample's two-level hash shape.
[[nodiscard]] uint64_t qwen_speculative_mix64(uint64_t x) noexcept;
// Uniform in the OPEN interval (0,1) with 53-bit precision, identical in
// shape to qwen_sample's uniform_open01: a pure function of
// (seed, round, index, tag); same arguments always give the same draw.
[[nodiscard]] double qwen_speculative_uniform(uint64_t seed, uint64_t round,
                                              uint64_t index, uint32_t tag) noexcept;

// ---------------------------------------------------------------------------
// Building blocks (public so the future device integration and tests share
// exactly the same math). Every function validates its inputs fail-closed.
// `temperature` must be finite and > 0 in all three.
// ---------------------------------------------------------------------------
// p = softmax(row / temperature)[index] under the limiting-distribution rules:
// +inf entries share the mass equally; -inf entries have zero mass; all -inf
// is a zero-mass failure; NaN is a failure.
[[nodiscard]] Status qwen_softmax_probability(luisa::span<const float> row,
                                              float temperature, uint32_t index,
                                              double &out) noexcept;
// Lowest index whose cumulative softmax mass strictly exceeds u (u in (0,1));
// identical selection semantics to qwen_sample given the same u.
[[nodiscard]] Status qwen_categorical_sample(luisa::span<const float> row,
                                             float temperature, double u,
                                             uint32_t &out) noexcept;
// Sample from normalize(max(p - q, 0)) where p = softmax(target/T),
// q = softmax(draft/T). Zero total residual mass is a failure.
[[nodiscard]] Status qwen_residual_sample(luisa::span<const float> target_row,
                                          luisa::span<const float> draft_row,
                                          float temperature, double u,
                                          uint32_t &out) noexcept;

// ---------------------------------------------------------------------------
// The acceptance engine. `out` is reset first; on Status failure its contents
// are unspecified and must not be used. Emitted/accepted counts and the bonus
// follow the file contract above for both modes.
// ---------------------------------------------------------------------------
[[nodiscard]] Status qwen_speculative_accept(const QwenSpeculativeInput &in,
                                             QwenSpeculativeResult &out) noexcept;

}// namespace luisa::qwen
