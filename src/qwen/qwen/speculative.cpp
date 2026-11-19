// Qwen3.5-4B -- pure-host speculative decoding acceptance engine. See
// qwen/speculative.h for the full contract (indexing, draw namespaces,
// non-finite policy). Fail closed: every malformed input is a recoverable
// Status; no exceptions, no RTTI, no mutable RNG state.
#include "qwen/speculative.h"

#include <cmath>
#include <limits>

#include <luisa/core/stl/format.h>

namespace luisa::qwen {
namespace {

[[nodiscard]] Status invalid(luisa::string msg, luisa::string detail = {}) {
    return Status::error(StatusCode::invalid_argument, std::move(msg), std::move(detail));
}

[[nodiscard]] Status check_temperature(float temperature, const char *detail) {
    if (!(temperature > 0.0f) || !std::isfinite(temperature)) {
        return invalid("temperature must be finite and strictly positive", detail);
    }
    return Status::success();
}

[[nodiscard]] Status check_unit_interval(double u, const char *detail) {
    if (!std::isfinite(u) || !(u > 0.0) || !(u < 1.0)) {
        return invalid("uniform draw must lie in the open interval (0,1)", detail);
    }
    return Status::success();
}

// Softmax statistics for one row under the limiting-distribution rules
// (speculative.h): NaN fails; +inf entries share the mass equally; -inf
// entries carry zero mass; an all -inf row is a zero-mass failure. With a
// finite max m, total = sum(exp((v - m) / temperature)) >= 1 (the max element
// contributes exp(0) = 1), so no overflow is possible.
struct SoftmaxStats {
    uint32_t pos_inf_count{0};
    double max{-std::numeric_limits<double>::infinity()};
    double total{0.0};
};

[[nodiscard]] Status softmax_stats(luisa::span<const float> row, float temperature,
                                   SoftmaxStats &s) {
    if (row.empty()) {
        return invalid("empty logits row", "row");
    }
    for (const float v : row) {
        if (std::isnan(v)) {
            return invalid("NaN logit in row", "row");
        }
    }
    for (const float v : row) {
        if (v == std::numeric_limits<float>::infinity()) { ++s.pos_inf_count; }
    }
    if (s.pos_inf_count != 0u) { return Status::success(); }
    for (const float v : row) {
        s.max = std::max(s.max, static_cast<double>(v));
    }
    if (s.max == -std::numeric_limits<double>::infinity()) {
        return invalid("categorical row has zero mass (all logits are -infinity)", "row");
    }
    const double inv_t = 1.0 / static_cast<double>(temperature);
    for (const float v : row) {
        if (v != -std::numeric_limits<float>::infinity()) {
            s.total += std::exp((static_cast<double>(v) - s.max) * inv_t);
        }
    }
    if (!(s.total > 0.0) || !std::isfinite(s.total)) {
        return Status::error(StatusCode::internal_error, "invalid categorical probability sum", "row");
    }
    return Status::success();
}

// Probability mass of element j under the row's limiting softmax.
[[nodiscard]] double softmax_mass_at(luisa::span<const float> row, float temperature,
                                     const SoftmaxStats &s, uint32_t j) noexcept {
    if (s.pos_inf_count != 0u) {
        return row[j] == std::numeric_limits<float>::infinity()
                   ? 1.0 / static_cast<double>(s.pos_inf_count)
                   : 0.0;
    }
    if (row[j] == -std::numeric_limits<float>::infinity()) { return 0.0; }
    const double w = std::exp((static_cast<double>(row[j]) - s.max) /
                              static_cast<double>(temperature));
    return w / s.total;
}

// Composed (round, index, tag) draw identity: nested SplitMix64 keeps every
// component fully avalanched, so distinct triples give distinct identities
// for all practical purposes.
[[nodiscard]] uint64_t speculative_draw_id(uint64_t round, uint64_t index,
                                           uint32_t tag) noexcept {
    return qwen_speculative_mix64(
        round ^ qwen_speculative_mix64(index + 0x9E3779B97F4A7C15ull *
                                                     (static_cast<uint64_t>(tag) + 1u)));
}

}// namespace

[[nodiscard]] Status QwenSpeculativeLogits::validate() const noexcept {
    if (rows == 0u) { return Status::success(); }// empty view (allowed when unused)
    if (base == nullptr) {
        return invalid("logits view has rows but a null base pointer", "base");
    }
    if (vocab == 0u) {
        return invalid("logits view has rows but zero vocab", "vocab");
    }
    if (stride != 0u && stride < vocab) {
        return invalid("logits view stride is smaller than the row vocab", "stride");
    }
    return Status::success();
}

[[nodiscard]] uint64_t qwen_speculative_mix64(uint64_t x) noexcept {
    // MUST bit-match the anonymous splitmix64 in engine.cpp: the uniform
    // below reuses qwen_sample's two-level hash shape.
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30u)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27u)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31u);
}

[[nodiscard]] double qwen_speculative_uniform(uint64_t seed, uint64_t round,
                                              uint64_t index, uint32_t tag) noexcept {
    // Same construction as qwen_sample's uniform_open01 (engine.cpp): 53
    // bits, open interval (0,1) -- endpoint-free so threshold walks never
    // select zero-probability elements or run past the row.
    const auto bits = qwen_speculative_mix64(seed ^ qwen_speculative_mix64(
                                                   speculative_draw_id(round, index, tag))) >>
                      11u;
    return (static_cast<double>(bits) + 0.5) * (1.0 / 9007199254740992.0);
}

[[nodiscard]] Status qwen_softmax_probability(luisa::span<const float> row,
                                              float temperature, uint32_t index,
                                              double &out) noexcept {
    if (auto s = check_temperature(temperature, "temperature"); !s.ok()) { return s; }
    if (index >= row.size()) {
        return invalid("probability index outside the logits row", "index");
    }
    SoftmaxStats stats;
    if (auto s = softmax_stats(row, temperature, stats); !s.ok()) { return s; }
    out = softmax_mass_at(row, temperature, stats, index);
    return Status::success();
}

[[nodiscard]] Status qwen_categorical_sample(luisa::span<const float> row,
                                             float temperature, double u,
                                             uint32_t &out) noexcept {
    if (auto s = check_temperature(temperature, "temperature"); !s.ok()) { return s; }
    if (auto s = check_unit_interval(u, "u"); !s.ok()) { return s; }
    SoftmaxStats stats;
    if (auto s = softmax_stats(row, temperature, stats); !s.ok()) { return s; }
    // +inf rows: mass shared equally among the +inf entries (rank selection,
    // identical to qwen_sample).
    if (stats.pos_inf_count != 0u) {
        auto rank = static_cast<uint32_t>(u * static_cast<double>(stats.pos_inf_count));
        for (uint32_t i = 0; i < static_cast<uint32_t>(row.size()); i++) {
            if (row[i] == std::numeric_limits<float>::infinity() && rank-- == 0u) {
                out = i;
                return Status::success();
            }
        }
        return Status::error(StatusCode::internal_error, "failed to rank a +infinity logit", "row");
    }
    const double threshold = u * stats.total;
    const double inv_t = 1.0 / static_cast<double>(temperature);
    double cumulative = 0.0;
    uint32_t last_nonzero = 0u;
    for (uint32_t i = 0; i < static_cast<uint32_t>(row.size()); i++) {
        const float v = row[i];
        if (v == -std::numeric_limits<float>::infinity()) { continue; }
        const double w = std::exp((static_cast<double>(v) - stats.max) * inv_t);
        if (w > 0.0) { last_nonzero = i; }
        cumulative += w;
        if (threshold < cumulative) {
            out = i;
            return Status::success();
        }
    }
    // Guard only against the final floating-point accumulation boundary.
    out = last_nonzero;
    return Status::success();
}

[[nodiscard]] Status qwen_residual_sample(luisa::span<const float> target_row,
                                          luisa::span<const float> draft_row,
                                          float temperature, double u,
                                          uint32_t &out) noexcept {
    if (auto s = check_temperature(temperature, "temperature"); !s.ok()) { return s; }
    if (auto s = check_unit_interval(u, "u"); !s.ok()) { return s; }
    if (target_row.size() != draft_row.size()) {
        return invalid("residual sampling requires rows of equal size", "draft_row");
    }
    SoftmaxStats p_stats;
    if (auto s = softmax_stats(target_row, temperature, p_stats); !s.ok()) { return s; }
    SoftmaxStats q_stats;
    if (auto s = softmax_stats(draft_row, temperature, q_stats); !s.ok()) { return s; }
    // Total residual mass: max(p - q, 0) summed over the vocabulary. Zero is
    // mathematically unreachable on a genuine rejection (it would require
    // p == q, whose accept ratio is 1), so fail closed if it ever shows up.
    double total = 0.0;
    for (uint32_t j = 0; j < static_cast<uint32_t>(target_row.size()); j++) {
        total += std::max(softmax_mass_at(target_row, temperature, p_stats, j) -
                              softmax_mass_at(draft_row, temperature, q_stats, j),
                          0.0);
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        return invalid("residual distribution has zero mass (target == draft)", "row");
    }
    const double threshold = u * total;
    double cumulative = 0.0;
    uint32_t last_positive = 0u;
    for (uint32_t j = 0; j < static_cast<uint32_t>(target_row.size()); j++) {
        const double r = std::max(softmax_mass_at(target_row, temperature, p_stats, j) -
                                      softmax_mass_at(draft_row, temperature, q_stats, j),
                                  0.0);
        if (r > 0.0) { last_positive = j; }
        cumulative += r;
        if (threshold < cumulative) {
            out = j;
            return Status::success();
        }
    }
    out = last_positive;// final floating-point boundary guard
    return Status::success();
}

[[nodiscard]] Status qwen_speculative_accept(const QwenSpeculativeInput &in,
                                             QwenSpeculativeResult &out) noexcept {
    out = QwenSpeculativeResult{};
    const uint64_t k = in.draft_tokens.size();
    if (k == 0u || k > 0xFFFFFFFFull) {
        return invalid("speculative rounds require between 1 and 2^32-1 draft tokens",
                       "draft_tokens");
    }
    if (auto s = in.target_logits.validate(); !s.ok()) { return s; }
    if (in.target_logits.rows < k + 1u) {
        return invalid("target logits need at least k+1 rows (draft tokens plus the "
                       "all-accept bonus row)",
                       "target_logits.rows");
    }
    const float temperature = in.options.temperature;
    if (!(temperature >= 0.0f) || !std::isfinite(temperature)) {
        return invalid("temperature must be zero (greedy) or finite and positive",
                       "options.temperature");
    }
    for (uint64_t j = 0; j < k; j++) {
        if (in.draft_tokens[static_cast<size_t>(j)] >= in.target_logits.vocab) {
            return invalid(luisa::format("draft token id {} outside the vocabulary",
                                         in.draft_tokens[static_cast<size_t>(j)]),
                           "draft_tokens");
        }
    }

    if (temperature == 0.0f) {
        // Greedy: accept a draft iff it equals the target's own greedy choice
        // at that position. Rows past the first mismatch are never examined.
        for (uint64_t j = 0; j < k; j++) {
            auto choice = qwen_argmax(in.target_logits.row(j));
            if (!choice.ok()) { return choice.status; }
            if (in.draft_tokens[static_cast<size_t>(j)] != *choice.value) {
                out.accepted = static_cast<uint32_t>(j);
                out.emitted = out.accepted + 1u;
                out.bonus = *choice.value;
                return Status::success();
            }
        }
        auto choice = qwen_argmax(in.target_logits.row(k));
        if (!choice.ok()) { return choice.status; }
        out.accepted = static_cast<uint32_t>(k);
        out.emitted = out.accepted + 1u;
        out.bonus = *choice.value;
        return Status::success();
    }

    // Sampling: p and q use the SAME temperature. Accept d iff
    // r * q(d) < p(d)  <=>  r < min(1, p(d)/q(d))  (r in (0,1) open).
    if (auto s = in.draft_logits.validate(); !s.ok()) { return s; }
    if (in.draft_logits.rows < k) {
        return invalid("draft logits need at least k rows (one draw distribution per "
                       "drafted token) in sampling mode",
                       "draft_logits.rows");
    }
    if (in.draft_logits.vocab != in.target_logits.vocab) {
        return invalid("draft and target logits must share one vocabulary", "vocab");
    }
    for (uint64_t j = 0; j < k; j++) {
        const uint32_t d = in.draft_tokens[static_cast<size_t>(j)];
        double p_d = 0.0;
        if (auto s = qwen_softmax_probability(in.target_logits.row(j), temperature, d, p_d);
            !s.ok()) {
            return s;
        }
        double q_d = 0.0;
        if (auto s = qwen_softmax_probability(in.draft_logits.row(j), temperature, d, q_d);
            !s.ok()) {
            return s;
        }
        const double r = qwen_speculative_uniform(in.options.seed, in.round, j,
                                                  kQwenSpecTagAccept);
        if (!(r * q_d < p_d)) {
            const double u = qwen_speculative_uniform(in.options.seed, in.round, j,
                                                      kQwenSpecTagBonus);
            if (auto s = qwen_residual_sample(in.target_logits.row(j), in.draft_logits.row(j),
                                              temperature, u, out.bonus);
                !s.ok()) {
                return s;
            }
            out.accepted = static_cast<uint32_t>(j);
            out.emitted = out.accepted + 1u;
            return Status::success();
        }
    }
    // All k drafts accepted: bonus from the EXTRA target row k.
    const double u = qwen_speculative_uniform(in.options.seed, in.round, k,
                                              kQwenSpecTagBonus);
    if (auto s = qwen_categorical_sample(in.target_logits.row(k), temperature, u, out.bonus);
        !s.ok()) {
        return s;
    }
    out.accepted = static_cast<uint32_t>(k);
    out.emitted = out.accepted + 1u;
    return Status::success();
}

}// namespace luisa::qwen
