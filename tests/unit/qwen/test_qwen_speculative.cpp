// Test for src/qwen/qwen/speculative.h/.cpp -- the pure-host speculative
// decoding acceptance engine (NextN/MTP draft+verify rounds). Host-only: no
// device, no backend, no model file.
//
// Coverage: greedy accept/reject shapes (all-accept, first-reject, partial),
// lowest-index qwen_argmax tie semantics, lazy row examination, +inf/NaN
// handling per mode, emitted/accepted counts, EOS-agnostic id handling,
// strided views, temperature sampling shapes (accept ratio, residual bonus,
// extra-row bonus), reproducibility and (round/index/tag) draw separation,
// temperature scaling of the accept ratio, and fail-closed malformed inputs.
#include "ut/ut.hpp"

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include <luisa/core/stl/vector.h>

#include "qwen/speculative.h"

using namespace luisa;
using namespace luisa::qwen;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// exp(-800) underflows to exactly 0.0 in double: rows built with kNeg as the
// "impossible" logit get EXACT point masses, so accept thresholds are exact.
constexpr float kNeg = -800.0f;
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// Rows of floats -> packed or NaN-padded strided matrix view. Padding is
// poison: any engine read outside a row's vocab fails comparisons loudly.
struct Matrix {
    luisa::vector<float> storage;
    QwenSpeculativeLogits view;
};

Matrix make_matrix(std::initializer_list<std::initializer_list<float>> rows,
                   uint64_t pad = 0u, bool force_packed_view = false) {
    const uint64_t vocab = static_cast<uint64_t>(rows.begin()->size());
    const uint64_t stride = vocab + pad;
    Matrix m;
    m.storage.assign(static_cast<size_t>(rows.size()) * static_cast<size_t>(stride), kNaN);
    uint64_t r = 0;
    for (const auto &row : rows) {
        uint64_t c = 0;
        for (const float v : row) {
            m.storage[static_cast<size_t>(r * stride + c)] = v;
            ++c;
        }
        ++r;
    }
    m.view = QwenSpeculativeLogits{m.storage.data(), static_cast<uint64_t>(rows.size()),
                                   vocab, force_packed_view ? 0u : stride};
    return m;
}

// Deterministic search over the caller's round counter using the PUBLIC draw
// function: returns the smallest round in [0, limit) satisfying pred.
template<typename F>
uint64_t find_round(uint64_t seed, uint64_t index, uint32_t tag, F &&pred,
                    uint64_t limit = 100000u) {
    for (uint64_t round = 0; round < limit; ++round) {
        if (pred(qwen_speculative_uniform(seed, round, index, tag))) { return round; }
    }
    return limit;
}

QwenSpeculativeInput make_input(luisa::span<const uint32_t> drafts, QwenSpeculativeLogits target,
                                QwenSpeculativeLogits draft, float temperature,
                                uint64_t round = 0u, uint64_t seed = 7u) {
    QwenSpeculativeInput in;
    in.draft_tokens = drafts;
    in.target_logits = target;
    in.draft_logits = draft;
    in.options.temperature = temperature;
    in.options.seed = seed;
    in.round = round;
    return in;
}

// ---------------------------------------------------------------------------
// Logistics view + uniform hash properties.
// ---------------------------------------------------------------------------

void test_view_validation() {
    expect(QwenSpeculativeLogits{}.validate().ok());// empty is allowed
    expect(!QwenSpeculativeLogits{nullptr, 3, 4, 4}.validate().ok());
    expect(QwenSpeculativeLogits{nullptr, 0, 4, 0}.validate().ok());// rows=0: null ok
    luisa::vector<float> buf(16, 0.0f);
    expect(!QwenSpeculativeLogits{buf.data(), 2, 0, 4}.validate().ok());// zero vocab
    expect(!QwenSpeculativeLogits{buf.data(), 2, 4, 3}.validate().ok());// stride < vocab
    expect(QwenSpeculativeLogits{buf.data(), 2, 4, 4}.validate().ok());
    expect(QwenSpeculativeLogits{buf.data(), 2, 4, 0}.validate().ok());// packed
}

void test_mix64_known_vector() {
    // Standard SplitMix64 first-output vector (state 0): pins the hash so the
    // qwen_sample-flavored uniforms stay cross-module stable.
    expect(qwen_speculative_mix64(0u) == 0xE220A8397B1DCDAFull);
}

void test_uniform_properties() {
    // Open interval (0,1), reproducible, and (round,index,tag)-separated.
    double first = 0.0;
    bool any_diff_index = false;
    bool any_diff_tag = false;
    bool any_diff_round = false;
    bool seen_first = false;
    for (uint64_t i = 0; i < 64; i++) {
        const double a = qwen_speculative_uniform(7u, 11u, i, kQwenSpecTagAccept);
        const double b = qwen_speculative_uniform(7u, 11u, i, kQwenSpecTagAccept);
        expect(a == b);            // same identity -> same draw
        expect(a > 0.0);           // open interval
        expect(a < 1.0);
        const double c = qwen_speculative_uniform(7u, 11u, i, kQwenSpecTagBonus);
        if (c != a) { any_diff_tag = true; }
        const double d = qwen_speculative_uniform(7u, 12u, i, kQwenSpecTagAccept);
        if (d != a) { any_diff_round = true; }
        if (!seen_first) {
            first = a;
            seen_first = true;
        } else if (a != first) {
            any_diff_index = true;
        }
    }
    expect(any_diff_index);
    expect(any_diff_tag);
    expect(any_diff_round);
}

// ---------------------------------------------------------------------------
// Greedy mode.
// ---------------------------------------------------------------------------

void test_greedy_all_accept() {
    // qwen_argmax returns the INDEX of the maximum: each row's max must sit
    // exactly at the index equal to the drafted id.
    const uint32_t drafts[] = {5u, 9u, 1u, 1u};
    auto t = make_matrix({{0, 0, 0, 0, 0, 9, 0, 0, 0, 0},// argmax index 5
                          {0, 0, 0, 0, 0, 0, 0, 0, 0, 9},// argmax index 9
                          {0, 9, 0, 0, 0, 0, 0, 0, 0, 0},// argmax index 1
                          {0, 9, 0, 0, 0, 0, 0, 0, 0, 0},// argmax index 1
                          {0, 0, 0, 9, 0, 0, 0, 0, 0, 0}});// extra row: bonus index 3
    auto in = make_input(drafts, t.view, QwenSpeculativeLogits{}, 0.0f);
    QwenSpeculativeResult out;
    expect(qwen_speculative_accept(in, out).ok());
    expect(out.accepted == 4u);
    expect(out.emitted == 5u);
    expect(out.bonus == 3u);
}

void test_greedy_first_reject_and_laziness() {
    const uint32_t drafts[] = {6u, 0u, 0u};
    // Rows 1..3 are NaN: they must NEVER be examined after the row-0 mismatch.
    auto t = make_matrix({{0, 0, 0, 0, 5, 0, 0}, {kNaN}, {kNaN}, {kNaN}});
    auto in = make_input(drafts, t.view, QwenSpeculativeLogits{}, 0.0f);
    QwenSpeculativeResult out;
    expect(qwen_speculative_accept(in, out).ok());
    expect(out.accepted == 0u);
    expect(out.emitted == 1u);
    expect(out.bonus == 4u);// argmax index of the 5
}

void test_greedy_partial_reject() {
    const uint32_t drafts[] = {5u, 7u, 8u};
    auto t = make_matrix({{0, 0, 0, 0, 0, 9, 0, 0, 0, 0},   // accept 5 (index 5)
                          {0, 0, 0, 0, 0, 0, 0, 9, 0, 0},   // accept 7 (index 7)
                          {0, 0, 0, 0, 0, 0, 0, 0, 0, 9},   // reject 8 vs index 9
                          {kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN}});// never examined
    auto in = make_input(drafts, t.view, QwenSpeculativeLogits{}, 0.0f);
    QwenSpeculativeResult out;
    expect(qwen_speculative_accept(in, out).ok());
      expect(out.accepted == 2u);
      expect(out.emitted == 3u);
      expect(out.bonus == 9u);
  }

  void test_greedy_tie_lowest_index() {
    // qwen_argmax must win the tie at indices 0/1 with value 3.
    const uint32_t bad[] = {1u};
    auto t = make_matrix({{3, 3, 1, 0}, {0, 0, 0, 0}});
    auto in_bad = make_input(bad, t.view, QwenSpeculativeLogits{}, 0.0f);
    QwenSpeculativeResult out;
    expect(qwen_speculative_accept(in_bad, out).ok());
    expect(out.accepted == 0u);
    expect(out.bonus == 0u);
    const uint32_t good[] = {0u};
    auto in_good = make_input(good, t.view, QwenSpeculativeLogits{}, 0.0f);
    expect(qwen_speculative_accept(in_good, out).ok());
    expect(out.accepted == 1u);
    expect(out.bonus == 0u);// extra row argmax
}

void test_greedy_plus_inf() {
    // qwen_argmax orders infinities normally: first +inf index wins.
    const uint32_t bad[] = {3u};
    auto t = make_matrix({{1, kInf, kInf, 2}, {0, 0, 0, 0}});
    QwenSpeculativeResult out;
    auto in_bad = make_input(bad, t.view, QwenSpeculativeLogits{}, 0.0f);
    expect(qwen_speculative_accept(in_bad, out).ok());
    expect(out.bonus == 1u);
    const uint32_t good[] = {1u};
    auto in_good = make_input(good, t.view, QwenSpeculativeLogits{}, 0.0f);
    expect(qwen_speculative_accept(in_good, out).ok());
    expect(out.accepted == 1u);
}

void test_greedy_nan_and_bonus_row() {
    QwenSpeculativeResult out;
    // NaN in the FIRST examined row -> failure.
    const uint32_t d0[] = {0u};
    auto t_nan = make_matrix({{0, kNaN, 0, 0}, {0, 0, 0, 0}});
    auto in_nan = make_input(d0, t_nan.view, QwenSpeculativeLogits{}, 0.0f);
    expect(!qwen_speculative_accept(in_nan, out).ok());
    // All drafts accepted but the EXTRA row is NaN -> failure (it IS examined).
    const uint32_t d1[] = {2u};
    auto t_extra = make_matrix({{0, 0, 2, 0}, {0, kNaN, 0, 0}});
    auto in_extra = make_input(d1, t_extra.view, QwenSpeculativeLogits{}, 0.0f);
    expect(!qwen_speculative_accept(in_extra, out).ok());
}

void test_greedy_ignores_draft_logits_and_strides() {
    // Greedy mode never looks at draft_logits: garbage view is fine.
    const uint32_t drafts[] = {1u, 2u};
    auto t = make_matrix({{0, 1, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 3}}, 2u);
    QwenSpeculativeLogits garbage{nullptr, 99u, 4u, 1u};// null base, tiny stride
    auto in = make_input(drafts, t.view, garbage, 0.0f);
    QwenSpeculativeResult out;
    expect(qwen_speculative_accept(in, out).ok());
    expect(out.accepted == 2u);
    expect(out.bonus == 3u);
    // Packed (stride 0) view of the same data gives the same answer.
    auto t_packed = make_matrix({{0, 1, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 3}}, 0u, true);
    auto in_packed = make_input(drafts, t_packed.view, garbage, 0.0f);
    expect(qwen_speculative_accept(in_packed, out).ok());
    expect(out.accepted == 2u && out.bonus == 3u);
}

void test_greedy_id_range() {
    QwenSpeculativeResult out;
    const uint32_t oob[] = {4u};// vocab is 4
    auto t = make_matrix({{0, 0, 0, 0}, {0, 0, 0, 0}});
    auto in = make_input(oob, t.view, QwenSpeculativeLogits{}, 0.0f);
    expect(!qwen_speculative_accept(in, out).ok());
    const uint32_t edge[] = {3u};// vocab-1 is valid; row max sits at index 3
    auto t_edge = make_matrix({{0, 0, 0, 9}, {0, 0, 0, 0}});
    auto in_edge = make_input(edge, t_edge.view, QwenSpeculativeLogits{}, 0.0f);
    expect(qwen_speculative_accept(in_edge, out).ok());
    expect(out.accepted == 1u);
}

void test_negative_zero_temperature_is_greedy() {
    const uint32_t drafts[] = {2u};
    auto t = make_matrix({{0, 0, 2, 0}, {0, 0, 0, 1}});
    auto in = make_input(drafts, t.view, QwenSpeculativeLogits{}, -0.0f);
    QwenSpeculativeResult out;
    expect(qwen_speculative_accept(in, out).ok());
    expect(out.accepted == 1u);// took the greedy path without draft logits
}

// ---------------------------------------------------------------------------
// Building blocks: softmax probability / categorical / residual.
// ---------------------------------------------------------------------------

void test_softmax_probability_basic() {
    double p = -1.0;
    const float uniform4[] = {0, 0, 0, 0};
    expect(qwen_softmax_probability(luisa::span<const float>{uniform4, 4}, 1.0f, 0u, p).ok());
    expect(p == 0.25);// exact: symmetric zeros
    expect(qwen_softmax_probability(luisa::span<const float>{uniform4, 4}, 2.0f, 3u, p).ok());
    expect(p == 0.25);// temperature preserves symmetry
    const float point[] = {0, kNeg, kNeg, kNeg};
    expect(qwen_softmax_probability(luisa::span<const float>{point, 4}, 1.0f, 0u, p).ok());
    expect(p == 1.0);
    expect(qwen_softmax_probability(luisa::span<const float>{point, 4}, 1.0f, 2u, p).ok());
    expect(p == 0.0);
}

void test_softmax_probability_pos_inf() {
    const float row[] = {kInf, kInf, 0, kNeg};
    double p = -1.0;
    expect(qwen_softmax_probability(luisa::span<const float>{row, 4}, 1.0f, 0u, p).ok());
    expect(p == 0.5);// limiting distribution: mass shared among +inf
    expect(qwen_softmax_probability(luisa::span<const float>{row, 4}, 1.0f, 1u, p).ok());
    expect(p == 0.5);
    expect(qwen_softmax_probability(luisa::span<const float>{row, 4}, 1.0f, 2u, p).ok());
    expect(p == 0.0);
    const float single[] = {kInf, 0, kNeg, kNeg};
    expect(qwen_softmax_probability(luisa::span<const float>{single, 4}, 1.0f, 0u, p).ok());
    expect(p == 1.0);
}

void test_softmax_probability_failures() {
    double p = 0.0;
    const float nan_row[] = {0, kNaN, 0, 0};
    expect(!qwen_softmax_probability(luisa::span<const float>{nan_row, 4}, 1.0f, 0u, p).ok());
    const float all_neg[] = {-kInf, -kInf, -kInf, -kInf};
    expect(!qwen_softmax_probability(luisa::span<const float>{all_neg, 4}, 1.0f, 0u, p).ok());
    expect(!qwen_softmax_probability(luisa::span<const float>{}, 1.0f, 0u, p).ok());
    const float row[] = {0, 0, 0, 0};
    expect(!qwen_softmax_probability(luisa::span<const float>{row, 4}, 0.0f, 0u, p).ok());
    expect(!qwen_softmax_probability(luisa::span<const float>{row, 4}, -1.0f, 0u, p).ok());
    expect(!qwen_softmax_probability(luisa::span<const float>{row, 4}, kNaN, 0u, p).ok());
    expect(!qwen_softmax_probability(luisa::span<const float>{row, 4}, kInf, 0u, p).ok());
    expect(!qwen_softmax_probability(luisa::span<const float>{row, 4}, 1.0f, 4u, p).ok());
}

void test_categorical_sample_basic() {
    uint32_t out = 999u;
    const float point[] = {0, kNeg, kNeg, kNeg};
    expect(qwen_categorical_sample(luisa::span<const float>{point, 4}, 1.0f, 0.5, out).ok());
    expect(out == 0u);// point mass: any u picks it
    const float half[] = {0, 0, kNeg, kNeg};
    // masses exactly [1/2, 1/2]: u < 0.5 -> 0, else 1.
    const uint64_t r_lo = find_round(7u, 0u, kQwenSpecTagBonus, [](double u) { return u < 0.5; });
    const uint64_t r_hi = find_round(7u, 0u, kQwenSpecTagBonus, [](double u) { return !(u < 0.5); });
    expect(r_lo < 100000u && r_hi < 100000u);
    expect(qwen_categorical_sample(luisa::span<const float>{half, 4}, 1.0f,
                                   qwen_speculative_uniform(7u, r_lo, 0u, kQwenSpecTagBonus), out)
               .ok());
    expect(out == 0u);
    expect(qwen_categorical_sample(luisa::span<const float>{half, 4}, 1.0f,
                                   qwen_speculative_uniform(7u, r_hi, 0u, kQwenSpecTagBonus), out)
               .ok());
    expect(out == 1u);
}

void test_categorical_sample_pos_inf() {
    uint32_t out = 999u;
    const float row[] = {kInf, kInf, 0, kNeg};
    const uint64_t r_lo = find_round(7u, 0u, kQwenSpecTagBonus, [](double u) { return u < 0.5; });
    expect(qwen_categorical_sample(luisa::span<const float>{row, 4}, 1.0f,
                                   qwen_speculative_uniform(7u, r_lo, 0u, kQwenSpecTagBonus), out)
               .ok());
    expect(out == 0u);
    expect(qwen_categorical_sample(luisa::span<const float>{row, 4}, 1.0f, 0.999999, out).ok());
    expect(out == 1u);
}

void test_categorical_sample_failures() {
    uint32_t out = 0u;
    const float nan_row[] = {0, kNaN, 0, 0};
    expect(!qwen_categorical_sample(luisa::span<const float>{nan_row, 4}, 1.0f, 0.5, out).ok());
    const float all_neg[] = {-kInf, -kInf, -kInf, -kInf};
    expect(!qwen_categorical_sample(luisa::span<const float>{all_neg, 4}, 1.0f, 0.5, out).ok());
    const float row[] = {0, 0, 0, 0};
    expect(!qwen_categorical_sample(luisa::span<const float>{}, 1.0f, 0.5, out).ok());
    expect(!qwen_categorical_sample(luisa::span<const float>{row, 4}, 1.0f, 0.0, out).ok());
    expect(!qwen_categorical_sample(luisa::span<const float>{row, 4}, 1.0f, 1.0, out).ok());
    expect(!qwen_categorical_sample(luisa::span<const float>{row, 4}, 1.0f, -0.1, out).ok());
    expect(!qwen_categorical_sample(luisa::span<const float>{row, 4}, 1.0f, kNaN, out).ok());
}

void test_residual_sample_basic() {
    uint32_t out = 999u;
    // p uniform [1/4 x4], q = [1/2, 1/2, 0, 0] -> residual [0, 0, 1/4, 1/4].
    const float p_row[] = {0, 0, 0, 0};
    const float q_row[] = {0, 0, kNeg, kNeg};
    const uint64_t r_lo = find_round(7u, 0u, kQwenSpecTagBonus, [](double u) { return u < 0.5; });
    const uint64_t r_hi = find_round(7u, 0u, kQwenSpecTagBonus, [](double u) { return !(u < 0.5); });
    expect(qwen_residual_sample(luisa::span<const float>{p_row, 4},
                                luisa::span<const float>{q_row, 4}, 1.0f,
                                qwen_speculative_uniform(7u, r_lo, 0u, kQwenSpecTagBonus), out)
               .ok());
    expect(out == 2u);
    expect(qwen_residual_sample(luisa::span<const float>{p_row, 4},
                                luisa::span<const float>{q_row, 4}, 1.0f,
                                qwen_speculative_uniform(7u, r_hi, 0u, kQwenSpecTagBonus), out)
               .ok());
    expect(out == 3u);
}

void test_residual_sample_pos_inf_and_failures() {
    uint32_t out = 999u;
    // p = point mass at 0 via +inf, q uniform: residual = [3/4, 0, 0, 0].
    const float p_row[] = {kInf, 0, kNeg, kNeg};
    const float q_row[] = {0, 0, 0, 0};
    for (double u : {0.1, 0.5, 0.9}) {
        expect(qwen_residual_sample(luisa::span<const float>{p_row, 4},
                                    luisa::span<const float>{q_row, 4}, 1.0f, u, out)
                   .ok());
        expect(out == 0u);
    }
    // Zero residual mass (identical rows) -> fail closed.
    const float same[] = {0, 0, kNeg, kNeg};
    expect(!qwen_residual_sample(luisa::span<const float>{same, 4},
                                 luisa::span<const float>{same, 4}, 1.0f, 0.5, out)
               .ok());
    // Size mismatch / NaN / all -inf / bad temperature / bad u.
    expect(!qwen_residual_sample(luisa::span<const float>{same, 4},
                                 luisa::span<const float>{same, 3}, 1.0f, 0.5, out)
               .ok());
    const float nan_row[] = {0, kNaN, 0, 0};
    expect(!qwen_residual_sample(luisa::span<const float>{nan_row, 4},
                                 luisa::span<const float>{same, 4}, 1.0f, 0.5, out)
               .ok());
    const float all_neg[] = {-kInf, -kInf, -kInf, -kInf};
    expect(!qwen_residual_sample(luisa::span<const float>{all_neg, 4},
                                 luisa::span<const float>{same, 4}, 1.0f, 0.5, out)
               .ok());
    expect(!qwen_residual_sample(luisa::span<const float>{same, 4},
                                 luisa::span<const float>{same, 4}, 0.0f, 0.5, out)
               .ok());
    expect(!qwen_residual_sample(luisa::span<const float>{same, 4},
                                 luisa::span<const float>{same, 4}, 1.0f, 1.0, out)
               .ok());
}

// ---------------------------------------------------------------------------
// Sampling mode: shapes.
// ---------------------------------------------------------------------------

void test_sampling_all_accept_bonus_from_extra_row() {
    // Guaranteed accept: every draft row puts ZERO mass on its drafted id
    // (q_d == 0 < p_d), so all drafts pass and the bonus comes from row k.
    const uint32_t d2[] = {0u, 1u};
    auto tt = make_matrix({{kNeg, 0, 0, 0},   // p row 0: mass at 1; d=0 -> p_d=0... no.
                           {0, 0, 0, 0},
                           {0, 0, 0, 0}});
    (void)tt;
    // Simplest guaranteed-accept construction: draft rows are all -inf except
    // an index the draft did NOT pick (q_d == 0 < any p_d > 0).
    auto t3 = make_matrix({{0, 0, kNeg, kNeg},   // p0: [1/2,1/2,0,0]; d0=0 -> p_d=1/2
                           {kNeg, 0, 0, kNeg},   // p1: [0,1/2,1/2,0]; d1=1 -> p_d=1/2
                           {0, kNeg, kNeg, kNeg}});// extra row: point mass at 0
    auto q3 = make_matrix({{kNeg, 0, kNeg, kNeg},   // q0: d=0 -> q_d = 0 -> accept
                           {0, kNeg, kNeg, kNeg}});// q1: d=1 -> q_d = 0 -> accept
    auto in = make_input(d2, t3.view, q3.view, 1.0f);
    QwenSpeculativeResult out;
    expect(qwen_speculative_accept(in, out).ok());
    expect(out.accepted == 2u);
    expect(out.emitted == 3u);
    expect(out.bonus == 0u);// softmax of the extra row picks its point mass
}

void test_sampling_first_reject_residual_bonus() {
    // k=1: p uniform (.25 each), q = [1/2,1/2,0,0]: d=0 -> accept iff r < 1/2.
    // On rejection the bonus comes from the residual [0,0,1/2,1/2] normalized.
    const uint32_t drafts[] = {0u};
    auto t = make_matrix({{0, 0, 0, 0}, {0, 0, 0, 0}});
    auto q = make_matrix({{0, 0, kNeg, kNeg}});
    const uint64_t seed = 7u;
    const uint64_t r_accept = find_round(seed, 0u, kQwenSpecTagAccept,
                                         [](double r) { return r < 0.5; });
    const uint64_t r_reject_lo = find_round(seed, 0u, kQwenSpecTagAccept,
                                            [](double r) { return !(r < 0.5); });
    // Among rejecting rounds, split again by the bonus draw (tag Bonus, index 0).
    uint64_t r_reject_bonus_lo = 100000u;
    uint64_t r_reject_bonus_hi = 100000u;
    for (uint64_t r = r_reject_lo; r < 100000u; ++r) {
        const double r_draw = qwen_speculative_uniform(seed, r, 0u, kQwenSpecTagAccept);
        if (r_draw < 0.5) { continue; }
        const double u_draw = qwen_speculative_uniform(seed, r, 0u, kQwenSpecTagBonus);
        if (u_draw < 0.5 && r_reject_bonus_lo == 100000u) { r_reject_bonus_lo = r; }
        if (!(u_draw < 0.5) && r_reject_bonus_hi == 100000u) { r_reject_bonus_hi = r; }
        if (r_reject_bonus_lo != 100000u && r_reject_bonus_hi != 100000u) { break; }
    }
    expect(r_reject_bonus_lo < 100000u && r_reject_bonus_hi < 100000u);

    QwenSpeculativeResult out;
    auto in_accept = make_input(drafts, t.view, q.view, 1.0f, r_accept, seed);
    expect(qwen_speculative_accept(in_accept, out).ok());
    expect(out.accepted == 1u);
    expect(out.emitted == 2u);
    // The extra row is uniform here, so assert the bonus against the
    // categorical reference at the engine's exact bonus draw.
    uint32_t ref = 999u;
    const float extra[] = {0, 0, 0, 0};
    expect(qwen_categorical_sample(luisa::span<const float>{extra, 4}, 1.0f,
                                   qwen_speculative_uniform(seed, r_accept, 1u,
                                                            kQwenSpecTagBonus),
                                   ref)
               .ok());
    expect(out.bonus == ref);

    auto in_reject_lo = make_input(drafts, t.view, q.view, 1.0f, r_reject_bonus_lo, seed);
    expect(qwen_speculative_accept(in_reject_lo, out).ok());
    expect(out.accepted == 0u);
    expect(out.emitted == 1u);
    expect(out.bonus == 2u);// residual [0,0,1/2,1/2]: u < 1/2 -> index 2

    auto in_reject_hi = make_input(drafts, t.view, q.view, 1.0f, r_reject_bonus_hi, seed);
    expect(qwen_speculative_accept(in_reject_hi, out).ok());
    expect(out.accepted == 0u);
    expect(out.bonus == 3u);
}

void test_sampling_partial_accept() {
    // d0 always accepted (q_d == 0), d1 coin-flips on r < 1/2; rejection emits
    // the residual bonus from row 1.
    const uint32_t drafts[] = {0u, 1u};
    auto t = make_matrix({{0, 0, kNeg, kNeg},// p0: [1/2,1/2,0,0]; d0 -> 1/2
                          {0, 0, 0, 0},      // p1: uniform; d1 -> 1/4
                          {0, 0, 0, 0}});
    auto q = make_matrix({{kNeg, 0, kNeg, kNeg},// q0: d0 -> 0 -> always accept
                          {0, 0, kNeg, kNeg}});// q1: [1/2,1/2,0,0]; d1 -> 1/2
    // Acceptance draw for d1 uses (round, index 1, tag Accept).
    const uint64_t seed = 9u;
    const uint64_t r_reject = find_round(seed, 1u, kQwenSpecTagAccept,
                                         [](double r) { return !(r < 0.5); });
    auto in = make_input(drafts, t.view, q.view, 1.0f, r_reject, seed);
    QwenSpeculativeResult out;
    expect(qwen_speculative_accept(in, out).ok());
    expect(out.accepted == 1u);
    expect(out.emitted == 2u);
    // Residual of p1(uniform) - q1([1/2,1/2,0,0]) = [0,0,1/4,1/4] -> bonus 2/3.
    expect(out.bonus == 2u || out.bonus == 3u);
    uint32_t ref = 999u;
    expect(qwen_residual_sample(t.view.row(1), q.view.row(1), 1.0f,
                                qwen_speculative_uniform(seed, r_reject, 1u, kQwenSpecTagBonus),
                                ref)
               .ok());
    expect(out.bonus == ref);
}

void test_sampling_reproducible_and_seed_round_sensitive() {
    const uint32_t drafts[] = {0u};
    auto t = make_matrix({{0, 0, 0, 0}, {0, 0, 0, 0}});
    auto q = make_matrix({{0, 0, kNeg, kNeg}});// accept iff r < 1/2
    const uint64_t seed = 7u;
    const uint64_t round = 3u;
    auto in = make_input(drafts, t.view, q.view, 1.0f, round, seed);
    QwenSpeculativeResult a, b;
    expect(qwen_speculative_accept(in, a).ok());
    expect(qwen_speculative_accept(in, b).ok());
    expect(a.accepted == b.accepted && a.emitted == b.emitted && a.bonus == b.bonus);
    // A different seed flips the decision somewhere (deterministic scan).
    bool flipped = false;
    for (uint64_t s = 0; s < 200 && !flipped; ++s) {
        const double r_a = qwen_speculative_uniform(seed, round, 0u, kQwenSpecTagAccept);
        const double r_b = qwen_speculative_uniform(s, round, 0u, kQwenSpecTagAccept);
        if ((r_a < 0.5) != (r_b < 0.5)) {
            auto in_b = make_input(drafts, t.view, q.view, 1.0f, round, s);
            QwenSpeculativeResult c;
            expect(qwen_speculative_accept(in_b, c).ok());
            expect(c.accepted != a.accepted);
            flipped = true;
        }
    }
    expect(flipped);
    // A different round flips the decision somewhere too.
    bool round_flipped = false;
    for (uint64_t r2 = 0; r2 < 200 && !round_flipped; ++r2) {
        const double r_a = qwen_speculative_uniform(seed, round, 0u, kQwenSpecTagAccept);
        const double r_b = qwen_speculative_uniform(seed, r2, 0u, kQwenSpecTagAccept);
        if ((r_a < 0.5) != (r_b < 0.5)) { round_flipped = true; }
    }
    expect(round_flipped);
}

void test_sampling_draw_tag_and_index_separation() {
    // k=1, forced reject: p puts all mass on 1/2, q point-masses d=0.
    // Residual = [0, 1/2, 1/2, 0]: the bonus must be driven by the BONUS-tagged
    // uniform at index 0 -- never by the acceptance draw (same index, tag Accept).
    const uint32_t drafts[] = {0u};
    auto t = make_matrix({{kNeg, 0, 0, kNeg}, {0, 0, 0, 0}});
    auto q = make_matrix({{0, kNeg, kNeg, kNeg}});
    const uint64_t seed = 5u;
    uint64_t hit = 100000u;
    for (uint64_t r = 0; r < 100000; ++r) {
        const double u_acc = qwen_speculative_uniform(seed, r, 0u, kQwenSpecTagAccept);
        const double u_bon = qwen_speculative_uniform(seed, r, 0u, kQwenSpecTagBonus);
        if ((u_acc < 0.5) != (u_bon < 0.5)) { hit = r; break; }
    }
    expect(hit < 100000u);
    auto in = make_input(drafts, t.view, q.view, 1.0f, hit, seed);
    QwenSpeculativeResult out;
    expect(qwen_speculative_accept(in, out).ok());
    expect(out.accepted == 0u);// p_d == 0 -> reject regardless of r
    // Expected bonus follows the BONUS uniform only.
    const bool bonus_lo = qwen_speculative_uniform(seed, hit, 0u, kQwenSpecTagBonus) < 0.5;
    expect(out.bonus == (bonus_lo ? 1u : 2u));
    // Index separation: k=2, forced reject at j=0; the bonus must use
    // (index 0, tag Bonus), not (index 1, tag Bonus).
    const uint32_t drafts2[] = {0u, 3u};
    auto t2 = make_matrix({{kNeg, 0, 0, kNeg}, {0, 0, 0, 0}, {0, 0, 0, 0}});
    auto q2 = make_matrix({{0, kNeg, kNeg, kNeg}, {0, 0, 0, 0}});
    uint64_t hit2 = 100000u;
    for (uint64_t r = 0; r < 100000; ++r) {
        const double u_b0 = qwen_speculative_uniform(seed, r, 0u, kQwenSpecTagBonus);
        const double u_b1 = qwen_speculative_uniform(seed, r, 1u, kQwenSpecTagBonus);
        if ((u_b0 < 0.5) != (u_b1 < 0.5)) { hit2 = r; break; }
    }
    expect(hit2 < 100000u);
    auto in2 = make_input(drafts2, t2.view, q2.view, 1.0f, hit2, seed);
    expect(qwen_speculative_accept(in2, out).ok());
    expect(out.accepted == 0u);
    const bool bonus2_lo = qwen_speculative_uniform(seed, hit2, 0u, kQwenSpecTagBonus) < 0.5;
    expect(out.bonus == (bonus2_lo ? 1u : 2u));
}

void test_sampling_temperature_scales_accept_ratio() {
    // d=1: p = [2,0,-800,-800], q = [2,2,-800,-800].
    //   T=1 : p_d/q_d small -> rounds with r above the ratio reject.
    //   T=8 : ratio near 1   -> rounds with r below the ratio accept.
    const uint32_t drafts[] = {1u};
    auto t = make_matrix({{2, 0, kNeg, kNeg}, {0, 0, 0, 0}});
    auto q = make_matrix({{2, 2, kNeg, kNeg}});
    const uint64_t seed = 3u;
    double p1 = 0.0, q1 = 0.0;
    expect(qwen_softmax_probability(t.view.row(0), 1.0f, 1u, p1).ok());
    expect(qwen_softmax_probability(q.view.row(0), 1.0f, 1u, q1).ok());
    expect(p1 < q1);// low temperature: reject possible
    const uint64_t r_reject = find_round(seed, 0u, kQwenSpecTagAccept,
                                         [&](double r) { return !(r * q1 < p1); });
    QwenSpeculativeResult out;
    auto in_cold = make_input(drafts, t.view, q.view, 1.0f, r_reject, seed);
    expect(qwen_speculative_accept(in_cold, out).ok());
    expect(out.accepted == 0u);
    expect(out.bonus == 0u || out.bonus == 1u);// residual subset of {0,1}

    double p8 = 0.0, q8 = 0.0;
    expect(qwen_softmax_probability(t.view.row(0), 8.0f, 1u, p8).ok());
    expect(qwen_softmax_probability(q.view.row(0), 8.0f, 1u, q8).ok());
    expect(p8 > p1);// higher temperature smooths the target mass toward d
    const uint64_t r_accept = find_round(seed, 0u, kQwenSpecTagAccept,
                                         [&](double r) { return r * q8 < p8; });
    auto in_hot = make_input(drafts, t.view, q.view, 8.0f, r_accept, seed);
    expect(qwen_speculative_accept(in_hot, out).ok());
    expect(out.accepted == 1u);
}

// ---------------------------------------------------------------------------
// Fail-closed malformed inputs at the engine boundary.
// ---------------------------------------------------------------------------

void test_engine_input_validation() {
    QwenSpeculativeResult out;
    const uint32_t one[] = {0u};
    auto t = make_matrix({{0, 0, 0, 0}, {0, 0, 0, 0}});
    auto q = make_matrix({{0, 0, 0, 0}});
    // Empty draft span.
    auto in_empty = make_input(luisa::span<const uint32_t>{}, t.view, q.view, 1.0f);
    expect(!qwen_speculative_accept(in_empty, out).ok());
    // Target rows < k+1.
    auto t_short = make_matrix({{0, 0, 0, 0}});
    auto in_short = make_input(one, t_short.view, q.view, 1.0f);
    expect(!qwen_speculative_accept(in_short, out).ok());
    // Sampling requires draft rows.
    auto in_no_draft = make_input(one, t.view, QwenSpeculativeLogits{}, 1.0f);
    expect(!qwen_speculative_accept(in_no_draft, out).ok());
    auto q_short = make_matrix({{0, 0, 0, 0}});// k=2 below needs 2 rows
    const uint32_t two[] = {0u, 1u};
    auto in_qshort = make_input(two, t.view, q_short.view, 1.0f);
    expect(!qwen_speculative_accept(in_qshort, out).ok());
    // Vocab mismatch between draft and target views.
    auto q_vocab = make_matrix({{0, 0, 0}});// vocab 3 vs 4
    auto in_vocab = make_input(one, t.view, q_vocab.view, 1.0f);
    expect(!qwen_speculative_accept(in_vocab, out).ok());
    // Bad temperatures.
    for (float temp : {kNaN, -1.0f, -0.5f, kInf}) {
        auto in_bad = make_input(one, t.view, q.view, temp);
        expect(!qwen_speculative_accept(in_bad, out).ok());
    }
    // Draft id outside vocab.
    const uint32_t oob[] = {4u};
    auto in_oob = make_input(oob, t.view, q.view, 1.0f);
    expect(!qwen_speculative_accept(in_oob, out).ok());
}

void test_engine_nan_and_zero_mass_rows_sampling() {
    QwenSpeculativeResult out;
    const uint32_t one[] = {0u};
    // NaN in the examined target row.
    auto t_nan = make_matrix({{0, kNaN, 0, 0}, {0, 0, 0, 0}});
    auto q_ok = make_matrix({{0, 0, 0, 0}});
    auto in_nan = make_input(one, t_nan.view, q_ok.view, 1.0f);
    expect(!qwen_speculative_accept(in_nan, out).ok());
    // NaN in the examined draft row.
    auto q_nan = make_matrix({{0, kNaN, 0, 0}});
    auto t_ok = make_matrix({{0, 0, 0, 0}, {0, 0, 0, 0}});
    auto in_qnan = make_input(one, t_ok.view, q_nan.view, 1.0f);
    expect(!qwen_speculative_accept(in_qnan, out).ok());
    // Zero-mass (all -inf) target row.
    auto t_neg = make_matrix({{-kInf, -kInf, -kInf, -kInf}, {0, 0, 0, 0}});
    auto in_neg = make_input(one, t_neg.view, q_ok.view, 1.0f);
    expect(!qwen_speculative_accept(in_neg, out).ok());
    // Zero-mass draft row.
    auto q_neg = make_matrix({{-kInf, -kInf, -kInf, -kInf}});
    auto in_qneg = make_input(one, t_ok.view, q_neg.view, 1.0f);
    expect(!qwen_speculative_accept(in_qneg, out).ok());
    // +inf rows are legal in sampling (limiting softmax mass).
    auto t_inf = make_matrix({{kInf, 0, kNeg, kNeg}, {0, 0, 0, 0}});
    auto q_inf = make_matrix({{0, kInf, kNeg, kNeg}});
    auto in_inf = make_input(one, t_inf.view, q_inf.view, 1.0f);
    expect(qwen_speculative_accept(in_inf, out).ok());
}

// ---------------------------------------------------------------------------
// EOS-agnosticism, counts, and the strided engine path.
// ---------------------------------------------------------------------------

void test_eos_agnostic_and_counts() {
    // No id is special: repeats and "sentinel-like" ids are ordinary tokens.
    const uint32_t drafts[] = {7u, 7u, 7u};
    auto t = make_matrix({{kNeg, kNeg, kNeg, kNeg, kNeg, kNeg, kNeg, 0},// argmax 7
                          {kNeg, kNeg, kNeg, kNeg, kNeg, kNeg, kNeg, 0},
                          {kNeg, kNeg, kNeg, kNeg, kNeg, kNeg, kNeg, 0},
                          {0, kNeg, kNeg, kNeg, kNeg, kNeg, kNeg, kNeg}},// bonus 0
                         3u);// strided with NaN padding
    auto in = make_input(drafts, t.view, QwenSpeculativeLogits{}, 0.0f);
    QwenSpeculativeResult out;
    expect(qwen_speculative_accept(in, out).ok());
    expect(out.accepted == 3u);
    expect(out.emitted == 4u);// accepted + bonus, always
    expect(out.bonus == 0u);
}

void test_strided_sampling_rows() {
    const uint32_t drafts[] = {0u};
    auto t = make_matrix({{0, 0, 0, 0}, {0, 0, kNeg, kNeg}}, 5u);// NaN-padded
    auto q = make_matrix({{kNeg, 0, kNeg, kNeg}}, 2u);// q_d(0) == 0 -> always accept
    auto in = make_input(drafts, t.view, q.view, 1.0f, 42u, 11u);
    QwenSpeculativeResult out;
    expect(qwen_speculative_accept(in, out).ok());
    expect(out.accepted == 1u);
    expect(out.emitted == 2u);
    expect(out.bonus == 0u || out.bonus == 1u);// extra row mass
}

}// namespace

static auto test_qwen_speculative_registration = [] {
    "spec_view_validation"_test = [] { test_view_validation(); };
    "spec_mix64_known_vector"_test = [] { test_mix64_known_vector(); };
    "spec_uniform_properties"_test = [] { test_uniform_properties(); };
    "spec_greedy_all_accept"_test = [] { test_greedy_all_accept(); };
    "spec_greedy_first_reject_and_laziness"_test = [] { test_greedy_first_reject_and_laziness(); };
    "spec_greedy_partial_reject"_test = [] { test_greedy_partial_reject(); };
    "spec_greedy_tie_lowest_index"_test = [] { test_greedy_tie_lowest_index(); };
    "spec_greedy_plus_inf"_test = [] { test_greedy_plus_inf(); };
    "spec_greedy_nan_and_bonus_row"_test = [] { test_greedy_nan_and_bonus_row(); };
    "spec_greedy_ignores_draft_logits_and_strides"_test =
        [] { test_greedy_ignores_draft_logits_and_strides(); };
    "spec_greedy_id_range"_test = [] { test_greedy_id_range(); };
    "spec_negative_zero_temperature_is_greedy"_test =
        [] { test_negative_zero_temperature_is_greedy(); };
    "spec_softmax_probability_basic"_test = [] { test_softmax_probability_basic(); };
    "spec_softmax_probability_pos_inf"_test = [] { test_softmax_probability_pos_inf(); };
    "spec_softmax_probability_failures"_test = [] { test_softmax_probability_failures(); };
    "spec_categorical_sample_basic"_test = [] { test_categorical_sample_basic(); };
    "spec_categorical_sample_pos_inf"_test = [] { test_categorical_sample_pos_inf(); };
    "spec_categorical_sample_failures"_test = [] { test_categorical_sample_failures(); };
    "spec_residual_sample_basic"_test = [] { test_residual_sample_basic(); };
    "spec_residual_sample_pos_inf_and_failures"_test =
        [] { test_residual_sample_pos_inf_and_failures(); };
    "spec_sampling_all_accept_bonus_from_extra_row"_test =
        [] { test_sampling_all_accept_bonus_from_extra_row(); };
    "spec_sampling_first_reject_residual_bonus"_test =
        [] { test_sampling_first_reject_residual_bonus(); };
    "spec_sampling_partial_accept"_test = [] { test_sampling_partial_accept(); };
    "spec_sampling_reproducible_and_seed_round_sensitive"_test =
        [] { test_sampling_reproducible_and_seed_round_sensitive(); };
    "spec_sampling_draw_tag_and_index_separation"_test =
        [] { test_sampling_draw_tag_and_index_separation(); };
    "spec_sampling_temperature_scales_accept_ratio"_test =
        [] { test_sampling_temperature_scales_accept_ratio(); };
    "spec_engine_input_validation"_test = [] { test_engine_input_validation(); };
    "spec_engine_nan_and_zero_mass_rows_sampling"_test =
        [] { test_engine_nan_and_zero_mass_rows_sampling(); };
    "spec_eos_agnostic_and_counts"_test = [] { test_eos_agnostic_and_counts(); };
    "spec_strided_sampling_rows"_test = [] { test_strided_sampling_rows(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}
