// Qwen3.5-4B MTP speculative-decoding target executor -- implementation.
// See mtp_executor.h for the session/failure contract. This file only drives
// the QwenDslEngine friend API, the QwenNextnDraft and the pure-host
// acceptance engine; it defines no kernels of its own.
#include "qwen/mtp_executor.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace luisa::qwen {

using compute::Buffer;
using compute::Device;
using compute::Stream;
namespace d = luisa::qwen::dsl;

namespace {

[[nodiscard]] Status err(StatusCode code, luisa::string msg, luisa::string detail = {}) {
    return Status::error(code, std::move(msg), std::move(detail));
}
[[nodiscard]] Status invalid(luisa::string msg, luisa::string detail = {}) {
    return err(StatusCode::invalid_argument, std::move(msg), std::move(detail));
}

}// namespace

// ---------------------------------------------------------------------------
// create
// ---------------------------------------------------------------------------
[[nodiscard]] Result<QwenMtpExecutor::Ptr> QwenMtpExecutor::create(
    Device &device, Stream &stream, const Qwen35Model &model,
    const QwenDeviceWeights &trunk_weights, const QwenDeviceWeights &draft_weights,
    const QwenEngineOptions &options, uint32_t draft_depth) {
    using R = Result<QwenMtpExecutor::Ptr>;
    if (!device) { return R::failure(invalid("invalid device")); }
    if (draft_depth == 0u) {
        return R::failure(invalid("draft_depth must be >= 1 (plain decode needs no executor)"));
    }
    const auto &g = model.config().geometry;
    if (options.max_context == 0u || options.max_context > g.context_length) {
        return R::failure(invalid("max_context must be in [1, model context_length]"));
    }
    if (options.prefill_chunk == 0u) {
        return R::failure(invalid("prefill_chunk must be >= 1"));
    }
    // The verify chunk holds draft_depth+1 rows and must fit the target
    // activation workspace in ONE forward_chunk call.
    if (options.prefill_chunk < draft_depth + 1u) {
        return R::failure(invalid("prefill_chunk must be >= draft_depth + 1 "
                                  "(the verify chunk rows must fit one forward pass)"));
    }
    const auto *embd = trunk_weights.find(QwenTensorRole::token_embd);
    if (embd == nullptr) {
        return R::failure(invalid("trunk token_embd weight missing (tied head/embedding required)"));
    }

    // The target always runs the DSL executor; options.kernel is forced.
    QwenEngineOptions forced = options;
    forced.kernel = QwenKernelKind::dsl;
    // Verify and a possible one-row accepted-prefix replay must execute the
    // SAME GDN numerical path. In particular, a zero-accept replay has rows=1
    // and would otherwise switch from multi-row WY/UT to the recurrence while
    // next_logits came from the WY verify row. Keep MTP on the exact recurrent
    // path until per-row chunk-state restore is implemented.
    forced.dsl_gdn_wy_ut = false;
    forced.dsl_gdn_auto = false;// keep the per-prompt rule from re-selecting WY/UT
    // Single-row decode specializations use different reduction orders from
    // multi-row verify. Keep target/replay/plain-reference numerics aligned.
    forced.dsl_decode_specialized = false;
    // MTP verify feeds the same attention kernel several accepted-prefix
    // reconstructions; keep them on the reference prefill attention so a verify
    // row and its replay differ only by the GDN path, not by softmax
    // reassociation. The ordinary (non-MTP) engine keeps the tiled kernel.
    forced.dsl_tiled_prefill_attn = false;
 // The same reason applies to the CUDA tensor-core prefill attention (opt_plan/09
 // N1b): it is a different numerical realization (BF16 operands and P weights) of
 // the same math, so verify and replay must not disagree by softmax precision.
 forced.native_tensor_attention = false;
    forced.dsl_cuda_graph = false;
    // The executor splits the prompt with the fixed chunk above and copies each
    // chunk's hidden rows into its own history buffer, so it keeps the exact
    // fixed-chunk schedule instead of the adaptive rule (which is a plain
    // single-sequence prefill feature).
    forced.prefill_chunk_adaptive = false;
    auto target = QwenDslEngine::create(device, stream, model, trunk_weights, forced);
    if (!target.ok()) { return R::failure(target.status); }

    // Allocate with Luisa's allocator; Deleter frees with the same allocator.
    auto *raw = ::new (luisa::allocate_with_allocator<QwenMtpExecutor>()) QwenMtpExecutor{};
    auto exec = QwenMtpExecutor::Ptr{raw};
    auto &e = *exec;
    e._device = device;
    e._draft_depth = draft_depth;
    e._target = std::move(*target.value);

    // Draft borrows the trunk embedding (tied) and the blk.<layers>.* set.
    auto draft = QwenNextnDraft::create(device, stream, model, *embd, draft_weights,
                                        forced.max_context, forced.prefill_chunk,
      forced.kv_page_bytes, forced.kv_dtype);
    if (!draft.ok()) { return R::failure(draft.status); }
    e._draft = std::move(*draft.value);

    auto *t = e.dsl_target();
    const uint64_t hidden = t->_cfg.hidden;
    const uint64_t vocab = t->_cfg.vocab;
    // Executor-owned buffers stay inside the uint32 kernel index envelope.
    uint64_t history_elems = 0u;
    if (!checked_mul(static_cast<uint64_t>(forced.max_context), hidden, history_elems) ||
        history_elems > 0xFFFFFFFFull) {
        return R::failure(invalid("hidden history exceeds the uint32 envelope"));
    }
    uint64_t verify_elems = 0u;
    if (!checked_mul(static_cast<uint64_t>(draft_depth) + 1u, vocab, verify_elems) ||
        verify_elems > 0xFFFFFFFFull) {
        return R::failure(invalid("verify logits buffer exceeds the uint32 envelope"));
    }
    e._history = device.create_buffer<float>(static_cast<size_t>(history_elems));
    if (!e._history) { return R::failure(err(StatusCode::out_of_memory, "history allocation failed")); }
    // One D2D snapshot per target GDN recurrent state / conv ring.
    const auto snap = [&](const luisa::vector<Buffer<float>> &src,
                          luisa::vector<Buffer<float>> &dst, const char *what) -> Status {
        dst.reserve(src.size());
        for (const auto &b : src) {
            auto s = device.create_buffer<float>(static_cast<size_t>(b.size()));
            if (!s) { return err(StatusCode::out_of_memory, luisa::string{what} + " allocation failed"); }
            e._snapshot_bytes += b.size() * 4u;
            dst.push_back(std::move(s));
        }
        return Status::success();
    };
    if (auto s = snap(t->_gdn_states, e._gdn_state_snaps, "gdn state snapshot"); !s.ok()) { return R::failure(s); }
    if (auto s = snap(t->_gdn_rings, e._gdn_ring_snaps, "gdn ring snapshot"); !s.ok()) { return R::failure(s); }
    e._verify_logits = device.create_buffer<float>(static_cast<size_t>(verify_elems));
    if (!e._verify_logits) {
        return R::failure(err(StatusCode::out_of_memory, "verify logits allocation failed"));
    }
    return R::success(std::move(exec));
}

// ---------------------------------------------------------------------------
// prefill: identical chunk schedule to QwenDslEngine::prefill, plus the
// per-chunk hidden-history D2D copies.
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenMtpExecutor::prefill(Stream &stream,
                                              luisa::span<const uint32_t> prompt_tokens,
                                              luisa::vector<float> &logits_out) {
    auto *t = dsl_target();
    if (t == nullptr) { return err(StatusCode::internal_error, "executor has no target"); }
    if (prompt_tokens.empty()) { return invalid("prefill prompt must be nonempty"); }
    if (prompt_tokens.size() > t->_max_context) { return invalid("prompt exceeds max_context"); }
    if (auto s = d::validate_qwen_embedding_inputs(prompt_tokens, t->_cfg.hidden, t->_cfg.vocab); !s.ok()) { return s; }
    const uint32_t total = static_cast<uint32_t>(prompt_tokens.size());
    const uint32_t hidden = t->_cfg.hidden;
    // Fresh target report for this public call (reset appends its own
    // zero-fill samples when benchmark is on, exactly like the engine).
    if (t->_benchmark) { t->_report.clear(); }
    if (auto s = t->reset(stream); !s.ok()) {
        t->_has_session = false;
        return s;
    }
    // ONE command batch for all chunks; each chunk stages a distinct span.
    t->_staged = 0;
    auto batch = QwenCommandBatch{stream, t->_benchmark ? &t->_report : nullptr,
                                  QwenBenchmarkPhase::prefill, 0u, total};
    uint32_t done = 0;
    while (done < total) {
        const uint32_t rows = std::min(t->_prefill_chunk, total - done);
        batch.set_context(QwenBenchmarkPhase::prefill, done, rows);
        if (auto s = t->stage_ids(batch, prompt_tokens.subspan(done, rows)); !s.ok()) {
            t->_has_session = false;
            return s;
        }
        if (auto s = t->forward_chunk(stream, batch, rows, done); !s.ok()) {
            t->_has_session = false;// a partial forward may have advanced device state
            return s;
        }
        // Record the chunk's output hidden rows into the history. Execution
        // order within the batch makes this read the just-written _hidden.
        batch.annotate("mtp.history");
        batch << _history.view(static_cast<size_t>(done) * hidden, static_cast<size_t>(rows) * hidden)
                     .copy_from(t->_hidden.view(0u, static_cast<size_t>(rows) * hidden));
        if (done + rows == total) {
            if (auto s = t->lm_head(batch, rows); !s.ok()) {
                t->_has_session = false;
                return s;
            }
        }
        done += rows;
    }
    batch.flush(false);
    logits_out.resize(t->_cfg.vocab);
    batch.layer(UINT32_MAX)
        .set_context(QwenBenchmarkPhase::prefill, total - 1u, 1u)
        .annotate("logits_download");
    batch << t->_logits.view().copy_to(luisa::span<float>{logits_out});
    batch.flush(true);// synchronizes: logits are host-readable
    t->_position = total;
    t->_has_session = true;
    _consumed.assign(prompt_tokens.begin(), prompt_tokens.end());
    _round = 0u;
    _rounds = 0u;
    _accepted = 0u;
    _drafted = 0u;
    return Status::success();
}

// ---------------------------------------------------------------------------
// round: NextN proposal, then the public verify path below.
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenMtpExecutor::round(Stream &stream, uint32_t pending_token,
                                            const QwenSampleOptions &sample,
                                            QwenMtpRoundResult &out) {
    auto *t = dsl_target();
    if (t == nullptr) { return err(StatusCode::internal_error, "executor has no target"); }
    if (_draft == nullptr) { return err(StatusCode::internal_error, "executor has no draft"); }
    if (!t->_has_session) { return invalid("round requires a prior nonempty prefill"); }
    const auto pos0 = t->_position;
    if (_consumed.size() != static_cast<size_t>(pos0) || pending_token >= t->_cfg.vocab) {
        return invalid("round pending token/history is inconsistent with the target");
    }
    if (pos0 + 1u >= t->_max_context) { return invalid("no room for speculation (use plain decode)"); }
    const auto room = t->_max_context - pos0 - 1u;
    const auto depth = std::min(_draft_depth, room);
    if (depth == 0u) { return invalid("draft depth clamped to zero (use plain decode)"); }
    luisa::vector<uint32_t> drafted;
    QwenSpeculativeLogits draft_logits{};
    if (auto s = _draft->propose(stream, _history.view(0u, static_cast<size_t>(pos0) * t->_cfg.hidden),
                                 luisa::span<const uint32_t>{_consumed.data(), _consumed.size()},
                                 pending_token, depth, sample, _round,
                                 drafted, draft_logits); !s.ok()) {
        return s;
    }
    if (drafted.size() != depth) {
        return err(StatusCode::internal_error, "draft returned an unexpected token count");
    }
    return verify(stream, pending_token, luisa::span<const uint32_t>{drafted},
                  draft_logits, sample, out);
}

// ---------------------------------------------------------------------------
// verify: snapshot -> verify (k+1 rows) -> accept -> maybe replay. Exposed so
// deterministic tests and alternative proposers can supply exact draft rows.
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenMtpExecutor::verify(
    Stream &stream, uint32_t pending_token, luisa::span<const uint32_t> drafted,
    const QwenSpeculativeLogits &draft_logits, const QwenSampleOptions &sample,
    QwenMtpRoundResult &out) {
    out = QwenMtpRoundResult{};
    auto *t = dsl_target();
    if (t == nullptr) { return err(StatusCode::internal_error, "executor has no target"); }
    if (!_history || !_verify_logits) {
        return err(StatusCode::internal_error, "executor state is not allocated");
    }
    if (_gdn_state_snaps.size() != t->_gdn_states.size() ||
        _gdn_ring_snaps.size() != t->_gdn_rings.size()) {
        return err(StatusCode::internal_error, "GDN snapshot count does not match the target");
    }
    if (!t->_has_session) { return invalid("round requires a prior nonempty prefill"); }
    const uint32_t pos0 = t->_position;
    const uint32_t max_context = t->_max_context;
    const uint32_t vocab = t->_cfg.vocab;
    const uint32_t hidden = t->_cfg.hidden;
    // The pending token is not consumed yet: the host mirror must be exactly
    // the consumed prefix.
    if (_consumed.size() != static_cast<size_t>(pos0)) {
        return invalid("host consumed-token mirror is out of sync with the target position");
    }
    if (pending_token >= vocab) { return invalid("pending token id out of vocabulary range"); }
    if (pos0 + 1u >= max_context) {
        return invalid("no room for a speculative round at the end of the context "
                       "(use plain decode)");
    }
    const uint32_t room = max_context - pos0 - 1u;
    if (drafted.empty() || drafted.size() > _draft_depth || drafted.size() > room) {
        return invalid("verify draft count must be in [1, min(configured depth, remaining context)]");
    }
    const uint32_t k = static_cast<uint32_t>(drafted.size());

    // ---- 1. verify ids [pending, d_1..d_k] --------------------------------
    _verify_ids.clear();
    _verify_ids.reserve(k + 1u);
    _verify_ids.push_back(pending_token);
    for (const uint32_t d : drafted) { _verify_ids.push_back(d); }

    // ---- 3. snapshot every GDN state + ring, then verify forward ----------
    // Snapshot copies are recorded FIRST in this batch, so they observe the
    // pre-verify state regardless of submission mode.
    auto batch = QwenCommandBatch{stream, t->_benchmark ? &t->_report : nullptr,
                                  QwenBenchmarkPhase::mtp, pos0, k + 1u};
    batch.annotate("mtp.snapshot");
    for (size_t i = 0; i < t->_gdn_states.size(); i++) {
        batch << _gdn_state_snaps[i].view().copy_from(t->_gdn_states[i].view());
    }
    for (size_t i = 0; i < t->_gdn_rings.size(); i++) {
        batch << _gdn_ring_snaps[i].view().copy_from(t->_gdn_rings[i].view());
    }
    t->_staged = 0;// distinct spans for this batch's id staging
    if (auto s = t->stage_ids(batch, luisa::span<const uint32_t>{_verify_ids.data(), _verify_ids.size()}); !s.ok()) {
        t->_has_session = false;
        return s;
    }
    if (auto s = t->forward_chunk(stream, batch, k + 1u, pos0); !s.ok()) {
        t->_has_session = false;// a partial forward may have advanced device state
        return s;
    }
    // History rows for the whole verify chunk (the replay overwrites the
    // rejected suffix with identical recomputed values when a < k).
    batch.annotate("mtp.history");
    batch << _history.view(static_cast<size_t>(pos0) * hidden, static_cast<size_t>(k + 1u) * hidden)
                 .copy_from(t->_hidden.view(0u, static_cast<size_t>(k + 1u) * hidden));
    // LM logits for EVERY verify row: output_norm on all rows, then the tied
    // token_embd projection into the [(depth+1), vocab] verify buffer.
    batch.annotate("mtp.output_norm");
    {
        auto plan = d::plan_qwen35_rmsnorm_dispatch(k + 1u, hidden, 256u);
        if (!plan.ok()) { t->_has_session = false; return plan.status; }
        batch << t->_rmsnorm_plain(t->_hidden.view(), t->_output_norm->f32.view(), t->_dummy_f.view(),
                                   t->_normed.view(), t->_dummy_f.view(), k + 1u, hidden, t->_cfg.norm_eps)
                     .dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    if (auto s = t->project(batch, t->_normed, *t->_token_embd, _verify_logits, k + 1u,
                            nullptr, "mtp.lm_head"); !s.ok()) {
        t->_has_session = false;
        return s;
    }
    batch.flush(false);
    _host_verify.resize(static_cast<size_t>(k + 1u) * vocab);
    batch.layer(UINT32_MAX)
        .set_context(QwenBenchmarkPhase::mtp, pos0 + k, 1u)
        .annotate("mtp.verify_logits");
    batch << _verify_logits.view(0u, static_cast<size_t>(k + 1u) * vocab)
                 .copy_to(luisa::span<float>{_host_verify});
    batch.flush(true);// synchronizes: verify logits are host-readable

    // ---- 4. host acceptance over the downloaded rows ----------------------
    QwenSpeculativeInput accept_in;
    accept_in.draft_tokens = luisa::span<const uint32_t>{drafted.data(), drafted.size()};
    accept_in.draft_logits = draft_logits;
    accept_in.target_logits = QwenSpeculativeLogits{_host_verify.data(), k + 1u, vocab, vocab};
    accept_in.options = sample;
    accept_in.round = _round;
    QwenSpeculativeResult acc{};
    if (auto s = qwen_speculative_accept(accept_in, acc); !s.ok()) {
        // The verify chunk already advanced the device state; the session can
        // no longer be trusted (logical position was never updated).
        t->_has_session = false;
        return s;
    }
    const uint32_t a = acc.accepted;
    if (a > k) {
        t->_has_session = false;
        return err(StatusCode::internal_error, "acceptance engine returned accepted > drafted");
    }

    // ---- 5. emitted prefix + bonus semantics ------------------------------
    out.accepted = a;
    out.emitted.reserve(a + 1u);
    out.emitted.push_back(pending_token);
    for (uint32_t j = 0; j < a; j++) { out.emitted.push_back(drafted[j]); }
    out.next_pending = acc.bonus;
    out.next_logits.assign(_host_verify.data() + static_cast<size_t>(a) * vocab,
                           _host_verify.data() + static_cast<size_t>(a + 1u) * vocab);

    // ---- 6. partial accept: restore GDN snapshots, replay ONLY the --------
    // emitted prefix (the bonus row is never replayed/consumed). -------------
    if (a < k) {
        auto replay = QwenCommandBatch{stream, t->_benchmark ? &t->_report : nullptr,
                                       QwenBenchmarkPhase::mtp, pos0, a + 1u};
        replay.annotate("mtp.restore");
        for (size_t i = 0; i < t->_gdn_states.size(); i++) {
            replay << t->_gdn_states[i].view().copy_from(_gdn_state_snaps[i].view());
        }
        for (size_t i = 0; i < t->_gdn_rings.size(); i++) {
            replay << t->_gdn_rings[i].view().copy_from(_gdn_ring_snaps[i].view());
        }
        // The verify batch was fully submitted+synced above, so reusing the
        // first spans of the engine staging buffer is safe.
        t->_staged = 0;
        if (auto s = t->stage_ids(replay, luisa::span<const uint32_t>{_verify_ids.data(), a + 1u}); !s.ok()) {
            t->_has_session = false;
            return s;
        }
        if (auto s = t->forward_chunk(stream, replay, a + 1u, pos0); !s.ok()) {
            t->_has_session = false;
            return s;
        }
        replay.annotate("mtp.history");
        replay << _history.view(static_cast<size_t>(pos0) * hidden, static_cast<size_t>(a + 1u) * hidden)
                      .copy_from(t->_hidden.view(0u, static_cast<size_t>(a + 1u) * hidden));
        replay.flush(false);
        // No host data depends on the replay, but the public contract
        // synchronizes before returning.
        stream.synchronize();
    }

    // ---- 7. logical position / consumed mirror / statistics ---------------
    t->_position = pos0 + a + 1u;
    t->_has_session = true;
    _consumed.insert(_consumed.end(), out.emitted.begin(), out.emitted.end());
    _rounds += 1u;
    _accepted += a;
    _drafted += k;
    _round += 1u;// next round's draw identity
    return Status::success();
}

// ---------------------------------------------------------------------------
// accessors
// ---------------------------------------------------------------------------
[[nodiscard]] QwenMtpExecutorInfo QwenMtpExecutor::info() const noexcept {
    QwenMtpExecutorInfo i;
    if (_target != nullptr) { i.target = _target->info(); }
    if (_draft != nullptr) { i.draft = _draft->info(); }
    i.draft_depth = _draft_depth;
    i.rounds = _rounds;
    i.accepted = _accepted;
    i.drafted = _drafted;
    i.history_bytes = _history ? _history.size() * 4u : 0u;
    i.snapshot_bytes = _snapshot_bytes;
    i.verify_logits_bytes = _verify_logits ? _verify_logits.size() * 4u : 0u;
    return i;
}

[[nodiscard]] const QwenBenchmarkReport *QwenMtpExecutor::benchmark_report() const noexcept {
    return _target != nullptr ? _target->benchmark_report() : nullptr;
}

[[nodiscard]] uint32_t QwenMtpExecutor::position() const noexcept {
    const auto *t = dsl_target();
    return t != nullptr ? t->_position : 0u;
}

[[nodiscard]] uint32_t QwenMtpExecutor::max_context() const noexcept {
    const auto *t = dsl_target();
    return t != nullptr ? t->_max_context : 0u;
}

[[nodiscard]] uint32_t QwenMtpExecutor::remaining_capacity() const noexcept {
    const auto *t = dsl_target();
    if (t == nullptr || t->_position >= t->_max_context) { return 0u; }
    return t->_max_context - t->_position;
}

[[nodiscard]] bool QwenMtpExecutor::can_speculate() const noexcept {
    const auto *t = dsl_target();
    // A round needs the pending slot plus at least one draft row.
    return t != nullptr && t->_has_session && t->_position + 2u <= t->_max_context;
}

}// namespace luisa::qwen
