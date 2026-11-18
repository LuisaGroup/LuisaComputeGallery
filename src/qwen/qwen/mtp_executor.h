#pragma once
// Qwen3.5-4B MTP (multi-token prediction / NextN) speculative-decoding
// TARGET executor. QwenMtpExecutor couples:
//
//   * a TARGET  -- the full 32-layer QwenDslEngine trunk (owned here, created
//                  with forced DSL + recurrent-GDN options so verify/replay
//                  use one numerical path), exposing its
//                  private reset/stage_ids/forward_chunk/lm_head/project
//                  members to this class through the QwenDslEngine friend
//                  declaration (no RTTI, no engine edits);
//   * a DRAFT   -- the one-layer Qwen3.5 NextN draft block (qwen/nextn_draft.h),
//                  driven by propose() over the target's hidden history;
//   * the acceptance engine (qwen/speculative.h) which consumes the already
//                  downloaded verify logits and decides accepted/bonus.
//
// Session contract (single sequence, mirrors QwenEngine):
//   prefill(prompt, logits) resets the target, consumes the whole prompt,
//   downloads next-token logits and records EVERY prompt row's output hidden
//   state into the hidden history. The host `_consumed` mirror equals the
//   prompt and target position == prompt.size().
//   round(pending, sample, out) runs ONE draft+verify speculation round.
//   `pending` is the token selected from the previous round's/logits
//   distribution; it is NOT yet emitted or consumed. On success:
//     out.emitted     = [pending, d_1..d_a]  (a+1 tokens; a == out.accepted)
//     out.next_pending = the bonus token (NOT emitted; the caller's next
//                       round input, or the final token after the last round)
//     out.next_logits  = the RAW target verify distribution after the emitted
//                       prefix (row a of k+1 rows). A sampling-rejection bonus
//                       uses the residual instead; never sample the bonus twice.
//   The target logically consumes exactly the emitted prefix; on partial
//   acceptance the polluted GDN state/conv rings are restored from D2D
//   snapshots and only the emitted prefix is replayed (the bonus row is NEVER
//   replayed). Full-attention KV rows beyond the logical position are never
//   read again, so they are left untouched. After the round,
//   target position == pos0 + a + 1 == _consumed.size().
//
//   Draw identity: the executor owns the speculation-round counter (starts at
//   0 after every prefill, increments per COMPLETED round) and passes it to
//   both the draft sampler and qwen_speculative_accept, so draft/accept draws
//   are reproducible and round-unique within a session.
//
// Failure contract (fail closed, recoverable Status, no exceptions/RTTI):
//   * argument/state validation failures leave everything untouched;
//   * a failure AFTER the target verify forward started (or inside the
//     acceptance computation) poisons the session: target _has_session is
//     cleared so no further round/decode can run on inconsistent state --
//     the caller must prefill again;
//   * draft-side failures (propose) happen before any target mutation and
//     leave the target session usable.
//
// Staging/lifetime: every public call records into QwenCommandBatch(es) and
// only downloads host data after a synchronizing flush; host staging vectors
// are members that outlive every queued command (the engine's _host_ids span
// rule is honored: `_staged` is reset per batch so each batch gets distinct
// spans). Annotations are string literals / engine-owned strings only.
//
// Benchmark: the target engine's own options.benchmark mode is preserved;
// verify/replay/restore commands are recorded with QwenBenchmarkPhase::mtp
// into the target's report (prefill keeps the engine's own prefill phase).

#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "qwen/benchmark.h"
#include "qwen/common.h"
#include "qwen/dsl_engine.h"
#include "qwen/engine.h"
#include "qwen/nextn_draft.h"
#include "qwen/speculative.h"

namespace luisa::qwen {

// Result of one successful QwenMtpExecutor::round() (see the file contract).
struct QwenMtpRoundResult {
    // Emitted prefix: pending token followed by the accepted drafts
    // [pending, d_1, ..., d_accepted]. The bonus is NOT emitted.
    luisa::vector<uint32_t> emitted;
    // The bonus token: the next round's pending input (or the final sampled
    // token once the caller stops speculating).
    uint32_t next_pending{0};
    // Raw target distribution [vocab] after the emitted prefix. Greedy and
    // all-accept bonuses come from this row. A sampling REJECTION bonus instead
    // comes from the residual max(target-draft,0), so callers must never redraw
    // that already-selected bonus from next_logits; it is diagnostic/context
    // state for the next round, where next_pending is verified first.
    luisa::vector<float> next_logits;
    // Number of accepted draft tokens (a in [0, k]); emitted.size() == a + 1.
    uint32_t accepted{0};
};

// Aggregate for CLI display (target + draft + executor statistics).
struct QwenMtpExecutorInfo {
    QwenEngineInfo target{};
    QwenNextnDraftInfo draft{};
    uint32_t draft_depth{0};
    uint64_t rounds{0};   // completed speculation rounds
    uint64_t accepted{0}; // accepted draft tokens (sum over rounds)
    uint64_t drafted{0};  // proposed draft tokens (sum over rounds)
    uint64_t history_bytes{0};
    uint64_t snapshot_bytes{0};// GDN state+ring D2D snapshots
    uint64_t verify_logits_bytes{0};
};

class QwenMtpExecutor {
public:
    // Allocator-matched deleter (same pattern as QwenEngineDeleter /
    // QwenNextnDraft::Deleter): the object is allocated with Luisa's
    // allocator; luisa::unique_ptr's default eastl::default_delete would free
    // through a different heap.
    struct Deleter {
        void operator()(QwenMtpExecutor *p) const noexcept {
            if (p != nullptr) {
                std::destroy_at(p);
                luisa::deallocate_with_allocator(p);
            }
        }
    };
    using Ptr = luisa::unique_ptr<QwenMtpExecutor, Deleter>;

    // Create the target+draft pair and the executor-owned device state.
    // `options.kernel` is FORCED to QwenKernelKind::dsl and
    // options.prefill_chunk must be >= draft_depth + 1 (the verify chunk has
    // draft_depth+1 rows and must fit the target activation workspace);
    // draft_depth must be >= 1; max_context must be in
    // [1, model context_length]. The trunk embedding
    // (trunk_weights.find(QwenTensorRole::token_embd)) is passed to the draft
    // for tied embeddings/LM head. `trunk_weights` and `draft_weights` are
    // BORROWED and must outlive the executor. The draft uses the same
    // max_context and prefill chunk as the target. Synchronizes nothing; the
    // created executor starts with a clean target session (reset at create).
    [[nodiscard]] static Result<Ptr> create(compute::Device &device, compute::Stream &stream,
                                            const Qwen35Model &model,
                                            const QwenDeviceWeights &trunk_weights,
                                            const QwenDeviceWeights &draft_weights,
                                            const QwenEngineOptions &options,
                                            uint32_t draft_depth);

    // Consume a nonempty prompt exactly like QwenDslEngine::prefill (reset +
    // chunked forward + last-position LM head + logits download), recording
    // every chunk's output hidden rows into the history. Resets the
    // round counter and the per-session speculation statistics, and mirrors
    // the prompt into the host consumed-token vector. Synchronizes before
    // returning.
    [[nodiscard]] Status prefill(compute::Stream &stream,
                                 luisa::span<const uint32_t> prompt_tokens,
                                 luisa::vector<float> &logits_out);

    // Run one draft+verify speculation round (see the file contract).
    // `pending_token` must be < vocab and is NOT yet emitted or consumed;
    // sample/round-counter drive both draft and acceptance draws.
    // On success the stream is synchronized and `out` fully populated.
    // depth is clamped so pos0 + 1 + depth <= max_context; a clamp to 0 is an
    // invalid_argument error (the caller should use plain target decode).
    [[nodiscard]] Status round(compute::Stream &stream, uint32_t pending_token,
                               const QwenSampleOptions &sample, QwenMtpRoundResult &out);

    // Verify an externally proposed sequence through the identical snapshot,
    // target-forward, acceptance and rollback path used by round(). Greedy mode
    // ignores draft_logits; sampling requires one row per drafted id. This is
    // also the deterministic seam for partial-accept and alternative-draft
    // tests. The proposal must contain 1..configured_depth ids and fit context.
    [[nodiscard]] Status verify(compute::Stream &stream, uint32_t pending_token,
                                luisa::span<const uint32_t> drafted,
                                const QwenSpeculativeLogits &draft_logits,
                                const QwenSampleOptions &sample, QwenMtpRoundResult &out);

    // Borrowed target engine (owned by this executor). Nonnull after create.
    [[nodiscard]] QwenEngine *target() noexcept { return _target.get(); }
    [[nodiscard]] const QwenEngine *target() const noexcept { return _target.get(); }

    [[nodiscard]] QwenMtpExecutorInfo info() const noexcept;
    // The target's own report; nullptr exactly when options.benchmark was
    // false. Valid until the next executor call that records commands.
    [[nodiscard]] const QwenBenchmarkReport *benchmark_report() const noexcept;

    // Session helpers for CLI driving.
    [[nodiscard]] uint32_t position() const noexcept;
    [[nodiscard]] uint32_t max_context() const noexcept;
    // max_context - position (0 when no session/target).
    [[nodiscard]] uint32_t remaining_capacity() const noexcept;
    // True when a round() could run: live session and room for at least one
    // draft token after the pending slot (position + 2 <= max_context).
    [[nodiscard]] bool can_speculate() const noexcept;
    [[nodiscard]] uint64_t rounds() const noexcept { return _rounds; }
    [[nodiscard]] uint64_t accepted_total() const noexcept { return _accepted; }
    [[nodiscard]] uint64_t drafted_total() const noexcept { return _drafted; }
    [[nodiscard]] double accept_rate() const noexcept {
        return _drafted == 0u ? 0.0 : static_cast<double>(_accepted) / static_cast<double>(_drafted);
    }

private:
    QwenMtpExecutor() noexcept = default;

    // The owned target as its concrete type (QwenDslEngine::create built it;
    // static downcast, no RTTI). Valid whenever _target is nonnull.
    [[nodiscard]] QwenDslEngine *dsl_target() noexcept {
        return static_cast<QwenDslEngine *>(_target.get());
    }
    [[nodiscard]] const QwenDslEngine *dsl_target() const noexcept {
        return static_cast<const QwenDslEngine *>(_target.get());
    }

    // ---- configuration ------------------------------------------------------
    uint32_t _draft_depth{0};

    // ---- owned children -----------------------------------------------------
    // Retained Device declared BEFORE every GPU child so reverse destruction
    // releases all buffers/shaders first; retaining the handle also makes
    // device-first caller teardown safe (same contract as the engines).
    compute::Device _device;
    // The target trunk (created with forced kernel=dsl options).
    luisa::unique_ptr<QwenEngine, QwenEngineDeleter> _target;
    QwenNextnDraft::Ptr _draft;

    // ---- persistent device state --------------------------------------------
    // Target hidden history: row i = trunk output hidden after consuming
    // consumed token i ([max_context, hidden]).
    compute::Buffer<float> _history;
    // D2D snapshots of every target GDN recurrent state / conv ring, taken
    // before each verify forward and restored before a partial-accept replay.
    luisa::vector<compute::Buffer<float>> _gdn_state_snaps;
    luisa::vector<compute::Buffer<float>> _gdn_ring_snaps;
    // Verify LM logits: [draft_depth+1, vocab] (one row per verify row).
    compute::Buffer<float> _verify_logits;

    // ---- persistent host staging (outlives every queued command) ------------
    // Downloaded verify logits of the CURRENT round; the QwenSpeculativeLogits
    // target view aliases this storage.
    luisa::vector<float> _host_verify;
    // Verify/replay id staging ([pending, d_1..d_k]); stage_ids copies it into
    // the engine staging span at record time, but keeping it in a member makes
    // the distinct-span rule explicit across the two batches of one round.
    luisa::vector<uint32_t> _verify_ids;
    // Host mirror of every target-consumed token id (prompt + emitted).
    luisa::vector<uint32_t> _consumed;

    // ---- session/statistics -------------------------------------------------
    uint64_t _round{0};    // speculation-round draw counter (per prefill)
    uint64_t _rounds{0};   // completed rounds (per prefill)
    uint64_t _accepted{0}; // accepted draft tokens (per prefill)
    uint64_t _drafted{0};  // proposed draft tokens (per prefill)
    uint64_t _snapshot_bytes{0};
};

}// namespace luisa::qwen
