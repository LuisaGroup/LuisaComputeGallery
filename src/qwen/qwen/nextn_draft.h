#pragma once
// Qwen3.5-4B NextN / MTP draft block -- ordinary-DSL implementation.
//
// QwenNextnDraft drives the ONE-layer Qwen3.5 NextN draft block (canonical
// converter name blk.32.*, see qwen/mtp.h) through the shared kernel
// factories in qwen/dsl_ops.h. NO Tile, NO native shaders, no fallback: every
// device kernel is an ordinary DSL kernel compiled once at create(). It is
// driven by a separate target engine that owns the 32-layer trunk and the
// acceptance engine (qwen/speculative.h consumes the logits this produces).
//
// Block semantics (frozen Qwen3.5-4B geometry; matches vLLM
// qwen3_5_mtp.py / the HF modular MTP layer):
//
//   Each draft row takes a target/draft feedback hidden H and one SHIFTED
//   input token id:
//     emb      = embedding(id)                     (tied trunk embedding, or
//                                                    the dedicated blk.32.token_embd)
//     fused    = eh_proj(concat(rmsnorm(emb, enorm), rmsnorm(H, hnorm)))
//     block    = ONE standard full-attention decoder block (blk.32 attn_norm,
//                q/k/v/output + q/k norms, partial NeoX RoPE, causal GQA over
//                the draft's OWN K/V cache, post_attention_norm, FFN gate/up,
//                SwiGLU, down; residual stream inside the block)
//     feedback = rmsnorm(block_residual_output, blk.32.shared_head.norm)
//     logits   = feedback @ head^T                 (dedicated blk.32.output, or
//                                                    the tied trunk embedding)
//   The RoPE position of a draft row is the position of the TARGET hidden it
//   consumes (positions are NOT shifted; the MTP layer shares the target's
//   position stream).
//
// Draft round (propose()), correctness-first form:
//   Let P = target_tokens.size() be the number of trunk-consumed tokens and
//   H_hist[P, hidden] the trunk hidden history (H_hist[i] = trunk output
//   hidden after consuming target_tokens[i]). The draft never trusts its
//   state across rounds: every propose() RESETS the draft K/V cache and
//   recomputes the draft prefill over the ENTIRE history:
//     row i in [0, P):   id_i = target_tokens[i+1] for i < P-1, else
//                        id_{P-1} = pending_token   (shifted ids)
//                        H_i = H_hist[i], position i
//   The last history row is draft step 1 at position P-1 (H[P-1] plus the
//   pending token), and emits d1. Later steps continue at P, P+1, ...:
//     step j >= 2: H = feedback(step j-1), id = drafted token d_{j-1}
//   Every step emits one feedback + one [vocab] logits row; the row is
//   downloaded and the next draft token is sampled on the HOST:
//     temperature == 0: qwen_argmax (lowest-index ties, NaN rejection)
//     temperature  > 0: qwen_speculative_uniform(seed, round, index=j-1,
//                        tag=kQwenNextnDraftDrawTag) + qwen_categorical_sample
//   The returned draft logits view has row(j) = the distribution d_{j+1} was
//   DRAWN FROM, exactly the contract of QwenSpeculativeLogits for
//   qwen_speculative_accept().
//
// Weight contract (canonical blk.<layers>.* names; <layers> = 32 frozen):
//   required packed matrices (storage must EQUAL the target embedding's):
//     blk.32.eh_proj.weight            [2*hidden, hidden]
//     blk.32.attn_q.weight             [hidden, 2*q_heads*head_dim]
//     blk.32.attn_k.weight             [hidden, kv_heads*head_dim]
//     blk.32.attn_v.weight             [hidden, kv_heads*head_dim]
//     blk.32.attn_output.weight        [q_heads*head_dim, hidden]
//     blk.32.ffn_gate.weight           [hidden, ffn]
//     blk.32.ffn_up.weight             [hidden, ffn]
//     blk.32.ffn_down.weight           [ffn, hidden]
//   required dense F32 (PRE-ADDED norm weights, same converter folding as the
//   trunk norms; multiplied directly):
//     blk.32.enorm.weight [hidden], blk.32.hnorm.weight [hidden],
//     blk.32.shared_head.norm.weight [hidden],
//     blk.32.attn_norm.weight [hidden], blk.32.post_attention_norm.weight
//     [hidden], blk.32.attn_q_norm.weight [head_dim],
//     blk.32.attn_k_norm.weight [head_dim]
//   optional packed (dedicated; absent = tied to the target embedding):
//     blk.32.token_embd.weight [hidden, vocab]  (draft input embedding)
//     blk.32.output.weight     [hidden, vocab]  (draft LM head)
//   Anything missing, malformed, storage-mismatched, or any unexpected extra
//   blk.32.* usage fails closed (recoverable Status, never a substitution).

#include <cstdint>
#include <memory>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

#include "qwen/common.h"
#include "qwen/dsl_ops.h"
#include "qwen/engine.h"
#include "qwen/kv_cache_allocator.h"
#include "qwen/model.h"
#include "qwen/speculative.h"
#include "qwen/weights.h"

namespace luisa::qwen {

// Draw namespace for draft-side sampling. speculative.h reserves tag 0
// (accept) and tag 1 (bonus) for the acceptance engine; the DRAFT token draws
// use tag 2 with (seed, round, index=draft step j-1, tag).
inline constexpr uint32_t kQwenNextnDraftDrawTag = 2u;

struct QwenNextnDraftInfo {
    WeightStorage weight_storage{WeightStorage::bf16_packed};
    bool dedicated_embeddings{false};
    bool dedicated_head{false};
    uint32_t max_context{0};
    uint32_t chunk{0};
    uint64_t cache_bytes{0};
    uint64_t workspace_bytes{0};
    uint64_t compiled_shader_count{0};
};

class QwenNextnDraft {
public:
    // Allocator-matched deleter (same pattern as QwenEngineDeleter): the
    // object is allocated with Luisa's allocator; luisa::unique_ptr's default
    // eastl::default_delete would free through a different heap.
    struct Deleter {
        void operator()(QwenNextnDraft *p) const noexcept {
            if (p != nullptr) {
                std::destroy_at(p);
                luisa::deallocate_with_allocator(p);
            }
        }
    };
    using Ptr = luisa::unique_ptr<QwenNextnDraft, Deleter>;

    // Validate the draft weight inventory WITHOUT touching a device: every
    // required blk.<layers>.* tensor present with frozen shape + storage
    // matching `target_embedding`, optional dedicated tensors well-formed.
    // Exposed for host tests; create() runs the same checks and fails closed.
    [[nodiscard]] static Status validate_weights(const Qwen35Model &model,
                                                 const QwenDeviceTensor &target_embedding,
                                                 const QwenDeviceWeights &draft_weights);

    // Create: validate + allocate the draft K/V cache ([max_context,
    // kv_heads, head_dim] x2) and chunk-sized activation workspace, compile
    // every kernel once, and reset() so the first propose starts clean.
    // `model` supplies the frozen geometry (cfg); `target_embedding` is the
    // trunk's tied token_embd tensor (used for draft embeddings and the LM
    // head unless dedicated blk.32.token_embd/output are present);
    // `draft_weights` carries the blk.<layers>.* set. Both weight sources are
    // BORROWED and must outlive the draft. The creation stream is retained;
    // reset()/propose() must use that same stream for sparse residency ordering.
    // `chunk` is the history-prefill row
    // batch (activation memory/launch tradeoff, >= 1).
    [[nodiscard]] static Result<Ptr> create(compute::Device &device, compute::Stream &stream,
                                            const Qwen35Model &model,
                                            const QwenDeviceTensor &target_embedding,
                                            const QwenDeviceWeights &draft_weights,
                                            uint32_t max_context, uint32_t chunk);

    ~QwenNextnDraft() noexcept;

    // Clear the draft K/V cache. Synchronizes before returning.
    [[nodiscard]] Status reset(compute::Stream &stream);

    // Run one draft round (see file contract): rebuild the draft prefill over
    // the entire provided history with shifted ids, then `draft_depth`
    // autoregressive steps. Synchronizes before returning (the per-step
    // logits downloads already synchronize the stream).
    //
    //   target_hidden_history: device rows [P, hidden]; row i is the trunk
    //                          output hidden for target_tokens[i]. The first
    //                          P rows are read; P == target_tokens.size().
    //   target_consumed_tokens: the P token ids the trunk has consumed.
    //   pending_token:          the trunk-sampled next token (< vocab), paired
    //                          with history row P-1.
//   draft_depth:            >= 1 drafted tokens; capacity is checked against
//                           the draft history and subsequent decode positions.
    //   sample/round:           sampling options + the caller's speculation
    //                           round counter (draft draw identity).
    //
    // Outputs (on success): drafted_ids_out holds draft_depth ids;
    // draft_logits_out is a QwenSpeculativeLogits view over INTERNAL host
    // storage with draft_depth rows -- valid until the next propose()/reset()
    // call (it aliases a persistent member that the next round rewrites).
    [[nodiscard]] Status propose(compute::Stream &stream,
                                 compute::BufferView<float> target_hidden_history,
                                 luisa::span<const uint32_t> target_consumed_tokens,
                                 uint32_t pending_token,
                                 uint32_t draft_depth,
                                 const QwenSampleOptions &sample,
                                 uint64_t round,
                                 luisa::vector<uint32_t> &drafted_ids_out,
                                 QwenSpeculativeLogits &draft_logits_out);

    [[nodiscard]] QwenNextnDraftInfo info() const noexcept;

private:
    QwenNextnDraft() noexcept = default;

    // ---- static geometry ---------------------------------------------------
    dsl::QwenDslConfig _cfg{};
    uint32_t _max_context{0};
    uint32_t _chunk{0};
    dsl::QwenWeightFormat _format{dsl::QwenWeightFormat::bf16};
    uint32_t _warp_size{0};

    // ---- borrowed weights / retained device --------------------------------
    const QwenDeviceWeights *_weights{nullptr};
    // Declared before every buffer/shader so reverse destruction releases all
    // GPU children first; retaining the handle also makes device-first caller
    // teardown safe (same lifetime contract as QwenDslEngine).
    compute::Device _device;
    compute::Stream _stream;

    // ---- resolved draft tensors (into *_weights / target embedding) --------
    struct DraftWeights {
        const QwenDeviceTensor *eh_proj{nullptr};
        const QwenDeviceTensor *enorm{nullptr};
        const QwenDeviceTensor *hnorm{nullptr};
        const QwenDeviceTensor *shared_head_norm{nullptr};
        const QwenDeviceTensor *attn_norm{nullptr};
        const QwenDeviceTensor *post_norm{nullptr};
        const QwenDeviceTensor *attn_q{nullptr};
        const QwenDeviceTensor *attn_k{nullptr};
        const QwenDeviceTensor *attn_v{nullptr};
        const QwenDeviceTensor *attn_output{nullptr};
        const QwenDeviceTensor *attn_q_norm{nullptr};
        const QwenDeviceTensor *attn_k_norm{nullptr};
        const QwenDeviceTensor *ffn_gate{nullptr};
        const QwenDeviceTensor *ffn_up{nullptr};
        const QwenDeviceTensor *ffn_down{nullptr};
    };
    DraftWeights _draft{};
    const QwenDeviceTensor *_embd{nullptr};// draft input embedding source
    const QwenDeviceTensor *_head{nullptr};// LM head source
    bool _dedicated_embeddings{false};
    bool _dedicated_head{false};

    // ---- persistent device state -------------------------------------------
    // The draft block owns one K/V pair in the same sparse allocator as the
    // target engines; propose() drops residency and rebuilds it every round.
    QwenKvCacheAllocator _kv_cache;
    uint64_t _cache_bytes{0};

    // ---- activation workspace (chunk-sized, reused per round) --------------
    compute::Buffer<uint32_t> _ids;      // [chunk] draft input ids
    compute::Buffer<uint32_t> _ids_host; // staging bounce for ids uploads
    compute::Buffer<int32_t> _positions; // [chunk] absolute positions
    compute::Buffer<int32_t> _positions_host;// staging bounce for positions
    // Persistent host staging: copy_from(span) borrows the host pointer until
    // the command encoder commits, so staging lives in members that outlive
    // every queued batch. Every chunk/step of one propose() uses a DISTINCT
    // slot ([_staged, _staged+rows)); the slot cursor resets per propose().
    luisa::vector<uint32_t> _host_ids;
    luisa::vector<int32_t> _host_pos;
    uint32_t _staged{0};
    compute::Buffer<float> _emb;     // [chunk, hidden] embedding gather
    compute::Buffer<float> _emb_n;   // [chunk, hidden] enorm(emb)
    compute::Buffer<float> _h_n;     // [chunk, hidden] hnorm(H)
    compute::Buffer<float> _fused;   // [chunk, 2*hidden] concat(emb_n, h_n)
    compute::Buffer<float> _hidden;  // [chunk, hidden] block residual stream
    compute::Buffer<float> _normed;  // [chunk, hidden] norm output
    compute::Buffer<float> _residual;// [chunk, hidden] residual snapshot
    compute::Buffer<float> _attn_proj_q; // [chunk, 2*q_heads*hd]
    compute::Buffer<float> _attn_proj_k; // [chunk, kv_heads*hd]
    compute::Buffer<float> _attn_proj_v; // [chunk, kv_heads*hd]
    compute::Buffer<float> _attn_q;      // [chunk, q_heads, hd]
    compute::Buffer<float> _attn_gate;   // [chunk, q_heads, hd]
    compute::Buffer<float> _attn_k;      // [chunk, kv_heads, hd]
    compute::Buffer<float> _attn_v;      // [chunk, kv_heads, hd]
    compute::Buffer<float> _attn_out;    // [chunk, q_heads, hd]
    compute::Buffer<float> _ffn_gate;    // [chunk, ffn]
    compute::Buffer<float> _ffn_up;      // [chunk, ffn]
    compute::Buffer<float> _ffn_act;     // [chunk, ffn]
    compute::Buffer<float> _feedback;    // [hidden] last-row / step feedback
    compute::Buffer<float> _logits;      // [vocab] one draft step's logits
    compute::Buffer<float> _dummy_f;     // 1-element dummy for unused args
    compute::Buffer<uint32_t> _dummy_u;  // 1-element dummy for unused args
    // Downloaded draft logits of the CURRENT round (rows == last draft_depth);
    // the QwenSpeculativeLogits returned by propose() aliases this storage.
    luisa::vector<float> _host_logits;
    uint64_t _workspace_bytes{0};

    // ---- compiled shaders (one per variant; shapes are runtime args) -------
    compute::Shader<2, compute::Buffer<uint32_t>, compute::Buffer<uint32_t>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _embedding;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, float> _rmsnorm_plain;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, float> _rmsnorm_snapshot;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<uint32_t>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t> _gemv;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<uint32_t>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t> _gemv_residual;
    compute::Shader<2, compute::Buffer<float>, compute::Buffer<uint32_t>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _gemm;
    compute::Shader<2, compute::Buffer<float>, compute::Buffer<uint32_t>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _gemm_residual;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, uint32_t> _swiglu;
    compute::Shader<1, compute::Buffer<uint32_t>, compute::Buffer<uint32_t>, uint32_t> _copy_u;
    compute::Shader<1, compute::Buffer<int32_t>, compute::Buffer<int32_t>, uint32_t> _copy_i;
    compute::Shader<2, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t> _concat_half;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<int32_t>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float,
                    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> _attn_prepare;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t, uint32_t> _attn_prefill;
    compute::Shader<1, compute::Buffer<float>, compute::Buffer<float>, compute::Buffer<float>,
                    compute::Buffer<float>, compute::Buffer<float>,
                    uint32_t, uint32_t, uint32_t, float, uint32_t, uint32_t> _attn_decode;
    uint64_t _shader_count{0};

    // ---- internal schedule helpers -----------------------------------------
    [[nodiscard]] Status alloc_state(compute::Device &device);
    [[nodiscard]] Status alloc_workspace(compute::Device &device);
    [[nodiscard]] Status compile_all(compute::Device &device);
    // Persistent-slot scalar staging (see the member lifetime rule above).
    [[nodiscard]] Status stage_ids(QwenCommandBatch &batch, luisa::span<const uint32_t> ids);
    [[nodiscard]] Status stage_positions(QwenCommandBatch &batch, uint32_t rows, uint32_t pos0);
    // packed GEMV (rows == 1) / GEMM projection with optional residual.
    [[nodiscard]] Status project(QwenCommandBatch &batch, const compute::BufferView<float> &x,
                                 const QwenDeviceTensor &w, const compute::BufferView<float> &y,
                                 uint32_t rows, const compute::Buffer<float> *residual,
                                 luisa::string_view label);
    // fused = eh_proj(concat(enorm(emb), hnorm(H))); H is a device view of
    // `rows` packed [rows, hidden] rows (history slice or the feedback row).
    [[nodiscard]] Status fused_input(QwenCommandBatch &batch, uint32_t rows,
                                     const compute::BufferView<float> &h);
    // One full-attention decoder block over the draft's own K/V cache at
    // absolute position pos0 (rows == 1 decode, rows > 1 causal prefill),
    // including the FFN; writes the [rows, hidden] residual output to _hidden.
    [[nodiscard]] Status block(QwenCommandBatch &batch, uint32_t rows, uint32_t pos0);
    // feedback = rmsnorm(_hidden last row, shared_head.norm) -> _feedback.
    [[nodiscard]] Status feedback_norm(QwenCommandBatch &batch, uint32_t rows);
    // logits = feedback @ head^T -> _logits.
    [[nodiscard]] Status lm_head(QwenCommandBatch &batch);
};

}// namespace luisa::qwen
