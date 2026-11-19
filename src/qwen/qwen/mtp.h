#pragma once
// Qwen3.5-4B shared engine layer -- isolated, host-only MTP (multi-token
// prediction / NextN speculative decoding) eligibility and default-policy
// probe. Host-only: no DSL/runtime device includes, no engine state, no
// weight loading. Pure inventory inspection over an open GgufReader.
//
// Honesty contract (the whole point of this module):
//
//  1. The HF config for Qwen3.5-4B declares exactly ONE MTP layer
//     (`mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false`)
//     and the official HF checkpoint carries `mtp.*` weights. The local
//     data/Qwen3.5-4B-BF16.gguf (Unsloth export, 426 tensors) OMITS them:
//     no `qwen35.nextn_predict_layers` metadata key and no `blk.32.*`
//     tensors. This probe reports that truthfully instead of guessing.
//
//  2. `available` means: the GGUF actually contains a COMPLETE one-layer
//     full-attention Qwen3.5 NextN draft block with the canonical converter
//     names, frozen-geometry shapes and acceptable dtypes (see below).
//     Present-but-malformed or incomplete inventories fail closed.
//
//  3. `benchmark_feasible` additionally requires a valid target trunk in the
//     SAME file (shared tied embedding / LM head and final norm), validated
//     through the OPT-IN combined-file trunk mode
//     (Qwen35Model::load_with_nextn: block_count=33, nextn_predict_layers=1,
//     all 426 trunk tensors bijective, only vetted blk.32.* extras). The
//     default Qwen35Model::load stays exact-426/block_count=32 for normal
//     engine use and intentionally rejects combined files. Weights alone
//     never prove acceleration.
//
//  4. `benchmark_passed` / `accelerates` / `default_enabled` are ALWAYS
//     false in this probe. Nothing here runs a benchmark, and no code path
//     in this module can set them true. The actual opt-in executor lives in
//     nextn_draft/mtp_executor; its default policy may only flip
//     after a REAL measured greedy draft+verify A/B benchmark against the
//     target-only decode baseline, on the same device/backend, that
//     (a) proves token-exact output equivalence with speculation disabled
//         under greedy decoding, covering every accept/reject combination
//         (all rejected, first-only, partial, all accepted, multi-round and
//         long-context speculation), with correct rollback of ALL state --
//         full-attention logical KV length, GDN FP32 recurrent state, GDN
//         conv state and position/mRoPE delta (truncating KV alone is NOT
//         sufficient for this hybrid model); and
//     (b) measures a net end-to-end decode speedup (draft cost + verify
//         cost + state snapshot/restore cost included).
//     Until such a benchmark exists and passes, MTP stays OFF by default.
//
// Canonical GGUF NextN names for the one-layer Qwen3.5-4B draft, derived
// from the converter (.kimix_cache/conv_qwen.py `_QwenMtpMixin`): `mtp.*`
// is remapped to `model.layers.{num_hidden_layers}.*`, i.e. block index 32,
// with fc->eh_proj, pre_fc_norm_embedding->enorm, pre_fc_norm_hidden->hnorm,
// norm->shared_head.norm; the block itself is a standard full-attention
// Qwen3.5 decoder block (llama.cpp models_qwen35.cpp load_block_mtp).
// 15 required tensors (ne-ordered dims under the frozen 4B geometry):
//   blk.32.eh_proj.weight            [5120, 2560]  (concat enorm(e), hnorm(h))
//   blk.32.enorm.weight              [2560]
//   blk.32.hnorm.weight              [2560]
//   blk.32.shared_head.norm.weight   [2560]
//   blk.32.attn_norm.weight          [2560]
//   blk.32.post_attention_norm.weight[2560]
//   blk.32.attn_q.weight             [2560, 8192]  (per head [Q(256),gate(256)])
//   blk.32.attn_k.weight             [2560, 1024]
//   blk.32.attn_v.weight             [2560, 1024]
//   blk.32.attn_output.weight        [4096, 2560]
//   blk.32.attn_q_norm.weight        [256]
//   blk.32.attn_k_norm.weight        [256]
//   blk.32.ffn_gate.weight           [2560, 9216]
//   blk.32.ffn_up.weight             [2560, 9216]
//   blk.32.ffn_down.weight           [9216, 2560]
// Optional (shared with the trunk; mtp_use_dedicated_embeddings=false):
//   blk.32.token_embd.weight         [2560, 248320] (dedicated embeddings)
//   blk.32.output.weight             [2560, 248320] (dedicated LM head)
// Any OTHER `blk.32.*` tensor (GDN/ssm tensors, scales, ...) makes the draft
// malformed and the probe fails closed. An optional tensor that IS present
// but malformed (wrong shape/dtype) also fails the draft closed -- optional
// means "may be absent", never "may be broken".

#include <cstdint>

#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "qwen/common.h"
#include "qwen/gguf.h"

namespace luisa::qwen {

// Metadata key the converter emits for NextN draft blocks
// (gguf_writer.add_nextn_predict_layers -> "<arch>.nextn_predict_layers").
inline constexpr luisa::string_view kMtpNextnMetadataKey = "qwen35.nextn_predict_layers";
// Frozen Qwen3.5-4B trunk block count; the one MTP draft block is blk.32.
inline constexpr uint64_t kMtpTrunkBlockCount = 32u;
inline constexpr uint64_t kMtpDraftBlockIndex = 32u;
// Required/optional canonical tensor counts for the one-layer draft.
inline constexpr uint64_t kMtpRequiredTensorCount = 15u;
inline constexpr uint64_t kMtpOptionalTensorCount = 2u;

enum class MtpTensorState : uint32_t {
    missing = 0,     // no tensor with this canonical name in the GGUF
    malformed,       // present, but wrong shape and/or unacceptable dtype
    ok,              // present with the expected shape and an acceptable dtype
};
[[nodiscard]] luisa::string_view mtp_tensor_state_name(MtpTensorState s) noexcept;

// Per-tensor audit entry. `required` marks the 15 mandatory draft tensors.
// `expected_dims` is ne-ordered under the frozen geometry; `actual_dims`,
// `actual_type` and `issue` are filled when the tensor is present.
struct MtpTensorCheck {
    luisa::string name;
    bool required{true};
    luisa::vector<uint64_t> expected_dims;
    MtpTensorState state{MtpTensorState::missing};
    luisa::vector<uint64_t> actual_dims;
    uint32_t actual_type{0};// raw GgufTensorType value (valid when present)
    luisa::string issue;    // human-readable mismatch description ("" when ok/missing)
};

// Detailed, honest MTP eligibility / default-policy report.
struct MtpEligibilityReport {
    // ---- metadata evidence ------------------------------------------------
    luisa::string architecture;              // general.architecture ("" when absent)
    bool architecture_is_qwen35{false};
    bool nextn_metadata_present{false};      // key exists in the GGUF
    bool nextn_metadata_valid{false};        // integer scalar in [0, block_count]
    uint64_t nextn_predict_layers{0};        // parsed count (0 when absent/invalid)
    luisa::optional<uint64_t> block_count;   // <arch>.block_count (nullopt when absent)

    // ---- inventory evidence ------------------------------------------------
    uint64_t total_tensor_count{0};          // all tensors in the file
    uint64_t detected_tensor_count{0};       // tensors named blk.32.*
    luisa::vector<luisa::string> detected_tensor_names;// sorted, canonical or not
    luisa::vector<luisa::string> unexpected_tensor_names;// blk.32.* outside the canonical set
    luisa::vector<MtpTensorCheck> tensor_checks;// 15 required + 2 optional, fixed order
    uint64_t required_ok_count{0};
    uint64_t required_missing_count{0};
    uint64_t required_malformed_count{0};
    uint64_t optional_present_count{0};
    uint64_t optional_malformed_count{0};// present-but-malformed optional tensors

    // ---- verdicts ----------------------------------------------------------
    // True ONLY when: qwen35 architecture, nextn metadata present+valid and
    // == 1, block_count == 33 (32 trunk + 1 draft), all 15 required tensors
    // ok, no present-but-malformed optional tensor, and no unexpected
    // blk.32.* tensors. Anything else fails closed.
    bool draft_complete{false};
    // The target trunk validates in the SAME file through the opt-in
    // combined-file mode (Qwen35Model::load_with_nextn: all 426 frozen trunk
    // tensors bijective + only vetted blk.32.* extras), so a real A/B
    // target-vs-draft benchmark is *possible* (weights feasibility only).
    bool target_trunk_ok{false};
    // Weights-only feasibility; NOT a performance claim.
    bool available{false};        // draft_complete
    bool benchmark_feasible{false};// draft_complete && target_trunk_ok

    // ---- default policy (honest, never optimistic) -------------------------
    // This probe never benchmarks; these stay false unconditionally. A real
    // measured greedy draft+verify A/B with full state-rollback correctness
    // (KV logical length + GDN recurrent/conv state + position/mRoPE delta)
    // AND a net measured decode speedup is the ONLY thing that may flip the
    // default to enabled -- see the header contract.
    bool benchmarked{false};
    bool benchmark_passed{false};
    bool accelerates{false};
    bool default_enabled{false};

    // Clear, human-readable explanation of the verdict (always non-empty).
    luisa::string reason;

    // One deterministic line per report field group, for CLI/test logging.
    [[nodiscard]] luisa::vector<luisa::string> report_lines() const;
};

// Inspect an open GGUF for the one-layer Qwen3.5-4B MTP draft. Pure host
// inventory/metadata inspection: no payload reads, no device, no engine
// state. Always succeeds with a report; malformed inputs are reflected as
// fail-closed verdicts inside the report, never as guesses.
[[nodiscard]] MtpEligibilityReport probe_mtp_eligibility(const GgufReader &reader);

// Device-free self test:
//  - deterministic synthetic GGUF fixtures through the REAL GgufReader:
//    no-MTP trunk slice (the current checkpoint situation), complete draft,
//    partially missing draft, wrong-shape draft, wrong-dtype draft,
//    unexpected blk.32.* tensor, missing/invalid nextn metadata;
//  - a FULL combined fixture: the complete valid frozen 426-tensor trunk
//    (tiny zero payloads, real shapes/metadata) PLUS the 15 draft tensors,
//    asserting the positive path draft_complete && target_trunk_ok &&
//    benchmark_feasible with all measured/default flags still false, and
//    that the DEFAULT Qwen35Model::load rejects the combined file;
//  - malformed optional blk.32.token_embd/output tensors fail closed;
//  - when data/Qwen3.5-4B-BF16.gguf is present, additionally probes the real
//    checkpoint and asserts the honest verdict: available=false,
//    benchmarked=false, accelerates=false, default_enabled=false, reason
//    "missing weights" (426 tensors, no nextn metadata, no blk.32.*).
// Returns ok on success; failures are recoverable Status, lines go to
// `report` when non-null.
[[nodiscard]] Status mtp_self_test(luisa::vector<luisa::string> *report = nullptr);

}// namespace luisa::qwen
