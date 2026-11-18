#pragma once
// Qwen3.5-4B shared engine layer -- frozen model geometry, config validation
// and exact tensor inventory / name resolution.
// Host-only: no DSL/runtime device includes.
//
// The geometry below is the frozen Qwen3.5-4B contract (see
// .kimix_cache/qwen_implementation_contract.md): 32 decoder blocks arranged
// as 8 x [GDN, GDN, GDN, full attention]; full-attention blocks are 3, 7,
// ..., 31. All shapes are validated against the GGUF metadata AND the exact
// tensor inventory (426 tensors for the reference BF16 checkpoint) -- any
// missing, extra, misnamed or mis-shapen tensor is a recoverable failure.
//
// GGUF dimensions are stored in ne order: matrices are [K, N] with payload
// row stride K; the depthwise convolution is [kernel, channels].

#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "qwen/common.h"
#include "qwen/gguf.h"

namespace luisa::qwen {

// Frozen Qwen3.5-4B geometry (text-only backbone; no vision tower / MTP).
struct Qwen35Geometry {
    uint64_t hidden{2560};
    uint64_t layers{32};
    uint64_t ffn{9216};
    uint64_t vocab{248320};
    // Full attention.
    uint64_t q_heads{16};
    uint64_t kv_heads{4};
    uint64_t head_dim{256};
    uint64_t q_proj_n{8192};// per head [Q(256), gate(256)]
    uint64_t kv_proj_n{1024};
    uint64_t o_proj_k{4096};
    uint64_t rope_dims{64};// partial NeoX rotate-half RoPE
    double rope_theta{1e7};
    // Gated DeltaNet.
    uint64_t gdn_qk_heads{16};
    uint64_t gdn_v_heads{32};
    uint64_t gdn_head_dim{128};
    uint64_t gdn_qkv_n{8192};// Q2048 + K2048 + V4096
    uint64_t gdn_z_n{4096};
    uint64_t gdn_gate_n{32}; // alpha/a and beta/b projections
    uint64_t gdn_out_k{4096};
    uint64_t conv_kernel{4};
    uint64_t full_attention_interval{4};// blocks 3, 7, ..., 31
    double rms_eps{1e-6};
    uint64_t context_length{262144};
};

[[nodiscard]] inline constexpr bool is_full_attention_layer(const Qwen35Geometry &g,
                                                            uint64_t layer) noexcept {
    return layer < g.layers && (layer + 1u) % g.full_attention_interval == 0u;
}

// Config parsed from GGUF metadata + cross-checked against the frozen
// geometry. `validate` fails closed on any mismatch; it never "adapts".
struct Qwen35Config {
    Qwen35Geometry geometry;
    luisa::string architecture;   // must be "qwen35"
    luisa::string name;           // general.name (informational)
    uint32_t gguf_file_type{0};   // general.file_type (informational)
    uint64_t quantization_version{0};
    // mRoPE sections as stored ([11, 11, 10, 0] in the reference file); the
    // first three must be 11/11/10. Text-only inference uses equal axes.
    luisa::vector<uint64_t> rope_dimension_sections;
    // Tokenizer metadata identity (values themselves stay in the GGUF).
    luisa::string tokenizer_model;// tokenizer.ggml.model, e.g. "gpt2"
    luisa::string tokenizer_pre;  // tokenizer.ggml.pre, e.g. "qwen35"
    uint64_t eos_token_id{0};
    uint64_t padding_token_id{0};
    bool has_chat_template{false};

    // Parse + validate. `reader` must already be open. Every required
    // metadata key must exist with the frozen value; unknown extra keys are
    // allowed (forward-compatible metadata), unknown tensors are NOT.
    // Default mode: qwen35.block_count must be exactly 32 and any
    // qwen35.nextn_predict_layers key is rejected (the engine trunk contract
    // has no NextN extension).
    // Opt-in mode `allow_nextn_extension` (MTP probe only, see
    // Qwen35Model::load_with_nextn): accepts exactly ONE declared NextN
    // extension, i.e. qwen35.block_count == 33 AND
    // qwen35.nextn_predict_layers == 1. Everything else stays frozen.
    [[nodiscard]] static Result<Qwen35Config> from_gguf(
        const GgufReader &reader, bool allow_nextn_extension = false);
};

// ---------------------------------------------------------------------------
// Exact tensor inventory and name resolution.
// ---------------------------------------------------------------------------
enum class QwenTensorRole : uint32_t {
    token_embd = 0,
    output_norm,
    attn_norm,
    post_attention_norm,
    ffn_gate,
    ffn_up,
    ffn_down,
    // full attention only
    attn_q,
    attn_k,
    attn_v,
    attn_output,
    attn_q_norm,
    attn_k_norm,
    // GDN only
    gdn_qkv,     // attn_qkv.weight
    gdn_gate,    // attn_gate.weight (z projection)
    gdn_alpha,   // ssm_alpha.weight (a projection)
    gdn_beta,    // ssm_beta.weight (b projection)
    gdn_out,     // ssm_out.weight
    gdn_conv1d,  // ssm_conv1d.weight
    gdn_a,       // ssm_a (per-head log decay parameter, F32)
    gdn_dt_bias, // ssm_dt.bias (F32)
    gdn_norm,    // ssm_norm.weight (per-head gated RMSNorm, direct weight)
};

[[nodiscard]] luisa::string_view qwen_tensor_role_name(QwenTensorRole role) noexcept;

// One resolved tensor: role, owning layer (UINT64_MAX for globals), the
// GGUF info pointer (owned by the reader) and the expected ne-ordered dims.
struct QwenTensorRef {
    QwenTensorRole role{QwenTensorRole::token_embd};
    uint64_t layer{UINT64_MAX};
    const GgufTensorInfo *info{nullptr};
};

// Canonical GGUF name for a role/layer (layer ignored for global roles).
// Returns an empty string for an invalid role/layer combination.
[[nodiscard]] luisa::string qwen_tensor_name(QwenTensorRole role, uint64_t layer = UINT64_MAX);

class Qwen35Model {
public:
    // Validate the config and resolve the EXACT tensor inventory:
    // 2 globals + 32 * 5 shared + 8 * 6 attention + 24 * 9 GDN = 426 tensors.
    // Fails on any missing/extra/mis-shapen tensor or dtype mismatch with the
    // frozen contract (matrices BF16 or F32 or Q8_0; norms/conv/a/dt F32).
    // DEFAULT engine path: requires exactly 426 tensors, block_count == 32
    // and no NextN declaration; a combined trunk+draft file FAILS here.
    [[nodiscard]] static Result<Qwen35Model> load(const GgufReader &reader);

    // Opt-in trunk validation for a combined trunk + one-layer NextN file,
    // used by both the eligibility probe and the explicit MTP executor.
    // Requires qwen35.block_count == 33 and qwen35.nextn_predict_layers == 1.
    // All 426 trunk tensors must still bijectively match the frozen
    // inventory (shapes + dtypes); the ONLY tolerated extras are vetted
    // `blk.32.*` names from the canonical NextN set (15 required + 2 optional
    // shared-head tensors). The exact set/shape/dtype validation of those
    // extras belongs to the MTP probe (qwen/mtp.h), which fails closed on
    // anything malformed. Any other extra tensor fails here.
    [[nodiscard]] static Result<Qwen35Model> load_with_nextn(const GgufReader &reader);

    [[nodiscard]] const Qwen35Config &config() const noexcept { return _config; }
    [[nodiscard]] luisa::span<const QwenTensorRef> tensors() const noexcept { return _tensors; }
    // Lookup by canonical name; nullptr when absent.
    [[nodiscard]] const QwenTensorRef *find(luisa::string_view name) const noexcept;
    // Lookup by role/layer; nullptr when the combination does not exist.
    [[nodiscard]] const QwenTensorRef *find(QwenTensorRole role,
                                            uint64_t layer = UINT64_MAX) const noexcept;
    [[nodiscard]] uint64_t full_attention_layers() const noexcept;
    [[nodiscard]] uint64_t gdn_layers() const noexcept;

private:
    // Shared implementation of load() (strict) and load_with_nextn() (opt-in).
    [[nodiscard]] static Result<Qwen35Model> load_impl(const GgufReader &reader, bool allow_nextn);

    Qwen35Config _config;
    luisa::vector<QwenTensorRef> _tensors;
};

}// namespace luisa::qwen
