// Qwen3.5-4B shared engine layer -- config validation + tensor inventory.
// Fail-closed: every mismatch is a recoverable Status, never an adaptation.
#include "qwen/model.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace luisa::qwen {
namespace {

[[nodiscard]] Status require_u64(const GgufReader &r, luisa::string_view key, uint64_t expected) {
    const auto v = r.meta_u64(key);
    if (!v.has_value()) {
        return Status::error(StatusCode::not_found,
                             "required Qwen3.5 metadata key missing or not an integer",
                             luisa::string{key});
    }
    if (*v != expected) {
        return Status::error(StatusCode::invalid_argument,
                             "Qwen3.5 metadata value does not match the frozen 4B geometry",
                             luisa::string{key});
    }
    return Status::success();
}

[[nodiscard]] Status require_f64_near(const GgufReader &r, luisa::string_view key,
                                      double expected, double rel_tol) {
    const auto v = r.meta_f64(key);
    if (!v.has_value()) {
        return Status::error(StatusCode::not_found,
                             "required Qwen3.5 metadata key missing or not a float",
                             luisa::string{key});
    }
    const double diff = std::abs(*v - expected);
    if (!(diff <= rel_tol * std::abs(expected))) {
        return Status::error(StatusCode::invalid_argument,
                             "Qwen3.5 metadata float does not match the frozen 4B geometry",
                             luisa::string{key});
    }
    return Status::success();
}

[[nodiscard]] Status require_string(const GgufReader &r, luisa::string_view key,
                                    luisa::string_view expected) {
    const auto v = r.meta_string(key);
    if (!v.has_value()) {
        return Status::error(StatusCode::not_found,
                             "required Qwen3.5 metadata string missing", luisa::string{key});
    }
    if (*v != expected) {
        return Status::error(StatusCode::invalid_argument,
                             "Qwen3.5 metadata string does not match the frozen contract",
                             luisa::string{key});
    }
    return Status::success();
}

// Expected ne-ordered dims for a role under the frozen geometry.
[[nodiscard]] luisa::vector<uint64_t> expected_dims(const Qwen35Geometry &g,
                                                    QwenTensorRole role) {
    using R = QwenTensorRole;
    switch (role) {
        case R::token_embd: return {g.hidden, g.vocab};
        case R::output_norm:
        case R::attn_norm:
        case R::post_attention_norm: return {g.hidden};
        case R::ffn_gate:
        case R::ffn_up: return {g.hidden, g.ffn};
        case R::ffn_down: return {g.ffn, g.hidden};
        case R::attn_q: return {g.hidden, g.q_proj_n};
        case R::attn_k:
        case R::attn_v: return {g.hidden, g.kv_proj_n};
        case R::attn_output: return {g.o_proj_k, g.hidden};
        case R::attn_q_norm:
        case R::attn_k_norm: return {g.head_dim};
        case R::gdn_qkv: return {g.hidden, g.gdn_qkv_n};
        case R::gdn_gate: return {g.hidden, g.gdn_z_n};
        case R::gdn_alpha:
        case R::gdn_beta: return {g.hidden, g.gdn_gate_n};
        case R::gdn_out: return {g.gdn_out_k, g.hidden};
        case R::gdn_conv1d: return {g.conv_kernel, g.gdn_qkv_n};
        case R::gdn_a:
        case R::gdn_dt_bias: return {g.gdn_v_heads};
        case R::gdn_norm: return {g.gdn_head_dim};
    }
    return {};
}

// Roles that must be stored dense F32 in the GGUF (sensitive tensors).
[[nodiscard]] bool role_requires_f32(QwenTensorRole role) noexcept {
    using R = QwenTensorRole;
    switch (role) {
        case R::output_norm:
        case R::attn_norm:
        case R::post_attention_norm:
        case R::attn_q_norm:
        case R::attn_k_norm:
        case R::gdn_conv1d:
        case R::gdn_a:
        case R::gdn_dt_bias:
        case R::gdn_norm: return true;
        default: return false;
    }
}

[[nodiscard]] bool role_is_global(QwenTensorRole role) noexcept {
    return role == QwenTensorRole::token_embd || role == QwenTensorRole::output_norm;
}

[[nodiscard]] bool role_is_attention(QwenTensorRole role) noexcept {
    using R = QwenTensorRole;
    switch (role) {
        case R::attn_q:
        case R::attn_k:
        case R::attn_v:
        case R::attn_output:
        case R::attn_q_norm:
        case R::attn_k_norm: return true;
        default: return false;
    }
}

[[nodiscard]] bool role_is_gdn(QwenTensorRole role) noexcept {
    using R = QwenTensorRole;
    switch (role) {
        case R::gdn_qkv:
        case R::gdn_gate:
        case R::gdn_alpha:
        case R::gdn_beta:
        case R::gdn_out:
        case R::gdn_conv1d:
        case R::gdn_a:
        case R::gdn_dt_bias:
        case R::gdn_norm: return true;
        default: return false;
    }
}

}// namespace

// Vetted extras for the opt-in NextN trunk mode: the canonical one-layer
// Qwen3.5-4B draft block (blk.32), 15 required + 2 optional shared-head
// tensors. Names mirror the frozen contract in qwen/mtp.h; the MTP probe
// owns the exact set/shape/dtype validation of these tensors -- this list
// only bounds WHICH extra names the trunk loader may tolerate.
[[nodiscard]] static bool is_vetted_nextn_extra(luisa::string_view name) noexcept {
    static constexpr luisa::string_view kAllowed[] = {
        "blk.32.eh_proj.weight",
        "blk.32.enorm.weight",
        "blk.32.hnorm.weight",
        "blk.32.shared_head.norm.weight",
        "blk.32.attn_norm.weight",
        "blk.32.post_attention_norm.weight",
        "blk.32.attn_q.weight",
        "blk.32.attn_k.weight",
        "blk.32.attn_v.weight",
        "blk.32.attn_output.weight",
        "blk.32.attn_q_norm.weight",
        "blk.32.attn_k_norm.weight",
        "blk.32.ffn_gate.weight",
        "blk.32.ffn_up.weight",
        "blk.32.ffn_down.weight",
        "blk.32.token_embd.weight",// optional dedicated embeddings
        "blk.32.output.weight",    // optional dedicated LM head
    };
    for (auto n : kAllowed) {
        if (name == n) { return true; }
    }
    return false;
}

[[nodiscard]] luisa::string_view qwen_tensor_role_name(QwenTensorRole role) noexcept {
    using R = QwenTensorRole;
    switch (role) {
        case R::token_embd: return "token_embd";
        case R::output_norm: return "output_norm";
        case R::attn_norm: return "attn_norm";
        case R::post_attention_norm: return "post_attention_norm";
        case R::ffn_gate: return "ffn_gate";
        case R::ffn_up: return "ffn_up";
        case R::ffn_down: return "ffn_down";
        case R::attn_q: return "attn_q";
        case R::attn_k: return "attn_k";
        case R::attn_v: return "attn_v";
        case R::attn_output: return "attn_output";
        case R::attn_q_norm: return "attn_q_norm";
        case R::attn_k_norm: return "attn_k_norm";
        case R::gdn_qkv: return "attn_qkv";
        case R::gdn_gate: return "attn_gate";
        case R::gdn_alpha: return "ssm_alpha";
        case R::gdn_beta: return "ssm_beta";
        case R::gdn_out: return "ssm_out";
        case R::gdn_conv1d: return "ssm_conv1d";
        case R::gdn_a: return "ssm_a";
        case R::gdn_dt_bias: return "ssm_dt_bias";
        case R::gdn_norm: return "ssm_norm";
    }
    return "unknown";
}

[[nodiscard]] luisa::string qwen_tensor_name(QwenTensorRole role, uint64_t layer) {
    using R = QwenTensorRole;
    if (role_is_global(role)) {
        if (role == R::token_embd) { return "token_embd.weight"; }
        return "output_norm.weight";
    }
    if (layer == UINT64_MAX) { return {}; }
    luisa::string name = luisa::string{"blk."} + luisa::string{std::to_string(layer)} + ".";
    switch (role) {
        case R::gdn_a: return name + "ssm_a";
        case R::gdn_dt_bias: return name + "ssm_dt.bias";
        default: return name + luisa::string{qwen_tensor_role_name(role)} + ".weight";
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
[[nodiscard]] Result<Qwen35Config> Qwen35Config::from_gguf(const GgufReader &reader,
                                                           bool allow_nextn_extension) {
    Qwen35Config cfg;
    const auto &g = cfg.geometry;
    const auto arch = reader.meta_string("general.architecture");
    if (!arch.has_value() || *arch != "qwen35") {
        return Result<Qwen35Config>::failure(StatusCode::invalid_argument,
                                             "GGUF architecture is not qwen35",
                                             "general.architecture");
    }
    cfg.architecture = luisa::string{*arch};
    if (const auto n = reader.meta_string("general.name"); n.has_value()) {
        cfg.name = luisa::string{*n};
    }
    if (const auto ft = reader.meta_u64("general.file_type"); ft.has_value()) {
        cfg.gguf_file_type = static_cast<uint32_t>(*ft);
    }
    if (const auto qv = reader.meta_u64("general.quantization_version"); qv.has_value()) {
        cfg.quantization_version = *qv;
    }
    // Frozen geometry keys (prefix qwen35.*).
    const struct {
        const char *key;
        uint64_t value;
    } required_u64[] = {
        {"qwen35.block_count", g.layers + (allow_nextn_extension ? 1u : 0u)},
        {"qwen35.context_length", g.context_length},
        {"qwen35.embedding_length", g.hidden},
        {"qwen35.feed_forward_length", g.ffn},
        {"qwen35.attention.head_count", g.q_heads},
        {"qwen35.attention.head_count_kv", g.kv_heads},
        {"qwen35.attention.key_length", g.head_dim},
        {"qwen35.attention.value_length", g.head_dim},
        {"qwen35.rope.dimension_count", g.rope_dims},
        {"qwen35.ssm.conv_kernel", g.conv_kernel},
        {"qwen35.ssm.state_size", g.gdn_head_dim},
        {"qwen35.ssm.group_count", g.gdn_qk_heads},
        {"qwen35.ssm.time_step_rank", g.gdn_v_heads},
        {"qwen35.ssm.inner_size", g.gdn_z_n},
        {"qwen35.full_attention_interval", g.full_attention_interval},
    };
    for (const auto &req : required_u64) {
        if (auto s = require_u64(reader, req.key, req.value); !s.ok()) {
            return Result<Qwen35Config>::failure(std::move(s));
        }
    }
    // NextN declaration contract. Default mode: the trunk-only engine
    // contract does not tolerate ANY nextn declaration. Opt-in mode (MTP
    // probe): exactly one declared NextN extension (block_count == 33 was
    // enforced above; nextn_predict_layers must be present and == 1).
    if (allow_nextn_extension) {
        if (auto s = require_u64(reader, "qwen35.nextn_predict_layers", 1u); !s.ok()) {
            return Result<Qwen35Config>::failure(std::move(s));
        }
    } else if (reader.metadata("qwen35.nextn_predict_layers") != nullptr) {
        return Result<Qwen35Config>::failure(
            StatusCode::invalid_argument,
            "qwen35.nextn_predict_layers declared but the default trunk contract "
            "requires exactly 32 blocks with no NextN extension",
            "qwen35.nextn_predict_layers");
    }
    if (auto s = require_f64_near(reader, "qwen35.rope.freq_base", g.rope_theta, 1e-6); !s.ok()) {
        return Result<Qwen35Config>::failure(std::move(s));
    }
    if (auto s = require_f64_near(reader, "qwen35.attention.layer_norm_rms_epsilon", g.rms_eps, 1e-3);
        !s.ok()) {
        return Result<Qwen35Config>::failure(std::move(s));
    }
    // mRoPE sections: [11, 11, 10] (+ optional trailing 0).
    const auto sections = reader.meta_array("qwen35.rope.dimension_sections");
    if (!sections.has_value() || sections->size() < 3u || sections->size() > 4u) {
        return Result<Qwen35Config>::failure(StatusCode::invalid_argument,
                                             "qwen35.rope.dimension_sections missing or wrong size",
                                             "qwen35.rope.dimension_sections");
    }
    constexpr uint64_t expected_sections[3] = {11u, 11u, 10u};
    for (size_t i = 0; i < sections->size(); ++i) {
        const auto &e = (*sections)[i];
        const uint64_t want = i < 3u ? expected_sections[i] : 0u;
        if (e.type == GgufMetaType::array || e.type == GgufMetaType::string) {
            return Result<Qwen35Config>::failure(StatusCode::invalid_argument,
                                                 "rope.dimension_sections element is not an integer",
                                                 "qwen35.rope.dimension_sections");
        }
        // Reuse the integer accessor semantics on the element directly.
        uint64_t v = 0;
        bool ok = true;
        switch (e.type) {
            case GgufMetaType::u8:
            case GgufMetaType::u16:
            case GgufMetaType::u32:
            case GgufMetaType::u64: v = e.u64; break;
            case GgufMetaType::i8:
            case GgufMetaType::i16:
            case GgufMetaType::i32:
            case GgufMetaType::i64:
                if (e.i64 < 0) { ok = false; } else { v = static_cast<uint64_t>(e.i64); }
                break;
            default: ok = false; break;
        }
        if (!ok || v != want) {
            return Result<Qwen35Config>::failure(StatusCode::invalid_argument,
                                                 "rope.dimension_sections do not match [11,11,10]",
                                                 "qwen35.rope.dimension_sections");
        }
        cfg.rope_dimension_sections.emplace_back(v);
    }
    // Tokenizer metadata identity. The token table itself stays in the GGUF
    // reader (meta_string_array("tokenizer.ggml.tokens") etc.).
    const auto tok_model = reader.meta_string("tokenizer.ggml.model");
    const auto tok_pre = reader.meta_string("tokenizer.ggml.pre");
    if (!tok_model.has_value() || !tok_pre.has_value()) {
        return Result<Qwen35Config>::failure(StatusCode::not_found,
                                             "tokenizer.ggml.model/pre metadata missing",
                                             "tokenizer.ggml");
    }
    cfg.tokenizer_model = luisa::string{*tok_model};
    cfg.tokenizer_pre = luisa::string{*tok_pre};
    const auto eos = reader.meta_u64("tokenizer.ggml.eos_token_id");
    const auto pad = reader.meta_u64("tokenizer.ggml.padding_token_id");
    if (!eos.has_value() || !pad.has_value() || *eos >= g.vocab || *pad >= g.vocab) {
        return Result<Qwen35Config>::failure(StatusCode::invalid_argument,
                                             "tokenizer eos/padding ids missing or outside vocab",
                                             "tokenizer.ggml");
    }
    cfg.eos_token_id = *eos;
    cfg.padding_token_id = *pad;
    cfg.has_chat_template = reader.meta_string("tokenizer.chat_template").has_value();
    // Sanity: derived projection sizes must agree with the frozen geometry.
    if (g.q_proj_n != 2u * g.q_heads * g.head_dim ||
        g.kv_proj_n != g.kv_heads * g.head_dim ||
        g.o_proj_k != g.q_heads * g.head_dim ||
        g.gdn_qkv_n != (2u * g.gdn_qk_heads + g.gdn_v_heads) * g.gdn_head_dim ||
        g.gdn_z_n != g.gdn_v_heads * g.gdn_head_dim ||
        g.gdn_out_k != g.gdn_z_n) {
        return Result<Qwen35Config>::failure(StatusCode::internal_error,
                                             "frozen geometry is internally inconsistent",
                                             "Qwen35Geometry");
    }
    return Result<Qwen35Config>::success(std::move(cfg));
}

// ---------------------------------------------------------------------------
// Model inventory
// ---------------------------------------------------------------------------
// Shared implementation of load() (strict) and load_with_nextn() (opt-in).
[[nodiscard]] Result<Qwen35Model> Qwen35Model::load_impl(const GgufReader &reader, bool allow_nextn) {
    auto cfg = Qwen35Config::from_gguf(reader, allow_nextn);
    if (!cfg.ok()) { return Result<Qwen35Model>::failure(std::move(cfg.status)); }
    Qwen35Model model;
    model._config = std::move(*cfg.value);
    const auto &g = model._config.geometry;

    // Build the exact expected inventory in a deterministic order.
    luisa::vector<std::pair<QwenTensorRole, uint64_t>> expected;
    expected.emplace_back(QwenTensorRole::token_embd, UINT64_MAX);
    expected.emplace_back(QwenTensorRole::output_norm, UINT64_MAX);
    const QwenTensorRole shared[] = {
        QwenTensorRole::attn_norm, QwenTensorRole::post_attention_norm,
        QwenTensorRole::ffn_gate, QwenTensorRole::ffn_up, QwenTensorRole::ffn_down,
    };
    const QwenTensorRole attention[] = {
        QwenTensorRole::attn_q, QwenTensorRole::attn_k, QwenTensorRole::attn_v,
        QwenTensorRole::attn_output, QwenTensorRole::attn_q_norm, QwenTensorRole::attn_k_norm,
    };
    const QwenTensorRole gdn[] = {
        QwenTensorRole::gdn_qkv, QwenTensorRole::gdn_gate, QwenTensorRole::gdn_alpha,
        QwenTensorRole::gdn_beta, QwenTensorRole::gdn_out, QwenTensorRole::gdn_conv1d,
        QwenTensorRole::gdn_a, QwenTensorRole::gdn_dt_bias, QwenTensorRole::gdn_norm,
    };
    for (uint64_t layer = 0; layer < g.layers; ++layer) {
        for (auto role : shared) { expected.emplace_back(role, layer); }
        if (is_full_attention_layer(g, layer)) {
            for (auto role : attention) { expected.emplace_back(role, layer); }
        } else {
            for (auto role : gdn) { expected.emplace_back(role, layer); }
        }
    }
    if (allow_nextn) {
        // Combined file: the 426 trunk tensors plus ONLY vetted blk.32.*
        // extras (at most the 17 canonical NextN tensors). Exact set/shape
        // validation of the extras is the MTP probe's job.
        if (reader.tensors().size() < expected.size() ||
            reader.tensors().size() > expected.size() + 17u) {
            return Result<Qwen35Model>::failure(
                StatusCode::invalid_argument,
                luisa::string{"GGUF tensor count does not match the Qwen3.5-4B trunk + "
                              "one-layer NextN inventory ("} +
                    luisa::string{std::to_string(expected.size())} + ".." +
                    luisa::string{std::to_string(expected.size() + 17u)} + " expected, " +
                    luisa::string{std::to_string(reader.tensors().size())} + " found)",
                "tensors");
        }
    } else if (reader.tensors().size() != expected.size()) {
        return Result<Qwen35Model>::failure(
            StatusCode::invalid_argument,
            luisa::string{"GGUF tensor count does not match the exact Qwen3.5-4B inventory ("} +
                luisa::string{std::to_string(expected.size())} + " expected, " +
                luisa::string{std::to_string(reader.tensors().size())} + " found)",
            "tensors");
    }
    luisa::unordered_map<luisa::string, size_t> resolved;
    for (const auto &[role, layer] : expected) {
        const auto name = qwen_tensor_name(role, layer);
        const auto *info = reader.find(name);
        if (info == nullptr) {
            return Result<Qwen35Model>::failure(StatusCode::not_found,
                                                "required Qwen3.5 tensor missing", name);
        }
        const auto dims = expected_dims(g, role);
        if (info->dims.size() != dims.size() ||
            !std::equal(info->dims.begin(), info->dims.end(), dims.begin())) {
            return Result<Qwen35Model>::failure(StatusCode::invalid_argument,
                                                "Qwen3.5 tensor shape does not match the frozen geometry",
                                                name);
        }
        if (role_requires_f32(role)) {
            if (info->type != static_cast<uint32_t>(GgufTensorType::f32)) {
                return Result<Qwen35Model>::failure(
                    StatusCode::invalid_argument,
                    "sensitive Qwen3.5 tensor must be stored F32 (found " +
                        luisa::string{gguf_tensor_type_name(info->type)} + ")",
                    name);
            }
        } else if (!gguf_tensor_type_supported(info->type)) {
            return Result<Qwen35Model>::failure(StatusCode::unsupported,
                                                "unsupported tensor dtype", name);
        }
        resolved.emplace(name, model._tensors.size());
        model._tensors.emplace_back(QwenTensorRef{role, layer, info});
    }
    // Reject extra tensors. Default mode: the count check above plus
    // per-name resolution already guarantees a bijection. Opt-in NextN
    // mode: every tensor outside the resolved trunk inventory must be a
    // vetted canonical blk.32.* name; anything else fails closed.
    if (resolved.size() != reader.tensors().size()) {
        if (!allow_nextn) {
            return Result<Qwen35Model>::failure(StatusCode::invalid_argument,
                                                "GGUF contains unexpected extra tensors", "tensors");
        }
        for (const auto &t : reader.tensors()) {
            if (resolved.find(t.name) == resolved.end() && !is_vetted_nextn_extra(t.name)) {
                return Result<Qwen35Model>::failure(
                    StatusCode::invalid_argument,
                    "GGUF contains an extra tensor outside the trunk inventory and the "
                    "vetted one-layer NextN set",
                    t.name);
            }
        }
    }
    // Store the name index implicitly via the reader; keep a small role index.
    return Result<Qwen35Model>::success(std::move(model));
}

[[nodiscard]] Result<Qwen35Model> Qwen35Model::load(const GgufReader &reader) {
    return load_impl(reader, false);
}

[[nodiscard]] Result<Qwen35Model> Qwen35Model::load_with_nextn(const GgufReader &reader) {
    return load_impl(reader, true);
}

[[nodiscard]] const QwenTensorRef *Qwen35Model::find(luisa::string_view name) const noexcept {
    for (const auto &t : _tensors) {
        if (t.info->name == name) { return &t; }
    }
    return nullptr;
}

[[nodiscard]] const QwenTensorRef *Qwen35Model::find(QwenTensorRole role, uint64_t layer) const noexcept {
    for (const auto &t : _tensors) {
        if (t.role == role && t.layer == layer) { return &t; }
    }
    return nullptr;
}

[[nodiscard]] uint64_t Qwen35Model::full_attention_layers() const noexcept {
    uint64_t n = 0;
    for (uint64_t l = 0; l < _config.geometry.layers; ++l) {
        if (is_full_attention_layer(_config.geometry, l)) { ++n; }
    }
    return n;
}

  [[nodiscard]] uint64_t Qwen35Model::gdn_layers() const noexcept {
      return _config.geometry.layers - full_attention_layers();
  }

  // ---------------------------------------------------------------------------
  // Merged input-projection inventory (opt_plan/03 section 3 + D5).
  // ---------------------------------------------------------------------------
  [[nodiscard]] luisa::string_view qwen_merge_kind_name(QwenMergeKind kind) noexcept {
      switch (kind) {
          case QwenMergeKind::none: return "none";
          case QwenMergeKind::gdn_qkv_z_ab: return "gdn_qkv_z_ab";
          case QwenMergeKind::attn_qkv: return "attn_qkv";
      }
      return "unknown";
  }

  [[nodiscard]] Result<luisa::vector<QwenMergedMatrixSpec>>
  qwen35_merged_projection_inventory(const Qwen35Model &model) {
      const auto &g = model.config().geometry;
      if (g.layers == 0u || g.hidden == 0u) {
          return Result<luisa::vector<QwenMergedMatrixSpec>>::failure(
              Status::error(StatusCode::invalid_argument,
                            "merged projection inventory needs a non-empty model geometry"));
      }
      luisa::vector<QwenMergedMatrixSpec> inventory;
      inventory.reserve(static_cast<size_t>(g.layers));
      for (uint64_t l = 0; l < g.layers; ++l) {
          QwenMergedMatrixSpec spec;
          spec.layer = l;
          spec.k = g.hidden;
          using R = QwenTensorRole;
          struct RoleRows {
              R role;
              uint64_t rows;
          };
          // Segment order is part of the ABI: the decode GEMV routes output rows to
          // destinations by exactly this order, and the offline packer writes the
          // rows in it. Changing it is a pack version change, never a free edit.
          luisa::vector<RoleRows> want;
          if (is_full_attention_layer(g, l)) {
              spec.kind = QwenMergeKind::attn_qkv;
              want = {{R::attn_q, g.q_proj_n},
                      {R::attn_k, g.kv_proj_n},
                      {R::attn_v, g.kv_proj_n}};
          } else {
              spec.kind = QwenMergeKind::gdn_qkv_z_ab;
              want = {{R::gdn_qkv, g.gdn_qkv_n},
                      {R::gdn_gate, g.gdn_z_n},
                      {R::gdn_alpha, g.gdn_gate_n},
                      {R::gdn_beta, g.gdn_gate_n}};
          }
          uint64_t rows = 0;
          for (const auto &w : want) {
              const auto *ref = model.find(w.role, l);
              if (ref == nullptr || ref->info == nullptr) {
                  return Result<luisa::vector<QwenMergedMatrixSpec>>::failure(Status::error(
                      StatusCode::not_found,
                      "merged projection segment source tensor is missing",
                      luisa::string{qwen_tensor_name(w.role, l)}));
              }
              // The source matrix must be exactly [K = hidden, rows]: the merge is a
              // row concatenation, so any other shape would silently re-segment.
              if (ref->info->dims.size() != 2u || ref->info->dims[0] != spec.k || ref->info->dims[1] != w.rows) {
                  return Result<luisa::vector<QwenMergedMatrixSpec>>::failure(Status::error(
                      StatusCode::invalid_argument,
                      "merged projection segment source shape does not match the frozen geometry",
                      luisa::string{qwen_tensor_name(w.role, l)}));
              }
              QwenTensorSegment seg;
              seg.role = w.role;
              seg.name = ref->info->name;
              seg.rows = w.rows;
              seg.row_offset = rows;
              spec.segments.emplace_back(std::move(seg));
              rows += w.rows;
          }
          spec.n = rows;
          if (spec.n == 0u || spec.n > 0xFFFFFFFFull || spec.k > 0xFFFFFFFFull) {
              return Result<luisa::vector<QwenMergedMatrixSpec>>::failure(Status::error(
                  StatusCode::invalid_argument,
                  "merged projection shape exceeds the uint32 kernel envelope"));
          }
          inventory.emplace_back(std::move(spec));
      }
      return Result<luisa::vector<QwenMergedMatrixSpec>>::success(std::move(inventory));
  }

  }// namespace luisa::qwen
