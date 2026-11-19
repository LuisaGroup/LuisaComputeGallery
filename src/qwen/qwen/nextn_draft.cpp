// Qwen3.5-4B NextN / MTP draft block -- ordinary-DSL implementation.
// See nextn_draft.h for the semantics/weight contract. All device compute
// goes through the dsl_ops.h kernel factories plus ONE small private concat
// kernel defined here (eh_proj needs [enorm(emb) | hnorm(H)] packed into a
// single [rows, 2*hidden] row-major buffer; the shared rmsnorm/embedding
// kernels use packed [rows, dim] layouts, so the concat is explicit).
#include "qwen/nextn_draft.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <luisa/dsl/sugar.h>

namespace luisa::qwen {

using compute::Buffer;
using compute::BufferView;
using compute::Device;
using compute::Stream;
namespace d = luisa::qwen::dsl;
using namespace luisa::compute;

namespace {

[[nodiscard]] Status err(StatusCode code, luisa::string msg, luisa::string detail = {}) {
    return Status::error(code, std::move(msg), std::move(detail));
}
[[nodiscard]] Status invalid(luisa::string msg, luisa::string detail = {}) {
    return err(StatusCode::invalid_argument, std::move(msg), std::move(detail));
}

[[nodiscard]] Result<uint32_t> u32(uint64_t v, const char *what) {
    if (v > 0xFFFFFFFFull) {
        return Result<uint32_t>::failure(invalid(luisa::string{what} + " exceeds uint32"));
    }
    return Result<uint32_t>::success(static_cast<uint32_t>(v));
}

// A matrix weight must be packed (bf16/fp8) with dims [K, N] (GGUF ne order).
[[nodiscard]] Status check_matrix(const QwenDeviceTensor &t, uint64_t k, uint64_t n,
                                  d::QwenWeightFormat format) {
    const auto want = format == d::QwenWeightFormat::bf16 ? WeightStorage::bf16_packed
                                                          : WeightStorage::fp8_packed;
    if (t.storage != want) {
        return invalid("draft weight storage does not match the target embedding (no substitution)", t.name);
    }
    if (t.dims.size() != 2u || t.dims[0] != k || t.dims[1] != n) {
        return invalid("draft weight matrix shape mismatch", t.name);
    }
    uint64_t elements = 0u;
    if (!checked_mul(k, n, elements) || t.element_count != elements) {
        return invalid("draft weight matrix element count mismatch/overflow", t.name);
    }
    const uint64_t codes_per_word = format == d::QwenWeightFormat::bf16 ? 2u : 4u;
    const uint64_t expected_words = (elements + codes_per_word - 1u) / codes_per_word;
    if (!t.packed || t.packed.size() != expected_words) {
        return invalid("draft weight packed buffer size mismatch", t.name);
    }
    if (format == d::QwenWeightFormat::fp8_e4m3) {
        const uint64_t expect = (k + 63u) / 64u;// ceil(K/64)
        uint64_t scale_count = 0u;
        if (!t.scales || t.groups_per_row != expect ||
            !checked_mul(n, expect, scale_count) || t.scales.size() != scale_count) {
            return invalid("draft fp8 weight scale geometry/buffer size mismatch", t.name);
        }
    }
    return Status::success();
}

[[nodiscard]] Status check_dense(const QwenDeviceTensor &t, uint64_t count) {
    if (t.storage != WeightStorage::f32_dense) {
        return invalid("draft norm must be dense F32", t.name);
    }
    if (t.element_count != count || !t.f32 || t.f32.size() != count) {
        return invalid("draft dense F32 tensor size/buffer mismatch", t.name);
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Private concat kernel: out[r, c] = c < half ? a[r, c] : b[r, c - half].
// Packs [enorm(emb) | hnorm(H)] into the [rows, 2*hidden] eh_proj input with
// the embedding half FIRST (vLLM concat order). Same launch convention as the
// dsl_ops 2D kernels: padded_cols == dispatch_size().x recovers (row, col)
// from the linear id, covering folded and unfolded plans with one code path.
// ---------------------------------------------------------------------------
using NextnConcatHalfKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*a [rows, half]*/,
    compute::Buffer<float> /*b [rows, half]*/,
    compute::Buffer<float> /*out [rows, 2*half]*/,
    uint32_t /*rows*/, uint32_t /*half*/, uint32_t /*padded_cols*/)>;
[[nodiscard]] NextnConcatHalfKernel make_nextn_concat_half_kernel() noexcept {
    NextnConcatHalfKernel kernel = [](BufferFloat a, BufferFloat b, BufferFloat out,
                                      Var<uint> rows, Var<uint> half, Var<uint> padded_cols) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("qwen_nextn_concat_half");
        const auto linear = dispatch_y() * dispatch_size().x + dispatch_x();
        const auto row = linear / padded_cols;
        const auto col = linear % padded_cols;
        $if (row < rows & col < 2u * half) {
            Var<float> v = def(0.0f);
            $if (col < half) {
                v = a.read(row * half + col);
            } $else {
                v = b.read(row * half + (col - half));
            };
            out.write(row * (2u * half) + col, v);
        };
    };
    return kernel;
}

// ---------------------------------------------------------------------------
// Draft weight resolution (shared by validate_weights and create).
// ---------------------------------------------------------------------------
struct ResolvedNextn {
    d::QwenWeightFormat format{d::QwenWeightFormat::bf16};
    const QwenDeviceTensor *embd{nullptr};
    const QwenDeviceTensor *head{nullptr};
    bool dedicated_embeddings{false};
    bool dedicated_head{false};
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

[[nodiscard]] Result<ResolvedNextn> resolve_nextn(const Qwen35Model &model,
                                                  const QwenDeviceTensor &target_embedding,
                                                  const QwenDeviceWeights &weights) {
    using R = Result<ResolvedNextn>;
    const auto &g = model.config().geometry;
    ResolvedNextn out;
    // Uniform packed storage follows the target (tied trunk) embedding; every
    // draft matrix must match it. Dense F32 norms stay dense in both variants.
    if (target_embedding.storage == WeightStorage::bf16_packed) {
        out.format = d::QwenWeightFormat::bf16;
    } else if (target_embedding.storage == WeightStorage::fp8_packed) {
        out.format = d::QwenWeightFormat::fp8_e4m3;
    } else {
        return R::failure(invalid("target embedding must be packed bf16 or fp8 (dense F32 unsupported)",
                                  target_embedding.name));
    }
    if (auto s = check_matrix(target_embedding, g.hidden, g.vocab, out.format); !s.ok()) {
        return R::failure(std::move(s));
    }

    const auto blk = luisa::string{"blk."} + luisa::string{std::to_string(g.layers)};
    const auto find_required = [&weights, &blk](const char *suffix) -> Result<const QwenDeviceTensor *> {
        using RR = Result<const QwenDeviceTensor *>;
        const auto name = blk + "." + suffix;
        const auto *t = weights.find(name);
        if (t == nullptr) {
            return RR::failure(invalid("draft weight missing", name));
        }
        return RR::success(t);
    };
    const auto get = [&](const char *suffix) -> Result<const QwenDeviceTensor *> {
        return find_required(suffix);
    };

    if (auto r = get("eh_proj.weight"); !r.ok()) { return R::failure(r.status); } else { out.eh_proj = *r.value; }
    if (auto r = get("enorm.weight"); !r.ok()) { return R::failure(r.status); } else { out.enorm = *r.value; }
    if (auto r = get("hnorm.weight"); !r.ok()) { return R::failure(r.status); } else { out.hnorm = *r.value; }
    if (auto r = get("shared_head.norm.weight"); !r.ok()) { return R::failure(r.status); } else { out.shared_head_norm = *r.value; }
    if (auto r = get("attn_norm.weight"); !r.ok()) { return R::failure(r.status); } else { out.attn_norm = *r.value; }
    if (auto r = get("post_attention_norm.weight"); !r.ok()) { return R::failure(r.status); } else { out.post_norm = *r.value; }
    if (auto r = get("attn_q.weight"); !r.ok()) { return R::failure(r.status); } else { out.attn_q = *r.value; }
    if (auto r = get("attn_k.weight"); !r.ok()) { return R::failure(r.status); } else { out.attn_k = *r.value; }
    if (auto r = get("attn_v.weight"); !r.ok()) { return R::failure(r.status); } else { out.attn_v = *r.value; }
    if (auto r = get("attn_output.weight"); !r.ok()) { return R::failure(r.status); } else { out.attn_output = *r.value; }
    if (auto r = get("attn_q_norm.weight"); !r.ok()) { return R::failure(r.status); } else { out.attn_q_norm = *r.value; }
    if (auto r = get("attn_k_norm.weight"); !r.ok()) { return R::failure(r.status); } else { out.attn_k_norm = *r.value; }
    if (auto r = get("ffn_gate.weight"); !r.ok()) { return R::failure(r.status); } else { out.ffn_gate = *r.value; }
    if (auto r = get("ffn_up.weight"); !r.ok()) { return R::failure(r.status); } else { out.ffn_up = *r.value; }
    if (auto r = get("ffn_down.weight"); !r.ok()) { return R::failure(r.status); } else { out.ffn_down = *r.value; }

    // Optional dedicated embedding / LM head (canonical NextN extras).
    const auto *embd = weights.find(blk + ".token_embd.weight");
    if (embd != nullptr) {
        if (auto s = check_matrix(*embd, g.hidden, g.vocab, out.format); !s.ok()) {
            return R::failure(std::move(s));
        }
        out.embd = embd;
        out.dedicated_embeddings = true;
    } else {
        out.embd = &target_embedding;
    }
    const auto *head = weights.find(blk + ".output.weight");
    if (head != nullptr) {
        if (auto s = check_matrix(*head, g.hidden, g.vocab, out.format); !s.ok()) {
            return R::failure(std::move(s));
        }
        out.head = head;
        out.dedicated_head = true;
    } else {
        // Tied LM head: the trunk's tied embedding (the real model head).
        out.head = &target_embedding;
    }

    // Frozen-geometry shape validation (fail closed, no adaptation).
    if (auto s = check_matrix(*out.eh_proj, 2u * g.hidden, g.hidden, out.format); !s.ok()) { return R::failure(std::move(s)); }
    if (auto s = check_dense(*out.enorm, g.hidden); !s.ok()) { return R::failure(std::move(s)); }
    if (auto s = check_dense(*out.hnorm, g.hidden); !s.ok()) { return R::failure(std::move(s)); }
    if (auto s = check_dense(*out.shared_head_norm, g.hidden); !s.ok()) { return R::failure(std::move(s)); }
    if (auto s = check_dense(*out.attn_norm, g.hidden); !s.ok()) { return R::failure(std::move(s)); }
    if (auto s = check_dense(*out.post_norm, g.hidden); !s.ok()) { return R::failure(std::move(s)); }
    if (auto s = check_matrix(*out.attn_q, g.hidden, g.q_proj_n, out.format); !s.ok()) { return R::failure(std::move(s)); }
    if (auto s = check_matrix(*out.attn_k, g.hidden, g.kv_proj_n, out.format); !s.ok()) { return R::failure(std::move(s)); }
    if (auto s = check_matrix(*out.attn_v, g.hidden, g.kv_proj_n, out.format); !s.ok()) { return R::failure(std::move(s)); }
    if (auto s = check_matrix(*out.attn_output, g.o_proj_k, g.hidden, out.format); !s.ok()) { return R::failure(std::move(s)); }
    if (auto s = check_dense(*out.attn_q_norm, g.head_dim); !s.ok()) { return R::failure(std::move(s)); }
    if (auto s = check_dense(*out.attn_k_norm, g.head_dim); !s.ok()) { return R::failure(std::move(s)); }
    if (auto s = check_matrix(*out.ffn_gate, g.hidden, g.ffn, out.format); !s.ok()) { return R::failure(std::move(s)); }
    if (auto s = check_matrix(*out.ffn_up, g.hidden, g.ffn, out.format); !s.ok()) { return R::failure(std::move(s)); }
    if (auto s = check_matrix(*out.ffn_down, g.ffn, g.hidden, out.format); !s.ok()) { return R::failure(std::move(s)); }
    return R::success(out);
}

// Host-side draft token sampling: greedy qwen_argmax at temperature == 0;
// otherwise one reproducible counter-based draw for THIS draft step and a
// categorical sample (speculative.h building blocks, tag 2 draft namespace).
[[nodiscard]] Status sample_draft_token(luisa::span<const float> row, const QwenSampleOptions &sample,
                                        uint64_t round, uint64_t index, uint32_t &out) noexcept {
    if (sample.temperature == 0.0f) {
        auto r = qwen_argmax(row);
        if (!r.ok()) { return r.status; }
        out = *r.value;
        return Status::success();
    }
    const double u = qwen_speculative_uniform(sample.seed, round, index, kQwenNextnDraftDrawTag);
    return qwen_categorical_sample(row, sample.temperature, u, out);
}

}// namespace

// ---------------------------------------------------------------------------
// Host-only weight validation (no device).
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenNextnDraft::validate_weights(const Qwen35Model &model,
                                                      const QwenDeviceTensor &target_embedding,
                                                      const QwenDeviceWeights &draft_weights) {
    auto r = resolve_nextn(model, target_embedding, draft_weights);
    if (!r.ok()) { return r.status; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
[[nodiscard]] Result<QwenNextnDraft::Ptr> QwenNextnDraft::create(
    Device &device, Stream &stream, const Qwen35Model &model,
    const QwenDeviceTensor &target_embedding, const QwenDeviceWeights &draft_weights,
    uint32_t max_context, uint32_t chunk) {
    using R = Result<QwenNextnDraft::Ptr>;
    if (!device) { return R::failure(invalid("invalid device")); }
    const auto &g = model.config().geometry;
    if (max_context == 0u || max_context > g.context_length) {
        return R::failure(invalid("max_context must be in [1, model context_length]"));
    }
    if (chunk == 0u) { return R::failure(invalid("chunk must be >= 1")); }
    auto resolved = resolve_nextn(model, target_embedding, draft_weights);
    if (!resolved.ok()) { return R::failure(std::move(resolved.status)); }

    auto *raw = ::new (luisa::allocate_with_allocator<QwenNextnDraft>()) QwenNextnDraft{};
    auto draft = QwenNextnDraft::Ptr{raw};
    auto &e = *draft;
    e._cfg.hidden = static_cast<uint32_t>(g.hidden);
    e._cfg.ffn = static_cast<uint32_t>(g.ffn);
    e._cfg.vocab = static_cast<uint32_t>(g.vocab);
    e._cfg.q_heads = static_cast<uint32_t>(g.q_heads);
    e._cfg.kv_heads = static_cast<uint32_t>(g.kv_heads);
    e._cfg.attn_head_dim = static_cast<uint32_t>(g.head_dim);
    e._cfg.rope_dim = static_cast<uint32_t>(g.rope_dims);
    e._cfg.rope_theta = static_cast<float>(g.rope_theta);
    e._cfg.gdn_qk_heads = static_cast<uint32_t>(g.gdn_qk_heads);
    e._cfg.gdn_v_heads = static_cast<uint32_t>(g.gdn_v_heads);
    e._cfg.gdn_head_dim = static_cast<uint32_t>(g.gdn_head_dim);
    e._cfg.conv_kernel = static_cast<uint32_t>(g.conv_kernel);
    e._cfg.norm_eps = static_cast<float>(g.rms_eps);
    e._cfg.l2_eps = static_cast<float>(g.rms_eps);
    e._cfg.softplus_threshold = 20.0f;
    e._cfg.fp8_group = kFp8GroupSize;
    if (auto s = d::validate_qwen_config(e._cfg); !s.ok()) { return R::failure(std::move(s)); }
    e._max_context = max_context;
    e._chunk = chunk;
    e._format = resolved.value->format;
    e._weights = &draft_weights;
    e._device = device;
    e._embd = resolved.value->embd;
    e._head = resolved.value->head;
    e._dedicated_embeddings = resolved.value->dedicated_embeddings;
    e._dedicated_head = resolved.value->dedicated_head;
    e._draft.eh_proj = resolved.value->eh_proj;
    e._draft.enorm = resolved.value->enorm;
    e._draft.hnorm = resolved.value->hnorm;
    e._draft.shared_head_norm = resolved.value->shared_head_norm;
    e._draft.attn_norm = resolved.value->attn_norm;
    e._draft.post_norm = resolved.value->post_norm;
    e._draft.attn_q = resolved.value->attn_q;
    e._draft.attn_k = resolved.value->attn_k;
    e._draft.attn_v = resolved.value->attn_v;
    e._draft.attn_output = resolved.value->attn_output;
    e._draft.attn_q_norm = resolved.value->attn_q_norm;
    e._draft.attn_k_norm = resolved.value->attn_k_norm;
    e._draft.ffn_gate = resolved.value->ffn_gate;
    e._draft.ffn_up = resolved.value->ffn_up;
    e._draft.ffn_down = resolved.value->ffn_down;

    if (auto s = e.alloc_state(device); !s.ok()) { return R::failure(std::move(s)); }
    if (auto s = e.alloc_workspace(device); !s.ok()) { return R::failure(std::move(s)); }
    if (auto s = e.compile_all(device); !s.ok()) { return R::failure(std::move(s)); }
    // The pinned-warp attention kernels require exactly 32 lanes. Query AFTER
    // the first compiles: some backends only report warp size post-compile.
    e._warp_size = device.compute_warp_size();
    if (e._warp_size != 32u) {
        return R::failure(err(StatusCode::unsupported,
                              "NextN draft DSL kernels require a 32-lane warp/subgroup; device reports " +
                                  luisa::string{std::to_string(e._warp_size)}));
    }
    if (auto s = e.reset(stream); !s.ok()) { return R::failure(std::move(s)); }
    return R::success(std::move(draft));
}

[[nodiscard]] Status QwenNextnDraft::alloc_state(Device &device) {
    const uint64_t kv_elems = static_cast<uint64_t>(_max_context) * _cfg.attn_kv_n();
    if (kv_elems > 0xFFFFFFFFull) {
        return invalid("draft KV element count exceeds the uint32 kernel envelope");
    }
    auto k = device.create_buffer<float>(static_cast<size_t>(kv_elems));
    auto v = device.create_buffer<float>(static_cast<size_t>(kv_elems));
    if (!k || !v) { return err(StatusCode::out_of_memory, "draft KV cache allocation failed"); }
    _k_cache = std::move(k);
    _v_cache = std::move(v);
    _cache_bytes = 2u * kv_elems * 4u;
    return Status::success();
}

[[nodiscard]] Status QwenNextnDraft::alloc_workspace(Device &device) {
    const uint64_t C = _chunk;
    const uint64_t hidden = _cfg.hidden;
    const uint64_t ffn = _cfg.ffn;
    const uint64_t max_proj = std::max<uint64_t>({_cfg.attn_q_proj_n(), _cfg.attn_kv_n(), ffn, hidden});
    const uint64_t q_elems = C * _cfg.q_heads * _cfg.attn_head_dim;
    const uint64_t kv_elems = C * _cfg.attn_kv_n();
    const uint64_t vocab = _cfg.vocab;
    const uint64_t counts[] = {C, C * hidden, 2u * C * hidden, C * max_proj, C * ffn,
                               q_elems, kv_elems, vocab, hidden, C * ffn};
    for (auto c : counts) {
        if (c > 0xFFFFFFFFull) {
            return invalid("draft workspace element count exceeds uint32 (reduce chunk)");
        }
    }
    _workspace_bytes = 0;
    const auto mk_f = [&](uint64_t n) -> Buffer<float> {
        auto b = device.create_buffer<float>(static_cast<size_t>(n));
        if (b) { _workspace_bytes += n * 4u; }
        return b;
    };
    const auto mk_u = [&](uint64_t n) -> Buffer<uint32_t> {
        auto b = device.create_buffer<uint32_t>(static_cast<size_t>(n));
        if (b) { _workspace_bytes += n * 4u; }
        return b;
    };
    _ids = mk_u(C);
    _ids_host = mk_u(C);
    _positions = [&] {
        auto b = device.create_buffer<int32_t>(static_cast<size_t>(C));
        if (b) { _workspace_bytes += C * 4u; }
        return b;
    }();
    _positions_host = [&] {
        auto b = device.create_buffer<int32_t>(static_cast<size_t>(C));
        if (b) { _workspace_bytes += C * 4u; }
        return b;
    }();
    // Host staging spans the whole draft context: one propose() stages P
    // history rows + draft_depth step rows, all in DISTINCT slots.
    _host_ids.resize(_max_context);
    _host_pos.resize(_max_context);
    _emb = mk_f(C * hidden);
    _emb_n = mk_f(C * hidden);
    _h_n = mk_f(C * hidden);
    _fused = mk_f(2u * C * hidden);
    _hidden = mk_f(C * hidden);
    _normed = mk_f(C * hidden);
    _residual = mk_f(C * hidden);
    _attn_proj_q = mk_f(C * _cfg.attn_q_proj_n());
    _attn_proj_k = mk_f(C * _cfg.attn_kv_n());
    _attn_proj_v = mk_f(C * _cfg.attn_kv_n());
    _attn_q = mk_f(q_elems);
    _attn_gate = mk_f(q_elems);
    _attn_k = mk_f(kv_elems);
    _attn_v = mk_f(kv_elems);
    _attn_out = mk_f(q_elems);
    _ffn_gate = mk_f(C * ffn);
    _ffn_up = mk_f(C * ffn);
    _ffn_act = mk_f(C * ffn);
    _feedback = mk_f(hidden);
    _logits = mk_f(vocab);
    _dummy_f = mk_f(1u);
    _dummy_u = mk_u(1u);
    if (!_ids || !_ids_host || !_positions || !_positions_host || !_emb || !_emb_n || !_h_n ||
        !_fused || !_hidden || !_normed || !_residual || !_attn_proj_q || !_attn_proj_k ||
        !_attn_proj_v || !_attn_q || !_attn_gate || !_attn_k || !_attn_v || !_attn_out ||
        !_ffn_gate || !_ffn_up || !_ffn_act || !_feedback || !_logits || !_dummy_f || !_dummy_u) {
        return err(StatusCode::out_of_memory, "draft workspace allocation failed");
    }
    return Status::success();
}

[[nodiscard]] Status QwenNextnDraft::compile_all(Device &device) {
    _shader_count = 0;
    // fast_math DISABLED for numerical fidelity (FP32 norms/softmax).
    compute::ShaderOption opt;
    opt.enable_fast_math = false;
    const auto compile1 = [&](auto kernel, auto &shader) -> Status {
        shader = device.compile(kernel, opt);
        if (!shader || !shader.compile_ok()) {
            return err(StatusCode::internal_error, "NextN draft shader compile failed");
        }
        _shader_count++;
        return Status::success();
    };
    if (auto s = compile1(d::make_qwen_embedding_kernel(_format), _embedding); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_rmsnorm_kernel(d::QwenNormEpilogue::plain), _rmsnorm_plain); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_rmsnorm_kernel(d::QwenNormEpilogue::snapshot), _rmsnorm_snapshot); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gemv_kernel(_format, d::QwenMatmulEpilogue::none), _gemv); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gemv_kernel(_format, d::QwenMatmulEpilogue::residual), _gemv_residual); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gemm_kernel(_format, d::QwenMatmulEpilogue::none), _gemm); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gemm_kernel(_format, d::QwenMatmulEpilogue::residual), _gemm_residual); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_swiglu_split_kernel(), _swiglu); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_zero_fill_kernel(), _zero_fill); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_copy_u_kernel(), _copy_u); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_copy_i_kernel(), _copy_i); !s.ok()) { return s; }
    if (auto s = compile1(make_nextn_concat_half_kernel(), _concat_half); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_attn_prepare_kernel(true), _attn_prepare); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_attn_prefill_kernel(), _attn_prefill); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_attn_decode_kernel(), _attn_decode); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenNextnDraft::reset(Stream &stream) {
    // No queued command may still observe the K/V cache when it is cleared.
    stream.synchronize();
    auto batch = QwenCommandBatch{stream, nullptr, QwenBenchmarkPhase::reset, 0u, 0u};
    const auto zero = [&](Buffer<float> &buf) -> Status {
        const auto n = u32(buf.size(), "reset buffer size");
        if (!n.ok()) { return n.status; }
        auto plan = d::plan_qwen_threads(buf.size());
        if (!plan.ok()) { return plan.status; }
        if (!plan.value->empty()) {
            batch.annotate("nextn.reset_cache");
            batch << _zero_fill(buf.view(), *n.value).dispatch(d::qwen_dispatch_threads(*plan.value));
        }
        return Status::success();
    };
    if (auto s = zero(_k_cache); !s.ok()) { return s; }
    if (auto s = zero(_v_cache); !s.ok()) { return s; }
    batch.flush(true);
    return Status::success();
}

// ---------------------------------------------------------------------------
// Scalar staging (persistent host slots + device bounce copies; NEVER
// copy_from directly into a kernel-read buffer that a later chunk rewrites).
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenNextnDraft::stage_ids(QwenCommandBatch &batch,
                                               luisa::span<const uint32_t> ids) {
    const auto rows = static_cast<uint32_t>(ids.size());
    if (static_cast<uint64_t>(_staged) + rows > _host_ids.size()) {
        return invalid("staging overflow: draft token rows");
    }
    const uint32_t base = _staged;
    for (uint32_t t = 0; t < rows; t++) { _host_ids[base + t] = ids[t]; }
    batch.annotate("nextn.ids_upload");
    batch << _ids_host.view(0, rows).copy_from(luisa::span<const uint32_t>{_host_ids.data() + base, rows});
    auto plan = d::plan_qwen_threads(rows);
    if (!plan.ok()) { return plan.status; }
    if (!plan.value->empty()) {
        batch.annotate("nextn.ids_copy");
        batch << _copy_u(_ids_host.view(), _ids.view(), rows).dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    _staged += rows;
    return Status::success();
}

[[nodiscard]] Status QwenNextnDraft::stage_positions(QwenCommandBatch &batch, uint32_t rows, uint32_t pos0) {
    // Distinct absolute-offset slot per chunk/step within one propose();
    // the device _positions buffer is rewritten per chunk in stream order.
    if (static_cast<uint64_t>(pos0) + rows > _host_pos.size()) {
        return invalid("staging overflow: draft position rows");
    }
    for (uint32_t t = 0; t < rows; t++) { _host_pos[pos0 + t] = static_cast<int32_t>(pos0 + t); }
    batch.annotate("nextn.positions_upload");
    batch << _positions_host.view(0, rows).copy_from(luisa::span<const int32_t>{_host_pos.data() + pos0, rows});
    auto plan = d::plan_qwen_threads(rows);
    if (!plan.ok()) { return plan.status; }
    if (!plan.value->empty()) {
        batch.annotate("nextn.positions_copy");
        batch << _copy_i(_positions_host.view(), _positions.view(), rows).dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Packed projection (GEMV for one row, GEMM otherwise).
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenNextnDraft::project(QwenCommandBatch &batch, const BufferView<float> &x,
                                             const QwenDeviceTensor &w, const BufferView<float> &y,
                                             uint32_t rows, const Buffer<float> *residual,
                                             luisa::string_view label) {
    const uint32_t k = static_cast<uint32_t>(w.dims[0]);
    const uint32_t n = static_cast<uint32_t>(w.dims[1]);
    const uint32_t group = _cfg.fp8_group;
    const auto scales = w.scales ? w.scales.view() : _dummy_f.view();
    const auto res = residual != nullptr ? residual->view() : _dummy_f.view();
    batch.annotate(label);
    if (rows == 1u) {
        auto plan = d::plan_qwen_gemv_dispatch(1u, n, k);
        if (!plan.ok()) { return plan.status; }
        if (plan.value->empty()) { return Status::success(); }
        if (residual != nullptr) {
            batch << _gemv_residual(x, w.packed.view(), scales, res, y, 1u, n, k, group)
                         .dispatch(d::qwen_dispatch_threads(*plan.value));
        } else {
            batch << _gemv(x, w.packed.view(), scales, res, y, 1u, n, k, group)
                         .dispatch(d::qwen_dispatch_threads(*plan.value));
        }
    } else {
        auto plan = d::plan_qwen_gemm_dispatch(rows, n, k);
        if (!plan.ok()) { return plan.status; }
        if (plan.value->empty()) { return Status::success(); }
        if (residual != nullptr) {
            batch << _gemm_residual(x, w.packed.view(), scales, res, y, rows, n, k, group,
                                    plan.value->padded_cols)
                         .dispatch(plan.value->total_x, plan.value->total_y);
        } else {
            batch << _gemm(x, w.packed.view(), scales, res, y, rows, n, k, group,
                           plan.value->padded_cols)
                         .dispatch(plan.value->total_x, plan.value->total_y);
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// fused = eh_proj(concat(enorm(emb), hnorm(H))) into _hidden [rows, hidden].
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenNextnDraft::fused_input(QwenCommandBatch &batch, uint32_t rows,
                                                 const BufferView<float> &h) {
    const uint32_t hidden = _cfg.hidden;
    const uint32_t vocab = _cfg.vocab;
    // emb = embedding(ids) [rows, hidden]
    batch.annotate("nextn.embedding");
    {
        auto plan = d::plan_qwen_embedding_dispatch(rows, hidden);
        if (!plan.ok()) { return plan.status; }
        const auto scales = _embd->scales ? _embd->scales.view() : _dummy_f.view();
        batch << _embedding(_ids.view(), _embd->packed.view(), scales, _emb.view(),
                            rows, hidden, vocab, _cfg.fp8_group, plan.value->padded_cols)
                  .dispatch(plan.value->total_x, plan.value->total_y);
    }
    auto np = d::plan_qwen_rmsnorm_dispatch(rows, hidden);
    if (!np.ok()) { return np.status; }
    // enorm(emb) -> _emb_n ; hnorm(H) -> _h_n (direct pre-added GGUF weights).
    batch.annotate("nextn.enorm");
    batch << _rmsnorm_plain(_emb.view(), _draft.enorm->f32.view(), _dummy_f.view(),
                            _emb_n.view(), _dummy_f.view(), rows, hidden, _cfg.norm_eps)
              .dispatch(d::qwen_dispatch_threads(*np.value));
    batch.annotate("nextn.hnorm");
    batch << _rmsnorm_plain(h, _draft.hnorm->f32.view(), _dummy_f.view(),
                            _h_n.view(), _dummy_f.view(), rows, hidden, _cfg.norm_eps)
              .dispatch(d::qwen_dispatch_threads(*np.value));
    // concat([emb_n | h_n]) -> _fused [rows, 2*hidden]
    auto cp = d::plan_qwen_threads_2d(rows, 2ull * hidden);
    if (!cp.ok()) { return cp.status; }
    batch.annotate("nextn.concat");
    batch << _concat_half(_emb_n.view(), _h_n.view(), _fused.view(),
                          rows, hidden, cp.value->padded_cols)
              .dispatch(cp.value->total_x, cp.value->total_y);
    // eh_proj: [rows, 2*hidden] @ [2*hidden, hidden] -> _hidden
    return project(batch, _fused.view(), *_draft.eh_proj, _hidden.view(), rows, nullptr, "nextn.eh_proj");
}

// ---------------------------------------------------------------------------
// One full-attention decoder block (+FFN) over the draft's own K/V cache.
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenNextnDraft::block(QwenCommandBatch &batch, uint32_t rows, uint32_t pos0) {
    const uint32_t hidden = _cfg.hidden;
    const uint32_t ffn = _cfg.ffn;
    const uint32_t hd = _cfg.attn_head_dim;
    const uint32_t qh = _cfg.q_heads, kvh = _cfg.kv_heads;
    // Snapshot the block input as the residual stream; normed input in one pass.
    auto np = d::plan_qwen_rmsnorm_dispatch(rows, hidden);
    if (!np.ok()) { return np.status; }
    batch.annotate("nextn.input_norm");
    batch << _rmsnorm_snapshot(_hidden.view(), _draft.attn_norm->f32.view(), _dummy_f.view(),
                               _normed.view(), _residual.view(), rows, hidden, _cfg.norm_eps)
              .dispatch(d::qwen_dispatch_threads(*np.value));
    // q/k/v projections.
    if (auto s = project(batch, _normed.view(), *_draft.attn_q, _attn_proj_q.view(), rows, nullptr, "nextn.q_proj"); !s.ok()) { return s; }
    if (auto s = project(batch, _normed.view(), *_draft.attn_k, _attn_proj_k.view(), rows, nullptr, "nextn.k_proj"); !s.ok()) { return s; }
    if (auto s = project(batch, _normed.view(), *_draft.attn_v, _attn_proj_v.view(), rows, nullptr, "nextn.v_proj"); !s.ok()) { return s; }
    // Fused split + QK norm + partial RoPE + KV append at pos0..pos0+rows-1.
    if (auto s = d::validate_qwen_attn_prepare_inputs(rows, _cfg, pos0, _max_context, true); !s.ok()) { return s; }
    batch.annotate("nextn.attn_prepare");
    {
        auto plan = d::plan_qwen_attn_prepare_dispatch(rows, _cfg);
        if (!plan.ok()) { return plan.status; }
        const uint32_t q_rows = rows * qh, k_rows = rows * kvh, v_rows = rows * kvh;
        batch << _attn_prepare(_attn_proj_q.view(), _attn_proj_k.view(), _attn_proj_v.view(),
                               _draft.attn_q_norm->f32.view(), _draft.attn_k_norm->f32.view(), _positions.view(),
                               _attn_q.view(), _attn_gate.view(), _attn_k.view(), _attn_v.view(),
                               _k_cache.view(), _v_cache.view(),
                               rows, qh, kvh, hd, _cfg.rope_dim, _cfg.rope_theta, _cfg.norm_eps,
                               pos0, _max_context, q_rows, k_rows, v_rows)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    const uint32_t kv_len = pos0 + rows;// cache now holds [0, kv_len)
    if (rows == 1u) {
        if (auto s = d::validate_qwen_attn_decode_inputs(qh, kvh, hd, kv_len, _max_context); !s.ok()) { return s; }
        auto plan = d::plan_qwen_attn_decode_dispatch(qh);
        if (!plan.ok()) { return plan.status; }
        batch.annotate("nextn.attn_decode");
        batch << _attn_decode(_attn_q.view(), _attn_gate.view(), _k_cache.view(), _v_cache.view(), _attn_out.view(),
                              qh, kvh, hd, _cfg.attn_scale(), kv_len, _max_context)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    } else {
        if (auto s = d::validate_qwen_attn_prefill_inputs(rows, qh, kvh, hd, pos0, kv_len, _max_context); !s.ok()) { return s; }
        auto plan = d::plan_qwen_attn_prefill_dispatch(rows, qh);
        if (!plan.ok()) { return plan.status; }
        batch.annotate("nextn.attn_prefill");
        batch << _attn_prefill(_attn_q.view(), _attn_gate.view(), _k_cache.view(), _v_cache.view(), _attn_out.view(),
                               rows, qh, kvh, hd, _cfg.attn_scale(), pos0, kv_len)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    // o_proj + residual: hidden = residual + attn_out @ W_o.
    if (auto s = project(batch, _attn_out.view(), *_draft.attn_output, _hidden.view(), rows, &_residual, "nextn.o_proj"); !s.ok()) { return s; }
    // Post-attention norm -> FFN.
    batch.annotate("nextn.post_norm");
    batch << _rmsnorm_plain(_hidden.view(), _draft.post_norm->f32.view(), _dummy_f.view(),
                            _normed.view(), _dummy_f.view(), rows, hidden, _cfg.norm_eps)
              .dispatch(d::qwen_dispatch_threads(*np.value));
    if (auto s = project(batch, _normed.view(), *_draft.ffn_gate, _ffn_gate.view(), rows, nullptr, "nextn.gate_proj"); !s.ok()) { return s; }
    if (auto s = project(batch, _normed.view(), *_draft.ffn_up, _ffn_up.view(), rows, nullptr, "nextn.up_proj"); !s.ok()) { return s; }
    {
        const uint32_t count = rows * ffn;
        auto plan = d::plan_qwen_swiglu_dispatch(count);
        if (!plan.ok()) { return plan.status; }
        batch.annotate("nextn.swiglu");
        batch << _swiglu(_ffn_gate.view(), _ffn_up.view(), _ffn_act.view(), count)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    // down + residual: hidden = hidden + down(act) (per-output epilogue; the
    // same-thread read-then-write of one element makes y == residual safe).
    return project(batch, _ffn_act.view(), *_draft.ffn_down, _hidden.view(), rows, &_hidden, "nextn.down_proj");
}

[[nodiscard]] Status QwenNextnDraft::feedback_norm(QwenCommandBatch &batch, uint32_t rows) {
    const uint32_t hidden = _cfg.hidden;
    auto plan = d::plan_qwen_rmsnorm_dispatch(1u, hidden);
    if (!plan.ok()) { return plan.status; }
    // feedback = rmsnorm(last block row, shared_head.norm) -> _feedback.
    const auto last = _hidden.view(static_cast<size_t>(rows - 1u) * hidden, hidden);
    batch.annotate("nextn.feedback_norm");
    batch << _rmsnorm_plain(last, _draft.shared_head_norm->f32.view(), _dummy_f.view(),
                            _feedback.view(), _dummy_f.view(), 1u, hidden, _cfg.norm_eps)
              .dispatch(d::qwen_dispatch_threads(*plan.value));
    return Status::success();
}

[[nodiscard]] Status QwenNextnDraft::lm_head(QwenCommandBatch &batch) {
    // logits[vocab] = feedback[hidden] @ head^T (tied or dedicated head).
    return project(batch, _feedback.view(), *_head, _logits.view(), 1u, nullptr, "nextn.lm_head");
}

// ---------------------------------------------------------------------------
// One draft round: rebuild the prefill over the whole history, then
// draft_depth autoregressive steps (see the header contract).
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenNextnDraft::propose(
    Stream &stream, compute::BufferView<float> target_hidden_history,
    luisa::span<const uint32_t> target_consumed_tokens, uint32_t pending_token,
    uint32_t draft_depth, const QwenSampleOptions &sample, uint64_t round,
    luisa::vector<uint32_t> &drafted_ids_out, QwenSpeculativeLogits &draft_logits_out) {
    const uint32_t hidden = _cfg.hidden;
    const uint32_t vocab = _cfg.vocab;
    // ---- fail-closed input validation ----
    const uint64_t P64 = target_consumed_tokens.size();
    if (P64 == 0u) { return invalid("propose requires a nonempty target history"); }
    if (P64 > 0xFFFFFFFFull) { return invalid("history rows exceed the uint32 envelope"); }
    const uint32_t P = static_cast<uint32_t>(P64);
    if (draft_depth == 0u) { return invalid("draft_depth must be >= 1"); }
    if (static_cast<uint64_t>(P) + draft_depth > _max_context) {
        return invalid("history rows + draft_depth exceed the draft max_context");
    }
    if (pending_token >= vocab) { return invalid("pending_token out of vocabulary range"); }
    for (const auto id : target_consumed_tokens) {
        if (static_cast<uint64_t>(id) >= vocab) { return invalid("target token id out of vocabulary range"); }
    }
    const float temperature = sample.temperature;
    if (!(temperature >= 0.0f) || !std::isfinite(temperature)) {
        return invalid("sample temperature must be finite and >= 0");
    }
    if (target_hidden_history.size() < static_cast<size_t>(P) * hidden) {
        return invalid("target_hidden_history is smaller than [history_rows, hidden]");
    }
    if (!_k_cache || !_v_cache) { return err(StatusCode::internal_error, "draft state is not allocated"); }

    // Host outputs. The logits view aliases _host_logits; resize keeps the
    // storage persistent across rounds (may reallocate on growth; the view
    // returned by the PREVIOUS propose is documented to die at the next one).
    _host_logits.resize(static_cast<size_t>(draft_depth) * vocab);
    drafted_ids_out.clear();
    _staged = 0;

    // Every propose() rebuilds the draft state from scratch (correctness
    // first): clear the K/V cache before recording this round.
    if (auto s = reset(stream); !s.ok()) { return s; }

    // Stage the SHIFTED history ids [done, done+rows): row i consumes
    // target_tokens[i+1], and the last history row consumes pending_token.
    const auto stage_shifted = [&](QwenCommandBatch &batch, uint32_t done, uint32_t rows) -> Status {
        if (static_cast<uint64_t>(_staged) + rows > _host_ids.size()) {
            return invalid("staging overflow: draft token rows");
        }
        const uint32_t base = _staged;
        for (uint32_t t = 0; t < rows; t++) {
            const uint32_t gpos = done + t;
            _host_ids[base + t] = (gpos + 1u < P) ? target_consumed_tokens[gpos + 1u] : pending_token;
        }
        batch.annotate("nextn.ids_upload");
        batch << _ids_host.view(0, rows).copy_from(luisa::span<const uint32_t>{_host_ids.data() + base, rows});
        auto plan = d::plan_qwen_threads(rows);
        if (!plan.ok()) { return plan.status; }
        if (!plan.value->empty()) {
            batch.annotate("nextn.ids_copy");
            batch << _copy_u(_ids_host.view(), _ids.view(), rows).dispatch(d::qwen_dispatch_threads(*plan.value));
        }
        _staged += rows;
        return Status::success();
    };

    // ---- batch A: draft prefill over the entire history + draft step 1 ----
    {
        auto batch = QwenCommandBatch{stream, nullptr, QwenBenchmarkPhase::mtp, 0u, P};
        uint32_t done = 0;
        uint32_t last_rows = 0;
        while (done < P) {
            const uint32_t rows = std::min(_chunk, P - done);
            if (auto s = stage_shifted(batch, done, rows); !s.ok()) { return s; }
            if (auto s = stage_positions(batch, rows, done); !s.ok()) { return s; }
            // H rows for this chunk: history[done, done+rows) packed [rows, hidden].
            const auto h = target_hidden_history.subview(static_cast<size_t>(done) * hidden,
                                                         static_cast<size_t>(rows) * hidden);
            if (auto s = fused_input(batch, rows, h); !s.ok()) { return s; }
            if (auto s = block(batch, rows, done); !s.ok()) { return s; }
            last_rows = rows;
            done += rows;
        }
        // The LAST history row is already draft step 1: it combines target
        // H[P-1] with pending_token at position P-1 and predicts d1. Adding
        // another pending-input step here would duplicate the token and shift
        // every subsequent draft position/cache entry by one.
        if (auto s = feedback_norm(batch, last_rows); !s.ok()) { return s; }
        if (auto s = lm_head(batch); !s.ok()) { return s; }
        batch.flush(false);
        batch.annotate("nextn.draft_logits");
        batch << _logits.view().copy_to(luisa::span<float>{_host_logits.data(), vocab});
        batch.flush(true);// synchronizes: row 0 is host-readable
    }
    {
        uint32_t token = 0u;
        if (auto s = sample_draft_token(luisa::span<const float>{_host_logits.data(), vocab},
                                        sample, round, 0u, token); !s.ok()) {
            return s;
        }
        drafted_ids_out.push_back(token);
    }

    // ---- draft steps 2..draft_depth (id = previous draft token) ----
    for (uint32_t j = 2u; j <= draft_depth; j++) {
        // Step 1 occupied history position P-1, so step 2 is position P.
        const uint32_t pos = P + j - 2u;
        const uint32_t prev = drafted_ids_out.back();
        auto batch = QwenCommandBatch{stream, nullptr, QwenBenchmarkPhase::mtp, pos, 1u};
        if (auto s = stage_ids(batch, luisa::span<const uint32_t>{&prev, 1u}); !s.ok()) { return s; }
        if (auto s = stage_positions(batch, 1u, pos); !s.ok()) { return s; }
        if (auto s = fused_input(batch, 1u, _feedback.view()); !s.ok()) { return s; }
        if (auto s = block(batch, 1u, pos); !s.ok()) { return s; }
        if (auto s = feedback_norm(batch, 1u); !s.ok()) { return s; }
        if (auto s = lm_head(batch); !s.ok()) { return s; }
        batch.flush(false);
        batch.annotate("nextn.draft_logits");
        batch << _logits.view().copy_to(
            luisa::span<float>{_host_logits.data() + static_cast<size_t>(j - 1u) * vocab, vocab});
        batch.flush(true);// synchronizes before the host samples step j
        uint32_t token = 0u;
        if (auto s = sample_draft_token(
                luisa::span<const float>{_host_logits.data() + static_cast<size_t>(j - 1u) * vocab, vocab},
                sample, round, j - 1u, token); !s.ok()) {
            return s;
        }
        drafted_ids_out.push_back(token);
    }

    draft_logits_out = QwenSpeculativeLogits{_host_logits.data(), draft_depth, vocab, vocab};
    return Status::success();
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------
[[nodiscard]] QwenNextnDraftInfo QwenNextnDraft::info() const noexcept {
    QwenNextnDraftInfo i;
    i.weight_storage = _format == d::QwenWeightFormat::bf16 ? WeightStorage::bf16_packed
                                                            : WeightStorage::fp8_packed;
    i.dedicated_embeddings = _dedicated_embeddings;
    i.dedicated_head = _dedicated_head;
    i.max_context = _max_context;
    i.chunk = _chunk;
    i.cache_bytes = _cache_bytes;
    i.workspace_bytes = _workspace_bytes;
    i.compiled_shader_count = _shader_count;
    return i;
}

}// namespace luisa::qwen
