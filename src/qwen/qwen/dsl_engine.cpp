// Qwen3.5-4B complete ordinary-DSL executor -- implementation.
// See dsl_engine.h for the schedule/state/ABI contract. All device work goes
// through the dsl_ops.h kernel factories; this file only plans, validates,
// allocates, compiles once, and records annotated QwenCommandBatch commands.
#include "qwen/dsl_engine.h"

#include <cmath>
#include <cstdio>

#include <luisa/core/logging.h>

namespace luisa::qwen {

using compute::Buffer;
using compute::BufferView;
using compute::CommandList;
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

// Checked uint64 -> uint32.
[[nodiscard]] Result<uint32_t> u32(uint64_t v, const char *what) {
    if (v > 0xFFFFFFFFull) {
        return Result<uint32_t>::failure(invalid(luisa::string{what} + " exceeds uint32"));
    }
    return Result<uint32_t>::success(static_cast<uint32_t>(v));
}

// A matrix weight must be packed (bf16/fp8) with dims [K, N].
[[nodiscard]] Status check_matrix(const QwenDeviceTensor &t, uint64_t k, uint64_t n,
                                  dsl::QwenWeightFormat format) {
    const auto want = format == dsl::QwenWeightFormat::bf16 ? WeightStorage::bf16_packed
                                                            : WeightStorage::fp8_packed;
    if (t.storage != want) {
        return invalid("weight storage does not match the engine variant (no substitution)", t.name);
    }
    if (t.dims.size() != 2u || t.dims[0] != k || t.dims[1] != n) {
        return invalid("weight matrix shape mismatch", t.name);
    }
    uint64_t elements = 0u;
    if (!checked_mul(k, n, elements) || t.element_count != elements) {
        return invalid("weight matrix element count mismatch/overflow", t.name);
    }
    const uint64_t codes_per_word = format == dsl::QwenWeightFormat::bf16 ? 2u : 4u;
    const uint64_t expected_words = (elements + codes_per_word - 1u) / codes_per_word;
    if (!t.packed || t.packed.size() != expected_words) {
        return invalid("weight packed buffer size mismatch", t.name);
    }
    if (format == dsl::QwenWeightFormat::fp8_e4m3) {
        const uint64_t expect = (k + 63u) / 64u;// ceil(K/64)
        uint64_t scale_count = 0u;
        if (!t.scales || t.groups_per_row != expect ||
            !checked_mul(n, expect, scale_count) || t.scales.size() != scale_count) {
            return invalid("fp8 weight scale geometry/buffer size mismatch", t.name);
        }
    }
    return Status::success();
}

[[nodiscard]] Status check_dense(const QwenDeviceTensor &t, uint64_t count) {
    if (t.storage != WeightStorage::f32_dense) {
        return invalid("sensitive tensor must be dense F32", t.name);
    }
    if (t.element_count != count || !t.f32 || t.f32.size() != count) {
        return invalid("dense F32 tensor size/buffer mismatch", t.name);
    }
    return Status::success();
}

}// namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
QwenDslEngine::~QwenDslEngine() noexcept {
    if (_stream) { _stream.synchronize(); }
}

[[nodiscard]] Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>> QwenDslEngine::create(
    Device &device, Stream &stream, const Qwen35Model &model,
    const QwenDeviceWeights &weights, const QwenEngineOptions &options) {
    if (options.kernel != QwenKernelKind::dsl) {
        return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(
            invalid("QwenDslEngine requires QwenKernelKind::dsl"));
    }
    if (!device) {
        return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(invalid("invalid device"));
    }
    const auto &g = model.config().geometry;
    if (options.max_context == 0u || options.max_context > g.context_length) {
        return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(
            invalid("max_context must be in [1, model context_length]"));
    }
    if (options.prefill_chunk == 0u) {
        return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(invalid("prefill_chunk must be >= 1"));
    }

    // Allocate with Luisa's allocator (mimalloc-backed); Deleter frees with
    // the same allocator. The private constructor is accessible here (member
    // function), unlike new_with_allocator's std::construct_at.
    auto *raw = ::new (luisa::allocate_with_allocator<QwenDslEngine>()) QwenDslEngine{};
    auto engine = luisa::unique_ptr<QwenDslEngine, QwenDslEngine::Deleter>{raw};
    auto &e = *engine;
    // Freeze geometry into the dsl config.
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
    if (auto s = d::validate_qwen_config(e._cfg); !s.ok()) {
        return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(std::move(s));
    }
    e._layers = static_cast<uint32_t>(g.layers);
    e._max_context = options.max_context;
    // Cap the chunk so activation scratch stays bounded and the GDN recurrent
    // kernel's per-launch token count stays inside the uint32 envelope.
    e._prefill_chunk = options.prefill_chunk;
    e._benchmark = options.benchmark;
    e._gdn_wy_ut = options.dsl_gdn_wy_ut;
    e._decode_specialized = options.dsl_decode_specialized;
    e._kv_page_bytes = options.kv_page_bytes;
    e._weights = &weights;
    e._device = device;
    e._stream = stream;

    if (auto s = e.resolve_weights(model, weights); !s.ok()) {
        return Result<luisa::unique_ptr<QwenEngine, Deleter>>::failure(std::move(s));
    }
    if (options.dsl_cuda_graph && !e._benchmark &&
        e._format == d::QwenWeightFormat::fp8_e4m3 &&
        device.backend_name() == luisa::string_view{"cuda"}) {
        e._cuda_graph_ext = device.extension<compute::CudaGraphExt>();
    }
    if (auto s = e.alloc_state(device); !s.ok()) {
        return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(std::move(s));
    }
    if (auto s = e.alloc_workspace(device); !s.ok()) {
        return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(std::move(s));
    }
    if (auto s = e.compile_all(device); !s.ok()) {
        return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(std::move(s));
    }
    // Warp capability: the pinned-warp kernels require exactly 32 lanes.
    // NOTE: on some backends compute_warp_size() is only valid after the first
    // shader compile, so query it AFTER compile_all.
    e._warp_size = device.compute_warp_size();
    if (e._warp_size != 32u) {
        return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(
            err(StatusCode::unsupported,
                "DSL Qwen kernels require a 32-lane warp/subgroup; device reports " +
                    luisa::string{std::to_string(e._warp_size)}));
    }
    // Reset state so the engine starts in a clean session.
    if (auto s = e.reset(stream); !s.ok()) {
        return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(std::move(s));
    }
    // Capability record (opt_plan/04-backend-accelerators.md section 7). Probed
    // once, AFTER compile_all so compute_warp_size() reports the real value, and
    // after alloc_state so the KV page geometry can be copied out of the live
    // allocator. Logged exactly once here: every accelerator decision downstream
    // reads this struct instead of re-querying (or guessing from the backend
    // name), and a benchmark log stays self-describing.
    e._caps = qwen_probe_backend_caps(device);
    qwen_fill_kv_page_caps(e._caps, e._kv_cache);
    qwen_log_backend_caps(e._caps);
    return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::success(std::move(engine));
}

[[nodiscard]] Status QwenDslEngine::resolve_weights(const Qwen35Model &model,
                                                    const QwenDeviceWeights &weights) {
    const auto &g = model.config().geometry;
    // Determine the uniform packed storage from the token embedding matrix.
    const auto *embd = weights.find(QwenTensorRole::token_embd);
    if (embd == nullptr) { return invalid("token_embd weight missing"); }
    if (embd->storage == WeightStorage::bf16_packed) {
        _format = dsl::QwenWeightFormat::bf16;
        _storage = WeightStorage::bf16_packed;
    } else if (embd->storage == WeightStorage::fp8_packed) {
        _format = dsl::QwenWeightFormat::fp8_e4m3;
        _storage = WeightStorage::fp8_packed;
    } else {
        return invalid("token_embd must be packed bf16 or fp8 (dense F32 embedding unsupported)");
    }
    _token_embd = embd;
    if (auto s = check_matrix(*embd, g.hidden, g.vocab, _format); !s.ok()) { return s; }
    const auto *onorm = weights.find(QwenTensorRole::output_norm);
    if (onorm == nullptr) { return invalid("output_norm weight missing"); }
    if (auto s = check_dense(*onorm, g.hidden); !s.ok()) { return s; }
    _output_norm = onorm;

    _layer_weights.resize(static_cast<size_t>(g.layers));
    // Cached per-layer command annotations (stable storage for annotate()).
    const auto lname = [](luisa::string_view op, uint64_t l) {
        return luisa::string{op} + ".l" + luisa::string{std::to_string(l)};
    };
    for (uint64_t l = 0; l < g.layers; l++) {
        auto &lw = _layer_weights[static_cast<size_t>(l)];
        lw.p_input_norm = lname("input_norm", l);
        lw.p_post_norm = lname("post_attn_norm", l);
        lw.p_swiglu = lname("swiglu", l);
        if (is_full_attention_layer(g, l)) {
            lw.p_attn_prepare = lname("attn_prepare_rope_kv_append", l);
            lw.p_attn_core = lname("attn_causal", l);
        } else {
            lw.p_conv = lname("gdn_conv_silu", l);
            lw.p_gates = lname("gdn_gates", l);
            lw.p_qk_norm = lname("gdn_qk_l2norm", l);
            lw.p_recurrent = lname("gdn_recurrent", l);
            lw.p_chunk = lname("gdn_wy_ut_chunk64", l);
            lw.p_gated_norm = lname("gdn_gated_norm", l);
        }
    }
    _full_slot.assign(static_cast<size_t>(g.layers), UINT32_MAX);
    _gdn_slot.assign(static_cast<size_t>(g.layers), UINT32_MAX);
    uint32_t full_slot = 0, gdn_slot = 0;
    for (uint64_t l = 0; l < g.layers; l++) {
        auto &lw = _layer_weights[static_cast<size_t>(l)];
        lw.full_attention = is_full_attention_layer(g, l);
        const auto get = [&](QwenTensorRole role) -> const QwenDeviceTensor * {
            return weights.find(role, l);
        };
        // Shared dense norms + packed FFN matrices.
        lw.attn_norm = get(QwenTensorRole::attn_norm);
        lw.post_norm = get(QwenTensorRole::post_attention_norm);
        lw.ffn_gate = get(QwenTensorRole::ffn_gate);
        lw.ffn_up = get(QwenTensorRole::ffn_up);
        lw.ffn_down = get(QwenTensorRole::ffn_down);
        if (lw.attn_norm == nullptr || lw.post_norm == nullptr || lw.ffn_gate == nullptr ||
            lw.ffn_up == nullptr || lw.ffn_down == nullptr) {
            return invalid("shared layer weight missing", luisa::string{std::to_string(l)});
        }
        if (auto s = check_dense(*lw.attn_norm, g.hidden); !s.ok()) { return s; }
        if (auto s = check_dense(*lw.post_norm, g.hidden); !s.ok()) { return s; }
        if (auto s = check_matrix(*lw.ffn_gate, g.hidden, g.ffn, _format); !s.ok()) { return s; }
        if (auto s = check_matrix(*lw.ffn_up, g.hidden, g.ffn, _format); !s.ok()) { return s; }
        if (auto s = check_matrix(*lw.ffn_down, g.ffn, g.hidden, _format); !s.ok()) { return s; }
        if (lw.full_attention) {
            lw.attn_q = get(QwenTensorRole::attn_q);
            lw.attn_k = get(QwenTensorRole::attn_k);
            lw.attn_v = get(QwenTensorRole::attn_v);
            lw.attn_output = get(QwenTensorRole::attn_output);
            lw.attn_q_norm = get(QwenTensorRole::attn_q_norm);
            lw.attn_k_norm = get(QwenTensorRole::attn_k_norm);
            if (lw.attn_q == nullptr || lw.attn_k == nullptr || lw.attn_v == nullptr ||
                lw.attn_output == nullptr || lw.attn_q_norm == nullptr || lw.attn_k_norm == nullptr) {
                return invalid("attention layer weight missing", luisa::string{std::to_string(l)});
            }
            if (auto s = check_matrix(*lw.attn_q, g.hidden, g.q_proj_n, _format); !s.ok()) { return s; }
            if (auto s = check_matrix(*lw.attn_k, g.hidden, g.kv_proj_n, _format); !s.ok()) { return s; }
            if (auto s = check_matrix(*lw.attn_v, g.hidden, g.kv_proj_n, _format); !s.ok()) { return s; }
            if (auto s = check_matrix(*lw.attn_output, g.o_proj_k, g.hidden, _format); !s.ok()) { return s; }
            if (auto s = check_dense(*lw.attn_q_norm, g.head_dim); !s.ok()) { return s; }
            if (auto s = check_dense(*lw.attn_k_norm, g.head_dim); !s.ok()) { return s; }
            _full_slot[static_cast<size_t>(l)] = full_slot++;
        } else {
            lw.gdn_qkv = get(QwenTensorRole::gdn_qkv);
            lw.gdn_gate = get(QwenTensorRole::gdn_gate);
            lw.gdn_alpha = get(QwenTensorRole::gdn_alpha);
            lw.gdn_beta = get(QwenTensorRole::gdn_beta);
            lw.gdn_out = get(QwenTensorRole::gdn_out);
            lw.gdn_conv1d = get(QwenTensorRole::gdn_conv1d);
            lw.gdn_a = get(QwenTensorRole::gdn_a);
            lw.gdn_dt_bias = get(QwenTensorRole::gdn_dt_bias);
            lw.gdn_norm = get(QwenTensorRole::gdn_norm);
            if (lw.gdn_qkv == nullptr || lw.gdn_gate == nullptr || lw.gdn_alpha == nullptr ||
                lw.gdn_beta == nullptr || lw.gdn_out == nullptr || lw.gdn_conv1d == nullptr ||
                lw.gdn_a == nullptr || lw.gdn_dt_bias == nullptr || lw.gdn_norm == nullptr) {
                return invalid("GDN layer weight missing", luisa::string{std::to_string(l)});
            }
            if (auto s = check_matrix(*lw.gdn_qkv, g.hidden, g.gdn_qkv_n, _format); !s.ok()) { return s; }
            if (auto s = check_matrix(*lw.gdn_gate, g.hidden, g.gdn_z_n, _format); !s.ok()) { return s; }
            if (auto s = check_matrix(*lw.gdn_alpha, g.hidden, g.gdn_gate_n, _format); !s.ok()) { return s; }
            if (auto s = check_matrix(*lw.gdn_beta, g.hidden, g.gdn_gate_n, _format); !s.ok()) { return s; }
            if (auto s = check_matrix(*lw.gdn_out, g.gdn_out_k, g.hidden, _format); !s.ok()) { return s; }
            if (auto s = check_dense(*lw.gdn_conv1d, g.conv_kernel * g.gdn_qkv_n); !s.ok()) { return s; }
            if (auto s = check_dense(*lw.gdn_a, g.gdn_v_heads); !s.ok()) { return s; }
            if (auto s = check_dense(*lw.gdn_dt_bias, g.gdn_v_heads); !s.ok()) { return s; }
            if (auto s = check_dense(*lw.gdn_norm, g.gdn_head_dim); !s.ok()) { return s; }
            _gdn_slot[static_cast<size_t>(l)] = gdn_slot++;
        }
    }
    return Status::success();
}

[[nodiscard]] Status QwenDslEngine::alloc_state(Device &device) {
    const uint64_t kv_n = _cfg.attn_kv_n();
    const uint64_t kv_elems = static_cast<uint64_t>(_max_context) * kv_n;
    const uint64_t state_elems = static_cast<uint64_t>(_cfg.gdn_v_heads) *
                                 _cfg.gdn_head_dim * _cfg.gdn_head_dim;
    const uint64_t ring_elems = static_cast<uint64_t>(_cfg.gdn_qkv_n()) * (_cfg.conv_kernel - 1u);
    if (kv_elems > 0xFFFFFFFFull || state_elems > 0xFFFFFFFFull || ring_elems > 0xFFFFFFFFull) {
        return invalid("state element count exceeds the uint32 kernel envelope");
    }
    uint32_t full_count = 0, gdn_count = 0;
    for (uint32_t l = 0; l < _layers; l++) {
        if (_layer_weights[l].full_attention) { full_count++; } else { gdn_count++; }
    }
    _gdn_states.reserve(gdn_count);
    _gdn_rings.reserve(gdn_count);
    if (auto s = _kv_cache.create(device, _stream, _max_context,
                                  static_cast<uint32_t>(kv_n), full_count,
                                  _kv_page_bytes);
        !s.ok()) {
        return s;
    }
    // Preserve the existing configured-capacity metric. Physical sparse
    // residency is exposed separately by QwenKvCacheAllocator and grows with use.
    _cache_bytes = _kv_cache.logical_bytes();
    for (uint32_t i = 0; i < gdn_count; i++) {
        auto st = device.create_buffer<float>(static_cast<size_t>(state_elems));
        auto ring = device.create_buffer<float>(static_cast<size_t>(ring_elems));
        if (!st || !ring) { return err(StatusCode::out_of_memory, "GDN state allocation failed"); }
        _gdn_states.push_back(std::move(st));
        _gdn_rings.push_back(std::move(ring));
        _cache_bytes += (state_elems + ring_elems) * 4u;
    }
    return Status::success();
}

[[nodiscard]] Status QwenDslEngine::alloc_workspace(Device &device) {
    const uint64_t C = _prefill_chunk;
    const uint64_t hidden = _cfg.hidden;
    const uint64_t ffn = _cfg.ffn;
    // Largest projection output width across the whole model.
    const uint64_t max_proj = std::max<uint64_t>({
        _cfg.attn_q_proj_n(), _cfg.attn_kv_n(), _cfg.gdn_qkv_n(), _cfg.gdn_v_n(),
        _cfg.gdn_v_heads, ffn, hidden});
    const uint64_t q_elems = C * _cfg.q_heads * _cfg.attn_head_dim;
    const uint64_t kv_elems = C * _cfg.kv_heads * _cfg.attn_head_dim;
    const uint64_t gdn_qk = C * _cfg.gdn_qk_heads * _cfg.gdn_head_dim;
    const uint64_t gdn_v = C * _cfg.gdn_v_heads * _cfg.gdn_head_dim;
    const uint64_t gdn_conv = C * _cfg.gdn_qkv_n();
    const uint64_t gdn_gate = C * _cfg.gdn_v_heads;
    const uint64_t vocab = _cfg.vocab;
    // Every workspace buffer must fit the uint32 kernel index envelope.
    const uint64_t counts[] = {C, C * hidden, C * max_proj, C * ffn, q_elems, kv_elems,
                               gdn_qk, gdn_v, gdn_conv, gdn_gate, vocab, hidden};
    for (auto c : counts) {
        if (c > 0xFFFFFFFFull) {
            return invalid("workspace element count exceeds uint32 (reduce prefill_chunk)");
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
    // Host staging spans the FULL context: prefill records every chunk into
    // one command list, so each chunk's borrowed span must stay distinct and
    // unmodified until the single commit.
    _host_ids.resize(_max_context);
    _hidden = mk_f(C * hidden);
    _normed = mk_f(C * hidden);
    _residual = mk_f(C * hidden);
    _proj_a = mk_f(C * max_proj);
    _proj_b = mk_f(C * max_proj);
    _proj_c = mk_f(C * max_proj);
    _ffn_gate = mk_f(C * ffn);
    _ffn_up = mk_f(C * ffn);
    _ffn_act = mk_f(C * ffn);
    _attn_q = mk_f(q_elems);
    _attn_gate = mk_f(q_elems);
    _attn_k = mk_f(kv_elems);
    _attn_v = mk_f(kv_elems);
    _attn_out = mk_f(q_elems);
    _attn_proj_q = mk_f(C * _cfg.attn_q_proj_n());
    _attn_proj_k = mk_f(C * _cfg.attn_kv_n());
    _attn_proj_v = mk_f(C * _cfg.attn_kv_n());
    _gdn_conv = mk_f(gdn_conv);
    _gdn_q = mk_f(gdn_qk);
    _gdn_k = mk_f(gdn_qk);
    _gdn_beta = mk_f(gdn_gate);
    _gdn_g = mk_f(gdn_gate);
    _gdn_alpha = mk_f(gdn_gate);
    _gdn_out = mk_f(gdn_v);
    _gdn_w = mk_f(gdn_v);
    _gdn_vnew = mk_f(gdn_v);
    _gdn_normed = mk_f(gdn_v);
    _last_normed = mk_f(hidden);
    _logits = mk_f(vocab);
    _greedy_history = mk_u(_max_context);
    _greedy_values = mk_f(d::qwen_greedy_argmax_blocks);
    _greedy_indices = mk_u(d::qwen_greedy_argmax_blocks);
    _greedy_invalid = mk_u(d::qwen_greedy_argmax_blocks);
    _dummy_f = mk_f(1u);
    _dummy_u = mk_u(1u);
    if (!_ids || !_ids_host || !_hidden || !_normed || !_residual || !_proj_a || !_proj_b ||
        !_proj_c || !_ffn_gate || !_ffn_up || !_ffn_act || !_attn_q || !_attn_gate || !_attn_k ||
        !_attn_v || !_attn_out || !_gdn_conv || !_gdn_q || !_gdn_k || !_gdn_beta || !_gdn_g ||
        !_gdn_out || !_gdn_w || !_gdn_vnew || !_gdn_normed || !_gdn_alpha || !_attn_proj_q || !_attn_proj_k || !_attn_proj_v ||
        !_last_normed || !_logits || !_greedy_history || !_greedy_values ||
        !_greedy_indices || !_greedy_invalid || !_dummy_f || !_dummy_u) {
        return err(StatusCode::out_of_memory, "workspace allocation failed");
    }
    return Status::success();
}

[[nodiscard]] Status QwenDslEngine::compile_all(Device &device) {
    _shader_count = 0;
    // Compile every variant once. Shapes are static; M/rows are runtime args.
    // fast_math is DISABLED for numerical fidelity (FP32 norms/softmax/state).
    compute::ShaderOption opt;
    opt.enable_fast_math = false;
    const auto compile1 = [&](auto kernel, auto &shader) -> Status {
        shader = device.compile(kernel, opt);
        if (!shader || !shader.compile_ok()) {
            return err(StatusCode::internal_error, "DSL shader compile failed");
        }
        _shader_count++;
        return Status::success();
    };
    if (auto s = compile1(d::make_qwen_embedding_kernel(_format), _embedding); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen35_rmsnorm_kernel(d::QwenNormEpilogue::plain, 256u), _rmsnorm_plain); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen35_rmsnorm_kernel(d::QwenNormEpilogue::snapshot, 256u), _rmsnorm_snapshot); !s.ok()) { return s; }
    // BF16 already runs at the DRAM roof with the generic packed loop; the
    // four-stream realization was slower in paired RTX 4080 SUPER A/B runs.
    // FP8 retains it for the exact shared-LUT decode experiment.
    const auto gemv_kernel = [&] {
        return _format == d::QwenWeightFormat::bf16 || !_decode_specialized
                   ? d::make_qwen_gemv_kernel(_format, d::QwenMatmulEpilogue::none)
                   : d::make_qwen35_decode_gemv_kernel(_format, d::QwenMatmulEpilogue::none);
    };
    const auto gemv_residual_kernel = [&] {
        return _format == d::QwenWeightFormat::bf16 || !_decode_specialized
                   ? d::make_qwen_gemv_kernel(_format, d::QwenMatmulEpilogue::residual)
                   : d::make_qwen35_decode_gemv_kernel(_format, d::QwenMatmulEpilogue::residual);
    };
    if (auto s = compile1(gemv_kernel(), _gemv); !s.ok()) { return s; }
    if (auto s = compile1(gemv_residual_kernel(), _gemv_residual); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gemm_kernel(_format, d::QwenMatmulEpilogue::none), _gemm); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gemm_kernel(_format, d::QwenMatmulEpilogue::residual), _gemm_residual); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_swiglu_split_kernel(), _swiglu); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_ffn_decode_kernel(_format), _ffn_decode); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_zero_fill_kernel(), _zero_fill); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_copy_u_kernel(), _copy_u); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_greedy_argmax_partial_kernel(), _greedy_argmax_partial); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_greedy_argmax_finish_kernel(), _greedy_argmax_finish); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_attn_prepare_kernel(true, true), _attn_prepare); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_attn_prefill_kernel(), _attn_prefill); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_attn_decode_kernel(), _attn_decode); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gdn_conv_silu_kernel(), _gdn_conv_sh); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gdn_gates_kernel(), _gdn_gates_sh); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gdn_decode_gates_kernel(_format), _gdn_decode_gates_sh); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gdn_qk_l2norm_kernel(), _gdn_qk_norm_sh); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gdn_recurrent_kernel(), _gdn_recurrent_sh); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen35_gdn_decode4_kernel(), _gdn_decode4_sh); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gdn_chunk_kernel(), _gdn_chunk_sh); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gdn_gated_norm_kernel(), _gdn_gated_norm_sh); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenDslEngine::reset(Stream &stream) {
    // _kv_cache.reset() below drains queued work before any state buffer or
    // host staging span is reused; do not add a duplicate stream sync here.
    // A standalone reset starts a fresh report. prefill clears before calling
    // reset and then appends its own samples, so reset work remains in TTFT.
    if (_benchmark) { _report.clear(); }
    // Build stable annotations once: reset.<kind>.<slot> (per-layer state).
    if (_reset_names.empty()) {
        const auto push = [&](const char *kind, size_t count) {
            for (size_t i = 0; i < count; i++) {
                _reset_names.emplace_back(luisa::string{"reset.zero_"} + kind + "." +
                                          luisa::string{std::to_string(i)});
            }
        };
        push("gdn_state", _gdn_states.size());
        push("gdn_ring", _gdn_rings.size());
    }
    // Decommit the old K/V tiles instead of zeroing the full configured
    // maximum context. reset() waits for prior use and retires heaps through
    // a stream completion callback; subsequent positions are written before read.
    if (auto s = _kv_cache.reset(stream); !s.ok()) { return s; }
    auto batch = QwenCommandBatch{stream, _benchmark ? &_report : nullptr,
                                  QwenBenchmarkPhase::reset, 0u, 0u};
    size_t name_i = 0;
    const auto zero = [&](Buffer<float> &buf) -> Status {
        const auto n = u32(buf.size(), "reset buffer size");
        if (!n.ok()) { return n.status; }
        auto plan = d::plan_qwen_threads(buf.size());
        if (!plan.ok()) { return plan.status; }
        if (!plan.value->empty()) {
            batch.annotate(_reset_names[name_i++]);
            batch << _zero_fill(buf.view(), *n.value).dispatch(d::qwen_dispatch_threads(*plan.value));
        }
        return Status::success();
    };
    for (auto &b : _gdn_states) { if (auto s = zero(b); !s.ok()) { return s; } }
    for (auto &b : _gdn_rings) { if (auto s = zero(b); !s.ok()) { return s; } }
    batch.flush(true);// no-op when empty; single synchronize otherwise
    _position = 0;
    _has_session = false;
    _greedy_next_valid = false;
    return Status::success();
}

// ---------------------------------------------------------------------------
// Projection helper (packed GEMV for 1 row, GEMM otherwise).
// ---------------------------------------------------------------------------
[[nodiscard]] luisa::string QwenDslEngine::proj_name(luisa::string_view label, const QwenDeviceTensor &w) {
    luisa::string s;
    s.reserve(label.size() + 1u + w.name.size());
    s.append(label);
    s.push_back('.');
    s.append(w.name);
    return s;
}

[[nodiscard]] Status QwenDslEngine::project(QwenCommandBatch &batch, const Buffer<float> &x,
                                            const QwenDeviceTensor &w, const Buffer<float> &y,
                                            uint32_t rows, const Buffer<float> *residual,
                                            luisa::string_view label) {
    const uint32_t k = static_cast<uint32_t>(w.dims[0]);
    const uint32_t n = static_cast<uint32_t>(w.dims[1]);
    const uint32_t group = _cfg.fp8_group;
    const auto scales = w.scales ? w.scales.view() : _dummy_f.view();
    const auto res = residual != nullptr ? residual->view() : _dummy_f.view();
    // One annotation per projection: <label>.<GGUF weight name> (includes the
    // layer). Build it only for profiling; normal recording performs no string
    // allocation. The local outlives the immediately following annotate call.
    luisa::string name;
    if (batch.instrumented()) {
        name = proj_name(label, w);
        batch.annotate(name);
    }
    if (rows == 1u) {
        auto plan = _format == d::QwenWeightFormat::bf16 || !_decode_specialized
                        ? d::plan_qwen_gemv_dispatch(1u, n, k)
                        : d::plan_qwen35_decode_gemv_dispatch(n, k, _format);
        if (!plan.ok()) { return plan.status; }
        if (plan.value->empty()) { return Status::success(); }
        if (residual != nullptr) {
            batch << _gemv_residual(x.view(), w.packed.view(), scales, res, y.view(), 1u, n, k, group)
                         .dispatch(d::qwen_dispatch_threads(*plan.value));
        } else {
            batch << _gemv(x.view(), w.packed.view(), scales, res, y.view(), 1u, n, k, group)
                         .dispatch(d::qwen_dispatch_threads(*plan.value));
        }
    } else {
        auto plan = d::plan_qwen_gemm_dispatch(rows, n, k);
        if (!plan.ok()) { return plan.status; }
        if (plan.value->empty()) { return Status::success(); }
        if (residual != nullptr) {
            batch << _gemm_residual(x.view(), w.packed.view(), scales, res, y.view(), rows, n, k, group,
                                    plan.value->padded_cols)
                         .dispatch(plan.value->total_x, plan.value->total_y);
        } else {
            batch << _gemm(x.view(), w.packed.view(), scales, res, y.view(), rows, n, k, group,
                           plan.value->padded_cols)
                         .dispatch(plan.value->total_x, plan.value->total_y);
        }
    }
    return Status::success();
}

// Stage `rows` token ids into the persistent host staging buffer and copy
// them into _ids with a device-side kernel. NEVER use copy_from(span) here:
// it records a borrowed host pointer that is only dereferenced when the
// command encoder commits; a caller-local staging vector would dangle.
[[nodiscard]] Status QwenDslEngine::stage_ids(QwenCommandBatch &batch, luisa::span<const uint32_t> ids) {
    const auto rows = static_cast<uint32_t>(ids.size());
    // Distinct span per chunk: never rewrite a span queued in the same batch.
    if (static_cast<uint64_t>(_staged) + rows > _host_ids.size()) {
        return invalid("staging overflow: token rows");
    }
    const uint32_t base = _staged;
    for (uint32_t t = 0; t < rows; t++) { _host_ids[base + t] = ids[t]; }
    batch.annotate("ids_upload");
    batch << _ids_host.view(0, rows).copy_from(luisa::span<const uint32_t>{_host_ids.data() + base, rows});
    auto plan = d::plan_qwen_threads(rows);
    if (!plan.ok()) { return plan.status; }
    if (!plan.value->empty()) {
        batch.annotate("ids_copy");
        batch << _copy_u(_ids_host.view(), _ids.view(), rows).dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    _staged += rows;
    return Status::success();
}

// ---------------------------------------------------------------------------
// Full attention block.
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenDslEngine::full_attention_block(QwenCommandBatch &batch, uint32_t layer,
                                                         uint32_t rows, uint32_t pos0) {
    const auto &lw = _layer_weights[layer];
    const uint32_t slot = _full_slot[layer];
    const auto kc = _kv_cache.k_view(slot);
    const auto vc = _kv_cache.v_view(slot);
    const uint32_t hd = _cfg.attn_head_dim;
    const uint32_t qh = _cfg.q_heads, kvh = _cfg.kv_heads;
    const uint32_t kv_n = _cfg.attn_kv_n();
    batch.layer(layer);
    // Projections: q (with per-head gate), k, v.
    if (auto s = project(batch, _normed, *lw.attn_q, _attn_proj_q, rows, nullptr, "proj"); !s.ok()) { return s; }
    if (auto s = project(batch, _normed, *lw.attn_k, _attn_proj_k, rows, nullptr, "proj"); !s.ok()) { return s; }
    if (auto s = project(batch, _normed, *lw.attn_v, _attn_proj_v, rows, nullptr, "proj"); !s.ok()) { return s; }
    // Fused split + QK norm + partial RoPE + KV append at pos0..pos0+rows-1.
    // Positions were uploaded ONCE by forward_chunk for the whole chunk.
    if (auto s = d::validate_qwen_attn_prepare_inputs(rows, _cfg, pos0, _max_context, true); !s.ok()) { return s; }
    batch.annotate(lw.p_attn_prepare);
    {
        auto plan = d::plan_qwen_attn_prepare_dispatch(rows, _cfg);
        if (!plan.ok()) { return plan.status; }
        const uint32_t q_rows = rows * qh, k_rows = rows * kvh, v_rows = rows * kvh;
        batch << _attn_prepare(_attn_proj_q.view(), _attn_proj_k.view(), _attn_proj_v.view(),
                            lw.attn_q_norm->f32.view(), lw.attn_k_norm->f32.view(), _ids_host.view().as<int32_t>(),
                            _attn_q.view(), _attn_gate.view(), _attn_k.view(), _attn_v.view(),
                            kc, vc,
                            rows, qh, kvh, hd, _cfg.rope_dim, _cfg.rope_theta, _cfg.norm_eps,
                            pos0, _max_context, q_rows, k_rows, v_rows)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    const uint32_t kv_len = pos0 + rows;// cache now holds [0, kv_len)
    if (rows == 1u) {
        // Decode: one query at absolute position pos0 == kv_len-1.
        if (auto s = d::validate_qwen_attn_decode_inputs(qh, kvh, hd, kv_len, _max_context); !s.ok()) { return s; }
        auto plan = d::plan_qwen_attn_decode_dispatch(qh);
        if (!plan.ok()) { return plan.status; }
        batch.annotate(lw.p_attn_core);
        batch << _attn_decode(_attn_q.view(), _attn_gate.view(), kc, vc, _attn_out.view(),
                           qh, kvh, hd, _cfg.attn_scale(), kv_len, _max_context)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    } else {
        if (auto s = d::validate_qwen_attn_prefill_inputs(rows, qh, kvh, hd, pos0, kv_len, _max_context); !s.ok()) { return s; }
        auto plan = d::plan_qwen_attn_prefill_dispatch(rows, qh);
        if (!plan.ok()) { return plan.status; }
        batch.annotate(lw.p_attn_core);
        batch << _attn_prefill(_attn_q.view(), _attn_gate.view(), kc, vc, _attn_out.view(),
                            rows, qh, kvh, hd, _cfg.attn_scale(), pos0, kv_len)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    // Output projection fused with the residual epilogue:
    // hidden = residual + attn_out @ W_o (no separate residual_add dispatch).
    if (auto s = project(batch, _attn_out, *lw.attn_output, _hidden, rows, &_residual, "proj_res"); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// GDN block.
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenDslEngine::gdn_block(QwenCommandBatch &batch, uint32_t layer,
                                              uint32_t rows, uint32_t pos0) {
    const auto &lw = _layer_weights[layer];
    const uint32_t slot = _gdn_slot[layer];
    const auto &state = _gdn_states[slot];
    const auto &ring = _gdn_rings[slot];
    const uint32_t hd = _cfg.gdn_head_dim;
    const uint32_t qkh = _cfg.gdn_qk_heads, vh = _cfg.gdn_v_heads;
    const uint32_t qkv_n = _cfg.gdn_qkv_n();
    const uint32_t v_n = _cfg.gdn_v_n();
    const uint32_t v_offset = 2u * _cfg.gdn_qk_n();
    batch.layer(layer);
    // Projections: packed QKV and z. Decode also fuses alpha/beta GEMVs and
    // their gate epilogues into one launch; prefill retains the reference path.
    if (auto s = project(batch, _normed, *lw.gdn_qkv, _proj_a, rows, nullptr, "proj"); !s.ok()) { return s; }
    if (auto s = project(batch, _normed, *lw.gdn_gate, _proj_b, rows, nullptr, "proj"); !s.ok()) { return s; }// z
    if (rows == 1u && _decode_specialized) {
        batch.annotate(lw.p_gates);
        auto plan = d::plan_qwen_gdn_decode_gates_dispatch(vh, _cfg.hidden, _format, _cfg.fp8_group);
        if (!plan.ok()) { return plan.status; }
        const auto as = lw.gdn_alpha->scales ? lw.gdn_alpha->scales.view() : _dummy_f.view();
        const auto bs = lw.gdn_beta->scales ? lw.gdn_beta->scales.view() : _dummy_f.view();
        batch << _gdn_decode_gates_sh(_normed.view(), lw.gdn_alpha->packed.view(), as,
                                      lw.gdn_beta->packed.view(), bs,
                                      lw.gdn_a->f32.view(), lw.gdn_dt_bias->f32.view(),
                                      _gdn_beta.view(), _gdn_g.view(),
                                      _cfg.hidden, vh, _cfg.fp8_group, _cfg.softplus_threshold)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    } else {
        if (auto s = project(batch, _normed, *lw.gdn_alpha, _gdn_alpha, rows, nullptr, "proj"); !s.ok()) { return s; }
        if (auto s = project(batch, _normed, *lw.gdn_beta, _proj_c, rows, nullptr, "proj"); !s.ok()) { return s; }
    }
    // Conv4 + SiLU with ring history over the packed QKV channels.
    if (auto s = d::validate_qwen_gdn_conv_inputs(rows, qkv_n, _cfg.conv_kernel, pos0); !s.ok()) { return s; }
    batch.annotate(lw.p_conv);
    {
        auto plan = d::plan_qwen_gdn_conv_dispatch(qkv_n);
        if (!plan.ok()) { return plan.status; }
        batch << _gdn_conv_sh(_proj_a.view(), lw.gdn_conv1d->f32.view(), ring.view(), _gdn_conv.view(),
                        rows, qkv_n, _cfg.conv_kernel, pos0)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    if (rows != 1u || !_decode_specialized) {
        // Reference gates: beta = sigmoid(b), g = ssm_a*softplus(a+dt_bias).
        batch.annotate(lw.p_gates);
        auto plan = d::plan_qwen_gdn_gates_dispatch(rows, vh);
        if (!plan.ok()) { return plan.status; }
        batch << _gdn_gates_sh(_gdn_alpha.view(), _proj_c.view(), lw.gdn_a->f32.view(), lw.gdn_dt_bias->f32.view(),
                               _gdn_beta.view(), _gdn_g.view(), rows, vh, _cfg.softplus_threshold)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    // QK L2 norm + Q scale (reads conv output, writes _gdn_q/_gdn_k).
    batch.annotate(lw.p_qk_norm);
    {
        auto plan = d::plan_qwen_gdn_qk_norm_dispatch(rows, qkh);
        if (!plan.ok()) { return plan.status; }
        batch << _gdn_qk_norm_sh(_gdn_conv.view(), _gdn_q.view(), _gdn_k.view(),
                           rows, qkh, hd, v_n, _cfg.l2_eps, _cfg.gdn_q_scale(), rows * qkh)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    // Prefill defaults to the WY representation + UT transform. Decode and
    // the explicit differential mode retain the exact per-token recurrence.
    if (rows == 1u && _decode_specialized) {
        if (auto s = d::validate_qwen35_gdn_decode4_inputs(rows, vh, qkh, hd); !s.ok()) { return s; }
        batch.annotate(lw.p_recurrent);
        auto plan = d::plan_qwen35_gdn_decode4_dispatch(vh);
        if (!plan.ok()) { return plan.status; }
        batch << _gdn_decode4_sh(_gdn_q.view(), _gdn_k.view(), _gdn_conv.view(),
                                 _gdn_beta.view(), _gdn_g.view(), state.view(), _gdn_out.view(),
                                 rows, vh, hd, qkh, v_offset)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    } else if (rows > 1u && _gdn_wy_ut) {
        if (auto s = d::validate_qwen_gdn_chunk_inputs(rows, vh, qkh, hd); !s.ok()) { return s; }
        if (d::qwen_gdn_chunk_shared_bytes() > 32ull * 1024ull) {
            return err(StatusCode::unsupported, "WY/UT chunk-64 shared-memory budget exceeded");
        }
        batch.annotate(lw.p_chunk);
        auto plan = d::plan_qwen_gdn_chunk_dispatch(vh);
        if (!plan.ok()) { return plan.status; }
        batch << _gdn_chunk_sh(_gdn_q.view(), _gdn_k.view(), _gdn_conv.view(),
                               _gdn_beta.view(), _gdn_g.view(), state.view(),
                               _gdn_w.view(), _gdn_vnew.view(), _gdn_out.view(),
                               rows, vh, hd, qkh, v_offset)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    } else {
        if (auto s = d::validate_qwen_gdn_recurrent_inputs(rows, vh, qkh, hd); !s.ok()) { return s; }
        if (auto s = d::check_qwen_gdn_shared_budget(hd, 32ull * 1024ull); !s.ok()) { return s; }
        batch.annotate(lw.p_recurrent);
        auto plan = d::plan_qwen_gdn_recurrent_dispatch(vh);
        if (!plan.ok()) { return plan.status; }
        batch << _gdn_recurrent_sh(_gdn_q.view(), _gdn_k.view(), _gdn_conv.view(),
                                   _gdn_beta.view(), _gdn_g.view(), state.view(), _gdn_out.view(),
                                   rows, vh, hd, qkh, v_offset)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    // Gated output RMSNorm (direct weight, norm before SiLU(z) gate). z=_proj_b.
    batch.annotate(lw.p_gated_norm);
    {
        auto plan = d::plan_qwen_gdn_gated_norm_dispatch(rows, vh);
        if (!plan.ok()) { return plan.status; }
        batch << _gdn_gated_norm_sh(_gdn_out.view(), _proj_b.view(), lw.gdn_norm->f32.view(), _gdn_normed.view(),
                              rows, vh, hd, _cfg.norm_eps)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    // Output projection fused with the residual epilogue:
    // hidden = residual + gdn_normed @ W_out (no separate residual_add).
    if (auto s = project(batch, _gdn_normed, *lw.gdn_out, _hidden, rows, &_residual, "proj_res"); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// FFN block (post-attention norm -> gate/up -> SwiGLU -> down + residual).
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenDslEngine::ffn_block(QwenCommandBatch &batch, uint32_t layer, uint32_t rows) {
    const auto &lw = _layer_weights[layer];
    const uint32_t hidden = _cfg.hidden, ffn = _cfg.ffn;
    batch.layer(layer);
    batch.annotate(lw.p_post_norm);
    // Post-attention RMSNorm (direct pre-added GGUF weight) on the hidden (which
    // already includes the attention/GDN residual). The FFN down projection
    // adds its output back into hidden via the residual epilogue.
    {
        auto plan = d::plan_qwen35_rmsnorm_dispatch(rows, hidden, 256u);
        if (!plan.ok()) { return plan.status; }
        batch << _rmsnorm_plain(_hidden.view(), lw.post_norm->f32.view(), _dummy_f.view(),
                             _normed.view(), _dummy_f.view(), rows, hidden, _cfg.norm_eps)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    if (rows == 1u && _decode_specialized) {
        // Decode fusion: gate GEMV + up GEMV + SwiGLU in one launch. The two
        // packed weight streams share each activation load and stay in FP32.
        batch.annotate(lw.p_swiglu);
        auto plan = d::plan_qwen_ffn_decode_dispatch(ffn, hidden, _format, _cfg.fp8_group);
        if (!plan.ok()) { return plan.status; }
        const auto gs = lw.ffn_gate->scales ? lw.ffn_gate->scales.view() : _dummy_f.view();
        const auto us = lw.ffn_up->scales ? lw.ffn_up->scales.view() : _dummy_f.view();
        batch << _ffn_decode(_normed.view(), lw.ffn_gate->packed.view(), gs,
                             lw.ffn_up->packed.view(), us, _ffn_act.view(),
                             ffn, hidden, _cfg.fp8_group)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    } else {
        // Reference/prefill route: separate gate/up projections and SwiGLU.
        if (auto s = project(batch, _normed, *lw.ffn_gate, _ffn_gate, rows, nullptr, "proj"); !s.ok()) { return s; }
        if (auto s = project(batch, _normed, *lw.ffn_up, _ffn_up, rows, nullptr, "proj"); !s.ok()) { return s; }
        batch.annotate(lw.p_swiglu);
        const uint32_t count = rows * ffn;
        auto plan = d::plan_qwen_swiglu_dispatch(count);
        if (!plan.ok()) { return plan.status; }
        batch << _swiglu(_ffn_gate.view(), _ffn_up.view(), _ffn_act.view(), count)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    // Down projection with residual: hidden = hidden + down(act).
    if (auto s = project(batch, _ffn_act, *lw.ffn_down, _hidden, rows, &_hidden, "proj_res"); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Forward one chunk of `rows` tokens at absolute position pos0.
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenDslEngine::forward_chunk(Stream &stream, QwenCommandBatch &batch,
                                                  uint32_t rows, uint32_t pos0) {
    if (rows == 0u || pos0 > _max_context || rows > _max_context - pos0) {
        return invalid("forward chunk exceeds sparse KV capacity");
    }
    // Growth only adds previously unmapped tiles. It is therefore safe for this
    // map submission to precede commands already recorded in `batch`: those
    // commands retain their existing tiles, and this chunk's maps precede its
    // later kernel submission on the same stream.
    if (auto s = _kv_cache.ensure_capacity(stream, pos0 + rows); !s.ok()) { return s; }
    const uint32_t hidden = _cfg.hidden;
    batch.layer(UINT32_MAX);// global (non-layer) commands until the first block
    // The engine's positions are exactly pos0 + row. The attention-prepare
    // specialization derives them in-kernel, eliminating per-chunk H2D/copy
    // commands and keeping decode graphs free of host-upload staging.
    // Embedding gather: hidden = table[ids].
    batch.annotate("embedding");
    {
        auto plan = d::plan_qwen_embedding_dispatch(rows, hidden);
        if (!plan.ok()) { return plan.status; }
        const auto scales = _token_embd->scales ? _token_embd->scales.view() : _dummy_f.view();
        batch << _embedding(_ids.view(), _token_embd->packed.view(), scales, _hidden.view(),
                         rows, hidden, _cfg.vocab, _cfg.fp8_group, plan.value->padded_cols)
                  .dispatch(plan.value->total_x, plan.value->total_y);
    }
    for (uint32_t l = 0; l < _layers; l++) {
        const auto &lw = _layer_weights[l];
        // Fused residual snapshot + input RMSNorm. The snapshot epilogue
        // writes the ORIGINAL hidden values to _residual during its sum pass,
        // then normalizes those exact values into _normed. This replaces two
        // launches and removes one full activation read.
        batch.layer(l).annotate(lw.p_input_norm);
        {
            auto plan = d::plan_qwen35_rmsnorm_dispatch(rows, hidden, 256u);
            if (!plan.ok()) { return plan.status; }
            batch << _rmsnorm_snapshot(_hidden.view(), lw.attn_norm->f32.view(), _dummy_f.view(),
                                       _normed.view(), _residual.view(), rows, hidden, _cfg.norm_eps)
                      .dispatch(d::qwen_dispatch_threads(*plan.value));
        }
        if (lw.full_attention) {
            if (auto s = full_attention_block(batch, l, rows, pos0); !s.ok()) { return s; }
        } else {
            if (auto s = gdn_block(batch, l, rows, pos0); !s.ok()) { return s; }
        }
        if (auto s = ffn_block(batch, l, rows); !s.ok()) { return s; }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// LM head on the last position.
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenDslEngine::lm_head(QwenCommandBatch &batch, uint32_t rows) {
    const uint32_t hidden = _cfg.hidden;
    batch.layer(UINT32_MAX);
    batch.annotate("output_norm");
    // Copy the last position's hidden state into _last_normed input slot.
    // rmsnorm reads [rows=1, hidden] from _hidden at offset (rows-1)*hidden.
    const auto last_hidden = _hidden.view(static_cast<size_t>(rows - 1u) * hidden, hidden);
    {
        auto plan = d::plan_qwen35_rmsnorm_dispatch(1u, hidden, 256u);
        if (!plan.ok()) { return plan.status; }
        batch << _rmsnorm_plain(last_hidden, _output_norm->f32.view(), _dummy_f.view(),
                             _last_normed.view(), _dummy_f.view(), 1u, hidden, _cfg.norm_eps)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    // Tied LM head: logits[vocab] = last_normed[hidden] @ token_embd^T.
    if (auto s = project(batch, _last_normed, *_token_embd, _logits, 1u, nullptr, "lm_head"); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// prefill
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenDslEngine::prefill(Stream &stream, luisa::span<const uint32_t> prompt_tokens,
                                            luisa::vector<float> &logits_out) {
    if (prompt_tokens.empty()) { return invalid("prefill prompt must be nonempty"); }
    if (prompt_tokens.size() > _max_context) {
        return invalid("prompt exceeds max_context");
    }
    if (auto s = d::validate_qwen_embedding_inputs(prompt_tokens, _cfg.hidden, _cfg.vocab); !s.ok()) { return s; }
    const uint32_t total = static_cast<uint32_t>(prompt_tokens.size());
    // Fresh report for this public call; reset() below may append its samples
    // (benchmark mode) and they are intentionally kept.
    if (_benchmark) { _report.clear(); }
    // A prefill always starts a fresh session.
    if (auto s = reset(stream); !s.ok()) {
        _has_session = false;
        return s;
    }
    // ONE command batch for ALL chunks: normal mode submits once, then the
    // logits download synchronizes. Each chunk stages a DISTINCT host span.
    _staged = 0;
    auto batch = QwenCommandBatch{stream, _benchmark ? &_report : nullptr,
                                  QwenBenchmarkPhase::prefill, 0u, total};
    uint32_t done = 0;
    while (done < total) {
        const uint32_t rows = std::min(_prefill_chunk, total - done);
        const uint32_t pos0 = done;
        batch.set_context(QwenBenchmarkPhase::prefill, pos0, rows);
        // Stage this chunk's token ids (persistent host buffer + device copy).
        if (auto s = stage_ids(batch, prompt_tokens.subspan(done, rows)); !s.ok()) {
            _has_session = false;
            return s;
        }
        if (auto s = forward_chunk(stream, batch, rows, pos0); !s.ok()) {
            _has_session = false;// a partial forward may have advanced device state
            return s;
        }
        // Only the LAST chunk computes the LM head.
        if (done + rows == total) {
            if (auto s = lm_head(batch, rows); !s.ok()) {
                _has_session = false;
                return s;
            }
        }
        done += rows;
    }
    // Single submission of the whole prefill (no-op in per-command mode).
    batch.flush(false);
    // Download logits in the same command list (single synchronize at end).
    logits_out.resize(_cfg.vocab);
    batch.layer(UINT32_MAX)
        .set_context(QwenBenchmarkPhase::prefill, total - 1u, 1u)
        .annotate("logits_download");
    batch << _logits.view().copy_to(luisa::span<float>{logits_out});
    batch.flush(true);
    _position = total;
    _has_session = true;
    _greedy_next_valid = false;
    return Status::success();
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenDslEngine::decode(Stream &stream, uint32_t input_token,
                                           luisa::vector<float> &logits_out) {
    if (!_has_session) { return invalid("decode requires a prior nonempty prefill"); }
    if (input_token >= _cfg.vocab) { return invalid("decode token id out of vocabulary range"); }
    if (_position >= _max_context) { return invalid("decode would exceed max_context"); }
    const uint32_t pos0 = _position;
    if (_benchmark) { _report.clear(); }
    _staged = 0;
    auto batch = QwenCommandBatch{stream, _benchmark ? &_report : nullptr,
                                  QwenBenchmarkPhase::decode, pos0, 1u};
    if (auto s = stage_ids(batch, luisa::span<const uint32_t>{&input_token, 1}); !s.ok()) {
        _has_session = false;
        return s;
    }
    if (auto s = forward_chunk(stream, batch, 1u, pos0); !s.ok()) {
        _has_session = false; // a partial forward may have advanced device state
        return s;
    }
    if (auto s = lm_head(batch, 1u); !s.ok()) {
        _has_session = false;
        return s;
    }
    // ONE command list per decode step; no synchronize until the download.
    batch.flush(false);
    logits_out.resize(_cfg.vocab);
    batch.layer(UINT32_MAX).annotate("logits_download");
    batch << _logits.view().copy_to(luisa::span<float>{logits_out});
    batch.flush(true);
    _position = pos0 + 1u;
    _greedy_next_valid = false;
    return Status::success();
}

[[nodiscard]] Status QwenDslEngine::launch_device_greedy_graph(
    Stream &stream, QwenCommandBatch &batch) {
    auto list = batch.take_commands();
    if (list.empty()) { return invalid("device greedy graph command list is empty"); }
    if (!_device_greedy_graph_exec.has_value()) {
        auto graph = _cuda_graph_ext->create_graph(std::move(list.commit()).command_list());
        if (graph.handle().handle == compute::CudaGraphExt::invalid_handle) {
            return err(StatusCode::unsupported,
                       "CUDA graph capture failed for FP8 device greedy decode");
        }
        auto exec = _cuda_graph_ext->instantiate(graph.handle().handle);
        if (exec.handle().handle == compute::CudaGraphExt::invalid_handle) {
            return err(StatusCode::unsupported,
                       "CUDA graph instantiation failed for FP8 device greedy decode");
        }
        _device_greedy_graph.emplace(std::move(graph));
        _device_greedy_graph_exec.emplace(std::move(exec));
    } else {
        compute::CudaGraphExt::GraphUpdateError error;
        if (!_cuda_graph_ext->update(_device_greedy_graph_exec->handle().handle,
                                     std::move(list.commit()).command_list(), &error)) {
            return err(StatusCode::internal_error,
                       "CUDA graph update failed for FP8 device greedy decode",
                       "node " + luisa::string{std::to_string(error.error_node)} +
                           ", type " + luisa::string{std::to_string(static_cast<uint32_t>(error.type))} +
                           ": " + error.detail);
        }
        _cuda_graph_updates++;
    }
    _cuda_graph_ext->launch(_device_greedy_graph_exec->handle().handle, stream.handle());
    _cuda_graph_steps++;
    return Status::success();
}

[[nodiscard]] Status QwenDslEngine::decode_device_greedy(
    Stream &stream, const uint32_t *input_token, uint32_t history_slot) {
    if (!_has_session) { return invalid("device greedy decode requires a prior prefill"); }
    if (history_slot >= _max_context) { return invalid("device greedy history slot exceeds max_context"); }
    if (_position >= _max_context) { return invalid("device greedy decode would exceed max_context"); }
    if (input_token == nullptr) {
        if (!_greedy_next_valid) { return invalid("device greedy decode has no device-resident input token"); }
    } else if (*input_token >= _cfg.vocab) {
        return invalid("device greedy input token id out of vocabulary range");
    }
    const uint32_t pos0 = _position;
    if (_benchmark) { _report.clear(); }
    _staged = 0;
    auto batch = QwenCommandBatch{stream, _benchmark ? &_report : nullptr,
                                  QwenBenchmarkPhase::decode, pos0, 1u};
    if (input_token != nullptr) {
        if (_cuda_graph_ext != nullptr) {
            // Host upload commands stay outside the updated graph. This avoids
            // releasing replaced graph staging while an earlier launch is in
            // flight; stream order still makes ids visible to the graph body.
            auto upload = QwenCommandBatch{stream, nullptr,
                                           QwenBenchmarkPhase::decode, pos0, 1u};
            if (auto s = stage_ids(upload, luisa::span<const uint32_t>{input_token, 1}); !s.ok()) {
                _has_session = false;
                return s;
            }
            upload.flush(false);
        } else if (auto s = stage_ids(batch, luisa::span<const uint32_t>{input_token, 1}); !s.ok()) {
            _has_session = false;
            return s;
        }
    }
    if (auto s = forward_chunk(stream, batch, 1u, pos0); !s.ok()) {
        _has_session = false;
        return s;
    }
    if (auto s = lm_head(batch, 1u); !s.ok()) {
        _has_session = false;
        return s;
    }
    auto partial_plan = d::plan_qwen_greedy_argmax_partial_dispatch(_cfg.vocab);
    auto finish_plan = d::plan_qwen_greedy_argmax_finish_dispatch(_cfg.vocab);
    if (!partial_plan.ok()) {
        _has_session = false;
        return partial_plan.status;
    }
    if (!finish_plan.ok()) {
        _has_session = false;
        return finish_plan.status;
    }
    batch.layer(UINT32_MAX).annotate("greedy_argmax_partial");
    batch << _greedy_argmax_partial(_logits.view(), _greedy_values.view(),
                                    _greedy_indices.view(), _greedy_invalid.view(), _cfg.vocab)
                 .dispatch(d::qwen_dispatch_threads(*partial_plan.value));
    batch.layer(UINT32_MAX).annotate("greedy_argmax_finish");
    batch << _greedy_argmax_finish(_greedy_values.view(), _greedy_indices.view(),
                                   _greedy_invalid.view(), _ids.view(), _greedy_history.view(),
                                   _cfg.vocab, history_slot)
                 .dispatch(d::qwen_dispatch_threads(*finish_plan.value));
    // Do not synchronize: the next in-flight step reads _ids produced above.
    // Stream order protects all reused activation/state buffers. Graph capture
    // or update consumes the list, so failures are returned without a fallback
    // submission of a potentially consumed command sequence.
    if (_cuda_graph_ext != nullptr) {
        if (auto s = launch_device_greedy_graph(stream, batch); !s.ok()) {
            _has_session = false;
            return s;
        }
    } else {
        batch.flush(false);
    }
    _position = pos0 + 1u;
    _greedy_next_valid = true;
    return Status::success();
}

[[nodiscard]] Status QwenDslEngine::download_device_greedy(
    Stream &stream, uint32_t first_slot, luisa::span<uint32_t> tokens_out) {
    if (tokens_out.empty()) { return Status::success(); }
    if (!_greedy_next_valid) { return invalid("device greedy download has no completed selection"); }
    if (first_slot >= _max_context || tokens_out.size() > _max_context - first_slot) {
        return invalid("device greedy download range exceeds history");
    }
    auto batch = QwenCommandBatch{stream, _benchmark ? &_report : nullptr,
                                  QwenBenchmarkPhase::decode, _position, 1u};
    batch.layer(UINT32_MAX).annotate("greedy_tokens_download");
    batch << _greedy_history.view(first_slot, tokens_out.size()).copy_to(tokens_out);
    batch.flush(true);
    for (auto token : tokens_out) {
        if (token >= _cfg.vocab) {
            _has_session = false;
            _greedy_next_valid = false;
            return err(StatusCode::internal_error,
                       "device greedy selection rejected NaN or invalid logits");
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------
[[nodiscard]] QwenEngineInfo QwenDslEngine::info() const noexcept {
    QwenEngineInfo i;
    i.kernel = QwenKernelKind::dsl;
    i.weight_storage = _storage;
    i.position = _position;
    i.max_context = _max_context;
    i.prefill_chunk = _prefill_chunk;
    i.cache_bytes = _cache_bytes;
    i.workspace_bytes = _workspace_bytes;
    i.compiled_shader_count = _shader_count;
    i.cuda_graph_steps = _cuda_graph_steps;
    i.cuda_graph_updates = _cuda_graph_updates;
    i.gdn_wy_ut = _gdn_wy_ut;
    i.caps = _caps;
    return i;
}

[[nodiscard]] const QwenBenchmarkReport *QwenDslEngine::benchmark_report() const noexcept {
    return _benchmark ? &_report : nullptr;
}

}// namespace luisa::qwen
