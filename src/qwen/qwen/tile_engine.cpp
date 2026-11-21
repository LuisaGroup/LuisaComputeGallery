// Qwen3.5-4B complete Tile executor -- implementation.
// See tile_engine.h for the schedule/state/ABI contract. All device COMPUTE
// goes through the tile_ops.h capture factories compiled with tile::compile;
// this file only plans, validates, allocates, caches shaders, and records
// commands through QwenCommandBatch (a plain CommandList wrapper in normal
// mode; a per-command commit+synchronize recorder in benchmark mode). Buffer
// copies (host staging uploads, logits download) carry no compute semantics.
// Every kernel/transfer is annotated with a unique operator/projection name;
// phase/position/rows are set per chunk and layer attribution per block.
#include "qwen/tile_engine.h"

#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstdlib>

#include <luisa/core/logging.h>

namespace luisa::qwen {

using compute::Buffer;
using compute::BufferView;
using compute::CommandList;
using compute::Device;
using compute::Stream;
namespace t = luisa::compute::tile;
namespace to = luisa::qwen::tile_ops;

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

// A matrix weight must be packed (bf16/fp8) with dims [K, N].
[[nodiscard]] Status check_matrix(const QwenDeviceTensor &t, uint64_t k, uint64_t n,
                                  to::QwenTileWeightFormat format) {
    const auto want = format == to::QwenTileWeightFormat::bf16 ? WeightStorage::bf16_packed
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
    const uint64_t codes_per_word = format == to::QwenTileWeightFormat::bf16 ? 2u : 4u;
    const uint64_t expected_words = (elements + codes_per_word - 1u) / codes_per_word;
    if (!t.packed || t.packed.size() != expected_words) {
        return invalid("weight packed buffer size mismatch", t.name);
    }
    if (format == to::QwenTileWeightFormat::fp8_e4m3) {
        const uint64_t expect = (k + 63u) / 64u;
        uint64_t scale_count = 0u;
        if (!t.scales || t.groups_per_row != expect ||
            !checked_mul(n, expect, scale_count) || t.scales.size() != scale_count) {
            return invalid("fp8 weight scale geometry/buffer size mismatch", t.name);
        }
        if (k % 4u != 0u) {
            return invalid("fp8 Tile GEMM requires K divisible by 4 (word-aligned packed rows)", t.name);
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

// Canonical cache key from typed semantic parameters. Integer overloads are
// templated so uint32_t/uint64_t/int callers are unambiguous vs the float one.
struct Key {
    luisa::string s;
    template<typename V>
        requires std::integral<V>
    Key &add(std::string_view tag, V v) {
        s.append(tag);
        s.append(std::to_string(static_cast<uint64_t>(v)).c_str());
        s.push_back(';');
        return *this;
    }
    Key &add(std::string_view tag, float v) {
        // Bit-exact float key (eps/theta/threshold are semantic constants).
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        s.append(tag);
        s.append(std::to_string(static_cast<uint64_t>(bits)).c_str());
        s.push_back(';');
        return *this;
    }
};

}// namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
QwenTileEngine::~QwenTileEngine() noexcept {
    if (_stream) { _stream.synchronize(); }
}

[[nodiscard]] Result<luisa::unique_ptr<QwenEngine, QwenTileEngine::Deleter>> QwenTileEngine::create(
    Device &device, Stream &stream, const Qwen35Model &model,
    const QwenDeviceWeights &weights, const QwenEngineOptions &options) {
    QwenTileEngineOptions topt;
    topt.base = options;
    return create(device, stream, model, weights, topt);
}

[[nodiscard]] Result<luisa::unique_ptr<QwenEngine, QwenTileEngine::Deleter>> QwenTileEngine::create(
    Device &device, Stream &stream, const Qwen35Model &model,
    const QwenDeviceWeights &weights, const QwenTileEngineOptions &options) {
    if (options.base.kernel != QwenKernelKind::tile) {
        return Result<luisa::unique_ptr<QwenEngine, QwenTileEngine::Deleter>>::failure(
            invalid("QwenTileEngine requires QwenKernelKind::tile"));
    }
    if (!device) {
        return Result<luisa::unique_ptr<QwenEngine, QwenTileEngine::Deleter>>::failure(invalid("invalid device"));
    }
    const auto &g = model.config().geometry;
    if (options.base.max_context == 0u || options.base.max_context > g.context_length) {
        return Result<luisa::unique_ptr<QwenEngine, QwenTileEngine::Deleter>>::failure(
            invalid("max_context must be in [1, model context_length]"));
    }
    if (options.base.prefill_chunk == 0u) {
        return Result<luisa::unique_ptr<QwenEngine, QwenTileEngine::Deleter>>::failure(invalid("prefill_chunk must be >= 1"));
    }
    if (options.gemm_tile_m == 0u || options.gemm_tile_n == 0u ||
        options.gemm_tile_m > 256u || options.gemm_tile_n > 256u ||
        options.gemm_tile_m * options.gemm_tile_n > 16384u) {
        return Result<luisa::unique_ptr<QwenEngine, QwenTileEngine::Deleter>>::failure(invalid("gemm tile out of range"));
    }
    // A bucket must be a sane key range; 0 means "no bucketing". Bucketing only
    // applies to the device-visible-length (dynamic) attention shaders, and the
    // bucket COUNT is what bounds the shader family, so that is what is
    // validated (a bucket >= max_context degenerates to the single full
    // capacity). 64 buckets at most: at most 64 decode + 64 full-chunk prefill
    // specializations exist regardless of max_context.
    if (options.attention_bucket_keys != 0u &&
        (options.attention_bucket_keys < 16u ||
         (static_cast<uint64_t>(options.base.max_context) + options.attention_bucket_keys - 1u) /
                 options.attention_bucket_keys >
             64u)) {
        return Result<luisa::unique_ptr<QwenEngine, QwenTileEngine::Deleter>>::failure(
            invalid("attention_bucket_keys must be 0 (off) or a range that yields at most 64 buckets"));
    }

    // Allocate with Luisa's allocator (mimalloc-backed); Deleter frees with
    // the SAME allocator (CRT new + eastl default_delete mixes heaps and
    // segfaults on destruction -- same pattern as QwenDslEngine).
    auto *raw = ::new (luisa::allocate_with_allocator<QwenTileEngine>()) QwenTileEngine{};
    auto engine = luisa::unique_ptr<QwenTileEngine, QwenTileEngine::Deleter>{raw};
    auto &e = *engine;
    e._opt = options;
    e._benchmark = options.base.benchmark;
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
    if (auto v = to::validate_qwen_tile_config(e._cfg); !v.ok()) {
        return Result<luisa::unique_ptr<QwenEngine, QwenTileEngine::Deleter>>::failure(
            invalid("tile config validation failed", v.message));
    }
    e._layers = static_cast<uint32_t>(g.layers);
    e._max_context = options.base.max_context;
    e._prefill_chunk = options.base.prefill_chunk;
    e._weights = &weights;
    e._device = device;
    e._stream = stream;
    const luisa::string_view backend{device.backend_name()};
    // Vulkan's legacy HLSL ABI requires storage-buffer view offsets aligned
    // to minStorageBufferOffsetAlignment (256 on NVIDIA). The one-token conv
    // step binds whole per-token scratch buffers instead of subviews, so we
    // pad the source row stride to a conservative 256-byte boundary on VK.
    // Other backends accept element-aligned views; keep the dense stride.
    if (backend == luisa::string_view{"vk"}) {
        e._view_align_bytes = 256;
    }
    // Normal-mode stage sync policy (detected once): DX drops the
    // block-internal stage flushes entirely and makes the every-N-layer
    // bounds commit-only (flush(false)) -- no host wait until the final
    // logits download. VK and every other backend keep the historical
    // stage-boundary + every-N-layer flush(true) policy. Benchmark mode
    // commits + syncs every command individually, so flushes are no-ops.
    e._stage_sync = backend != luisa::string_view{"dx"};

  // KV dtype (opt_plan/05 Q3): the Tile engine's own attention schedule addresses the
  // cache with the DENSE FP32 [position][kv_head][channel] arithmetic baked into every
  // captured Tile kernel, and there is no BF16 blocked-layout Tile realization. Refuse
  // the request EXPLICITLY instead of quietly running an FP32 cache inside a BF16
  // session (which would be a silent dtype substitution): --kernel dsl is the path
  // that implements BF16 KV.
  if (options.base.kv_dtype != QwenKvDType::f32) {
      return Result<luisa::unique_ptr<QwenEngine, QwenTileEngine::Deleter>>::failure(
          Status::error(StatusCode::unsupported,
                        "the Tile engine has no BF16/blocked KV cache realization; use --kernel dsl",
                        luisa::string{"kv_dtype "} + luisa::string{qwen_kv_dtype_name(options.base.kv_dtype)}));
  }
    if (auto s = e.resolve_weights(model, weights); !s.ok()) {
        return Result<luisa::unique_ptr<QwenEngine, QwenTileEngine::Deleter>>::failure(std::move(s));
    }
    if (auto s = e.alloc_state(device); !s.ok()) {
        return Result<luisa::unique_ptr<QwenEngine, QwenTileEngine::Deleter>>::failure(std::move(s));
    }
    if (auto s = e.alloc_workspace(device); !s.ok()) {
        return Result<luisa::unique_ptr<QwenEngine, QwenTileEngine::Deleter>>::failure(std::move(s));
    }
    if (auto s = e.precompile(device, stream); !s.ok()) {
        return Result<luisa::unique_ptr<QwenEngine, QwenTileEngine::Deleter>>::failure(std::move(s));
    }
    if (auto s = e.reset(stream); !s.ok()) {
        return Result<luisa::unique_ptr<QwenEngine, QwenTileEngine::Deleter>>::failure(std::move(s));
    }
    // Capability record (opt_plan/04-backend-accelerators.md section 7): probed
    // once, after precompile (so compute_warp_size() is meaningful) and after
    // alloc_state (so the KV page geometry comes from the live allocator), then
    // logged exactly once. Nothing downstream re-queries these values.
    e._caps = qwen_probe_backend_caps(device);
    qwen_fill_kv_page_caps(e._caps, e._kv_cache);
    qwen_log_backend_caps(e._caps);
    return Result<luisa::unique_ptr<QwenEngine, QwenTileEngine::Deleter>>::success(std::move(engine));
}

[[nodiscard]] Status QwenTileEngine::resolve_weights(const Qwen35Model &model,
                                                     const QwenDeviceWeights &weights) {
    const auto &g = model.config().geometry;
    const auto *embd = weights.find(QwenTensorRole::token_embd);
    if (embd == nullptr) { return invalid("token_embd weight missing"); }
    if (embd->storage == WeightStorage::bf16_packed) {
        _format = to::QwenTileWeightFormat::bf16;
        _storage = WeightStorage::bf16_packed;
    } else if (embd->storage == WeightStorage::fp8_packed) {
        _format = to::QwenTileWeightFormat::fp8_e4m3;
        _storage = WeightStorage::fp8_packed;
    } else {
        return invalid("token_embd must be packed bf16 or fp8 (dense F32 embedding and the "
                       "native Q8_0 storage are not implemented by the Tile variants; "
                       "use --version dsl-q8 for a Q8_0 checkpoint)");
    }
    _token_embd = embd;
    if (auto s = check_matrix(*embd, g.hidden, g.vocab, _format); !s.ok()) { return s; }
    if (_format == to::QwenTileWeightFormat::fp8_e4m3 && g.hidden % to::kTileFp8Group != 0u) {
        return invalid("fp8 Tile embedding requires hidden divisible by 64");
    }
    const auto *onorm = weights.find(QwenTensorRole::output_norm);
    if (onorm == nullptr) { return invalid("output_norm weight missing"); }
    if (auto s = check_dense(*onorm, g.hidden); !s.ok()) { return s; }
    _output_norm = onorm;

    _layer_weights.resize(static_cast<size_t>(g.layers));
    _full_slot.assign(static_cast<size_t>(g.layers), UINT32_MAX);
    _gdn_slot.assign(static_cast<size_t>(g.layers), UINT32_MAX);
    uint32_t full_slot = 0, gdn_slot = 0;
    for (uint64_t l = 0; l < g.layers; l++) {
        auto &lw = _layer_weights[static_cast<size_t>(l)];
        lw.full_attention = is_full_attention_layer(g, l);
        const auto get = [&](QwenTensorRole role) -> const QwenDeviceTensor * {
            return weights.find(role, l);
        };
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

[[nodiscard]] Status QwenTileEngine::alloc_state(Device &device) {
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
    _gdn_rings.reserve(2u * gdn_count);
    _gdn_ring_parity.assign(static_cast<size_t>(gdn_count), 0u);
    // FP32 + position_major are pinned EXPLICITLY (the refusal above already
    // rejected every other dtype): the Tile attention kernels hard-code that
    // addressing, so the allocator must not derive a different geometry.
    if (auto s = _kv_cache.create(device, _stream, _max_context,
                                  static_cast<uint32_t>(kv_n), full_count,
                                  _opt.base.kv_page_bytes, QwenKvDType::f32,
                                  _cfg.kv_heads, QwenKvLayout::position_major);
        !s.ok()) {
        return s;
    }
    _cache_bytes = _kv_cache.logical_bytes();
    for (uint32_t i = 0; i < gdn_count; i++) {
        auto st = device.create_buffer<float>(static_cast<size_t>(state_elems));
        // Ping-pong pair: token at pos reads ring[2*i + pos%2] and writes
        // ring[2*i + (pos+1)%2]. No kernel ever reads and writes one ring.
        auto ring_a = device.create_buffer<float>(static_cast<size_t>(ring_elems));
        auto ring_b = device.create_buffer<float>(static_cast<size_t>(ring_elems));
        if (!st || !ring_a || !ring_b) { return err(StatusCode::out_of_memory, "GDN state allocation failed"); }
        _gdn_states.push_back(std::move(st));
        _gdn_rings.push_back(std::move(ring_a));
        _gdn_rings.push_back(std::move(ring_b));
        _cache_bytes += (state_elems + 2u * ring_elems) * 4u;
    }
    return Status::success();
}

[[nodiscard]] Status QwenTileEngine::alloc_workspace(Device &device) {
    const uint64_t C = _prefill_chunk;
    const uint64_t hidden = _cfg.hidden;
    const uint64_t ffn = _cfg.ffn;
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
    const auto mk_i = [&](uint64_t n) -> Buffer<int32_t> {
        auto b = device.create_buffer<int32_t>(static_cast<size_t>(n));
        if (b) { _workspace_bytes += n * 4u; }
        return b;
    };
    _ids = mk_i(C);
    _positions = mk_i(C);
    _attn_len = mk_i(2);
    _kv_len_stride = static_cast<uint32_t>(_view_align_bytes / 4u);
    _kv_len = mk_i(_kv_len_stride);
    const uint64_t rope_table_elems = static_cast<uint64_t>(_max_context) * (_cfg.rope_dim / 2u);
    if (rope_table_elems > 0xFFFFFFFFull) {
        return invalid("rope table element count exceeds uint32 (reduce max_context)");
    }
    _rope_cos = mk_f(rope_table_elems);
    _rope_sin = mk_f(rope_table_elems);
    // Whole-prompt host staging: prefill records ALL chunks into one batch
    // without intermediate syncs, so each chunk owns a distinct absolute
    // span (ids/pos at offset pos0, len pair at offset 2*pos0). The prompt
    // is bounded by max_context; decode (one chunk per batch) uses slot 0.
    _host_ids.resize(_max_context);
    _host_pos.resize(_max_context);
    _host_len.resize(2u * static_cast<size_t>(_max_context));
    _hidden = mk_f(C * hidden);
    _normed = mk_f(C * hidden);
    _residual = mk_f(C * hidden);
    _zero = mk_f(C * hidden);
    _proj_a = mk_f(C * max_proj);
    _proj_b = mk_f(C * max_proj);
    _proj_c = mk_f(C * max_proj);
    _kseg_acc = mk_f(C * max_proj);
    _ffn_gate = mk_f(C * ffn);
    _ffn_up = mk_f(C * ffn);
    _ffn_act = mk_f(C * ffn);
    _attn_proj_q = mk_f(C * _cfg.attn_q_proj_n());
    _attn_proj_k = mk_f(C * _cfg.attn_kv_n());
    _attn_proj_v = mk_f(C * _cfg.attn_kv_n());
    _attn_q = mk_f(q_elems);
    _attn_gate = mk_f(q_elems);
    _attn_k = mk_f(kv_elems);
    _attn_v = mk_f(kv_elems);
    _attn_qr = mk_f(q_elems);
    _attn_kr = mk_f(kv_elems);
    _attn_out = mk_f(q_elems);
    _gdn_conv = mk_f(gdn_conv);
    // Dense [chunk, qkv_n] packed-QKV projection output: the multi-token conv
    // chunk reads rows in place (stride == qkv_n), so chunk prefill needs no
    // per-token source staging copy.
    _gdn_qkv = mk_f(gdn_conv);
    {
        const uint64_t align_elems = _view_align_bytes / 4u;
        const uint64_t qkv_n = _cfg.gdn_qkv_n();
        _conv_src_stride = static_cast<uint32_t>(((qkv_n + align_elems - 1u) / align_elems) * align_elems);
        const uint64_t src_elems = C * _conv_src_stride;
        if (src_elems > 0xFFFFFFFFull) {
            return invalid("conv scratch element count exceeds uint32");
        }
        _conv_src = mk_f(src_elems);
        _conv_out = mk_f(gdn_conv);
        _conv_pos_stride = static_cast<uint32_t>(align_elems);
        _conv_pos = mk_i(C * _conv_pos_stride);
    }
    _gdn_q = mk_f(gdn_qk);
    _gdn_k = mk_f(gdn_qk);
    _gdn_beta = mk_f(gdn_gate);
    _gdn_g = mk_f(gdn_gate);
    _gdn_braw = mk_f(gdn_gate);
    _gdn_out = mk_f(gdn_v);
    _gdn_normed = mk_f(gdn_v);
    _last_normed = mk_f(hidden);
    _logits = mk_f(vocab);
    _dummy_f = mk_f(1u);
    _dummy_f2 = mk_f(1u);
    _dummy_u = [&] {
        auto b = device.create_buffer<uint32_t>(1u);
        if (b) { _workspace_bytes += 4u; }
        return b;
    }();
    if (!_ids || !_positions || !_attn_len || !_kv_len || !_rope_cos || !_rope_sin || !_hidden || !_normed || !_residual || !_zero ||
        !_proj_a || !_proj_b || !_proj_c || !_kseg_acc || !_ffn_gate || !_ffn_up || !_ffn_act ||
        !_attn_proj_q || !_attn_proj_k || !_attn_proj_v || !_attn_q || !_attn_gate || !_attn_k ||
        !_attn_v || !_attn_qr || !_attn_kr || !_attn_out || !_gdn_conv || !_gdn_qkv || !_gdn_q || !_gdn_k ||
        !_gdn_beta || !_gdn_g || !_gdn_braw || !_gdn_out || !_gdn_normed || !_last_normed || !_logits ||
          !_conv_src || !_conv_out || !_conv_pos ||
        !_dummy_f || !_dummy_f2 || !_dummy_u) {
        return err(StatusCode::out_of_memory, "workspace allocation failed");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Shader cache
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenTileEngine::shader_for(Device &device, const luisa::string &key,
                                                const to::TileCapture &capture,
                                                const t::Shader **out, luisa::string &detail) {
    if (auto it = _shaders.find(key); it != _shaders.end()) {
        *out = it->second.get();
        return Status::success();
    }
    if (!capture.ok()) {
        detail = capture.status.message;
        for (auto &d : capture.diagnostics) {
            detail.append("\n  diag: ");
            detail.append(d);
        }
        return err(StatusCode::internal_error, "Tile capture failed verification", detail);
    }
    t::CompileOptions copt;
    copt.lowering = _opt.lowering;
    compute::ShaderOption sopt;
    sopt.enable_fast_math = false;// FP32 fidelity for norms/softmax/state
    auto shader = t::compile(device, *capture.kernel, copt, sopt);
    if (!shader) {
        detail = shader.metadata().error;
        for (auto &d : capture.diagnostics) {
            detail.append("\n  diag: ");
            detail.append(d);
        }
        return err(StatusCode::internal_error, "Tile shader compile failed (no lowering fallback)", detail);
    }
    _shader_count++;
    auto owned = luisa::make_unique<t::Shader>(std::move(shader));
    *out = owned.get();
    _shaders.emplace(key, std::move(owned));
    return Status::success();
}

template<typename F>
[[nodiscard]] Status QwenTileEngine::shader_lazy(Device &device, const luisa::string &key,
                                                 F &&make_capture, const t::Shader **out) {
    if (auto it = _shaders.find(key); it != _shaders.end()) {
        *out = it->second.get();
        return Status::success();
    }
    auto capture = make_capture();
    luisa::string detail;
    return shader_for(device, key, capture, out, detail);
}

[[nodiscard]] uint32_t QwenTileEngine::attention_capacity(uint32_t kv_len) const noexcept {
    // ONE policy, shared with the tests (tile_ops): bucket ceiling clamped to
    // max_context, or the whole capacity when bucketing is disabled. kv_len <=
    // max_context is an engine invariant, so the clamp only ever applies to a
    // partially filled last bucket and the result is never below kv_len.
    return to::tile_attention_bucket_capacity(kv_len, _opt.attention_bucket_keys, _max_context);
}

template<typename F>
[[nodiscard]] Status QwenTileEngine::attention_shader(Device &device, const luisa::string &key,
                                                      uint32_t rows, uint32_t cap, F &&make_capture,
                                                      const t::Shader **out) {
    if (auto it = _shaders.find(key); it != _shaders.end()) {
        *out = it->second.get();
        return Status::success();
    }
    // Not precompiled at create: ragged chunk tails and (rows, bucket) pairs a
    // prompt rule can produce that the create-time policy did not enumerate.
    // Bounded, counted and LOGGED, because a compile on the request path is
    // exactly the TTFT stall this engine is not allowed to hide.
    if (_attention_bucket_lazy >= _opt.max_attention_bucket_lazy) {
        return err(StatusCode::unsupported,
                   "attention bucket lazy-compile bound exceeded",
                   luisa::string{"rows "} + luisa::string{std::to_string(rows).c_str()} +
                       ", key capacity " + luisa::string{std::to_string(cap).c_str()} +
                       ", max_attention_bucket_lazy " +
                       luisa::string{std::to_string(_opt.max_attention_bucket_lazy).c_str()});
    }
    _attention_bucket_lazy++;
    LUISA_INFO("tile attention: compiling bucket shader on the REQUEST path "
               "(rows={}, key capacity={}, lazy {}/{}) -- precompile() did not "
               "enumerate this pair",
               rows, cap, _attention_bucket_lazy, _opt.max_attention_bucket_lazy);
    auto capture = make_capture();
    luisa::string detail;
    return shader_for(device, key, capture, out, detail);
}

[[nodiscard]] const Buffer<float> *QwenTileEngine::take_ring_pair(
    uint32_t slot, const Buffer<float> *&out) noexcept {
    const auto parity = _gdn_ring_parity[slot] & 1u;
    _gdn_ring_parity[slot] = parity ^ 1u;
    out = &_gdn_rings[2u * slot + (parity ^ 1u)];
    return &_gdn_rings[2u * slot + parity];
}

[[nodiscard]] Status QwenTileEngine::precompile(Device &device, Stream &stream) {
    const t::Shader *sh = nullptr;
    const auto hidden = _cfg.hidden;
    const auto ffn = _cfg.ffn;
    const auto vocab = _cfg.vocab;
    const auto fmt = static_cast<uint64_t>(_format);
    // RoPE sin/cos tables: host-precomputed FP32 (double-precision libm angle,
    // single rounding), indexed by absolute position. Uploaded once; the
    // stream is synchronized before the staging vectors go out of scope.
    {
        luisa::vector<float> cos_tab, sin_tab;
        if (auto r = to::tile_host_rope_tables(_cfg.rope_dim, _cfg.rope_theta, _max_context,
                                               cos_tab, sin_tab); !r.ok()) {
            return invalid("rope table computation failed", r.message);
        }
        stream << _rope_cos.copy_from(luisa::span<const float>{cos_tab})
               << _rope_sin.copy_from(luisa::span<const float>{sin_tab}) << compute::synchronize();
    }
    // Embedding for every chunk size that can occur (1..prefill_chunk and the
    // ragged tail sizes are covered by 1..C; precompile all to avoid mid-run JIT).
    for (uint32_t rows = 1; rows <= _prefill_chunk; rows++) {
        Key k; k.add("emb", fmt).add("t", rows).add("h", hidden).add("v", vocab);
        if (auto s = shader_lazy(device, k.s, [&] { return to::capture_tile_embedding(_format, rows, hidden, vocab); }, &sh); !s.ok()) { return s; }
        // RMSNorm plain over [rows, hidden] (input/post norms) and [1, hidden] (LM head).
        Key kn; kn.add("rms", 0).add("r", rows).add("d", hidden).add("eps", _cfg.norm_eps);
        if (auto s = shader_lazy(device, kn.s, [&] { return to::capture_tile_rmsnorm(to::TileNormEpilogue::plain, rows, hidden, _cfg.norm_eps); }, &sh); !s.ok()) { return s; }
        // SwiGLU split over rows*ffn.
        Key ks; ks.add("swiglu", rows * ffn);
        if (auto s = shader_lazy(device, ks.s, [&] { return to::capture_tile_swiglu_split(rows * ffn); }, &sh); !s.ok()) { return s; }
        // Residual add over rows*hidden.
        Key ka; ka.add("add", rows * hidden);
        if (auto s = shader_lazy(device, ka.s, [&] { return to::capture_tile_residual_add(rows * hidden); }, &sh); !s.ok()) { return s; }
    }
    // LM-head norm (1 row) + zero fills for state.
    {
        Key kn; kn.add("rms", 0).add("r", 1).add("d", hidden).add("eps", _cfg.norm_eps);
        if (auto s = shader_lazy(device, kn.s, [&] { return to::capture_tile_rmsnorm(to::TileNormEpilogue::plain, 1, hidden, _cfg.norm_eps); }, &sh); !s.ok()) { return s; }
    }
    // GEMM specializations for every (rows, n, k, epilogue) used by the model.
    // rows in 1..C; the distinct (n,k) pairs are enumerated from the weights.
    struct NK { uint64_t n, k; bool operator==(const NK &o) const noexcept { return n == o.n && k == o.k; } };
    luisa::vector<NK> shapes;
    const auto add_nk = [&](uint64_t n, uint64_t k) {
        NK nk{n, k};
        for (auto &s : shapes) { if (s == nk) { return; } }
        shapes.push_back(nk);
    };
    add_nk(vocab, hidden);// LM head
    for (uint32_t l = 0; l < _layers; l++) {
        const auto &lw = _layer_weights[l];
        add_nk(lw.ffn_gate->dims[1], lw.ffn_gate->dims[0]);
        add_nk(lw.ffn_down->dims[1], lw.ffn_down->dims[0]);
        if (lw.full_attention) {
            add_nk(lw.attn_q->dims[1], lw.attn_q->dims[0]);
            add_nk(lw.attn_k->dims[1], lw.attn_k->dims[0]);
            add_nk(lw.attn_output->dims[1], lw.attn_output->dims[0]);
        } else {
            add_nk(lw.gdn_qkv->dims[1], lw.gdn_qkv->dims[0]);
            add_nk(lw.gdn_gate->dims[1], lw.gdn_gate->dims[0]);
            add_nk(lw.gdn_alpha->dims[1], lw.gdn_alpha->dims[0]);
            add_nk(lw.gdn_out->dims[1], lw.gdn_out->dims[0]);
        }
    }
    for (uint32_t rows = 1; rows <= _prefill_chunk; rows++) {
        for (auto nk : shapes) {
            for (uint32_t epi = 0; epi < 2; epi++) {
                // The residual epilogue is used for every n=hidden projection:
                // ffn_down, attn o_proj and gdn o_proj (all project directly
                // into _hidden with &_residual).
                if (epi == 1 && nk.n != hidden) { continue; }
                Key kg; kg.add("gemm", fmt).add("e", epi).add("m", rows).add("n", nk.n).add("k", nk.k)
                       .add("tm", _opt.gemm_tile_m).add("tn", _opt.gemm_tile_n);
                if (auto s = shader_lazy(device, kg.s, [&] {
                        to::TileGemmShape gs;
                        gs.m = rows; gs.n = static_cast<uint32_t>(nk.n); gs.k = static_cast<uint32_t>(nk.k);
                        gs.tile_m = _opt.gemm_tile_m; gs.tile_n = _opt.gemm_tile_n;
    // Backend-specific realization proven by A/B: one-row MMA is fastest on
    // DX but reproduces DEVICE_LOST on VK; VK uses the safe reduction path.
    gs.gemv_mma = device.backend_name() == luisa::string_view{"dx"};
                        return to::capture_tile_gemm(_format, epi == 1 ? to::TileMatmulEpilogue::residual
                                                                        : to::TileMatmulEpilogue::none, gs);
                    }, &sh); !s.ok()) { return s; }
            }
        }
    }
    // Attention prepare / rope / kv-append / attention for each chunk size.
    for (uint32_t rows = 1; rows <= _prefill_chunk; rows++) {
        Key kpq; kpq.add("prepq", rows).add("qh", _cfg.q_heads).add("hd", _cfg.attn_head_dim);
        if (auto s = shader_lazy(device, kpq.s, [&] { return to::capture_tile_attn_prepare_q(_cfg, rows); }, &sh); !s.ok()) { return s; }
        Key kpk; kpk.add("prepk", rows).add("kvh", _cfg.kv_heads).add("hd", _cfg.attn_head_dim);
        if (auto s = shader_lazy(device, kpk.s, [&] { return to::capture_tile_attn_prepare_k(_cfg, rows); }, &sh); !s.ok()) { return s; }
        Key kpv; kpv.add("prepv", rows).add("kvh", _cfg.kv_heads).add("hd", _cfg.attn_head_dim);
        if (auto s = shader_lazy(device, kpv.s, [&] { return to::capture_tile_attn_prepare_v(_cfg, rows); }, &sh); !s.ok()) { return s; }
        Key krq; krq.add("ropet", rows).add("h", _cfg.q_heads).add("hd", _cfg.attn_head_dim).add("rd", _cfg.rope_dim).add("mc", _max_context);
        if (auto s = shader_lazy(device, krq.s, [&] { return to::capture_tile_rope_partial_table(rows, _cfg.q_heads, _cfg.attn_head_dim, _cfg.rope_dim, _max_context); }, &sh); !s.ok()) { return s; }
        Key krk; krk.add("ropet", rows).add("h", _cfg.kv_heads).add("hd", _cfg.attn_head_dim).add("rd", _cfg.rope_dim).add("mc", _max_context);
        if (auto s = shader_lazy(device, krk.s, [&] { return to::capture_tile_rope_partial_table(rows, _cfg.kv_heads, _cfg.attn_head_dim, _cfg.rope_dim, _max_context); }, &sh); !s.ok()) { return s; }
    }
    // KV append is position-specialized (pos0 static). Precompile the decode
    // position 0..? No -- positions vary at runtime. The append capture takes
    // pos0 statically, so we compile on demand keyed by (rows, pos0). To bound
    // mid-run JIT we precompile nothing here; the cache handles it. (The
    // dynamic attention shaders below are position-independent.)
    if (_opt.dynamic_attention) {
        // Canonical, BOUNDED length-bucket list: 0 disables bucketing and
        // streams the whole capacity (the historical behaviour), otherwise every
        // multiple of the bucket below max_context plus max_context itself for a
        // partially filled last bucket.
        _attention_caps.clear();
        if (_opt.attention_bucket_keys == 0u) {
            _attention_caps.push_back(_max_context);
        } else {
            for (uint64_t cap = _opt.attention_bucket_keys; cap < _max_context;
                 cap += _opt.attention_bucket_keys) {
                _attention_caps.push_back(static_cast<uint32_t>(cap));
            }
            _attention_caps.push_back(_max_context);
        }
        // Precompile policy: EVERY decode bucket (the latency-critical path) and
        // EVERY bucket at the full configured chunk size (the steady prefill
        // path), plus every chunk size at the SMALLEST bucket (a prompt that fits
        // in one chunk lands here). Anything else -- in practice a ragged prefill
        // tail above the smallest bucket -- uses the bounded, logged lazy cache
        // in attention_shader(); no path silently falls back to the full
        // capacity.
        const auto count_new = [&](size_t before) {
            _attention_bucket_precompiled += static_cast<uint32_t>(_shaders.size() - before);
        };
        for (auto cap : _attention_caps) {
            Key kd; kd.add("attndec_dyn", 0).add("cap", cap).add("qh", _cfg.q_heads).add("kvh", _cfg.kv_heads).add("hd", _cfg.attn_head_dim);
            const auto before = _shaders.size();
            if (auto s = shader_lazy(device, kd.s, [&] { return to::capture_tile_attn_decode_dynamic(_cfg, cap); }, &sh); !s.ok()) { return s; }
            count_new(before);
        }
        for (auto cap : _attention_caps) {
            Key kpf; kpf.add("attnpf_dyn", _prefill_chunk).add("cap", cap).add("qh", _cfg.q_heads).add("kvh", _cfg.kv_heads).add("hd", _cfg.attn_head_dim);
            const auto before = _shaders.size();
            if (auto s = shader_lazy(device, kpf.s, [&] { return to::capture_tile_attn_prefill_dynamic(_cfg, _prefill_chunk, cap); }, &sh); !s.ok()) { return s; }
            count_new(before);
        }
        if (_prefill_chunk > 1u) {
            const auto smallest = _attention_caps.front();
            for (uint32_t rows = 1; rows < _prefill_chunk; rows++) {
                Key kpf; kpf.add("attnpf_dyn", rows).add("cap", smallest).add("qh", _cfg.q_heads).add("kvh", _cfg.kv_heads).add("hd", _cfg.attn_head_dim);
                const auto before = _shaders.size();
                if (auto s = shader_lazy(device, kpf.s, [&] { return to::capture_tile_attn_prefill_dynamic(_cfg, rows, smallest); }, &sh); !s.ok()) { return s; }
                count_new(before);
            }
        }
        LUISA_INFO("tile attention: {} buckets ([{}..{}] keys), {} shaders precompiled, "
                   "{} bounded lazy slots",
                   _attention_caps.size(), _attention_caps.front(),
                   _attention_caps.back(), _attention_bucket_precompiled,
                   _opt.max_attention_bucket_lazy);
    }
    // Conv ping-pong step: exactly (kernel-1) pos_mod variants, shared by all
    // positions (the absolute pos is device-visible). No per-position JIT.
    // Kept even when the chunk variant is selected: it is the differential and
    // A-B path and costs nothing at run time.
    for (uint32_t pm = 0; pm + 1u < _cfg.conv_kernel; pm++) {
        Key kc; kc.add("gdnconvstep", _cfg.gdn_qkv_n()).add("ks", _cfg.conv_kernel).add("pm", pm);
        if (auto s = shader_lazy(device, kc.s, [&] { return to::capture_tile_gdn_conv_silu_step(_cfg.gdn_qkv_n(), _cfg.conv_kernel, pm); }, &sh); !s.ok()) { return s; }
    }
    // Multi-token conv chunk: one specialization per chunk size that can
    // actually select it (more than one row), position device-visible (no
    // per-position or per-token compile). With prefill_chunk == 1 the kernel can
    // never be used, so nothing is compiled for it.
    if (_opt.chunk_conv && _prefill_chunk > 1u) {
        for (uint32_t rows = 2; rows <= _prefill_chunk; rows++) {
            Key kc; kc.add("gdnconvchunk", _cfg.gdn_qkv_n()).add("ks", _cfg.conv_kernel).add("t", rows);
            if (auto s = shader_lazy(device, kc.s, [&] { return to::capture_tile_gdn_conv_silu_chunk(_cfg.gdn_qkv_n(), _cfg.conv_kernel, rows); }, &sh); !s.ok()) { return s; }
        }
    }
    // GDN per-chunk kernels.
    for (uint32_t rows = 1; rows <= _prefill_chunk; rows++) {
        Key kg; kg.add("gdngates", rows).add("vh", _cfg.gdn_v_heads).add("sp", _cfg.softplus_threshold);
        if (auto s = shader_lazy(device, kg.s, [&] { return to::capture_tile_gdn_gates(rows, _cfg.gdn_v_heads, _cfg.softplus_threshold); }, &sh); !s.ok()) { return s; }
        Key kq; kq.add("gdnq", rows).add("qkh", _cfg.gdn_qk_heads).add("hd", _cfg.gdn_head_dim)
                 .add("vc", _cfg.gdn_v_n()).add("eps", _cfg.l2_eps).add("qs", _cfg.gdn_q_scale());
        if (auto s = shader_lazy(device, kq.s, [&] { return to::capture_tile_gdn_q_l2norm(rows, _cfg.gdn_qk_heads, _cfg.gdn_head_dim, _cfg.gdn_v_n(), _cfg.l2_eps, _cfg.gdn_q_scale()); }, &sh); !s.ok()) { return s; }
        Key kk; kk.add("gdnk", rows).add("qkh", _cfg.gdn_qk_heads).add("hd", _cfg.gdn_head_dim)
                 .add("vc", _cfg.gdn_v_n()).add("eps", _cfg.l2_eps);
        if (auto s = shader_lazy(device, kk.s, [&] { return to::capture_tile_gdn_k_l2norm(rows, _cfg.gdn_qk_heads, _cfg.gdn_head_dim, _cfg.gdn_v_n(), _cfg.l2_eps); }, &sh); !s.ok()) { return s; }
        Key kr; kr.add("gdnreccols", rows).add("vh", _cfg.gdn_v_heads).add("qkh", _cfg.gdn_qk_heads).add("hd", _cfg.gdn_head_dim);
        if (auto s = shader_lazy(device, kr.s, [&] { return to::capture_tile_gdn_recurrent_columns(rows, _cfg.gdn_v_heads, _cfg.gdn_qk_heads, _cfg.gdn_head_dim); }, &sh); !s.ok()) { return s; }
        Key kn; kn.add("gdngnorm", rows).add("vh", _cfg.gdn_v_heads).add("hd", _cfg.gdn_head_dim).add("eps", _cfg.norm_eps);
        if (auto s = shader_lazy(device, kn.s, [&] { return to::capture_tile_gdn_gated_norm(rows, _cfg.gdn_v_heads, _cfg.gdn_head_dim, _cfg.norm_eps); }, &sh); !s.ok()) { return s; }
    }
    // Zero-fill shaders for state reset (sizes are static).
    {
        const uint64_t state_elems = static_cast<uint64_t>(_cfg.gdn_v_heads) * _cfg.gdn_head_dim * _cfg.gdn_head_dim;
        const uint64_t ring_elems = static_cast<uint64_t>(_cfg.gdn_qkv_n()) * (_cfg.conv_kernel - 1u);
        for (auto n : {state_elems, ring_elems}) {
            Key kz; kz.add("zero", n);
            if (auto s = shader_lazy(device, kz.s, [&] { return to::capture_tile_zero_fill(static_cast<uint32_t>(n)); }, &sh); !s.ok()) { return s; }
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenTileEngine::zero_buffer(QwenCommandBatch &cl, Device &device, Buffer<float> &buf,
                                                 luisa::string_view name) {
    const auto n = u32(buf.size(), "reset buffer size");
    if (!n.ok()) { return n.status; }
    if (*n.value == 0u) { return Status::success(); }
    Key k; k.add("zero", buf.size());
    const t::Shader *sh = nullptr;
    if (auto s = shader_lazy(device, k.s, [&] { return to::capture_tile_zero_fill(*n.value); }, &sh); !s.ok()) { return s; }
    cl.annotate(name) << (*sh)(buf.view()).dispatch();
    return Status::success();
}

[[nodiscard]] Status QwenTileEngine::reset(Stream &stream) {
    // Fresh report for a standalone reset; prefill appends after this reset.
    if (_benchmark) { _report.clear(); }
    // The allocator drains the stream before staging/state reuse and before
    // its host-immediate CUDA unmap path; avoid a duplicate engine-level sync.
    if (auto s = _kv_cache.reset(stream); !s.ok()) { return s; }
    QwenCommandBatch cl{stream, _benchmark ? &_report : nullptr, QwenBenchmarkPhase::reset, 0u, 0u};
    cl.layer(UINT32_MAX);
    Device device = _device;
    // GDN recurrent states and ping-pong conv rings, in layer order.
    for (uint32_t l = 0; l < _layers; l++) {
        const uint32_t slot = _gdn_slot[l];
        if (slot == UINT32_MAX) { continue; }
        const auto ls = luisa::string{std::to_string(l)};
        cl.layer(l);
        if (auto s = zero_buffer(cl, device, _gdn_states[slot], luisa::string{"reset.gdn_state.l"} + ls); !s.ok()) { return s; }
        if (auto s = zero_buffer(cl, device, _gdn_rings[2u * slot], luisa::string{"reset.gdn_ring0.l"} + ls); !s.ok()) { return s; }
        if (auto s = zero_buffer(cl, device, _gdn_rings[2u * slot + 1u], luisa::string{"reset.gdn_ring1.l"} + ls); !s.ok()) { return s; }
        // Both rings are zero, so the pair parity restarts at 0
        // (_gdn_rings[2*slot] is the read buffer of the next invocation).
        _gdn_ring_parity[slot] = 0u;
    }
    cl.layer(UINT32_MAX);
    // The persistent zero operand for the residual-snapshot copy.
    if (auto s = zero_buffer(cl, device, _zero, "reset.zero_operand"); !s.ok()) { return s; }
    // Conv source scratch: the padding lanes beyond qkv_n must stay zero.
    if (auto s = zero_buffer(cl, device, _conv_src, "reset.conv_src_pad"); !s.ok()) { return s; }
    cl.flush();
    _position = 0;
    _has_session = false;
    return Status::success();
}

// ---------------------------------------------------------------------------
// Staging (persistent host buffers; see the lifetime rule in the header).
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenTileEngine::stage_ids(QwenCommandBatch &cl, luisa::span<const uint32_t> ids, uint32_t pos0) {
    const auto rows = static_cast<uint32_t>(ids.size());
    // Absolute-offset span [pos0, pos0+rows): distinct per chunk, never
    // rewritten before the owning batch commits (the staged cursor IS pos0).
    if (static_cast<uint64_t>(pos0) + rows > _host_ids.size()) { return invalid("staging overflow: token rows"); }
    for (uint32_t t = 0; t < rows; t++) { _host_ids[pos0 + t] = static_cast<int32_t>(ids[t]); }
    cl.annotate("upload.token_ids")
      << _ids.view(0, rows).copy_from(luisa::span<const int32_t>{_host_ids.data() + pos0, rows});
    return Status::success();
}

[[nodiscard]] Status QwenTileEngine::stage_positions(QwenCommandBatch &cl, uint32_t rows, uint32_t pos0) {
    // Absolute-offset span [pos0, pos0+rows), shared by the RoPE kernels and
    // the per-token conv position uploads of every layer in this chunk.
    if (static_cast<uint64_t>(pos0) + rows > _host_pos.size()) { return invalid("staging overflow: position rows"); }
    for (uint32_t t = 0; t < rows; t++) { _host_pos[pos0 + t] = static_cast<int32_t>(pos0 + t); }
    cl.annotate("upload.rope_positions")
      << _positions.view(0, rows).copy_from(luisa::span<const int32_t>{_host_pos.data() + pos0, rows});
    return Status::success();
}

[[nodiscard]] Status QwenTileEngine::stage_attn_len(QwenCommandBatch &cl, uint32_t pos0, uint32_t kv_len) {
    // Fresh pair slot [2*pos0, 2*pos0+2) per chunk; earlier chunks' slots are
    // never rewritten before the owning batch commits.
    if (2ull * pos0 + 2ull > _host_len.size()) { return invalid("staging overflow: attention length slots"); }
    const auto *slot = _host_len.data() + 2u * static_cast<size_t>(pos0);
    _host_len[2u * static_cast<size_t>(pos0)] = static_cast<int32_t>(pos0);
    _host_len[2u * static_cast<size_t>(pos0) + 1u] = static_cast<int32_t>(kv_len);
    cl.annotate("upload.attn_len")
      << _attn_len.view(0, 2).copy_from(luisa::span<const int32_t>{slot, 2});
    cl.annotate("upload.kv_len")
      << _kv_len.view(0, 1).copy_from(luisa::span<const int32_t>{slot + 1, 1});
    return Status::success();
}

// ---------------------------------------------------------------------------
// Projection helper (packed Tile GEMM; rows==1 uses the same specialization).
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenTileEngine::project(Device &device, Stream &stream, QwenCommandBatch &cl,
                                             const Buffer<float> &x, const QwenDeviceTensor &w,
                                             const Buffer<float> &y, uint32_t rows,
                                             const Buffer<float> *residual, luisa::string_view name) {
    const auto k = u32(w.dims[0], "projection K");
    const auto n = u32(w.dims[1], "projection N");
    if (!k.ok()) { return k.status; }
    if (!n.ok()) { return n.status; }
    const uint32_t epi = residual != nullptr ? 1u : 0u;
    // Benchmark labels derived from the projection name (static string_views;
    // suffixes are string literals that outlive every sample). In normal mode
    // these annotations are inert.
    luisa::string seg_name, acc_name, epi_name, copy_name;
    if (cl.instrumented()) {
        seg_name = luisa::string{name} + ".kseg";
        acc_name = luisa::string{name} + ".acc";
        epi_name = luisa::string{name} + (epi != 0u ? ".residual" : ".store");
        copy_name = luisa::string{name} + ".store_copy";
    }
    const auto scales = w.scales ? w.scales.view() : _dummy_f.view();
    const t::Shader *sh = nullptr;
    to::TileGemmShape gs;
    gs.m = rows; gs.n = *n.value; gs.k = *k.value;
    gs.tile_m = _opt.gemm_tile_m; gs.tile_n = _opt.gemm_tile_n;
    // Large-K projections (ffn_down K=9216) exceed the serial K-loop size the
    // VK backend executes reliably in a single dispatch (K<=4096 is fine;
    // K=8192 hangs, K=9216 loses the device). Split K into <=4096 segments,
    // accumulate the partials, and apply the residual after the last segment.
    // The segment order is fixed (ascending K), so the FP32 accumulation is
    // deterministic. Small-K projections use the exact single-dispatch GEMM.
    constexpr uint32_t kSegMax = 4096;
    if (*k.value > kSegMax && *k.value % to::kTileGemmKBlock != 0u) {
        return err(StatusCode::unsupported,
                   "large-K Tile projection requires block-aligned K; refusing the unstable single-dispatch fallback",
                   w.name);
    }
    if (*k.value > kSegMax) {
        const uint32_t nseg = (*k.value + kSegMax - 1u) / kSegMax;
        // Round the nominal segment UP to the Tile K block. Rounding down can
        // push the final segment above kSegMax (e.g. K=12224). With ceil(K/N)
        // sizing, every fixed segment and the final tail are <= kSegMax.
        const uint32_t nominal = (*k.value + nseg - 1u) / nseg;
        const uint32_t seg = ((nominal + to::kTileGemmKBlock - 1u) /
                              to::kTileGemmKBlock) * to::kTileGemmKBlock;
        // Accumulate the K-segment partials. NO residual_add ever aliases an
        // input with its output (the XIR route mishandles in-place a/out
        // aliasing): partials go to _proj_c, the running sum ping-pongs between
        // _proj_b and _kseg_acc, and the final result is written to y.
        const auto count = u32(static_cast<uint64_t>(rows) * *n.value, "K-split accumulation size");
        if (!count.ok()) { return count.status; }
        Key ka; ka.add("add", *count.value);
        const t::Shader *ash = nullptr;
        if (auto st = shader_lazy(device, ka.s, [&] { return to::capture_tile_residual_add(*count.value); }, &ash); !st.ok()) { return st; }
        const Buffer<float> *acc = &_proj_b;
        for (uint32_t s = 0; s < nseg; s++) {
            const uint32_t ko = s * seg;
            const uint32_t ks = (s + 1u == nseg) ? (*k.value - ko) : seg;
            if (ks == 0u || ks > kSegMax || ks % to::kTileGemmKBlock != 0u) {
                return err(StatusCode::internal_error, "invalid Tile K-segment plan", w.name);
            }
            Key key; key.add("gemmkseg", static_cast<uint64_t>(_format)).add("m", rows)
                        .add("n", *n.value).add("k", *k.value).add("ko", ko).add("ks", ks)
                        .add("tm", _opt.gemm_tile_m).add("tn", _opt.gemm_tile_n)
                        .add("gvm", gs.gemv_mma);
            if (auto st = shader_lazy(device, key.s, [&] {
                    return to::capture_tile_gemm_kseg(_format, gs, ko, ks);
                }, &sh); !st.ok()) { return st; }
            cl.annotate(seg_name);
            if (rows == 1u) {
                // The M=1 GEMV shares the general five-argument ABI (the
                // residual input is unused for a partial K segment). The
                // general M>1 K-segment kernel retains its four-argument ABI.
                cl << (*sh)(x.view(), w.packed.view(), scales, _dummy_f2.view(), _proj_c.view()).dispatch();
            } else {
                cl << (*sh)(x.view(), w.packed.view(), scales, _proj_c.view()).dispatch();
            }
            if (s == 0) {
                // acc = partial (add with the zero operand; out != inputs).
                cl.annotate(acc_name) << (*ash)(_proj_c.view(), _zero.view(), acc->view()).dispatch();
            } else {
                // next = acc + partial, written to the buffer that is not acc.
                const Buffer<float> *next = (acc == &_proj_b) ? &_kseg_acc : &_proj_b;
                cl.annotate(acc_name) << (*ash)(acc->view(), _proj_c.view(), next->view()).dispatch();
                acc = next;
            }
        }
        // Write the accumulated product to y, adding the residual if present.
        // When y IS the residual buffer (ffn_down in-place), route through a
        // scratch so no residual_add aliases an input with its output.
        if (residual != nullptr) {
            const bool inplace = (&y == residual);
            if (inplace) {
                const Buffer<float> *next = (acc == &_proj_b) ? &_kseg_acc : &_proj_b;
                cl.annotate(epi_name) << (*ash)(acc->view(), residual->view(), next->view()).dispatch();
                cl.annotate(copy_name) << (*ash)(next->view(), _zero.view(), y.view()).dispatch();
            } else {
                cl.annotate(epi_name) << (*ash)(acc->view(), residual->view(), y.view()).dispatch();
            }
        } else {
            cl.annotate(epi_name) << (*ash)(acc->view(), _zero.view(), y.view()).dispatch();
        }
        return Status::success();
    }
    Key key; key.add("gemm", static_cast<uint64_t>(_format)).add("e", epi).add("m", rows)
                .add("n", *n.value).add("k", *k.value)
                .add("tm", _opt.gemm_tile_m).add("tn", _opt.gemm_tile_n)
                .add("gvm", gs.gemv_mma);
    if (auto s = shader_lazy(device, key.s, [&] {
            return to::capture_tile_gemm(_format, epi == 1 ? to::TileMatmulEpilogue::residual
                                                           : to::TileMatmulEpilogue::none, gs);
        }, &sh); !s.ok()) { return s; }
    cl.annotate(name);
    if (residual != nullptr) {
        cl << (*sh)(x.view(), w.packed.view(), scales, residual->view(), y.view()).dispatch();
    } else {
        cl << (*sh)(x.view(), w.packed.view(), scales, _dummy_f2.view(), y.view()).dispatch();
    }
    return Status::success();
}

[[nodiscard]] Status QwenTileEngine::rmsnorm(Device &device, QwenCommandBatch &cl,
                                             const BufferView<float> &x, const Buffer<float> &w,
                                             const BufferView<float> &y, uint32_t rows, uint32_t dim,
                                             luisa::string_view name) {
    Key key; key.add("rms", 0).add("r", rows).add("d", dim).add("eps", _cfg.norm_eps);
    const t::Shader *sh = nullptr;
    if (auto s = shader_lazy(device, key.s, [&] {
            return to::capture_tile_rmsnorm(to::TileNormEpilogue::plain, rows, dim, _cfg.norm_eps);
        }, &sh); !s.ok()) { return s; }
    cl.annotate(name) << (*sh)(x, w.view(), _dummy_f2.view(), y, _dummy_f2.view()).dispatch();
    return Status::success();
}

// ---------------------------------------------------------------------------
// Full attention block.
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenTileEngine::full_attention_block(Device &device, Stream &stream, QwenCommandBatch &cl,
                                                          uint32_t layer, uint32_t rows, uint32_t pos0) {
    cl.layer(layer);
    const auto &lw = _layer_weights[layer];
    const uint32_t slot = _full_slot[layer];
    const auto kc = _kv_cache.k_view(slot);
    const auto vc = _kv_cache.v_view(slot);
    const uint32_t hd = _cfg.attn_head_dim;
    const uint32_t qh = _cfg.q_heads, kvh = _cfg.kv_heads;
    const uint32_t kv_n = _cfg.attn_kv_n();
    const t::Shader *sh = nullptr;
    // q/k/v projections.
    if (auto s = project(device, stream, cl, _normed, *lw.attn_q, _attn_proj_q, rows, nullptr, "attn.q_proj"); !s.ok()) { return s; }
    if (auto s = project(device, stream, cl, _normed, *lw.attn_k, _attn_proj_k, rows, nullptr, "attn.k_proj"); !s.ok()) { return s; }
    if (auto s = project(device, stream, cl, _normed, *lw.attn_v, _attn_proj_v, rows, nullptr, "attn.v_proj"); !s.ok()) { return s; }
    // Prepare: three host-selected kernels (Q+gate / K / V), each writing
    // only its own outputs with in-bounds stores. Zero-centered per-head QK
    // norm (DIRECT pre-added GGUF weight), no RoPE yet.
    {
        Key key; key.add("prepq", rows).add("qh", qh).add("hd", hd);
        if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_attn_prepare_q(_cfg, rows); }, &sh); !s.ok()) { return s; }
        cl.annotate("attn.prepare_q")
          << (*sh)(_attn_proj_q.view(), lw.attn_q_norm->f32.view(),
                   _attn_q.view(), _attn_gate.view()).dispatch();
    }
    {
        Key key; key.add("prepk", rows).add("kvh", kvh).add("hd", hd);
        if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_attn_prepare_k(_cfg, rows); }, &sh); !s.ok()) { return s; }
        cl.annotate("attn.prepare_k")
          << (*sh)(_attn_proj_k.view(), lw.attn_k_norm->f32.view(), _attn_k.view()).dispatch();
    }
    {
        Key key; key.add("prepv", rows).add("kvh", kvh).add("hd", hd);
        if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_attn_prepare_v(_cfg, rows); }, &sh); !s.ok()) { return s; }
        cl.annotate("attn.prepare_v")
          << (*sh)(_attn_proj_v.view(), _attn_v.view()).dispatch();
    }
    // Separate NeoX rotate-half partial RoPE on Q and K (host tables).
    // _positions was uploaded ONCE by forward_chunk for this chunk and is
    // shared by every full-attention layer (values are layer-independent).
    {
        Key key; key.add("ropet", rows).add("h", qh).add("hd", hd).add("rd", _cfg.rope_dim).add("mc", _max_context);
        if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_rope_partial_table(rows, qh, hd, _cfg.rope_dim, _max_context); }, &sh); !s.ok()) { return s; }
        cl.annotate("attn.rope_q")
          << (*sh)(_attn_q.view(), _positions.view(), _rope_cos.view(), _rope_sin.view(), _attn_qr.view()).dispatch();
    }
    {
        Key key; key.add("ropet", rows).add("h", kvh).add("hd", hd).add("rd", _cfg.rope_dim).add("mc", _max_context);
        if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_rope_partial_table(rows, kvh, hd, _cfg.rope_dim, _max_context); }, &sh); !s.ok()) { return s; }
        cl.annotate("attn.rope_k")
          << (*sh)(_attn_k.view(), _positions.view(), _rope_cos.view(), _rope_sin.view(), _attn_kr.view()).dispatch();
    }
    // KV append at absolute positions pos0..pos0+rows-1 (pos0-static capture).
    {
        Key key; key.add("kvapp", rows).add("c", kv_n).add("p0", pos0).add("cap", _max_context);
        if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_kv_append(rows, kv_n, pos0, _max_context); }, &sh); !s.ok()) { return s; }
        cl.annotate("attn.kv_append")
          << (*sh)(_attn_kr.view(), _attn_v.view(), kc, vc).dispatch();
    }
    const uint32_t kv_len = pos0 + rows;
    // Static key range of the device-visible-length shader: the length bucket
    // ceil(kv_len/attention_bucket_keys)*bucket clamped to max_context (the full
    // capacity when bucketing is disabled). The length itself stays
    // device-visible, so one shader per bucket serves every position inside it;
    // pos0+tokens <= kv_len <= cap holds, which is the captures' requirement.
    const uint32_t attn_cap = attention_capacity(kv_len);
    // _attn_len/_kv_len were uploaded ONCE by forward_chunk for this chunk;
    // every full-attention layer sees the same {pos0, kv_len}.
    if (rows == 1u) {
        // Decode: one query at absolute position pos0 == kv_len-1.
        if (_opt.dynamic_attention) {
            Key key; key.add("attndec_dyn", 0).add("cap", attn_cap).add("qh", qh).add("kvh", kvh).add("hd", hd);
            if (auto s = attention_shader(device, key.s, 0u, attn_cap, [&] { return to::capture_tile_attn_decode_dynamic(_cfg, attn_cap); }, &sh); !s.ok()) { return s; }
            cl.annotate("attn.decode_dyn")
              << (*sh)(_attn_qr.view(), _attn_gate.view(), kc, vc,
                       _kv_len.view(0, _kv_len_stride), _attn_out.view()).dispatch();
        } else {
            Key key; key.add("attndec", kv_len).add("cap", _max_context).add("qh", qh).add("kvh", kvh).add("hd", hd);
            const bool is_new = _shaders.find(key.s) == _shaders.end();
            if (is_new && _static_attention_count >= _opt.max_static_attention_shaders) {
                return err(StatusCode::unsupported, "static attention shader cache bound exceeded");
            }
            if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_attn_decode(_cfg, kv_len, _max_context); }, &sh); !s.ok()) { return s; }
            if (is_new) { _static_attention_count++; }
            cl.annotate("attn.decode_static")
              << (*sh)(_attn_qr.view(), _attn_gate.view(), kc, vc, _attn_out.view()).dispatch();
        }
    } else {
        if (_opt.dynamic_attention) {
            Key key; key.add("attnpf_dyn", rows).add("cap", attn_cap).add("qh", qh).add("kvh", kvh).add("hd", hd);
            if (auto s = attention_shader(device, key.s, rows, attn_cap, [&] { return to::capture_tile_attn_prefill_dynamic(_cfg, rows, attn_cap); }, &sh); !s.ok()) { return s; }
            cl.annotate("attn.prefill_dyn")
              << (*sh)(_attn_qr.view(), _attn_gate.view(), kc, vc,
                       _attn_len.view(), _attn_out.view()).dispatch();
        } else {
            Key key; key.add("attnpf", rows).add("p0", pos0).add("kv", kv_len).add("cap", _max_context)
                        .add("qh", qh).add("kvh", kvh).add("hd", hd);
            const bool is_new = _shaders.find(key.s) == _shaders.end();
            if (is_new && _static_attention_count >= _opt.max_static_attention_shaders) {
                return err(StatusCode::unsupported, "static attention shader cache bound exceeded");
            }
            if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_attn_prefill(_cfg, rows, pos0, kv_len, _max_context); }, &sh); !s.ok()) { return s; }
            if (is_new) { _static_attention_count++; }
            cl.annotate("attn.prefill_static")
              << (*sh)(_attn_qr.view(), _attn_gate.view(), kc, vc, _attn_out.view()).dispatch();
        }
    }
    // Output projection with fused residual epilogue:
    // hidden = residual + attn_out @ o_proj, written directly to _hidden
    // (y != residual, so the GEMM epilogue never aliases). K = o_proj_k
    // (4096) takes the exact single-dispatch residual path in project();
    // larger K keeps the alias-free K-split accumulation there. Same FP32
    // math as the former separate attn.residual add kernel.
    if (auto s = project(device, stream, cl, _attn_out, *lw.attn_output, _hidden, rows, &_residual, "attn.o_proj"); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// GDN block.
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenTileEngine::gdn_block(Device &device, Stream &stream, QwenCommandBatch &cl,
                                               uint32_t layer, uint32_t rows, uint32_t pos0) {
    cl.layer(layer);
    const auto &lw = _layer_weights[layer];
    const uint32_t slot = _gdn_slot[layer];
    const auto &state = _gdn_states[slot];
    const uint32_t hd = _cfg.gdn_head_dim;
    const uint32_t qkh = _cfg.gdn_qk_heads, vh = _cfg.gdn_v_heads;
    const uint32_t qkv_n = _cfg.gdn_qkv_n();
    const uint32_t v_n = _cfg.gdn_v_n();
    const t::Shader *sh = nullptr;
    // Flush the recorded command list at a block-internal stage boundary
    // ONLY when _stage_sync (VK and other non-DX backends: long unflushed
    // recordings lose the device even within one block). On DX the stage
    // boundaries never flush; only the every-N-layer bounds in forward_chunk
    // remain, as commit-only flush(false). See
    // QwenTileEngineOptions::sync_every_layers. The flush routes through
    // QwenCommandBatch::flush, which is a no-op when empty (benchmark mode
    // already committed every command individually).
#define QWEN_TS() do { if (_stage_sync && _opt.sync_every_layers != 0u) { cl.flush(); } } while (0)
    // Projections: packed QKV, z (gate), alpha (a), beta (b).
    QWEN_TS();
    // The packed-QKV projection writes a DENSE [rows, qkv_n] buffer: the
    // multi-token conv chunk consumes those rows in place (row stride == qkv_n),
    // so prefill needs no per-token source staging copy.
    if (auto s = project(device, stream, cl, _normed, *lw.gdn_qkv, _gdn_qkv, rows, nullptr, "gdn.qkv_proj"); !s.ok()) { return s; }
    QWEN_TS();
    if (auto s = project(device, stream, cl, _normed, *lw.gdn_gate, _proj_b, rows, nullptr, "gdn.z_proj"); !s.ok()) { return s; }// z
    QWEN_TS();
    if (auto s = project(device, stream, cl, _normed, *lw.gdn_alpha, _proj_c, rows, nullptr, "gdn.alpha_proj"); !s.ok()) { return s; }// a
    // Raw beta projection into a dedicated scratch buffer (_gdn_braw). The
    // gates kernel reads b=_gdn_braw and writes g=_gdn_g; keeping the input
    // and output in separate buffers avoids an in-place read/write alias.
    QWEN_TS();
    if (auto s = project(device, stream, cl, _normed, *lw.gdn_beta, _gdn_braw, rows, nullptr, "gdn.beta_proj"); !s.ok()) { return s; }// b (raw)
    // Conv4 + SiLU.
    //
    // chunk_conv (default): ONE capture_tile_gdn_conv_silu_chunk launch for the
    // whole chunk. It reads the packed-QKV rows in place, carries the causal
    // window in registers, writes the SiLU rows in place and produces the ring
    // image, with the chunk's absolute position device-visible (uploaded ONCE
    // per chunk in forward_chunk, shared by every GDN layer). Per chunk/layer
    // that is 1 command instead of 1 + 3*rows (position upload, source staging
    // copy, conv launch, output gather copy, all per token).
    //
    // Otherwise the historical one-token path: per token a position upload, a
    // source-row staging copy, one conv step launch and an output gather copy,
    // alternating the ring pair (a full history copy per token).
    //
    // Both arms ping-pong the pair per INVOCATION, so ring_in and ring_out are
    // never the same buffer (the Tile runtime rejects a buffer bound to two
    // arguments when either is writable) regardless of the row count.
    {
        const uint32_t r = _cfg.conv_kernel - 1u;
        const auto pos_stride = _conv_pos_stride;
        // A single row keeps the per-token step: it is ONE launch plus three
        // cheap copies, and the generalized chunk kernel's initial window build
        // costs ~2% more than the specialized tap loop at T=1 (measured, see
        // .kimix_cache/qwen_benchmark/phase9d/REPORT.md). Prefill chunks with
        // more than one row use the chunk kernel, where the per-token copies and
        // launches are the whole point.
        if (_opt.chunk_conv && rows > 1u) {
            const Buffer<float> *ring_out = nullptr;
            const auto *ring_in = take_ring_pair(slot, ring_out);
            Key key; key.add("gdnconvchunk", qkv_n).add("ks", _cfg.conv_kernel).add("t", rows);
            if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_gdn_conv_silu_chunk(qkv_n, _cfg.conv_kernel, rows); }, &sh); !s.ok()) { return s; }
            cl.annotate("gdn.conv_silu_chunk")
              << (*sh)(_gdn_qkv.view(), lw.gdn_conv1d->f32.view(),
                       ring_in->view(), _conv_pos.view(0, pos_stride), ring_out->view(),
                       _gdn_conv.view()).dispatch();
        } else {
            const auto src_stride = _conv_src_stride;
            // Stage absolute positions into the aligned per-token pos scratch
            // from the chunk's ABSOLUTE host span [pos0, pos0+rows) that
            // forward_chunk staged once. No host cell is rewritten here: every
            // chunk's span stays intact until the owning batch commits (prefill
            // records all chunks into one batch). Padding lanes are never read
            // (the kernel reads index 0).
            if (static_cast<uint64_t>(pos0) + rows > _host_pos.size()) { return invalid("staging overflow: conv position rows"); }
            cl.annotate("gdn.conv.upload_pos");
            for (uint32_t t = 0; t < rows; t++) {
                cl << _conv_pos.view(t * pos_stride, 1).copy_from(luisa::span<const int32_t>{_host_pos.data() + pos0 + t, 1});
            }
            // Stage the packed QKV rows into the aligned per-token scratch
            // (device-to-device copies; padding lanes stay zero from reset).
            cl.annotate("gdn.conv.stage_src");
            for (uint32_t t = 0; t < rows; t++) {
                cl << _gdn_qkv.view(t * qkv_n, qkv_n).copy_to(_conv_src.view(t * src_stride, qkv_n));
            }
            cl.annotate("gdn.conv_silu_step");
            for (uint32_t t = 0; t < rows; t++) {
                const uint32_t pos = pos0 + t;
                const uint32_t pm = pos % r;
                const Buffer<float> *ring_out = nullptr;
                const auto *ring_in = take_ring_pair(slot, ring_out);
                Key key; key.add("gdnconvstep", qkv_n).add("ks", _cfg.conv_kernel).add("pm", pm);
                if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_gdn_conv_silu_step(qkv_n, _cfg.conv_kernel, pm); }, &sh); !s.ok()) { return s; }
                // Whole-buffer bindings (offset 0, aligned sizes): valid storage
                // descriptors on every backend, including Vulkan's legacy HLSL.
                cl << (*sh)(_conv_src.view(t * src_stride, src_stride), lw.gdn_conv1d->f32.view(),
                            ring_in->view(), _conv_pos.view(t * pos_stride, pos_stride), ring_out->view(),
                            _conv_out.view(t * qkv_n, qkv_n)).dispatch();
            }
            // Gather the per-token outputs into the packed chunk conv buffer.
            cl.annotate("gdn.conv.gather");
            for (uint32_t t = 0; t < rows; t++) {
                cl << _conv_out.view(t * qkv_n, qkv_n).copy_to(_gdn_conv.view(t * qkv_n, qkv_n));
            }
        }
    }
    QWEN_TS();
    // Gates: beta = sigmoid(b); g = ssm_a*softplus(a+dt_bias) -- GGUF ssm_a
    // is ALREADY -exp(A_log) (negative, folded offline), no device exp.
    // a=_proj_c, b=_gdn_braw (raw); outputs beta=_gdn_beta, g=_gdn_g.
    {
        Key key; key.add("gdngates", rows).add("vh", vh).add("sp", _cfg.softplus_threshold);
        if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_gdn_gates(rows, vh, _cfg.softplus_threshold); }, &sh); !s.ok()) { return s; }
        cl.annotate("gdn.gates")
          << (*sh)(_proj_c.view(), _gdn_braw.view(), lw.gdn_a->f32.view(), lw.gdn_dt_bias->f32.view(),
                   _gdn_beta.view(), _gdn_g.view()).dispatch();
    }
    QWEN_TS();
    // Q/K L2 norm + Q scale: two host-selected kernels, each writing only
    // its own output with in-bounds stores.
    {
        Key key; key.add("gdnq", rows).add("qkh", qkh).add("hd", hd).add("vc", v_n)
                    .add("eps", _cfg.l2_eps).add("qs", _cfg.gdn_q_scale());
        if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_gdn_q_l2norm(rows, qkh, hd, v_n, _cfg.l2_eps, _cfg.gdn_q_scale()); }, &sh); !s.ok()) { return s; }
        cl.annotate("gdn.q_l2norm")
          << (*sh)(_gdn_conv.view(), _gdn_q.view()).dispatch();
    }
    {
        Key key; key.add("gdnk", rows).add("qkh", qkh).add("hd", hd).add("vc", v_n)
                    .add("eps", _cfg.l2_eps);
        if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_gdn_k_l2norm(rows, qkh, hd, v_n, _cfg.l2_eps); }, &sh); !s.ok()) { return s; }
        cl.annotate("gdn.k_l2norm")
          << (*sh)(_gdn_conv.view(), _gdn_k.view()).dispatch();
    }
    QWEN_TS();
    // FP32 delta-rule recurrence (same operator for prefill chunks and decode).
    // Column-parallel variant: the [hd,hd] SSA tile carry exceeds the DX/VK
    // XIR snapshot budget at head_dim=128; the column form keeps state in the
    // buffer with scalar carries (zero static snapshots).
    {
        Key key; key.add("gdnreccols", rows).add("vh", vh).add("qkh", qkh).add("hd", hd);
        if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_gdn_recurrent_columns(rows, vh, qkh, hd); }, &sh); !s.ok()) { return s; }
        cl.annotate("gdn.recurrent")
          << (*sh)(_gdn_q.view(), _gdn_k.view(), _gdn_conv.view(),
                   _gdn_beta.view(), _gdn_g.view(), state.view(), _gdn_out.view()).dispatch();
    }
    QWEN_TS();
    // Gated output RMSNorm (direct weight, norm before SiLU(z) gate). z=_proj_b.
    {
        Key key; key.add("gdngnorm", rows).add("vh", vh).add("hd", hd).add("eps", _cfg.norm_eps);
        if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_gdn_gated_norm(rows, vh, hd, _cfg.norm_eps); }, &sh); !s.ok()) { return s; }
        cl.annotate("gdn.gated_norm")
          << (*sh)(_gdn_out.view(), _proj_b.view(), lw.gdn_norm->f32.view(), _gdn_normed.view()).dispatch();
    }
    QWEN_TS();
    // Output projection with fused residual epilogue:
    // hidden = residual + gdn_normed @ out_proj, written directly to _hidden
    // (y != residual, no aliasing). K = gdn_out_k (4096) takes the exact
    // single-dispatch residual path in project(); larger K keeps the
    // alias-free K-split accumulation there. Same FP32 math as the former
    // separate gdn.residual add kernel.
    if (auto s = project(device, stream, cl, _gdn_normed, *lw.gdn_out, _hidden, rows, &_residual, "gdn.o_proj"); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// FFN block (post-attention norm -> gate/up -> SwiGLU -> down + residual).
#undef QWEN_TS
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenTileEngine::ffn_block(Device &device, Stream &stream, QwenCommandBatch &cl,
                                               uint32_t layer, uint32_t rows) {
    cl.layer(layer);
    const auto &lw = _layer_weights[layer];
    const uint32_t hidden = _cfg.hidden, ffn = _cfg.ffn;
    const t::Shader *sh = nullptr;
    // Flush the recorded command list at a block-internal stage boundary
    // ONLY when _stage_sync (VK and other non-DX backends lose the device on
    // long unflushed recordings even within one block). On DX the stage
    // boundaries never flush; only the every-N-layer bounds in forward_chunk
    // remain, as commit-only flush(false). See
    // QwenTileEngineOptions::sync_every_layers. The flush routes through
    // QwenCommandBatch::flush (empty flush = no-op, so benchmark mode adds no
    // redundant empty submissions/synchronizations).
#define QWEN_TS() do { if (_stage_sync && _opt.sync_every_layers != 0u) { cl.flush(); } } while (0)
    if (auto s = rmsnorm(device, cl, _hidden.view(), lw.post_norm->f32, _normed.view(), rows, hidden, "ffn.post_norm"); !s.ok()) { return s; }
    QWEN_TS();
    if (auto s = project(device, stream, cl, _normed, *lw.ffn_gate, _ffn_gate, rows, nullptr, "ffn.gate_proj"); !s.ok()) { return s; }
    QWEN_TS();
    if (auto s = project(device, stream, cl, _normed, *lw.ffn_up, _ffn_up, rows, nullptr, "ffn.up_proj"); !s.ok()) { return s; }
    {
    QWEN_TS();
        const uint32_t count = rows * ffn;
        Key key; key.add("swiglu", count);
        if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_swiglu_split(count); }, &sh); !s.ok()) { return s; }
        cl.annotate("ffn.swiglu")
          << (*sh)(_ffn_gate.view(), _ffn_up.view(), _ffn_act.view()).dispatch();
    }
    QWEN_TS();
    // Down projection with residual: hidden = hidden + down(act). The K-split
    // path in project() accumulates partials alias-free and adds the residual
    // into y; the single-dispatch path uses the GEMM residual epilogue.
    if (auto s = project(device, stream, cl, _ffn_act, *lw.ffn_down, _hidden, rows, &_hidden, "ffn.down_proj"); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Forward one chunk of `rows` tokens at absolute position pos0.
#undef QWEN_TS
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenTileEngine::forward_chunk(Device &device, Stream &stream, QwenCommandBatch &cl,
                                                   uint32_t rows, uint32_t pos0) {
    if (rows == 0u || pos0 > _max_context || rows > _max_context - pos0) {
        return invalid("forward chunk exceeds sparse KV capacity");
    }
    if (auto s = _kv_cache.ensure_capacity(stream, pos0 + rows); !s.ok()) { return s; }
    const uint32_t hidden = _cfg.hidden;
    const t::Shader *sh = nullptr;
    // Upload the chunk-shared dynamic state ONCE: absolute RoPE positions
    // (read by every full-attention layer's rope_q/rope_k) and the dynamic
    // attention lengths {pos0, kv_len} (read by every full-attention layer).
    // Both are layer-independent; the host spans are absolute-offset and
    // never rewritten before this batch commits. The conv steps read the
    // same host position span for their per-token aligned uploads.
    cl.layer(UINT32_MAX);
    if (auto s = stage_positions(cl, rows, pos0); !s.ok()) { return s; }
    if (auto s = stage_attn_len(cl, pos0, pos0 + rows); !s.ok()) { return s; }
    if (_opt.chunk_conv && rows > 1u) {
        // ONE device-visible chunk position per chunk, shared by every GDN
        // layer's multi-token conv kernel (the per-token path uploads one
        // position per token per layer). Host slot = the chunk's absolute start
        // (never rewritten inside the batch); device slot 0, because the copies
        // are stream-ordered ahead of the kernels that read them.
        if (static_cast<uint64_t>(pos0) >= _host_pos.size()) {
            return invalid("staging overflow: conv chunk position");
        }
        cl.annotate("gdn.conv.upload_chunk_pos")
          << _conv_pos.view(0, _conv_pos_stride)
                 .copy_from(luisa::span<const int32_t>{_host_pos.data() + pos0, 1});
    }
    // Embedding gather: hidden = table[ids].
    {
        Key key; key.add("emb", static_cast<uint64_t>(_format)).add("t", rows).add("h", hidden).add("v", _cfg.vocab);
        if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_embedding(_format, rows, hidden, _cfg.vocab); }, &sh); !s.ok()) { return s; }
        const auto scales = _token_embd->scales ? _token_embd->scales.view() : _dummy_f.view();
        cl.annotate("embed.gather")
          << (*sh)(_ids.view(), _token_embd->packed.view(), scales, _hidden.view()).dispatch();
    }
    for (uint32_t l = 0; l < _layers; l++) {
        cl.layer(l);
        const auto &lw = _layer_weights[l];
        // Snapshot the block input as the residual stream. Tile has no pure
        // copy operator; residual = hidden + 0 via the persistent all-zero
        // buffer is a single Tile kernel with no hidden compute semantics.
        {
            const uint32_t count = rows * hidden;
            Key key; key.add("add", count);
            if (auto s = shader_lazy(device, key.s, [&] { return to::capture_tile_residual_add(count); }, &sh); !s.ok()) { return s; }
            cl.annotate("block.residual_snapshot")
              << (*sh)(_hidden.view(), _zero.view(), _residual.view()).dispatch();
        }
        // Input RMSNorm (direct pre-added GGUF weight).
        if (auto s = rmsnorm(device, cl, _hidden.view(), lw.attn_norm->f32, _normed.view(), rows, hidden, "block.input_norm"); !s.ok()) { return s; }
        if (lw.full_attention) {
            if (auto s = full_attention_block(device, stream, cl, l, rows, pos0); !s.ok()) { return s; }
        } else {
            if (auto s = gdn_block(device, stream, cl, l, rows, pos0); !s.ok()) { return s; }
        }
          if (auto s = ffn_block(device, stream, cl, l, rows); !s.ok()) { return s; }
          // Bound the recorded command list: flush every sync_every_layers
          // layers. Whole-chunk recordings lose the device on the current
          // DX/VK backends; see QwenTileEngineOptions::sync_every_layers.
          // _stage_sync backends (VK et al.) commit + synchronize here as
          // before; DX commits WITHOUT waiting (flush(false)) so the host
          // never blocks until the final logits download. flush() is a
          // no-op when empty (benchmark mode already committed and
          // synchronized every command individually).
          if (_opt.sync_every_layers != 0u && (l + 1u) % _opt.sync_every_layers == 0u) {
              cl.flush(_stage_sync);
          }
      }
      return Status::success();
  }

// ---------------------------------------------------------------------------
// LM head on the last position.
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenTileEngine::lm_head(Device &device, Stream &stream, QwenCommandBatch &cl, uint32_t rows) {
    cl.layer(UINT32_MAX);
    const uint32_t hidden = _cfg.hidden;
    const auto last_hidden = _hidden.view(static_cast<size_t>(rows - 1u) * hidden, hidden);
    if (auto s = rmsnorm(device, cl, last_hidden, _output_norm->f32, _last_normed.view(), 1u, hidden, "head.output_norm"); !s.ok()) { return s; }
    // Tied LM head: logits[vocab] = last_normed[hidden] @ token_embd^T.
    if (auto s = project(device, stream, cl, _last_normed, *_token_embd, _logits, 1u, nullptr, "head.lm_head"); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// prefill
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenTileEngine::prefill(Stream &stream, luisa::span<const uint32_t> prompt_tokens,
                                             luisa::vector<float> &logits_out) {
    if (prompt_tokens.empty()) { return invalid("prefill prompt must be nonempty"); }
    if (prompt_tokens.size() > _max_context) { return invalid("prompt exceeds max_context"); }
    for (auto id : prompt_tokens) {
        if (id >= _cfg.vocab) { return invalid("prompt token id out of vocabulary range"); }
    }
    // Fresh report per public call; the reset trace is part of prefill's.
    _report.clear();
    if (auto s = reset(stream); !s.ok()) { return s; }
    Device device = _device;
    const uint32_t total = static_cast<uint32_t>(prompt_tokens.size());
    // ALL prefill chunks are recorded into ONE batch -- no per-chunk sync.
    // copy_from borrows host pointers until encoding commits, so each chunk
    // stages into a DISTINCT absolute-offset host span (ids/pos at pos0, the
    // attn-len pair at 2*pos0) that no later chunk rewrites; the device
    // scratch reuse across chunks is ordered by the stream. The only
    // remaining submissions are the existing every-N-layer bounds inside
    // forward_chunk (commit-only on DX, commit+sync elsewhere) and the
    // final commit below. On a mid-recording failure the batch destructor
    // drops every uncommitted command and the next reset() synchronizes the
    // stream before any host span is rewritten.
    QwenCommandBatch cl{stream, _benchmark ? &_report : nullptr,
                        QwenBenchmarkPhase::prefill, 0u, total};
    cl.layer(UINT32_MAX);
    uint32_t done = 0;
    while (done < total) {
        const uint32_t rows = std::min(_prefill_chunk, total - done);
        const uint32_t pos0 = done;
        // Per-chunk benchmark attribution (phase stays prefill; position and
        // rows track the chunk exactly as with the former per-chunk batches).
        cl.set_context(QwenBenchmarkPhase::prefill, pos0, rows);
        cl.layer(UINT32_MAX);
        if (auto s = stage_ids(cl, prompt_tokens.subspan(done, rows), pos0); !s.ok()) { return s; }
        if (auto s = forward_chunk(device, stream, cl, rows, pos0); !s.ok()) { return s; }
        if (done + rows == total) {
            if (auto s = lm_head(device, stream, cl, rows); !s.ok()) { return s; }
        }
        done += rows;
    }
    // Commit the whole prefill recording without waiting: the logits
    // readback below synchronizes the stream (same ordering as decode).
    cl.flush(false);
    logits_out.resize(_cfg.vocab);
    QwenCommandBatch dl{stream, _benchmark ? &_report : nullptr,
                        QwenBenchmarkPhase::prefill, total - 1u, total};
    dl.layer(UINT32_MAX).annotate("download.logits")
      << _logits.view().copy_to(luisa::span<float>{logits_out});
    dl.flush();
    _position = total;
    _has_session = true;
    return Status::success();
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenTileEngine::decode(Stream &stream, uint32_t input_token,
                                            luisa::vector<float> &logits_out) {
    if (!_has_session) { return invalid("decode requires a prior nonempty prefill"); }
    if (input_token >= _cfg.vocab) { return invalid("decode token id out of vocabulary range"); }
    if (_position >= _max_context) { return invalid("decode would exceed max_context"); }
    // Fresh report per public call.
    _report.clear();
    Device device = _device;
    const uint32_t pos0 = _position;
    QwenCommandBatch cl{stream, _benchmark ? &_report : nullptr,
                        QwenBenchmarkPhase::decode, pos0, 1u};
    cl.layer(UINT32_MAX);
    if (auto s = stage_ids(cl, luisa::span<const uint32_t>{&input_token, 1}, pos0); !s.ok()) {
        _has_session = false;
        return s;
    }
    if (auto s = forward_chunk(device, stream, cl, 1u, pos0); !s.ok()) {
        _has_session = false; // a partial forward may have advanced device state
        return s;
    }
    if (auto s = lm_head(device, stream, cl, 1u); !s.ok()) {
        _has_session = false;
        return s;
    }
    // Commit without waiting: the logits readback below synchronizes the
    // whole stream (same ordering as the original commit + copy + sync).
    cl.flush(false);
    logits_out.resize(_cfg.vocab);
    QwenCommandBatch dl{stream, _benchmark ? &_report : nullptr,
                        QwenBenchmarkPhase::decode, pos0, 1u};
    dl.layer(UINT32_MAX).annotate("download.logits")
      << _logits.view().copy_to(luisa::span<float>{logits_out});
    dl.flush();
    _position = pos0 + 1u;
    return Status::success();
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------
[[nodiscard]] QwenEngineInfo QwenTileEngine::info() const noexcept {
    QwenEngineInfo i;
    i.kernel = QwenKernelKind::tile;
    i.weight_storage = _storage;
    i.position = _position;
    i.max_context = _max_context;
    i.prefill_chunk = _prefill_chunk;
    i.cache_bytes = _cache_bytes;
  // KV storage actually built. The Tile engine is FP32 + position_major ONLY (it
  // refuses every other dtype in create()), so these are constants of this engine
  // rather than a request echo -- reported so a log can compare tiers/engines.
  i.kv_dtype = QwenKvDType::f32;
  i.kv_layout = QwenKvLayout::position_major;
  i.kv_elements_per_word = _kv_cache.elements_per_word();
  i.kv_region_words = _kv_cache.region_words();
  i.kv_payload_bytes = _kv_cache.payload_bytes();
  i.kv_page_bytes = _kv_cache.page_size_bytes();
  i.kv_pages_per_region = _kv_cache.pages_per_region();
  i.kv_heap_slots = _kv_cache.heap_slot_count();
  i.kv_min_commit_pages = _kv_cache.min_commit_pages();
    i.workspace_bytes = _workspace_bytes;
    i.compiled_shader_count = _shader_count;
    i.tile_chunk_conv = _opt.chunk_conv;
    i.tile_attention_bucket_keys = _opt.dynamic_attention ? _opt.attention_bucket_keys : 0u;
    i.tile_attention_buckets = static_cast<uint32_t>(_attention_caps.size());
    i.tile_attention_precompiled = _attention_bucket_precompiled;
    i.tile_attention_lazy_compiles = _attention_bucket_lazy;
    i.caps = _caps;
    return i;
}

[[nodiscard]] const QwenBenchmarkReport *QwenTileEngine::benchmark_report() const noexcept {
    return _benchmark ? &_report : nullptr;
}

}// namespace luisa::qwen
