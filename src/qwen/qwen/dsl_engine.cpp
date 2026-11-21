// Qwen3.5-4B complete ordinary-DSL executor -- implementation.
// See dsl_engine.h for the schedule/state/ABI contract. All device work goes
// through the dsl_ops.h kernel factories; this file only plans, validates,
// allocates, compiles once, and records annotated QwenCommandBatch commands.
#include "qwen/dsl_engine.h"
#include "qwen/native_gemm.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <luisa/backends/ext/cuda/cuda_pdl_ext.h>
#include <luisa/backends/ext/pinned_memory_ext.hpp>
#include <luisa/core/stl/format.h>
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

    // A matrix weight must be packed (bf16/fp8/q8_0) with dims [K, N].
  [[nodiscard]] Status check_matrix(const QwenDeviceTensor &t, uint64_t k, uint64_t n,
                                    dsl::QwenWeightFormat format, uint32_t group = 0u) {
      const auto want = format == dsl::QwenWeightFormat::bf16 ? WeightStorage::bf16_packed
                      : format == dsl::QwenWeightFormat::fp8_e4m3 ? WeightStorage::fp8_packed
                      : format == dsl::QwenWeightFormat::w4 ? WeightStorage::w4_packed
                                                           : WeightStorage::q8_0_packed;
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
      const uint64_t codes_per_word = dsl::qwen_weight_pack(format);
      const uint64_t expected_words = (elements + codes_per_word - 1u) / codes_per_word;
      // A merged segment alias binds a SUB-VIEW of the merged matrix, so the extent
      // check must use the derived view, not the parent buffer.
      if (!t.packed || packed_element_count(t) != expected_words) {
          return invalid("weight packed buffer size mismatch", t.name);
      }
      if (t.merge_parent_index != UINT64_MAX) {
          // The sub-view must start at the alias's row offset inside the parent.
          if (t.packed_view.offset() % expected_words != 0u ||
              t.packed_view.size() != expected_words) {
              return invalid("merged segment sub-view does not match its row range", t.name);
          }
      }
      // The FP32 scale extent of the matrix this tensor REPRESENTS (an alias's
      // `scales` is the parent's buffer, so its own row range is the sub-view).
      const uint64_t scales_extent = t.merge_parent_index == UINT64_MAX
                                         ? (t.scales ? static_cast<uint64_t>(t.scales.size()) : 0ull)
                                         : static_cast<uint64_t>(t.scales_view.size());
    if (format == dsl::QwenWeightFormat::q8_0) {
        // Q8_0: whole 32-weight blocks, K/32 FP32 scales per row.
        if (auto s = d::validate_qwen_q8_0_geometry(k, dsl::kQwenQ8BlockSize); !s.ok()) { return s; }
        const uint64_t expect = k / dsl::kQwenQ8BlockSize;
        uint64_t scale_count = 0u;
        if (!t.scales || scales_extent == 0u || t.groups_per_row != expect ||
            !checked_mul(n, expect, scale_count) || scales_extent != scale_count) {
            return invalid("q8_0 weight scale geometry/buffer size mismatch", t.name);
        }
    } else if (format == dsl::QwenWeightFormat::fp8_e4m3) {
        const uint64_t g = group != 0u ? group : (t.group != 0u ? t.group : 64u);
        if (g == 0u || k % g != 0u) {
            return invalid("fp8 weight scale group does not divide K", t.name);
        }
        const uint64_t expect = k / g;
        uint64_t scale_count = 0u;
        if (!t.scales || scales_extent == 0u || t.groups_per_row != expect ||
            !checked_mul(n, expect, scale_count) || scales_extent != scale_count) {
            return invalid("fp8 weight scale geometry/buffer size mismatch", t.name);
        }
    } else if (format == dsl::QwenWeightFormat::w4) {
        // W4A16: whole 8-code words and whole scale groups (32/64/128), one FP32
        // scale per group per row. Ragged geometry is refused, never padded. The
        // group comes from the caller, else from the pack record (t.group).
        const uint32_t g = group != 0u ? group : t.group;
        if (auto s = d::validate_qwen_w4_geometry(k, g); !s.ok()) { return s; }
        const uint64_t expect = k / g;
        uint64_t scale_count = 0u;
        if (!t.scales || scales_extent == 0u || t.groups_per_row != expect ||
            !checked_mul(n, expect, scale_count) || scales_extent != scale_count) {
            return invalid("w4 weight scale geometry/buffer size mismatch", t.name);
        }
        if (group != 0u && t.group != 0u && t.group != group) {
            return invalid("w4 weight scale group does not match the pack record", t.name);
        }
    }
    return Status::success();
}

  // OFFLINE merged projection descriptor check (opt_plan/03 section 3 + D5). The
  // descriptor is the ONLY thing the merged decode route trusts, so it is validated
  // against the frozen geometry at create: kind, segment roles/names/rows/offsets in
  // the canonical order, K = hidden, N = the sum of the segments, the engine's weight
  // format/group and the real buffer extents. Any deviation is fail-closed: a
  // hand-edited or foreign pack can never route rows into the wrong destination.
  [[nodiscard]] Status check_merged_parent(const QwenDeviceTensor &parent, QwenMergeKind kind,
                                          luisa::span<const QwenTensorRole> roles,
                                          luisa::span<const uint64_t> rows, uint64_t k,
                                          WeightStorage storage, uint32_t group) {
      if (parent.merge_kind != kind) {
          return invalid("merged weight descriptor has the wrong kind", parent.name);
      }
      if (parent.storage != storage) {
          return invalid("merged weight format does not match the engine variant", parent.name);
      }
      if (parent.group != group && !(group == 0u && parent.group == 0u)) {
          return invalid("merged weight scale group does not match the engine variant", parent.name);
      }
      if (parent.dims.size() != 2u || parent.dims[0] != k) {
          return invalid("merged weight K does not match the model hidden size", parent.name);
      }
      if (roles.size() != rows.size() || parent.segments.size() != roles.size()) {
          return invalid("merged weight segment count does not match the frozen geometry", parent.name);
      }
      uint64_t n = 0u, offset = 0u;
      for (size_t i = 0; i < roles.size(); ++i) {
          const auto &seg = parent.segments[i];
          if (seg.role != roles[i]) {
              return invalid("merged weight segment order/role does not match the frozen geometry",
                             parent.name);
          }
          if (seg.rows != rows[i] || seg.row_offset != offset) {
              return invalid("merged weight segment row range does not match the frozen geometry",
                             parent.name);
          }
          if (seg.name.empty() || seg.source_hash == 0u || seg.source_bytes == 0u) {
              return invalid("merged weight segment is missing its source identity", parent.name);
          }
          offset += seg.rows;
          n += seg.rows;
      }
      if (parent.dims[1] != n) {
          return invalid("merged weight N does not equal the segment row sum", parent.name);
      }
      uint64_t elements = 0u;
      if (!checked_mul(k, n, elements) || parent.element_count != elements) {
          return invalid("merged weight element count mismatch", parent.name);
      }
      const uint64_t codes = storage == WeightStorage::bf16_packed ? 2u : 4u;
      if (!parent.packed || parent.packed.size() != (elements + codes - 1u) / codes) {
          return invalid("merged weight packed buffer size mismatch", parent.name);
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
    // Native-shader instances must be destroyed before the Device and after their
    // dispatches completed, which the synchronization above guarantees.
      _native_gemm.reset();
      _native_attn.reset();
      // Same contract for the native Tile shaders: they borrow the Device and must be
      // destroyed after their dispatches completed (synchronized above) and before it.
      _cuda_tile_gemm.reset();
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
    if (options.prefill_chunk_max == 0u) {
        return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(
            invalid("prefill_chunk_max must be >= 1"));
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
    // Adaptive outer chunking (opt_plan/02 section 4): the workspace is sized for
    // the largest chunk the rule can pick, and each prefill then selects its own
    // chunk inside [64, max]. A fixed chunk keeps the historical geometry.
    e._prefill_chunk_adaptive = options.prefill_chunk_adaptive;
    e._prefill_chunk_max = options.prefill_chunk_max;
    e._prefill_chunk_backend_limit =
        d::qwen_prefill_chunk_backend_limit(device.backend_name());
    // Cap the chunk so activation scratch stays bounded and the GDN recurrent
    // kernel's per-launch token count stays inside the uint32 envelope.
    e._prefill_chunk = e._prefill_chunk_adaptive ? options.prefill_chunk_max : options.prefill_chunk;
    e._benchmark = options.benchmark;
    e._gdn_wy_ut = options.dsl_gdn_wy_ut;
          // Calibration-only mode needs the reference M=1 route: the fused FFN/GDN
        // decode kernels compute the ffn_down input in-kernel, where no accumulator
        // can see it (see QwenEngineOptions::calibration_only).
        e._calibration_only = options.calibration_only;
        e._decode_specialized = options.dsl_decode_specialized && !options.calibration_only;
      e._w4_prefill_in_kernel = options.dsl_w4_prefill_in_kernel;
      e._w4_lm_head_fp8 = options.dsl_w4_lm_head_8bit;
      (void)options.dsl_dual_prefill;// resolved from the uploaded copies in resolve_weights
    // Split-KV decode attention with GQA dedup (opt_plan/03 section 4.2). The
    // THRESHOLD has to be known before compile_all, because the legacy per-q_head
    // pair is compiled there and its static guard must be the exact complement of
    // the split family's: one compile-time constant on both sides, so no kv_len is
    // ever served by neither kernel or by both. Whether the split family is
    // compiled at all is decided AFTER the capability probe below, and a refusal
    // there recompiles the legacy pair unguarded -- so this is only the REQUEST.
    e._attn_split = false;
    e._attn_split_threshold = 0u;
    e._attn_split_cfg = d::QwenAttnDecodeSplitConfig{};
    if (options.dsl_split_kv && !e._decode_specialized) {
        // MTP verify, accepted-prefix replay and the plain reference decode must
        // share ONE reduction order; the split reduction changes it, so that path
        // keeps the per-q_head kernel (same rule as dsl_decode_specialized).
        LUISA_INFO("qwen dsl: split-KV decode attention disabled because "
            "dsl_decode_specialized is false (MTP/reference reduction order)");
    } else if (options.dsl_split_kv) {
        if (options.dsl_split_kv_threshold == 0u) {
            return Result<luisa::unique_ptr<QwenEngine, Deleter>>::failure(
                err(StatusCode::invalid_argument,
                    "dsl_split_kv_threshold must be >= 1 (the per-q_head kernel owns kv_len 0)"));
        }
        e._attn_split_threshold = options.dsl_split_kv_threshold;
         }
 // Fused decode GDN conv+SiLU+QK norm (opt_plan/09 N3.3). Same rule as the
      // specializations: the option only REQUESTS it; the model geometry and a
      // successful compile decide, and compile_all() clears it otherwise.
      e._gdn_decode_fused = options.dsl_gdn_decode_fused && e._decode_specialized;
      // FUSED one-CTA recurrence + gated output RMSNorm (opt_plan/03 section 5).
      // Same rule: a REQUEST here, resolved against the geometry in compile_all().
      // dsl_decode_specialized is part of the request itself: MTP verify/replay,
      // calibration-only and the plain reference decode keep the exact recurrence +
      // separate norm pair they share today, so they never see this kernel at all.
      e._gdn_recurrent_norm = options.dsl_gdn_recurrent_norm && e._decode_specialized;
      if (options.dsl_gdn_recurrent_norm && !e._decode_specialized) {
          e._gdn_recurrent_norm_reason =
              "dsl_decode_specialized is false (MTP/reference reduction order)";
      }
      // GDN state-column specialization (opt_plan/09 V4): requested here, resolved
      // against the model geometry in compile_all().
      e._gdn_state_specialized = options.dsl_gdn_state_specialized;
      e._rmsnorm_reg = options.dsl_rmsnorm_reg;

      // Vectorized 128-bit decode realizations (opt_plan/09 N2). A REQUEST: the
      // weight geometry and a successful compile decide in compile_all().
      e._decode_packed = options.dsl_decode_packed;
      e._decode_packed_fp8 = options.dsl_decode_packed_fp8;
      // Model-specialized decode attention prepare (opt_plan/09 N3.2). Same rule.
      e._attn_prepare_model = options.dsl_attn_prepare_model;
      // OFFLINE merged input projections (opt_plan/03 section 3). A REQUEST here: the
      // uploaded weight set decides whether a pack exists (resolve_weights records
      // `_merged_available` and the per-layer descriptors) and compile_all() decides
      // whether the merged kernels can run for this format. Without a pack the flag
      // changes nothing: every projection is uploaded and launched as before.
      e._merged_decode = options.dsl_merged_decode;

 // Full-attention KV STORED DTYPE (opt_plan/05 Q3). The request is resolved
 // here, before any allocation, into the ONE contract the allocator and every
 // attention kernel read (qwen/kv_cache_layout.h). No silent fallback: an
 // unusable geometry fails create() below, and info() reports what was built.
 e._kv_storage = d::qwen_kv_storage_native(options.kv_dtype);
 e._kv_packed_append = e._kv_storage.packed_words();
 // fp8_e4m3 stores packed quad words AND needs the per-(head,64-block) scale
 // update, so it appends through its own kernel (never through the packed-word
 // one, whose two-codes-per-word encoder would mis-address a quad).
 e._kv_fp8_append = e._kv_storage.dtype == QwenKvDType::fp8_e4m3;
 {
     QwenKvLayoutDesc kv_desc{};
     if (auto s = qwen_kv_layout_desc(e._kv_storage.dtype, e._kv_storage.layout,
                                      e._max_context, e._cfg.kv_heads,
                                      static_cast<uint32_t>(e._cfg.attn_kv_n() / e._cfg.kv_heads),
                                      kv_desc);
         !s.ok()) {
         return Result<luisa::unique_ptr<QwenEngine, Deleter>>::failure(s);
     }
     LUISA_INFO("qwen dsl: KV cache dtype {} layout {} ({} code(s)/word, {} B/token/layer); "
                "{} append kernel",
                qwen_kv_dtype_name(e._kv_storage.dtype),
                qwen_kv_layout_name(e._kv_storage.layout),
                kv_desc.elements_per_word(), kv_desc.bytes_per_token(),
                e._kv_fp8_append ? "separate fp8 online-scale"
                                     : (e._kv_packed_append ? "separate packed" : "inline"));
 }
      // Per-prompt GDN prefill selection (opt_plan/09 V4 follow-up). An explicit
      // --gdn-wy-ut pin wins and turns auto off; otherwise the measured crossover
      // decides per prompt.
      e._gdn_auto = options.dsl_gdn_auto;
      e._gdn_wy_ut_active = options.dsl_gdn_wy_ut;
      e._gdn_auto_threshold = options.dsl_gdn_auto_threshold;
      if (e._gdn_auto) {
          LUISA_INFO("qwen dsl: per-prompt GDN selection enabled (recurrent for prompts <= {}, "
                     "WY/UT chunk-64 above)", e._gdn_auto_threshold);
      } else {
          LUISA_INFO("qwen dsl: per-prompt GDN selection disabled; fixed gdn_wy_ut={}",
                     e._gdn_wy_ut);
      }
    e._kv_page_bytes = options.kv_page_bytes;
    e._weights = &weights;
    e._device = device;
    e._stream = stream;

    if (auto s = e.resolve_weights(model, weights); !s.ok()) {
        return Result<luisa::unique_ptr<QwenEngine, Deleter>>::failure(std::move(s));
    }
    // Static CUDA decode graph (opt_plan/03 section 2.3). Format-agnostic: BF16
    // and FP8 differ only in shader bodies, and both get the launch amortization
    // (BF16 is the numerics reference, so it must be measurable under the same
    // schedule). Disabled for: benchmark mode (it synchronizes per command and
    // reports host-wall samples), MTP verify/replay (forces dsl_cuda_graph=false
    // because it needs host-visible logits), the host-logits decode() path (which
    // never uses the graph at all) and non-CUDA devices.
    if (options.dsl_cuda_graph && !e._benchmark && device.backend_name() == luisa::string_view{"cuda"}) {
        e._cuda_graph_ext = device.extension<compute::CudaGraphExt>();
        if (e._cuda_graph_ext == nullptr) {
            LUISA_WARNING("qwen dsl: CudaGraphExt is unavailable on this CUDA device; "
                          "device-greedy decode keeps ordinary command lists");
        }
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

    // Prefill attention realization. The tiled kernel pins warp size 32 and a
    // shared-memory tile, so the choice is fail-closed against the capability
    // record captured above: a device that cannot host it keeps the reference
    // kernel and reports that in QwenEngineInfo (never a silent run-time fallback).
    e._attn_tiled = false;
    e._attn_tiled_cfg = d::QwenAttnPrefillTiledConfig{};
    if (options.dsl_tiled_prefill_attn) {
        d::QwenAttnPrefillTiledConfig tiled_cfg{};
        const auto selected = d::select_qwen35_attn_prefill_tiled_config(
            e._caps.warp_size, e._caps.max_shared_memory_bytes, tiled_cfg);
        if (selected.ok()) {
            e._attn_tiled = true;
            e._attn_tiled_cfg = tiled_cfg;
        } else {
            LUISA_WARNING("qwen dsl: tiled prefill attention unavailable ({}: {}); using the reference kernel",
                selected.message, selected.detail);
        }
    }
    // Compile the tiled prefill attention shader now: the capability record it
    // depends on is only valid after compile_all, and the reference kernel above
    // is always compiled so the switch stays a pure dispatch decision.
    if (e._attn_tiled) {
        compute::ShaderOption tiled_option;
        tiled_option.enable_fast_math = false;
        e._attn_prefill_tiled = device.compile(
            d::make_qwen35_attn_prefill_tiled_kernel(e._attn_tiled_cfg, e._kv_storage), tiled_option);
        if (!e._attn_prefill_tiled || !e._attn_prefill_tiled.compile_ok()) {
            return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(
                err(StatusCode::internal_error, "DSL tiled prefill attention shader compile failed"));
        }
        e._shader_count++;
    }
    qwen_log_backend_caps(e._caps);
    // ---- PINNED download staging (opt_plan/03 section 6, opt_plan/09 N4.3) ----
    // Selected once, AFTER the workspace exists (the vocabulary and the history
    // capacity are known) and BEFORE any download can happen. Fail-closed: an absent
    // extension, a refused allocation or the opt-out keeps the previous pageable
    // path and the reason is reported in QwenEngineInfo. Benchmark mode deliberately
    // keeps the pageable path: `--benchmark` instruments per CommandList command, and
    // a pinned download is not one of those, so mixing them would make the report
    // lie about where the decode time went.
    e._pinned_selection = QwenPinnedSelection{};
    e._pinned_selection.requested = options.dsl_pinned_transfers;
    e._pinned_ext = device.extension<compute::PinnedMemoryExt>();
    e._pinned_selection.extension = e._pinned_ext != nullptr;
    if (!options.dsl_pinned_transfers) {
        e._pinned_selection.reason = "disabled by option (--pinned-transfers 0)";
    } else if (e._pinned_ext == nullptr) {
        e._pinned_selection.reason = "PinnedMemoryExt is not available on this backend";
    } else if (e._benchmark) {
        e._pinned_selection.reason =
            "benchmark mode keeps the pageable path (per-command instrumentation)";
    } else {
        // Ring capacity: the bounded in-flight chunk can never exceed the context, and
        // 1024 tokens (4 KiB) covers the largest useful chunk several times over while
        // a caller asking for more than the ring is served by ring-sized slices.
        const uint32_t ring_tokens = std::min<uint32_t>(e._max_context, 1024u);
        e._pinned = QwenPinnedRing::create(device, e._pinned_ext, ring_tokens,
                                           static_cast<uint64_t>(e._cfg.vocab));
        e._pinned_selection.tokens = static_cast<bool>(e._pinned.token_buffer());
        e._pinned_selection.logits = static_cast<bool>(e._pinned.logits_buffer());
        e._pinned_selection.ring_tokens = e._pinned.capacity();
        e._pinned_selection.logits_bytes = static_cast<uint64_t>(e._cfg.vocab) * sizeof(float);
        e._pinned_selection.selected = e._pinned.valid();
        e._pinned_selection.reason = e._pinned_selection.selected
                                         ? "selected"
                                         : "pinned allocation failed (kept the pageable path)";
    }
    LUISA_INFO("qwen dsl: pinned downloads {}", e._pinned_selection.summary());
    // ---- PDL state (opt_plan/03 section 2.5) --------------------------------
    // PDL is enabled at DEVICE CREATION (CUDADeviceConfigExt), so the engine can only
    // report what that device actually offers. Nothing here changes kernel selection:
    // no engine kernel emits grid_dependency_synchronize(), and the backend's
    // automatic prologue is inert below compute capability 9.0.
    {
        auto &pdl = e._pdl;
        pdl.requested = options.dsl_pdl;
        if (auto *ext = device.extension<compute::CUDAPdlStreamExt>()) {
            pdl.available = true;
            pdl.supported = ext->pdl_supported();
            pdl.auto_prologue = ext->auto_prologue_enabled();
            pdl.stream_enabled = ext->pdl_stream_enabled();
            const auto stats = ext->launch_stats();
            pdl.relaxed_launches = stats.relaxed_launches;
            pdl.strict_launches = stats.strict_launches;
            pdl.driver_fallbacks = stats.driver_fallbacks;
            if (!pdl.supported) {
                pdl.reason = "not available on this device (CUDAPdlStreamExt::pdl_supported() "
                             "is false: programmatic dependent launch needs compute capability "
                             ">= 9.0 and a driver >= 12.0); no engine kernel is PDL-eligible";
            } else if (!pdl.requested) {
                pdl.reason = "supported, not requested (the launch policy stays on the "
                             "backend default and no engine kernel waits for a dependency)";
            } else if (!pdl.auto_prologue) {
                pdl.reason = "supported, but the automatic wait-at-entry prologue is off: the "
                             "device was created without "
                             "CUDADeviceConfigExt::enable_programmatic_stream_serialization()";
            } else {
                pdl.reason = "supported with the automatic prologue; no engine kernel is "
                             "PDL-eligible (no weight-stream/activation overlap realization)";
            }
        } else {
            pdl.reason = "CUDAPdlStreamExt is not available on this backend";
        }
        LUISA_INFO("qwen dsl: PDL {}", pdl.summary());
    }
    // ---- CONDITIONAL WHILE decode chunk (opt_plan/09 N4.2) ------------------
    // Two independent gates, never the tier and never the capability flag alone:
    //   1. the request must be valid on its own (host-only validation), and
    //   2. the DEDICATED instantiate/execute/zero-trip probe must pass on the running
    //      driver. A refusal keeps the per-token static graph as the default and is
    //      reported verbatim (QwenEngineInfo::cuda_graph_while_reason).
    e._while_requested = options.dsl_cuda_graph_while;
    e._while_chunk = options.dsl_cuda_graph_while_tokens;
    e._while_eos = options.dsl_eos_token_id;
    if (e._while_requested) {
        QwenWhileRequest request;
        request.requested = true;
        request.chunk = e._while_chunk;
        request.eos_token = e._while_eos;
        request.vocab = e._cfg.vocab;
        request.position = 0u;
        request.max_context = e._max_context;
        auto validated = qwen_validate_while_request(request);
        if (e._cuda_graph_ext == nullptr) {
            e._while_reason = "conditional WHILE decode requires CudaGraphExt "
                              "(and dsl_cuda_graph, which is off here)";
        } else if (!validated.ok()) {
            e._while_reason = luisa::string{validated.message} +
                              (validated.detail.empty() ? "" : " [" + validated.detail + "]");
        } else if (!e._caps.cuda_graph_conditional_nodes) {
            e._while_reason = "CudaGraphExt reports no conditional-node support";
        } else {
            e._while_probe = qwen_probe_cuda_graph_while(device, stream);
            if (e._while_probe.supported) {
                e._while_selected = true;
                e._while_reason = "selected (probe: 3 body executions, replay 3, zero-trip 0)";
            } else {
                e._while_reason = "probe refused: " + e._while_probe.summary();
            }
        }
        LUISA_INFO("qwen dsl: conditional WHILE decode requested (chunk {}, eos {}): {}",
                   e._while_chunk, e._while_eos, e._while_reason);
        if (!e._while_selected) {
            LUISA_WARNING("qwen dsl: conditional WHILE decode unavailable ({}); the bounded "
                          "chunk is served by the per-token static graph",
                          e._while_reason.c_str());
        }
    }
    // Split-KV decode attention with GQA dedup (opt_plan/03 section 4.2). Selected
    // from the capability record probed above (warp size, REPORTED shared-memory
    // limit and -- when the backend answers it -- the SM count), then compiled once
    // for the whole session. A refusal here is fail-closed: the legacy pair is
    // recompiled UNGUARDED so decode attention still works for every kv_len, and the
    // engine reports decode_attn_split == false instead of a half-armed guard.
    if (e._attn_split_threshold != 0u) {
        d::QwenAttnDecodeSplitConfig split_cfg{};
        auto selected = d::select_qwen_attn_decode_split_config(
            e._caps.warp_size, e._caps.max_shared_memory_bytes, e._caps.sm_count, split_cfg);
        // The threshold is the one the legacy pair was just compiled with; never let
        // the selection change it (the two guards must stay exact complements).
        split_cfg.threshold = e._attn_split_threshold;
        // A caller override only moves the compile-time grid bound, and it goes
        // through the same fail-closed gate as the selection: above the plan cap or
        // a config whose fixed partial workspace the kernels could not index, the
        // override is refused and the legacy pair stays in charge (never a
        // half-applied config). It cannot spend shared memory, so the K/V tile the
        // reported limit selected stays within budget.
        if (selected.ok() && options.dsl_split_kv_max_splits != 0u) {
            if (auto applied = d::qwen_attn_decode_split_apply_max_splits_override(
                    split_cfg, options.dsl_split_kv_max_splits, e._caps.max_shared_memory_bytes);
                !applied.ok()) {
                selected = applied;
            }
        }
        // The family is model-specific (compile-time q_heads/kv_heads/head_dim), so a
        // config that does not describe THIS trunk is refused instead of mis-indexed.
        if (selected.ok()) {
            if (e._cfg.q_heads != split_cfg.q_heads || e._cfg.kv_heads != split_cfg.kv_heads ||
                e._cfg.attn_head_dim != split_cfg.head_dim) {
                selected = Status::error(StatusCode::unsupported,
                                         "split-KV decode attention requires the model geometry it "
                                         "was compiled for (q_heads 16 / kv_heads 4 / head_dim 256)");
            } else {
                selected = d::validate_qwen_attn_decode_split_config(split_cfg);
            }
        }
        if (!selected.ok()) {
            LUISA_WARNING("qwen dsl: split-KV decode attention unavailable ({}: {}); "
                          "keeping the per-q_head kernel for every kv_len",
                          selected.message, selected.detail);
            compute::ShaderOption plain_option;
            plain_option.enable_fast_math = false;
            e._attn_decode = device.compile(d::make_qwen_attn_decode_kernel(0u, e._kv_storage), plain_option);
            e._attn_decode_dev = device.compile(d::make_qwen_attn_decode_dev_kernel(0u, e._kv_storage), plain_option);
            if (!e._attn_decode || !e._attn_decode.compile_ok() ||
                !e._attn_decode_dev || !e._attn_decode_dev.compile_ok()) {
                return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(
                    err(StatusCode::internal_error, "DSL decode attention shader recompile failed"));
            }
            e._attn_split_threshold = 0u;
        } else {
            const auto acc_elems = static_cast<uint64_t>(split_cfg.max_splits) *
                                   split_cfg.q_heads * split_cfg.head_dim;
            const auto ml_elems = static_cast<uint64_t>(split_cfg.max_splits) * split_cfg.q_heads * 2u;
            if (acc_elems > 0xFFFFFFFFull || ml_elems > 0xFFFFFFFFull) {
                return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(
                    err(StatusCode::internal_error, "split-KV decode partial workspace exceeds uint32"));
            }
            e._attn_partial_acc = device.create_buffer<float>(static_cast<size_t>(acc_elems));
            e._attn_partial_ml = device.create_buffer<float>(static_cast<size_t>(ml_elems));
            if (!e._attn_partial_acc || !e._attn_partial_ml) {
                return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(
                    err(StatusCode::out_of_memory, "split-KV decode partial workspace allocation failed"));
            }
            e._workspace_bytes += (acc_elems + ml_elems) * sizeof(float);
            compute::ShaderOption split_option;
            split_option.enable_fast_math = false;
            e._attn_decode_split = device.compile(
                d::make_qwen_attn_decode_split_kernel(split_cfg, e._kv_storage), split_option);
            e._attn_decode_split_dev = device.compile(
                d::make_qwen_attn_decode_split_dev_kernel(split_cfg, e._kv_storage), split_option);
            e._attn_decode_combine = device.compile(
                d::make_qwen_attn_decode_combine_kernel(split_cfg), split_option);
            e._attn_decode_combine_dev = device.compile(
                d::make_qwen_attn_decode_combine_dev_kernel(split_cfg), split_option);
            if (!e._attn_decode_split || !e._attn_decode_split.compile_ok() ||
                !e._attn_decode_split_dev || !e._attn_decode_split_dev.compile_ok() ||
                !e._attn_decode_combine || !e._attn_decode_combine.compile_ok() ||
                !e._attn_decode_combine_dev || !e._attn_decode_combine_dev.compile_ok()) {
                return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(
                    err(StatusCode::internal_error, "DSL split-KV decode attention shader compile failed"));
            }
            e._shader_count += 4u;
            e._attn_split = true;
            e._attn_split_cfg = split_cfg;
            LUISA_INFO("qwen dsl: split-KV decode attention enabled (threshold {}, k_tile {}, "
                       "max_splits {}, shared {} B, partials {} KiB, sm source '{}')",
                       split_cfg.threshold, split_cfg.k_tile, split_cfg.max_splits,
                       d::qwen_attn_decode_split_shared_bytes(split_cfg),
                       d::qwen_attn_decode_split_partial_bytes(split_cfg) / 1024u,
                       e._caps.sm_count_source);
        }
    }
    // Weight-reusing prefill GEMM (opt_plan/02 section 3). Compiled here for the
    // same reason as the tiled attention above: the selection depends on the
    // warp size the device reports only after the first compile. The reference
    // GEMM is always compiled, so this stays a pure dispatch decision.
    e._prefill_gemm_mt = false;
    e._gemm_mt_cfg = d::QwenGemmMtConfig{};
    if (options.dsl_prefill_gemm_mt) {
        d::QwenGemmMtConfig mt_cfg{};
        mt_cfg.mt = options.dsl_prefill_gemm_mt_rows;
        if (auto s = d::validate_qwen35_gemm_mt_config(mt_cfg); !s.ok()) {
            LUISA_WARNING("qwen dsl: mt prefill GEMM disabled ({}: {})", s.message, s.detail);
        } else if (auto s = d::select_qwen35_gemm_mt_config(e._caps.warp_size, mt_cfg); !s.ok()) {
            LUISA_WARNING("qwen dsl: mt prefill GEMM unavailable ({}: {}); using the reference GEMM",
                s.message, s.detail);
        } else {
            compute::ShaderOption mt_option;
            mt_option.enable_fast_math = false;
            // Fail closed once, at create time: every projection weight the mt
            // family would see must satisfy its packed-word / fp8-group contract,
            // otherwise the engine keeps the reference GEMM instead of falling
            // back per shape in the middle of a prompt.
            // The family is compiled for the format the PREFILL route will read:
            // the 8-bit second copy in a dual-copy W4 session (plan option (c)),
            // otherwise the decode format itself (option (b) / non-W4 variants).
            const auto pf_fmt = e._dual_prefill ? e._prefill_format : e._format;
            const auto pf_group = e._dual_prefill ? e._prefill_group : e._decode_group;
            Status shape_status = Status::success();
            const auto check_shape = [&](const QwenDeviceTensor *t) {
                if (t == nullptr || !shape_status.ok()) { return; }
                auto planned = d::plan_qwen35_gemm_mt_dispatch(
                    2u, t->dims[1], t->dims[0], pf_fmt, pf_group, mt_cfg);
                if (!planned.ok()) { shape_status = planned.status; }
            };
            // In a dual-copy session only the partner tensors reach rows > 1.
            for (const auto &lw : e._layer_weights) {
                const auto &pf = lw.pf;
                const auto ffn_gate = e._dual_prefill ? pf.ffn_gate : lw.ffn_gate;
                const auto ffn_up = e._dual_prefill ? pf.ffn_up : lw.ffn_up;
                const auto ffn_down = e._dual_prefill ? pf.ffn_down : lw.ffn_down;
                check_shape(ffn_gate);
                check_shape(ffn_up);
                check_shape(ffn_down);
                if (lw.full_attention) {
                    check_shape(e._dual_prefill ? pf.attn_q : lw.attn_q);
                    check_shape(e._dual_prefill ? pf.attn_k : lw.attn_k);
                    check_shape(e._dual_prefill ? pf.attn_v : lw.attn_v);
                    check_shape(e._dual_prefill ? pf.attn_output : lw.attn_output);
                } else {
                    check_shape(e._dual_prefill ? pf.gdn_qkv : lw.gdn_qkv);
                    check_shape(e._dual_prefill ? pf.gdn_gate : lw.gdn_gate);
                    check_shape(e._dual_prefill ? pf.gdn_alpha : lw.gdn_alpha);
                    check_shape(e._dual_prefill ? pf.gdn_beta : lw.gdn_beta);
                    check_shape(e._dual_prefill ? pf.gdn_out : lw.gdn_out);
                }
            }
            if (!shape_status.ok()) {
                LUISA_WARNING("qwen dsl: mt prefill GEMM not applicable to a projection shape ({}: {}); using the reference GEMM",
                    shape_status.message, shape_status.detail);
              } else if (e._dual_prefill) {
                  e._gemm_mt_pf = device.compile(d::make_qwen35_gemm_mt_kernel(
                                                     mt_cfg, pf_fmt, d::QwenMatmulEpilogue::none),
                                                 mt_option);
                  e._gemm_mt_pf_residual = device.compile(d::make_qwen35_gemm_mt_kernel(
                                                              mt_cfg, pf_fmt, d::QwenMatmulEpilogue::residual),
                                                          mt_option);
                  if (!e._gemm_mt_pf || !e._gemm_mt_pf.compile_ok() ||
                      !e._gemm_mt_pf_residual || !e._gemm_mt_pf_residual.compile_ok()) {
                      return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(
                          err(StatusCode::internal_error, "DSL mt prefill GEMM (8-bit copy) shader compile failed"));
                  }
                  e._prefill_gemm_mt_pf = true;
                  e._shader_count += 2u;
                  e._gemm_mt_cfg = mt_cfg;
              } else {
                  e._gemm_mt = device.compile(d::make_qwen35_gemm_mt_kernel(
                                                  mt_cfg, e._format, d::QwenMatmulEpilogue::none),
                                              mt_option);
                  e._gemm_mt_residual = device.compile(d::make_qwen35_gemm_mt_kernel(
                                                           mt_cfg, e._format, d::QwenMatmulEpilogue::residual),
                                                       mt_option);
                  if (!e._gemm_mt || !e._gemm_mt.compile_ok() ||
                      !e._gemm_mt_residual || !e._gemm_mt_residual.compile_ok()) {
                      return Result<luisa::unique_ptr<QwenEngine, QwenDslEngine::Deleter>>::failure(
                          err(StatusCode::internal_error, "DSL mt prefill GEMM shader compile failed"));
                  }
                  e._prefill_gemm_mt = true;
                  e._shader_count += 2u;
                  e._gemm_mt_cfg = mt_cfg;
              }

        }
    }
    // CUDA TENSOR-CORE prefill GEMM (opt_plan/09 N1a). Independent of the DSL MT
    // selection above: this is a NativeShaderExt / CUDA_NVRTC `mma.sync` kernel, so
    // it needs the CUDA backend, BF16 packed weights and a geometry its tiling can
    // index. Every gate is checked here, once: a projection that failed the shape
    // contract, a missing extension or an NVRTC/reflection error leaves the object
    // invalid and the DSL GEMM in charge for the whole session (never a per-shape
    // fallback in the middle of a prompt).
    if (options.native_tensor_gemm) {
        // Plan option (c): the tensor-core prefill GEMM consumes the 8-BIT PREFILL
        // copy when the session has one (its existing FP8 kernel), and the decode
        // format only in a single-copy session. It has no W4 realization.
        const auto nf_fmt = e._dual_prefill ? e._prefill_format : e._format;
        const auto nf_group = e._dual_prefill ? e._prefill_group : e._decode_group;
        if (!d::qwen_native_gemm_supported_format(nf_fmt)) {
            LUISA_INFO("qwen dsl: native tensor GEMM off ({} weights keep the DSL MT path)",
                       d::qwen_weight_format_name(nf_fmt));
        } else if (luisa::string_view{device.backend_name()} != "cuda") {
            LUISA_INFO("qwen dsl: native tensor GEMM off (backend '{}'; NativeShaderExt "
                       "CUDA_NVRTC exists on cuda only)", device.backend_name());
        } else {
            Status native_shape = Status::success();
            const auto check_native = [&](const QwenDeviceTensor *t) {
                if (t == nullptr || !native_shape.ok()) { return; }
                if (auto st = d::validate_qwen_native_gemm_shape(
                        d::qwen_native_gemm_min_rows, t->dims[1], t->dims[0], nf_fmt,
                        nf_group);
                    !st.ok()) {
                    native_shape = st;
                }
            };
            for (const auto &lw : e._layer_weights) {
                const auto &pf = lw.pf;
                const auto g = [&](const QwenDeviceTensor *a, const QwenDeviceTensor *b) {
                    return e._dual_prefill ? a : b;
                };
                check_native(g(pf.ffn_gate, lw.ffn_gate));
                check_native(g(pf.ffn_up, lw.ffn_up));
                check_native(g(pf.ffn_down, lw.ffn_down));
                if (lw.full_attention) {
                    check_native(g(pf.attn_q, lw.attn_q));
                    check_native(g(pf.attn_k, lw.attn_k));
                    check_native(g(pf.attn_v, lw.attn_v));
                    check_native(g(pf.attn_output, lw.attn_output));
                } else {
                    check_native(g(pf.gdn_qkv, lw.gdn_qkv));
                    check_native(g(pf.gdn_gate, lw.gdn_gate));
                    check_native(g(pf.gdn_alpha, lw.gdn_alpha));
                    check_native(g(pf.gdn_beta, lw.gdn_beta));
                    check_native(g(pf.gdn_out, lw.gdn_out));
                }
            }
            d::QwenNativeGemmConfig native_cfg{};
            // The FP8 scale group is baked into the generated source, so the engine hands
            // the one its weight buffers were quantized with.
            native_cfg.fp8_group = nf_group;
            if (!native_shape.ok()) {
                LUISA_WARNING("qwen dsl: native tensor GEMM off ({}: {}); using the DSL GEMM",
                              native_shape.message, native_shape.detail);
            } else if (auto st = e._native_gemm.create(device, nf_fmt, native_cfg); !st.ok()) {
                LUISA_WARNING("qwen dsl: native tensor GEMM off ({}: {}); using the DSL GEMM",
                              st.message, st.detail);
                e._native_gemm.reset();
            } else {
                LUISA_INFO("qwen dsl: native tensor GEMM on ({} prefill rows, tile {}x{}x{}, "
                           "{} warps, cold JIT {:.1f} ms)",
                           d::qwen_native_gemm_min_rows, native_cfg.block_m, native_cfg.block_n,
                           native_cfg.block_k, native_cfg.warps, e._native_gemm.jit_ms());
            }
          }
      }
      // ---- EXPERIMENTAL: CUDA native-Tile typed-FP8 prefill GEMM (strictly opt-in) --
      // QwenEngineOptions::cuda_tile_fp8_prefill is a REQUEST the caller makes and the
      // engine never infers from a weight format or a backend. When it is false, this
      // block does not run at all: no probe, no JIT, no dispatch gate, so the measured
      // default (native NVRTC, then the DSL MT / legacy GEMM) is byte for byte the
      // behaviour of the tree before this path existed, and QwenEngineInfo reports
      // cuda_tile_prefill_gemm == false instead of an "available" claim.
      // When it IS requested the whole feature is FAIL CLOSED here: a bad tiling, a
      // prefill copy that is not FP8 E4M3 with the 64-element scale group, a non-cuda
      // backend, a missing LUISA_CUDA_TILE_IR=1 or a device/toolkit whose typed-FP8
      // Tile compile probe fails all make CREATE fail with the helper's diagnostic.
      // There is no quiet no-op and no mid-prompt fallback to the other GEMMs (nor to
      // another lowering): an explicit request that cannot be honoured is an error.
      // The format/group of the copy PREFILL reads is the one that matters: in a
      // dual-copy W4 session that is the FP8 second copy (plan option (c)), otherwise
      // the decode format itself (so an ordinary dsl-fp8 session qualifies and
      // bf16/q8_0 do not).
      const auto tile_prefill_format = e._dual_prefill ? e._prefill_format : e._format;
      const auto tile_prefill_group = e._dual_prefill ? e._prefill_group : e._decode_group;
      if (options.cuda_tile_fp8_prefill) {
          d::QwenCudaTileGemmConfig tile_cfg{};
          tile_cfg.tile_m = options.cuda_tile_m;
          tile_cfg.tile_n = options.cuda_tile_n;
          if (auto st = d::validate_qwen_cuda_tile_gemm_config(tile_cfg); !st.ok()) {
              return Result<luisa::unique_ptr<QwenEngine, Deleter>>::failure(
                  err(st.code, "cuda tile FP8 prefill GEMM: the requested tiling is invalid "
                               "(cuda_tile_m / cuda_tile_n)",
                      st.detail.empty() ? st.message : st.message + ": " + st.detail));
          }
          if (luisa::string_view{device.backend_name()} != "cuda") {
              return Result<luisa::unique_ptr<QwenEngine, Deleter>>::failure(err(
                  StatusCode::unsupported,
                  "cuda tile FP8 prefill GEMM requires the cuda backend",
                  luisa::string{"device backend = "} + luisa::string{device.backend_name()} +
                      " (the DX/VK Tile route rejects FP8 storage and is unchanged)"));
          }
          if (!d::qwen_cuda_tile_gemm_native_route_requested()) {
              return Result<luisa::unique_ptr<QwenEngine, Deleter>>::failure(err(
                  StatusCode::unsupported,
                  "cuda tile FP8 prefill GEMM requires LUISA_CUDA_TILE_IR=1 in the environment",
                  "the CUDA backend requires an explicit opt-in for its experimental native Tile "
                  "IR route; export LUISA_CUDA_TILE_IR=1 for CUDA 13.5, or use the "
                  "default native NVRTC GEMM"));
          }
          if (!d::qwen_cuda_tile_gemm_supported_format(tile_prefill_format)) {
              return Result<luisa::unique_ptr<QwenEngine, Deleter>>::failure(err(
                  StatusCode::unsupported,
                  "cuda tile FP8 prefill GEMM requires FP8 E4M3 prefill weights",
                  luisa::string{"the copy the prefill route reads is "} +
                      luisa::string{d::qwen_weight_format_name(tile_prefill_format)} +
                      " (an ordinary dsl-fp8 session, or the FP8 second copy of a dual-copy "
                      "W4 session, is required); the BF16/Q8_0/W4 prefill paths stay on the "
                      "native NVRTC / DSL MT GEMM"));
          }
          if (tile_prefill_group != d::qwen_cuda_tile_gemm_group) {
              return Result<luisa::unique_ptr<QwenEngine, Deleter>>::failure(err(
                  StatusCode::unsupported,
                  "cuda tile FP8 prefill GEMM requires the 64-element FP8 scale group",
                  luisa::format("prefill scale group = {}", tile_prefill_group)));
          }
          if (auto st = e._cuda_tile_gemm.create(device, tile_cfg); !st.ok()) {
              return Result<luisa::unique_ptr<QwenEngine, Deleter>>::failure(
                  err(st.code, "cuda tile FP8 prefill GEMM is unavailable: " + st.message, st.detail));
          }
          // Static per-projection gate. Only the shapes the dispatch rule can route
          // here are checked (N >= qwen_native_gemm_min_columns, the existing
          // native-eligible column rule; the narrow N = 32 gate/alpha/beta pair keeps
          // its current path), and they are checked NOW so the requested path can
          // never die in the middle of a prompt for a statically known reason: a K
          // that is not a multiple of 64, an extent outside the static bounds, or a
          // ragged N on a device whose toolkit cannot compile a PARTIAL typed-FP8 tile
          // load (measured: sm_89 cannot, so such an N is refused at create instead of
          // being assumed to work).
          Status tile_shape = Status::success();
          const auto check_tile = [&](const QwenDeviceTensor *t) {
              if (t == nullptr || !tile_shape.ok()) { return; }
              const auto n = t->dims[1];
              const auto k = t->dims[0];
              if (!d::qwen_native_gemm_prefers_columns(n)) { return; }
              if (auto st = d::validate_qwen_cuda_tile_gemm_shape(
                      d::qwen_native_gemm_min_rows, n, k, tile_prefill_format, tile_prefill_group);
                  !st.ok()) {
                  tile_shape = err(st.code, luisa::string{t->name} + ": " + st.message, st.detail);
                  return;
              }
              if (!e._cuda_tile_gemm.supports_ragged_columns() && n % tile_cfg.tile_n != 0u) {
                  tile_shape = err(
                      StatusCode::unsupported,
                      luisa::string{t->name} +
                          ": N is not a multiple of cuda_tile_n and this device/toolkit cannot "
                          "compile a PARTIAL (masked) typed-FP8 tile load",
                      luisa::format("n = {}, tile_n = {}: pick a tile_n that divides every eligible "
                                    "N (64 or 128 for this model), or drop --cuda-tile-fp8",
                                    n, tile_cfg.tile_n));
              }
          };
          for (const auto &lw : e._layer_weights) {
              const auto &pf = lw.pf;
              const auto g = [&](const QwenDeviceTensor *a, const QwenDeviceTensor *b) {
                  return e._dual_prefill ? a : b;
              };
              check_tile(g(pf.ffn_gate, lw.ffn_gate));
              check_tile(g(pf.ffn_up, lw.ffn_up));
              check_tile(g(pf.ffn_down, lw.ffn_down));
              if (lw.full_attention) {
                  check_tile(g(pf.attn_q, lw.attn_q));
                  check_tile(g(pf.attn_k, lw.attn_k));
                  check_tile(g(pf.attn_v, lw.attn_v));
                  check_tile(g(pf.attn_output, lw.attn_output));
              } else {
                  check_tile(g(pf.gdn_qkv, lw.gdn_qkv));
                  check_tile(g(pf.gdn_gate, lw.gdn_gate));
                  check_tile(g(pf.gdn_alpha, lw.gdn_alpha));
                  check_tile(g(pf.gdn_beta, lw.gdn_beta));
                  check_tile(g(pf.gdn_out, lw.gdn_out));
              }
          }
          if (!tile_shape.ok()) {
              return Result<luisa::unique_ptr<QwenEngine, Deleter>>::failure(
                  err(tile_shape.code,
                      "cuda tile FP8 prefill GEMM does not cover this weight shape: " +
                          tile_shape.message,
                      tile_shape.detail));
          }
          LUISA_INFO("qwen dsl: CUDA native Tile FP8 prefill GEMM ON (EXPERIMENTAL, opt-in; tile "
                     "{}x{}x{}, prefill rows >= {}, lazily JIT-compiled per static shape, probe "
                     "{:.1f} ms, ragged-N {}); it is tried BEFORE the native NVRTC GEMM and is "
                     "measured SLOWER on sm_89 - the default stays NVRTC",
                     tile_cfg.tile_m, tile_cfg.tile_n, tile_cfg.tile_k,
                     d::qwen_native_gemm_min_rows, e._cuda_tile_gemm.probe_ms(),
                     e._cuda_tile_gemm.supports_ragged_columns() ? "supported" : "refused");
      }
      // CUDA TENSOR-CORE prefill attention (opt_plan/09 N1b). Independent of the
      // tiled/reference DSL selection above: this is a NativeShaderExt / CUDA_NVRTC
      // FlashAttention-2 kernel, so it needs the CUDA backend, the model geometry its
      // compile-time tiling was generated for (head_dim 256, 16 q / 4 kv heads) and a
      // successful NVRTC compile. Every gate is checked here, once: any failure leaves
      // the object invalid and the DSL tiled kernel keeps prefill attention for the
      // whole session (never a per-shape fallback in the middle of a prompt).
      // KV dtype gate (opt_plan/05 Q3): the native FlashAttention-2 kernel reads the
      // K/V cache as DENSE FP32 [capacity][kv_heads][head_dim]. A BF16 cache stores two
      // codes per word in the 64-position blocked layout, so handing it that pointer
      // would reinterpret codes as floats -- refused here, explicitly, and the DSL
      // BF16 tiled/reference kernel keeps prefill attention for the whole session.
      // This is a fail-closed selection, not a fallback: info().native_prefill_attn
      // stays false and the log names the reason.
      if (options.native_tensor_attention && e._kv_storage.dtype != QwenKvDType::f32) {
          LUISA_INFO("qwen dsl: native tensor attention off (KV dtype {} is not the dense "
                     "FP32 layout the native kernel binds); using the DSL prefill attention",
                     qwen_kv_dtype_name(e._kv_storage.dtype));
      } else if (options.native_tensor_attention) {
          if (luisa::string_view{device.backend_name()} != "cuda") {
              LUISA_INFO("qwen dsl: native tensor attention off (backend '{}'; NativeShaderExt "
                         "CUDA_NVRTC exists on cuda only)", device.backend_name());
          } else {
              d::QwenNativeAttnConfig native_attn_cfg{};
              native_attn_cfg.head_dim = e._cfg.attn_head_dim;
              native_attn_cfg.q_heads = e._cfg.q_heads;
              native_attn_cfg.kv_heads = e._cfg.kv_heads;
              if (auto st = d::validate_qwen_native_attn_config(native_attn_cfg); !st.ok()) {
                  LUISA_WARNING("qwen dsl: native tensor attention off ({}: {}); using the DSL "
                                "prefill attention",
                                st.message, st.detail);
              } else if (auto st = e._native_attn.create(device, native_attn_cfg); !st.ok()) {
                  LUISA_WARNING("qwen dsl: native tensor attention off ({}: {}); using the DSL "
                                "prefill attention",
                                st.message, st.detail);
                  e._native_attn.reset();
              } else {
                  LUISA_INFO("qwen dsl: native tensor attention on (tile {}x{}, {} warps, {} B "
                             "static shared, cold JIT {:.1f} ms)",
                             native_attn_cfg.q_tile, native_attn_cfg.kv_tile,
                             native_attn_cfg.warps(),
                             d::qwen_native_attn_shared_bytes(native_attn_cfg),
                             e._native_attn.jit_ms());
              }
          }
      }
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
    } else if (embd->storage == WeightStorage::q8_0_packed) {
        // Native Q8_0: the engine consumes the checkpoint's own int8 block codes.
        // Nothing is re-quantized and nothing is routed through FP8.
        _format = dsl::QwenWeightFormat::q8_0;
        _storage = WeightStorage::q8_0_packed;
        // The per-row scale group is the GGUF Q8_0 block (32), not the FP8 64.
    } else if (embd->storage == WeightStorage::w4_packed) {
        // W4A16 decode weights from an offline pack (qwen/w4_pack.h). The pack
        // records its scale group (32/64/128); a missing record falls back to the
        // canonical 128 only when the geometry proves it (the checker below).
        _format = dsl::QwenWeightFormat::w4;
        _storage = WeightStorage::w4_packed;
    } else {
        return invalid("token_embd must be packed bf16, fp8, native q8_0 or w4 "
                       "(dense F32 embedding unsupported)");
    }
    _decode_group = embd->group != 0u ? embd->group : dsl::qwen_weight_group(_format);
    _cfg.fp8_group = _format == dsl::QwenWeightFormat::q8_0 ? dsl::kQwenQ8BlockSize
                    : _format == dsl::QwenWeightFormat::w4 ? _decode_group
                                                           : kFp8GroupSize;
    _prefill_format = _format;
    _prefill_storage = _storage;
    _prefill_group = _decode_group;
    _token_embd = embd;
    if (auto s = check_matrix(*embd, g.hidden, g.vocab, _format); !s.ok()) { return s; }
    const auto *onorm = weights.find(QwenTensorRole::output_norm);
    if (onorm == nullptr) { return invalid("output_norm weight missing"); }
    if (auto s = check_dense(*onorm, g.hidden); !s.ok()) { return s; }
    _output_norm = onorm;
    // Plan option (c): the LM head must stay 8-bit in a W4 session. The dual-copy
    // pack carries the FP8 payload of token_embd under "<name>.fp8"; when it is
    // present it owns the LM head (W4 decoding of the embedding rows is fine, the
    // LM head is not), and every prefill projection must have its own partner.
    _lm_head = _token_embd;
    if (_format == dsl::QwenWeightFormat::w4) {
        const auto *lm8 = weights.find(luisa::string{_token_embd->name} + ".fp8");
        if (lm8 != nullptr) {
            if (lm8->storage != WeightStorage::fp8_packed) {
                return invalid("token_embd.fp8 partner must be fp8_packed");
            }
            if (auto s = check_matrix(*lm8, g.hidden, g.vocab, dsl::QwenWeightFormat::fp8_e4m3,
                                      lm8->group != 0u ? lm8->group : kFp8GroupSize);
                !s.ok()) {
                return s;
            }
            _lm_head = lm8;
            _dual_prefill = true;
            _prefill_format = dsl::QwenWeightFormat::fp8_e4m3;
            _prefill_storage = WeightStorage::fp8_packed;
            _prefill_group = lm8->group != 0u ? lm8->group : kFp8GroupSize;
        } else if (_w4_lm_head_fp8) {
            return invalid("a W4 session requires the 8-bit LM head copy "
                           "(token_embd.fp8 from the dual-copy pack); pass "
                           "dsl_w4_lm_head_8bit=false only for an explicit A/B arm");
        }
    }

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
            // Split-KV decode attention nodes (opt_plan/03 section 4.2). Both are
            // recorded for every decode token when the family is enabled; the
            // compile-time guards decide which one runs, so the graph never changes.
            lw.p_attn_split = lname("attn_decode_split", l);
            lw.p_attn_combine = lname("attn_decode_combine", l);
        } else {
            lw.p_conv = lname("gdn_conv_silu", l);
            lw.p_gates = lname("gdn_gates", l);
            lw.p_qk_norm = lname("gdn_qk_l2norm", l);
            lw.p_recurrent = lname("gdn_recurrent", l);
            lw.p_chunk = lname("gdn_wy_ut_chunk64", l);
            lw.p_gated_norm = lname("gdn_gated_norm", l);
            lw.p_recurrent_norm = lname("gdn_recurrent_norm", l);
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
            // The activation workspace geometry (dsl_ops.h 4c) sizes the z arena
            // by gdn_v_n and the alpha/beta arenas by gdn_v_heads, so a model
            // whose GDN projection widths differ would silently overflow. Fail
            // closed instead.
            if (g.gdn_z_n != _cfg.gdn_v_n() || g.gdn_gate_n != _cfg.gdn_v_heads) {
                return invalid("model GDN projection widths do not match the activation workspace geometry "
                               "(z must equal gdn_v_n, alpha/beta must equal gdn_v_heads)");
            }
            if (auto s = check_dense(*lw.gdn_conv1d, g.conv_kernel * g.gdn_qkv_n); !s.ok()) { return s; }
            if (auto s = check_dense(*lw.gdn_a, g.gdn_v_heads); !s.ok()) { return s; }
            if (auto s = check_dense(*lw.gdn_dt_bias, g.gdn_v_heads); !s.ok()) { return s; }
            if (auto s = check_dense(*lw.gdn_norm, g.gdn_head_dim); !s.ok()) { return s; }
            _gdn_slot[static_cast<size_t>(l)] = gdn_slot++;
        }
  }
  // ---------------------------------------------------------------------------
  // OFFLINE merged input projections (opt_plan/03 section 3 + D5).
  // A merged session's per-role pointers above are segment ALIASES into a merged
  // matrix, so every existing consumer (prefill GEMMs, the FFN path, the calibration
  // route) keeps reading its own rows. What this block does:
  //  * proves each layer's descriptor against the frozen geometry (kind, segment
  //    roles/rows/offsets, K = hidden, N = sum of the segments, format/group);
  //  * records the row boundaries the merged decode kernel routes on;
  //  * enforces the all-or-nothing rule: a weight set with SOME layers merged is a
  //    corrupt pack, not a tuning opportunity.
  // The route itself is only ENABLED in compile_all() once the merged kernels compile.
  // ---------------------------------------------------------------------------
  {
      using R = QwenTensorRole;
      QwenTensorRole attn_roles[3] = {R::attn_q, R::attn_k, R::attn_v};
      uint64_t attn_rows[3] = {g.q_proj_n, g.kv_proj_n, g.kv_proj_n};
      QwenTensorRole gdn_roles[4] = {R::gdn_qkv, R::gdn_gate, R::gdn_alpha, R::gdn_beta};
      uint64_t gdn_rows[4] = {g.gdn_qkv_n, g.gdn_z_n, g.gdn_gate_n, g.gdn_gate_n};
      uint64_t merged_layers = 0u;
      for (uint64_t l = 0; l < g.layers; l++) {
          auto &lw = _layer_weights[static_cast<size_t>(l)];
          if (lw.full_attention) {
              const auto *pq = weights.merged_parent_of(*lw.attn_q);
              const auto *pk = weights.merged_parent_of(*lw.attn_k);
              const auto *pv = weights.merged_parent_of(*lw.attn_v);
              if (pq == nullptr && pk == nullptr && pv == nullptr) { continue; }
              if (pq == nullptr || pk != pq || pv != pq) {
                  return invalid("merged attention projection: Q/K/V do not share one merged "
                                 "matrix (corrupt pack)", luisa::string{std::to_string(l)});
              }
              if (auto s = check_merged_parent(*pq, QwenMergeKind::attn_qkv, attn_roles, attn_rows,
                                               g.hidden, _storage, _decode_group);
                  !s.ok()) {
                  return s;
              }
              lw.merged_attn = pq;
              lw.merged_q_rows = static_cast<uint32_t>(g.q_proj_n);
              lw.merged_kv_rows = static_cast<uint32_t>(g.kv_proj_n);
              lw.p_merged_proj = lname("proj_merged_attn_qkv", l);
          } else {
              const auto *pqkv = weights.merged_parent_of(*lw.gdn_qkv);
              const auto *pz = weights.merged_parent_of(*lw.gdn_gate);
              const auto *pa = weights.merged_parent_of(*lw.gdn_alpha);
              const auto *pb = weights.merged_parent_of(*lw.gdn_beta);
              if (pqkv == nullptr && pz == nullptr && pa == nullptr && pb == nullptr) { continue; }
              if (pqkv == nullptr || pz != pqkv || pa != pqkv || pb != pqkv) {
                  return invalid("merged GDN projection: QKV/Z/alpha/beta do not share one merged "
                                 "matrix (corrupt pack)", luisa::string{std::to_string(l)});
              }
              if (auto s = check_merged_parent(*pqkv, QwenMergeKind::gdn_qkv_z_ab, gdn_roles,
                                               gdn_rows, g.hidden, _storage, _decode_group);
                  !s.ok()) {
                  return s;
              }
              lw.merged_gdn = pqkv;
              lw.merged_qkv_rows = static_cast<uint32_t>(g.gdn_qkv_n);
              lw.merged_z_rows = static_cast<uint32_t>(g.gdn_z_n);
              lw.merged_gate_heads = static_cast<uint32_t>(g.gdn_gate_n);
              lw.p_merged_proj = lname("proj_merged_gdn_qkv_z_ab", l);
          }
          merged_layers++;
      }
      if (merged_layers != 0u && merged_layers != g.layers) {
          return invalid("the weight set is only partially merged: every layer must come from "
                         "one projection pack",
                         luisa::string{std::to_string(merged_layers)} + "/" +
                             luisa::string{std::to_string(g.layers)});
      }
      _merged_available = merged_layers != 0u;
      if (_merged_available) {
          const auto merged = weights.merged_matrices();
          _merged_matrices = merged.size();
          for (const auto *m : merged) { _merged_segments += m->segments.size(); }
          _merged_pack_name = luisa::string{weights.projection_pack_source()};
          _merged_pack_format = luisa::string{weights.projection_pack_format()};
          LUISA_INFO("qwen dsl: offline merged input projections present ({} matrices / {} segment "
                     "aliases, format {}, source {})",
                     _merged_matrices, _merged_segments,
                     _merged_pack_format.empty() ? "unknown" : _merged_pack_format.c_str(),
                     _merged_pack_name.empty() ? "unknown" : _merged_pack_name.c_str());
        }
    }
  // Plan option (c): resolve the FP8 prefill partners of every matrix the rows>1
  // route touches. The rule is all-or-nothing per session (a partially populated
  // pack is a corrupt pack, not a tuning opportunity) and it is decided HERE, at
  // create, so nothing can change mid-request.
      if (_format == dsl::QwenWeightFormat::w4) {
        uint32_t found = 0u, expected = 0u;
        const auto partner = [&](const QwenDeviceTensor *t) -> const QwenDeviceTensor * {
            if (t == nullptr) { return nullptr; }
            const auto *p = weights.find(luisa::string{t->name} + ".fp8");
            if (p != nullptr) { found++; }
            return p;
        };
        for (auto &lw : _layer_weights) {
            expected += 3u;// ffn gate/up/down
            lw.pf.ffn_gate = partner(lw.ffn_gate);
            lw.pf.ffn_up = partner(lw.ffn_up);
            lw.pf.ffn_down = partner(lw.ffn_down);
            if (lw.full_attention) {
                expected += 4u;
                lw.pf.attn_q = partner(lw.attn_q);
                lw.pf.attn_k = partner(lw.attn_k);
                lw.pf.attn_v = partner(lw.attn_v);
                lw.pf.attn_output = partner(lw.attn_output);
            } else {
                expected += 5u;
                lw.pf.gdn_qkv = partner(lw.gdn_qkv);
                lw.pf.gdn_gate = partner(lw.gdn_gate);
                lw.pf.gdn_alpha = partner(lw.gdn_alpha);
                lw.pf.gdn_beta = partner(lw.gdn_beta);
                lw.pf.gdn_out = partner(lw.gdn_out);
            }
        }
        const bool have_lm = _lm_head != _token_embd;
        if (_w4_prefill_in_kernel) {
            // Plan option (b), explicitly requested: prefill and the decode GEMV
            // both read the W4 tensors and the prefill GEMM dequantizes in-kernel.
            // The 8-bit LM head copy is still used when the pack carries it (the
            // plan requires the head to stay 8-bit), but the prefill partners are
            // deliberately NOT wired up, so this is a real A/B and not a relabel.
            LUISA_INFO("qwen dsl: W4 decode + IN-KERNEL W4 prefill (plan option (b) A/B arm, "
                       "explicitly requested); prefill runs the 4-bit GEMM family");
            if (!have_lm && _w4_lm_head_fp8) {
                return invalid("the option-(b) W4 arm still needs the 8-bit LM head copy "
                               "(token_embd.fp8); pass dsl_w4_lm_head_8bit=false for a "
                               "fully 4-bit A/B (not recommended)");
            }
            return Status::success();
        }
        if (found != 0u && (!have_lm || found != expected)) {
            return invalid("dual-copy W4 pack is incomplete: expected an FP8 partner for "
                           "every packed matrix and the LM head, found " +
                           luisa::string{std::to_string(found)} + "/" +
                           luisa::string{std::to_string(expected)} +
                           (have_lm ? "" : " plus the LM head copy"));
        }
        if (found == expected && expected != 0u && have_lm) {
            _dual_prefill = true;
            _prefill_format = dsl::QwenWeightFormat::fp8_e4m3;
            _prefill_storage = WeightStorage::fp8_packed;
            // Validate every partner against the rows>1 shapes with the FP8 group
            // recorded in its own tensor (canonical 64 when the pack omits it).
            const auto check_pf = [&](const QwenDeviceTensor *p, uint64_t k, uint64_t n) -> Status {
                if (p == nullptr) { return Status::success(); }
                return check_matrix(*p, k, n, dsl::QwenWeightFormat::fp8_e4m3);
            };
            for (uint64_t l = 0; l < g.layers; l++) {
                const auto &lw = _layer_weights[static_cast<size_t>(l)];
                if (auto s = check_pf(lw.pf.ffn_gate, g.hidden, g.ffn); !s.ok()) { return s; }
                if (auto s = check_pf(lw.pf.ffn_up, g.hidden, g.ffn); !s.ok()) { return s; }
                if (auto s = check_pf(lw.pf.ffn_down, g.ffn, g.hidden); !s.ok()) { return s; }
                if (lw.full_attention) {
                    if (auto s = check_pf(lw.pf.attn_q, g.hidden, g.q_proj_n); !s.ok()) { return s; }
                    if (auto s = check_pf(lw.pf.attn_k, g.hidden, g.kv_proj_n); !s.ok()) { return s; }
                    if (auto s = check_pf(lw.pf.attn_v, g.hidden, g.kv_proj_n); !s.ok()) { return s; }
                    if (auto s = check_pf(lw.pf.attn_output, g.o_proj_k, g.hidden); !s.ok()) { return s; }
                } else {
                    if (auto s = check_pf(lw.pf.gdn_qkv, g.hidden, g.gdn_qkv_n); !s.ok()) { return s; }
                    if (auto s = check_pf(lw.pf.gdn_gate, g.hidden, g.gdn_z_n); !s.ok()) { return s; }
                    if (auto s = check_pf(lw.pf.gdn_alpha, g.hidden, g.gdn_gate_n); !s.ok()) { return s; }
                    if (auto s = check_pf(lw.pf.gdn_beta, g.hidden, g.gdn_gate_n); !s.ok()) { return s; }
                    if (auto s = check_pf(lw.pf.gdn_out, g.gdn_out_k, g.hidden); !s.ok()) { return s; }
                }
            }
            LUISA_INFO("qwen dsl: W4 decode + FP8 prefill second copy (plan option (c)); "
                       "{} partner matrices, group {}", expected, _prefill_group);
        } else if (!_w4_prefill_in_kernel) {
            return invalid("a W4 session needs an FP8 prefill copy (plan option (c)); the "
                           "in-kernel W4 prefill arm (plan option (b)) is opt-in via "
                           "dsl_w4_prefill_in_kernel and is never the silent default");
        } else {
            LUISA_INFO("qwen dsl: W4 decode with in-kernel W4 prefill (plan option (b) A/B "
                       "arm, explicitly requested): prefill runs the 4-bit GEMM family");
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
                           _kv_page_bytes, _kv_storage.dtype,
                           _cfg.kv_heads, _kv_storage.layout);
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
    // Aliased activation pool (opt_plan/09 V6 + N3.5). qwen_workspace_geometry
    // is the single source of truth for the per-row sizes; every member below is
    // a handle copy of its arena, so the executor keeps stable buffer handles
    // while the footprint drops from ~116k to ~52k floats per chunk row.
    const auto geo = d::qwen_workspace_geometry(_cfg, C);
    const uint64_t q_elems = C * _cfg.q_heads * _cfg.attn_head_dim;
    const uint64_t gdn_conv = C * _cfg.gdn_qkv_n();
    const uint64_t gdn_gate = C * _cfg.gdn_v_heads;
    const uint64_t vocab = _cfg.vocab;
    // Every workspace buffer must fit the uint32 kernel index envelope.
    const uint64_t counts[] = {C, C * hidden, C * ffn, geo.pool_elems, geo.hidden_elems,
                               geo.normed_elems, geo.residual_elems, q_elems, gdn_conv,
                               gdn_gate, vocab, hidden};
    for (auto c : counts) {
        if (c > 0xFFFFFFFFull) {
            return invalid("workspace element count exceeds uint32 (reduce prefill_chunk)");
        }
    }
    for (uint32_t i = 0u; i < d::kQwenWorkspaceArenaCount; i++) {
        if (C * geo.arena_floats[i] > 0xFFFFFFFFull) {
            return invalid("workspace arena element count exceeds uint32 (reduce prefill_chunk)");
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
    // Aliased pool: one allocation per arena (see dsl_ops.h for the lifetime
    // proof and qwen_workspace_members() for the member -> arena mapping).
    Buffer<float> arena[d::kQwenWorkspaceArenaCount];
    for (uint32_t i = 0u; i < d::kQwenWorkspaceArenaCount; i++) {
        arena[i] = mk_f(C * geo.arena_floats[i]);
        if (!arena[i]) {
            return err(StatusCode::out_of_memory, "workspace arena allocation failed");
        }
    }
    _ids = mk_u(C);
    _ids_host = mk_u(C);
    // Host staging spans the FULL context: prefill records every chunk into
    // one command list, so each chunk's borrowed span must stay distinct and
    // unmodified until the single commit.
    _host_ids.resize(_max_context);
    // Permanent buffers: the residual stream lives across a whole layer and both
    // residual projections read it, so it is never aliased.
    _hidden = mk_f(geo.hidden_elems);
    _normed = mk_f(geo.normed_elems);
    _residual = mk_f(geo.residual_elems);
    // Aliases. A0: ffn gate | gdn qkv | attention q projection.
    _ffn_gate = arena[0];
    _proj_a = arena[0];
    _attn_proj_q = arena[0];
    // A1: ffn up | gdn conv output | attention q.
    _ffn_up = arena[1];
    _gdn_conv = arena[1];
    _attn_q = arena[1];
    // A2: ffn act | gdn out | attention gate.
    _ffn_act = arena[2];
    _gdn_out = arena[2];
    _attn_gate = arena[2];
    // A3: gdn z | attention out.
    _proj_b = arena[3];
    _attn_out = arena[3];
    // A4: gdn w | gdn normed | gdn alpha | attention k projection.
    _gdn_w = arena[4];
    _gdn_normed = arena[4];
    _gdn_alpha = arena[4];
    _attn_proj_k = arena[4];
    // A5: gdn v_new | raw beta | attention v projection.
    _gdn_vnew = arena[5];
    _proj_c = arena[5];
    _attn_proj_v = arena[5];
    // A6/A7: gdn q/k | attention k/v.
    _gdn_q = arena[6];
    _attn_k = arena[6];
    _gdn_k = arena[7];
    _attn_v = arena[7];
    // A8/A9: tiny GDN decay buffers (live next to every other GDN buffer).
    _gdn_beta = arena[8];
    _gdn_g = arena[9];
    _last_normed = mk_f(hidden);
    _logits = mk_f(vocab);
    _greedy_history = mk_u(_max_context);
    _greedy_values = mk_f(d::qwen_greedy_argmax_blocks);
    _greedy_indices = mk_u(d::qwen_greedy_argmax_blocks);
    _greedy_invalid = mk_u(d::qwen_greedy_argmax_blocks);
    // Device-resident decode state (opt_plan/03 section 2.3): one 16-byte buffer
    // the static graph reads instead of per-token host scalars. Host staging for
    // its uploads is a persistent member (see the note on _host_state).
    _dev_state = mk_u(d::qwen_dec_state_count);
    _host_state.assign(d::qwen_dec_state_count, 0u);
    _dummy_f = mk_f(1u);
    _dummy_u = mk_u(1u);
    if (!_ids || !_ids_host || !_hidden || !_normed || !_residual || !_proj_a || !_proj_b ||
        !_proj_c || !_ffn_gate || !_ffn_up || !_ffn_act || !_attn_q || !_attn_gate || !_attn_k ||
        !_attn_v || !_attn_out || !_gdn_conv || !_gdn_q || !_gdn_k || !_gdn_beta || !_gdn_g ||
        !_gdn_out || !_gdn_w || !_gdn_vnew || !_gdn_normed || !_gdn_alpha || !_attn_proj_q || !_attn_proj_k || !_attn_proj_v ||
        !_last_normed || !_logits || !_greedy_history || !_greedy_values ||
        !_greedy_indices || !_greedy_invalid || !_dev_state || !_dummy_f || !_dummy_u) {
        return err(StatusCode::out_of_memory, "workspace allocation failed");
    }
    _workspace_chunk_bytes = geo.chunk_bytes;
    _workspace_pool_floats_per_row = geo.pool_floats_per_row;
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
    // Norm specialization (opt_plan/09 N3.1): with a compile-time dim and block
// size each thread caches its dim/block elements (10) plus the norm weight in
// registers, so the normalization pass does not re-read x/sum_out from global
// memory (the generic wide kernel does, once per launch, 32+ times a token).
// The generic factory stays reachable for every other hidden size and is what
// the differential tests keep validating against.
_rmsnorm_reg = _rmsnorm_reg && _cfg.hidden == d::qwen35_rmsnorm_model_dim;
const auto norm_block = 256u;
const auto norm_plain = _rmsnorm_reg
    ? d::make_qwen35_rmsnorm_reg_kernel(d::QwenNormEpilogue::plain, _cfg.hidden, norm_block)
    : d::make_qwen35_rmsnorm_kernel(d::QwenNormEpilogue::plain, norm_block);
if (auto s = compile1(norm_plain, _rmsnorm_plain); !s.ok()) { return s; }
const auto norm_snapshot = _rmsnorm_reg
    ? d::make_qwen35_rmsnorm_reg_kernel(d::QwenNormEpilogue::snapshot, _cfg.hidden, norm_block)
    : d::make_qwen35_rmsnorm_kernel(d::QwenNormEpilogue::snapshot, norm_block);
if (auto s = compile1(norm_snapshot, _rmsnorm_snapshot); !s.ok()) { return s; }
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
    if (auto s = compile1(d::make_qwen_greedy_argmax_finish_dev_kernel(), _greedy_argmax_finish_dev); !s.ok()) { return s; }
    // A BF16 KV cache cannot be appended inside the prepare kernel (two codes share
 // a word and adjacent channels belong to different lanes), so that path compiles
 // the prepare WITHOUT the inline append and runs the packed append kernel below.
 const bool kv_inline_append = !_kv_packed_append;
 if (auto s = compile1(d::make_qwen_attn_prepare_kernel(kv_inline_append, true, _kv_storage), _attn_prepare); !s.ok()) { return s; }
    // Device-state realizations of the four per-token scalar consumers of a
    // decode step. They are compiled for every variant (BF16 and FP8) because the
    // static graph is not format-specific; the reference kernels above stay for
    // prefill, graph-off debugging and the MTP verify/replay paths.
    if (auto s = compile1(d::make_qwen_attn_prepare_dev_kernel(kv_inline_append, _kv_storage), _attn_prepare_dev); !s.ok()) { return s; }
 // Packed (BF16) KV append: one thread per 32-bit word, both codes in one thread.
 // Compiled only when the request actually stores packed words, so the FP32 arm
 // keeps exactly its previous launch list.
 if (_kv_fp8_append) {
     // FP8 online-scale append (one CTA per (64-position block, kv_head)): the
     // ONLY writer of an fp8 cache. Host-position and device-state realizations,
     // so the static decode graph keeps a body that needs no per-token update.
     if (auto s = compile1(d::make_qwen_kv_append_fp8_kernel(_kv_storage, false), _attn_kv_append_fp8); !s.ok()) { return s; }
     if (auto s = compile1(d::make_qwen_kv_append_fp8_kernel(_kv_storage, true), _attn_kv_append_fp8_dev); !s.ok()) { return s; }
 } else if (_kv_packed_append) {
     if (auto s = compile1(d::make_qwen_kv_append_packed_kernel(_kv_storage, false), _attn_kv_append); !s.ok()) { return s; }
     if (auto s = compile1(d::make_qwen_kv_append_packed_kernel(_kv_storage, true), _attn_kv_append_dev); !s.ok()) { return s; }
 }
    if (auto s = compile1(d::make_qwen_attn_prefill_kernel(_kv_storage), _attn_prefill); !s.ok()) { return s; }
    // `_attn_split_threshold` (0 = unguarded) makes both legacy realizations return
    // immediately once the split-KV family owns the range; the guard is a
    // compile-time constant and kv_len stays a runtime/device value, which is what
    // keeps one captured graph valid across the threshold (opt_plan/03 section 4.2).
    if (auto s = compile1(d::make_qwen_attn_decode_kernel(_attn_split_threshold, _kv_storage), _attn_decode); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_attn_decode_dev_kernel(_attn_split_threshold, _kv_storage), _attn_decode_dev); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gdn_conv_silu_kernel(), _gdn_conv_sh); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gdn_conv_silu_dev_kernel(), _gdn_conv_dev); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gdn_gates_kernel(), _gdn_gates_sh); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gdn_decode_gates_kernel(_format), _gdn_decode_gates_sh); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gdn_qk_l2norm_kernel(), _gdn_qk_norm_sh); !s.ok()) { return s; }
 // Fused decode conv+SiLU+QK-norm (opt_plan/09 N3.3). Compiled only when the
 // model geometry supports it; the reference conv/qk-norm pair above stays
 // compiled as the differential path and for rows > 1.
 if (_gdn_decode_fused) {
     const auto fused_plan = d::plan_qwen35_gdn_decode_fused_dispatch(
         _cfg.gdn_qk_heads, _cfg.gdn_head_dim, _cfg.gdn_v_n());
     if (!fused_plan.ok()) {
         LUISA_WARNING("qwen dsl: fused GDN decode unavailable ({}); using the two-launch pair",
                       fused_plan.status.message);
         _gdn_decode_fused = false;
     } else {
         if (auto s = compile1(d::make_qwen35_gdn_decode_fused_kernel(), _gdn_decode_fused_sh); !s.ok()) { return s; }
         if (auto s = compile1(d::make_qwen35_gdn_decode_fused_dev_kernel(), _gdn_decode_fused_dev); !s.ok()) { return s; }
     }
 }    // Model specializations with a COMPILE-TIME GDN head dimension
 // (opt_plan/09 V4): the lane-owned 128-float state column and its key-axis
 // indices become constants, so the column is register-resident instead of a
 // runtime-indexed local array. The generic factories stay the reference for
 // every other geometry and for the differential tests; _gdn_state_specialized
 // records which realization was compiled.
 const auto gdn_hd = _cfg.gdn_head_dim;
 _gdn_state_specialized = _gdn_state_specialized && gdn_hd == d::qwen35_gdn_model_head_dim;
 // 0 tells the factories to return the generic runtime-extent realization (the
 // only value the specializations do not claim), so the A/B flag really selects
 // which kernel is compiled.
 const auto gdn_specialize_hd = _gdn_state_specialized ? gdn_hd : 0u;
 if (auto s = compile1(d::make_qwen35_gdn_recurrent_kernel(gdn_specialize_hd), _gdn_recurrent_sh); !s.ok()) { return s; }
 if (auto s = compile1(d::make_qwen35_gdn_decode4_kernel(), _gdn_decode4_sh); !s.ok()) { return s; }
 // FUSED one-CTA recurrence + gated output RMSNorm (opt_plan/03 section 5).
 // Only the model geometry (T=1, head_dim 128) can host it, and the
 // shared-memory budget is checked before the compile instead of assumed. Any
 // refusal clears the SELECTED state and keeps the recorded reason: the decode
 // schedule then stays exactly what it is today (decode4 + separate gated norm).
 if (_gdn_recurrent_norm) {
     const auto fused_status = d::validate_qwen35_gdn_recurrent_norm_inputs(
         1u, _cfg.gdn_v_heads, _cfg.gdn_qk_heads, _cfg.gdn_head_dim);
     if (!fused_status.ok()) {
         _gdn_recurrent_norm = false;
         _gdn_recurrent_norm_reason =
             luisa::format("not the model GDN geometry: {}", fused_status.message);
         LUISA_INFO("qwen dsl: fused GDN recurrent+norm not used ({})",
                    fused_status.message);
     } else if (d::qwen_gdn_recurrent_norm_shared_bytes(_cfg.gdn_head_dim) > 32ull * 1024ull) {
         _gdn_recurrent_norm = false;
         _gdn_recurrent_norm_reason = "shared-memory budget exceeded";
         LUISA_INFO("qwen dsl: fused GDN recurrent+norm not used (shared-memory budget)");
     } else {
         if (auto s = compile1(d::make_qwen35_gdn_recurrent_norm_kernel(), _gdn_recurrent_norm_sh); !s.ok()) { return s; }
         _gdn_recurrent_norm_reason =
             "fused one-CTA-per-head recurrent + gated norm (1 launch)";
     }
 } else if (_gdn_recurrent_norm_reason.empty()) {
     _gdn_recurrent_norm_reason = "not requested (dsl_gdn_recurrent_norm false)";
 }
 if (auto s = compile1(d::make_qwen35_gdn_chunk_kernel(gdn_specialize_hd), _gdn_chunk_sh); !s.ok()) { return s; }
    if (auto s = compile1(d::make_qwen_gdn_gated_norm_kernel(), _gdn_gated_norm_sh); !s.ok()) { return s; }

    // ---------------------------------------------------------------------
    // Vectorized 128-bit decode realizations (opt_plan/09 N2). The 32-bit
    // realizations above stay compiled as the reference path and as the ragged
    // fallback; this block only ADDS the uint4/float4 family and records which one
    // the engine will dispatch. Fail closed ONCE, at create time, against every
    // projection shape the decode route would see -- never per shape mid-token.
    // MEASURED format rule: the 128-bit branchless-LUT-free E4M3 decode lost to the
    // 32-bit shared-LUT realization on every model shape on this box (RTX 4080 SUPER:
    // gdn_z 529 -> 425 GB/s, gdn_out 583 -> 556, attn_q 613 -> 584, ffn_gate_up
    // 637 -> 596, ffn_down 631 -> 613, gdn gates 6.05 -> 6.25 us), while the BF16
    // 128-bit realization is never slower and wins where the 32-bit load count
    // matters (gdn gates 5.86 -> 5.44 us, gdn_z +2.0 %, attn_q +0.8 %). FP8
    // therefore keeps the LUT path unless dsl_decode_packed_fp8 explicitly asks
    // for the rejected variant (so the measurement stays reproducible).
    if (_decode_packed && _format == d::QwenWeightFormat::w4) {
        // No 128-bit W4 realization exists (one uint4 would be 32 codes, a
        // 1024-element lane stride, and it has not been measured): the 32-bit
        // decode GEMV stays in charge instead of pretending the family covers W4.
        _decode_packed = false;
        LUISA_INFO("qwen dsl: 128-bit decode realizations not implemented for W4; "
                   "using the 32-bit decode GEMV");
    }
    if (_decode_packed && _format == d::QwenWeightFormat::fp8_e4m3 &&
        !_decode_packed_fp8) {
        _decode_packed = false;
        LUISA_INFO("qwen dsl: 128-bit decode realizations not used for FP8 (measured "
                   "slower than the 32-bit shared-LUT decode; --decode-packed-fp8 1 to "
                   "override)");
    }
    if (_decode_packed && !_decode_specialized) {
        // MTP verify/replay and the plain reference decode must keep ONE reduction
        // order; the packed family re-partitions the K sum, so it is disabled there.
        LUISA_INFO("qwen dsl: 128-bit decode realizations disabled because "
                   "dsl_decode_specialized is false (MTP/reference reduction order)");
        _decode_packed = false;
    }
    if (_decode_packed) {
        const auto group = _decode_group;
        Status shape_status = Status::success();
        const auto check_shape = [&](const QwenDeviceTensor *t) {
            if (t == nullptr || !shape_status.ok()) { return; }
            auto planned = d::plan_qwen35_decode_gemv_packed_dispatch(
                t->dims[1], t->dims[0], _format, group);
            if (!planned.ok()) { shape_status = planned.status; }
        };
        for (const auto &lw : _layer_weights) {
            check_shape(lw.ffn_gate);
            check_shape(lw.ffn_up);
            check_shape(lw.ffn_down);
            if (lw.full_attention) {
                check_shape(lw.attn_q);
                check_shape(lw.attn_k);
                check_shape(lw.attn_v);
                check_shape(lw.attn_output);
            } else {
                check_shape(lw.gdn_qkv);
                check_shape(lw.gdn_gate);
                check_shape(lw.gdn_alpha);
                check_shape(lw.gdn_beta);
                check_shape(lw.gdn_out);
                // The tied LM head is not a layer weight but the largest single weight
                // stream of a decode step, so it goes through the same fail-closed check.
                check_shape(_token_embd);
            }
        }
        if (!shape_status.ok()) {
            LUISA_WARNING("qwen dsl: 128-bit decode realizations not applicable to a "
                          "projection shape ({}: {}); keeping the 32-bit path",
                          shape_status.message, shape_status.detail);
            _decode_packed = false;
        }
    }
    if (_decode_packed) {
        if (auto s = compile1(d::make_qwen35_decode_gemv_packed_kernel(
                              _format, d::QwenMatmulEpilogue::none), _gemv_packed); !s.ok()) { return s; }
        if (auto s = compile1(d::make_qwen35_decode_gemv_packed_kernel(
                              _format, d::QwenMatmulEpilogue::residual), _gemv_packed_residual); !s.ok()) { return s; }
        if (auto s = compile1(d::make_qwen_ffn_decode_packed_kernel(_format), _ffn_decode_packed); !s.ok()) { return s; }
        if (auto s = compile1(d::make_qwen_gdn_decode_gates_packed_kernel(_format),
                              _gdn_decode_gates_packed); !s.ok()) { return s; }
    }
  // ---------------------------------------------------------------------------
  // OFFLINE merged input projections (opt_plan/03 section 3 + D5).
  // The merged decode GEMV is ADDITIVE: it replaces 3 launches per attention layer
  // and 3 per GDN layer at rows == 1 (attention Q+K+V -> 1; GDN QKV+Z+gate -> 1)
  // and nothing else. Everything that could change numerics or the recorded body's
  // invariance is decided HERE, once, fail closed:
  //  * a pack must be present (resolve_weights proved the descriptors);
  //  * W4 is refused (no 4-bit merged realization), and so is the 128-bit packed
  //    decode family (it re-partitions K, i.e. it would no longer reproduce the
  //    unmerged accumulation order the merged route is contracted to keep);
  //  * dsl_decode_specialized must be true (MTP verify/replay and the plain
  //    reference decode keep ONE reduction order and stay unmerged);
  //  * every layer's merged geometry must satisfy the launch plan.
  // A refusal is reported and falls back to the ordinary per-projection GEMVs on the
  // SAME (merged, sub-view) buffers, which is numerically the unmerged path.
  // ---------------------------------------------------------------------------
  if (_merged_decode) {
      if (!_merged_available) {
          _merged_decode = false;
      } else if (_format == d::QwenWeightFormat::w4) {
          LUISA_INFO("qwen dsl: merged decode projections not available for W4 (no 4-bit merged "
                     "realization); using the per-projection decode GEMVs");
          _merged_decode = false;
      } else if (!_decode_specialized) {
          LUISA_INFO("qwen dsl: merged decode projections disabled because dsl_decode_specialized "
                     "is false (MTP/reference reduction order)");
          _merged_decode = false;
      } else if (_decode_packed) {
          LUISA_INFO("qwen dsl: merged decode projections disabled because the 128-bit packed "
                     "decode realization re-partitions K (its accumulation order differs from the "
                     "merged route's contract); --decode-packed 0 enables the merged route");
          _merged_decode = false;
      }
  }
  if (_merged_decode) {
      Status plan_status = Status::success();
      uint32_t saved = 0u;
      for (const auto &lw : _layer_weights) {
          if (lw.full_attention) {
              if (lw.merged_attn == nullptr) {
                  plan_status = invalid("merged attention projection is missing from the weight set");
                  break;
              }
              auto planned = d::plan_qwen35_merged_attn_gemv_dispatch(
                  lw.merged_q_rows, lw.merged_kv_rows, _cfg.hidden, _format, _decode_group);
              if (!planned.ok()) {
                  plan_status = planned.status;
                  break;
              }
              saved += 2u;
          } else {
              if (lw.merged_gdn == nullptr) {
                  plan_status = invalid("merged GDN projection is missing from the weight set");
                  break;
              }
              auto planned = d::plan_qwen35_merged_gdn_gemv_dispatch(
                  lw.merged_qkv_rows, lw.merged_z_rows, lw.merged_gate_heads, _cfg.hidden,
                  _format, _decode_group);
              if (!planned.ok()) {
                  plan_status = planned.status;
                  break;
              }
              saved += 2u;
          }
      }
      if (!plan_status.ok()) {
          LUISA_WARNING("qwen dsl: merged decode projections not applicable ({}: {}); keeping the "
                        "per-projection decode GEMVs on the merged segment views",
                        plan_status.message, plan_status.detail);
          _merged_decode = false;
      } else {
          // A compile failure is a refusal, not a broken session: the unmerged route on
          // the same segment views is numerically the unmerged path and stays available.
          auto gdn_status = compile1(d::make_qwen35_merged_gdn_gemv_kernel(_format),
                                     _gemv_merged_gdn);
          auto attn_status = gdn_status.ok()
                                 ? compile1(d::make_qwen35_merged_attn_gemv_kernel(_format),
                                            _gemv_merged_attn)
                                 : gdn_status;
          if (!gdn_status.ok() || !attn_status.ok()) {
              LUISA_WARNING("qwen dsl: merged decode projection compile failed ({}); keeping the "
                            "per-projection decode GEMVs",
                            gdn_status.ok() ? attn_status.message : gdn_status.message);
              _merged_decode = false;
          } else {
              LUISA_INFO("qwen dsl: merged decode projections ENABLED ({} launches saved per "
                         "decode token, GDN N={} + attention N={}, format {}, group {})",
                         saved, _cfg.gdn_qkv_n() + _cfg.gdn_v_n() + 2u * _cfg.gdn_v_heads,
                         _cfg.attn_q_proj_n() + 2u * _cfg.attn_kv_n(),
                         d::qwen_weight_format_name(_format), _decode_group);
              _merged_launches_saved = saved;
          }
      }
  }
  // Plan option (c): the 8-bit PREFILL copy needs its own GEMM/LM-head shader
  // set compiled for the prefill format (the decode set above is the W4 one).
    // Same factories, same shape contracts, one extra compile per variant; the
    // group is a runtime argument so the pack's group needs no specialization.
    _prefill_gemm_mt_pf = false;
    const bool lm_head_8bit = _lm_head != nullptr && _lm_head != _token_embd;
    if (lm_head_8bit) {
        const auto lm_pair = [&](d::QwenMatmulEpilogue epilogue) {
            return _decode_specialized
                       ? d::make_qwen35_decode_gemv_kernel(d::QwenWeightFormat::fp8_e4m3, epilogue)
                       : d::make_qwen_gemv_kernel(d::QwenWeightFormat::fp8_e4m3, epilogue);
        };
        if (auto s = compile1(lm_pair(d::QwenMatmulEpilogue::none), _gemv_lm); !s.ok()) { return s; }
        if (auto s = compile1(lm_pair(d::QwenMatmulEpilogue::residual), _gemv_lm_residual); !s.ok()) { return s; }
    }
    if (_dual_prefill) {
        if (auto s = compile1(d::make_qwen_gemm_kernel(_prefill_format, d::QwenMatmulEpilogue::none), _gemm_pf); !s.ok()) { return s; }
        if (auto s = compile1(d::make_qwen_gemm_kernel(_prefill_format, d::QwenMatmulEpilogue::residual), _gemm_pf_residual); !s.ok()) { return s; }
    }
    // Model-specialized decode attention prepare (opt_plan/09 N3.2). Compiled only
    // when the model geometry is exactly the specialized one (head_dim 256,
    // rope_dim 64, 16 q / 4 kv heads); the generic pair above stays the reference
    // for every other geometry and for the differential A/B.
    if (_attn_prepare_model) {
        if (auto s = d::validate_qwen35_attn_prepare_model_inputs(1u, _cfg, 0u, _max_context); !s.ok()) {
            LUISA_WARNING("qwen dsl: model attention prepare unavailable ({}: {}); using the generic kernel",
                          s.message, s.detail);
            _attn_prepare_model = false;
        }
    }
    if (_attn_prepare_model) {
        if (auto s = compile1(d::make_qwen35_attn_prepare_model_kernel(!_kv_packed_append, _kv_storage), _attn_prepare_model_sh); !s.ok()) { return s; }
        if (auto s = compile1(d::make_qwen35_attn_prepare_model_dev_kernel(!_kv_packed_append, _kv_storage), _attn_prepare_model_dev_sh); !s.ok()) { return s; }
    }
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
    for (auto &b : _gdn_states) { if (auto s = zero(b); !s.ok()) { return s; }
 }
    for (auto &b : _gdn_rings) { if (auto s = zero(b); !s.ok()) { return s; }
 }
    batch.flush(true);// no-op when empty; single synchronize otherwise
    _position = 0;
    _has_session = false;
    _greedy_next_valid = false;
    // The device decode state belongs to a session: force a re-upload before the
    // next static-graph launch. The captured graph itself stays valid (it only
    // holds buffer addresses, which reset() never changes) and is reused; a
    // previously failed capture may be retried in the new session.
    _dev_state_position = UINT32_MAX;
    _dev_state_kv_len = 0u;
    _dev_state_slot = 0u;
    _dev_state_synced = false;
    _cuda_graph_capture_failed = false;
    // A failed conditional-WHILE capture/compile is retried in the new session; an
    // already captured WHILE graph holds only stable buffer addresses and is reused
    // unchanged, so nothing about it is invalidated by a reset.
    _while_capture_failed = false;
    return Status::success();
    }

// ---------------------------------------------------------------------------
// Projection helper (packed GEMV for 1 row, GEMM otherwise).
// ---------------------------------------------------------------------------
[[nodiscard]] const QwenDeviceTensor *QwenDslEngine::prefill_partner(const QwenDeviceTensor &w) const noexcept {
    if (!_dual_prefill) { return nullptr; }
    // The partner pointers were resolved once in resolve_weights; the layer table
    // is the authority, so a lookup by identity keeps project() branch-free.
    for (const auto &lw : _layer_weights) {
        if (&w == lw.ffn_gate) { return lw.pf.ffn_gate; }
        if (&w == lw.ffn_up) { return lw.pf.ffn_up; }
        if (&w == lw.ffn_down) { return lw.pf.ffn_down; }
        if (&w == lw.attn_q) { return lw.pf.attn_q; }
        if (&w == lw.attn_k) { return lw.pf.attn_k; }
        if (&w == lw.attn_v) { return lw.pf.attn_v; }
        if (&w == lw.attn_output) { return lw.pf.attn_output; }
        if (&w == lw.gdn_qkv) { return lw.pf.gdn_qkv; }
        if (&w == lw.gdn_gate) { return lw.pf.gdn_gate; }
        if (&w == lw.gdn_alpha) { return lw.pf.gdn_alpha; }
        if (&w == lw.gdn_beta) { return lw.pf.gdn_beta; }
        if (&w == lw.gdn_out) { return lw.pf.gdn_out; }
    }
    if (&w == _lm_head || &w == _token_embd) { return _lm_head; }
    return nullptr;
}

[[nodiscard]] luisa::string QwenDslEngine::proj_name(luisa::string_view label, const QwenDeviceTensor &w) {
    luisa::string s;
    s.reserve(label.size() + 1u + w.name.size());
    s.append(label);
    s.push_back('.');
    s.append(w.name);
    return s;
}

[[nodiscard]] Status QwenDslEngine::project(QwenCommandBatch &batch, const Buffer<float> &x,
                                            const QwenDeviceTensor &w_in, const Buffer<float> &y,
                                            uint32_t rows, const Buffer<float> *residual,
                                            luisa::string_view label) {
    // Plan option (c): rows > 1 consumes the PREFILL copy (8-bit) when the session
    // has one; rows == 1 (decode) always consumes the decode copy (W4). The
    // choice is a per-session constant, so this is one pointer test, not a mode.
    const bool prefill_route = rows > 1u && _dual_prefill;
    const QwenDeviceTensor *wp = &w_in;
    if (prefill_route) {
        wp = prefill_partner(w_in);
        if (wp == nullptr) {
            return invalid("dual-copy W4 session has no prefill partner for a projection", w_in.name);
        }
    }
    const auto &w = *wp;
    const uint32_t k = static_cast<uint32_t>(w.dims[0]);
    const uint32_t n = static_cast<uint32_t>(w.dims[1]);
    // The scale group of the COPY BEING READ: decode (fp8 64 / q8_0 32 / w4 group)
    // versus the 8-bit prefill copy. Kernels take it as a runtime argument.
    const uint32_t group = prefill_route ? _prefill_group : _decode_group;
    // The packed/scales views a consumer binds: for a merged segment alias these are
    // the alias's row range inside the merged matrix (packed_view_of/scales_view_of),
    // for an ordinary tensor the whole buffer. One resolution point, so nothing below
    // has to know whether this matrix came from an offline pack.
    const auto wv = packed_view_of(w);
    const auto scales = w.scales ? scales_view_of(w) : _dummy_f.view();
    const auto res = residual != nullptr ? residual->view() : _dummy_f.view();
 // Calibration-only (opt_plan/05 section 3 item 2): accumulate this projection's
 // squared input activation into its device accumulator, in the SAME batch and
 // before the dispatch that consumes `x`, so the statistic reads exactly the
 // activation the projection reads. The identity is the DECODE tensor, i.e. the
 // name the pack manifest uses (never the .fp8 partner).
 if (auto s = record_calibration(batch, x, w_in, rows, k); !s.ok()) { return s; }
    // One annotation per projection: <label>.<GGUF weight name> (includes the
    // layer). Build it only for profiling; normal recording performs no string
    // allocation. The local outlives the immediately following annotate call.
    luisa::string name;
    if (batch.instrumented()) {
        name = proj_name(label, w);
        batch.annotate(name);
    }
    if (rows == 1u && _decode_packed && _format != d::QwenWeightFormat::w4) {
        // Vectorized 128-bit M=1 projection (opt_plan/09 N2). create() proved every
        // projection shape satisfies the packed contract, so a failure here is a real
        // error, not a ragged fallback. The activation view is 16-byte aligned (offset
        // 0 of a float buffer) and the pack view inherits the row alignment.
        const auto elems = d::qwen_packed_vec_elements(_format);
        const auto vecs = static_cast<uint32_t>(static_cast<uint64_t>(k) / elems);
        auto plan = d::plan_qwen35_decode_gemv_packed_dispatch(n, k, _format, group);
        if (!plan.ok()) { return plan.status; }
        if (plan.value->empty()) { return Status::success(); }
        auto x4 = x.view().as<float4>();
        auto w4 = wv.as<uint4>();
        if (residual != nullptr) {
            batch << _gemv_packed_residual(x4, w4, scales, res, y.view(), n, k, group, vecs)
                         .dispatch(d::qwen_dispatch_threads(*plan.value));
        } else {
            batch << _gemv_packed(x4, w4, scales, res, y.view(), n, k, group, vecs)
                         .dispatch(d::qwen_dispatch_threads(*plan.value));
        }
    } else if (rows == 1u) {
                        auto plan = _format == d::QwenWeightFormat::bf16 || !_decode_specialized
                                        ? d::plan_qwen_gemv_dispatch(1u, n, k)
                        : d::plan_qwen35_decode_gemv_dispatch(n, k, _format, group);
        if (!plan.ok()) { return plan.status; }
        if (plan.value->empty()) { return Status::success(); }
        if (residual != nullptr) {
            batch << _gemv_residual(x.view(), wv, scales, res, y.view(), 1u, n, k, group)
                         .dispatch(d::qwen_dispatch_threads(*plan.value));
        } else {
            batch << _gemv(x.view(), wv, scales, res, y.view(), 1u, n, k, group)
                         .dispatch(d::qwen_dispatch_threads(*plan.value));
        }
    } else {
        // Prefill (rows > 1).
        // CUDA tensor-core path first (opt_plan/09 N1a): exact for every shape
        // create() validated and strictly faster than the warp-per-column family
        // there (see the prefill-GEMM arm of benchmark_qwen_decode). A nullptr
        // command is a real error, not a fallback: create() proved the geometry.
        // Prefill (rows > 1). The model path uses the weight-reusing mt GEMM:
        // one warp per output column, K split across lanes, MT-row m tile, so
        // weight traffic is weight_bytes * ceil(rows / MT) with coalesced reads
        // instead of ceil(rows / 16) with 8x sector amplification. create()
        // pre-validates the mt shape contract for every projection weight, so a
        // planning failure here is a real error, not a silent fallback.
        const bool prefill_fmt = prefill_route;
        const auto &gemm_shader = prefill_fmt ? _gemm_pf : _gemm;
        const auto &gemm_residual_shader = prefill_fmt ? _gemm_pf_residual : _gemm_residual;
        const auto &gemm_mt_shader = prefill_fmt ? _gemm_mt_pf : _gemm_mt;
        const auto &gemm_mt_residual_shader = prefill_fmt ? _gemm_mt_pf_residual : _gemm_mt_residual;
        const bool mt_ok = prefill_fmt ? _prefill_gemm_mt_pf : _prefill_gemm_mt;
        // ---- EXPERIMENTAL CUDA native-Tile typed-FP8 prefill GEMM ----------------
        // Strictly opt-in: `_cuda_tile_gemm.valid()` is true ONLY when the caller set
        // QwenEngineOptions::cuda_tile_fp8_prefill and create() passed the whole gate
        // (cuda + LUISA_CUDA_TILE_IR=1 + a typed-FP8 Tile compile probe + the static
        // per-projection gate), so a default engine never reaches this branch and every
        // row below is the exact code of the tree before this path existed. When it is
        // armed it takes the shapes the MEASURED native rule serves (prefill rows >= 16
        // and N >= 256) BEFORE the native NVRTC kernel and reuses the SAME x / packed w
        // / scales / residual / y views: no repack, no widened copy, no host staging,
        // and the launch is counted so a report can prove it engaged. A nullptr command
        // is an ENGINE ERROR, not a fallback: the caller asked for this realization, so
        // silently running a different GEMM would make the reported path a lie.
        const auto tile_format = prefill_fmt ? _prefill_format : _format;
        if (_cuda_tile_gemm.valid() && rows >= d::qwen_native_gemm_min_rows &&
            d::qwen_native_gemm_prefers_columns(n) &&
            // Only the typed FP8 copy with the 64-element scale group exists on that
            // route; BF16/Q8_0/W4 prefill keeps the native/DSL chain untouched.
            d::qwen_cuda_tile_gemm_supported_format(tile_format) &&
            group == d::qwen_cuda_tile_gemm_group) {
            luisa::string tile_error;
            auto command = _cuda_tile_gemm.build_dispatch(
                residual != nullptr, x.view(), wv, scales, res, y.view(),
                rows, n, k, &tile_error);
            if (command == nullptr) {
                return err(StatusCode::internal_error,
                           "CUDA native Tile FP8 prefill GEMM dispatch failed", std::move(tile_error));
            }
            batch << std::move(command);
            _cuda_tile_gemm_launches++;
            return Status::success();
        }
        if (mt_ok) {
            auto planned = d::plan_qwen35_gemm_mt_dispatch(rows, n, k,
                                                           prefill_fmt ? _prefill_format : _format,
                                                           group, _gemm_mt_cfg);
            if (!planned.ok()) { return planned.status; }
            const auto &mt_plan = *planned.value;
            if (mt_plan.empty()) { return Status::success(); }
            // Measured selection, both arms fail closed: rows >= 16 (below it the
            // decode/MTP GEMV family owns the shape) and N >= 256 (the only
            // projection width where the DSL MT kernel was measured faster is the
            // narrow N = 32 gate/alpha/beta pair, see qwen_native_gemm_min_columns).
            if (_native_gemm.valid() && rows >= d::qwen_native_gemm_min_rows &&
                d::qwen_native_gemm_prefers_columns(n) &&
                // The native tensor-core GEMM implements BF16/FP8 only: it may
                // consume the FP8 PREFILL copy (plan option (c)) but never W4.
                d::qwen_native_gemm_supported_format(prefill_fmt ? _prefill_format : _format)) {
                luisa::string native_error;
                auto command = _native_gemm.build_dispatch(
                    residual != nullptr, x.view(), wv, scales, res, y.view(),
                    rows, n, k, &native_error);
                if (command == nullptr) {
                    return err(StatusCode::internal_error,
                               "native tensor GEMM dispatch failed", std::move(native_error));
                }
                batch << std::move(command);
                _native_gemm_launches++;
                return Status::success();
            }
            if (residual != nullptr) {
                batch << gemm_mt_residual_shader(x.view(), wv, scales, res, y.view(),
                                                 rows, n, k, group, mt_plan.column_blocks)
                             .dispatch(mt_plan.total_x, mt_plan.total_y);
            } else {
                batch << gemm_mt_shader(x.view(), wv, scales, res, y.view(),
                                        rows, n, k, group, mt_plan.column_blocks)
                             .dispatch(mt_plan.total_x, mt_plan.total_y);
            }
            return Status::success();
        }
        auto plan = d::plan_qwen_gemm_dispatch(rows, n, k);
        if (!plan.ok()) { return plan.status; }
        if (plan.value->empty()) { return Status::success(); }
        if (residual != nullptr) {
            batch << gemm_residual_shader(x.view(), wv, scales, res, y.view(), rows, n, k, group,
                                          plan.value->padded_cols)
                         .dispatch(plan.value->total_x, plan.value->total_y);
        } else {
            batch << gemm_shader(x.view(), wv, scales, res, y.view(), rows, n, k, group,
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
[[nodiscard]] int QwenDslEngine::calibration_index(const QwenDeviceTensor &w) const noexcept {
    for (size_t i = 0; i < _calibration_entries.size(); ++i) {
        if (_calibration_entries[i].weight == &w) { return static_cast<int>(i); }
    }
    return -1;
}

[[nodiscard]] Status QwenDslEngine::record_calibration(QwenCommandBatch &batch,
                                                       const compute::Buffer<float> &x,
                                                       const QwenDeviceTensor &w,
                                                       uint32_t rows, uint32_t k) {
    if (!_calibration_active) { return Status::success(); }
    const int index = calibration_index(w);
    if (index < 0) {
        // The weight is not in the calibrated set: refuse instead of silently
        // leaving a projection uncounted (a missing statistic must never look like
        // a measured one).
        return err(StatusCode::internal_error,
                   "calibration: projection is not in the calibrated set", w.name);
    }
    auto &entry = _calibration_entries[static_cast<size_t>(index)];
    if (entry.k != static_cast<uint64_t>(k)) {
        return err(StatusCode::internal_error, "calibration: projection K changed", w.name);
    }
    if (rows == 0u) { return Status::success(); }
    auto plan = d::plan_qwen_threads(k);
    if (!plan.ok()) { return plan.status; }
    if (plan.value->empty()) { return Status::success(); }
    batch.annotate("calibrate");
    batch << _calibration_acc(x.view(0u, static_cast<size_t>(rows) * k), entry.acc.view(),
                              rows, k)
                 .dispatch(d::qwen_dispatch_threads(*plan.value));
    entry.tokens += rows;
    _calibration_projections++;
    return Status::success();
}

[[nodiscard]] Status QwenDslEngine::begin_activation_calibration(compute::Stream &stream) {
    if (_calibration_active) {
        return Status::error(StatusCode::invalid_argument,
                             "calibration mode is already active");
    }
    if (!_calibration_only) {
        return Status::error(
            StatusCode::unsupported,
            "activation calibration requires QwenEngineOptions::calibration_only: the fused "
            "decode kernels compute the ffn_down input activation in-kernel, so the reference "
            "route is needed to observe every projection input");
    }
    if (!_calibration_acc || !_calibration_acc.compile_ok()) {
        // Compile ONLY the accumulator: the calibration mode must not disturb the
        // ordinary kernel set (it is entered after create() on a live engine).
        _calibration_acc = _device.compile(d::make_qwen_importance_accumulate_kernel());
        if (!_calibration_acc || !_calibration_acc.compile_ok()) {
            return Status::error(StatusCode::internal_error,
                                 "calibration: the activation accumulator kernel failed to compile");
        }
    }
    // Enumerate every projection weight the executor can consume: the twelve
    // per-layer projections and the tied LM head. One accumulator per weight.
    _calibration_entries.clear();
    const auto add = [&](const QwenDeviceTensor *w) -> Status {
        if (w == nullptr) { return Status::success(); }
        for (const auto &e : _calibration_entries) {
            if (e.weight == w) { return Status::success(); }
        }
        if (w->dims.size() != 2u) {
            return err(StatusCode::internal_error,
                       "calibration: projection weight is not a matrix", w->name);
        }
        CalibrationEntry entry;
        entry.weight = w;
        entry.k = w->dims[0];
        entry.acc = _device.create_buffer<float>(static_cast<size_t>(entry.k));
        if (!entry.acc) {
            return Status::error(StatusCode::out_of_memory,
                                 "calibration accumulator allocation failed", w->name);
        }
        entry.acc.set_name(w->name.c_str());
        _calibration_entries.push_back(std::move(entry));
        return Status::success();
    };
    for (const auto &lw : _layer_weights) {
        for (const QwenDeviceTensor *w : {lw.ffn_gate, lw.ffn_up, lw.ffn_down, lw.attn_q,
                                          lw.attn_k, lw.attn_v, lw.attn_output, lw.gdn_qkv,
                                          lw.gdn_gate, lw.gdn_alpha, lw.gdn_beta, lw.gdn_out}) {
            if (auto s = add(w); !s.ok()) { return s; }
        }
    }
    if (auto s = add(_lm_head); !s.ok()) { return s; }
    if (_calibration_entries.empty()) {
        return Status::error(StatusCode::internal_error,
                             "calibration: the executor resolved no projection weights");
    }
    // Zero every accumulator in stream order (no host staging that could dangle).
    for (auto &entry : _calibration_entries) {
        auto plan = d::plan_qwen_threads(static_cast<uint32_t>(entry.k));
        if (!plan.ok()) { return plan.status; }
        if (plan.value->empty()) { continue; }
        stream << _zero_fill(entry.acc.view(), static_cast<uint32_t>(entry.k))
                      .dispatch(d::qwen_dispatch_threads(*plan.value));
        entry.tokens = 0;
    }
    stream << compute::synchronize();
    _calibration_projections = 0;
    _calibration_active = true;
    return Status::success();
}

[[nodiscard]] Result<QwenCalibrationReport> QwenDslEngine::finish_activation_calibration(
    compute::Stream &stream) {
    if (!_calibration_active) {
        return Result<QwenCalibrationReport>::failure(StatusCode::invalid_argument,
                                                     "calibration mode is not active");
    }
    QwenCalibrationReport report;
    report.projections = _calibration_projections;
    uint64_t tokens = 0;
    for (auto &entry : _calibration_entries) {
        if (entry.tokens == 0u) {
            return Result<QwenCalibrationReport>::failure(
                StatusCode::unvalidated,
                "calibration: this projection never consumed an activation row (the run did not "
                "execute a forward pass that reaches it); refusing to invent statistics",
                entry.weight->name);
        }
        if (tokens == 0u) { tokens = entry.tokens; }
        QwenCalibrationTensor t;
        t.name = entry.weight->name;
        t.k = entry.k;
        t.tokens = entry.tokens;
        luisa::vector<float> sums(static_cast<size_t>(entry.k), 0.0f);
        stream << entry.acc.view().copy_to(luisa::span<float>{sums});
        stream << compute::synchronize();
        t.second_moment.resize(sums.size());
        const auto inv = 1.0f / static_cast<float>(entry.tokens);
        for (size_t j = 0; j < sums.size(); ++j) {
            const float mean_square = sums[j] * inv;
            if (!std::isfinite(mean_square) || mean_square < 0.0f) {
                return Result<QwenCalibrationReport>::failure(
                    StatusCode::unvalidated,
                    "calibration: accumulated statistics are not finite/non-negative",
                    t.name);
            }
            t.second_moment[j] = mean_square;
        }
        report.observed.push_back(std::move(t));
    }
    report.tensors = report.observed.size();
    report.tokens = tokens;
    report.diagnostics.push_back(Diagnostic{
        DiagnosticSeverity::info, "w4_importance_collected",
        luisa::string{"projections="} + luisa::string{std::to_string(report.projections).c_str()} +
            " tensors=" + luisa::string{std::to_string(report.tensors).c_str()} +
            " tokens=" + luisa::string{std::to_string(report.tokens).c_str()}});
    _calibration_active = false;
    return Result<QwenCalibrationReport>::success(std::move(report));
}

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
                                                         uint32_t rows, uint32_t pos0,
                                                         bool device_state) {
    const auto &lw = _layer_weights[layer];
    const uint32_t slot = _full_slot[layer];
    const auto kc = _kv_cache.k_view(slot);
    const auto vc = _kv_cache.v_view(slot);
    // fp8: the dense per-region scale arrays of THIS layer (a dummy 1-element
    // buffer for f32/bf16, where the kernels fold every scale load away). Both
    // views are captured once per call; the allocator owns them for its lifetime.
    const auto ksc = _kv_fp8_append ? _kv_cache.k_scale_view(slot) : _dummy_f.view();
    const auto vsc = _kv_fp8_append ? _kv_cache.v_scale_view(slot) : _dummy_f.view();
    const uint32_t hd = _cfg.attn_head_dim;
    const uint32_t qh = _cfg.q_heads, kvh = _cfg.kv_heads;
    const uint32_t kv_n = _cfg.attn_kv_n();
    batch.layer(layer);
  // Projections: q (with per-head gate), k, v.
  // OFFLINE merged projection (opt_plan/03 section 3 + D5): at rows == 1 the whole
  // group is ONE launch over the merged [hidden, 10240] matrix, whose rows are the
  // q/k/v rows in that order. Each warp writes its scalar straight into the
  // destination buffer of its segment, so no intermediate is created, and the
  // per-output accumulation order equals the unmerged GEMV's (bit-identical logits).
  // `lw.merged_attn` is a build-time (per-layer) fact, so the recorded body is
  // argument-invariant and the static CUDA graph needs no update. rows > 1 keeps the
  // ordinary projections (native tensor-core / DSL MT / reference GEMM untouched).
  const bool merged_proj = rows == 1u && _merged_decode && lw.merged_attn != nullptr;
  if (merged_proj) {
      batch.annotate(lw.p_merged_proj);
      auto planned = d::plan_qwen35_merged_attn_gemv_dispatch(
          lw.merged_q_rows, lw.merged_kv_rows, _cfg.hidden, _format, _decode_group);
      if (!planned.ok()) { return planned.status; }
      const auto &mp = *lw.merged_attn;
      batch << _gemv_merged_attn(_normed.view(), mp.packed.view(),
                                 mp.scales ? mp.scales.view() : _dummy_f.view(),
                                 _attn_proj_q.view(), _attn_proj_k.view(), _attn_proj_v.view(),
                                 _cfg.hidden, _decode_group, lw.merged_q_rows, lw.merged_kv_rows)
                   .dispatch(d::qwen_dispatch_threads(*planned.value));
  } else {
      if (auto s = project(batch, _normed, *lw.attn_q, _attn_proj_q, rows, nullptr, "proj"); !s.ok()) { return s; }
      if (auto s = project(batch, _normed, *lw.attn_k, _attn_proj_k, rows, nullptr, "proj"); !s.ok()) { return s; }
      if (auto s = project(batch, _normed, *lw.attn_v, _attn_proj_v, rows, nullptr, "proj"); !s.ok()) { return s; }
  }
    // Fused split + QK norm + partial RoPE + KV append at pos0..pos0+rows-1.
    // `pos0` is the real argument of the reference kernel; the device-state
    // realization reads the same position from _dev_state, which is what keeps
    // the captured decode body argument-invariant. Validation uses the recording
    // position: the capacity/int32 bounds it checks are monotone in the position,
    // so an upper bound at capture time covers every later token.
    if (auto s = d::validate_qwen_attn_prepare_inputs(rows, _cfg, pos0, _max_context, true); !s.ok()) { return s; }
    {
        if (_attn_prepare_model) {
            // Model-specialized prepare (opt_plan/09 N3.2): compile-time head_dim 256 /
            // rope_dim 64 / 16 q / 4 kv heads, one register cache per lane, one RoPE
            // pair per lane. Same append semantics, same layout, same guarded no-op on
            // a geometry mismatch. The generic realization below stays the reference.
            if (auto s = d::validate_qwen35_attn_prepare_model_inputs(rows, _cfg, pos0, _max_context); !s.ok()) { return s; }
            auto plan = d::plan_qwen35_attn_prepare_model_dispatch(rows);
            if (!plan.ok()) { return plan.status; }
            if (device_state) {
                batch << _attn_prepare_model_dev_sh(_attn_proj_q.view(), _attn_proj_k.view(), _attn_proj_v.view(),
                                                    lw.attn_q_norm->f32.view(), lw.attn_k_norm->f32.view(),
                                                    _attn_q.view(), _attn_gate.view(), _attn_k.view(), _attn_v.view(),
                                                    kc, vc, _dev_state.view(),
                                                    rows, _cfg.rope_theta, _cfg.norm_eps, _max_context)
                             .dispatch(d::qwen_dispatch_threads(*plan.value));
            } else {
                batch << _attn_prepare_model_sh(_attn_proj_q.view(), _attn_proj_k.view(), _attn_proj_v.view(),
                                                lw.attn_q_norm->f32.view(), lw.attn_k_norm->f32.view(),
                                                _attn_q.view(), _attn_gate.view(), _attn_k.view(), _attn_v.view(),
                                                kc, vc,
                                                rows, _cfg.rope_theta, _cfg.norm_eps, pos0, _max_context)
                         .dispatch(d::qwen_dispatch_threads(*plan.value));
            }
        } else {
            auto plan = d::plan_qwen_attn_prepare_dispatch(rows, _cfg);
        if (!plan.ok()) { return plan.status; }
        const uint32_t q_rows = rows * qh, k_rows = rows * kvh, v_rows = rows * kvh;
        const auto positions = _ids_host.view().as<int32_t>();
        if (device_state) {
            batch << _attn_prepare_dev(_attn_proj_q.view(), _attn_proj_k.view(), _attn_proj_v.view(),
                                   lw.attn_q_norm->f32.view(), lw.attn_k_norm->f32.view(), positions,
                                   _attn_q.view(), _attn_gate.view(), _attn_k.view(), _attn_v.view(),
                                   kc, vc, _dev_state.view(),
                                   rows, qh, kvh, hd, _cfg.rope_dim, _cfg.rope_theta, _cfg.norm_eps,
                                   _max_context, q_rows, k_rows, v_rows)
                      .dispatch(d::qwen_dispatch_threads(*plan.value));
        } else {
            batch << _attn_prepare(_attn_proj_q.view(), _attn_proj_k.view(), _attn_proj_v.view(),
                                lw.attn_q_norm->f32.view(), lw.attn_k_norm->f32.view(), positions,
                                _attn_q.view(), _attn_gate.view(), _attn_k.view(), _attn_v.view(),
                                kc, vc,
                                rows, qh, kvh, hd, _cfg.rope_dim, _cfg.rope_theta, _cfg.norm_eps,
                                pos0, _max_context, q_rows, k_rows, v_rows)
                      .dispatch(d::qwen_dispatch_threads(*plan.value));
        }
    }
    }
 // KV append outside the prepare kernel (opt_plan/05 Q3). The prepare kernels
 // above were compiled with append_kv = false on both packed paths, so the cache
 // is untouched there and the chunk is written here:
 //  * bf16: one thread per 32-bit WORD, two RNE codes per word at the
 //    ((head*blocks + pos/64)*64 + pos%64)*head_dim + channel address of the
 //    blocked layout (a per-code store would be a read-modify-write race);
 //  * fp8_e4m3: one CTA per (64-position block, kv_head) which owns the block
 //    scale as well (see above).
 // Both device-state realizations read the position from _dev_state, so a
 // captured static graph needs no update. Cost: ONE extra launch per prepare
 // call, reported as QwenEngineInfo::kv_packed_append / kv_scaled_append.
      if (_kv_fp8_append) {
          // FP8 E4M3 append with ONLINE per-(head, 64-position block) scales: one CTA
          // per (touched 64-position block, kv_head), K then V. It re-derives the block
          // maximum over every VALID value (the stored prefix plus this chunk), keeps
          // the old scale when the maximum does not grow, and otherwise rescales the
          // whole valid prefix so no earlier code is left paired with a new scale. It
          // also writes the scale, so it is the only writer of an fp8 cache.
          // Device-state realization reads the position from _dev_state, so a captured
          // decode graph still needs no update (QwenEngineInfo::kv_scaled_append).
          const auto append_plan = d::plan_qwen_kv_append_fp8_dispatch(pos0, rows, kvh, _kv_storage);
          if (!append_plan.ok()) { return append_plan.status; }
          batch.annotate(lw.p_attn_prepare);
          if (device_state) {
              batch << _attn_kv_append_fp8_dev(_attn_k.view(), _attn_v.view(),
                                              _kv_cache.k_word_view(slot), _kv_cache.v_word_view(slot),
                                              ksc, vsc, _dev_state.view(),
                                              rows, kvh, hd, pos0, _max_context)
                    .dispatch(d::qwen_dispatch_threads(*append_plan.value));
          } else {
              batch << _attn_kv_append_fp8(_attn_k.view(), _attn_v.view(),
                                          _kv_cache.k_word_view(slot), _kv_cache.v_word_view(slot),
                                          ksc, vsc, _dummy_u.view(),
                                          rows, kvh, hd, pos0, _max_context)
                    .dispatch(d::qwen_dispatch_threads(*append_plan.value));
          }
 } else if (_kv_packed_append) {
     // Packed BF16: one thread per 32-bit word, both codes in the same thread.
     const auto append_plan = d::plan_qwen_kv_append_packed_dispatch(rows, kvh, hd, _kv_storage);
     if (!append_plan.ok()) { return append_plan.status; }
     batch.annotate(lw.p_attn_prepare);
     if (device_state) {
         batch << _attn_kv_append_dev(_attn_k.view(), _attn_v.view(),
                                     _kv_cache.k_word_view(slot), _kv_cache.v_word_view(slot),
                                     _dev_state.view(), rows, kvh, hd, pos0, _max_context)
               .dispatch(d::qwen_dispatch_threads(*append_plan.value));
     } else {
         batch << _attn_kv_append(_attn_k.view(), _attn_v.view(),
                                 _kv_cache.k_word_view(slot), _kv_cache.v_word_view(slot),
                                 _dummy_u.view(), rows, kvh, hd, pos0, _max_context)
               .dispatch(d::qwen_dispatch_threads(*append_plan.value));
     }
 }
    const uint32_t kv_len = pos0 + rows;// cache now holds [0, kv_len)
    if (rows == 1u) {
        // Decode: one query at absolute position pos0 == kv_len-1.
        if (auto s = d::validate_qwen_attn_decode_inputs(qh, kvh, hd, kv_len, _max_context); !s.ok()) { return s; }
        auto plan = d::plan_qwen_attn_decode_dispatch(qh);
        if (!plan.ok()) { return plan.status; }
          // Split-KV decode attention with GQA dedup (opt_plan/03 section 4.2). When the
          // family is compiled, BOTH realizations are recorded for EVERY token and the
          // compile-time thresholds guard them, so exactly one runs and the captured
          // static graph stays argument-invariant while kv_len crosses the threshold.
          if (_attn_split) {
              auto split_plan = d::plan_qwen_attn_decode_split_dispatch(_attn_split_cfg);
              if (!split_plan.ok()) { return split_plan.status; }
              auto combine_plan = d::plan_qwen_attn_decode_combine_dispatch(_attn_split_cfg);
              if (!combine_plan.ok()) { return combine_plan.status; }
              if (auto s = d::validate_qwen_attn_decode_split_inputs(kv_len, _max_context, _attn_split_cfg); !s.ok()) { return s; }
              batch.annotate(lw.p_attn_core);
              if (device_state) {
                  batch << _attn_decode_dev(_attn_q.view(), _attn_gate.view(), kc, vc, ksc, vsc, _attn_out.view(),
                      _dev_state.view(), qh, kvh, hd, _cfg.attn_scale(), _max_context)
                      .dispatch(d::qwen_dispatch_threads(*plan.value));
              } else {
                  batch << _attn_decode(_attn_q.view(), _attn_gate.view(), kc, vc, ksc, vsc, _attn_out.view(),
                      qh, kvh, hd, _cfg.attn_scale(), kv_len, _max_context)
                      .dispatch(d::qwen_dispatch_threads(*plan.value));
              }
              batch.annotate(lw.p_attn_split);
              if (device_state) {
                  batch << _attn_decode_split_dev(_attn_q.view(), kc, vc, ksc, vsc, _attn_partial_acc.view(),
                      _attn_partial_ml.view(), _dev_state.view(), _cfg.attn_scale(), _max_context)
                      .dispatch(d::qwen_dispatch_threads(*split_plan.value));
              } else {
                  batch << _attn_decode_split(_attn_q.view(), kc, vc, ksc, vsc, _attn_partial_acc.view(),
                      _attn_partial_ml.view(), kv_len, _cfg.attn_scale(), _max_context)
                      .dispatch(d::qwen_dispatch_threads(*split_plan.value));
              }
              batch.annotate(lw.p_attn_combine);
              if (device_state) {
                  batch << _attn_decode_combine_dev(_attn_partial_acc.view(), _attn_partial_ml.view(),
                      _attn_gate.view(), _attn_out.view(), _dev_state.view(), _max_context)
                      .dispatch(d::qwen_dispatch_threads(*combine_plan.value));
              } else {
                  batch << _attn_decode_combine(_attn_partial_acc.view(), _attn_partial_ml.view(),
                      _attn_gate.view(), _attn_out.view(), kv_len, _max_context)
                      .dispatch(d::qwen_dispatch_threads(*combine_plan.value));
              }
          } else {
            batch.annotate(lw.p_attn_core);
            if (device_state) {
                batch << _attn_decode_dev(_attn_q.view(), _attn_gate.view(), kc, vc, ksc, vsc, _attn_out.view(),
                                      _dev_state.view(), qh, kvh, hd, _cfg.attn_scale(), _max_context)
                          .dispatch(d::qwen_dispatch_threads(*plan.value));
            } else {
                batch << _attn_decode(_attn_q.view(), _attn_gate.view(), kc, vc, ksc, vsc, _attn_out.view(),
                                  qh, kvh, hd, _cfg.attn_scale(), kv_len, _max_context)
                          .dispatch(d::qwen_dispatch_threads(*plan.value));
            }
          }
} else {
      // CUDA tensor-core FlashAttention-2 path first (opt_plan/09 N1b): the same
      // buffers, causal window, cache layout and sigmoid gate as the DSL tiled
      // kernel below, dispatched as a native command. A nullptr command is a real
      // error, not a fallback: create() proved the geometry and every input is
      // re-validated here with the DSL kernels' own contract.
      if (_native_attn.valid() && d::qwen_native_attn_prefers_rows(rows)) {
          if (auto s = d::validate_qwen_native_attn_inputs(rows, pos0, kv_len, _max_context,
                                                          _native_attn.config()); !s.ok()) {
              return s;
          }
          luisa::string native_error;
          auto command = _native_attn.build_dispatch(
              _attn_q.view(), _attn_gate.view(), kc, vc, _attn_out.view(),
              rows, _cfg.attn_scale(), pos0, kv_len, _max_context, &native_error);
          if (command == nullptr) {
              return err(StatusCode::internal_error,
                         "native tensor attention dispatch failed", std::move(native_error));
          }
          batch.annotate(lw.p_attn_core);
          batch << std::move(command);
          _native_attn_launches++;
      } else if (_attn_tiled) {
            // Tiled model path: identical buffers, causal window, cache layout and gate.
            if (auto s = d::validate_qwen35_attn_prefill_tiled_inputs(rows, pos0, kv_len, _max_context,
                                                             _attn_tiled_cfg); !s.ok()) {
                    return s;
            }
            auto plan = d::plan_qwen35_attn_prefill_tiled_dispatch(rows, _attn_tiled_cfg);
            if (!plan.ok()) { return plan.status; }
            batch.annotate(lw.p_attn_core);
            batch << _attn_prefill_tiled(_attn_q.view(), _attn_gate.view(), kc, vc, ksc, vsc, _attn_out.view(),
                                            rows, _cfg.attn_scale(), pos0, kv_len, _max_context)
                              .dispatch(d::qwen_dispatch_threads(*plan.value));
        } else {
            if (auto s = d::validate_qwen_attn_prefill_inputs(rows, qh, kvh, hd, pos0, kv_len, _max_context); !s.ok()) { return s; }
            auto plan = d::plan_qwen_attn_prefill_dispatch(rows, qh);
            if (!plan.ok()) { return plan.status; }
            batch.annotate(lw.p_attn_core);
            batch << _attn_prefill(_attn_q.view(), _attn_gate.view(), kc, vc, ksc, vsc, _attn_out.view(),
            rows, qh, kvh, hd, _cfg.attn_scale(), pos0, kv_len, _max_context)
            .dispatch(d::qwen_dispatch_threads(*plan.value));
        }
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
                                              uint32_t rows, uint32_t pos0, bool device_state) {
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
    // OFFLINE merged projection (opt_plan/03 section 3 + D5): at rows == 1 the whole
    // group is ONE launch over the merged [hidden, 12352] matrix [QKV | Z | alpha | beta].
    // The QKV and Z warps store their scalar straight into _proj_a / _proj_b, and the 64
    // gate warps run the SAME dot and the SAME epilogue as
    // make_qwen_gdn_decode_gates_kernel (g = ssm_a*softplus(a + dt_bias), beta = sigmoid(b)
    // into _gdn_g / _gdn_beta), so this launch REPLACES the two GEMVs AND the fused gate
    // launch without creating a 12352-wide intermediate. The per-output accumulation order
    // is the unmerged one, so decode logits stay bit-identical. `lw.merged_gdn` is a
    // per-layer build-time fact: the recorded body stays argument-invariant for the static
    // CUDA graph, and prefill (rows > 1) keeps the ordinary per-projection path.
    const bool merged_proj = rows == 1u && _merged_decode && lw.merged_gdn != nullptr;
    if (merged_proj) {
        batch.annotate(lw.p_merged_proj);
        auto planned = d::plan_qwen35_merged_gdn_gemv_dispatch(
            lw.merged_qkv_rows, lw.merged_z_rows, lw.merged_gate_heads, _cfg.hidden,
            _format, _decode_group);
        if (!planned.ok()) { return planned.status; }
        const auto &mp = *lw.merged_gdn;
        batch << _gemv_merged_gdn(_normed.view(), mp.packed.view(),
                                  mp.scales ? mp.scales.view() : _dummy_f.view(),
                                  _proj_a.view(), _proj_b.view(),
                                  lw.gdn_a->f32.view(), lw.gdn_dt_bias->f32.view(),
                                  _gdn_beta.view(), _gdn_g.view(),
                                  _cfg.hidden, _decode_group,
                                  lw.merged_qkv_rows, lw.merged_z_rows, lw.merged_gate_heads,
                                  _cfg.softplus_threshold)
                    .dispatch(d::qwen_dispatch_threads(*planned.value));
    } else {
        if (auto s = project(batch, _normed, *lw.gdn_qkv, _proj_a, rows, nullptr, "proj"); !s.ok()) { return s; }
        if (auto s = project(batch, _normed, *lw.gdn_gate, _proj_b, rows, nullptr, "proj"); !s.ok()) { return s; }
    }
    // z
    if (merged_proj) {
        // The alpha/beta GEMVs and the fused gate epilogue are inside the merged launch
        // above; the raw alpha/beta scratch buffers are deliberately not written.
    } else if (rows == 1u && _decode_specialized) {
        batch.annotate(lw.p_gates);
        auto plan = d::plan_qwen_gdn_decode_gates_dispatch(vh, _cfg.hidden, _format, _decode_group);
        if (!plan.ok()) { return plan.status; }
        const auto as = lw.gdn_alpha->scales ? scales_view_of(*lw.gdn_alpha) : _dummy_f.view();
        const auto bs = lw.gdn_beta->scales ? scales_view_of(*lw.gdn_beta) : _dummy_f.view();
        batch << _gdn_decode_gates_sh(_normed.view(), packed_view_of(*lw.gdn_alpha), as,
                                      packed_view_of(*lw.gdn_beta), bs,
                                      lw.gdn_a->f32.view(), lw.gdn_dt_bias->f32.view(),
                                      _gdn_beta.view(), _gdn_g.view(),
                                      _cfg.hidden, vh, _decode_group, _cfg.softplus_threshold)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    } else {
        if (auto s = project(batch, _normed, *lw.gdn_alpha, _gdn_alpha, rows, nullptr, "proj"); !s.ok()) { return s; }
        if (auto s = project(batch, _normed, *lw.gdn_beta, _proj_c, rows, nullptr, "proj"); !s.ok()) { return s; }
    }
    // Conv4 + SiLU with ring history over the packed QKV channels. The
    // device-state realization reads the absolute position from _dev_state
    // (opt_plan/09 V3): the 24 GDN layers of the static decode body would
    // otherwise all change their `pos0` argument every token.
    if (auto s = d::validate_qwen_gdn_conv_inputs(rows, qkv_n, _cfg.conv_kernel, pos0); !s.ok()) { return s; }
    batch.annotate(lw.p_conv);
 // Fused decode realization (opt_plan/09 N3.3): ONE launch replaces the conv4 +
 // SiLU + ring update AND the QK L2 norm, so the Q/K half of the conv output
 // never round-trips through global memory and 24 launches/token disappear. The
 // choice is a host-side engine-build decision (`_gdn_decode_fused`), so the
 // recorded body is argument-invariant either way and the static graph is
 // unaffected. With the flag off, the reference pair below runs unchanged.
 const bool fused_decode = rows == 1u && _decode_specialized && _gdn_decode_fused;
 if (fused_decode) {
     auto plan = d::plan_qwen35_gdn_decode_fused_dispatch(qkh, hd, v_n);
     if (!plan.ok()) { return plan.status; }
     if (device_state) {
         batch << _gdn_decode_fused_dev(_proj_a.view(), lw.gdn_conv1d->f32.view(), ring.view(),
                                        _gdn_q.view(), _gdn_k.view(), _gdn_conv.view(),
                                        _dev_state.view(), qkh, hd, v_n, _cfg.conv_kernel,
                                        _cfg.l2_eps, _cfg.gdn_q_scale())
                       .dispatch(d::qwen_dispatch_threads(*plan.value));
     } else {
         batch << _gdn_decode_fused_sh(_proj_a.view(), lw.gdn_conv1d->f32.view(), ring.view(),
                                       _gdn_q.view(), _gdn_k.view(), _gdn_conv.view(),
                                       qkh, hd, v_n, _cfg.conv_kernel,
                                       _cfg.l2_eps, _cfg.gdn_q_scale(), pos0)
                       .dispatch(d::qwen_dispatch_threads(*plan.value));
     }
 } else {
     {
         auto plan = d::plan_qwen_gdn_conv_dispatch(qkv_n);
         if (!plan.ok()) { return plan.status; }
         if (device_state) {
             batch << _gdn_conv_dev(_proj_a.view(), lw.gdn_conv1d->f32.view(), ring.view(), _gdn_conv.view(),
                                    _dev_state.view(), rows, qkv_n, _cfg.conv_kernel)
                       .dispatch(d::qwen_dispatch_threads(*plan.value));
         } else {
             batch << _gdn_conv_sh(_proj_a.view(), lw.gdn_conv1d->f32.view(), ring.view(), _gdn_conv.view(),
                                   rows, qkv_n, _cfg.conv_kernel, pos0)
                       .dispatch(d::qwen_dispatch_threads(*plan.value));
         }
     }
 }
 if (rows == 1u && _decode_specialized && _decode_packed) {
     // 128-bit realization of the same T=1 alpha/beta fusion (opt_plan/09 N2).
     batch.annotate(lw.p_gates);
     const auto elems = d::qwen_packed_vec_elements(_format);
     const auto vecs = static_cast<uint32_t>(static_cast<uint64_t>(_cfg.hidden) / elems);
     auto plan = d::plan_qwen_gdn_decode_gates_packed_dispatch(
         vh, _cfg.hidden, _format, _decode_group);
     if (!plan.ok()) { return plan.status; }
     const auto as = lw.gdn_alpha->scales ? scales_view_of(*lw.gdn_alpha) : _dummy_f.view();
     const auto bs = lw.gdn_beta->scales ? scales_view_of(*lw.gdn_beta) : _dummy_f.view();
     batch << _gdn_decode_gates_packed(_normed.view().as<float4>(),
                                      packed_view_of(*lw.gdn_alpha).as<uint4>(), as,
                                      packed_view_of(*lw.gdn_beta).as<uint4>(), bs,
                                      lw.gdn_a->f32.view(), lw.gdn_dt_bias->f32.view(),
                                      _gdn_beta.view(), _gdn_g.view(),
                                      _cfg.hidden, vh, _decode_group, _cfg.softplus_threshold, vecs)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
 } else if (rows != 1u || !_decode_specialized) {
// Reference gates: beta = sigmoid(b), g = ssm_a*softplus(a+dt_bias).
     batch.annotate(lw.p_gates);
     auto plan = d::plan_qwen_gdn_gates_dispatch(rows, vh);
     if (!plan.ok()) { return plan.status; }
     batch << _gdn_gates_sh(_gdn_alpha.view(), _proj_c.view(), lw.gdn_a->f32.view(), lw.gdn_dt_bias->f32.view(),
                            _gdn_beta.view(), _gdn_g.view(), rows, vh, _cfg.softplus_threshold)
               .dispatch(d::qwen_dispatch_threads(*plan.value));
 }
 if (!fused_decode) {
     // QK L2 norm + Q scale (reads conv output, writes _gdn_q/_gdn_k).
     batch.annotate(lw.p_qk_norm);
     {
         auto plan = d::plan_qwen_gdn_qk_norm_dispatch(rows, qkh);
         if (!plan.ok()) { return plan.status; }
         batch << _gdn_qk_norm_sh(_gdn_conv.view(), _gdn_q.view(), _gdn_k.view(),
                                  rows, qkh, hd, v_n, _cfg.l2_eps, _cfg.gdn_q_scale(), rows * qkh)
                   .dispatch(d::qwen_dispatch_threads(*plan.value));
     }
 }

 // Prefill defaults to the WY representation + UT transform. Decode and
    // the explicit differential mode retain the exact per-token recurrence.
    // FUSED one-CTA recurrence + gated output RMSNorm (opt_plan/03 section 5).
    // The ONE-CTA recurrence owns a whole head per block, so the gated RMSNorm
    // reduces inside the SAME launch: no _gdn_out round trip and one launch per
    // GDN layer instead of two. The choice is a host-side engine-build decision
    // (_gdn_recurrent_norm), so the recorded body is argument-invariant and the
    // static decode graph is unaffected; with the flag off the schedule below is
    // exactly what it is today.
    const bool fused_recurrent_norm = rows == 1u && _gdn_recurrent_norm;
    if (fused_recurrent_norm) {
        if (auto s = d::validate_qwen35_gdn_recurrent_norm_inputs(rows, vh, qkh, hd); !s.ok()) { return s; }
        if (d::qwen_gdn_recurrent_norm_shared_bytes(hd) > 32ull * 1024ull) {
            return err(StatusCode::unsupported, "fused GDN recurrent+norm shared-memory budget exceeded");
        }
        batch.annotate(lw.p_recurrent_norm);
        auto plan = d::plan_qwen35_gdn_recurrent_norm_dispatch(vh);
        if (!plan.ok()) { return plan.status; }
        batch << _gdn_recurrent_norm_sh(_gdn_q.view(), _gdn_k.view(), _gdn_conv.view(),
                                        _gdn_beta.view(), _gdn_g.view(), state.view(),
                                        _proj_b.view(), lw.gdn_norm->f32.view(),
                                        _gdn_normed.view(),
                                        rows, vh, hd, qkh, v_offset, _cfg.norm_eps)
                     .dispatch(d::qwen_dispatch_threads(*plan.value));
    } else if (rows == 1u && _decode_specialized) {
        if (auto s = d::validate_qwen35_gdn_decode4_inputs(rows, vh, qkh, hd); !s.ok()) { return s; }
        batch.annotate(lw.p_recurrent);
        auto plan = d::plan_qwen35_gdn_decode4_dispatch(vh);
        if (!plan.ok()) { return plan.status; }
        batch << _gdn_decode4_sh(_gdn_q.view(), _gdn_k.view(), _gdn_conv.view(),
                                 _gdn_beta.view(), _gdn_g.view(), state.view(), _gdn_out.view(),
                                 rows, vh, hd, qkh, v_offset)
                  .dispatch(d::qwen_dispatch_threads(*plan.value));
    } else if (rows > 1u && _gdn_wy_ut_active) {
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
      // Skipped ONLY when the fused one-CTA launch above already produced
      // _gdn_normed; every other schedule keeps this launch unchanged.
      if (!fused_recurrent_norm) {
        // Gated output RMSNorm (direct weight, norm before SiLU(z) gate). z=_proj_b.
        batch.annotate(lw.p_gated_norm);
        {
            auto plan = d::plan_qwen_gdn_gated_norm_dispatch(rows, vh);
            if (!plan.ok()) { return plan.status; }
            batch << _gdn_gated_norm_sh(_gdn_out.view(), _proj_b.view(), lw.gdn_norm->f32.view(), _gdn_normed.view(),
                                  rows, vh, hd, _cfg.norm_eps)
                      .dispatch(d::qwen_dispatch_threads(*plan.value));
        }
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
    if (rows == 1u && _decode_specialized && _decode_packed) {
        // 128-bit realization of the same fusion (opt_plan/09 N2).
        batch.annotate(lw.p_swiglu);
        const auto elems = d::qwen_packed_vec_elements(_format);
        const auto vecs = static_cast<uint32_t>(static_cast<uint64_t>(hidden) / elems);
        auto plan = d::plan_qwen_ffn_decode_packed_dispatch(ffn, hidden, _format, _decode_group);
        if (!plan.ok()) { return plan.status; }
        const auto gs = lw.ffn_gate->scales ? lw.ffn_gate->scales.view() : _dummy_f.view();
        const auto us = lw.ffn_up->scales ? lw.ffn_up->scales.view() : _dummy_f.view();
        batch << _ffn_decode_packed(_normed.view().as<float4>(),
                                    lw.ffn_gate->packed.view().as<uint4>(), gs,
                                    lw.ffn_up->packed.view().as<uint4>(), us, _ffn_act.view(),
                                    ffn, hidden, _decode_group, vecs)
                     .dispatch(d::qwen_dispatch_threads(*plan.value));
    } else if (rows == 1u && _decode_specialized) {
// Decode fusion: gate GEMV + up GEMV + SwiGLU in one launch. The two
        // packed weight streams share each activation load and stay in FP32.
        batch.annotate(lw.p_swiglu);
        auto plan = d::plan_qwen_ffn_decode_dispatch(ffn, hidden, _format, _decode_group);
        if (!plan.ok()) { return plan.status; }
        const auto gs = lw.ffn_gate->scales ? lw.ffn_gate->scales.view() : _dummy_f.view();
        const auto us = lw.ffn_up->scales ? lw.ffn_up->scales.view() : _dummy_f.view();
        batch << _ffn_decode(_normed.view(), lw.ffn_gate->packed.view(), gs,
                             lw.ffn_up->packed.view(), us, _ffn_act.view(),
                             ffn, hidden, _decode_group)
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
                                                  uint32_t rows, uint32_t pos0, bool device_state) {
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
                         rows, hidden, _cfg.vocab, _decode_group, plan.value->padded_cols)
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
            if (auto s = full_attention_block(batch, l, rows, pos0, device_state); !s.ok()) { return s; }
        } else {
            if (auto s = gdn_block(batch, l, rows, pos0, device_state); !s.ok()) { return s; }
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
    // Plan option (c) keeps the LM head at 8 bits (the most quality-sensitive
    // matrix): when the session carries the FP8 partner it is read here through
    // its own GEMV pair, and only the explicit in-kernel A/B arm falls back to the
    // 4-bit table.
    if (_lm_head == nullptr || _lm_head == _token_embd) {
        if (auto s = project(batch, _last_normed, *_token_embd, _logits, 1u, nullptr, "lm_head"); !s.ok()) { return s; }
        return Status::success();
    }
    const uint32_t k = hidden;
    const uint32_t n = _cfg.vocab;
    const uint32_t group = _lm_head->group != 0u ? _lm_head->group : kFp8GroupSize;
    const auto scales = _lm_head->scales ? _lm_head->scales.view() : _dummy_f.view();
    batch.annotate("lm_head");
    luisa::string name;
    if (batch.instrumented()) {
        name = proj_name("lm_head", *_lm_head);
        batch.annotate(name);
    }
 // Same calibration hook as project(): this arm bypasses it. In a calibration
 // session `_lm_head` IS the tied token_embd pointer (no FP8 partner exists), so it
 // is a registered projection; a partner pointer would fail closed.
 if (auto s = record_calibration(batch, _last_normed, *_lm_head, 1u, k); !s.ok()) { return s; }
    auto plan = _decode_specialized
                    ? d::plan_qwen35_decode_gemv_dispatch(n, k, d::QwenWeightFormat::fp8_e4m3, group)
                    : d::plan_qwen_gemv_dispatch(1u, n, k);
    if (!plan.ok()) { return plan.status; }
    if (plan.value->empty()) { return Status::success(); }
    batch << _gemv_lm(_last_normed.view(), _lm_head->packed.view(), scales, _dummy_f.view(),
                      _logits.view(), 1u, n, k, group)
                 .dispatch(d::qwen_dispatch_threads(*plan.value));
    return Status::success();
}

// ---------------------------------------------------------------------------
// prefill
// ---------------------------------------------------------------------------
[[nodiscard]] uint32_t QwenDslEngine::prefill_chunk_for(uint64_t rows) const noexcept {
    if (!_prefill_chunk_adaptive) { return _prefill_chunk; }
    const auto plan = d::qwen_plan_prefill_chunks(rows, _prefill_chunk_max,
                                                  _prefill_chunk_backend_limit);
    // Never exceed the workspace row capacity sized at create().
    return std::max(1u, std::min(plan.chunk, _prefill_chunk));
}

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
    // Chunking for THIS prompt (opt_plan/02 section 4). A fixed chunk keeps the
    // historical single size; the adaptive rule picks per prompt and reports the
    // selection in QwenEngineInfo so a benchmark log can attribute TTFT.

    // Per-prompt GDN prefill selection (opt_plan/09 V4 follow-up). Decided ONCE
    // for the whole prompt -- every chunk of a prompt must stay on one numerical
    // path -- from the MEASURED crossover, never from the chunk size:
    //   128..8192 tokens: exact per-token recurrence, 15-17 % faster TTFT
    //   above 8192:      WY/UT chunk-64 (unmeasured region keeps the old default)
    _gdn_wy_ut_active = _gdn_auto ? total > _gdn_auto_threshold : _gdn_wy_ut;
    if (_gdn_auto) {
        LUISA_INFO("qwen dsl: GDN prefill for {} tokens -> {}", total,
                   _gdn_wy_ut_active ? "WY/UT chunk-64" : "exact per-token recurrence");
    }
    const uint32_t chunk = prefill_chunk_for(total);
    _prefill_chunk_selected = chunk;
    _prefill_chunks = (total + chunk - 1u) / chunk;
    // ONE command batch for ALL chunks: normal mode submits once, then the
    // logits download synchronizes. Each chunk stages a DISTINCT host span.
    _staged = 0;
    auto batch = QwenCommandBatch{stream, _benchmark ? &_report : nullptr,
                                  QwenBenchmarkPhase::prefill, 0u, total};
    uint32_t done = 0;
    while (done < total) {
        const uint32_t rows = std::min(chunk, total - done);
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
    // Download logits (single synchronize at end) through the SELECTED realization:
    // pinned staging when PinnedMemoryExt is available and selected, otherwise the
    // same pageable copy as before.
    batch.layer(UINT32_MAX).set_context(QwenBenchmarkPhase::prefill, total - 1u, 1u);
    if (auto s = download_logits(stream, batch, logits_out); !s.ok()) {
        _has_session = false;
        return s;
    }
    _position = total;
    _has_session = true;
    _greedy_next_valid = false;
    // The prefill advanced the position on the host-scalar path, so the device
    // decode state no longer describes this session: force an upload before the
    // first static-graph launch (the captured graph itself stays valid).
    _dev_state_synced = false;
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
    // ONE command list per decode step; no synchronize until the download (which
    // uses the pinned staging when it is selected).
    batch.flush(false);
    if (auto s = download_logits(stream, batch, logits_out); !s.ok()) {
        _has_session = false;
        return s;
    }
    _position = pos0 + 1u;
    _greedy_next_valid = false;
    // A host-logits decode advanced _position without touching the device decode
    // state; the next device-greedy step must re-upload it.
    _dev_state_synced = false;
    return Status::success();
}

// ---------------------------------------------------------------------------
// Static CUDA decode graph (opt_plan/03 section 2.3, milestone M-c).
//
// The reference decode step is a command list whose kernel ARGUMENTS change every
// token (absolute position, KV length, greedy history slot), so a graph built
// from it can only be replayed after a whole-graph update -- which costs more
// than the launches it saves. The body recorded below is argument-invariant: the
// four consumers of those scalars (attention prepare, decode attention, the 24
// GDN conv layers and the argmax finish) read them from _dev_state, and every
// other argument is a device buffer or a session-constant scalar. One capture per
// session is therefore enough; later tokens are a bare cuGraphLaunch.
//
// Deliberately NOT captured, in stream order before the launch:
//   * the host token upload (copy_from + device copy into _ids),
//   * the device-state upload (session start / prefill / caller slot drift),
//   * sparse KV page maps (ensure_capacity), which must never appear inside a
//     graph: growth is page-granular, keeps every virtual address stable and
//     therefore needs no graph update either.
// ---------------------------------------------------------------------------
[[nodiscard]] Status QwenDslEngine::upload_dev_state(Stream &stream, uint32_t position,
                                                     uint32_t kv_len, uint32_t history_slot,
                                                     uint32_t remaining) {
    if (!_dev_state || _host_state.size() < d::qwen_dec_state_count) {
        return err(StatusCode::internal_error, "device decode state buffer is not allocated");
    }
    _host_state[d::qwen_dec_state_position] = position;
    _host_state[d::qwen_dec_state_kv_len] = kv_len;
    _host_state[d::qwen_dec_state_history_slot] = history_slot;
    // [eos] is the conditional WHILE's stop flag: cleared on every upload so a
    // chunk starts from "not stopped" and the WHILE finish kernel owns the field.
    // The per-token kernels never read or write it.
    _host_state[d::qwen_dec_state_eos] = 0u;
    // [remaining] is the WHILE token budget of the NEXT launch (0 for the
    // per-token path, which never reads it).
    _host_state[d::qwen_dec_state_remaining] = remaining;
    auto batch = QwenCommandBatch{stream, _benchmark ? &_report : nullptr,
                                  QwenBenchmarkPhase::decode, position, 1u};
    batch.layer(UINT32_MAX).annotate("decode_state_upload");
    batch << _dev_state.view().copy_from(
        luisa::span<const uint32_t>{_host_state.data(), d::qwen_dec_state_count});
    // Commit immediately: copy_from borrows the host span until the encoder runs,
    // and _host_state is reused by the next upload. This is a plain pre-graph
    // stream command; nothing about the graph is touched.
    batch.flush(false);
    _dev_state_position = position;
    _dev_state_kv_len = kv_len;
    _dev_state_slot = history_slot;
    _dev_state_synced = true;
    return Status::success();
}

[[nodiscard]] Status QwenDslEngine::ensure_kv_for_decode(Stream &stream, uint32_t tokens) {
    if (tokens == 0u) { return Status::success(); }
    if (tokens > _max_context) { tokens = _max_context; }
    if (tokens <= _kv_cache.committed_tokens()) { return Status::success(); }
    if (!_kv_cache.valid()) {
        return err(StatusCode::internal_error, "decode KV growth on an invalid allocator");
    }
    // Round the request up to a whole page. Sparse residency is page-granular and
    // the allocator maps whole pages, so a page-granular watermark keeps this
    // call at the page boundaries instead of every token.
    uint32_t covered = tokens;
    const uint64_t page_bytes = _kv_cache.page_size_bytes();
    const uint64_t bytes_per_token =
        static_cast<uint64_t>(_kv_cache.values_per_token()) * sizeof(float);
    if (page_bytes != 0u && bytes_per_token != 0u) {
        const auto page_tokens = std::max<uint64_t>(1u, page_bytes / bytes_per_token);
        const uint64_t rounded =
            ((static_cast<uint64_t>(tokens) + page_tokens - 1u) / page_tokens) * page_tokens;
        covered = static_cast<uint32_t>(std::min<uint64_t>(rounded, _max_context));
    }
    // Stream-ordered map submission, OUTSIDE the graph (never captured). Growth
    // only adds previously unmapped pages and never moves the region bases, so
    // the captured nodes keep pointing at live memory.
    return _kv_cache.ensure_capacity(stream, covered);
}

[[nodiscard]] Status QwenDslEngine::record_greedy_step(Stream &stream, QwenCommandBatch &batch,
                                                       uint32_t pos0, uint32_t history_slot,
                                                       bool device_state, uint64_t while_handle) {
    if (auto s = forward_chunk(stream, batch, 1u, pos0, device_state); !s.ok()) { return s; }
    if (auto s = lm_head(batch, 1u); !s.ok()) { return s; }
    auto partial_plan = d::plan_qwen_greedy_argmax_partial_dispatch(_cfg.vocab);
    auto finish_plan = d::plan_qwen_greedy_argmax_finish_dispatch(_cfg.vocab);
    if (!partial_plan.ok()) { return partial_plan.status; }
    if (!finish_plan.ok()) { return finish_plan.status; }
    batch.layer(UINT32_MAX).annotate("greedy_argmax_partial");
    batch << _greedy_argmax_partial(_logits.view(), _greedy_values.view(),
                                    _greedy_indices.view(), _greedy_invalid.view(), _cfg.vocab)
                 .dispatch(d::qwen_dispatch_threads(*partial_plan.value));
    batch.layer(UINT32_MAX).annotate("greedy_argmax_finish");
    if (device_state && while_handle != 0ull) {
        // CONDITIONAL WHILE body terminal (opt_plan/09 N4.2): the same selection and
        // the same state advance as the per-token device-state kernel, plus the
        // device-computed continuation flag written into the enclosing WHILE node.
        // `while_handle` is the cudaGraphConditionalHandle created for the parent
        // graph, baked in here because the body is only ever populated after the
        // handle exists (build_body_graph), and it is a session constant, so the
        // body stays argument-invariant across launches.
        batch << _greedy_argmax_finish_while(_greedy_values.view(), _greedy_indices.view(),
                                            _greedy_invalid.view(), _ids.view(),
                                            _greedy_history.view(), _dev_state.view(),
                                            _cfg.vocab, _max_context, while_handle, _while_eos)
                     .dispatch(d::qwen_dispatch_threads(*finish_plan.value));
    } else if (device_state) {
        // Writes next_ids[0] (read by the next step's embedding), the history
        // slot from the device state, and advances position/kv_len/slot.
        batch << _greedy_argmax_finish_dev(_greedy_values.view(), _greedy_indices.view(),
                                          _greedy_invalid.view(), _ids.view(),
                                          _greedy_history.view(), _dev_state.view(),
                                          _cfg.vocab, _max_context)
                     .dispatch(d::qwen_dispatch_threads(*finish_plan.value));
    } else {
        batch << _greedy_argmax_finish(_greedy_values.view(), _greedy_indices.view(),
                                      _greedy_invalid.view(), _ids.view(),
                                      _greedy_history.view(), _cfg.vocab, history_slot)
                     .dispatch(d::qwen_dispatch_threads(*finish_plan.value));
    }
    return Status::success();
}

[[nodiscard]] Status QwenDslEngine::capture_static_decode_graph(Stream &stream, QwenCommandBatch &batch) {
    if (_cuda_graph_ext == nullptr) {
        return err(StatusCode::unsupported, "static CUDA decode graph requires CudaGraphExt");
    }
    if (_device_greedy_graph_exec.has_value()) {
        return err(StatusCode::internal_error, "static CUDA decode graph was captured twice");
    }
    auto list = batch.take_commands();
    if (list.empty()) { return invalid("static CUDA decode graph command list is empty"); }
    auto graph = _cuda_graph_ext->create_graph(std::move(list.commit()).command_list());
    if (graph.handle().handle == compute::CudaGraphExt::invalid_handle) {
        return err(StatusCode::unsupported, "CUDA graph capture failed for device greedy decode");
    }
    auto exec = _cuda_graph_ext->instantiate(graph.handle().handle);
    if (exec.handle().handle == compute::CudaGraphExt::invalid_handle) {
        return err(StatusCode::unsupported, "CUDA graph instantiation failed for device greedy decode");
    }
    _device_greedy_graph.emplace(std::move(graph));
    _device_greedy_graph_exec.emplace(std::move(exec));
    // Upload the template once: the body is static, so the driver can pre-resolve
    // its allocations before the first launch. Failure is not fatal (CUDA treats
    // it as an optimization), so the result is deliberately ignored.
    _cuda_graph_ext->upload(_device_greedy_graph_exec->handle().handle, stream.handle());
    _cuda_graph_ext->launch(_device_greedy_graph_exec->handle().handle, stream.handle());
    _cuda_graph_steps++;
    return Status::success();
}

[[nodiscard]] Status QwenDslEngine::launch_static_decode_graph(Stream &stream) {
    if (_cuda_graph_ext == nullptr || !_device_greedy_graph_exec.has_value()) {
        return err(StatusCode::internal_error, "static CUDA decode graph is not instantiated");
    }
    // Zero per-token graph maintenance: no update(), no node update, no rebuild.
    _cuda_graph_ext->launch(_device_greedy_graph_exec->handle().handle, stream.handle());
    _cuda_graph_steps++;
    return Status::success();
}

// ---------------------------------------------------------------------------
// Conditional WHILE decode chunk (opt_plan/09 N4.2)
// ---------------------------------------------------------------------------
// The parent graph owns exactly one non-conditional node -- a 1-thread zero-fill
// of the unused dummy buffer -- because CUDA refuses a graph with no nodes, and
// every alternative candidate (an upload of a staging span, a copy of a live buffer)
// would either perturb session state or read host memory the graph does not own.
// The WHILE container is wired after it, its body is populated with the already
// recorded argument-invariant step, and the condition is assigned a launch default
// of 1 (do-while): a chunk always makes progress, and the body's finish kernel
// clears the condition when the device decides to stop.
[[nodiscard]] Status QwenDslEngine::capture_while_decode_graph(Stream &stream, uint32_t pos0,
                                                              uint32_t history_slot) {
    if (_cuda_graph_ext == nullptr) {
        return err(StatusCode::unsupported, "conditional WHILE decode requires CudaGraphExt");
    }
    if (_while_graph_exec.has_value()) {
        return err(StatusCode::internal_error, "conditional WHILE decode graph was captured twice");
    }
    // (a) Parent graph. CUDA refuses a node-less graph, so it starts from a 1-thread
    // zero-fill of the unused dummy buffer: the cheapest node whose arguments are
    // session constants and whose execution cannot perturb any live buffer (the dummy
    // is never read by a kernel; it only fills unused argument slots, and uninitialized
    // device memory already had no contract).
    auto parent = QwenCommandBatch{stream, nullptr, QwenBenchmarkPhase::decode, 0u, 0u};
    parent.layer(UINT32_MAX).annotate("while_graph_seed");
    {
        auto plan = d::plan_qwen_threads(1u);
        if (!plan.ok()) { return plan.status; }
        parent << _zero_fill(_dummy_f.view(), 1u).dispatch(d::qwen_dispatch_threads(*plan.value));
    }
    auto parent_list = parent.take_commands();
    if (parent_list.empty()) { return invalid("conditional WHILE parent graph is empty"); }
    auto graph = _cuda_graph_ext->create_graph(std::move(parent_list.commit()).command_list());
    if (graph.handle().handle == compute::CudaGraphExt::invalid_handle) {
        return err(StatusCode::unsupported, "CUDA graph capture failed for the WHILE parent");
    }
    auto graph_handle = graph.handle().handle;
    // Assigned default 1 (do-while): a chunk always executes at least one token, and
    // the value is re-applied on EVERY launch, so a replay is reproducible.
    auto handle = _cuda_graph_ext->create_conditional_handle(graph_handle, 1u, true);
    if (handle == 0u) {
        return err(StatusCode::unsupported, "create_conditional_handle failed");
    }
    luisa::vector<compute::CudaGraphExt::GraphHandle> bodies(
        1u, compute::CudaGraphExt::invalid_handle);
    size_t conditional_index = ~size_t{0u};
    if (!_cuda_graph_ext->add_conditional_node(graph_handle,
                                              compute::CudaGraphExt::ConditionalType::WHILE,
                                              handle, 1u, luisa::span{bodies},
                                              &conditional_index)) {
        return err(StatusCode::unsupported, "add_conditional_node(WHILE) failed");
    }
    if (bodies[0] == compute::CudaGraphExt::invalid_handle) {
        return err(StatusCode::unsupported, "WHILE node returned no body graph");
    }
    // (b) Body: the argument-invariant device-state step, recorded NOW because the
    // finish kernel needs the conditional handle value and that only exists after the
    // parent graph was created. Every other argument is a device buffer or a session
    // constant, so the body stays valid for every chunk of the session.
    auto body = QwenCommandBatch{stream, nullptr, QwenBenchmarkPhase::decode, pos0, 1u};
    if (auto s = record_greedy_step(stream, body, pos0, history_slot, true, handle); !s.ok()) {
        return s;
    }
    auto body_list = body.take_commands();
    if (body_list.empty()) { return invalid("conditional WHILE decode body is empty"); }
    if (!_cuda_graph_ext->add_dependency(graph_handle, 0u, conditional_index)) {
        return err(StatusCode::unsupported, "WHILE dependency wiring failed");
    }
    if (!_cuda_graph_ext->build_body_graph(bodies[0],
                                          std::move(body_list.commit()).command_list())) {
        return err(StatusCode::unsupported, "build_body_graph refused the decode step");
    }
    // (c) Instantiate and run the first chunk through the same bare-launch path every
    // later chunk takes (so no second code path can drift from it).
    auto exec = _cuda_graph_ext->instantiate(graph_handle);
    if (exec.handle().handle == compute::CudaGraphExt::invalid_handle) {
        return err(StatusCode::unsupported, "WHILE graph instantiation failed");
    }
    _while_graph.emplace(std::move(graph));
    _while_graph_exec.emplace(std::move(exec));
    // Pre-resolve the template's allocations once: CUDA treats the upload as an
    // optimization, so a failure is deliberately ignored.
    _cuda_graph_ext->upload(_while_graph_exec->handle().handle, stream.handle());
    _cuda_graph_ext->launch(_while_graph_exec->handle().handle, stream.handle());
    _while_launches++;
    return Status::success();
}

[[nodiscard]] Status QwenDslEngine::launch_while_decode_graph(Stream &stream) {
    if (_cuda_graph_ext == nullptr || !_while_graph_exec.has_value()) {
        return err(StatusCode::internal_error, "conditional WHILE decode graph is not instantiated");
    }
    _cuda_graph_ext->launch(_while_graph_exec->handle().handle, stream.handle());
    _while_launches++;
    return Status::success();
}

[[nodiscard]] Status QwenDslEngine::read_back_dev_state(Stream &stream,
                                                        DevStateReadback &state) {
    if (!_dev_state || _host_state.size() < d::qwen_dec_state_count) {
        return err(StatusCode::internal_error, "device decode state buffer is not allocated");
    }
    if (_pinned_selection.selected && _pinned.valid() &&
        _pinned.capacity() >= d::qwen_dec_state_count) {
        // Same pinned staging the token/logits downloads use; 5 words, but the code
        // path (ordered copy -> synchronize -> host memcpy) is identical, so the
        // readback cannot diverge from the download contract. A ring smaller than the
        // state (a <= 4-token context) keeps the pageable readback instead of refusing.
        auto span = luisa::span<uint32_t>{_host_state.data(), d::qwen_dec_state_count};
        if (auto s = _pinned.download_tokens(stream, _dev_state.view(), 0u, span,
                                             _pinned_stats);
            !s.ok()) {
            return s;
        }
    } else {
        auto batch = QwenCommandBatch{stream, _benchmark ? &_report : nullptr,
                                      QwenBenchmarkPhase::decode, _position, 1u};
        batch.layer(UINT32_MAX).annotate("decode_state_readback");
        batch << _dev_state.view().copy_to(
            luisa::span<uint32_t>{_host_state.data(), d::qwen_dec_state_count});
        batch.flush(true);
    }
    state.position = _host_state[d::qwen_dec_state_position];
    state.kv_len = _host_state[d::qwen_dec_state_kv_len];
    state.slot = _host_state[d::qwen_dec_state_history_slot];
    state.eos = _host_state[d::qwen_dec_state_eos];
    state.remaining = _host_state[d::qwen_dec_state_remaining];
    return Status::success();
}

// One logits download through the SELECTED realization (QwenEngineInfo::pinned_*
// reports which one and why); both branches resize the caller's vector first so an
// early failure cannot leave it half-filled.
[[nodiscard]] Status QwenDslEngine::download_logits(Stream &stream, QwenCommandBatch &batch,
                                                    luisa::vector<float> &logits_out) {
    logits_out.resize(_cfg.vocab);
    if (_pinned_selection.selected && _pinned.valid()) {
        // Flush the recorded step first: the pinned download issues its own
        // stream-ordered copy, so everything already recorded must be committed
        // before it to keep stream order.
        batch.flush(false);
        if (auto s = _pinned.download_floats(stream, _logits.view(),
                                             luisa::span<float>{logits_out}, _pinned_stats);
            !s.ok()) {
            return s;
        }
        return Status::success();
    }
    batch.layer(UINT32_MAX).annotate("logits_download");
    batch << _logits.view().copy_to(luisa::span<float>{logits_out});
    batch.flush(true);
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
    const bool static_graph = _cuda_graph_ext != nullptr;
    if (static_graph) {
        if (_cuda_graph_capture_failed) {
            return err(StatusCode::unsupported,
                       "static CUDA decode graph is unavailable after a capture/instantiate failure");
        }
        // (1) Host token upload: a plain stream command BEFORE the launch, never
        // part of the graph. Keeping it outside also avoids releasing replaced
        // upload staging while an earlier launch is still in flight.
        if (input_token != nullptr) {
            auto upload = QwenCommandBatch{stream, nullptr,
                                           QwenBenchmarkPhase::decode, pos0, 1u};
            if (auto s = stage_ids(upload, luisa::span<const uint32_t>{input_token, 1}); !s.ok()) {
                _has_session = false;
                return s;
            }
            upload.flush(false);
        }
        // (2) Device decode state. `kv_len` is the length the decode attention of
        // THIS step may read: the same step's prepare appends [pos0], so the
        // invariant is kv_len == position + 1. Uploaded only when the host mirror
        // disagrees (session start, prefill/chunk boundary, caller slot drift,
        // or a host-logits decode() that advanced _position on the scalar path).
        const uint32_t kv_len = pos0 + 1u;
        if (!dev_state_matches(pos0, kv_len, history_slot)) {
            if (auto s = upload_dev_state(stream, pos0, kv_len, history_slot); !s.ok()) {
                _has_session = false;
                return s;
            }
        }
        // (3) Sparse KV growth stays outside the graph and out of the capture.
        if (auto s = ensure_kv_for_decode(stream, kv_len); !s.ok()) {
            _has_session = false;
            return s;
        }
        // (4) Capture once, then launch. The steady state records nothing at all:
        // that is where the ~438 host-side node records per token go away.
        if (!_device_greedy_graph_exec.has_value()) {
            auto batch = QwenCommandBatch{stream, nullptr,
                                          QwenBenchmarkPhase::decode, pos0, 1u};
            if (auto s = record_greedy_step(stream, batch, pos0, history_slot, true); !s.ok()) {
                _has_session = false;
                return s;
            }
            if (auto s = capture_static_decode_graph(stream, batch); !s.ok()) {
                // The list was consumed by the capture attempt: fail closed, never
                // submit a partial sequence. A later reset() may retry.
                _cuda_graph_capture_failed = true;
                _has_session = false;
                LUISA_WARNING("qwen dsl: static CUDA decode graph unavailable ({}); "
                              "device greedy decode fails closed until reset",
                    s.detail.empty() ? s.message : s.detail);
                return s;
            }
        } else if (auto s = launch_static_decode_graph(stream); !s.ok()) {
            _has_session = false;
            return s;
        }
        // Mirror the advance the finish kernel performed on the device.
        _dev_state_position = pos0 + 1u;
        _dev_state_kv_len = pos0 + 2u;
        _dev_state_slot = history_slot + 1u;
        _dev_state_synced = true;
    } else {
        auto batch = QwenCommandBatch{stream, _benchmark ? &_report : nullptr,
                                      QwenBenchmarkPhase::decode, pos0, 1u};
        if (input_token != nullptr) {
            if (auto s = stage_ids(batch, luisa::span<const uint32_t>{input_token, 1}); !s.ok()) {
                _has_session = false;
                return s;
            }
        }
        if (auto s = record_greedy_step(stream, batch, pos0, history_slot, false); !s.ok()) {
            _has_session = false;
            return s;
        }
        // Do not synchronize: the next in-flight step reads _ids produced above.
        // Stream order protects all reused activation/state buffers.
        batch.flush(false);
    }
    _position = pos0 + 1u;
    _greedy_next_valid = true;
    return Status::success();
}

// Chunked device-greedy decode (opt_plan/09 N4.2). ONE conditional WHILE graph
// launch when the chunked path is selected; otherwise the SAME tokens are produced
// by per-token launches (the documented fallback), so the caller's ids never depend
// on which path served the chunk.
[[nodiscard]] Status QwenDslEngine::decode_device_greedy_chunk(
    Stream &stream, const uint32_t *input_token, uint32_t history_slot,
    uint32_t count, QwenChunkResult &result) {
    result = QwenChunkResult{};
    result.requested = count;
    if (count == 0u) { return invalid("device greedy chunk size must be at least 1"); }
    if (!_has_session) { return invalid("device greedy chunk requires a prior prefill"); }
    if (history_slot >= _max_context || count > _max_context - history_slot) {
        return invalid("device greedy chunk range exceeds history");
    }
    // ONE fallback for every "no WHILE" reason (not selected, sticky capture failure,
    // a failure discovered below): exactly the per-token steps the caller would have
    // issued itself, so the ids are identical and only the launch count differs.
    // `result.fallback` keeps "the accelerator never engaged" visible.
    const auto per_token_fallback = [&]() -> Status {
        result.fallback = true;
        const uint32_t *next = input_token;
        for (uint32_t i = 0; i < count; i++) {
            if (auto s = decode_device_greedy(stream, next, history_slot + i); !s.ok()) {
                result.executed = i;
                return s;
            }
            next = nullptr;
        }
        result.executed = count;
        _while_fallbacks++;
        return Status::success();
    };
    if (!_while_selected || _while_capture_failed) { return per_token_fallback(); }
    if (_position >= _max_context) {
        return invalid("device greedy chunk would exceed max_context");
    }
    if (count > _max_context - _position) {
        return invalid("device greedy chunk exceeds the remaining context");
    }
    if (input_token == nullptr) {
        if (!_greedy_next_valid) {
            return invalid("device greedy chunk has no device-resident input token");
        }
    } else if (*input_token >= _cfg.vocab) {
        return invalid("device greedy chunk input token id out of vocabulary range");
    }
    const uint32_t pos0 = _position;
    if (_benchmark) { _report.clear(); }
    _staged = 0;
    // (1) Host token upload: a plain stream command BEFORE the launch, never part of
    // the graph (the same contract as the per-token static graph).
    if (input_token != nullptr) {
        auto upload = QwenCommandBatch{stream, nullptr, QwenBenchmarkPhase::decode, pos0, 1u};
        if (auto s = stage_ids(upload, luisa::span<const uint32_t>{input_token, 1}); !s.ok()) {
            _has_session = false;
            return s;
        }
        upload.flush(false);
    }
    // (2) Device decode state INCLUDING this launch's token budget. Uploaded per chunk
    // (the budget changes every launch) instead of per token.
    const uint32_t kv_len = pos0 + 1u;
    if (auto s = upload_dev_state(stream, pos0, kv_len, history_slot, count); !s.ok()) {
        _has_session = false;
        return s;
    }
    // (3) Sparse KV growth for the WHOLE chunk, outside the graph and at page
    // granularity. The last token of the chunk reads kv_len = pos0 + count.
    if (auto s = ensure_kv_for_decode(stream, pos0 + count); !s.ok()) {
        _has_session = false;
        return s;
    }
    // (4) Capture once, then bare launches.
    if (!_while_graph_exec.has_value()) {
        if (!_greedy_argmax_finish_while) {
            compute::ShaderOption option;
            option.enable_fast_math = false;
            _greedy_argmax_finish_while = _device.compile(
                d::make_qwen_greedy_argmax_finish_while_kernel(), option);
            if (!_greedy_argmax_finish_while || !_greedy_argmax_finish_while.compile_ok()) {
                _while_capture_failed = true;
                _while_reason = "conditional WHILE finish kernel failed to compile";
                LUISA_WARNING("qwen dsl: {}; the chunk falls back to per-token launches",
                              _while_reason.c_str());
                return per_token_fallback();
            }
            _shader_count++;
        }
        if (auto s = capture_while_decode_graph(stream, pos0, history_slot); !s.ok()) {
            _while_capture_failed = true;
            // The device state may hold a partially uploaded budget from this call;
            // force the next launch to re-upload it.
            _dev_state_synced = false;
            _while_reason = "conditional WHILE capture failed: " +
                            luisa::string{s.detail.empty() ? s.message : s.detail};
            LUISA_WARNING("qwen dsl: {}; the chunk falls back to per-token launches",
                          _while_reason.c_str());
            return per_token_fallback();
        }
    } else if (auto s = launch_while_decode_graph(stream); !s.ok()) {
        _has_session = false;
        return s;
    }
    // (5) Read the device state back: `executed` is measured, never assumed. The
    // readback synchronizes, so every token of the chunk is already in _greedy_history
    // before the caller downloads it.
    DevStateReadback state;
    if (auto s = read_back_dev_state(stream, state); !s.ok()) {
        _has_session = false;
        return s;
    }
    if (state.remaining > count) {
        _has_session = false;
        _greedy_next_valid = false;
        return err(StatusCode::internal_error,
                   "conditional WHILE chunk reported an impossible iteration budget");
    }
    const uint32_t executed = count - state.remaining;
    if (executed == 0u) {
        _has_session = false;
        _greedy_next_valid = false;
        return err(StatusCode::internal_error,
                   "conditional WHILE chunk executed no token (the do-while default was lost)");
    }
    if (state.position != pos0 + executed) {
        _has_session = false;
        _greedy_next_valid = false;
        return err(StatusCode::internal_error,
                   "conditional WHILE chunk advanced the device position inconsistently");
    }
    result.executed = executed;
    result.eos = state.eos != 0u;
    // Only three stops exist: EOS / an invalid selection (both set [eos]) or the
    // context limit, so an early exit without [eos] is the context limit.
    result.context_limit = executed < count && !result.eos;
    // Host accounting follows the DEVICE, so an early exit can never look like
    // unexecuted tokens ran.
    _position = state.position;
    _dev_state_position = state.position;
    _dev_state_kv_len = state.kv_len;
    _dev_state_slot = state.slot;
    _dev_state_synced = true;
    _greedy_next_valid = true;
    _while_tokens += executed;
    if (executed < count) { _while_early_exits++; }
    return Status::success();
}

[[nodiscard]] Status QwenDslEngine::download_device_greedy(
    Stream &stream, uint32_t first_slot, luisa::span<uint32_t> tokens_out) {
    if (tokens_out.empty()) { return Status::success(); }
    if (!_greedy_next_valid) { return invalid("device greedy download has no completed selection"); }
    if (first_slot >= _max_context || tokens_out.size() > _max_context - first_slot) {
        return invalid("device greedy download range exceeds history");
    }
    if (_pinned_selection.selected && _pinned.valid()) {
        // PINNED ring: an arbitrary caller range [first_slot, first_slot + size) is
        // downloaded in ring-sized slices -- stream-ordered device->device copy into
        // page-locked host memory, synchronize, host memcpy into the caller's span --
        // so a range larger than the ring is correct, and the ring is reusable.
        if (auto s = _pinned.download_tokens(stream, _greedy_history.view(), first_slot,
                                             tokens_out, _pinned_stats);
            !s.ok()) {
            return s;
        }
    } else {
        auto batch = QwenCommandBatch{stream, _benchmark ? &_report : nullptr,
                                      QwenBenchmarkPhase::decode, _position, 1u};
        batch.layer(UINT32_MAX).annotate("greedy_tokens_download");
        batch << _greedy_history.view(first_slot, tokens_out.size()).copy_to(tokens_out);
        batch.flush(true);
    }
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
  i.prefill_storage = _prefill_storage;
  i.w4_decode = _format == d::QwenWeightFormat::w4;
  i.w4_prefill_in_kernel = i.w4_decode && !_dual_prefill;
    i.position = _position;
    i.max_context = _max_context;
    i.prefill_chunk = _prefill_chunk;
    i.prefill_chunk_adaptive = _prefill_chunk_adaptive;
    i.prefill_chunk_max = _prefill_chunk_max;
    i.prefill_chunk_backend_limit = _prefill_chunk_backend_limit;
    i.prefill_chunk_selected = _prefill_chunk_selected;
    i.prefill_chunks = _prefill_chunks;
    i.cache_bytes = _cache_bytes;
 i.kv_dtype = _kv_storage.dtype;
 i.kv_layout = _kv_storage.layout;
 i.kv_elements_per_word = _kv_cache.elements_per_word();
 i.kv_region_words = _kv_cache.region_words();
 i.kv_payload_bytes = _kv_cache.payload_bytes();
 i.kv_page_bytes = _kv_cache.page_size_bytes();
 i.kv_pages_per_region = _kv_cache.pages_per_region();
 i.kv_heap_slots = _kv_cache.heap_slot_count();
 i.kv_min_commit_pages = _kv_cache.min_commit_pages();
 i.kv_packed_append = _kv_packed_append;
 i.kv_scaled_append = _kv_fp8_append;
 i.kv_scales_per_region = _kv_cache.scales_per_region();
 i.kv_scale_bytes = _kv_cache.scale_bytes();
    i.workspace_bytes = _workspace_bytes;
    i.workspace_chunk_bytes = _workspace_chunk_bytes;
    i.workspace_pool_floats_per_row = _workspace_pool_floats_per_row;
    i.compiled_shader_count = _shader_count;
      i.cuda_graph_steps = _cuda_graph_steps;
      i.cuda_graph_updates = _cuda_graph_updates;
      // Conditional WHILE decode chunk (opt_plan/09 N4.2): the SELECTED state, whether
      // the dedicated probe ran, and the counters that prove what actually launched.
      i.cuda_graph_while = _while_selected && !_while_capture_failed;
      i.cuda_graph_while_probe = _while_probe.supported;
      i.cuda_graph_while_chunk = _while_selected ? _while_chunk : 0u;
      i.cuda_graph_while_launches = _while_launches;
      i.cuda_graph_while_tokens = _while_tokens;
      i.cuda_graph_while_early_exits = _while_early_exits;
      i.cuda_graph_while_fallbacks = _while_fallbacks;
      i.cuda_graph_while_reason = _while_reason;
      // Pinned download staging (opt_plan/03 section 6): selection + accounting.
      i.pinned_transfers = _pinned_selection.selected;
      i.pinned_available = _pinned_selection.extension;
      i.pinned_ring_tokens = _pinned_selection.ring_tokens;
      i.pinned_downloads = _pinned_stats.downloads;
      i.pinned_bytes = _pinned_stats.bytes;
      i.pinned_host_copies = _pinned_stats.host_copies;
      i.pinned_reason = _pinned_selection.reason;
      // PDL: refresh the live counters (the extension owns them) so a log reports the
      // launches THIS session made instead of the creation-time snapshot.
      i.pdl = _pdl;
      if (auto *pdl_ext = _device ? _device.extension<compute::CUDAPdlStreamExt>() : nullptr) {
          i.pdl.available = true;
          i.pdl.supported = pdl_ext->pdl_supported();
          i.pdl.auto_prologue = pdl_ext->auto_prologue_enabled();
          i.pdl.stream_enabled = pdl_ext->pdl_stream_enabled();
          const auto pdl_stats = pdl_ext->launch_stats();
          i.pdl.relaxed_launches = pdl_stats.relaxed_launches;
          i.pdl.strict_launches = pdl_stats.strict_launches;
          i.pdl.driver_fallbacks = pdl_stats.driver_fallbacks;
      }

          i.gdn_wy_ut = _gdn_wy_ut_active;
      // OFFLINE merged input projections (opt_plan/03 section 3 + D5).
      i.merged_decode = _merged_decode;
      i.merged_available = _merged_available;
      if (_merged_available) {
          i.merged_gdn_n = _cfg.gdn_qkv_n() + _cfg.gdn_v_n() + 2u * _cfg.gdn_v_heads;
          i.merged_attn_n = _cfg.attn_q_proj_n() + 2u * _cfg.attn_kv_n();
      }
      i.merged_launches_saved = _merged_decode ? _merged_launches_saved : 0u;
      i.merged_matrices = _merged_matrices;
      i.merged_segments = _merged_segments;
      i.projection_pack_source = _merged_pack_name;
      i.projection_pack_format = _merged_pack_format;
    // Either addressable prefill GEMM set counts: in a dual-copy session the
  // decode-format one is deliberately not compiled.
  i.prefill_gemm_mt = _prefill_gemm_mt || _prefill_gemm_mt_pf;
    i.gdn_state_specialized = _gdn_state_specialized;
 i.rmsnorm_reg = _rmsnorm_reg;
 i.gdn_decode_fused = _gdn_decode_fused;
 i.gdn_recurrent_norm = _gdn_recurrent_norm;
 i.gdn_recurrent_norm_reason = _gdn_recurrent_norm_reason;
 i.decode_packed = _decode_packed;
 i.decode_packed_fp8 = _decode_packed && _format == d::QwenWeightFormat::fp8_e4m3;
 i.attn_prepare_model = _attn_prepare_model;
 i.gdn_auto = _gdn_auto;
 i.gdn_auto_threshold = _gdn_auto ? _gdn_auto_threshold : 0u;
    i.prefill_gemm_mt_rows = (_prefill_gemm_mt || _prefill_gemm_mt_pf) ? _gemm_mt_cfg.mt : 0u;
    i.native_prefill_attn = _native_attn.valid();
      if (i.native_prefill_attn) {
          i.native_prefill_attn_q_tile = _native_attn.config().q_tile;
          i.native_prefill_attn_kv_tile = _native_attn.config().kv_tile;
          i.native_prefill_attn_warps = _native_attn.config().warps();
          i.native_prefill_attn_shared_bytes = _native_attn.shared_bytes();
          i.native_prefill_attn_min_rows = d::qwen_native_attn_min_rows;
          i.native_prefill_attn_jit_ms = _native_attn.jit_ms();
      }
      i.native_prefill_attn_launches = _native_attn_launches;
      i.native_prefill_gemm = _native_gemm.valid();
    if (i.native_prefill_gemm) {
        i.native_prefill_gemm_min_rows = d::qwen_native_gemm_min_rows;
        i.native_prefill_gemm_block_m = _native_gemm.config().block_m;
        i.native_prefill_gemm_block_n = _native_gemm.config().block_n;
        i.native_prefill_gemm_block_k = _native_gemm.config().block_k;
        i.native_prefill_gemm_jit_ms = _native_gemm.jit_ms();
    }
    i.native_prefill_gemm_launches = _native_gemm_launches;
    // EXPERIMENTAL CUDA native-Tile typed-FP8 prefill GEMM. `valid()` is only true
    // when the caller requested it AND the whole create-time gate passed, so the
    // default report states false/zero and can never claim the Tile realization
    // engaged. The counters are the helper's own: lazy per-static-shape compiles,
    // their cache hits, the compiled shader count, their summed JIT time, the
    // create-time probe and the MEASURED ragged-N capability.
    i.cuda_tile_prefill_gemm = _cuda_tile_gemm.valid();
    i.cuda_tile_prefill_gemm_launches = _cuda_tile_gemm_launches;
    if (i.cuda_tile_prefill_gemm) {
        i.cuda_tile_prefill_gemm_m = _cuda_tile_gemm.config().tile_m;
        i.cuda_tile_prefill_gemm_n = _cuda_tile_gemm.config().tile_n;
        i.cuda_tile_prefill_gemm_k = _cuda_tile_gemm.config().tile_k;
        i.cuda_tile_prefill_gemm_min_rows = d::qwen_native_gemm_min_rows;
        i.cuda_tile_prefill_gemm_compiles = _cuda_tile_gemm.compile_count();
        i.cuda_tile_prefill_gemm_cache_hits = _cuda_tile_gemm.cache_hits();
        i.cuda_tile_prefill_gemm_shaders = _cuda_tile_gemm.shader_count();
        i.cuda_tile_prefill_gemm_jit_ms = _cuda_tile_gemm.jit_ms();
        i.cuda_tile_prefill_gemm_probe_ms = _cuda_tile_gemm.probe_ms();
        i.cuda_tile_prefill_gemm_ragged_n = _cuda_tile_gemm.supports_ragged_columns();
    }
    i.prefill_attn_tiled = _attn_tiled;
    // Split-KV decode attention (opt_plan/03 section 4.2). Everything here is derived
    // from the SELECTED configuration and the session position, never from a
    // per-token counter: in the static-graph path the nodes are recorded once, so a
    // counter would report the capture instead of the step. `decode_attn_kv_len` is
    // the key count the last/next decode step reads (the prepare of a step appends at
    // position, so kv_len == position), and `decode_attn_dedup` states whether that
    // kv_len takes the deduped path -- the exact flag the roof model needs.
    i.decode_attn_split = _attn_split;
    if (_attn_split) {
        i.decode_attn_threshold = _attn_split_cfg.threshold;
        i.decode_attn_max_splits = _attn_split_cfg.max_splits;
        i.decode_attn_k_tile = _attn_split_cfg.k_tile;
        i.decode_attn_shared_bytes = d::qwen_attn_decode_split_shared_bytes(_attn_split_cfg);
        i.decode_attn_partial_bytes = static_cast<uint32_t>(
            std::min<uint64_t>(d::qwen_attn_decode_split_partial_bytes(_attn_split_cfg), 0xFFFFFFFFull));
        i.decode_attn_kv_len = std::min(_position, _max_context);
        i.decode_attn_dedup = i.decode_attn_kv_len >= _attn_split_cfg.threshold;
        if (i.decode_attn_dedup) {
            i.decode_attn_splits = d::qwen_attn_decode_split_count(i.decode_attn_kv_len, _attn_split_cfg);
            i.decode_attn_blocks = d::qwen_attn_decode_split_blocks(_attn_split_cfg) + _cfg.q_heads;
        } else {
            i.decode_attn_splits = 0u;
            i.decode_attn_blocks = _cfg.q_heads;
        }
    } else {
        // Reference path: one block per q_head for every key count.
        i.decode_attn_kv_len = std::min(_position, _max_context);
        i.decode_attn_blocks = _cfg.q_heads;
    }
        // blocks/SM evidence (gate 11). The helper keeps a 0 SM count -- "not
        // queryable" -- at 0 instead of dividing by a device property nobody measured.
        i.decode_attn_blocks_per_sm =
            qwen_decode_attn_blocks_per_sm(i.decode_attn_blocks, _caps.sm_count);
    if (_attn_tiled) {
        i.prefill_attn_q_tile = _attn_tiled_cfg.q_tile;
        i.prefill_attn_k_tile = _attn_tiled_cfg.k_tile;
        i.prefill_attn_shared_bytes = d::qwen35_attn_prefill_tiled_shared_bytes(_attn_tiled_cfg);
    }
    i.caps = _caps;
    return i;
}

[[nodiscard]] const QwenBenchmarkReport *QwenDslEngine::benchmark_report() const noexcept {
    return _benchmark ? &_report : nullptr;
}

}// namespace luisa::qwen
