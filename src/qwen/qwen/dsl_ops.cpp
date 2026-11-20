// Qwen3.5-4B ordinary DSL operators -- implementation.
// Device-free kernel factories, overflow-checked host planning/validation,
// independent double-accumulation CPU oracles, and a device-free self-test.
// The exact contracts/layouts are declared in dsl_ops.h.
#include "qwen/dsl_ops.h"

#include <cmath>
#include <cstring>

#include <luisa/dsl/sugar.h>

namespace luisa::qwen::dsl {

using namespace luisa::compute;

// ---------------------------------------------------------------------------
// Status helpers (shared ocr::Status/Result contract; see dsl_ops.h).
// ---------------------------------------------------------------------------
namespace {

[[nodiscard]] Status err_invalid(luisa::string msg) {
    return Status::error(StatusCode::invalid_argument, std::move(msg));
}
[[nodiscard]] Status err_overflow(luisa::string msg) {
    return Status::error(StatusCode::invalid_argument, std::move(msg), "overflow");
}

// Checked uint64 -> uint32 for kernel scalar arguments.
[[nodiscard]] Result<uint32_t> checked_u32(uint64_t v, const char *what) {
    if (v > 0xFFFFFFFFull) {
        return Result<uint32_t>::failure(err_overflow(luisa::string{what} + " exceeds the uint32 kernel envelope"));
    }
    return Result<uint32_t>::success(static_cast<uint32_t>(v));
}

// Checked product (fail-closed on uint64 overflow AND the uint32 envelope).
[[nodiscard]] Result<uint64_t> checked_product_u32(uint64_t a, uint64_t b, const char *what) {
    if (a != 0u && b > 0xFFFFFFFFull / a) {
        return Result<uint64_t>::failure(err_overflow(luisa::string{what} + " product exceeds the uint32 envelope"));
    }
    return Result<uint64_t>::success(a * b);
}

}// namespace

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------
[[nodiscard]] float QwenDslConfig::attn_scale() const noexcept {
    return 1.0f / std::sqrt(static_cast<float>(attn_head_dim));
}
[[nodiscard]] float QwenDslConfig::gdn_q_scale() const noexcept {
    return 1.0f / std::sqrt(static_cast<float>(gdn_head_dim));
}

[[nodiscard]] Status validate_qwen_config(const QwenDslConfig &cfg) noexcept {
    if (cfg.hidden == 0u || cfg.ffn == 0u || cfg.vocab == 0u) {
        return err_invalid("qwen config: hidden/ffn/vocab must be nonzero");
    }
    if (cfg.q_heads == 0u || cfg.kv_heads == 0u || cfg.q_heads % cfg.kv_heads != 0u) {
        return err_invalid("qwen config: q_heads must be a positive multiple of kv_heads (GQA)");
    }
    if (cfg.attn_head_dim == 0u || cfg.attn_head_dim % 32u != 0u || cfg.attn_head_dim > 256u) {
        return err_invalid("qwen config: attn_head_dim must be a multiple of 32 in [32,256] (decode shared-tile bound)");
    }
    if (cfg.rope_dim == 0u || cfg.rope_dim % 2u != 0u || cfg.rope_dim > cfg.attn_head_dim) {
        return err_invalid("qwen config: rope_dim must be even and <= attn_head_dim");
    }
    if (cfg.gdn_qk_heads == 0u || cfg.gdn_v_heads == 0u || cfg.gdn_v_heads % cfg.gdn_qk_heads != 0u) {
        return err_invalid("qwen config: gdn_v_heads must be a positive multiple of gdn_qk_heads");
    }
    if (cfg.gdn_head_dim == 0u || cfg.gdn_head_dim % 32u != 0u || cfg.gdn_head_dim > 128u) {
        return err_invalid("qwen config: gdn_head_dim must be a multiple of 32 in [32,128] (one state column per lane)");
    }
    if (cfg.conv_kernel < 2u) {
        return err_invalid("qwen config: conv_kernel must be >= 2");
    }
    if (!(cfg.norm_eps > 0.0f) || !(cfg.l2_eps > 0.0f) || !(cfg.rope_theta > 0.0f)) {
        return err_invalid("qwen config: eps/theta must be positive");
    }
    if (cfg.fp8_group == 0u) {
        return err_invalid("qwen config: fp8_group must be nonzero");
    }
    // Fixed per-layer shapes must fit the uint32 index envelope (checked).
    const uint64_t pairs[][2] = {
        {cfg.vocab, cfg.hidden},
        {cfg.attn_q_proj_n(), cfg.hidden},
        {cfg.attn_kv_n(), cfg.hidden},
        {cfg.hidden, cfg.attn_o_k()},
        {cfg.gdn_qkv_n(), cfg.hidden},
        {cfg.gdn_v_n(), cfg.hidden},
        {cfg.hidden, cfg.gdn_v_n()},
        {cfg.ffn, cfg.hidden},
    };
    for (auto &p : pairs) {
        if (!checked_product_u32(p[0], p[1], "qwen config weight shape").ok()) {
            return err_overflow("qwen config: a fixed per-layer weight shape exceeds uint32");
        }
    }
    const auto state_n = checked_product_u32(cfg.gdn_v_heads, cfg.gdn_head_dim, "qwen config state");
    if (!state_n.ok() || !checked_product_u32(*state_n.value, cfg.gdn_head_dim, "qwen config state").ok()) {
        return err_overflow("qwen config: GDN state shape exceeds uint32");
    }
    return Status::success();
}

[[nodiscard]] luisa::string_view qwen_weight_format_name(QwenWeightFormat format) noexcept {
    switch (format) {
        case QwenWeightFormat::bf16: return "bf16";
        case QwenWeightFormat::fp8_e4m3: return "fp8_e4m3";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// Host bit helpers (exact parity with the device decode paths)
// ---------------------------------------------------------------------------
[[nodiscard]] uint16_t host_f32_to_bf16_bits(float v) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &v, 4u);
    if (((bits >> 23u) & 0xFFu) == 0xFFu && (bits & 0x7FFFFFu) != 0u) {
        // NaN: keep a quiet NaN payload (truncate, force quiet bit).
        return static_cast<uint16_t>((bits >> 16u) | 0x0040u);
    }
    // Round-to-nearest-even on the discarded low 16 bits.
    const uint32_t rounding_bias = 0x7FFFu + ((bits >> 16u) & 1u);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16u);
}

[[nodiscard]] float host_bf16_bits_to_f32(uint16_t bits) noexcept {
    const uint32_t out = static_cast<uint32_t>(bits) << 16u;
    float f;
    std::memcpy(&f, &out, 4u);
    return f;
}

[[nodiscard]] float host_fp8_e4m3_bits_to_f32(uint8_t code) noexcept {
    const uint32_t sign = (code >> 7u) & 1u;
    const uint32_t exp = (code >> 3u) & 0xFu;
    const uint32_t mant = code & 0x7u;
    uint32_t out;
    if (exp == 0u) {
        if (mant == 0u) {
            out = sign << 31u;// +/- zero
        } else {
            // Subnormal: value = mant * 2^-9. Normalize.
            uint32_t m = mant;
            uint32_t e = 0u;
            while ((m & 0x8u) == 0u) {
                m <<= 1u;
                e += 1u;
            }
            m &= 0x7u;
            out = (sign << 31u) | ((127u - 7u - e + 1u) << 23u) | (m << 20u);
        }
    } else if (exp == 0xFu && mant == 0x7u) {
        out = (sign << 31u) | 0x7FC00000u;// NaN (OCP E4M3FN: no infinities)
    } else {
        out = (sign << 31u) | ((exp + 127u - 7u) << 23u) | (mant << 20u);
    }
    float f;
    std::memcpy(&f, &out, 4u);
    return f;
}

[[nodiscard]] uint8_t host_f32_to_fp8_e4m3_bits(float v) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &v, 4u);
    const uint32_t sign = (bits >> 31u) & 1u;
    const int32_t exp = static_cast<int32_t>((bits >> 23u) & 0xFFu);
    const uint32_t mant = bits & 0x7FFFFFu;
    if (exp == 0xFF) {
        // NaN -> E4M3 NaN; Inf -> saturate to max finite (448).
        if (mant != 0u) { return static_cast<uint8_t>((sign << 7u) | 0x7Fu); }
        return static_cast<uint8_t>((sign << 7u) | 0x7Eu);
    }
    const int32_t unbiased = exp - 127;
    if (exp == 0) {
        // FP32 subnormal: magnitude < 2^-126, far below the E4M3 subnormal
        // floor 2^-9; rounds to zero (ties-to-even keeps zero).
        return static_cast<uint8_t>(sign << 7u);
    }
    // Value = 1.mant * 2^unbiased (or subnormal E4M3 when unbiased < -6).
    // E4M3 normal: exp field e = unbiased + 7 in [1,15], 3 mantissa bits.
    // E4M3 subnormal: e = 0, value = m/8 * 2^-6, m in [0,7].
    int32_t e;
    uint32_t m23;// mantissa in units of 2^-23 relative to the leading 1
    if (unbiased >= -6) {
        e = unbiased + 7;
        m23 = mant;
    } else {
        // Subnormal: shift the implicit-1 mantissa right by (-6 - unbiased).
        const int32_t shift = -6 - unbiased;// >= 1
        if (shift > 24) { return static_cast<uint8_t>(sign << 7u); }
        const uint32_t full = (1u << 23u) | mant;
        // Round-to-nearest-even at bit position (shift + 20).
        const int32_t drop = shift + 20;
        if (drop >= 32) { return static_cast<uint8_t>(sign << 7u); }
        const uint32_t keep = full >> drop;
        const uint32_t rem = full & ((1u << drop) - 1u);
        const uint32_t half = 1u << (drop - 1u);
        uint32_t rounded = keep;
        if (rem > half || (rem == half && (keep & 1u) != 0u)) { rounded += 1u; }
        if (rounded >= 8u) {
            // Promoted to the smallest normal.
            return static_cast<uint8_t>((sign << 7u) | (1u << 3u));
        }
        return static_cast<uint8_t>((sign << 7u) | rounded);
    }
    if (e >= 15) {
        // e == 15 with mantissa 7 is NaN; saturate finite overflow to 0x7E
        // (e=15, m=6 -> 448). Round first: values >= 464 (midpoint 448..480)
        // would round to 480 which is not representable -> saturate.
        if (e > 15) { return static_cast<uint8_t>((sign << 7u) | 0x7Eu); }
        // e == 15: round mantissa to 3 bits, cap at 6.
        const uint32_t keep = m23 >> 20u;
        const uint32_t rem = m23 & 0xFFFFFu;
        uint32_t rounded = keep;
        if (rem > 0x80000u || (rem == 0x80000u && (keep & 1u) != 0u)) { rounded += 1u; }
        if (rounded > 6u) { rounded = 6u; }
        return static_cast<uint8_t>((sign << 7u) | (15u << 3u) | rounded);
    }
    // Normal range: round mantissa to 3 bits (RNE), carry into exponent.
    const uint32_t keep = m23 >> 20u;
    const uint32_t rem = m23 & 0xFFFFFu;
    uint32_t rounded = keep;
    if (rem > 0x80000u || (rem == 0x80000u && (keep & 1u) != 0u)) { rounded += 1u; }
    if (rounded == 8u) {
        rounded = 0u;
        e += 1;
        if (e >= 15) {
            // 256 * 2^8 = ... e==15,m==0 is 256? No: e=15 -> 2^8=256, m=0 -> 256.
            // Actually e=15,m=0 = 2^(15-7)=256; fine unless it overflowed past
            // the max finite: e==15 && m>6 saturates.
            return static_cast<uint8_t>((sign << 7u) | (15u << 3u));
        }
    }
    if (e == 15 && rounded > 6u) { rounded = 6u; }
    return static_cast<uint8_t>((sign << 7u) | (static_cast<uint32_t>(e) << 3u) | rounded);
}

[[nodiscard]] Result<uint64_t> packed_matrix_words(QwenWeightFormat format, uint64_t k, uint64_t n) {
    const auto elems = checked_product_u32(k, n, "packed matrix");
    if (!elems.ok()) { return Result<uint64_t>::failure(elems.status); }
    const uint64_t e = *elems.value;
    switch (format) {
        case QwenWeightFormat::bf16:
            if ((e & 1u) != 0u) {
                return Result<uint64_t>::failure(err_invalid("bf16 packing requires an even element count (K*N)"));
            }
            return Result<uint64_t>::success(e / 2u);
        case QwenWeightFormat::fp8_e4m3:
            return Result<uint64_t>::success((e + 3u) / 4u);
    }
    return Result<uint64_t>::failure(err_invalid("unknown weight format"));
}

[[nodiscard]] Result<uint64_t> fp8_scale_elements(uint64_t k, uint64_t n, uint32_t group) {
    if (group == 0u) { return Result<uint64_t>::failure(err_invalid("fp8 group must be nonzero")); }
    const uint64_t groups_per_row = (k + group - 1u) / group;
    return checked_product_u32(groups_per_row, n, "fp8 scales");
}

[[nodiscard]] Status host_pack_matrix(QwenWeightFormat format, luisa::span<const float> w,
                                         uint64_t k, uint64_t n, uint32_t group,
                                         luisa::vector<uint32_t> &words,
                                         luisa::vector<float> &scales) {
    const auto wc = packed_matrix_words(format, k, n);
    if (!wc.ok()) { return wc.status; }
    if (w.size() != k * n) { return err_invalid("host_pack_matrix: weight span size mismatch"); }
    words.assign(static_cast<size_t>(*wc.value), 0u);
    scales.clear();
    if (format == QwenWeightFormat::bf16) {
        for (uint64_t i = 0; i < k * n; i += 2u) {
            const uint32_t lo = host_f32_to_bf16_bits(w[i]);
            const uint32_t hi = host_f32_to_bf16_bits(w[i + 1u]);
            words[static_cast<size_t>(i / 2u)] = lo | (hi << 16u);
        }
        return Status::success();
    }
    if (group == 0u) { return err_invalid("host_pack_matrix: fp8 group must be nonzero"); }
    const auto sc = fp8_scale_elements(k, n, group);
    if (!sc.ok()) { return sc.status; }
    scales.resize(static_cast<size_t>(*sc.value));
    const uint64_t gpr = (k + group - 1u) / group;
    for (uint64_t row = 0; row < n; row++) {
        for (uint64_t g = 0; g < gpr; g++) {
            const uint64_t begin = g * group;
            const uint64_t end = begin + group < k ? begin + group : k;
            double max_abs = 0.0;
            for (uint64_t i = begin; i < end; i++) {
                const double a = std::fabs(static_cast<double>(w[row * k + i]));
                if (a > max_abs) { max_abs = a; }
            }
            // scale = max_abs/448; zero groups get scale 0 (codes all zero).
            const float scale = static_cast<float>(max_abs / 448.0);
            scales[static_cast<size_t>(row * gpr + g)] = scale;
            const double inv = scale > 0.0f ? 1.0 / static_cast<double>(scale) : 0.0;
            for (uint64_t i = begin; i < end; i++) {
                const double q = static_cast<double>(w[row * k + i]) * inv;
                // Saturate to the finite E4M3 range before RNE conversion.
                const double clamped = q > 448.0 ? 448.0 : (q < -448.0 ? -448.0 : q);
                const uint8_t code = host_f32_to_fp8_e4m3_bits(static_cast<float>(clamped));
                const uint64_t idx = row * k + i;
                words[static_cast<size_t>(idx / 4u)] |= static_cast<uint32_t>(code) << ((idx & 3u) * 8u);
            }
        }
    }
    return Status::success();
}

[[nodiscard]] float host_decode_packed(QwenWeightFormat format, luisa::span<const uint32_t> words,
                                       luisa::span<const float> scales, uint64_t k,
                                       uint32_t group, uint64_t index) noexcept {
    if (format == QwenWeightFormat::bf16) {
        const uint32_t word = words[static_cast<size_t>(index >> 1u)];
        const uint32_t bits = (word >> ((index & 1u) * 16u)) & 0xFFFFu;
        return host_bf16_bits_to_f32(static_cast<uint16_t>(bits));
    }
    const uint32_t word = words[static_cast<size_t>(index >> 2u)];
    const uint32_t code = (word >> ((index & 3u) * 8u)) & 0xFFu;
    const uint64_t row = index / k;
    const uint64_t gpr = (k + group - 1u) / group;
    const float scale = scales[static_cast<size_t>(row * gpr + (index % k) / group)];
    return host_fp8_e4m3_bits_to_f32(static_cast<uint8_t>(code)) * scale;
}

// ---------------------------------------------------------------------------
// Launch planning
// ---------------------------------------------------------------------------
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_threads(uint64_t count, uint32_t block) {
    if (block == 0u) { return Result<QwenLaunch1D>::failure(err_invalid("block size must be nonzero")); }
    QwenLaunch1D plan;
    plan.block = block;
    plan.work_items = count;
    if (count == 0u) { return Result<QwenLaunch1D>::success(plan); }
    if (count > 0xFFFFFFFFull) {
        return Result<QwenLaunch1D>::failure(err_overflow("work item count exceeds uint32"));
    }
    const uint64_t groups = (count + block - 1u) / block;
    if (groups <= kQwenMaxGroupsPerDim) {
        plan.total_x = static_cast<uint32_t>(groups * block);
        plan.total_y = 1u;
    } else {
        // Flatten: x stays one block-row wide times a bounded group count,
        // y carries the remaining block rows. Keep total_x a multiple of
        // block and both group counts inside the portable envelope.
        const uint64_t gx = kQwenMaxGroupsPerDim;
        const uint64_t gy = (groups + gx - 1u) / gx;
        if (gy > kQwenMaxGroupsPerDim) {
            return Result<QwenLaunch1D>::failure(err_overflow("flattened grid exceeds the 2D group envelope"));
        }
        const uint64_t padded_threads = gx * block * gy;
        if (padded_threads > 0xFFFFFFFFull) {
            return Result<QwenLaunch1D>::failure(err_overflow("flattened total-thread count exceeds uint32"));
        }
        plan.total_x = static_cast<uint32_t>(gx * block);
        plan.total_y = static_cast<uint32_t>(gy);
    }
    return Result<QwenLaunch1D>::success(plan);
}

[[nodiscard]] Result<QwenLaunch2D> plan_qwen_threads_2d(uint64_t rows, uint64_t cols,
                                                          uint32_t block_x, uint32_t block_y) {
    if (block_x == 0u || block_y == 0u) {
        return Result<QwenLaunch2D>::failure(err_invalid("block sizes must be nonzero"));
    }
    QwenLaunch2D plan;
    plan.rows = rows;
    plan.cols = cols;
    plan.block_x = block_x;
    plan.block_y = block_y;
    if (rows == 0u || cols == 0u) { return Result<QwenLaunch2D>::success(plan); }
    const auto c32 = checked_u32(cols, "cols");
    if (!c32.ok()) { return Result<QwenLaunch2D>::failure(c32.status); }
    const uint64_t gx = (cols + block_x - 1u) / block_x;
    const uint64_t gy = (rows + block_y - 1u) / block_y;
    if (gx > kQwenMaxGroupsPerDim) {
        return Result<QwenLaunch2D>::failure(err_overflow("column group count exceeds the portable envelope"));
    }
    plan.padded_cols = static_cast<uint32_t>(gx * block_x);
    if (gy <= kQwenMaxGroupsPerDim) {
        plan.total_x = static_cast<uint32_t>(gx * block_x);
        plan.total_y = static_cast<uint32_t>(gy * block_y);
        plan.folded = false;
    } else {
        // Fold block-rows into x: the linear id spans padded_cols columns
        // per row; the kernel recovers row = linear / padded_cols.
        const uint64_t total_threads = static_cast<uint64_t>(plan.padded_cols) * gy * block_y;
        const uint64_t groups = (total_threads + block_x - 1u) / block_x;
        if (groups > static_cast<uint64_t>(kQwenMaxGroupsPerDim) * kQwenMaxGroupsPerDim) {
            return Result<QwenLaunch2D>::failure(err_overflow("folded grid exceeds the 2D group envelope"));
        }
        const uint64_t fx = kQwenMaxGroupsPerDim;
        const uint64_t fy = (groups + fx - 1u) / fx;
        plan.total_x = static_cast<uint32_t>(fx * block_x);
        plan.total_y = static_cast<uint32_t>(fy * block_y);
        plan.folded = true;
    }
    return Result<QwenLaunch2D>::success(plan);
}

// Warp-per-row plan: `rows` warps of 32 lanes, blocks of `block` threads
// (multiple of 32). work_items = rows (the kernel guards warp_id < rows).
namespace {
[[nodiscard]] Result<QwenLaunch1D> plan_warp_rows(uint64_t rows, uint32_t block) {
    if (block == 0u || block % 32u != 0u) {
        return Result<QwenLaunch1D>::failure(err_invalid("warp-row block must be a nonzero multiple of 32"));
    }
    const auto threads = checked_product_u32(rows, 32u, "warp rows");
    if (!threads.ok()) { return Result<QwenLaunch1D>::failure(threads.status); }
    auto plan = plan_qwen_threads(*threads.value, block);
    if (!plan.ok()) { return Result<QwenLaunch1D>::failure(plan.status); }
    plan.value->work_items = rows;
    plan.value->warp_size = 32u;
    return plan;
}
}// namespace

// ---------------------------------------------------------------------------
// Device-side shared helpers (recorded into the AST; no host state).
// ---------------------------------------------------------------------------
namespace {

// Linear id for a (possibly flattened) 1D launch. dispatch_size() is the
// TOTAL thread count per dimension.
[[nodiscard]] auto qwen_linear_id() noexcept {
    return dispatch_y() * dispatch_size().x + dispatch_x();
}

// (row, col) recovery for a folded 2D launch (padded_cols stride).
[[nodiscard]] auto qwen_folded_row() noexcept { return qwen_linear_id(); }

// Exact bf16 -> fp32.
[[nodiscard]] auto qwen_bf16_to_f32(Var<uint32_t> bits) noexcept {
    return as<float>(bits << 16u);
}

// Exact E4M3 -> fp32 (finite codes incl. subnormals; S.1111.111 -> NaN).
[[nodiscard]] auto qwen_fp8_e4m3_to_f32(Var<uint32_t> code) noexcept {
    Var<uint32_t> sign = (code >> 7u) & 1u;
    Var<uint32_t> exp = (code >> 3u) & 0xFu;
    Var<uint32_t> mant = code & 0x7u;
    Var<uint32_t> out_bits = def(0u);
    $if (exp == 0u) {
        $if (mant != 0u) {
            Var<uint32_t> m = def(mant);
            Var<uint32_t> e = def(0u);
            $while ((m & 0x8u) == 0u) {
                m = m << 1u;
                e += 1u;
            };
            m = m & 0x7u;
            out_bits = (sign << 31u) | ((127u - 7u - e + 1u) << 23u) | (m << 20u);
        } $else {
            out_bits = sign << 31u;
        };
    } $elif (exp == 0xFu & mant == 0x7u) {
        out_bits = (sign << 31u) | 0x7FC00000u;
    } $else {
        out_bits = (sign << 31u) | ((exp + 127u - 7u) << 23u) | (mant << 20u);
    };
    return as<float>(out_bits);
}

// Decode one packed weight element. `format` is a compile-time host value.
// fp8 scale index = row * groups_per_row + (k_index / group); the caller
// passes the row base scale offset and the within-row k index.
[[nodiscard]] auto qwen_read_weight(const BufferVar<uint32_t> &packed,
                                    const BufferVar<float> &scales,
                                    Var<uint32_t> index, Var<uint32_t> row_scale_base,
                                    Var<uint32_t> k_index, Var<uint32_t> group,
                                    QwenWeightFormat format) noexcept {
    if (format == QwenWeightFormat::bf16) {
        const auto word = packed.read(index >> 1u);
        const auto bits = (word >> ((index & 1u) * 16u)) & 0xFFFFu;
        return qwen_bf16_to_f32(bits);
    }
    const auto word = packed.read(index >> 2u);
    const auto code = (word >> ((index & 3u) * 8u)) & 0xFFu;
    const auto scale = scales.read(row_scale_base + k_index / group);
    return qwen_fp8_e4m3_to_f32(code) * scale;
}

[[nodiscard]] auto qwen_silu(Var<float> x) noexcept {
    return x / (1.0f + exp(-x));
}
[[nodiscard]] auto qwen_sigmoid(Var<float> x) noexcept {
    return 1.0f / (1.0f + exp(-x));
}

}// namespace

// ---------------------------------------------------------------------------
// 1. Embedding gather
// ---------------------------------------------------------------------------
[[nodiscard]] QwenEmbeddingKernel make_qwen_embedding_kernel(QwenWeightFormat format) noexcept {
    QwenEmbeddingKernel kernel = [format](BufferUInt ids, BufferUInt table, BufferFloat scales,
                                          BufferFloat out, Var<uint> tokens, Var<uint> hidden,
                                          Var<uint> vocab, Var<uint> group, Var<uint> padded_cols) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name(format == QwenWeightFormat::bf16 ? "qwen_embedding_bf16" : "qwen_embedding_fp8");
        // Folded launches recover (row,col) from the linear id; unfolded
        // launches use dispatch ids directly. padded_cols == dispatch_size().x
        // in the unfolded case, so one code path covers both.
        const auto linear = qwen_linear_id();
        const auto row = linear / padded_cols;
        const auto col = linear % padded_cols;
        $if (row < tokens & col < hidden) {
            const auto id = ids.read(row);
            // Defensive: an out-of-range id leaves the row untouched rather
            // than reading OOB (host validation should have rejected it).
            $if (id < vocab) {
                const auto index = id * hidden + col;
                Var<float> v = def(0.0f);
                if (format == QwenWeightFormat::bf16) {
                    const auto word = table.read(index >> 1u);
                    v = qwen_bf16_to_f32((word >> ((index & 1u) * 16u)) & 0xFFFFu);
                } else {
                    const auto word = table.read(index >> 2u);
                    const auto code = (word >> ((index & 3u) * 8u)) & 0xFFu;
                    const auto gpr = (hidden + group - 1u) / group;
                    v = qwen_fp8_e4m3_to_f32(code) * scales.read(id * gpr + col / group);
                }
                out.write(row * hidden + col, v);
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch2D> plan_qwen_embedding_dispatch(uint64_t tokens, uint64_t hidden) {
    const auto n = checked_product_u32(tokens, hidden, "embedding output");
    if (!n.ok()) { return Result<QwenLaunch2D>::failure(n.status); }
    return plan_qwen_threads_2d(tokens, hidden);
}

[[nodiscard]] Status validate_qwen_embedding_inputs(luisa::span<const uint32_t> ids,
                                                       uint64_t hidden, uint64_t vocab) {
    if (hidden == 0u || vocab == 0u) { return err_invalid("embedding hidden/vocab must be nonzero"); }
    if (!checked_product_u32(hidden, vocab, "embedding table").ok()) {
        return err_overflow("embedding table exceeds uint32");
    }
    for (auto id : ids) {
        if (static_cast<uint64_t>(id) >= vocab) {
            return err_invalid("embedding token id out of range");
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// 2. Zero-centered RMSNorm (warp per row)
// ---------------------------------------------------------------------------
[[nodiscard]] QwenRmsNormKernel make_qwen_rmsnorm_kernel(QwenNormEpilogue epilogue) noexcept {
    QwenRmsNormKernel kernel = [epilogue](BufferFloat x, BufferFloat w, BufferFloat residual,
                                          BufferFloat y, BufferFloat sum_out,
                                          Var<uint> rows, Var<uint> dim, Var<float> eps) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name(epilogue == QwenNormEpilogue::residual ? "qwen_rmsnorm_residual"
                 : epilogue == QwenNormEpilogue::snapshot ? "qwen_rmsnorm_snapshot"
                                                          : "qwen_rmsnorm");
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        // Block-uniform guard: every lane of a participating warp enters.
        $if (warp < rows & dim > 0u) {
            const auto base = warp * dim;
            Var<float> sumsq = def(0.0f);
            $for (i, lane, dim, 32u) {
                Var<float> v = x.read(base + i);
                if (epilogue == QwenNormEpilogue::residual) {
                    v += residual.read(base + i);
                    sum_out.write(base + i, v);
                } else if (epilogue == QwenNormEpilogue::snapshot) {
                    // Residual snapshot fused into the sum-of-squares pass:
                    // store the ORIGINAL x bit-exactly (no arithmetic on v),
                    // so the second pass below can normalize from sum_out
                    // with identical FP32 math to plain. x/w/y/sum_out must
                    // be distinct buffers (see header aliasing contract).
                    sum_out.write(base + i, v);
                }
                sumsq += v * v;
            };
            sumsq = warp_active_sum(sumsq);
            const auto inv = rsqrt(sumsq / cast<float>(dim) + eps);
            $for (i, lane, dim, 32u) {
                Var<float> v = def(0.0f);
                if (epilogue == QwenNormEpilogue::plain) {
                    v = x.read(base + i);
                } else {
                    // residual and snapshot both normalize the value they
                    // just wrote to sum_out (same-lane store->load; the
                    // warp_active_sum above separates the two passes).
                    v = sum_out.read(base + i);
                }
                // GGUF norm weights are PRE-ADDED (the llama.cpp converter
                // folds the HF zero-centered 1+w offline); multiply DIRECTLY.
                y.write(base + i, v * inv * w.read(i));
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_rmsnorm_dispatch(uint64_t rows, uint64_t dim) {
    if (dim > 0xFFFFFFFFull) { return Result<QwenLaunch1D>::failure(err_overflow("norm dim exceeds uint32")); }
    const auto n = checked_product_u32(rows, dim, "norm elements");
    if (!n.ok()) { return Result<QwenLaunch1D>::failure(n.status); }
    return plan_warp_rows(rows, 128u);
}

[[nodiscard]] QwenRmsNormKernel make_qwen35_rmsnorm_kernel(
    QwenNormEpilogue epilogue, uint32_t block_size) noexcept {
    QwenRmsNormKernel kernel = [epilogue, block_size](BufferFloat x, BufferFloat w, BufferFloat residual,
                                          BufferFloat y, BufferFloat sum_out,
                                          Var<uint> rows, Var<uint> dim, Var<float> eps) noexcept {
        set_block_size(block_size, 1u, 1u);
        set_warp_size(32u);
        set_name(epilogue == QwenNormEpilogue::residual ? "qwen35_rmsnorm_residual"
                 : epilogue == QwenNormEpilogue::snapshot ? "qwen35_rmsnorm_snapshot"
                                                          : "qwen35_rmsnorm");
        Shared<float> partial{8u};
        const auto row = qwen_linear_id() / block_size;
        const auto tid = thread_x();
        const auto lane = warp_lane_id();
        const auto wid = tid / 32u;
        $if (row < rows & dim > 0u) {
            const auto base = row * dim;
            Var<float> sumsq = def(0.0f);
            $for (i, tid, dim, block_size) {
                Var<float> v = x.read(base + i);
                if (epilogue == QwenNormEpilogue::residual) {
                    v += residual.read(base + i);
                    sum_out.write(base + i, v);
                } else if (epilogue == QwenNormEpilogue::snapshot) {
                    sum_out.write(base + i, v);
                }
                sumsq += v * v;
            };
            sumsq = warp_active_sum(sumsq);
            $if (lane == 0u) { partial[wid] = sumsq; };
            sync_block();
            // Every thread uses the same four-partial order, then normalizes
            // its own strided elements. No second cross-warp round trip.
            sumsq = 0.0f;
            for (uint32_t w = 0u; w < block_size / 32u; w++) {
                sumsq += partial.read(w);
            }
            const auto inv = rsqrt(sumsq / cast<float>(dim) + eps);
            $for (i, tid, dim, block_size) {
                const auto v = epilogue == QwenNormEpilogue::plain
                                   ? x.read(base + i)
                                   : sum_out.read(base + i);
                y.write(base + i, v * inv * w.read(i));
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen35_rmsnorm_dispatch(
    uint64_t rows, uint64_t dim, uint32_t block_size) {
    if (block_size != 128u && block_size != 256u) {
        return Result<QwenLaunch1D>::failure(err_invalid("qwen35 norm block must be 128 or 256"));
    }
    if (dim > 0xFFFFFFFFull) {
        return Result<QwenLaunch1D>::failure(err_overflow("qwen35 norm dim exceeds uint32"));
    }
    const auto n = checked_product_u32(rows, dim, "qwen35 norm elements");
    if (!n.ok()) { return Result<QwenLaunch1D>::failure(n.status); }
    const auto threads = checked_product_u32(rows, block_size, "qwen35 norm threads");
    if (!threads.ok()) { return Result<QwenLaunch1D>::failure(threads.status); }
    auto plan = plan_qwen_threads(*threads.value, block_size);
    if (!plan.ok()) { return plan; }
    plan.value->work_items = rows;
    plan.value->warp_size = 32u;
    return plan;
}

// ---------------------------------------------------------------------------
// 3/4. Packed GEMV / GEMM
// ---------------------------------------------------------------------------
namespace {

// Shared body: y[m,n] = dot(x[m,:], W[n,:]) (+residual). `epilogue` and
// `format` are compile-time. Used by both the warp GEMV and the thread GEMM.
[[nodiscard]] auto qwen_matmul_dot(const BufferVar<float> &x, const BufferVar<uint32_t> &w,
                                   const BufferVar<float> &scales,
                                   Var<uint> x_base, Var<uint> w_base, Var<uint> scale_base,
                                   Var<uint> k, Var<uint> group, Var<uint> begin, Var<uint> step,
                                   QwenWeightFormat format) noexcept {
    Var<float> acc = def(0.0f);
    $for (ki, begin, k, step) {
        const auto xv = x.read(x_base + ki);
        const auto wv = qwen_read_weight(w, scales, w_base + ki, scale_base, ki, group, format);
        acc += xv * wv;
    };
    return acc;
}

// Packed-word-oriented GEMV dot (warp lanes own whole packed words).
// Each lane owns packed word `begin` and strides by `step_words` words:
//   bf16: ki = 2*begin, K stride 64  -> one uint32, two bf16 halves;
//   fp8 : ki = 4*begin, K stride 128 -> one uint32, four e4m3 bytes.
// Each packed word is decoded exactly once per lane iteration and the
// products accumulate in ascending K order. For fp8 the caller guarantees
// K % 4 == 0 AND group % 4 == 0, so a quad never crosses a scale boundary:
// ONE scale load per word, no per-element group division.
// NOTE: this repartitions K across lanes (lane owns a contiguous quad/pair
// instead of single strided elements), so the warp reduction order -- and
// therefore the final rounding -- differs from qwen_matmul_dot. The FP32
// math per product is identical; results are NOT bit-identical to the
// strided path, only within accumulation-order noise.
[[nodiscard]] auto qwen_gemv_dot_packed(const BufferVar<float> &x, const BufferVar<uint32_t> &w,
                                        const BufferVar<float> &scales,
                                        Var<uint> x_base, Var<uint> w_base, Var<uint> scale_base,
                                        Var<uint> k, Var<uint> group,
                                        Var<uint> begin_word, Var<uint> step_words,
                                        QwenWeightFormat format) noexcept {
    Var<float> acc = def(0.0f);
    // `words` = k / pack (caller guarantees k % pack == 0). The word base
    // is computed once on the caller side (warp * words).
    const auto words = format == QwenWeightFormat::bf16 ? (k >> 1u) : (k >> 2u);
    // group % 4 == 0 and group > 0 are guaranteed for fp8 by the caller, so
    // ki / group == wi / (group / 4) for the quad's base index.
    const auto group_words = group >> 2u;
    $for (wi, begin_word, words, step_words) {
        const auto word = w.read(w_base + wi);
        if (format == QwenWeightFormat::bf16) {
            const auto ki = wi * 2u;
            const auto w0 = qwen_bf16_to_f32(word & 0xFFFFu);
            const auto w1 = qwen_bf16_to_f32((word >> 16u) & 0xFFFFu);
            acc += x.read(x_base + ki) * w0;
            acc += x.read(x_base + ki + 1u) * w1;
        } else {
            const auto ki = wi * 4u;
            const auto scale = scales.read(scale_base + wi / group_words);
            const auto w0 = qwen_fp8_e4m3_to_f32(word & 0xFFu) * scale;
            const auto w1 = qwen_fp8_e4m3_to_f32((word >> 8u) & 0xFFu) * scale;
            const auto w2 = qwen_fp8_e4m3_to_f32((word >> 16u) & 0xFFu) * scale;
            const auto w3 = qwen_fp8_e4m3_to_f32((word >> 24u) & 0xFFu) * scale;
            acc += x.read(x_base + ki) * w0;
            acc += x.read(x_base + ki + 1u) * w1;
            acc += x.read(x_base + ki + 2u) * w2;
            acc += x.read(x_base + ki + 3u) * w3;
        }
    };
    return acc;
}

// Fast-path eligibility (host-computed shape facts, uniform across the
// warp): K divides the pack width, and for fp8 the group divides the quad.
[[nodiscard]] auto qwen_gemv_packed_ok(Var<uint> k, Var<uint> group,
                                       QwenWeightFormat format) noexcept {
    if (format == QwenWeightFormat::bf16) {
        return (k & 1u) == 0u;
    }
    return (k & 3u) == 0u & (group & 3u) == 0u & group > 0u;
}

}// namespace

[[nodiscard]] QwenGemvKernel make_qwen_gemv_kernel(QwenWeightFormat format,
                                                   QwenMatmulEpilogue epilogue) noexcept {
    QwenGemvKernel kernel = [format, epilogue](BufferFloat x, BufferUInt w, BufferFloat scales,
                                               BufferFloat residual, BufferFloat y,
                                               Var<uint> m, Var<uint> n, Var<uint> k, Var<uint> group) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name(format == QwenWeightFormat::bf16 ? "qwen_gemv_bf16" : "qwen_gemv_fp8");
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        $if (warp < n & k > 0u & m > 0u) {
            const auto w_base = warp * k;
            Var<uint> scale_base = def(0u);
            if (format == QwenWeightFormat::fp8_e4m3) {
                scale_base = warp * ((k + group - 1u) / group);
            }
            // Each lane accumulates its K subset for every m row, then one
            // warp reduction per row. Fast path: lane owns whole packed
            // words (one word read + decode per pair/quad, one fp8 scale
            // load per quad, no per-element group division); lanes read
            // consecutive words (coalesced along K). Ragged K (or fp8 group
            // not divisible by 4) falls back to the legacy per-element
            // strided loop. The $if condition is warp-uniform, so
            // warp_active_sum stays block-uniform on both paths. The two
            // paths partition K differently -> rounding differs (see
            // qwen_gemv_dot_packed).
            $for (mi, m) {
                Var<float> acc = def(0.0f);
                $if (qwen_gemv_packed_ok(k, group, format)) {
                    const auto word_base = format == QwenWeightFormat::bf16 ? w_base / 2u : w_base / 4u;
                    acc = qwen_gemv_dot_packed(x, w, scales, mi * k, word_base, scale_base, k, group,
                                               lane, 32u, format);
                } $else {
                    acc = qwen_matmul_dot(x, w, scales, mi * k, w_base, scale_base, k, group,
                                          lane, 32u, format);
                };
                acc = warp_active_sum(acc);
                $if (lane == 0u) {
                    const auto idx = mi * n + warp;
                    Var<float> v = acc;
                    if (epilogue == QwenMatmulEpilogue::residual) {
                        v += residual.read(idx);
                    }
                    y.write(idx, v);
                };
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gemv_dispatch(uint64_t m, uint64_t n, uint64_t k) {
    if (m == 0u || m > kQwenGemvMaxM) {
        return Result<QwenLaunch1D>::failure(err_invalid("gemv M must be in [1,4]"));
    }
    const auto xk = checked_product_u32(m, k, "gemv x");
    if (!xk.ok()) { return Result<QwenLaunch1D>::failure(xk.status); }
    const auto wn = checked_product_u32(n, k, "gemv weights");
    if (!wn.ok()) { return Result<QwenLaunch1D>::failure(wn.status); }
    const auto yn = checked_product_u32(m, n, "gemv output");
    if (!yn.ok()) { return Result<QwenLaunch1D>::failure(yn.status); }
    return plan_warp_rows(n, 128u);
}

namespace {

// Decode-only Qwen shape specialization. `words` is a multiple of 128, so
// every lane executes four valid, independent streams without tail guards.
// Keeping the streams in separate accumulators also breaks the single FMA
// dependency chain of the generic packed-word loop.
[[nodiscard]] auto qwen35_decode_dot(const BufferVar<float> &x, const BufferVar<uint32_t> &w,
                                     const BufferVar<float> &scales, Shared<float> &lut,
                                     Var<uint> w_base, Var<uint> scale_base,
                                     Var<uint> k, Var<uint> group,
                                     Var<uint> lane, QwenWeightFormat format) noexcept {
    Var<float> a0 = def(0.0f), a1 = def(0.0f), a2 = def(0.0f), a3 = def(0.0f);
    const auto words = format == QwenWeightFormat::bf16 ? (k >> 1u) : (k >> 2u);
    const auto group_words = group >> 2u;
    const auto word_base = format == QwenWeightFormat::bf16 ? w_base / 2u : w_base / 4u;
    const auto accumulate = [&](Var<uint> wi, Var<float> &acc) {
        const auto word = w.read(word_base + wi);
        if (format == QwenWeightFormat::bf16) {
            const auto ki = wi * 2u;
            acc += x.read(ki) * qwen_bf16_to_f32(word & 0xFFFFu);
            acc += x.read(ki + 1u) * qwen_bf16_to_f32(word >> 16u);
        } else {
            const auto ki = wi * 4u;
            const auto scale = scales.read(scale_base + wi / group_words);
            acc += x.read(ki) * lut.read(word & 0xFFu) * scale;
            acc += x.read(ki + 1u) * lut.read((word >> 8u) & 0xFFu) * scale;
            acc += x.read(ki + 2u) * lut.read((word >> 16u) & 0xFFu) * scale;
            acc += x.read(ki + 3u) * lut.read((word >> 24u) & 0xFFu) * scale;
        }
    };
    $for (wi, lane, words, 128u) {
        accumulate(wi, a0);
        accumulate(wi + 32u, a1);
        accumulate(wi + 64u, a2);
        accumulate(wi + 96u, a3);
    };
    return (a0 + a1) + (a2 + a3);
}

}// namespace

[[nodiscard]] QwenGemvKernel make_qwen35_decode_gemv_kernel(
    QwenWeightFormat format, QwenMatmulEpilogue epilogue) noexcept {
    QwenGemvKernel kernel = [format, epilogue](BufferFloat x, BufferUInt w, BufferFloat scales,
                                               BufferFloat residual, BufferFloat y,
                                               Var<uint> m, Var<uint> n, Var<uint> k,
                                               Var<uint> group) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name(format == QwenWeightFormat::bf16 ? "qwen35_decode_gemv_bf16"
                                                  : "qwen35_decode_gemv_fp8");
        Shared<float> lut{format == QwenWeightFormat::fp8_e4m3 ? 256u : 1u};
        if (format == QwenWeightFormat::fp8_e4m3) {
            // Two entries per thread. This exact conversion is paid once per
            // block, not once per packed weight; all consumers follow the
            // block barrier. Random LUT reads remain in 1 KiB of shared memory.
            $for (i, thread_x(), 256u, 128u) {
                lut[i] = qwen_fp8_e4m3_to_f32(i);
            };
            sync_block();
        }
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        $if (warp < n & m == 1u & k > 0u) {
            const auto w_base = warp * k;
            Var<uint> scale_base = def(0u);
            if (format == QwenWeightFormat::fp8_e4m3) {
                scale_base = warp * ((k + group - 1u) / group);
            }
            auto acc = qwen35_decode_dot(x, w, scales, lut, w_base, scale_base,
                                         k, group, lane, format);
            acc = warp_active_sum(acc);
            $if (lane == 0u) {
                Var<float> v = acc;
                if (epilogue == QwenMatmulEpilogue::residual) {
                    v += residual.read(warp);
                }
                y.write(warp, v);
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen35_decode_gemv_dispatch(
    uint64_t n, uint64_t k, QwenWeightFormat format) {
    const uint64_t pack = format == QwenWeightFormat::bf16 ? 2u : 4u;
    const uint64_t words_per_launch = 128u * pack;
    if (k == 0u || k % words_per_launch != 0u) {
        return Result<QwenLaunch1D>::failure(err_invalid(
            "qwen35 decode GEMV requires K divisible by 128 packed words"));
    }
    return plan_qwen_gemv_dispatch(1u, n, k);
}

[[nodiscard]] QwenGemmKernel make_qwen_gemm_kernel(QwenWeightFormat format,
                                                   QwenMatmulEpilogue epilogue) noexcept {
    QwenGemmKernel kernel = [format, epilogue](BufferFloat x, BufferUInt w, BufferFloat scales,
                                               BufferFloat residual, BufferFloat y,
                                               Var<uint> m, Var<uint> n, Var<uint> k, Var<uint> group,
                                               Var<uint> padded_cols) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name(format == QwenWeightFormat::bf16 ? "qwen_gemm_bf16" : "qwen_gemm_fp8");
        const auto linear = qwen_linear_id();
        const auto row = linear / padded_cols;
        const auto col = linear % padded_cols;
        $if (row < m & col < n & k > 0u) {
            Var<uint> scale_base = def(0u);
            if (format == QwenWeightFormat::fp8_e4m3) {
                scale_base = col * ((k + group - 1u) / group);
            }
            auto acc = qwen_matmul_dot(x, w, scales, row * k, col * k, scale_base, k, group,
                                       0u, 1u, format);
            const auto idx = row * n + col;
            Var<float> v = acc;
            if (epilogue == QwenMatmulEpilogue::residual) {
                v += residual.read(idx);
            }
            y.write(idx, v);
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch2D> plan_qwen_gemm_dispatch(uint64_t m, uint64_t n, uint64_t k) {
    const auto xk = checked_product_u32(m, k, "gemm x");
    if (!xk.ok()) { return Result<QwenLaunch2D>::failure(xk.status); }
    const auto wn = checked_product_u32(n, k, "gemm weights");
    if (!wn.ok()) { return Result<QwenLaunch2D>::failure(wn.status); }
    const auto yn = checked_product_u32(m, n, "gemm output");
    if (!yn.ok()) { return Result<QwenLaunch2D>::failure(yn.status); }
    return plan_qwen_threads_2d(m, n);
}

// ---------------------------------------------------------------------------
// 5. SwiGLU
// ---------------------------------------------------------------------------
[[nodiscard]] QwenSwiGluSplitKernel make_qwen_swiglu_split_kernel() noexcept {
    QwenSwiGluSplitKernel kernel = [](BufferFloat gate, BufferFloat up, BufferFloat out,
                                      Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_swiglu_split");
        const auto i = qwen_linear_id();
        $if (i < count) {
            out.write(i, qwen_silu(gate.read(i)) * up.read(i));
        };
    };
    return kernel;
}

[[nodiscard]] QwenSwiGluFusedKernel make_qwen_swiglu_fused_kernel() noexcept {
    QwenSwiGluFusedKernel kernel = [](BufferFloat proj, BufferFloat out,
                                      Var<uint> m, Var<uint> f, Var<uint> padded_cols) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("qwen_swiglu_fused");
        const auto linear = qwen_linear_id();
        const auto row = linear / padded_cols;
        const auto col = linear % padded_cols;
        $if (row < m & col < f) {
            const auto g = proj.read(row * (2u * f) + col);
            const auto u = proj.read(row * (2u * f) + f + col);
            out.write(row * f + col, qwen_silu(g) * u);
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_swiglu_dispatch(uint64_t count) {
    return plan_qwen_threads(count);
}
[[nodiscard]] Result<QwenLaunch2D> plan_qwen_swiglu_fused_dispatch(uint64_t m, uint64_t f) {
    const auto n = checked_product_u32(m, f, "swiglu output");
    if (!n.ok()) { return Result<QwenLaunch2D>::failure(n.status); }
    return plan_qwen_threads_2d(m, f);
}

[[nodiscard]] QwenFfnDecodeKernel make_qwen_ffn_decode_kernel(QwenWeightFormat format) noexcept {
    QwenFfnDecodeKernel kernel = [format](BufferFloat x, BufferUInt gate_w,
                                          BufferFloat gate_scales, BufferUInt up_w,
                                          BufferFloat up_scales, BufferFloat out,
                                          Var<uint> f, Var<uint> k, Var<uint> group) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name(format == QwenWeightFormat::bf16 ? "qwen_ffn_decode_bf16"
                                                  : "qwen_ffn_decode_fp8");
        Shared<float> lut{format == QwenWeightFormat::fp8_e4m3 ? 256u : 1u};
        if (format == QwenWeightFormat::fp8_e4m3) {
            $for (i, thread_x(), 256u, 128u) { lut[i] = qwen_fp8_e4m3_to_f32(i); };
            sync_block();
        }
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        $if (warp < f & k > 0u) {
            const auto words = format == QwenWeightFormat::bf16 ? (k >> 1u) : (k >> 2u);
            const auto element_base = warp * k;
            const auto word_base = format == QwenWeightFormat::bf16
                                       ? element_base / 2u : element_base / 4u;
            Var<uint> scale_base = def(0u);
            if (format == QwenWeightFormat::fp8_e4m3) {
                scale_base = warp * ((k + group - 1u) / group);
            }
            const auto group_words = group >> 2u;
            Var<float> gate_acc = def(0.0f), up_acc = def(0.0f);
            $for (wi, lane, words, 32u) {
                const auto gw = gate_w.read(word_base + wi);
                const auto uw = up_w.read(word_base + wi);
                if (format == QwenWeightFormat::bf16) {
                    const auto ki = wi * 2u;
                    const auto x0 = x.read(ki);
                    const auto x1 = x.read(ki + 1u);
                    gate_acc += x0 * qwen_bf16_to_f32(gw & 0xFFFFu);
                    gate_acc += x1 * qwen_bf16_to_f32(gw >> 16u);
                    up_acc += x0 * qwen_bf16_to_f32(uw & 0xFFFFu);
                    up_acc += x1 * qwen_bf16_to_f32(uw >> 16u);
                } else {
                    const auto ki = wi * 4u;
                    const auto gs = gate_scales.read(scale_base + wi / group_words);
                    const auto us = up_scales.read(scale_base + wi / group_words);
                    const auto x0 = x.read(ki);
                    const auto x1 = x.read(ki + 1u);
                    const auto x2 = x.read(ki + 2u);
                    const auto x3 = x.read(ki + 3u);
                    gate_acc += x0 * lut.read(gw & 0xFFu) * gs;
                    gate_acc += x1 * lut.read((gw >> 8u) & 0xFFu) * gs;
                    gate_acc += x2 * lut.read((gw >> 16u) & 0xFFu) * gs;
                    gate_acc += x3 * lut.read((gw >> 24u) & 0xFFu) * gs;
                    up_acc += x0 * lut.read(uw & 0xFFu) * us;
                    up_acc += x1 * lut.read((uw >> 8u) & 0xFFu) * us;
                    up_acc += x2 * lut.read((uw >> 16u) & 0xFFu) * us;
                    up_acc += x3 * lut.read((uw >> 24u) & 0xFFu) * us;
                }
            };
            gate_acc = warp_active_sum(gate_acc);
            up_acc = warp_active_sum(up_acc);
            $if (lane == 0u) { out.write(warp, qwen_silu(gate_acc) * up_acc); };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_ffn_decode_dispatch(
    uint64_t f, uint64_t k, QwenWeightFormat format, uint32_t group) {
    const uint64_t pack = format == QwenWeightFormat::bf16 ? 2u : 4u;
    if (k == 0u || k % pack != 0u ||
        (format == QwenWeightFormat::fp8_e4m3 &&
         (group == 0u || group % 4u != 0u))) {
        return Result<QwenLaunch1D>::failure(err_invalid("fused FFN decode K/group is not packed-word aligned"));
    }
    const auto xk = checked_product_u32(1u, k, "fused FFN x");
    const auto wn = checked_product_u32(f, k, "fused FFN weights");
    if (!xk.ok()) { return Result<QwenLaunch1D>::failure(xk.status); }
    if (!wn.ok()) { return Result<QwenLaunch1D>::failure(wn.status); }
    return plan_warp_rows(f, 128u);
}

// ---------------------------------------------------------------------------
// 6. Elementwise utilities.
// ---------------------------------------------------------------------------
[[nodiscard]] QwenResidualAddKernel make_qwen_residual_add_kernel() noexcept {
    QwenResidualAddKernel kernel = [](BufferFloat a, BufferFloat b, BufferFloat out,
                                      Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_residual_add");
        const auto i = qwen_linear_id();
        $if (i < count) { out.write(i, a.read(i) + b.read(i)); };
    };
    return kernel;
}

[[nodiscard]] QwenZeroFillKernel make_qwen_zero_fill_kernel() noexcept {
    QwenZeroFillKernel kernel = [](BufferFloat out, Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_zero_fill");
        const auto i = qwen_linear_id();
        $if (i < count) { out.write(i, 0.0f); };
    };
    return kernel;
}

[[nodiscard]] QwenCopyKernel make_qwen_copy_kernel() noexcept {
    QwenCopyKernel kernel = [](BufferFloat src, BufferFloat out, Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_copy");
  const auto i = qwen_linear_id();
    $if (i < count) { out.write(i, src.read(i)); };
  };
  return kernel;
}

[[nodiscard]] QwenCopyUKernel make_qwen_copy_u_kernel() noexcept {
  QwenCopyUKernel kernel = [](BufferUInt src, BufferUInt out, Var<uint> count) noexcept {
    set_block_size(256u, 1u, 1u);
    set_name("qwen_copy_u32");
    const auto i = qwen_linear_id();
    $if (i < count) { out.write(i, src.read(i)); };
  };
  return kernel;
}

[[nodiscard]] QwenCopyIKernel make_qwen_copy_i_kernel() noexcept {
  QwenCopyIKernel kernel = [](BufferInt src, BufferInt out, Var<uint> count) noexcept {
    set_block_size(256u, 1u, 1u);
    set_name("qwen_copy_i32");
    const auto i = qwen_linear_id();
    $if (i < count) { out.write(i, src.read(i)); };
  };
  return kernel;
}

[[nodiscard]] QwenGreedyArgmaxKernel make_qwen_greedy_argmax_kernel() noexcept {
    QwenGreedyArgmaxKernel kernel = [](BufferFloat logits, BufferUInt next_ids,
                                       BufferUInt history, Var<uint> vocab,
                                       Var<uint> slot) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_greedy_argmax");
        Shared<float> values{256u};
        Shared<uint> indices{256u};
        Shared<uint> invalid{256u};
        const auto tid = thread_x();
        Var<float> best = as<float>(0xFF800000u); // negative infinity, including all -inf rows
        Var<uint> best_index = def(vocab);
        Var<uint> bad = def(0u);
        $for (i, tid, vocab, 256u) {
            const auto v = logits.read(i);
            $if (isnan(v)) {
                bad = 1u;
            } $else {
                $if (v > best | (v == best & i < best_index)) {
                    best = v;
                    best_index = i;
                };
            };
        };
        values[tid] = best;
        indices[tid] = best_index;
        invalid[tid] = bad;
        sync_block();
        // Each step reads the upper half and writes only the disjoint lower
        // half, so one barrier per level is sufficient. Explicit index compare
        // makes the result independent of reduction-tree shape.
        for (uint32_t stride = 128u; stride != 0u; stride >>= 1u) {
            $if (tid < stride) {
                const auto other_v = values.read(tid + stride);
                const auto other_i = indices.read(tid + stride);
                $if (other_v > values.read(tid) |
                     (other_v == values.read(tid) & other_i < indices.read(tid))) {
                    values[tid] = other_v;
                    indices[tid] = other_i;
                };
                invalid[tid] = invalid.read(tid) | invalid.read(tid + stride);
            };
            sync_block();
        }
        $if (tid == 0u) {
            Var<uint> token = indices.read(0u);
            $if (invalid.read(0u) != 0u) { token = vocab; };
            next_ids.write(0u, token);
            history.write(slot, token);
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_greedy_argmax_dispatch(uint64_t vocab) {
    if (vocab == 0u || vocab > 0xFFFFFFFFull) {
        return Result<QwenLaunch1D>::failure(err_invalid("greedy argmax vocab must be in [1,uint32_max]"));
    }
    auto plan = plan_qwen_threads(256u, 256u);
    if (plan.ok()) { plan.value->work_items = 1u; }
    return plan;
}

namespace {

void qwen_greedy_reduce_shared(Shared<float> &values, Shared<uint> &indices,
                               Shared<uint> &invalid) noexcept {
    sync_block();
    for (uint32_t stride = 128u; stride != 0u; stride >>= 1u) {
        const auto tid = thread_x();
        $if (tid < stride) {
            const auto other_v = values.read(tid + stride);
            const auto other_i = indices.read(tid + stride);
            $if (other_v > values.read(tid) |
                 (other_v == values.read(tid) & other_i < indices.read(tid))) {
                values[tid] = other_v;
                indices[tid] = other_i;
            };
            invalid[tid] = invalid.read(tid) | invalid.read(tid + stride);
        };
        sync_block();
    }
}

}// namespace

[[nodiscard]] QwenGreedyArgmaxPartialKernel make_qwen_greedy_argmax_partial_kernel() noexcept {
    QwenGreedyArgmaxPartialKernel kernel = [](BufferFloat logits, BufferFloat out_values,
                                              BufferUInt out_indices, BufferUInt out_invalid,
                                              Var<uint> vocab) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_greedy_argmax_partial");
        Shared<float> values{256u};
        Shared<uint> indices{256u};
        Shared<uint> invalid{256u};
        const auto block = qwen_linear_id() / 256u;
        const auto tid = thread_x();
        Var<float> best = as<float>(0xFF800000u);
        Var<uint> best_index = def(vocab);
        Var<uint> bad = def(0u);
        $for (i, block * 256u + tid, vocab, qwen_greedy_argmax_blocks * 256u) {
            const auto v = logits.read(i);
            $if (isnan(v)) {
                bad = 1u;
            } $else {
                $if (v > best | (v == best & i < best_index)) {
                    best = v;
                    best_index = i;
                };
            };
        };
        values[tid] = best;
        indices[tid] = best_index;
        invalid[tid] = bad;
        qwen_greedy_reduce_shared(values, indices, invalid);
        $if (tid == 0u) {
            out_values.write(block, values.read(0u));
            out_indices.write(block, indices.read(0u));
            out_invalid.write(block, invalid.read(0u));
        };
    };
    return kernel;
}

[[nodiscard]] QwenGreedyArgmaxFinishKernel make_qwen_greedy_argmax_finish_kernel() noexcept {
    QwenGreedyArgmaxFinishKernel kernel = [](BufferFloat in_values, BufferUInt in_indices,
                                             BufferUInt in_invalid, BufferUInt next_ids,
                                             BufferUInt history, Var<uint> vocab,
                                             Var<uint> slot) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_greedy_argmax_finish");
        Shared<float> values{256u};
        Shared<uint> indices{256u};
        Shared<uint> invalid{256u};
        const auto tid = thread_x();
        values[tid] = in_values.read(tid);
        indices[tid] = in_indices.read(tid);
        invalid[tid] = in_invalid.read(tid);
        qwen_greedy_reduce_shared(values, indices, invalid);
        $if (tid == 0u) {
            Var<uint> token = indices.read(0u);
            $if (invalid.read(0u) != 0u) { token = vocab; };
            next_ids.write(0u, token);
            history.write(slot, token);
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_greedy_argmax_partial_dispatch(uint64_t vocab) {
    if (vocab == 0u || vocab > 0xFFFFFFFFull) {
        return Result<QwenLaunch1D>::failure(err_invalid("greedy partial argmax vocab is invalid"));
    }
    auto plan = plan_qwen_threads(qwen_greedy_argmax_blocks * 256u, 256u);
    if (plan.ok()) { plan.value->work_items = qwen_greedy_argmax_blocks; }
    return plan;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_greedy_argmax_finish_dispatch(uint64_t vocab) {
    return plan_qwen_greedy_argmax_dispatch(vocab);
}

// ---------------------------------------------------------------------------
// 7. Full attention
// ---------------------------------------------------------------------------
namespace {

// NeoX rotate-half partial RoPE for one lane-owned pair (i, i+half) of a
// head row, half = rope_dim/2. inv_freq[i] = theta^(-2i/rope_dim).
// Text mRoPE: the three position axes collapse to the same scalar position,
// so one frequency table serves all of them (HF recomposition_frequencies
// duplicates cos/sin across the two halves).
[[nodiscard]] auto qwen_rope_pair(Var<float> x0, Var<float> x1, Var<float> pos,
                                  Var<uint> pair, Var<uint> rope_dim, Var<float> theta) noexcept {
    const auto angle = pos * exp2(log2(theta) * (-cast<float>(2u * pair) / cast<float>(rope_dim)));
    const auto c = cos(angle);
    const auto s = sin(angle);
    return compose(x0 * c - x1 * s, x0 * s + x1 * c);
}

}// namespace

[[nodiscard]] QwenAttnPrepareKernel make_qwen_attn_prepare_kernel(
    bool append_kv, bool implicit_position_ramp) noexcept {
    QwenAttnPrepareKernel kernel = [append_kv, implicit_position_ramp](
                                       BufferFloat q_proj, BufferFloat k_proj, BufferFloat v_proj,
                                       BufferFloat q_norm_w, BufferFloat k_norm_w, BufferInt positions,
                                       BufferFloat q_out, BufferFloat gate_out, BufferFloat k_out, BufferFloat v_out,
                                       BufferFloat k_cache, BufferFloat v_cache,
                                       Var<uint> tokens, Var<uint> q_heads, Var<uint> kv_heads,
                                       Var<uint> head_dim, Var<uint> rope_dim, Var<float> theta, Var<float> eps,
                                       Var<uint> pos0, Var<uint> cache_capacity,
                                       Var<uint> q_rows, Var<uint> k_rows, Var<uint> v_rows) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name(append_kv ? "qwen_attn_prepare_append" : "qwen_attn_prepare");
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        const auto total_rows = q_rows + k_rows + v_rows;
        // One warp per (token, head, kind) row; kind order Q, K, V.
        $if (warp < total_rows & head_dim > 0u & tokens > 0u) {
            Var<uint> kind = def(0u);// 0=Q,1=K,2=V
            Var<uint> row = def(warp);
            $if (row >= q_rows) {
                row -= q_rows;
                kind = 1u;
                $if (row >= k_rows) {
                    row -= k_rows;
                    kind = 2u;
                };
            };
            // row is now (token, head) within its kind.
            Var<uint> heads = q_heads;
            $if (kind != 0u) { heads = kv_heads; };
            const auto t = row / heads;
            const auto h = row % heads;
        const auto pos = implicit_position_ramp
                             ? cast<float>(pos0 + t)
                             : cast<float>(positions.read(t));
        $if (kind == 0u) {
                // Q row: source stride 2*head_dim per head ([Q|gate]).
                const auto src = (t * q_heads + h) * (2u * head_dim);
                const auto dst = (t * q_heads + h) * head_dim;
                Var<float> sumsq = def(0.0f);
                $for (i, lane, head_dim, 32u) {
                    const auto v = q_proj.read(src + i);
                    sumsq += v * v;
                };
                sumsq = warp_active_sum(sumsq);
                const auto inv = rsqrt(sumsq / cast<float>(head_dim) + eps);
                $for (i, lane, head_dim, 32u) {
                    const auto v = q_proj.read(src + i) * inv * q_norm_w.read(i);
                    Var<float> out_v = def(v);
                    $if (i < rope_dim) {
                        // NeoX rotate-half: pair (p, p+half), half = rope_dim/2.
                        // Each lane recomputes both elements of its pair so it
                        // writes exactly its own element.
                        const auto half = rope_dim / 2u;
                        const auto p = i % half;
                        const auto lo = p;
                        const auto hi = p + half;
                        const auto x0 = q_proj.read(src + lo) * inv * q_norm_w.read(lo);
                        const auto x1 = q_proj.read(src + hi) * inv * q_norm_w.read(hi);
                        const auto rr = qwen_rope_pair(x0, x1, pos, p, rope_dim, theta);
                        out_v = ite(i < half, rr.get<0>(), rr.get<1>());
                    };
                    q_out.write(dst + i, out_v);
                };
                // Gate: raw copy, no norm/rope.
                $for (i, lane, head_dim, 32u) {
                    gate_out.write(dst + i, q_proj.read(src + head_dim + i));
                };
            } $else {
                const auto src = (t * kv_heads + h) * head_dim;
                const auto dst = (t * kv_heads + h) * head_dim;
                $if (kind == 1u) {
                    Var<float> sumsq = def(0.0f);
                    $for (i, lane, head_dim, 32u) {
                        const auto v = k_proj.read(src + i);
                        sumsq += v * v;
                    };
                    sumsq = warp_active_sum(sumsq);
                    const auto inv = rsqrt(sumsq / cast<float>(head_dim) + eps);
                    $for (i, lane, head_dim, 32u) {
                        const auto v = k_proj.read(src + i) * inv * k_norm_w.read(i);
                        Var<float> out_v = def(v);
                        $if (i < rope_dim) {
                            const auto half = rope_dim / 2u;
                            const auto p = i % half;
                            const auto lo = p;
                            const auto hi = p + half;
                            const auto x0 = k_proj.read(src + lo) * inv * k_norm_w.read(lo);
                            const auto x1 = k_proj.read(src + hi) * inv * k_norm_w.read(hi);
                            const auto rr = qwen_rope_pair(x0, x1, pos, p, rope_dim, theta);
                            out_v = ite(i < half, rr.get<0>(), rr.get<1>());
                        };
                        k_out.write(dst + i, out_v);
                        if (append_kv) {
                            const auto slot = (pos0 + t) * (kv_heads * head_dim) + h * head_dim + i;
                            $if (pos0 + t < cache_capacity) { k_cache.write(slot, out_v); };
                        }
                    };
                } $else {
                    $for (i, lane, head_dim, 32u) {
                        const auto v = v_proj.read(src + i);
                        v_out.write(dst + i, v);
                        if (append_kv) {
                            const auto slot = (pos0 + t) * (kv_heads * head_dim) + h * head_dim + i;
                            $if (pos0 + t < cache_capacity) { v_cache.write(slot, v); };
                        }
                    };
                };
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_attn_prepare_dispatch(uint64_t tokens, const QwenDslConfig &cfg) {
    const uint64_t rows = tokens * (static_cast<uint64_t>(cfg.q_heads) + 2u * cfg.kv_heads);
    return plan_warp_rows(rows, 128u);
}

[[nodiscard]] Status validate_qwen_attn_prepare_inputs(uint64_t tokens, const QwenDslConfig &cfg,
                                                          uint64_t pos0, uint64_t cache_capacity, bool append_kv) {
    const auto cfg_ok = validate_qwen_config(cfg);
    if (!cfg_ok.ok()) { return cfg_ok; }
    if (tokens == 0u) { return Status::success(); }
    if (tokens > 0xFFFFFFFFull) { return err_overflow("attention tokens exceed uint32"); }
    if (!checked_product_u32(tokens, cfg.attn_q_proj_n(), "attention q projection").ok() ||
        !checked_product_u32(tokens, cfg.attn_kv_n(), "attention kv projection").ok()) {
        return err_overflow("attention projection element count exceeds uint32");
    }
    if (pos0 > 0x7FFFFFFFull || tokens > 0x7FFFFFFFull - pos0) {
        return err_overflow("attention absolute position exceeds int32");
    }
    if (append_kv) {
        if (pos0 + tokens > cache_capacity) {
            return err_invalid("attention KV append would exceed cache capacity");
        }
        if (!checked_product_u32(cache_capacity, cfg.attn_kv_n(), "KV cache").ok()) {
            return err_overflow("KV cache element count exceeds uint32");
        }
    }
    return Status::success();
}

[[nodiscard]] QwenRopePartialKernel make_qwen_rope_partial_neox_kernel() noexcept {
    QwenRopePartialKernel kernel = [](BufferFloat x, BufferFloat out, BufferInt positions,
                                      Var<uint> tokens, Var<uint> heads, Var<uint> head_dim,
                                      Var<uint> rope_dim, Var<float> theta, Var<uint> rows) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name("qwen_rope_partial_neox");
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        $if (warp < rows & head_dim > 0u & tokens > 0u) {
            const auto t = warp / heads;
            const auto base = warp * head_dim;
            const auto pos = cast<float>(positions.read(t));
            // Rotated prefix: NeoX pairs (p, p+half), half = rope_dim/2;
            // each lane handles pairs p = lane, lane+32, ... and writes both
            // elements of its pair exclusively.
            const auto half = rope_dim / 2u;
            $for (p, lane, half, 32u) {
                const auto x0 = x.read(base + p);
                const auto x1 = x.read(base + p + half);
                const auto rr = qwen_rope_pair(x0, x1, pos, p, rope_dim, theta);
                out.write(base + p, rr.get<0>());
                out.write(base + p + half, rr.get<1>());
            };
            // Tail passthrough.
            $for (i, rope_dim + lane, head_dim, 32u) {
                out.write(base + i, x.read(base + i));
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_rope_dispatch(uint64_t tokens, uint64_t heads) {
    const auto rows = checked_product_u32(tokens, heads, "rope rows");
    if (!rows.ok()) { return Result<QwenLaunch1D>::failure(rows.status); }
    return plan_warp_rows(*rows.value, 128u);
}

[[nodiscard]] QwenKvAppendKernel make_qwen_kv_append_kernel() noexcept {
    QwenKvAppendKernel kernel = [](BufferFloat k, BufferFloat v, BufferFloat k_cache, BufferFloat v_cache,
                                   Var<uint> tokens, Var<uint> channels, Var<uint> pos0,
                                   Var<uint> capacity, Var<uint> padded_cols) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("qwen_kv_append");
        const auto linear = qwen_linear_id();
        const auto row = linear / padded_cols;
        const auto col = linear % padded_cols;
        $if (row < tokens & col < channels) {
            const auto p = pos0 + row;
            $if (p < capacity) {
                const auto src = row * channels + col;
                const auto dst = p * channels + col;
                k_cache.write(dst, k.read(src));
                v_cache.write(dst, v.read(src));
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch2D> plan_qwen_kv_append_dispatch(uint64_t tokens, uint64_t channels) {
    const auto n = checked_product_u32(tokens, channels, "kv append");
    if (!n.ok()) { return Result<QwenLaunch2D>::failure(n.status); }
    return plan_qwen_threads_2d(tokens, channels);
}

[[nodiscard]] QwenAttnPrefillKernel make_qwen_attn_prefill_kernel() noexcept {
    QwenAttnPrefillKernel kernel = [](BufferFloat q, BufferFloat gate, BufferFloat k_cache, BufferFloat v_cache,
                                      BufferFloat out, Var<uint> tokens, Var<uint> q_heads, Var<uint> kv_heads,
                                      Var<uint> head_dim, Var<float> scale, Var<uint> pos0, Var<uint> kv_len) noexcept {
        set_block_size(128u, 1u, 1u);
        set_name("qwen_attn_prefill");
        const auto id = qwen_linear_id();
        $if (id < tokens * q_heads & head_dim > 0u & kv_heads > 0u) {
            const auto t = id / q_heads;
            const auto h = id % q_heads;
            const auto kvh = h / (q_heads / kv_heads);
            const auto q_base = (t * q_heads + h) * head_dim;
            const auto kv_stride = kv_heads * head_dim;
            const auto k_end = pos0 + t + 1u;// causal: keys [0, pos0+t]
            // Online softmax: running max m, sum l, output acc[d].
            Var<float> run_m = def(-3.0e38f);
            Var<float> run_l = def(0.0f);
            // Online softmax: running max m, sum l, output acc[d]. The
            // accumulator is a fixed-size local array (max head_dim 1024);
            // only [0, head_dim) is used, explicitly zeroed here.
            Local<float> acc{1024u};
            $for (d, head_dim) { acc[d] = 0.0f; };
            $for (key, k_end) {
                const auto k_base = key * kv_stride + kvh * head_dim;
                Var<float> score = def(0.0f);
                $for (d, head_dim) {
                    score += q.read(q_base + d) * k_cache.read(k_base + d);
                };
                score *= scale;
                const auto new_m = max(run_m, score);
                const auto corr = exp(run_m - new_m);
                const auto p = exp(score - new_m);
                run_l = run_l * corr + p;
                $for (d, head_dim) {
                    acc[d] = acc[d] * corr + p * v_cache.read(k_base + d);
                };
                run_m = new_m;
            };
            const auto inv_l = 1.0f / run_l;
            $for (d, head_dim) {
                const auto g = gate.read(q_base + d);
                out.write(q_base + d, acc[d] * inv_l * qwen_sigmoid(g));
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_attn_prefill_dispatch(uint64_t tokens, uint64_t q_heads) {
    const auto n = checked_product_u32(tokens, q_heads, "attention prefill rows");
    if (!n.ok()) { return Result<QwenLaunch1D>::failure(n.status); }
    return plan_qwen_threads(*n.value, 128u);
}

[[nodiscard]] Status validate_qwen_attn_prefill_inputs(uint64_t tokens, uint64_t q_heads, uint64_t kv_heads,
                                                          uint64_t head_dim, uint64_t pos0, uint64_t kv_len,
                                                          uint64_t capacity) {
    if (q_heads == 0u || kv_heads == 0u || q_heads % kv_heads != 0u) {
        return err_invalid("attention GQA requires q_heads % kv_heads == 0");
    }
    if (head_dim == 0u || head_dim > 1024u) {
        return err_invalid("attention head_dim must be in [1,1024] (local accumulator bound)");
    }
    if (tokens == 0u) { return Status::success(); }
    if (pos0 > kv_len || tokens > kv_len - pos0) {
        return err_invalid("attention prefill causal window exceeds kv_len (append first)");
    }
    if (kv_len > capacity) { return err_invalid("attention kv_len exceeds cache capacity"); }
    const auto cache_n = checked_product_u32(capacity, kv_heads, "attention cache");
    const auto q_n = checked_product_u32(tokens, q_heads, "attention queries");
    if (!cache_n.ok() || !q_n.ok() ||
        !checked_product_u32(*cache_n.value, head_dim, "attention cache").ok() ||
        !checked_product_u32(*q_n.value, head_dim, "attention queries").ok()) {
        return err_overflow("attention element count exceeds uint32");
    }
    if (pos0 > 0xFFFFFFFFull || tokens > 0xFFFFFFFFull - pos0) {
        return err_overflow("attention position exceeds uint32");
    }
    return Status::success();
}

[[nodiscard]] QwenAttnDecodeKernel make_qwen_attn_decode_kernel() noexcept {
    QwenAttnDecodeKernel kernel = [](BufferFloat q, BufferFloat gate, BufferFloat k_cache, BufferFloat v_cache,
                                     BufferFloat out, Var<uint> q_heads, Var<uint> kv_heads,
                                     Var<uint> head_dim, Var<float> scale, Var<uint> kv_len,
                                     Var<uint> capacity) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name("qwen_attn_decode");
        // Shared merge buffers at kernel scope (uniform allocation).
        Shared<float> sm{4u};
        Shared<float> sl{4u};
        Shared<float> sacc{4u * 256u};// head_dim <= 256 for decode (validator)
        // Block index from the linear thread id (works for flattened plans).
        const auto h = qwen_linear_id() / 128u;
        // Block-uniform guard: all 128 threads of a participating block enter.
        $if (h < q_heads & head_dim > 0u & head_dim <= 256u & kv_heads > 0u & kv_len > 0u) {
            const auto lane = warp_lane_id();
            const auto wid = thread_x() / 32u;// warp index in block (0..3)
            const auto kvh = h / (q_heads / kv_heads);
            const auto q_base = h * head_dim;
            const auto kv_stride = kv_heads * head_dim;
            const auto nkeys = min(kv_len, capacity);// defensive clamp
            // Per-warp online softmax over the strided key subset
            // key = wid, wid+4, wid+8, ... Lane l owns dims l, l+32, ...
            // A warp with an EMPTY key subset keeps m=-inf sentinel, l=0,
            // acc=0; the merge below skips it (exp(-inf - m_max) = 0).
            Var<float> run_m = def(-3.0e38f);
            Var<float> run_l = def(0.0f);
            Local<float> acc{256u};// decode validator: head_dim <= 256
            $for (d, lane, head_dim, 32u) { acc[d] = 0.0f; };
            $for (key, wid, nkeys, 4u) {
                const auto k_base = key * kv_stride + kvh * head_dim;
                Var<float> partial = def(0.0f);
                $for (d, lane, head_dim, 32u) {
                    partial += q.read(q_base + d) * k_cache.read(k_base + d);
                };
                const auto score = warp_active_sum(partial) * scale;
                const auto new_m = max(run_m, score);
                const auto corr = exp(run_m - new_m);
                const auto p = exp(score - new_m);
                run_l = run_l * corr + p;
                $for (d, lane, head_dim, 32u) {
                    acc[d] = acc[d] * corr + p * v_cache.read(k_base + d);
                };
                run_m = new_m;
            };
            // Publish per-warp partials: m/l in shared scalars, acc in a
            // shared [4, head_dim] tile (buffers declared at kernel scope).
            $if (lane == 0u) {
                sm[wid] = run_m;
                sl[wid] = run_l;
            };
            $for (d, lane, head_dim, 32u) {
                sacc[wid * 256u + d] = acc[d];
            };
            sync_block();
            // Merge in warp order, but distribute independent output channels over
            // all 128 threads. The baseline gave thread 0 a serial 256-channel
            // loop and recomputed the same four exponentials for every channel.
            Var<float> m_max = def(-3.0e38f);
            $for (w, 4u) {
                $if (sl.read(w) > 0.0f) { m_max = max(m_max, sm.read(w)); };
            };
            Var<float> l_tot = def(0.0f);
            $for (w, 4u) {
                $if (sl.read(w) > 0.0f) { l_tot += sl.read(w) * exp(sm.read(w) - m_max); };
            };
            const auto inv_l = 1.0f / l_tot;
            Local<float> corr{4u};
            $for (w, 4u) {
                corr[w] = 0.0f;
                $if (sl.read(w) > 0.0f) { corr[w] = exp(sm.read(w) - m_max); };
            };
            $for (d, thread_x(), head_dim, 128u) {
                Var<float> o = def(0.0f);
                $for (w, 4u) {
                    o += sacc.read(w * 256u + d) * corr[w];
                };
                out.write(q_base + d, o * inv_l * qwen_sigmoid(gate.read(q_base + d)));
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_attn_decode_dispatch(uint64_t q_heads) {
    // One 128-thread block per q_head.
    const auto threads = checked_product_u32(q_heads, 128u, "decode threads");
    if (!threads.ok()) { return Result<QwenLaunch1D>::failure(threads.status); }
    auto plan = plan_qwen_threads(*threads.value, 128u);
    if (!plan.ok()) { return Result<QwenLaunch1D>::failure(plan.status); }
    plan.value->work_items = q_heads;
    plan.value->warp_size = 32u;
    return plan;
}

[[nodiscard]] Status validate_qwen_attn_decode_inputs(uint64_t q_heads, uint64_t kv_heads,
                                                         uint64_t head_dim, uint64_t kv_len, uint64_t capacity) {
    if (q_heads == 0u || kv_heads == 0u || q_heads % kv_heads != 0u) {
        return err_invalid("decode GQA requires q_heads % kv_heads == 0");
    }
    if (head_dim == 0u || head_dim % 32u != 0u || head_dim > 256u) {
        return err_invalid("decode head_dim must be a multiple of 32 in [32,256] (shared tile bound)");
    }
    if (kv_len == 0u) { return err_invalid("decode requires kv_len >= 1 (append the current token first)"); }
    if (kv_len > capacity) { return err_invalid("decode kv_len exceeds cache capacity"); }
    const auto cache_n = checked_product_u32(capacity, kv_heads, "decode cache");
    const auto q_n = checked_product_u32(q_heads, head_dim, "decode queries");
    if (!cache_n.ok() || !q_n.ok() ||
        !checked_product_u32(*cache_n.value, head_dim, "decode cache").ok()) {
        return err_overflow("decode element count exceeds uint32");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// 8. Gated DeltaNet
// ---------------------------------------------------------------------------
[[nodiscard]] QwenGdnConvKernel make_qwen_gdn_conv_silu_kernel() noexcept {
    QwenGdnConvKernel kernel = [](BufferFloat src, BufferFloat weight, BufferFloat ring, BufferFloat out,
                                  Var<uint> tokens, Var<uint> channels, Var<uint> kernel_size,
                                  Var<uint> pos0) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_gdn_conv_silu");
        const auto c = qwen_linear_id();
        $if (c < channels & kernel_size >= 2u & tokens > 0u) {
            const auto rw = kernel_size - 1u;
            // Causal order inside the thread: a ring slot is overwritten
            // only AFTER every output that reads the older tap was computed.
            $for (t, tokens) {
                const auto p = pos0 + t;
                Var<float> acc = def(0.0f);
                $for (j, kernel_size) {
                    const auto signed_pos = cast<int>(p) - cast<int>(rw) + cast<int>(j);
                    Var<float> xv = def(0.0f);
                    $if (signed_pos >= 0) {
                        const auto abs_pos = cast<uint>(signed_pos);
                        $if (abs_pos < pos0) {
                            xv = ring.read(c * rw + (abs_pos % rw));
                        } $else {
                            xv = src.read((abs_pos - pos0) * channels + c);
                        };
                    };
                    // GGUF ssm_conv1d [kernel, channels] payload is CHANNEL-major:
                    // weight[c*kernel + j] is tap j of channel c.
                    acc += weight.read(c * kernel_size + j) * xv;
                };
                out.write(t * channels + c, qwen_silu(acc));
                ring.write(c * rw + (p % rw), src.read(t * channels + c));
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_conv_dispatch(uint64_t channels) {
    return plan_qwen_threads(channels);
}

[[nodiscard]] Status validate_qwen_gdn_conv_inputs(uint64_t tokens, uint64_t channels,
                                                      uint64_t kernel_size, uint64_t pos0) {
    if (channels == 0u || kernel_size < 2u) {
        return err_invalid("GDN conv geometry must be nonzero (kernel >= 2)");
    }
    if (!checked_product_u32(tokens, channels, "GDN conv src").ok() ||
        !checked_product_u32(channels, kernel_size, "GDN conv weight").ok() ||
        !checked_product_u32(channels, kernel_size - 1u, "GDN conv ring").ok()) {
        return err_overflow("GDN conv element count exceeds uint32");
    }
    if (pos0 > 0x7FFFFFFFull || tokens > 0x7FFFFFFFull - pos0) {
        return err_overflow("GDN conv absolute position exceeds int32 (signed tap math)");
    }
    return Status::success();
}

[[nodiscard]] QwenGdnGatesKernel make_qwen_gdn_gates_kernel() noexcept {
    QwenGdnGatesKernel kernel = [](BufferFloat a, BufferFloat b, BufferFloat ssm_a, BufferFloat dt_bias,
                                   BufferFloat beta_out, BufferFloat g_out,
                                   Var<uint> tokens, Var<uint> heads, Var<float> softplus_threshold) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_gdn_gates");
        const auto id = qwen_linear_id();
        $if (id < tokens * heads & heads > 0u) {
            const auto h = id % heads;
            const auto av = a.read(id) + dt_bias.read(h);
            Var<float> sp = def(0.0f);
            $if (av > softplus_threshold) {
                sp = av;
            } $else {
                sp = log(1.0f + exp(av));
            };
            beta_out.write(id, qwen_sigmoid(b.read(id)));
            // GGUF ssm_a is ALREADY -exp(A_log) (negative, folded offline by
            // the converter): g = ssm_a[h] * softplus(a + dt_bias[h]).
            g_out.write(id, ssm_a.read(h) * sp);
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_gates_dispatch(uint64_t tokens, uint64_t heads) {
    const auto n = checked_product_u32(tokens, heads, "gdn gates");
    if (!n.ok()) { return Result<QwenLaunch1D>::failure(n.status); }
    return plan_qwen_threads(*n.value);
}

[[nodiscard]] QwenGdnDecodeGatesKernel make_qwen_gdn_decode_gates_kernel(
    QwenWeightFormat format) noexcept {
    QwenGdnDecodeGatesKernel kernel = [format](
        BufferFloat x, BufferUInt alpha_w, BufferFloat alpha_scales,
        BufferUInt beta_w, BufferFloat beta_scales, BufferFloat ssm_a,
        BufferFloat dt_bias, BufferFloat beta_out, BufferFloat g_out,
        Var<uint> k, Var<uint> heads, Var<uint> group,
        Var<float> softplus_threshold) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name(format == QwenWeightFormat::bf16 ? "qwen_gdn_decode_gates_bf16"
                                                  : "qwen_gdn_decode_gates_fp8");
        Shared<float> lut{format == QwenWeightFormat::fp8_e4m3 ? 256u : 1u};
        if (format == QwenWeightFormat::fp8_e4m3) {
            $for (i, thread_x(), 256u, 128u) { lut[i] = qwen_fp8_e4m3_to_f32(i); };
            sync_block();
        }
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        $if (warp < heads * 2u & heads > 0u & k > 0u) {
            const auto is_beta = warp >= heads;
            Var<uint> h = warp;
            $if (is_beta) { h = warp - heads; };
            const auto dot = [&](const BufferVar<uint32_t> &w, const BufferVar<float> &scales,
                                 Var<uint> row) {
                const auto w_base = row * k;
                Var<uint> scale_base = def(0u);
                if (format == QwenWeightFormat::fp8_e4m3) {
                    scale_base = row * ((k + group - 1u) / group);
                }
                return qwen35_decode_dot(x, w, scales, lut, w_base, scale_base,
                                         k, group, lane, format);
            };
            Var<float> acc = def(0.0f);
            $if (is_beta) {
                acc = dot(beta_w, beta_scales, h);
            } $else {
                acc = dot(alpha_w, alpha_scales, h);
            };
            acc = warp_active_sum(acc);
            $if (lane == 0u) {
                $if (is_beta) {
                    beta_out.write(h, qwen_sigmoid(acc));
                } $else {
                    const auto av = acc + dt_bias.read(h);
                    Var<float> sp = def(0.0f);
                    $if (av > softplus_threshold) {
                        sp = av;
                    } $else {
                        sp = log(1.0f + exp(av));
                    };
                    g_out.write(h, ssm_a.read(h) * sp);
                };
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_decode_gates_dispatch(
    uint64_t heads, uint64_t k, QwenWeightFormat format, uint32_t group) {
    const auto rows = checked_product_u32(heads, 2u, "gdn decode gate rows");
    if (!rows.ok()) { return Result<QwenLaunch1D>::failure(rows.status); }
    if (k == 0u || (format == QwenWeightFormat::bf16 && k % 256u != 0u) ||
        (format == QwenWeightFormat::fp8_e4m3 &&
         (group == 0u || group % 4u != 0u || k % 512u != 0u))) {
        return Result<QwenLaunch1D>::failure(err_invalid("gdn decode gate packed geometry mismatch"));
    }
    const auto wn = checked_product_u32(heads, k, "gdn decode gate weights");
    if (!wn.ok()) { return Result<QwenLaunch1D>::failure(wn.status); }
    return plan_warp_rows(*rows.value, 128u);
}

[[nodiscard]] QwenGdnQkNormKernel make_qwen_gdn_qk_l2norm_kernel() noexcept {
    QwenGdnQkNormKernel kernel = [](BufferFloat conv_out, BufferFloat q_out, BufferFloat k_out,
                                    Var<uint> tokens, Var<uint> qk_heads, Var<uint> head_dim,
                                    Var<uint> v_channels, Var<float> eps, Var<float> q_scale, Var<uint> rows) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name("qwen_gdn_qk_l2norm");
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        // rows = tokens*qk_heads; each warp does the Q row AND the K row of
        // one (token, qk_head). Block-uniform guard.
        $if (warp < rows & head_dim > 0u & tokens > 0u) {
            const auto t = warp / qk_heads;
            const auto h = warp % qk_heads;
            const auto stride = 2u * qk_heads * head_dim + v_channels;
            const auto q_src = t * stride + h * head_dim;
            const auto k_src = q_src + qk_heads * head_dim;
            const auto dst = (t * qk_heads + h) * head_dim;
            Var<float> q_ss = def(0.0f);
            Var<float> k_ss = def(0.0f);
            $for (i, lane, head_dim, 32u) {
                const auto qv = conv_out.read(q_src + i);
                const auto kv = conv_out.read(k_src + i);
                q_ss += qv * qv;
                k_ss += kv * kv;
            };
            q_ss = warp_active_sum(q_ss);
            k_ss = warp_active_sum(k_ss);
            // FLA/HF l2norm: x * rsqrt(sum(x*x) + eps) -- eps INSIDE the
            // rsqrt (near-zero rows stay near zero, never blown up to 1/eps).
            const auto q_inv = rsqrt(q_ss + eps);
            const auto k_inv = rsqrt(k_ss + eps);
            $for (i, lane, head_dim, 32u) {
                q_out.write(dst + i, conv_out.read(q_src + i) * q_inv * q_scale);
                k_out.write(dst + i, conv_out.read(k_src + i) * k_inv);
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_qk_norm_dispatch(uint64_t tokens, uint64_t qk_heads) {
    const auto rows = checked_product_u32(tokens, qk_heads, "gdn qk rows");
    if (!rows.ok()) { return Result<QwenLaunch1D>::failure(rows.status); }
    return plan_warp_rows(*rows.value, 128u);
}

[[nodiscard]] QwenGdnRecurrentKernel make_qwen_gdn_recurrent_kernel() noexcept {
    QwenGdnRecurrentKernel kernel = [](BufferFloat q, BufferFloat k, BufferFloat conv_out,
                                       BufferFloat beta, BufferFloat g, BufferFloat state, BufferFloat out,
                                       Var<uint> tokens, Var<uint> v_heads, Var<uint> head_dim,
                                       Var<uint> qk_heads, Var<uint> v_offset) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name("qwen_gdn_recurrent");
        // Shared staging buffers MUST be declared at kernel scope (not inside
        // a divergent $if) so every block allocates them uniformly.
        Shared<float> sk{512u};
        Shared<float> sq{512u};
        // Block index from the linear thread id (works for flattened plans).
        const auto h = qwen_linear_id() / 128u;
        // Block-uniform guard. One lane owns one state column, so head_dim
        // must not exceed 128 lanes across the four 32-lane warps.
        $if (h < v_heads & head_dim > 0u & head_dim <= 128u & (head_dim % 32u) == 0u & tokens > 0u) {
            const auto lane = warp_lane_id();
            const auto wid = thread_x() / 32u;
            const auto cols_per_warp = head_dim / 4u;
            // This thread owns value column v = wid*cols_per_warp + lane
            // (lanes >= cols_per_warp own nothing; all their reads/writes are
            // guarded and their local state column stays zero).
            const auto v_col = wid * cols_per_warp + lane;
            const auto owns_col = lane < cols_per_warp;
            // GGUF V-side tensors are TILED (converter reorders grouped HF
            // heads [K0_v0,K0_v1,K1_v0,...] -> [K0_v0,K1_v0,...,K0_v1,...]):
            // V head h pairs with QK head h % qk_heads (ggml tiled broadcast).
            const auto qkh = h % qk_heads;
            const auto s_base = h * head_dim * head_dim;
            const auto v_src_base = v_offset + h * head_dim;
            const auto c_stride = v_offset + v_heads * head_dim;
            // Thread-local state column S[:, v_col] (head_dim floats; fixed
            // 512-entry array, only [0,head_dim) used). Non-owning lanes keep
            // zeros so their reads never touch the state buffer.
            Local<float> col{512u};
            $for (ki, head_dim) { col[ki] = 0.0f; };
            $if (owns_col) {
                $for (ki, head_dim) {
                    col[ki] = state.read(s_base + ki * head_dim + v_col);
                };
            };
            // Staged k/q vectors (shared, per token; declared at kernel scope).
            $for (t, tokens) {
                const auto th = t * v_heads + h;
                const auto alpha = exp(g.read(th));
                const auto bt = beta.read(th);
                const auto qk_row = (t * qk_heads + qkh) * head_dim;
                // Stage k and q for this token (all 128 threads cooperate).
                $for (i, thread_x(), head_dim, 128u) {
                    sk[i] = k.read(qk_row + i);
                    sq[i] = q.read(qk_row + i);
                };
                sync_block();
                // Decay the column and compute err_v = v[v] - sum_k S[k,v]
                // k[k]. Each lane owns a DIFFERENT value column, so the dot
                // is a per-lane serial reduction over the key axis (NOT a
                // warp collective -- that would mix columns). The k vector
                // is shared-broadcast, so all lanes read the same sk[ki].
                Var<float> dot = def(0.0f);
                $for (ki, head_dim) {
                    col[ki] = col[ki] * alpha;
                    dot += col[ki] * sk[ki];
                };
                // Guard the READ itself: non-owning lanes must not touch
                // conv_out (their v_col can exceed head_dim).
                Var<float> vv = def(0.0f);
                $if (owns_col) {
                    vv = conv_out.read(t * c_stride + v_src_base + v_col);
                };
                const auto err = vv - dot;
                // Rank-1 update: S[k,v] += beta * k[k] * err.
                $for (ki, head_dim) {
                    col[ki] = col[ki] + bt * sk[ki] * err;
                };
                // o[v] = sum_k S[k,v] q[k].
                Var<float> o = def(0.0f);
                $for (ki, head_dim) { o += col[ki] * sq[ki]; };
                $if (owns_col) {
                    out.write((t * v_heads + h) * head_dim + v_col, o);
                };
                sync_block();
            };
            // Write the state column back once.
            $if (owns_col) {
                $for (ki, head_dim) {
                    state.write(s_base + ki * head_dim + v_col, col[ki]);
                };
            };
        };
    };
    return kernel;
}

[[nodiscard]] QwenGdnRecurrentKernel make_qwen35_gdn_decode4_kernel() noexcept {
    QwenGdnRecurrentKernel kernel = [](BufferFloat q, BufferFloat k, BufferFloat conv_out,
                                       BufferFloat beta, BufferFloat g, BufferFloat state,
                                       BufferFloat out, Var<uint> tokens, Var<uint> v_heads,
                                       Var<uint> head_dim, Var<uint> qk_heads,
                                       Var<uint> v_offset) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name("qwen35_gdn_decode4");
        Shared<float> sk{128u};
        Shared<float> sq{128u};
        const auto work = qwen_linear_id() / 128u;
        $if (work < v_heads * 4u & tokens == 1u & head_dim == 128u) {
            const auto h = work / 4u;
            const auto col_group = work % 4u;
            const auto lane = warp_lane_id();
            const auto wid = thread_x() / 32u;
            const auto key_part = lane / 8u;
            const auto col_in_group = wid * 8u + (lane & 7u);
            const auto v_col = col_group * 32u + col_in_group;
            const auto key_begin = key_part * 32u;
            const auto qkh = h % qk_heads;
            const auto qk_row = qkh * 128u;
            $for (i, thread_x(), 128u, 128u) {
                sk[i] = k.read(qk_row + i);
                sq[i] = q.read(qk_row + i);
            };
            sync_block();
            const auto s_base = h * 128u * 128u + v_col;
            Local<float> col{32u};
            $for (j, 32u) {
                col[j] = state.read(s_base + (key_begin + j) * 128u);
            };
            const auto alpha = exp(g.read(h));
            const auto bt = beta.read(h);
            Var<float> dot = def(0.0f);
            $for (j, 32u) {
                col[j] = col[j] * alpha;
                dot += col[j] * sk[key_begin + j];
            };
            // Four key slices are lanes {c,c+8,c+16,c+24} in each warp.
            // Every lane participates in both shuffles and receives the sum.
            dot += warp_read_lane(dot, lane ^ 8u);
            dot += warp_read_lane(dot, lane ^ 16u);
            const auto vv = conv_out.read(v_offset + h * 128u + v_col);
            const auto err = vv - dot;
            Var<float> o = def(0.0f);
            $for (j, 32u) {
                col[j] = col[j] + bt * sk[key_begin + j] * err;
                o += col[j] * sq[key_begin + j];
            };
            o += warp_read_lane(o, lane ^ 8u);
            o += warp_read_lane(o, lane ^ 16u);
            $if (key_part == 0u) {
                out.write(h * 128u + v_col, o);
            };
            $for (j, 32u) {
                state.write(s_base + (key_begin + j) * 128u, col[j]);
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen35_gdn_decode4_dispatch(uint64_t v_heads) {
    const auto blocks = checked_product_u32(v_heads, 4u, "gdn decode4 blocks");
    if (!blocks.ok()) { return Result<QwenLaunch1D>::failure(blocks.status); }
    const auto threads = checked_product_u32(*blocks.value, 128u, "gdn decode4 threads");
    if (!threads.ok()) { return Result<QwenLaunch1D>::failure(threads.status); }
    auto plan = plan_qwen_threads(*threads.value, 128u);
    if (!plan.ok()) { return plan; }
    plan.value->work_items = *blocks.value;
    plan.value->warp_size = 32u;
    return plan;
}

[[nodiscard]] Status validate_qwen35_gdn_decode4_inputs(
    uint64_t tokens, uint64_t v_heads, uint64_t qk_heads, uint64_t head_dim) {
    if (tokens != 1u || head_dim != 128u) {
        return err_invalid("qwen35 GDN decode4 requires T=1 and head_dim=128");
    }
    return validate_qwen_gdn_recurrent_inputs(tokens, v_heads, qk_heads, head_dim);
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_recurrent_dispatch(uint64_t v_heads) {
    const auto threads = checked_product_u32(v_heads, 128u, "gdn recurrent threads");
    if (!threads.ok()) { return Result<QwenLaunch1D>::failure(threads.status); }
    auto plan = plan_qwen_threads(*threads.value, 128u);
    if (!plan.ok()) { return Result<QwenLaunch1D>::failure(plan.status); }
    plan.value->work_items = v_heads;
    plan.value->warp_size = 32u;
    return plan;
}

[[nodiscard]] Status validate_qwen_gdn_recurrent_inputs(uint64_t tokens, uint64_t v_heads,
                                                           uint64_t qk_heads, uint64_t head_dim) {
    if (v_heads == 0u || qk_heads == 0u || v_heads % qk_heads != 0u) {
        return err_invalid("GDN requires v_heads % qk_heads == 0");
    }
    if (head_dim == 0u || head_dim % 32u != 0u || head_dim > 128u) {
        return err_invalid("GDN head_dim must be a multiple of 32 in [32,128] (one state column per lane)");
    }
    if (tokens == 0u) { return Status::success(); }
    const auto v_n = checked_product_u32(tokens, v_heads, "GDN v rows");
    const auto s_n = checked_product_u32(v_heads, head_dim, "GDN state");
    const auto q_n = checked_product_u32(tokens, qk_heads, "GDN qk rows");
    if (!v_n.ok() || !s_n.ok() || !q_n.ok() ||
        !checked_product_u32(*v_n.value, head_dim, "GDN v elements").ok() ||
        !checked_product_u32(*s_n.value, head_dim, "GDN state elements").ok() ||
        !checked_product_u32(*q_n.value, head_dim, "GDN qk elements").ok()) {
        return err_overflow("GDN recurrent element count exceeds uint32");
    }
    return Status::success();
}

[[nodiscard]] uint64_t qwen_gdn_recurrent_shared_bytes(uint64_t head_dim) noexcept {
    // sk/sq are statically sized at 512 floats each (max head_dim).
    (void)head_dim;
    return 2u * 512u * sizeof(float);
}

[[nodiscard]] Status check_qwen_gdn_shared_budget(uint64_t head_dim, uint64_t backend_shared_bytes) {
    if (qwen_gdn_recurrent_shared_bytes(head_dim) > backend_shared_bytes) {
        return Status::error(StatusCode::unsupported,
                                "GDN recurrent kernel needs 4 KiB shared; backend limit is smaller");
    }
    return Status::success();
}

[[nodiscard]] QwenGdnChunkKernel make_qwen_gdn_chunk_kernel() noexcept {
    QwenGdnChunkKernel kernel = [](BufferFloat q, BufferFloat k, BufferFloat conv_out,
                                   BufferFloat beta, BufferFloat g, BufferFloat state,
                                   BufferFloat w, BufferFloat v_new, BufferFloat out,
                                   Var<uint> tokens, Var<uint> v_heads, Var<uint> head_dim,
                                   Var<uint> qk_heads, Var<uint> v_offset) noexcept {
        set_block_size(128u, 1u, 1u);
        set_name("qwen_gdn_wy_ut_chunk64");
        // A holds the strict-lower KKT matrix, then its unit-lower UT inverse,
        // and finally the causal decay-scaled QK matrix. Reusing this 16 KiB
        // tile keeps the whole kernel below a 32 KiB shared-memory budget.
        Shared<float> a{qwen_gdn_chunk_size * qwen_gdn_chunk_size};
        Shared<float> sg{qwen_gdn_chunk_size};// chunk-local cumulative log decay
        Shared<float> sb{qwen_gdn_chunk_size};
        Shared<float> row{qwen_gdn_chunk_size};// UT forward-substitution staging
        const auto h = qwen_linear_id() / 128u;
        const auto lane = thread_x();
        $if (h < v_heads & head_dim > 0u & head_dim <= 128u & qk_heads > 0u & tokens > 0u) {
            const auto qkh = h % qk_heads;
            const auto s_base = h * head_dim * head_dim;
            const auto v_base = v_offset + h * head_dim;
            const auto c_stride = v_offset + v_heads * head_dim;
            const auto owns_col = lane < head_dim;
            // The lane-owned state column is the chunk-start H_c. Keeping it
            // in registers avoids materializing NT copies of a 128x128 state.
            Local<float> col{128u};
            $for (ki, head_dim) { col[ki] = 0.0f; };
            $if (owns_col) {
                $for (ki, head_dim) { col[ki] = state.read(s_base + ki * head_dim + lane); };
            };
            $for (chunk, (tokens + qwen_gdn_chunk_size - 1u) / qwen_gdn_chunk_size) {
                const auto t0 = chunk * qwen_gdn_chunk_size;
                const auto count = min(qwen_gdn_chunk_size, tokens - t0);
                // Chunk-local FP32 cumsum. No global Gamma and no exp(-Gamma)
                // is formed, so long prompts cannot overflow transformed K.
                $if (lane == 0u) {
                    Var<float> sum = def(0.0f);
                    $for (i, count) {
                        const auto th = (t0 + i) * v_heads + h;
                        sum += g.read(th);
                        sg[i] = sum;
                        sb[i] = beta.read(th);
                    };
                };
                sync_block();

                // A[i,j] = beta_i (k_i.k_j) exp(G_i-G_j), i>j.
                $for (idx, lane, qwen_gdn_chunk_size * qwen_gdn_chunk_size, 128u) {
                    const auto i = idx / qwen_gdn_chunk_size;
                    const auto j = idx % qwen_gdn_chunk_size;
                    Var<float> value = def(0.0f);
                    $if (i > j & i < count & j < count) {
                        const auto ki = ((t0 + i) * qk_heads + qkh) * head_dim;
                        const auto kj = ((t0 + j) * qk_heads + qkh) * head_dim;
                        Var<float> dot = def(0.0f);
                        $for (d, head_dim) { dot += k.read(ki + d) * k.read(kj + d); };
                        value = sb[i] * dot * exp(sg[i] - sg[j]);
                    };
                    a[idx] = value;
                };
                sync_block();

                // UT transform: A <- (I + strict_lower(A))^-1. Row i is
                // staged before any element overwrites the still-needed L_i,*
                // source row; preceding rows already contain inverse rows.
                $for (i, count) {
                    $for (j, lane, i + 1u, 128u) {
                        Var<float> value = def(ite(i == j, 1.0f, 0.0f));
                        $for (l, j, i) { value -= a[i * qwen_gdn_chunk_size + l] *
                                                  a[l * qwen_gdn_chunk_size + j]; };
                        row[j] = value;
                    };
                    sync_block();
                    $for (j, lane, i + 1u, 128u) {
                        a[i * qwen_gdn_chunk_size + j] = row[j];
                    };
                    sync_block();
                };

                // W = A_inv * diag(beta * exp(G)) * K. Each chunk/head has
                // private scratch, so subsequent block-synchronized reads are
                // race-free without a second WY kernel launch.
                $for (idx, lane, count * head_dim, 128u) {
                    const auto i = idx / head_dim;
                    const auto d = idx % head_dim;
                    Var<float> value = def(0.0f);
                    $for (j, i + 1u) {
                        const auto kj = ((t0 + j) * qk_heads + qkh) * head_dim + d;
                        value += a[i * qwen_gdn_chunk_size + j] * sb[j] * exp(sg[j]) * k.read(kj);
                    };
                    w.write(((t0 + i) * v_heads + h) * head_dim + d, value);
                };
                sync_block();

                // v_new = U - W H_c, U=A_inv*diag(beta)*V. This remains
                // UNSCALED for the output equation; the state update below
                // applies exp(G_last-G_i) to its own copy of each value.
                $for (i, count) {
                    $if (owns_col) {
                        Var<float> u = def(0.0f);
                        $for (j, i + 1u) {
                            const auto vv = conv_out.read((t0 + j) * c_stride + v_base + lane);
                            u += a[i * qwen_gdn_chunk_size + j] * sb[j] * vv;
                        };
                        Var<float> correction = def(0.0f);
                        $for (d, head_dim) {
                            correction += w.read(((t0 + i) * v_heads + h) * head_dim + d) * col[d];
                        };
                        v_new.write(((t0 + i) * v_heads + h) * head_dim + lane, u - correction);
                    };
                };
                sync_block();

                // The UT inverse is dead now. Reuse A for the causal output
                // coefficient QK^T * exp(G_i-G_j), including the diagonal.
                $for (idx, lane, qwen_gdn_chunk_size * qwen_gdn_chunk_size, 128u) {
                    const auto i = idx / qwen_gdn_chunk_size;
                    const auto j = idx % qwen_gdn_chunk_size;
                    Var<float> value = def(0.0f);
                    $if (i >= j & i < count & j < count) {
                        const auto qi = ((t0 + i) * qk_heads + qkh) * head_dim;
                        const auto kj = ((t0 + j) * qk_heads + qkh) * head_dim;
                        Var<float> dot = def(0.0f);
                        $for (d, head_dim) { dot += q.read(qi + d) * k.read(kj + d); };
                        value = dot * exp(sg[i] - sg[j]);
                    };
                    a[idx] = value;
                };
                sync_block();

                // o_i = exp(G_i) q_i H_c + sum_{j<=i} A_out[i,j] v_new[j].
                $for (i, count) {
                    $if (owns_col) {
                        const auto qi = ((t0 + i) * qk_heads + qkh) * head_dim;
                        Var<float> cross = def(0.0f);
                        $for (d, head_dim) { cross += q.read(qi + d) * col[d]; };
                        cross *= exp(sg[i]);
                        Var<float> intra = def(0.0f);
                        $for (j, i + 1u) {
                            intra += a[i * qwen_gdn_chunk_size + j] *
                                     v_new.read(((t0 + j) * v_heads + h) * head_dim + lane);
                        };
                        out.write(((t0 + i) * v_heads + h) * head_dim + lane, cross + intra);
                    };
                };

                // H_{c+1} = exp(G_last) H_c + K^T diag(exp(G_last-G_i)) v_new.
                // Each col[d] reads only its own previous value, so in-place
                // register updates cannot mix chunk-start and chunk-end states.
                const auto g_last = sg[count - 1u];
                $if (owns_col) {
                    $for (d, head_dim) {
                        Var<float> value = def(col[d] * exp(g_last));
                        $for (i, count) {
                            const auto ki = ((t0 + i) * qk_heads + qkh) * head_dim + d;
                            const auto vn = v_new.read(((t0 + i) * v_heads + h) * head_dim + lane);
                            value += k.read(ki) * vn * exp(g_last - sg[i]);
                        };
                        col[d] = value;
                    };
                };
                // No thread may overwrite A/W/v_new for the next chunk until
                // every output and state read from this chunk has completed.
                sync_block();
            };
            $if (owns_col) {
                $for (d, head_dim) { state.write(s_base + d * head_dim + lane, col[d]); };
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_chunk_dispatch(uint64_t v_heads) {
    return plan_qwen_gdn_recurrent_dispatch(v_heads);
}

[[nodiscard]] Status validate_qwen_gdn_chunk_inputs(uint64_t tokens, uint64_t v_heads,
                                                    uint64_t qk_heads, uint64_t head_dim) {
    if (auto s = validate_qwen_gdn_recurrent_inputs(tokens, v_heads, qk_heads, head_dim); !s.ok()) { return s; }
    const auto rows = checked_product_u32(tokens, v_heads, "GDN chunk scratch rows");
    if (!rows.ok() || !checked_product_u32(*rows.value, head_dim, "GDN chunk scratch").ok()) {
        return err_overflow("GDN chunk scratch element count exceeds uint32");
    }
    return Status::success();
}

[[nodiscard]] uint64_t qwen_gdn_chunk_shared_bytes() noexcept {
    return (qwen_gdn_chunk_size * qwen_gdn_chunk_size + 3u * qwen_gdn_chunk_size) * sizeof(float);
}

[[nodiscard]] Status dsl_ops_chunk_prefill_status() noexcept {
    return Status::success();
}

[[nodiscard]] QwenGdnGatedNormKernel make_qwen_gdn_gated_norm_kernel() noexcept {
    QwenGdnGatedNormKernel kernel = [](BufferFloat o, BufferFloat z, BufferFloat norm_w, BufferFloat out,
                                       Var<uint> tokens, Var<uint> heads, Var<uint> head_dim,
                                       Var<float> eps) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name("qwen_gdn_gated_norm");
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        $if (warp < tokens * heads & head_dim > 0u) {
            const auto base = warp * head_dim;
            Var<float> sumsq = def(0.0f);
            $for (i, lane, head_dim, 32u) {
                const auto v = o.read(base + i);
                sumsq += v * v;
            };
            sumsq = warp_active_sum(sumsq);
            const auto inv = rsqrt(sumsq / cast<float>(head_dim) + eps);
            // DIRECT weight (not 1+w); norm BEFORE the SiLU(z) gate.
            $for (i, lane, head_dim, 32u) {
                out.write(base + i, o.read(base + i) * inv * norm_w.read(i) * qwen_silu(z.read(base + i)));
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_gated_norm_dispatch(uint64_t tokens, uint64_t heads) {
    const auto rows = checked_product_u32(tokens, heads, "gdn gated norm rows");
    if (!rows.ok()) { return Result<QwenLaunch1D>::failure(rows.status); }
    return plan_warp_rows(*rows.value, 128u);
}

// ---------------------------------------------------------------------------
// 9. CPU oracles (double accumulation; independent scalar code paths)
// ---------------------------------------------------------------------------
namespace {

[[nodiscard]] double ref_silu(double x) { return x / (1.0 + std::exp(-x)); }
[[nodiscard]] double ref_sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

[[nodiscard]] double ref_decode_weight(QwenWeightFormat format, luisa::span<const uint32_t> words,
                                       luisa::span<const float> scales, uint64_t k, uint32_t group,
                                       uint64_t index) {
    return static_cast<double>(host_decode_packed(format, words, scales, k, group, index));
}

}// namespace

[[nodiscard]] Status ref_qwen_embedding(QwenWeightFormat format, luisa::span<const uint32_t> ids,
                                           luisa::span<const uint32_t> table, luisa::span<const float> scales,
                                           uint64_t hidden, uint64_t vocab, uint32_t group,
                                           luisa::vector<float> &out) {
    const auto v = validate_qwen_embedding_inputs(ids, hidden, vocab);
    if (!v.ok()) { return v; }
    const auto wc = packed_matrix_words(format, hidden, vocab);
    if (!wc.ok()) { return wc.status; }
    if (table.size() != *wc.value) { return err_invalid("embedding oracle: table word count mismatch"); }
    if (format == QwenWeightFormat::fp8_e4m3) {
        const auto sc = fp8_scale_elements(hidden, vocab, group);
        if (!sc.ok()) { return sc.status; }
        if (scales.size() != *sc.value) { return err_invalid("embedding oracle: scale count mismatch"); }
    }
    out.resize(ids.size() * static_cast<size_t>(hidden));
    for (size_t t = 0; t < ids.size(); t++) {
        for (uint64_t c = 0; c < hidden; c++) {
            out[t * static_cast<size_t>(hidden) + static_cast<size_t>(c)] =
                host_decode_packed(format, table, scales, hidden, group,
                                   static_cast<uint64_t>(ids[t]) * hidden + c);
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_rmsnorm(luisa::span<const float> x, luisa::span<const float> w,
                                         luisa::span<const float> residual, bool has_residual,
                                         uint32_t rows, uint32_t dim, float eps,
                                         luisa::vector<float> &y, luisa::vector<float> &sum_out) {
    if (dim == 0u) { return err_invalid("rmsnorm dim must be nonzero"); }
    const uint64_t n = static_cast<uint64_t>(rows) * dim;
    if (x.size() != n || w.size() != dim) { return err_invalid("rmsnorm oracle size mismatch"); }
    if (has_residual && residual.size() != n) { return err_invalid("rmsnorm residual size mismatch"); }
    y.resize(static_cast<size_t>(n));
    if (has_residual) { sum_out.resize(static_cast<size_t>(n)); }
    for (uint32_t r = 0; r < rows; r++) {
        const uint64_t base = static_cast<uint64_t>(r) * dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < dim; i++) {
            double v = x[base + i];
            if (has_residual) {
                v += residual[base + i];
                sum_out[base + i] = static_cast<float>(v);
            }
            ss += v * v;
        }
        const double inv = 1.0 / std::sqrt(ss / dim + eps);
        for (uint32_t i = 0; i < dim; i++) {
            const double v = has_residual ? static_cast<double>(sum_out[base + i]) : static_cast<double>(x[base + i]);
            // GGUF weights are pre-added (1+w folded offline); direct multiply.
            y[base + i] = static_cast<float>(v * inv * static_cast<double>(w[i]));
        }
    }
    return Status::success();
}

// Independent oracle for QwenNormEpilogue::snapshot: sum_out is a BIT-EXACT
// copy of the original x, y is the plain direct-pre-added-weight RMSNorm of
// that copy. Deliberately a separate code path from ref_qwen_rmsnorm (plain
// branch) so a snapshot regression cannot hide behind a shared helper bug.
[[nodiscard]] Status ref_qwen_rmsnorm_snapshot(luisa::span<const float> x, luisa::span<const float> w,
                                               uint32_t rows, uint32_t dim, float eps,
                                               luisa::vector<float> &y, luisa::vector<float> &sum_out) {
    if (dim == 0u) { return err_invalid("rmsnorm snapshot dim must be nonzero"); }
    const uint64_t n = static_cast<uint64_t>(rows) * dim;
    if (x.size() != n || w.size() != dim) { return err_invalid("rmsnorm snapshot oracle size mismatch"); }
    y.resize(static_cast<size_t>(n));
    sum_out.assign(x.begin(), x.end());// exact copy, no arithmetic
    for (uint32_t r = 0; r < rows; r++) {
        const uint64_t base = static_cast<uint64_t>(r) * dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < dim; i++) {
            const double v = sum_out[base + i];// normalize FROM the snapshot
            ss += v * v;
        }
        const double inv = 1.0 / std::sqrt(ss / dim + eps);
        for (uint32_t i = 0; i < dim; i++) {
            // GGUF weights are pre-added (1+w folded offline); direct multiply.
            y[base + i] = static_cast<float>(static_cast<double>(sum_out[base + i]) * inv *
                                             static_cast<double>(w[i]));
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_matmul(QwenWeightFormat format, luisa::span<const float> x,
                                        luisa::span<const uint32_t> w, luisa::span<const float> scales,
                                        luisa::span<const float> residual, bool has_residual,
                                        uint32_t m, uint32_t n, uint32_t k, uint32_t group,
                                        luisa::vector<float> &y) {
    if (k == 0u) { return err_invalid("matmul K must be nonzero"); }
    const auto wc = packed_matrix_words(format, k, n);
    if (!wc.ok()) { return wc.status; }
    if (w.size() != *wc.value) { return err_invalid("matmul oracle weight word count mismatch"); }
    if (x.size() != static_cast<uint64_t>(m) * k) { return err_invalid("matmul oracle x size mismatch"); }
    if (has_residual && residual.size() != static_cast<uint64_t>(m) * n) {
        return err_invalid("matmul oracle residual size mismatch");
    }
    if (format == QwenWeightFormat::fp8_e4m3) {
        const auto sc = fp8_scale_elements(k, n, group);
        if (!sc.ok()) { return sc.status; }
        if (scales.size() != *sc.value) { return err_invalid("matmul oracle scale count mismatch"); }
    }
    y.resize(static_cast<size_t>(m) * n);
    for (uint32_t mi = 0; mi < m; mi++) {
        for (uint32_t ni = 0; ni < n; ni++) {
            double acc = 0.0;
            for (uint32_t ki = 0; ki < k; ki++) {
                acc += static_cast<double>(x[static_cast<uint64_t>(mi) * k + ki]) *
                       ref_decode_weight(format, w, scales, k, group, static_cast<uint64_t>(ni) * k + ki);
            }
            if (has_residual) { acc += residual[static_cast<uint64_t>(mi) * n + ni]; }
            y[static_cast<size_t>(mi) * n + ni] = static_cast<float>(acc);
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_swiglu(luisa::span<const float> gate, luisa::span<const float> up,
                                        luisa::vector<float> &out) {
    if (gate.size() != up.size()) { return err_invalid("swiglu gate/up size mismatch"); }
    out.resize(gate.size());
    for (size_t i = 0; i < gate.size(); i++) {
        out[i] = static_cast<float>(ref_silu(gate[i]) * static_cast<double>(up[i]));
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_swiglu_fused(luisa::span<const float> proj, uint32_t m, uint32_t f,
                                              luisa::vector<float> &out) {
    if (proj.size() != static_cast<uint64_t>(m) * 2u * f) { return err_invalid("fused swiglu size mismatch"); }
    out.resize(static_cast<size_t>(m) * f);
    for (uint32_t mi = 0; mi < m; mi++) {
        for (uint32_t fi = 0; fi < f; fi++) {
            const uint64_t base = static_cast<uint64_t>(mi) * 2u * f;
            out[static_cast<size_t>(mi) * f + fi] =
                static_cast<float>(ref_silu(proj[base + fi]) * static_cast<double>(proj[base + f + fi]));
        }
    }
    return Status::success();
}

namespace {

// Shared norm+rope row transform for the prepare oracle.
void ref_norm_rope_row(luisa::span<const float> src, luisa::span<const float> nw,
                       float pos, uint32_t head_dim, uint32_t rope_dim, float theta, float eps,
                       luisa::span<float> dst) {
    double ss = 0.0;
    for (uint32_t i = 0; i < head_dim; i++) { ss += static_cast<double>(src[i]) * src[i]; }
    const double inv = 1.0 / std::sqrt(ss / head_dim + eps);
    // Norm first into dst (direct pre-added weight), then NeoX rotate-half
    // on the prefix: pairs (p, p+half), half = rope_dim/2.
    for (uint32_t i = 0; i < head_dim; i++) {
        dst[i] = static_cast<float>(src[i] * inv * static_cast<double>(nw[i]));
    }
    const uint32_t half = rope_dim / 2u;
    for (uint32_t p = 0; p < half; p++) {
        const double freq = std::pow(static_cast<double>(theta), -static_cast<double>(2u * p) / rope_dim);
        const double angle = static_cast<double>(pos) * freq;
        const double c = std::cos(angle), s = std::sin(angle);
        const double x0 = dst[p], x1 = dst[p + half];
        dst[p] = static_cast<float>(x0 * c - x1 * s);
        dst[p + half] = static_cast<float>(x0 * s + x1 * c);
    }
}

}// namespace

[[nodiscard]] Status ref_qwen_attn_prepare(luisa::span<const float> q_proj, luisa::span<const float> k_proj,
                                              luisa::span<const float> v_proj, luisa::span<const float> q_norm_w,
                                              luisa::span<const float> k_norm_w, luisa::span<const int32_t> positions,
                                              uint32_t tokens, const QwenDslConfig &cfg, uint32_t pos0,
                                              bool append_kv, luisa::span<float> k_cache, luisa::span<float> v_cache,
                                              luisa::vector<float> &q, luisa::vector<float> &gate,
                                              luisa::vector<float> &k, luisa::vector<float> &v) {
    const auto val = validate_qwen_attn_prepare_inputs(tokens, cfg, pos0,
                                                       append_kv ? k_cache.size() / cfg.attn_kv_n() : 0u, append_kv);
    if (!val.ok()) { return val; }
    const uint32_t hd = cfg.attn_head_dim;
    const uint64_t qn = static_cast<uint64_t>(tokens) * cfg.attn_q_proj_n();
    const uint64_t kvn = static_cast<uint64_t>(tokens) * cfg.attn_kv_n();
    if (q_proj.size() != qn || k_proj.size() != kvn || v_proj.size() != kvn ||
        q_norm_w.size() != hd || k_norm_w.size() != hd || positions.size() != tokens) {
        return err_invalid("attn prepare oracle size mismatch");
    }
    if (append_kv) {
        const uint64_t cap = k_cache.size() / cfg.attn_kv_n();
        if (k_cache.size() != cap * cfg.attn_kv_n() || v_cache.size() != k_cache.size()) {
            return err_invalid("attn prepare oracle cache size mismatch");
        }
    }
    q.resize(static_cast<size_t>(tokens) * cfg.q_heads * hd);
    gate.resize(q.size());
    k.resize(static_cast<size_t>(kvn));
    v.resize(static_cast<size_t>(kvn));
    for (uint32_t t = 0; t < tokens; t++) {
        const float pos = static_cast<float>(positions[t]);
        for (uint32_t h = 0; h < cfg.q_heads; h++) {
            const uint64_t src = (static_cast<uint64_t>(t) * cfg.q_heads + h) * 2u * hd;
            const uint64_t dst = (static_cast<uint64_t>(t) * cfg.q_heads + h) * hd;
            ref_norm_rope_row(q_proj.subspan(src, hd), q_norm_w, pos, hd, cfg.rope_dim,
                              cfg.rope_theta, cfg.norm_eps, luisa::span<float>{q}.subspan(dst, hd));
            for (uint32_t i = 0; i < hd; i++) { gate[dst + i] = q_proj[src + hd + i]; }
        }
        for (uint32_t h = 0; h < cfg.kv_heads; h++) {
            const uint64_t off = (static_cast<uint64_t>(t) * cfg.kv_heads + h) * hd;
            ref_norm_rope_row(k_proj.subspan(off, hd), k_norm_w, pos, hd, cfg.rope_dim,
                              cfg.rope_theta, cfg.norm_eps, luisa::span<float>{k}.subspan(off, hd));
            for (uint32_t i = 0; i < hd; i++) { v[off + i] = v_proj[off + i]; }
            if (append_kv) {
                const uint64_t slot = (static_cast<uint64_t>(pos0) + t) * cfg.attn_kv_n() + h * hd;
                for (uint32_t i = 0; i < hd; i++) {
                    k_cache[slot + i] = k[off + i];
                    v_cache[slot + i] = v[off + i];
                }
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_rope_partial(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                              uint32_t tokens, uint32_t heads, uint32_t head_dim,
                                              uint32_t rope_dim, float theta, luisa::vector<float> &out) {
    if (rope_dim % 2u != 0u || rope_dim > head_dim) { return err_invalid("rope_dim must be even and <= head_dim"); }
    const uint64_t n = static_cast<uint64_t>(tokens) * heads * head_dim;
    if (x.size() != n || positions.size() != tokens) { return err_invalid("rope oracle size mismatch"); }
    out.resize(static_cast<size_t>(n));
    for (uint32_t t = 0; t < tokens; t++) {
        const double pos = positions[t];
        for (uint32_t h = 0; h < heads; h++) {
            const uint64_t base = (static_cast<uint64_t>(t) * heads + h) * head_dim;
            for (uint32_t i = 0; i < head_dim; i++) { out[base + i] = x[base + i]; }
            // NeoX rotate-half: pairs (p, p+half), duplicated cos/sin.
            const uint32_t half = rope_dim / 2u;
            for (uint32_t p = 0; p < half; p++) {
                const double freq = std::pow(static_cast<double>(theta), -static_cast<double>(2u * p) / rope_dim);
                const double angle = pos * freq;
                const double c = std::cos(angle), s = std::sin(angle);
                const double x0 = x[base + p], x1 = x[base + p + half];
                out[base + p] = static_cast<float>(x0 * c - x1 * s);
                out[base + p + half] = static_cast<float>(x0 * s + x1 * c);
            }
        }
    }
    return Status::success();
}

namespace {

// Naive two-pass causal GQA attention over the cache (oracle core).
[[nodiscard]] Status ref_attn_core(luisa::span<const float> q, luisa::span<const float> gate,
                                      luisa::span<const float> k_cache, luisa::span<const float> v_cache,
                                      uint32_t tokens, uint32_t q_heads, uint32_t kv_heads,
                                      uint32_t head_dim, float scale, uint32_t pos0, uint32_t kv_len,
                                      luisa::vector<float> &out) {
    if (q_heads == 0u || kv_heads == 0u || q_heads % kv_heads != 0u) {
        return err_invalid("attention oracle GQA ratio invalid");
    }
    const uint64_t qn = static_cast<uint64_t>(tokens) * q_heads * head_dim;
    if (q.size() != qn || gate.size() != qn) { return err_invalid("attention oracle q/gate size mismatch"); }
    const uint64_t kv_stride = static_cast<uint64_t>(kv_heads) * head_dim;
    if (k_cache.size() < static_cast<uint64_t>(kv_len) * kv_stride ||
        v_cache.size() < static_cast<uint64_t>(kv_len) * kv_stride) {
        return err_invalid("attention oracle cache too small for kv_len");
    }
    out.resize(static_cast<size_t>(qn));
    luisa::vector<double> scores(kv_len);
    for (uint32_t t = 0; t < tokens; t++) {
        const uint32_t k_end = pos0 + t + 1u;
        if (k_end > kv_len) { return err_invalid("attention oracle causal window exceeds kv_len"); }
        for (uint32_t h = 0; h < q_heads; h++) {
            const uint32_t kvh = h / (q_heads / kv_heads);
            const uint64_t q_base = (static_cast<uint64_t>(t) * q_heads + h) * head_dim;
            double max_s = -1e300;
            for (uint32_t key = 0; key < k_end; key++) {
                const uint64_t k_base = static_cast<uint64_t>(key) * kv_stride + kvh * head_dim;
                double s = 0.0;
                for (uint32_t d = 0; d < head_dim; d++) {
                    s += static_cast<double>(q[q_base + d]) * k_cache[k_base + d];
                }
                s *= scale;
                scores[key] = s;
                if (s > max_s) { max_s = s; }
            }
            double sum = 0.0;
            for (uint32_t key = 0; key < k_end; key++) {
                scores[key] = std::exp(scores[key] - max_s);
                sum += scores[key];
            }
            for (uint32_t d = 0; d < head_dim; d++) {
                double o = 0.0;
                for (uint32_t key = 0; key < k_end; key++) {
                    const uint64_t v_base = static_cast<uint64_t>(key) * kv_stride + kvh * head_dim;
                    o += scores[key] * v_cache[v_base + d];
                }
                out[q_base + d] = static_cast<float>(o / sum * ref_sigmoid(gate[q_base + d]));
            }
        }
    }
    return Status::success();
}

}// namespace

[[nodiscard]] Status ref_qwen_attn_prefill(luisa::span<const float> q, luisa::span<const float> gate,
                                              luisa::span<const float> k_cache, luisa::span<const float> v_cache,
                                              uint32_t tokens, uint32_t q_heads, uint32_t kv_heads,
                                              uint32_t head_dim, float scale, uint32_t pos0, uint32_t kv_len,
                                              luisa::vector<float> &out) {
    return ref_attn_core(q, gate, k_cache, v_cache, tokens, q_heads, kv_heads, head_dim, scale, pos0, kv_len, out);
}

[[nodiscard]] Status ref_qwen_attn_decode(luisa::span<const float> q, luisa::span<const float> gate,
                                             luisa::span<const float> k_cache, luisa::span<const float> v_cache,
                                             uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                                             float scale, uint32_t kv_len, luisa::vector<float> &out) {
    if (kv_len == 0u) { return err_invalid("decode oracle requires kv_len >= 1"); }
    // The decode query sits at absolute position kv_len-1.
    return ref_attn_core(q, gate, k_cache, v_cache, 1u, q_heads, kv_heads, head_dim, scale,
                         kv_len - 1u, kv_len, out);
}

[[nodiscard]] Status ref_qwen_gdn_conv(luisa::span<const float> src, luisa::span<const float> weight,
                                          luisa::span<float> ring, uint32_t tokens, uint32_t channels,
                                          uint32_t kernel_size, uint32_t pos0, luisa::vector<float> &out) {
    const auto val = validate_qwen_gdn_conv_inputs(tokens, channels, kernel_size, pos0);
    if (!val.ok()) { return val; }
    const uint64_t rw = kernel_size - 1u;
    if (src.size() != static_cast<uint64_t>(tokens) * channels ||
        weight.size() != static_cast<uint64_t>(channels) * kernel_size ||
        ring.size() != static_cast<uint64_t>(channels) * rw) {
        return err_invalid("gdn conv oracle size mismatch");
    }
    out.resize(static_cast<size_t>(tokens) * channels);
    for (uint32_t t = 0; t < tokens; t++) {
        const uint64_t p = static_cast<uint64_t>(pos0) + t;
        for (uint32_t c = 0; c < channels; c++) {
            double acc = 0.0;
            for (uint32_t j = 0; j < kernel_size; j++) {
                const int64_t abs_pos = static_cast<int64_t>(p) - static_cast<int64_t>(rw) + j;
                double xv = 0.0;
                if (abs_pos >= 0) {
                    if (abs_pos < pos0) {
                        xv = ring[static_cast<size_t>(c) * rw + static_cast<size_t>(abs_pos % static_cast<int64_t>(rw))];
                    } else {
                        xv = src[static_cast<size_t>(abs_pos - pos0) * channels + c];
                    }
                }
                // GGUF ssm_conv1d payload is channel-major: weight[c*K + j].
                acc += static_cast<double>(weight[static_cast<size_t>(c) * kernel_size + j]) * xv;
            }
            out[static_cast<size_t>(t) * channels + c] = static_cast<float>(ref_silu(acc));
        }
        for (uint32_t c = 0; c < channels; c++) {
            ring[static_cast<size_t>(c) * rw + static_cast<size_t>(p % rw)] = src[static_cast<size_t>(t) * channels + c];
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_gdn_gates(luisa::span<const float> a, luisa::span<const float> b,
                                           luisa::span<const float> ssm_a, luisa::span<const float> dt_bias,
                                           uint32_t tokens, uint32_t heads, float softplus_threshold,
                                           luisa::vector<float> &beta, luisa::vector<float> &g) {
    if (heads == 0u) { return err_invalid("gdn gates heads must be nonzero"); }
    const uint64_t n = static_cast<uint64_t>(tokens) * heads;
    if (a.size() != n || b.size() != n || ssm_a.size() != heads || dt_bias.size() != heads) {
        return err_invalid("gdn gates oracle size mismatch");
    }
    beta.resize(static_cast<size_t>(n));
    g.resize(static_cast<size_t>(n));
    for (uint64_t i = 0; i < n; i++) {
        const uint32_t h = static_cast<uint32_t>(i % heads);
        const double av = static_cast<double>(a[i]) + dt_bias[h];
        const double sp = av > softplus_threshold ? av : std::log1p(std::exp(av));
        beta[static_cast<size_t>(i)] = static_cast<float>(ref_sigmoid(b[i]));
        // GGUF ssm_a is already -exp(A_log) (negative); direct multiply.
        g[static_cast<size_t>(i)] = static_cast<float>(static_cast<double>(ssm_a[h]) * sp);
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_gdn_qk_norm(luisa::span<const float> conv_out, uint32_t tokens,
                                             uint32_t qk_heads, uint32_t head_dim, uint32_t v_channels,
                                             float eps, float q_scale,
                                             luisa::vector<float> &q, luisa::vector<float> &k) {
    if (qk_heads == 0u || head_dim == 0u) { return err_invalid("gdn qk norm geometry must be nonzero"); }
    const uint64_t stride = 2u * static_cast<uint64_t>(qk_heads) * head_dim + v_channels;
    if (conv_out.size() != static_cast<uint64_t>(tokens) * stride) {
        return err_invalid("gdn qk norm oracle conv_out size mismatch");
    }
    const uint64_t n = static_cast<uint64_t>(tokens) * qk_heads * head_dim;
    q.resize(static_cast<size_t>(n));
    k.resize(static_cast<size_t>(n));
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t h = 0; h < qk_heads; h++) {
            const uint64_t q_src = static_cast<uint64_t>(t) * stride + h * head_dim;
            const uint64_t k_src = q_src + static_cast<uint64_t>(qk_heads) * head_dim;
            const uint64_t dst = (static_cast<uint64_t>(t) * qk_heads + h) * head_dim;
            double qss = 0.0, kss = 0.0;
            for (uint32_t i = 0; i < head_dim; i++) {
                qss += static_cast<double>(conv_out[q_src + i]) * conv_out[q_src + i];
                kss += static_cast<double>(conv_out[k_src + i]) * conv_out[k_src + i];
            }
            // FLA/HF l2norm: eps inside the rsqrt.
            const double q_inv = 1.0 / std::sqrt(qss + static_cast<double>(eps));
            const double k_inv = 1.0 / std::sqrt(kss + static_cast<double>(eps));
            for (uint32_t i = 0; i < head_dim; i++) {
                q[dst + i] = static_cast<float>(conv_out[q_src + i] * q_inv * q_scale);
                k[dst + i] = static_cast<float>(conv_out[k_src + i] * k_inv);
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_gdn_recurrent(luisa::span<const float> q, luisa::span<const float> k,
                                               luisa::span<const float> conv_out, luisa::span<const float> beta,
                                               luisa::span<const float> g, luisa::span<float> state,
                                               uint32_t tokens, uint32_t v_heads, uint32_t head_dim,
                                               uint32_t qk_heads, luisa::vector<float> &out) {
    const auto val = validate_qwen_gdn_recurrent_inputs(tokens, v_heads, qk_heads, head_dim);
    if (!val.ok()) { return val; }
    const uint64_t qn = static_cast<uint64_t>(tokens) * qk_heads * head_dim;
    const uint64_t vn = static_cast<uint64_t>(tokens) * v_heads * head_dim;
    const uint64_t sn = static_cast<uint64_t>(v_heads) * head_dim * head_dim;
    const uint64_t v_offset = 2u * static_cast<uint64_t>(qk_heads) * head_dim;
    const uint64_t c_stride = v_offset + static_cast<uint64_t>(v_heads) * head_dim;
    if (q.size() != qn || k.size() != qn || conv_out.size() != static_cast<uint64_t>(tokens) * c_stride ||
        beta.size() != vn / head_dim || g.size() != vn / head_dim || state.size() != sn) {
        return err_invalid("gdn recurrent oracle size mismatch");
    }
    out.resize(static_cast<size_t>(vn));
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t h = 0; h < v_heads; h++) {
            // Tiled GGUF V order: V head h pairs with QK head h % qk_heads.
            const uint32_t qkh = h % qk_heads;
            const uint64_t th = static_cast<uint64_t>(t) * v_heads + h;
            const uint64_t qk_row = (static_cast<uint64_t>(t) * qk_heads + qkh) * head_dim;
            const uint64_t v_src = static_cast<uint64_t>(t) * c_stride + v_offset + h * head_dim;
            const uint64_t sb = static_cast<uint64_t>(h) * head_dim * head_dim;
            const double alpha = std::exp(static_cast<double>(g[th]));
            const double bt = beta[th];
            for (uint64_t i = 0; i < static_cast<uint64_t>(head_dim) * head_dim; i++) {
                state[sb + i] = static_cast<float>(static_cast<double>(state[sb + i]) * alpha);
            }
            for (uint32_t vi = 0; vi < head_dim; vi++) {
                double sk = 0.0;
                for (uint32_t ki = 0; ki < head_dim; ki++) {
                    sk += static_cast<double>(state[sb + static_cast<uint64_t>(ki) * head_dim + vi]) * k[qk_row + ki];
                }
                const double err = static_cast<double>(conv_out[v_src + vi]) - sk;
                for (uint32_t ki = 0; ki < head_dim; ki++) {
                    const uint64_t idx = sb + static_cast<uint64_t>(ki) * head_dim + vi;
                    state[idx] = static_cast<float>(static_cast<double>(state[idx]) + bt * k[qk_row + ki] * err);
                }
            }
            for (uint32_t vi = 0; vi < head_dim; vi++) {
                double o = 0.0;
                for (uint32_t ki = 0; ki < head_dim; ki++) {
                    o += static_cast<double>(state[sb + static_cast<uint64_t>(ki) * head_dim + vi]) * q[qk_row + ki];
                }
                out[(static_cast<uint64_t>(t) * v_heads + h) * head_dim + vi] = static_cast<float>(o);
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_gdn_gated_norm(luisa::span<const float> o, luisa::span<const float> z,
                                                luisa::span<const float> norm_w, uint32_t tokens,
                                                uint32_t heads, uint32_t head_dim, float eps,
                                                luisa::vector<float> &out) {
    if (heads == 0u || head_dim == 0u) { return err_invalid("gdn gated norm geometry must be nonzero"); }
    const uint64_t n = static_cast<uint64_t>(tokens) * heads * head_dim;
    if (o.size() != n || z.size() != n || norm_w.size() != head_dim) {
        return err_invalid("gdn gated norm oracle size mismatch");
    }
    out.resize(static_cast<size_t>(n));
    for (uint64_t r = 0; r < static_cast<uint64_t>(tokens) * heads; r++) {
        const uint64_t base = r * head_dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < head_dim; i++) { ss += static_cast<double>(o[base + i]) * o[base + i]; }
        const double inv = 1.0 / std::sqrt(ss / head_dim + eps);
        for (uint32_t i = 0; i < head_dim; i++) {
            out[base + i] = static_cast<float>(o[base + i] * inv * static_cast<double>(norm_w[i]) *
                                               ref_silu(z[base + i]));
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// 10. Device-free self-test
// ---------------------------------------------------------------------------
namespace {

struct SelfTestRng {
    uint64_t state{0x9E3779B97F4A7C15ull};
    [[nodiscard]] uint32_t next() noexcept {
        state ^= state >> 12u;
        state ^= state << 25u;
        state ^= state >> 27u;
        return static_cast<uint32_t>((state * 0x2545F4914F6CDD1Dull) >> 32u);
    }
    [[nodiscard]] float uniform(float lo, float hi) noexcept {
        return lo + (hi - lo) * (static_cast<float>(next() >> 8u) / 16777216.0f);
    }
};

[[nodiscard]] bool close_enough(float a, float b, float rel, float abs) noexcept {
    const double d = std::fabs(static_cast<double>(a) - b);
    return d <= abs + rel * std::fabs(static_cast<double>(b));
}

struct TestCtx {
    luisa::vector<luisa::string> *report;
    uint32_t failures{0};
    void check(bool ok, const char *name) {
        if (report != nullptr) {
            report->push_back(luisa::string{ok ? "PASS " : "FAIL "} + name);
        }
        if (!ok) { failures++; }
    }
};

}// namespace

[[nodiscard]] Status dsl_ops_self_test(luisa::vector<luisa::string> *report) {
    TestCtx ctx{report, 0u};
    SelfTestRng rng;

    // --- host bit helpers: bf16 round-trip of exactly representable values.
    {
        bool ok = true;
        for (int i = 0; i < 1000; i++) {
            const float v = rng.uniform(-100.0f, 100.0f);
            const uint16_t b = host_f32_to_bf16_bits(v);
            const float back = host_bf16_bits_to_f32(b);
            // bf16 has 8 mantissa bits: relative error <= 2^-8.
            ok = ok && close_enough(back, v, 1.0f / 128.0f, 1e-30f);
        }
        // Exact values round-trip bit-exactly.
        ok = ok && host_bf16_bits_to_f32(host_f32_to_bf16_bits(1.5f)) == 1.5f;
        ok = ok && host_bf16_bits_to_f32(host_f32_to_bf16_bits(-0.25f)) == -0.25f;
        ctx.check(ok, "bf16 host round-trip");
    }
    // --- fp8 E4M3: known codes, subnormals, saturation, NaN.
    {
        bool ok = true;
        ok = ok && host_fp8_e4m3_bits_to_f32(0x00u) == 0.0f;
        ok = ok && host_fp8_e4m3_bits_to_f32(0x80u) == 0.0f;// -0
        ok = ok && host_fp8_e4m3_bits_to_f32(0x38u) == 1.0f;// e=7,m=0
        ok = ok && host_fp8_e4m3_bits_to_f32(0x7Eu) == 448.0f;// max finite
        ok = ok && host_fp8_e4m3_bits_to_f32(0x01u) == std::ldexp(1.0f, -9);// min subnormal
        ok = ok && std::isnan(host_fp8_e4m3_bits_to_f32(0x7Fu));
        ok = ok && host_f32_to_fp8_e4m3_bits(1.0f) == 0x38u;
        ok = ok && host_f32_to_fp8_e4m3_bits(448.0f) == 0x7Eu;
        ok = ok && host_f32_to_fp8_e4m3_bits(1000.0f) == 0x7Eu;// saturate
        ok = ok && host_f32_to_fp8_e4m3_bits(-1000.0f) == 0xFEu;
        ok = ok && host_f32_to_fp8_e4m3_bits(0.0f) == 0x00u;
        // RNE tie: 1.0625 is exactly between 1.0 (m=0) and 1.125 (m=1) -> even m=0.
        ok = ok && host_f32_to_fp8_e4m3_bits(1.0625f) == 0x38u;
        // Round-trip of all finite codes.
        for (uint32_t c = 0; c < 256u; c++) {
            if ((c & 0x7Fu) == 0x7Fu) { continue; }// NaN
            const float v = host_fp8_e4m3_bits_to_f32(static_cast<uint8_t>(c));
            ok = ok && host_f32_to_fp8_e4m3_bits(v) == static_cast<uint8_t>(c);
        }
        ctx.check(ok, "fp8 e4m3 host codec");
    }
    // --- packing + matmul oracle vs an independent dequantized reference,
    //     ragged K (odd for fp8 partial groups/words), small M/N.
    for (auto format : {QwenWeightFormat::bf16, QwenWeightFormat::fp8_e4m3}) {
        const uint32_t K = 70u, N = 5u, M = 3u;// K=70: partial fp8 group (64+6), odd word tail
        luisa::vector<float> w(static_cast<size_t>(K) * N), x(static_cast<size_t>(M) * K);
        for (auto &v : w) { v = rng.uniform(-1.0f, 1.0f); }
        for (auto &v : x) { v = rng.uniform(-1.0f, 1.0f); }
        luisa::vector<uint32_t> words;
        luisa::vector<float> scales;
        const auto pack = host_pack_matrix(format, w, K, N, 64u, words, scales);
        ctx.check(pack.ok(), format == QwenWeightFormat::bf16 ? "pack bf16 ragged" : "pack fp8 ragged");
        // Independent reference: dequantize via host_decode_packed, double dot.
        luisa::vector<float> expect(static_cast<size_t>(M) * N);
        for (uint32_t mi = 0; mi < M; mi++) {
            for (uint32_t ni = 0; ni < N; ni++) {
                double acc = 0.0;
                for (uint32_t ki = 0; ki < K; ki++) {
                    acc += static_cast<double>(x[mi * K + ki]) *
                           host_decode_packed(format, words, scales, K, 64u, ni * K + ki);
                }
                expect[mi * N + ni] = static_cast<float>(acc);
            }
        }
        luisa::vector<float> y;
        const auto mm = ref_qwen_matmul(format, x, words, scales, {}, false, M, N, K, 64u, y);
        bool ok = mm.ok() && y.size() == expect.size();
        const float tol = format == QwenWeightFormat::bf16 ? 2e-2f : 0.35f;
        for (size_t i = 0; ok && i < y.size(); i++) { ok = close_enough(y[i], expect[i], tol, 1e-4f); }
        ctx.check(ok, format == QwenWeightFormat::bf16 ? "matmul bf16 oracle ragged" : "matmul fp8 oracle ragged");
        // Zero group: all-zero weights must give scale 0 and exact zero output.
        luisa::vector<float> zw(static_cast<size_t>(K) * N, 0.0f);
        luisa::vector<uint32_t> zw_words;
        luisa::vector<float> zw_scales;
        const auto zpack = host_pack_matrix(format, zw, K, N, 64u, zw_words, zw_scales);
        luisa::vector<float> zy;
        const auto zmm = ref_qwen_matmul(format, x, zw_words, zw_scales, {}, false, M, N, K, 64u, zy);
        bool zok = zpack.ok() && zmm.ok();
        for (auto v : zy) { zok = zok && v == 0.0f; }
        if (format == QwenWeightFormat::fp8_e4m3) {
            for (auto s : zw_scales) { zok = zok && s == 0.0f; }
        }
        ctx.check(zok, format == QwenWeightFormat::bf16 ? "matmul bf16 zero group" : "matmul fp8 zero group");
    }
    // --- embedding gather oracle (both formats, ragged hidden).
    for (auto format : {QwenWeightFormat::bf16, QwenWeightFormat::fp8_e4m3}) {
        const uint64_t hidden = 34u, vocab = 7u;// hidden=34: even for bf16, partial fp8 group
        luisa::vector<float> table(static_cast<size_t>(hidden * vocab));
        for (auto &v : table) { v = rng.uniform(-1.0f, 1.0f); }
        luisa::vector<uint32_t> words;
        luisa::vector<float> scales;
        const auto pack = host_pack_matrix(format, table, hidden, vocab, 64u, words, scales);
        const luisa::vector<uint32_t> ids{6u, 0u, 3u, 6u};
        luisa::vector<float> out;
        const auto emb = ref_qwen_embedding(format, ids, words, scales, hidden, vocab, 64u, out);
        bool ok = pack.ok() && emb.ok() && out.size() == ids.size() * hidden;
        for (size_t t = 0; ok && t < ids.size(); t++) {
            for (uint64_t c = 0; c < hidden; c++) {
                const float expect = host_decode_packed(format, words, scales, hidden, 64u, ids[t] * hidden + c);
                ok = close_enough(out[t * hidden + c], expect, 0.0f, 0.0f);
            }
        }
        ctx.check(ok, format == QwenWeightFormat::bf16 ? "embedding bf16 oracle" : "embedding fp8 oracle");
        // Out-of-range id must be rejected.
        const luisa::vector<uint32_t> bad{7u};
        luisa::vector<float> dummy;
        ctx.check(!ref_qwen_embedding(format, bad, words, scales, hidden, vocab, 64u, dummy).ok(),
                  "embedding rejects out-of-range id");
    }
    // --- rmsnorm (plain + residual), ragged dim.
    {
        const uint32_t rows = 3u, dim = 37u;
        luisa::vector<float> x(rows * dim), w(dim), res(rows * dim);
        for (auto &v : x) { v = rng.uniform(-2.0f, 2.0f); }
        // GGUF norm weights are PRE-ADDED (converter folds 1+w offline):
        // generate them around 1, exactly like the real checkpoint.
        for (auto &v : w) { v = rng.uniform(0.5f, 1.5f); }
        for (auto &v : res) { v = rng.uniform(-1.0f, 1.0f); }
        luisa::vector<float> y, sum;
        const auto r1 = ref_qwen_rmsnorm(x, w, {}, false, rows, dim, 1e-6f, y, sum);
        // Independent scalar check of row 0 (direct weight multiply).
        double ss = 0.0;
        for (uint32_t i = 0; i < dim; i++) { ss += static_cast<double>(x[i]) * x[i]; }
        const double inv = 1.0 / std::sqrt(ss / dim + 1e-6);
        bool ok = r1.ok();
        for (uint32_t i = 0; ok && i < dim; i++) {
            ok = close_enough(y[i], static_cast<float>(x[i] * inv * w[i]), 1e-5f, 1e-6f);
        }
        ctx.check(ok, "rmsnorm pre-added direct-weight oracle");
        const auto r2 = ref_qwen_rmsnorm(x, w, res, true, rows, dim, 1e-6f, y, sum);
        ok = r2.ok();
        double ss2 = 0.0;
        for (uint32_t i = 0; i < dim; i++) {
            const double v = static_cast<double>(x[i]) + res[i];
            ss2 += v * v;
        }
        const double inv2 = 1.0 / std::sqrt(ss2 / dim + 1e-6);
        for (uint32_t i = 0; ok && i < dim; i++) {
            ok = close_enough(y[i], static_cast<float>((x[i] + res[i]) * inv2 * w[i]), 1e-5f, 1e-6f);
        }
        ctx.check(ok, "rmsnorm residual oracle");
    }
    // --- rmsnorm snapshot: sum_out is x bit-exactly; y == plain norm math.
    {
        const uint32_t rows = 3u, dim = 37u;// ragged: not a multiple of 32
        luisa::vector<float> x(rows * dim), w(dim);
        for (auto &v : x) { v = rng.uniform(-2.0f, 2.0f); }
        for (auto &v : w) { v = rng.uniform(0.5f, 1.5f); }// pre-added GGUF weights
        luisa::vector<float> y, sum;
        const auto r = ref_qwen_rmsnorm_snapshot(x, w, rows, dim, 1e-6f, y, sum);
        bool ok = r.ok() && sum.size() == x.size() && y.size() == x.size();
        for (size_t i = 0; ok && i < x.size(); i++) { ok = sum[i] == x[i]; }// bit-exact
        // Independent scalar check of every row (direct weight multiply).
        for (uint32_t row = 0; ok && row < rows; row++) {
            double ss = 0.0;
            for (uint32_t i = 0; i < dim; i++) {
                const double v = x[row * dim + i];
                ss += v * v;
            }
            const double inv = 1.0 / std::sqrt(ss / dim + 1e-6);
            for (uint32_t i = 0; ok && i < dim; i++) {
                ok = close_enough(y[row * dim + i],
                                  static_cast<float>(x[row * dim + i] * inv * w[i]), 1e-5f, 1e-6f);
            }
        }
        // y must equal the PLAIN oracle exactly (same inputs, same math).
        luisa::vector<float> py, psum;
        const auto rp = ref_qwen_rmsnorm(x, w, {}, false, rows, dim, 1e-6f, py, psum);
        ok = ok && rp.ok() && py.size() == y.size();
        for (size_t i = 0; ok && i < y.size(); i++) { ok = y[i] == py[i]; }
        // Size validation fails closed (never reads OOB).
        luisa::vector<float> ty, tsum;
        ok = ok && !ref_qwen_rmsnorm_snapshot(x, w, rows, 0u, 1e-6f, ty, tsum).ok();
        ok = ok && !ref_qwen_rmsnorm_snapshot(luisa::span<const float>(x.data(), x.size() - 1u),
                                              w, rows, dim, 1e-6f, ty, tsum).ok();
        ctx.check(ok, "rmsnorm snapshot oracle");
    }
    // --- swiglu split/fused parity.
    {
        const uint32_t m = 2u, f = 9u;
        luisa::vector<float> proj(m * 2u * f);
        for (auto &v : proj) { v = rng.uniform(-3.0f, 3.0f); }
        // Row 0 of the fused projection is [gate f | up f].
        luisa::vector<float> gate(proj.begin(), proj.begin() + f);
        luisa::vector<float> up(proj.begin() + f, proj.begin() + 2 * f);
        luisa::vector<float> o1, o2;
        const auto r1 = ref_qwen_swiglu(gate, up, o1);
        const auto r2 = ref_qwen_swiglu_fused(proj, m, f, o2);
        bool ok = r1.ok() && r2.ok() && o2.size() == static_cast<size_t>(m) * f;
        // Compare row 0 against the split path; check row 1 independently.
        for (size_t i = 0; ok && i < o1.size(); i++) { ok = close_enough(o1[i], o2[i], 1e-6f, 1e-7f); }
        for (uint32_t i = 0; ok && i < f; i++) {
            const double expect = ref_silu(proj[2u * f + i]) * static_cast<double>(proj[3u * f + i]);
            ok = close_enough(o2[f + i], static_cast<float>(expect), 1e-6f, 1e-7f);
        }
        ctx.check(ok, "swiglu split/fused parity");
    }
    // --- attention prepare + prefill + decode end-to-end oracle chain with
    //     GQA (4 q heads, 2 kv heads), partial rope, ragged cache.
    {
        QwenDslConfig cfg;
        cfg.q_heads = 4u;
        cfg.kv_heads = 2u;
        cfg.attn_head_dim = 64u;
        cfg.rope_dim = 16u;
        const uint32_t hd = cfg.attn_head_dim;
        const uint32_t T = 5u;
        const uint32_t pos0 = 3u;// non-zero history
        const uint32_t capacity = 11u;
        luisa::vector<float> q_proj(T * cfg.attn_q_proj_n()), k_proj(T * cfg.attn_kv_n()), v_proj(T * cfg.attn_kv_n());
        luisa::vector<float> qnw(hd), knw(hd);
        luisa::vector<int32_t> positions(T);
        for (auto &v : q_proj) { v = rng.uniform(-1.0f, 1.0f); }
        for (auto &v : k_proj) { v = rng.uniform(-1.0f, 1.0f); }
        for (auto &v : v_proj) { v = rng.uniform(-1.0f, 1.0f); }
        for (auto &v : qnw) { v = rng.uniform(0.7f, 1.3f); }// pre-added GGUF weights
        for (auto &v : knw) { v = rng.uniform(0.7f, 1.3f); }
        for (uint32_t t = 0; t < T; t++) { positions[t] = static_cast<int32_t>(pos0 + t); }
        // Prefill the history [0, pos0) with random normed+roped K/V directly.
        luisa::vector<float> k_cache(capacity * cfg.attn_kv_n()), v_cache(capacity * cfg.attn_kv_n());
        for (auto &v : k_cache) { v = rng.uniform(-1.0f, 1.0f); }
        for (auto &v : v_cache) { v = rng.uniform(-1.0f, 1.0f); }
        luisa::vector<float> q, gate, k, v;
        const auto prep = ref_qwen_attn_prepare(q_proj, k_proj, v_proj, qnw, knw, positions, T, cfg, pos0,
                                                true, k_cache, v_cache, q, gate, k, v);
        ctx.check(prep.ok(), "attn prepare oracle");
        // RoPE tail passthrough + norm-before-rope spot check on q row 0:
        // recompute independently.
        bool ok = prep.ok();
        {
            double ss = 0.0;
            for (uint32_t i = 0; i < hd; i++) { ss += static_cast<double>(q_proj[i]) * q_proj[i]; }
            const double inv = 1.0 / std::sqrt(ss / hd + cfg.norm_eps);
            // Tail element (>= rope_dim) must equal normed value exactly.
            for (uint32_t i = cfg.rope_dim; ok && i < hd; i++) {
                ok = close_enough(q[i], static_cast<float>(q_proj[i] * inv * qnw[i]), 1e-5f, 1e-6f);
            }
            // Gate is raw.
            for (uint32_t i = 0; ok && i < hd; i++) { ok = gate[i] == q_proj[hd + i]; }
        }
        ctx.check(ok, "attn prepare norm/rope/gate semantics");
        const uint32_t kv_len = pos0 + T;
        luisa::vector<float> attn_out;
        const auto pre = ref_qwen_attn_prefill(q, gate, k_cache, v_cache, T, cfg.q_heads, cfg.kv_heads,
                                               hd, cfg.attn_scale(), pos0, kv_len, attn_out);
        ctx.check(pre.ok(), "attn prefill oracle");
        // Decode at the last position must equal the last prefill row.
        luisa::vector<float> q_last(q.begin() + (T - 1u) * cfg.q_heads * hd, q.end());
        luisa::vector<float> gate_last(gate.begin() + (T - 1u) * cfg.q_heads * hd, gate.end());
        luisa::vector<float> dec_out;
        const auto dec = ref_qwen_attn_decode(q_last, gate_last, k_cache, v_cache, cfg.q_heads, cfg.kv_heads,
                                              hd, cfg.attn_scale(), kv_len, dec_out);
        ok = dec.ok() && dec_out.size() == cfg.q_heads * hd;
        for (size_t i = 0; ok && i < dec_out.size(); i++) {
            ok = close_enough(dec_out[i], attn_out[(T - 1u) * cfg.q_heads * hd + i], 1e-5f, 1e-6f);
        }
        ctx.check(ok, "attn decode == prefill last row");
        // Standalone rope oracle parity with the prepare's q rotation.
        luisa::vector<float> rope_out;
        const auto rr = ref_qwen_rope_partial(q, positions, T, cfg.q_heads, hd, cfg.rope_dim, cfg.rope_theta, rope_out);
        ctx.check(rr.ok(), "rope partial oracle runs");
    }
    // --- GDN chain: conv with history across two chunks, gates, qk norm,
    //     recurrent prefill vs decode step-by-step, gated norm.
    {
        QwenDslConfig cfg;
        cfg.gdn_qk_heads = 2u;
        cfg.gdn_v_heads = 4u;
        cfg.gdn_head_dim = 32u;
        const uint32_t hd = cfg.gdn_head_dim;
        const uint32_t C = cfg.gdn_qkv_n();// 2*2*32 + 4*32 = 256
        const uint32_t K = cfg.conv_kernel;
        const uint32_t T1 = 3u, T2 = 4u;
        luisa::vector<float> src1(T1 * C), src2(T2 * C), weight(K * C);
        luisa::vector<float> ring(C * (K - 1u), 0.0f);
        for (auto &v : src1) { v = rng.uniform(-1.0f, 1.0f); }
        for (auto &v : src2) { v = rng.uniform(-1.0f, 1.0f); }
        for (auto &v : weight) { v = rng.uniform(-0.5f, 0.5f); }
        luisa::vector<float> conv1, conv2;
        const auto c1 = ref_qwen_gdn_conv(src1, weight, ring, T1, C, K, 0u, conv1);
        const auto c2 = ref_qwen_gdn_conv(src2, weight, ring, T2, C, K, T1, conv2);
        ctx.check(c1.ok() && c2.ok(), "gdn conv two-chunk history");
        // Independent full-sequence conv (no chunk boundary) must match.
        luisa::vector<float> src_all(src1);
        src_all.insert(src_all.end(), src2.begin(), src2.end());
        luisa::vector<float> ring_zero(C * (K - 1u), 0.0f), conv_all;
        const auto ca = ref_qwen_gdn_conv(src_all, weight, ring_zero, T1 + T2, C, K, 0u, conv_all);
        bool ok = ca.ok();
        for (size_t i = 0; ok && i < conv1.size(); i++) { ok = close_enough(conv1[i], conv_all[i], 1e-5f, 1e-6f); }
        for (size_t i = 0; ok && i < conv2.size(); i++) {
            ok = close_enough(conv2[i], conv_all[conv1.size() + i], 1e-5f, 1e-6f);
        }
        ctx.check(ok, "gdn conv chunked == full");
        // Gates.
        const uint32_t H = cfg.gdn_v_heads;
        const uint32_t T = T1 + T2;
        luisa::vector<float> a(T * H), b(T * H), ssm_a(H), dt_bias(H);
        for (auto &v : a) { v = rng.uniform(-4.0f, 4.0f); }
        for (auto &v : b) { v = rng.uniform(-4.0f, 4.0f); }
        // GGUF ssm_a = -exp(A_log), A_log = log(U(0.01,16)): strictly negative.
        for (auto &v : ssm_a) { v = -std::exp(static_cast<double>(rng.uniform(-4.6f, 2.77f))); }
        for (auto &v : dt_bias) { v = rng.uniform(-1.0f, 1.0f); }
        luisa::vector<float> beta, g;
        const auto gt = ref_qwen_gdn_gates(a, b, ssm_a, dt_bias, T, H, cfg.softplus_threshold, beta, g);
        ok = gt.ok();
        for (auto gv : g) { ok = ok && gv <= 0.0f; }
        // Independent scalar check of element 0 with the GGUF convention.
        {
            const double av = static_cast<double>(a[0]) + dt_bias[0];
            const double sp = av > cfg.softplus_threshold ? av : std::log1p(std::exp(av));
            ok = ok && close_enough(g[0], static_cast<float>(static_cast<double>(ssm_a[0]) * sp), 1e-5f, 1e-6f);
        }
        ctx.check(ok, "gdn gates (negative ssm_a, g <= 0)");
        // QK norm.
        luisa::vector<float> q, k;
        const auto qk = ref_qwen_gdn_qk_norm(conv_all, T, cfg.gdn_qk_heads, hd, cfg.gdn_v_n(),
                                             cfg.l2_eps, cfg.gdn_q_scale(), q, k);
        ok = qk.ok();
        // Verify Q row 0 norm: ||q|| == q_scale (when ||raw|| > eps).
        {
            double ss = 0.0;
            for (uint32_t i = 0; i < hd; i++) { ss += static_cast<double>(q[i]) * q[i]; }
            ok = ok && close_enough(static_cast<float>(std::sqrt(ss)), cfg.gdn_q_scale(), 1e-4f, 1e-6f);
        }
        ctx.check(ok, "gdn qk l2 norm + q scale");
        // Recurrent prefill (T tokens) vs step-by-step decode (T x 1 token).
        luisa::vector<float> state(static_cast<size_t>(H) * hd * hd);
        for (auto &v : state) { v = rng.uniform(-0.1f, 0.1f); }
        luisa::vector<float> state_copy = state;
        luisa::vector<float> out_full;
        const auto rec = ref_qwen_gdn_recurrent(q, k, conv_all, beta, g, state, T, H, hd, cfg.gdn_qk_heads, out_full);
        ctx.check(rec.ok(), "gdn recurrent prefill oracle");
        // Step-by-step: slice per-token q/k/conv/beta/g.
        luisa::vector<float> out_step(static_cast<size_t>(T) * H * hd);
        bool step_ok = true;
        const uint64_t qk_stride = static_cast<uint64_t>(cfg.gdn_qk_heads) * hd;
        const uint64_t c_stride = static_cast<uint64_t>(C);
        for (uint32_t t = 0; t < T && step_ok; t++) {
            luisa::vector<float> qt(q.begin() + t * qk_stride, q.begin() + (t + 1u) * qk_stride);
            luisa::vector<float> kt(k.begin() + t * qk_stride, k.begin() + (t + 1u) * qk_stride);
            luisa::vector<float> ct(conv_all.begin() + t * c_stride, conv_all.begin() + (t + 1u) * c_stride);
            luisa::vector<float> bt(beta.begin() + t * H, beta.begin() + (t + 1u) * H);
            luisa::vector<float> gt1(g.begin() + t * H, g.begin() + (t + 1u) * H);
            luisa::vector<float> ot;
            const auto r = ref_qwen_gdn_recurrent(qt, kt, ct, bt, gt1, state_copy, 1u, H, hd, cfg.gdn_qk_heads, ot);
            step_ok = r.ok();
            for (uint32_t i = 0; i < H * hd && step_ok; i++) { out_step[static_cast<size_t>(t) * H * hd + i] = ot[i]; }
        }
        ok = step_ok;
        for (size_t i = 0; ok && i < out_full.size(); i++) {
            ok = close_enough(out_full[i], out_step[i], 2e-4f, 2e-5f);
        }
        // Final states must match too.
        for (size_t i = 0; ok && i < state.size(); i++) {
            ok = close_enough(state[i], state_copy[i], 2e-4f, 2e-5f);
        }
        ctx.check(ok, "gdn recurrent prefill == step-by-step decode");
        // Gated output norm: direct weight, norm before gate.
        luisa::vector<float> z(static_cast<size_t>(T) * H * hd), nw(hd);
        for (auto &v : z) { v = rng.uniform(-2.0f, 2.0f); }
        for (auto &v : nw) { v = rng.uniform(0.5f, 1.5f); }
        luisa::vector<float> normed;
        const auto gn = ref_qwen_gdn_gated_norm(out_full, z, nw, T, H, hd, cfg.norm_eps, normed);
        ok = gn.ok();
        // Independent row-0 check.
        {
            double ss = 0.0;
            for (uint32_t i = 0; i < hd; i++) { ss += static_cast<double>(out_full[i]) * out_full[i]; }
            const double inv = 1.0 / std::sqrt(ss / hd + cfg.norm_eps);
            for (uint32_t i = 0; ok && i < hd; i++) {
                const double expect = out_full[i] * inv * nw[i] * ref_silu(z[i]);
                ok = close_enough(normed[i], static_cast<float>(expect), 1e-5f, 1e-6f);
            }
        }
        ctx.check(ok, "gdn gated norm (direct weight, norm before gate)");
    }
    // --- planning: overflow refusal, empty plans, flattening.
    {
        const auto empty = plan_qwen_threads(0u);
        ctx.check(empty.ok() && empty.value->empty(), "plan empty");
        const auto big = plan_qwen_threads(0x100000000ull);
        ctx.check(!big.ok(), "plan refuses > uint32 count");
        const auto flat = plan_qwen_threads(static_cast<uint64_t>(kQwenMaxGroupsPerDim) * 256u + 1u);
        ctx.check(flat.ok() && flat.value->is_2d(), "plan flattens past group envelope");
        const auto g2 = plan_qwen_threads_2d(3u, 7u);
        ctx.check(g2.ok() && g2.value->padded_cols >= 7u, "plan 2d ragged");
        const auto bad = plan_qwen_gemv_dispatch(5u, 8u, 16u);
        ctx.check(!bad.ok(), "gemv plan refuses M > 4");
        const auto ov = plan_qwen_gemm_dispatch(0x10000ull, 0x10000ull, 0x10000ull);
        ctx.check(!ov.ok(), "gemm plan refuses overflow");
    }
    // --- config validation.
    {
        QwenDslConfig cfg;
        ctx.check(validate_qwen_config(cfg).ok(), "default config valid");
        cfg.q_heads = 15u;// not a multiple of kv_heads=4
        ctx.check(!validate_qwen_config(cfg).ok(), "config rejects bad GQA ratio");
    }
    // --- WY/UT chunk prefill is part of the supported operator surface.
    ctx.check(dsl_ops_chunk_prefill_status().ok(), "chunk-64 WY/UT prefill available");
    ctx.check(qwen_gdn_chunk_shared_bytes() <= 32u * 1024u, "chunk-64 shared budget <= 32 KiB");
    ctx.check(validate_qwen_gdn_chunk_inputs(65u, 4u, 2u, 32u).ok(), "chunk-64 ragged plan accepted");
    ctx.check(!validate_qwen_gdn_chunk_inputs(65u, 3u, 2u, 32u).ok(), "chunk-64 rejects invalid head pairing");

    // ------------------------------------------------------------------
    // EXTERNAL-CONTRACT FIXTURES (GGUF semantics; independent expectations
    // so a shared oracle/kernel bug cannot pass both sides). Each fixture
    // pins one converter/HF convention that the generic kernels must honor.
    // ------------------------------------------------------------------
    // (a) RMS norm with KNOWN pre-added weights: w = [2, 0.5, 1, 1.5],
    //     x = [3, 4, 0, -1] -> y = x * rsqrt(26/4 + eps) * w (direct).
    {
        const luisa::vector<float> x{3.0f, 4.0f, 0.0f, -1.0f};
        const luisa::vector<float> w{2.0f, 0.5f, 1.0f, 1.5f};
        luisa::vector<float> y, sum;
        const auto r = ref_qwen_rmsnorm(x, w, {}, false, 1u, 4u, 1e-6f, y, sum);
        const double inv = 1.0 / std::sqrt(26.0 / 4.0 + 1e-6);
        bool ok = r.ok();
        const float expect[4] = {float(3.0 * inv * 2.0), float(4.0 * inv * 0.5),
                                 0.0f, float(-1.0 * inv * 1.5)};
        for (uint32_t i = 0; ok && i < 4u; i++) { ok = close_enough(y[i], expect[i], 1e-6f, 1e-7f); }
        ctx.check(ok, "fixture rmsnorm pre-added direct weight");
    }
    // (b) Gates with KNOWN negative ssm_a (GGUF -exp(A_log)):
    //     ssm_a = [-2, -0.5], dt_bias = [0.25, -1], a = [1, 30], b = [0, 2].
    //     g0 = -2 * log1p(exp(1.25)); g1 = -0.5 * 29 (softplus threshold 20).
    {
        const luisa::vector<float> a{1.0f, 30.0f}, b{0.0f, 2.0f};
        const luisa::vector<float> ssm_a{-2.0f, -0.5f}, dt{0.25f, -1.0f};
        luisa::vector<float> beta, g;
        const auto r = ref_qwen_gdn_gates(a, b, ssm_a, dt, 1u, 2u, 20.0f, beta, g);
        const double g0 = -2.0 * std::log1p(std::exp(1.25));
        const double g1 = -0.5 * 29.0;
        const double b0 = 0.5, b1 = 1.0 / (1.0 + std::exp(-2.0));
        const bool ok = r.ok() &&
                        close_enough(g[0], float(g0), 1e-6f, 1e-7f) &&
                        close_enough(g[1], float(g1), 1e-6f, 1e-7f) &&
                        close_enough(beta[0], float(b0), 1e-6f, 1e-7f) &&
                        close_enough(beta[1], float(b1), 1e-6f, 1e-7f);
        ctx.check(ok, "fixture gdn gates negative ssm_a direct");
    }
    // (c) Channel-major impulse conv: C=2, K=4, weight[c*K+j] = c*10+j+1.
    //     src = impulse at t=0 on channel 0 only. Causal depthwise conv (tap
    //     j reads position p-(K-1)+j, i.e. HF F.conv1d with K-1 left padding
    //     truncated to T): the impulse at position 0 is seen by output t
    //     through tap j = K-1-t, so out[t,0] = silu(w[0*K + (K-1-t)]) for
    //     t in [0,K), zero afterwards; channel 1 stays zero. A kernel-major
    //     misread would use w[(K-1-t)*C + 0] and fail. Ring history zero.
    {
        const uint32_t C = 2u, K = 4u, T = 4u;
        luisa::vector<float> w(C * K), src(T * C, 0.0f), ring(C * (K - 1u), 0.0f);
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t j = 0; j < K; j++) { w[c * K + j] = float(c * 10 + j + 1); }
        }
        src[0u * C + 0u] = 1.0f;// impulse: token 0, channel 0
        luisa::vector<float> out;
        const auto r = ref_qwen_gdn_conv(src, w, ring, T, C, K, 0u, out);
        bool ok = r.ok();
        for (uint32_t t = 0; t < T && ok; t++) {
            for (uint32_t c = 0; c < C && ok; c++) {
                const double acc = (c == 0u && t < K) ? static_cast<double>(w[0u * K + (K - 1u - t)]) : 0.0;
                ok = close_enough(out[t * C + c], float(ref_silu(acc)), 1e-6f, 1e-7f);
            }
        }
        ctx.check(ok, "fixture conv channel-major impulse");
    }
    // (d) Tiled V-head pairing: qk_heads=2, v_heads=4, hd=32. Distinct unit
    //     q/k per QK head, distinct v per V head, beta=1, g=0, zero state.
    //     After one token S = k v^T so o[vh] = (q[qk] . k[qk]) * v[vh];
    //     tiled mapping requires qk = vh % 2 (grouped vh/2 would differ).
    {
        const uint32_t QK = 2u, VH = 4u, HD = 32u;
        luisa::vector<float> q(QK * HD, 0.0f), k(QK * HD, 0.0f);
        q[0] = 1.0f; q[HD + 1] = 1.0f;
        k[0] = 1.0f; k[HD + 1] = 1.0f;
        const uint32_t C = 2u * QK * HD + VH * HD;
        luisa::vector<float> conv(C, 0.0f);
        for (uint32_t vh = 0; vh < VH; vh++) { conv[2u * QK * HD + vh * HD + 0u] = float(vh + 1); }
        luisa::vector<float> beta(VH, 1.0f), g(VH, 0.0f), state(VH * HD * HD, 0.0f), out;
        const auto r = ref_qwen_gdn_recurrent(q, k, conv, beta, g, state, 1u, VH, HD, QK, out);
        bool ok = r.ok();
        for (uint32_t vh = 0; vh < VH && ok; vh++) {
            ok = close_enough(out[vh * HD + 0u], float(vh + 1), 1e-6f, 1e-7f);
            for (uint32_t i = 1; i < HD && ok; i++) { ok = out[vh * HD + i] == 0.0f; }
        }
        ctx.check(ok, "fixture gdn tiled v-head pairing (h % qk_heads)");
    }
    // (e) NeoX rotate-half rope: hd=8, rope_dim=4 (half=2), theta=10000,
    //     pos=1, x=[1,2,3,4,5,6,7,8]. freq[p]=10000^(-2p/4).
    //     out[0]=c0-3s0; out[2]=s0+3c0; out[1]=2c1-4s1; out[3]=2s1+4c1;
    //     tail [4..8) passthrough.
    {
        const luisa::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
        const luisa::vector<int32_t> pos{1};
        luisa::vector<float> out;
        const auto r = ref_qwen_rope_partial(x, pos, 1u, 1u, 8u, 4u, 10000.0f, out);
        const double f0 = 1.0, f1 = std::pow(10000.0, -0.5);
        const double c0 = std::cos(f0), s0 = std::sin(f0);
        const double c1 = std::cos(f1), s1 = std::sin(f1);
        const float expect[8] = {float(1 * c0 - 3 * s0), float(2 * c1 - 4 * s1),
                                 float(1 * s0 + 3 * c0), float(2 * s1 + 4 * c1),
                                 5, 6, 7, 8};
        bool ok = r.ok();
        for (uint32_t i = 0; ok && i < 8u; i++) { ok = close_enough(out[i], expect[i], 1e-5f, 1e-6f); }
        ctx.check(ok, "fixture rope NeoX rotate-half + tail");
    }
    // (f) Near-zero L2 rows: conv row q = [1e-4, 0...], k = [0...].
    //     FLA l2norm: q_out = x * rsqrt(sum+eps) * q_scale (tiny, NOT ~1/eps
    //     blown up); k_out exactly 0.
    {
        const uint32_t QK = 1u, HD = 32u, VN = 32u;
        const uint32_t C = 2u * QK * HD + VN;
        luisa::vector<float> conv(C, 0.0f);
        conv[0] = 1e-4f;
        luisa::vector<float> q, k;
        const float qs = 1.0f / std::sqrt(32.0f);
        const auto r = ref_qwen_gdn_qk_norm(conv, 1u, QK, HD, VN, 1e-6f, qs, q, k);
        const double inv = 1.0 / std::sqrt(1e-8 + 1e-6);
        bool ok = r.ok() && close_enough(q[0], float(1e-4 * inv * qs), 1e-5f, 1e-9f);
        for (uint32_t i = 1; i < HD && ok; i++) { ok = q[i] == 0.0f; }
        for (uint32_t i = 0; i < HD && ok; i++) { ok = k[i] == 0.0f; }
        ctx.check(ok, "fixture gdn l2norm near-zero rows (eps inside rsqrt)");
    }

    if (ctx.failures != 0u) {
        return Status::error(StatusCode::invalid_argument,
                                luisa::format("dsl_ops_self_test: {} check(s) failed", ctx.failures));
    }
    return Status::success();
}

}// namespace luisa::qwen::dsl
