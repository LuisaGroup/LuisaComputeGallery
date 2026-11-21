// Qwen3.5-4B shared engine layer -- streaming device weight upload.
// One tensor at a time, bounded host staging, recoverable failures only.
#include "qwen/weights.h"

#include "qwen/mtp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>

namespace luisa::qwen {
namespace {

constexpr uint64_t kMinChunkBytes = 64ull * 1024ull;
constexpr uint64_t kMaxChunkBytes = 64ull * 1024ull * 1024ull;

[[nodiscard]] uint64_t clamp_chunk_bytes(uint64_t requested) noexcept {
    const uint64_t c = std::clamp<uint64_t>(requested, kMinChunkBytes, kMaxChunkBytes);
    return c & ~15ull;// multiple of 16 (uint32 words / fp8 quads / f32)
}

// Upload `bytes` (host staging) into a float buffer in bounded chunks.
[[nodiscard]] Status upload_f32_stream(compute::Stream &stream, compute::Buffer<float> &dst,
                                       GgufReader &reader, const GgufTensorInfo &info,
                                       luisa::vector<float> &staging) {
    const uint64_t total = info.element_count;
    uint64_t done = 0;
    while (done < total) {
        const uint64_t n = std::min<uint64_t>(staging.size(), total - done);
        if (info.type == static_cast<uint32_t>(GgufTensorType::f32)) {
            if (auto s = reader.read_tensor_bytes_at(
                    info, done * 4u,
                    luisa::span<std::byte>{reinterpret_cast<std::byte *>(staging.data()),
                                           static_cast<size_t>(n * 4u)});
                !s.ok()) {
                return s;
            }
        } else {// BF16 -> explicit decode
            luisa::vector<uint16_t> raw(static_cast<size_t>(n));
            if (auto s = reader.read_tensor_bytes_at(
                    info, done * 2u,
                    luisa::span<std::byte>{reinterpret_cast<std::byte *>(raw.data()),
                                           static_cast<size_t>(n * 2u)});
                !s.ok()) {
                return s;
            }
            for (uint64_t i = 0; i < n; ++i) { staging[i] = bf16_to_f32(raw[i]); }
        }
        stream << dst.view(static_cast<size_t>(done), static_cast<size_t>(n))
                      .copy_from(luisa::span<const float>{staging.data(), static_cast<size_t>(n)});
        done += n;
    }
    return Status::success();
}

}// namespace

// ---------------------------------------------------------------------------
// Host packers
// ---------------------------------------------------------------------------
[[nodiscard]] Status pack_bf16_tensor(luisa::span<const std::byte> bytes,
                                      uint64_t element_count,
                                      luisa::span<uint32_t> packed_out) noexcept {
    uint64_t expect_bytes = 0;
    if (!checked_mul(element_count, 2u, expect_bytes) ||
        static_cast<uint64_t>(bytes.size()) != expect_bytes) {
        return Status::error(StatusCode::invalid_argument,
                             "bf16 payload byte size mismatch", "bytes");
    }
    const uint64_t words = (element_count + 1u) / 2u;
    if (static_cast<uint64_t>(packed_out.size()) != words) {
        return Status::error(StatusCode::invalid_argument,
                             "bf16 packed output size mismatch", "packed_out");
    }
    // Endian-explicit: GGUF BF16 codes are little-endian uint16 on disk; the
    // ABI word is lo | hi<<16 with the low element first.
    for (uint64_t w = 0; w < words; ++w) {
        uint32_t word = 0;
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint64_t i = 2u * w + h;
            if (i < element_count) {
                const auto lo = static_cast<uint32_t>(bytes[2u * i]);
                const auto hi = static_cast<uint32_t>(bytes[2u * i + 1u]);
                word |= (lo | (hi << 8u)) << (16u * h);
            }
        }
        packed_out[w] = word;
    }
    return Status::success();
}

[[nodiscard]] Status quantize_fp8_row(luisa::span<const float> values,
                                      luisa::span<uint8_t> codes_out,
                                      luisa::span<float> scales_out) noexcept {
    const uint64_t k = values.size();
    if (static_cast<uint64_t>(codes_out.size()) != k) {
        return Status::error(StatusCode::invalid_argument,
                             "fp8 row codes output size mismatch", "codes_out");
    }
    const uint64_t groups = fp8_group_count(k);
    if (static_cast<uint64_t>(scales_out.size()) != groups) {
        return Status::error(StatusCode::invalid_argument,
                             "fp8 row scales output size mismatch", "scales_out");
    }
    for (uint64_t i = 0; i < k; ++i) {
        if (!std::isfinite(values[i])) {
            return Status::error(StatusCode::invalid_argument,
                                 "fp8 quantization input is not finite", "values");
        }
    }
    for (uint64_t g = 0; g < groups; ++g) {
        const uint64_t begin = g * kFp8GroupSize;
        const uint64_t end = std::min<uint64_t>(begin + kFp8GroupSize, k);
        float max_abs = 0.0f;
        for (uint64_t i = begin; i < end; ++i) {
            max_abs = std::max(max_abs, std::abs(values[i]));
        }
        // scale = max_abs / 448; zero groups get scale 0 and all-zero codes.
        const float scale = max_abs / kFp8E4M3Max;
        scales_out[g] = scale;
        if (scale == 0.0f) {
            for (uint64_t i = begin; i < end; ++i) { codes_out[i] = 0u; }
            continue;
        }
        for (uint64_t i = begin; i < end; ++i) {
            // values[i]/scale is bounded by 448 in exact math; the encoder
            // saturates any rounding overshoot to +/-448 as well.
            codes_out[i] = f32_to_fp8_e4m3(values[i] / scale);
        }
    }
    return Status::success();
}

[[nodiscard]] Status quantize_fp8_matrix(luisa::span<const float> values,
                                         uint64_t k, uint64_t n,
                                         luisa::span<uint32_t> packed_out,
                                         luisa::span<float> scales_out) noexcept {
    uint64_t elements = 0;
    if (!checked_mul(k, n, elements) || static_cast<uint64_t>(values.size()) != elements) {
        return Status::error(StatusCode::invalid_argument,
                             "fp8 matrix value count mismatch", "values");
    }
    const uint64_t words = (elements + 3u) / 4u;
    if (static_cast<uint64_t>(packed_out.size()) != words) {
        return Status::error(StatusCode::invalid_argument,
                             "fp8 matrix packed output size mismatch", "packed_out");
    }
    uint64_t groups_per_row = 0, scale_count = 0;
    groups_per_row = fp8_group_count(k);
    if (!checked_mul(groups_per_row, n, scale_count) ||
        static_cast<uint64_t>(scales_out.size()) != scale_count) {
        return Status::error(StatusCode::invalid_argument,
                             "fp8 matrix scales output size mismatch", "scales_out");
    }
    if (k == 0u || n == 0u) { return Status::success(); }
    luisa::vector<uint8_t> row_codes(static_cast<size_t>(k));
    for (uint64_t row = 0; row < n; ++row) {
        const auto row_values = values.subspan(static_cast<size_t>(row * k), static_cast<size_t>(k));
        const auto row_scales = scales_out.subspan(static_cast<size_t>(row * groups_per_row),
                                                   static_cast<size_t>(groups_per_row));
        if (auto s = quantize_fp8_row(row_values, row_codes, row_scales); !s.ok()) { return s; }
        // Pack this row's codes into the global quad stream. K is even for
        // every matrix in this model, but the packer handles any K: rows are
        // concatenated in element order before quad packing.
        for (uint64_t i = 0; i < k; ++i) {
            const uint64_t global = row * k + i;
            const uint64_t w = global / 4u;
            const uint32_t shift = 8u * (global % 4u);
            if (shift == 0u) { packed_out[w] = 0u; }
            packed_out[w] |= static_cast<uint32_t>(row_codes[i]) << shift;
        }
    }
    return Status::success();
}

[[nodiscard]] Status dequantize_fp8_matrix(luisa::span<const uint32_t> packed,
                                           luisa::span<const float> scales,
                                           uint64_t k, uint64_t n,
                                           luisa::span<float> out) noexcept {
    uint64_t elements = 0;
    if (!checked_mul(k, n, elements) || static_cast<uint64_t>(out.size()) != elements) {
        return Status::error(StatusCode::invalid_argument,
                             "fp8 dequant output size mismatch", "out");
    }
    if (static_cast<uint64_t>(packed.size()) != (elements + 3u) / 4u) {
        return Status::error(StatusCode::invalid_argument,
                             "fp8 dequant packed size mismatch", "packed");
    }
    const uint64_t groups_per_row = fp8_group_count(k);
    uint64_t scale_count = 0;
    if (!checked_mul(groups_per_row, n, scale_count) ||
        static_cast<uint64_t>(scales.size()) != scale_count) {
        return Status::error(StatusCode::invalid_argument,
                             "fp8 dequant scales size mismatch", "scales");
    }
    for (uint64_t row = 0; row < n; ++row) {
        for (uint64_t i = 0; i < k; ++i) {
            const uint64_t global = row * k + i;
            const uint8_t code = static_cast<uint8_t>((packed[global / 4u] >> (8u * (global % 4u))) & 0xffu);
            const float scale = scales[row * groups_per_row + i / kFp8GroupSize];
            out[global] = fp8_e4m3_to_f32(code) * scale;
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Device upload
// ---------------------------------------------------------------------------
[[nodiscard]] Result<QwenDeviceWeights> QwenDeviceWeights::upload_refs(
    compute::Device &device, compute::Stream &stream,
    const GgufReader &reader, luisa::span<const QwenTensorRef> refs,
    const QwenWeightUploadOptions &options,
    QwenWeightUploadReport *report) {
    QwenDeviceWeights weights;
    // Declared AFTER weights: on any early failure return this guard is
    // destroyed FIRST and quiesces borrowed staging copies before the partial
    // device tensors are released. Successful uploads retain the caller-sync
    // contract and pay no extra full-stream wait here.
    struct FailureSync {
        compute::Stream &stream;
        bool completed{false};
        ~FailureSync() noexcept {
            if (!completed) { stream.synchronize(); }
        }
    } failure_sync{stream};
    QwenWeightUploadReport local_report;
    auto &rep = report != nullptr ? *report : local_report;
    rep = QwenWeightUploadReport{};
    const uint64_t chunk_bytes = clamp_chunk_bytes(options.chunk_bytes);
    // The reader is logically const for payload reads; the API takes a
    // non-const reader because FILE* positioning is internal state.
    auto &mutable_reader = const_cast<GgufReader &>(reader);

      for (const auto &ref : refs) {
          const auto &info = *ref.info;
          // OFFLINE merged input projection (opt_plan/03 section 3): a tensor that a
          // merged matrix covers is NOT uploaded at all. The merged payload holds the
          // same bytes (a merged row IS the source row), so the session's device
          // footprint stays the same instead of doubling, and every consumer keeps
          // binding its segment alias (see append_merged_projections).
          if (options.projection_pack != nullptr &&
              projection_pack_covers(*options.projection_pack->manifest, ref.role, ref.layer)) {
              rep.projection_pack_skipped_tensors++;
              continue;
          }
          auto storage = classify_weight_storage(info.name, info.type, info.dims, options.fp8, options.q8_0);
        if (!storage.ok()) { return Result<QwenDeviceWeights>::failure(std::move(storage.status)); }
        QwenDeviceTensor t;
        t.role = ref.role;
        t.layer = ref.layer;
        t.name = info.name;
        t.storage = *storage.value;
        t.dims = info.dims;
        t.element_count = info.element_count;

        if (t.storage == WeightStorage::f32_dense) {
            t.f32 = device.create_buffer<float>(static_cast<size_t>(info.element_count));
            if (!t.f32) {
                return Result<QwenDeviceWeights>::failure(StatusCode::out_of_memory,
                                                          "device buffer allocation failed", t.name);
            }
            t.f32.set_name(t.name.c_str());
            luisa::vector<float> staging(static_cast<size_t>(
                std::max<uint64_t>(1u, chunk_bytes / 4u)));
            if (info.type != static_cast<uint32_t>(GgufTensorType::f32) &&
                info.type != static_cast<uint32_t>(GgufTensorType::bf16)) {
                return Result<QwenDeviceWeights>::failure(
                    StatusCode::unsupported,
                    "dense F32 upload requires an F32/BF16 source (Q8_0 needs explicit dequantization)",
                    t.name);
            }
            if (auto s = upload_f32_stream(stream, t.f32, mutable_reader, info, staging); !s.ok()) {
                return Result<QwenDeviceWeights>::failure(std::move(s));
            }
            rep.f32_count++;
            rep.device_bytes += info.element_count * 4u;
        } else if (t.storage == WeightStorage::bf16_packed) {
            // Raw copy of the GGUF BF16 payload into the uint32 pair ABI.
            const uint64_t words = (info.element_count + 1u) / 2u;
            t.packed = device.create_buffer<uint32_t>(static_cast<size_t>(words));
            if (!t.packed) {
                return Result<QwenDeviceWeights>::failure(StatusCode::out_of_memory,
                                                          "device buffer allocation failed", t.name);
            }
            t.packed.set_name(t.name.c_str());
            luisa::vector<uint32_t> staging(static_cast<size_t>(
                std::max<uint64_t>(1u, chunk_bytes / 4u)));
            uint64_t done = 0;// elements
            while (done < info.element_count) {
                const uint64_t n = std::min<uint64_t>(staging.size() * 2u,
                                                      info.element_count - done);
                // Read raw bytes and pack endian-explicitly.
                luisa::vector<std::byte> raw(static_cast<size_t>(n * 2u));
                if (auto s = mutable_reader.read_tensor_bytes_at(info, done * 2u, raw); !s.ok()) {
                    return Result<QwenDeviceWeights>::failure(std::move(s));
                }
                const uint64_t chunk_words = (n + 1u) / 2u;
                if (auto s = pack_bf16_tensor(raw, n,
                                              luisa::span<uint32_t>{staging.data(),
                                                                    static_cast<size_t>(chunk_words)});
                    !s.ok()) {
                    return Result<QwenDeviceWeights>::failure(std::move(s));
                }
                const uint64_t word_off = done / 2u;
                stream << t.packed.view(static_cast<size_t>(word_off), static_cast<size_t>(chunk_words))
                              .copy_from(luisa::span<const uint32_t>{staging.data(),
                                                                     static_cast<size_t>(chunk_words)});
                done += n;
            }
            rep.bf16_count++;
            rep.device_bytes += words * 4u;
        } else if (t.storage == WeightStorage::q8_0_packed) {
            // NATIVE Q8_0: the GGUF int8 block codes are copied verbatim into
            // the 4-codes-per-word ABI and each block's FP16 scale is widened
            // exactly to FP32. ZERO host re-quantization: the device words are
            // byte-identical to the checkpoint payload.
            if (info.dims.size() != 2u) {
                return Result<QwenDeviceWeights>::failure(
                    StatusCode::internal_error,
                    "q8_0 storage selected for a non-matrix tensor", t.name);
            }
            const uint64_t k = info.dims[0];
            const uint64_t n = info.dims[1];
            if (k == 0u || k % kQ8_0BlockSize != 0u) {
                return Result<QwenDeviceWeights>::failure(
                    StatusCode::unsupported,
                    "q8_0 upload requires K % 32 == 0 (whole blocks and whole 8-word groups)",
                    t.name);
            }
            const uint64_t words = info.element_count / 4u;
            const uint64_t groups_per_row = q8_0_group_count(k);
            uint64_t scale_count = 0;
            if (!checked_mul(groups_per_row, n, scale_count)) {
                return Result<QwenDeviceWeights>::failure(StatusCode::invalid_argument,
                                                          "q8_0 scale count overflow", t.name);
            }
            t.packed = device.create_buffer<uint32_t>(static_cast<size_t>(words));
            t.scales = device.create_buffer<float>(static_cast<size_t>(scale_count));
            if (!t.packed || !t.scales) {
                return Result<QwenDeviceWeights>::failure(StatusCode::out_of_memory,
                                                          "device buffer allocation failed", t.name);
            }
            t.packed.set_name(t.name.c_str());
            t.scales.set_name((t.name + ".scales").c_str());
            t.groups_per_row = groups_per_row;
            // Stream WHOLE blocks: chunk = min(blocks remaining, blocks that fit
            // the staging budget). A block is 34 bytes, so the chunk boundary is
            // always a block boundary and no block is ever split.
            const uint64_t blocks = info.element_count / kQ8_0BlockSize;
            const uint64_t blocks_per_chunk = std::max<uint64_t>(
                1u, chunk_bytes / kQ8_0BlockBytes);
            luisa::vector<std::byte> raw;
            luisa::vector<uint32_t> word_staging;
            luisa::vector<float> scale_staging;
            uint64_t block = 0;
            while (block < blocks) {
                const uint64_t chunk_blocks = std::min<uint64_t>(blocks_per_chunk, blocks - block);
                const uint64_t chunk_elems = chunk_blocks * kQ8_0BlockSize;
                raw.resize(static_cast<size_t>(chunk_blocks * kQ8_0BlockBytes));
                if (auto s = mutable_reader.read_tensor_bytes_at(
                        info, block * kQ8_0BlockBytes, raw);
                    !s.ok()) {
                    return Result<QwenDeviceWeights>::failure(std::move(s));
                }
                word_staging.resize(static_cast<size_t>(chunk_elems / 4u));
                scale_staging.resize(static_cast<size_t>(chunk_blocks));
                if (auto s = pack_q8_0_blocks(raw, chunk_elems,
                                              luisa::span<uint32_t>{word_staging.data(),
                                                                    word_staging.size()},
                                              luisa::span<float>{scale_staging.data(),
                                                                 scale_staging.size()});
                    !s.ok()) {
                    return Result<QwenDeviceWeights>::failure(std::move(s));
                }
                const uint64_t word_off = (block * kQ8_0BlockSize) / 4u;
                stream << t.packed.view(static_cast<size_t>(word_off),
                                        static_cast<size_t>(word_staging.size()))
                              .copy_from(luisa::span<const uint32_t>{word_staging.data(),
                                                                     word_staging.size()});
                stream << t.scales.view(static_cast<size_t>(block),
                                        static_cast<size_t>(scale_staging.size()))
                              .copy_from(luisa::span<const float>{scale_staging.data(),
                                                                  scale_staging.size()});
                block += chunk_blocks;
            }
            rep.q8_0_count++;
            rep.device_bytes += words * 4u + scale_count * 4u;
        } else {
            // FP8 E4M3: quantize from the BF16/F32 source at load time.
            // Matrices only: dims [K, N].
            if (info.dims.size() != 2u) {
                return Result<QwenDeviceWeights>::failure(
                    StatusCode::internal_error,
                    "fp8 storage selected for a non-matrix tensor", t.name);
            }
            const uint64_t k = info.dims[0];
            const uint64_t n = info.dims[1];
            const uint64_t words = (info.element_count + 3u) / 4u;
            const uint64_t groups_per_row = fp8_group_count(k);
            uint64_t scale_count = 0;
            if (!checked_mul(groups_per_row, n, scale_count)) {
                return Result<QwenDeviceWeights>::failure(StatusCode::invalid_argument,
                                                          "fp8 scale count overflow", t.name);
            }
            t.packed = device.create_buffer<uint32_t>(static_cast<size_t>(words));
            t.scales = device.create_buffer<float>(static_cast<size_t>(scale_count));
            if (!t.packed || !t.scales) {
                return Result<QwenDeviceWeights>::failure(StatusCode::out_of_memory,
                                                          "device buffer allocation failed", t.name);
            }
            t.packed.set_name(t.name.c_str());
            t.scales.set_name((t.name + ".scales").c_str());
            t.groups_per_row = groups_per_row;
            // Stream row-blocks: decode a bounded set of rows to FP32,
            // quantize, upload packed words + scales. Rows are processed in
            // order so the global quad stream stays deterministic.
            const uint64_t rows_per_chunk = std::max<uint64_t>(1u, chunk_bytes / (4u * std::max<uint64_t>(k, 1u)));
            luisa::vector<float> row_values;
            luisa::vector<uint32_t> packed_staging;
            luisa::vector<float> scale_staging;
            uint64_t row = 0;
            while (row < n) {
                const uint64_t rows = std::min<uint64_t>(rows_per_chunk, n - row);
                const uint64_t elems = rows * k;
                row_values.resize(static_cast<size_t>(elems));
                // Decode the row block from the GGUF (BF16 or F32 source).
                const uint64_t elem_off = row * k;
                if (info.type == static_cast<uint32_t>(GgufTensorType::f32)) {
                    if (auto s = mutable_reader.read_tensor_bytes_at(
                            info, elem_off * 4u,
                            luisa::span<std::byte>{reinterpret_cast<std::byte *>(row_values.data()),
                                                   static_cast<size_t>(elems * 4u)});
                        !s.ok()) {
                        return Result<QwenDeviceWeights>::failure(std::move(s));
                    }
                } else {
                    luisa::vector<uint16_t> raw(static_cast<size_t>(elems));
                    if (auto s = mutable_reader.read_tensor_bytes_at(
                            info, elem_off * 2u,
                            luisa::span<std::byte>{reinterpret_cast<std::byte *>(raw.data()),
                                                   static_cast<size_t>(elems * 2u)});
                        !s.ok()) {
                        return Result<QwenDeviceWeights>::failure(std::move(s));
                    }
                    for (uint64_t i = 0; i < elems; ++i) { row_values[i] = bf16_to_f32(raw[i]); }
                }
                // Quantize row by row into a contiguous code block, then
                // pack the whole block (row boundaries are element-order
                // concatenation, matching quantize_fp8_matrix).
                luisa::vector<uint8_t> block_codes(static_cast<size_t>(elems));
                scale_staging.resize(static_cast<size_t>(rows * groups_per_row));
                for (uint64_t r = 0; r < rows; ++r) {
                    if (auto s = quantize_fp8_row(
                            luisa::span<const float>{row_values.data() + r * k, static_cast<size_t>(k)},
                            luisa::span<uint8_t>{block_codes.data() + r * k, static_cast<size_t>(k)},
                            luisa::span<float>{scale_staging.data() + r * groups_per_row,
                                               static_cast<size_t>(groups_per_row)});
                        !s.ok()) {
                        return Result<QwenDeviceWeights>::failure(std::move(s));
                    }
                }
                // The block starts at element row*k; quad alignment of the
                // block start is (row*k) % 4. For this model K is a multiple
                // of 4, so blocks are quad-aligned; handle the general case
                // by packing into a scratch quad span with the same global
                // offsets.
                const uint64_t global_begin = row * k;
                const uint64_t global_end = global_begin + elems;
                const uint64_t word_begin = global_begin / 4u;
                const uint64_t word_end = (global_end + 3u) / 4u;
                packed_staging.assign(static_cast<size_t>(word_end - word_begin), 0u);
                // Preserve partial-word bytes at the block edges: for K % 4
                // != 0 the first/last word may share bytes with neighboring
                // blocks. Read-modify-write is avoided by requiring quad-
                // aligned row starts (true for every Qwen3.5 matrix).
                if (global_begin % 4u != 0u) {
                    return Result<QwenDeviceWeights>::failure(
                        StatusCode::unsupported,
                        "fp8 streaming requires K % 4 == 0 (row quad alignment)", t.name);
                }
                for (uint64_t i = 0; i < elems; ++i) {
                    const uint64_t global = global_begin + i;
                    packed_staging[static_cast<size_t>(global / 4u - word_begin)] |=
                        static_cast<uint32_t>(block_codes[i]) << (8u * (global % 4u));
                }
                stream << t.packed.view(static_cast<size_t>(word_begin),
                                        static_cast<size_t>(word_end - word_begin))
                              .copy_from(luisa::span<const uint32_t>{packed_staging.data(),
                                                                     packed_staging.size()});
                stream << t.scales.view(static_cast<size_t>(row * groups_per_row),
                                        static_cast<size_t>(rows * groups_per_row))
                              .copy_from(luisa::span<const float>{scale_staging.data(),
                                                                  scale_staging.size()});
                row += rows;
            }
            rep.fp8_count++;
            rep.device_bytes += words * 4u + scale_count * 4u;
            rep.diagnostics.emplace_back(Diagnostic{
                DiagnosticSeverity::info, "fp8_quantized",
                "matrix quantized from " + luisa::string{gguf_tensor_type_name(info.type)} +
                    " to FP8 E4M3 at load time (per-row groups of 64, FP32 scales): " + t.name});
        }
          rep.tensor_count++;
          weights._tensors.emplace_back(std::move(t));
      }
      // OFFLINE merged input projections: append the merged matrices and their
      // per-segment aliases AFTER the ordinary tensors, so `find()` returns the
      // alias for every replaced role and the merged parent stays reachable through
      // merged_parent_of(). Nothing was appended on a failure.
      if (options.projection_pack != nullptr) {
          const auto &pack = *options.projection_pack;
          if (auto s = append_merged_projections(device, stream, pack.dir, *pack.manifest,
                                                weights._tensors, chunk_bytes, &rep);
              !s.ok()) {
              return Result<QwenDeviceWeights>::failure(std::move(s));
          }
          rep.projection_pack_format = luisa::string{weight_storage_name(pack.manifest->storage)};
          weights._projection_pack_source = pack.manifest->source_name;
          weights._projection_pack_format = rep.projection_pack_format;
          rep.diagnostics.emplace_back(Diagnostic{
              DiagnosticSeverity::info, "projection_pack",
              luisa::format("offline merged input projections: {} matrices, {} segment aliases "
                            "(sub-views, no duplicated bytes), {} source tensors skipped, "
                            "format {}",
                            rep.projection_pack_matrices, rep.projection_pack_segments,
                            rep.projection_pack_skipped_tensors, rep.projection_pack_format)});
      }
      failure_sync.completed = true;
    if (rep.q8_0_count != 0u) {
        rep.diagnostics.emplace_back(Diagnostic{
            DiagnosticSeverity::info, "q8_0_native",
            luisa::format("Q8_0 matrices uploaded as the checkpoint's own int8 block codes "
                          "(byte copy) with exact FP16->FP32 scale widening: {} matrices, "
                          "zero re-quantization",
                          rep.q8_0_count)});
    }
    return Result<QwenDeviceWeights>::success(std::move(weights));
}

  [[nodiscard]] Result<QwenDeviceWeights> QwenDeviceWeights::upload(
      compute::Device &device, compute::Stream &stream, const GgufReader &reader,
      const Qwen35Model &model, const QwenWeightUploadOptions &options,
      QwenWeightUploadReport *report) {
      // OFFLINE merged input projections (opt_plan/03 section 3 + D5). Every gate
      // that can be decided from the model + manifest is decided HERE, before a
      // single byte is uploaded: the pack must describe exactly the canonical
      // inventory of THIS model and must have been built in the format this upload
      // produces. A pack of the wrong format/geometry is a recoverable failure, never
      // a silent re-interpretation.
      if (options.projection_pack != nullptr) {
          using R = Result<QwenDeviceWeights>;
          const auto &pack = *options.projection_pack;
          if (pack.manifest == nullptr) {
                              return R::failure(StatusCode::invalid_argument,
                                  "projection pack pointer is set but its manifest is null");
          }
          auto inventory = qwen35_merged_projection_inventory(model);
          if (!inventory.ok()) { return R::failure(std::move(inventory.status)); }
          if (auto s = validate_projection_pack_manifest(*pack.manifest, *inventory.value); !s.ok()) {
              return R::failure(std::move(s));
          }
          const auto want = options.q8_0 ? WeightStorage::q8_0_packed
                            : options.fp8 ? WeightStorage::fp8_packed
                                          : WeightStorage::bf16_packed;
          if (pack.manifest->storage != want) {
              return R::failure(Status::error(
                  StatusCode::invalid_argument,
                  "the projection pack was built in a different weight format than this "
                  "variant uploads (refusing to re-quantize a merged matrix)",
                  luisa::string{weight_storage_name(pack.manifest->storage)} + " vs " +
                      luisa::string{weight_storage_name(want)}));
          }
      }
      return upload_refs(device, stream, reader, model.tensors(), options, report);
  }

[[nodiscard]] Result<QwenDeviceWeights> QwenDeviceWeights::upload_nextn(
    compute::Device &device, compute::Stream &stream, const GgufReader &reader,
    const QwenWeightUploadOptions &options, QwenWeightUploadReport *report) {
    using R = Result<QwenDeviceWeights>;
    const auto eligibility = probe_mtp_eligibility(reader);
    if (!eligibility.benchmark_feasible) {
        return R::failure(StatusCode::unsupported,
                          "NextN upload requires a complete draft and validated target trunk",
                          eligibility.reason);
    }
    struct Spec {
        const char *suffix;
        QwenTensorRole role;
        bool required;
    };
    static constexpr Spec specs[] = {
        {"eh_proj.weight", QwenTensorRole::gdn_qkv, true},
        {"enorm.weight", QwenTensorRole::gdn_norm, true},
        {"hnorm.weight", QwenTensorRole::gdn_a, true},
        {"shared_head.norm.weight", QwenTensorRole::output_norm, true},
        {"attn_norm.weight", QwenTensorRole::attn_norm, true},
        {"post_attention_norm.weight", QwenTensorRole::post_attention_norm, true},
        {"attn_q.weight", QwenTensorRole::attn_q, true},
        {"attn_k.weight", QwenTensorRole::attn_k, true},
        {"attn_v.weight", QwenTensorRole::attn_v, true},
        {"attn_output.weight", QwenTensorRole::attn_output, true},
        {"attn_q_norm.weight", QwenTensorRole::attn_q_norm, true},
        {"attn_k_norm.weight", QwenTensorRole::attn_k_norm, true},
        {"ffn_gate.weight", QwenTensorRole::ffn_gate, true},
        {"ffn_up.weight", QwenTensorRole::ffn_up, true},
        {"ffn_down.weight", QwenTensorRole::ffn_down, true},
        {"token_embd.weight", QwenTensorRole::token_embd, false},
        {"output.weight", QwenTensorRole::token_embd, false},
    };
      luisa::vector<QwenTensorRef> refs;
      for (const auto &spec : specs) {
          const auto name = luisa::string{"blk.32."} + spec.suffix;
          const auto *info = reader.find(name);
          if (info == nullptr) {
              if (spec.required) {
                  return R::failure(StatusCode::unsupported, "required NextN tensor disappeared after validation", name);
              }
              continue;
          }
          refs.emplace_back(QwenTensorRef{spec.role, kMtpDraftBlockIndex, info});
      }
      // The draft block is never merged (the merged projection inventory covers the
      // 32 trunk layers), so the NextN upload must NOT consume the pack: it would
      // otherwise append a second copy of every merged matrix.
      auto draft_options = options;
      draft_options.projection_pack = nullptr;
      return upload_refs(device, stream, reader, luisa::span<const QwenTensorRef>{refs}, draft_options, report);
}

[[nodiscard]] const QwenDeviceTensor *QwenDeviceWeights::find(luisa::string_view name) const noexcept {
    for (const auto &t : _tensors) {
        if (t.name == name) { return &t; }
    }
    return nullptr;
}

  [[nodiscard]] const QwenDeviceTensor *QwenDeviceWeights::find(QwenTensorRole role,
                                                                uint64_t layer) const noexcept {
      for (const auto &t : _tensors) {
          // A merged MATRIX is never the answer to a role lookup: its segments keep
          // the roles and their aliases are what consumers bind. The parent is
          // reachable through merged_parent_of()/merged_matrices().
          if (t.merge_kind != QwenMergeKind::none) { continue; }
          if (t.role == role && t.layer == layer) { return &t; }
      }
      return nullptr;
  }

  [[nodiscard]] const QwenDeviceTensor *QwenDeviceWeights::merged_parent_of(
      const QwenDeviceTensor &t) const noexcept {
      if (t.merge_parent_index >= _tensors.size()) { return nullptr; }
      const auto &parent = _tensors[static_cast<size_t>(t.merge_parent_index)];
      return &parent;
  }

  [[nodiscard]] luisa::vector<const QwenDeviceTensor *> QwenDeviceWeights::merged_matrices() const noexcept {
      luisa::vector<const QwenDeviceTensor *> out;
      for (const auto &t : _tensors) {
          if (t.merge_kind != QwenMergeKind::none) { out.emplace_back(&t); }
      }
      return out;
  }

  }// namespace luisa::qwen
