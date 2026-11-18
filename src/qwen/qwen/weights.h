#pragma once
// Qwen3.5-4B shared engine layer -- device weight upload (storage ABI).
//
// Storage contract (see .kimix_cache/qwen_implementation_contract.md):
// - Dense F32 tensors (norms, conv, ssm_a/dt, any F32-stored tensor) upload
//   as Buffer<float>, one element per value.
// - BF16 matrices upload as Buffer<uint32_t>: two BF16 code units per
//   little-endian word, LOW element first. This is a raw copy of the GGUF
//   BF16 payload bytes (the GGUF stores exactly this layout on LE hosts).
// - FP8 matrices upload as Buffer<uint32_t>: four true E4M3 codes per
//   little-endian word, LOW element first, plus a Buffer<float> of FP32
//   scales, one per contiguous group of 64 K elements per output row
//   (scale = max_abs / 448; dequantized weight = fp8(code) * scale).
//   FP8 conversion happens at load time from the BF16/F32 source and is
//   REPORTED via a diagnostic -- it is never silent, and Q8_0 is never
//   reinterpreted as FP8.
// - Q8_0 sources are rejected by the uploader (explicit dequantize +
//   re-quantize tooling is a separate, reporting step; double quantization
//   must be visible). The GGUF reader exposes read_tensor_q8_0_f32 for that.
//
// Streaming: tensors are read from the GGUF one at a time in bounded host
// chunks (default 4 MiB); the multi-GB file is never held in host RAM.
// Uploads are enqueued on the caller's Stream; call synchronize() (or use
// the returned report after a stream synchronize) before consuming weights.
//
// Host-side packers (pack_bf16_tensor / quantize_fp8_rows) are exposed for
// tests and offline conversion; they operate on caller-provided spans.

#include <cstdint>
#include <utility>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "qwen/common.h"
#include "qwen/gguf.h"
#include "qwen/model.h"

namespace luisa::qwen {

// One uploaded tensor. Exactly one of the storage forms is populated:
// - f32_dense: `f32` holds element_count floats.
// - bf16_packed: `packed` holds ceil(element_count/2) uint32 words.
// - fp8_packed: `packed` holds ceil(element_count/4) uint32 words and
//   `scales` holds N * ceil(K/64) floats (row-major over [K, N] matrices).
struct QwenDeviceTensor {
    QwenTensorRole role{QwenTensorRole::token_embd};
    uint64_t layer{UINT64_MAX};
    luisa::string name;
    WeightStorage storage{WeightStorage::f32_dense};
    luisa::vector<uint64_t> dims;// ne order, as in the GGUF
    uint64_t element_count{0};
    compute::Buffer<float> f32;
    compute::Buffer<uint32_t> packed;
    compute::Buffer<float> scales;
    // FP8 only: groups per output row (ceil(K/64)); 0 otherwise.
    uint64_t groups_per_row{0};
};

struct QwenWeightUploadOptions {
    // Select the FP8 E4M3 variant for eligible BF16/F32 matrices. Sensitive
    // tensors stay dense F32 in both variants.
    bool fp8{false};
    // Host staging chunk size in bytes (bounded streaming). Clamped to
    // [64 KiB, 64 MiB]; rounded down to a multiple of 16.
    uint64_t chunk_bytes{4ull * 1024ull * 1024ull};
};

struct QwenWeightUploadReport {
    uint64_t tensor_count{0};
    uint64_t f32_count{0};
    uint64_t bf16_count{0};
    uint64_t fp8_count{0};
    uint64_t device_bytes{0};
    // Non-fatal diagnostics, e.g. "fp8_quantized" for every matrix converted
    // from BF16/F32 to E4M3 at load time (the conversion is never silent).
    luisa::vector<Diagnostic> diagnostics;
};

class QwenDeviceWeights {
public:
    // Upload every tensor of the validated model, one tensor at a time.
    // On failure any buffers created so far are released with the partial
    // result; the Status explains the first failure.
    [[nodiscard]] static Result<QwenDeviceWeights> upload(
        compute::Device &device, compute::Stream &stream,
        const GgufReader &reader, const Qwen35Model &model,
        const QwenWeightUploadOptions &options,
        QwenWeightUploadReport *report = nullptr);

    // Upload the canonical blk.32.* NextN draft tensors after the complete
    // MTP eligibility probe has validated the combined GGUF. The ordinary
    // upload above still reads only the 426 trunk tensors from the model.
    [[nodiscard]] static Result<QwenDeviceWeights> upload_nextn(
        compute::Device &device, compute::Stream &stream,
        const GgufReader &reader, const QwenWeightUploadOptions &options,
        QwenWeightUploadReport *report = nullptr);

    // Inject already-owned device tensors for deterministic operator/executor
    // tests and offline tools. Consumers (engine/draft create) still validate
    // every name, shape, storage kind and buffer extent; no checkpoint is
    // fabricated or reported by this helper.
    [[nodiscard]] static QwenDeviceWeights from_device_tensors(
        luisa::vector<QwenDeviceTensor> tensors) noexcept {
        QwenDeviceWeights weights;
        weights._tensors = std::move(tensors);
        return weights;
    }

    [[nodiscard]] luisa::span<const QwenDeviceTensor> tensors() const noexcept { return _tensors; }
    [[nodiscard]] const QwenDeviceTensor *find(luisa::string_view name) const noexcept;
    [[nodiscard]] const QwenDeviceTensor *find(QwenTensorRole role,
                                               uint64_t layer = UINT64_MAX) const noexcept;

private:
    [[nodiscard]] static Result<QwenDeviceWeights> upload_refs(
        compute::Device &device, compute::Stream &stream, const GgufReader &reader,
        luisa::span<const QwenTensorRef> refs, const QwenWeightUploadOptions &options,
        QwenWeightUploadReport *report);

    luisa::vector<QwenDeviceTensor> _tensors;
};

// ---------------------------------------------------------------------------
// Host-side packers (also used by the uploader; exposed for tests/tools).
// ---------------------------------------------------------------------------
// Pack a BF16 GGUF payload (raw bytes, 2 per element) into the uint32 pair
// ABI. `bytes.size()` must be exactly 2*element_count; `packed_out.size()`
// must be ceil(element_count/2). On little-endian hosts this is a raw copy;
// the implementation is endian-explicit regardless.
[[nodiscard]] Status pack_bf16_tensor(luisa::span<const std::byte> bytes,
                                      uint64_t element_count,
                                      luisa::span<uint32_t> packed_out) noexcept;

// Quantize one matrix row of K FP32 values to E4M3 with per-group-of-64
// scales. codes_out.size() == K, scales_out.size() == ceil(K/64).
// Zero groups get scale 0 (all codes 0); finite values saturate at +/-448
// after scaling; NaN/inf inputs are rejected (recoverable).
[[nodiscard]] Status quantize_fp8_row(luisa::span<const float> values,
                                      luisa::span<uint8_t> codes_out,
                                      luisa::span<float> scales_out) noexcept;

// Quantize a whole [K, N] matrix (row-major, row stride K) to the packed
// FP8 ABI. packed_out.size() == ceil(K*N/4), scales_out.size() ==
// N * ceil(K/64). Rows are quantized independently.
[[nodiscard]] Status quantize_fp8_matrix(luisa::span<const float> values,
                                         uint64_t k, uint64_t n,
                                         luisa::span<uint32_t> packed_out,
                                         luisa::span<float> scales_out) noexcept;

// Dequantize the packed FP8 ABI back to FP32 (host reference for tests).
[[nodiscard]] Status dequantize_fp8_matrix(luisa::span<const uint32_t> packed,
                                           luisa::span<const float> scales,
                                           uint64_t k, uint64_t n,
                                           luisa::span<float> out) noexcept;

}// namespace luisa::qwen
