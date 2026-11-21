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
// - Q8_0 matrices upload as Buffer<uint32_t> of ceil(K*N/4) words, four
//   SIGNED int8 codes per little-endian word, LOW element first, plus a
//   Buffer<float> of FP32 scales, one per 32-element block along K per output
//   row (K/32 per row). The words are a BYTE COPY of the GGUF block's int8
//   payload and the scale is the exact FP16->FP32 widening of the block's
//   FP16 scale: the loader performs ZERO host re-quantization, so
//   dequantized weight = int8(code) * scale reproduces the checkpoint
//   bit for bit. Only whole blocks are allowed (K % 32 == 0), which also makes
//   every 32-element block one packed-word group of 8 words.
// - W4A16 matrices (WeightStorage::w4_packed) are uploaded from an OFFLINE W4
//   PACK (qwen/w4_pack.h), never quantized inside a request: 8 signed int4 codes
//   per uint32 word (low nibble first) plus one FP32 scale per group of
//   `group` K elements per output row, produced by the deterministic clipping
//   search in common.h. A dual-copy pack additionally carries the FP8 payload of
//   every matrix (a separate tensor named "<name>.fp8"), which the engine uses
//   for prefill and for the LM head (plan option (c)).
// - Q8_0 sources are rejected unless QwenWeightUploadOptions::q8_0 is set
//   (the dsl-q8 variant). Without that opt-in the rejection is explicit
//   (explicit dequantize + re-quantize tooling is a separate, reporting step;
//   double quantization must be visible). The GGUF reader exposes
//   read_tensor_q8_0_f32 for host references.
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
  #include "qwen/projection_pack.h"

namespace luisa::qwen {

// One uploaded tensor. Exactly one of the storage forms is populated:
// - f32_dense: `f32` holds element_count floats.
// - bf16_packed: `packed` holds ceil(element_count/2) uint32 words.
// - fp8_packed: `packed` holds ceil(element_count/4) uint32 words and
//   `scales` holds N * ceil(K/64) floats (row-major over [K, N] matrices).
// - q8_0_packed: `packed` holds element_count/4 uint32 words (4 signed int8
//   codes each, byte-copied from the GGUF blocks) and `scales` holds
//   N * (K/32) floats (row-major over [K, N] matrices), one per 32-element
//   block, each the exact FP16->FP32 widening of the checkpoint's block scale.
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
    // fp8_packed: groups per output row (ceil(K/64)); q8_0_packed: K/32; 0 else.
    // w4_packed: K / group.
    uint64_t groups_per_row{0};
      // Scale group along K this tensor was packed with (0 = the loader did not
      // record one, in which case consumers fall back to the per-format canonical
      // group: fp8 64, q8_0 32, w4 128). A W4 pack manifest records 32/64/128.
      uint32_t group{0};
      // ---- offline merged input projection (opt_plan/03 section 3 + D5) ------
      // * merge_kind != none: this tensor IS a merged matrix (name
      //   "blk.<layer>.<kind>.merged"); `segments` is its first-class descriptor:
      //   the source name, role, row count and row offset of every segment, in the
      //   canonical order of qwen/model.h (which is also the payload row order).
      // * merge_parent_index != UINT64_MAX: this tensor is a first-class PER-SEGMENT
      //   ALIAS of tensors[merge_parent_index]: `packed`/`scales` are SUB-VIEWS of
      //   the merged buffers (no byte is duplicated) while role/name/dims/storage
      //   are the SOURCE tensor's, so `find(role, layer)`, the segment descriptor
      //   record and every existing projection consumer keep working unchanged.
      //   `merge_kind`/`segments` are empty on an alias; `merge_row_offset` is the
      //   alias's first row inside the merged matrix.
      QwenMergeKind merge_kind{QwenMergeKind::none};
      uint64_t merge_parent_index{UINT64_MAX};
      uint64_t merge_row_offset{0};
      luisa::vector<QwenTensorSegment> segments;
      // Sub-view REALIZATION of `packed`/`scales` of a segment alias. An alias's
      // `packed`/`scales` are the PARENT's buffers (so extents/validity stay
      // checkable), and these views select the alias's row range inside them.
      // Consumers bind them through packed_view_of()/scales_view_of(), which return
      // the whole buffer for an ordinary tensor. Both are empty for a parent/ordinary
      // tensor.
        compute::BufferView<uint32_t> packed_view;
        compute::BufferView<float> scales_view;
  };

  // The packed-weight view a consumer must bind: the alias's row-range sub-view of
  // the merged matrix when `t` is a segment alias, otherwise the whole buffer. This
  // is the ONLY place the "merged or not" distinction is resolved, so a consumer
  // never has to know whether a matrix came from a pack.
  [[nodiscard]] inline compute::BufferView<uint32_t> packed_view_of(const QwenDeviceTensor &t) noexcept {
      return t.merge_parent_index == UINT64_MAX ? t.packed.view() : t.packed_view;
  }
  // Same for the FP32 scales. Returns an empty view when the tensor has no scales
  // (bf16), so callers keep their `t.scales ? ... : dummy` shape by testing `t.scales`.
  [[nodiscard]] inline compute::BufferView<float> scales_view_of(const QwenDeviceTensor &t) noexcept {
      return t.merge_parent_index == UINT64_MAX ? t.scales.view() : t.scales_view;
  }
  // Element count of the packed view a consumer binds (row range of an alias).
  [[nodiscard]] inline uint64_t packed_element_count(const QwenDeviceTensor &t) noexcept {
      return t.merge_parent_index == UINT64_MAX ? (t.packed ? t.packed.size() : 0u)
                                                : static_cast<uint64_t>(t.packed_view.size());
  }

  struct QwenWeightUploadOptions {
      // Select the FP8 E4M3 variant for eligible BF16/F32 matrices. Sensitive
    // tensors stay dense F32 in both variants.
    bool fp8{false};
    // Select the NATIVE Q8_0 device format for eligible Q8_0 matrices: the
    // GGUF int8 block codes are copied verbatim and the FP16 block scales are
    // widened exactly to FP32. No host re-quantization, no FP8 routing.
    // Mutually exclusive with `fp8`. False keeps the historical explicit
    // rejection of Q8_0 checkpoints.
    bool q8_0{false};
      // Host staging chunk size in bytes (bounded streaming). Clamped to
      // [64 KiB, 64 MiB]; rounded down to a multiple of 16.
      uint64_t chunk_bytes{4ull * 1024ull * 1024ull};
      // OFFLINE merged input projections (opt_plan/03 section 3 + D5). When
      // nonnull the upload consumes this pack: every (role, layer) that a merged
      // segment covers is SKIPPED (never uploaded a second time) and the merged
      // matrix is uploaded from the pack payload instead, followed by one ALIAS
      // tensor per segment whose buffers are sub-views of the merged matrix. That
      // is what makes a merged session cost the same device bytes as the unmerged
      // one instead of twice as much. The pack must already have passed
      // verify_projection_pack_source() for THIS checkpoint, and its format must be
      // exactly the format this upload produces (fp8/bf16/q8_0) -- a mismatch is a
      // recoverable failure, never a substitution. W4 sessions refuse a pack (no
      // merged 4-bit realization exists).
      const QwenProjectionPackSource *projection_pack{nullptr};
  };

struct QwenWeightUploadReport {
    // Number of DEVICE tensors created by the upload. It always equals
    // f32_count + bf16_count + fp8_count + q8_0_count + w4_count, so a summary
    // line built from these counters is internally consistent. (A W4 pack creates
    // MORE device tensors than it has manifest records: one record can yield a W4
    // decode tensor AND its "<name>.fp8" prefill partner.)
    uint64_t tensor_count{0};
    // Number of manifest RECORDS consumed (one per GGUF tensor). For a W4 pack
    // this is smaller than tensor_count whenever FP8 partner payloads exist.
    uint64_t manifest_records{0};
    uint64_t f32_count{0};
    uint64_t bf16_count{0};
    uint64_t fp8_count{0};
    uint64_t q8_0_count{0};
      uint64_t w4_count{0};
      // Bytes actually allocated on the device (packed words + scale arrays).
      uint64_t device_bytes{0};
      // Offline merged projection pack (opt_plan/03 section 3): how many merged
      // matrices and segment aliases this upload created, and how many source
      // tensors it SKIPPED because a merged matrix replaced them. A merged session
      // reports projection_pack_matrices + projection_pack_segments >=
      // projection_pack_skipped_tensors (one alias per skipped tensor), and
      // device_bytes stays ~equal to the unmerged upload (aliases are sub-views).
      uint64_t projection_pack_matrices{0};
      uint64_t projection_pack_segments{0};
      uint64_t projection_pack_skipped_tensors{0};
      uint64_t projection_pack_bytes{0};
      // Format the pack was recorded in (empty when no pack was consumed).
      luisa::string projection_pack_format;
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
      // Merged matrix a per-segment ALIAS points at, or nullptr when `t` is not an
      // alias (or belongs to another weight set).
      [[nodiscard]] const QwenDeviceTensor *merged_parent_of(const QwenDeviceTensor &t) const noexcept;
      // Every merged matrix of this weight set (merge_kind != none), in upload order.
      [[nodiscard]] luisa::vector<const QwenDeviceTensor *> merged_matrices() const noexcept;
      // Provenance of the offline pack this weight set was uploaded from (empty when
      // none was consumed): the manifest's source checkpoint name and pack format.
      [[nodiscard]] luisa::string_view projection_pack_source() const noexcept { return _projection_pack_source; }
      [[nodiscard]] luisa::string_view projection_pack_format() const noexcept { return _projection_pack_format; }
  private:
      [[nodiscard]] static Result<QwenDeviceWeights> upload_refs(
          compute::Device &device, compute::Stream &stream, const GgufReader &reader,
          luisa::span<const QwenTensorRef> refs, const QwenWeightUploadOptions &options,
          QwenWeightUploadReport *report);
      // Append the merged projection tensors of a VERIFIED pack: one merged parent
      // per matrix (merge_kind + segments recorded) and one alias per segment
      // (sub-views into the parent, role/name/dims of the source). Built in a local
      // vector and spliced in ONLY on success, so a failed upload leaves nothing
      // behind. `base` is the index the first appended tensor will occupy.
      [[nodiscard]] static Status append_merged_projections(
          compute::Device &device, compute::Stream &stream,
          const luisa::filesystem::path &dir, const ProjectionPackManifest &manifest,
          luisa::vector<QwenDeviceTensor> &tensors, uint64_t chunk_bytes,
          QwenWeightUploadReport *report) noexcept;
      // True when a merged segment of `manifest` covers this (role, layer).
      [[nodiscard]] static bool projection_pack_covers(const ProjectionPackManifest &manifest,
                                                       QwenTensorRole role, uint64_t layer) noexcept;
        luisa::vector<QwenDeviceTensor> _tensors;
        luisa::string _projection_pack_source;
        luisa::string _projection_pack_format;
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
