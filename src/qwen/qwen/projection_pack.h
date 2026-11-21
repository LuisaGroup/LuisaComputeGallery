#pragma once
// Qwen3.5-4B -- OFFLINE merged input-projection pack
// (opt_plan/03-decode-latency.md section 3 "Fuse the packed projections" +
//  risk D5, opt_plan/05-quantization.md section 5 "per-backend weight layouts").
//
// WHAT IT IS
// ----------
// The two decode-time input projection groups are concatenated OFFLINE along N
// into one row-per-output matrix each (see qwen/model.h for the canonical segment
// order and the numerics argument):
//   gdn_qkv_z_ab  attn_qkv(8192) | attn_gate(4096) | ssm_alpha(32) | ssm_beta(32)
//                 -> one [2560, 12352] matrix per GDN layer (24 of them)
//   attn_qkv      attn_q(8192) | attn_k(1024) | attn_v(1024)
//                 -> one [2560, 10240] matrix per full-attention layer (8)
// A merged row IS the source row: the merge never re-quantizes, never reorders
// elements inside a row and never re-groups the per-row scales, so the merged
// payload is byte-identical to the concatenation of the source payloads, and a
// per-row (FP8/Q8_0) scale stays next to its own row.
//
// WHY A PACK AND NOT A LOAD-TIME REPACK
// -------------------------------------
// The plan requires the concatenation to happen offline: at 4 B parameters the
// repack reads/writes ~2 GiB and must not be inside a request. The pack also
// REPLACES the source tensors at upload (see QwenWeightUploadOptions::
// projection_pack), so the device never holds both the merged matrix and its
// segments.
//
// DIRECTORY LAYOUT (self-describing, manifest written LAST)
// --------------------------------------------------------
//   <dir>/manifest.txt                 line-oriented, order-stable, hashed
//   <dir>/matrices/<index>.bin         merged packed words [N][K/pack] uint32 LE
//   <dir>/matrices/<index>.scales.bin  merged FP32 scales [N][scales_per_row]
//                                      (fp8_e4m3 / q8_0 only; absent for bf16)
// The payloads are the DEVICE ABI itself (see qwen/weights.h and qwen/common.h):
// two BF16 code units / four E4M3 / four int8 codes per little-endian uint32 word,
// low element first, plus one FP32 scale per contiguous K group per output row.
// Streaming them into a device buffer is therefore a plain byte copy.
//
// MANIFEST CONTENTS (fail closed on any mismatch)
// -----------------------------------------------
// * version, format (bf16/fp8_e4m3/q8_0), scale group, code units per word, scales
//   per row and the segment order the payload follows;
// * the SOURCE checkpoint identity: file name, byte size and a ROLLING digest over
//   the raw stored payloads of exactly the merged source tensors, in inventory
//   order (the same rolling rule as the W4 pack: qwen/w4_pack.h
//   w4_roll_source_hash). A same-size/different-weights checkpoint is refused;
// * per merged matrix: name, kind, layer, k, n, K width, format, group, the word
//   and scale payload byte sizes + FNV-1a 64 hashes;
// * per segment: the SOURCE tensor name, role, layer, row count, row offset, the
//   source payload byte size + FNV-1a 64 hash, and the merged tensor's own name.
// W4 is NOT mergeable here: it stays an explicitly unsupported format (a W4 pack
// already carries its own dual payloads; see qwen/w4_pack.h).
#include <cstddef>
#include <cstdint>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include "qwen/common.h"
#include "qwen/gguf.h"
#include "qwen/model.h"

namespace luisa::qwen {

inline constexpr uint32_t kProjectionPackVersion = 1u;
// Streaming chunk bounds of the offline packer and of the payload upload.
inline constexpr uint64_t kProjectionPackMinChunkBytes = 64ull * 1024ull;
inline constexpr uint64_t kProjectionPackMaxChunkBytes = 128ull * 1024ull * 1024ull;
inline constexpr uint64_t kProjectionPackDefaultChunkBytes = 16ull * 1024ull * 1024ull;

// Merged tensor name convention: blk.<layer>.<kind>.merged
[[nodiscard]] luisa::string projection_pack_matrix_name(QwenMergeKind kind, uint64_t layer);
// Payload paths of matrix `index` (scales_path is empty for bf16).
[[nodiscard]] Status projection_pack_payload_paths(const luisa::filesystem::path &dir,
                                                   uint64_t index,
                                                   luisa::filesystem::path &words_path,
                                                   luisa::filesystem::path &scales_path) noexcept;
// Elements covered by one uint32 payload word for a pack format: 2 (bf16) or 4
// (fp8_e4m3 / q8_0). W4 is refused: the merged projection path has no 4-bit
// realization (see qwen/dsl_ops.h).
[[nodiscard]] Result<uint32_t> projection_pack_elements_per_word(WeightStorage storage) noexcept;
// Scale groups per output row of one matrix: 0 (bf16), ceil(K/group) (fp8_e4m3),
// K/32 (q8_0). Fail closed when the format's geometry contract is violated.
[[nodiscard]] Result<uint64_t> projection_pack_scales_per_row(WeightStorage storage,
                                                              uint32_t group, uint64_t k) noexcept;

struct ProjectionPackOptions {
    // bf16_packed, fp8_packed or q8_0_packed. The pack records exactly one format:
    // a runtime that asks for a different one is refused instead of being served a
    // silently re-quantized matrix.
    WeightStorage format{WeightStorage::bf16_packed};
    // Scale group along K for fp8_e4m3 (must be the runtime's 64) / q8_0 (32 is
    // implied by the block geometry). Ignored for bf16.
    uint32_t group{kFp8GroupSize};
    // Host read/write chunk in bytes (clamped to the bounds above).
    uint64_t chunk_bytes{kProjectionPackDefaultChunkBytes};
};

// One segment of one merged matrix in a manifest (role/name/rows/row_offset plus
// the source payload identity).
struct ProjectionPackSegmentRecord {
    QwenTensorSegment segment;// role, source name, rows, row_offset, bytes, hash
};

struct ProjectionPackMatrixRecord {
    luisa::string name;// merged tensor name
    QwenMergeKind kind{QwenMergeKind::none};
    uint64_t layer{UINT64_MAX};
    uint64_t k{0};
    uint64_t n{0};
    WeightStorage storage{WeightStorage::bf16_packed};
    uint32_t group{0};
    uint64_t scales_per_row{0};
    uint64_t word_bytes{0};
    uint64_t word_hash{0};
    uint64_t scale_bytes{0};
    uint64_t scale_hash{0};
    luisa::vector<QwenTensorSegment> segments;
};

struct ProjectionPackManifest {
    uint32_t version{0};
    luisa::string source_name;
    uint64_t source_size{0};
    // Rolling digest over the merged source tensors' raw payloads, in inventory
    // order (w4_roll_source_hash of every per-tensor FNV-1a 64 hash).
    uint64_t source_hash{0};
    WeightStorage storage{WeightStorage::bf16_packed};
    uint32_t group{0};
    uint32_t elements_per_word{0};
    uint64_t scales_per_row{0};
    // "bf16" / "fp8_e4m3" / "q8_0" -- the single format entry the files follow.
    luisa::string format_order;
    luisa::vector<ProjectionPackMatrixRecord> matrices;
};

// A pack the uploader is told to consume: the directory plus the manifest that was
// already read (and verified against the checkpoint) by the caller. Borrowed for
// the duration of the upload; nothing here is cached inside the device weights.
struct QwenProjectionPackSource {
    luisa::filesystem::path dir;
    const ProjectionPackManifest *manifest{nullptr};
};

// Write/read the manifest (the packer writes it LAST, so a pack without one is an
// incomplete pack and is refused).
[[nodiscard]] Status write_projection_pack_manifest(const luisa::filesystem::path &dir,
                                                    const ProjectionPackManifest &manifest) noexcept;
[[nodiscard]] Result<ProjectionPackManifest> read_projection_pack_manifest(
    const luisa::filesystem::path &dir) noexcept;

// Rolling digest of the merged source tensors of an inventory (the identity the
// manifest records). `model.tensors()` is the production inventory; the packer
// and every verifier call this one rule.
[[nodiscard]] Result<uint64_t> projection_pack_source_digest(
    GgufReader &reader, const Qwen35Model &model,
    uint64_t chunk_bytes = kProjectionPackDefaultChunkBytes) noexcept;
[[nodiscard]] Result<uint64_t> projection_pack_source_digest(
    GgufReader &reader, luisa::span<const QwenMergedMatrixSpec> inventory,
    uint64_t chunk_bytes = kProjectionPackDefaultChunkBytes) noexcept;

// Verify that the checkpoint the runner was told to load is the one the pack was
// built from: file SIZE and the recomputed rolling source digest must both match.
// Fails closed with both values, before anything is uploaded.
[[nodiscard]] Status verify_projection_pack_source(
    GgufReader &reader, const Qwen35Model &model, const ProjectionPackManifest &manifest,
    uint64_t chunk_bytes = kProjectionPackDefaultChunkBytes) noexcept;

// Verify every payload file's byte size and FNV-1a 64 hash (the upload does the
// same check while it streams, so this is for tooling/tests and for a dry run).
[[nodiscard]] Status verify_projection_pack_payloads(
    const luisa::filesystem::path &dir, const ProjectionPackManifest &manifest,
    uint64_t chunk_bytes = kProjectionPackDefaultChunkBytes) noexcept;

// Structural validation of one manifest against the canonical inventory: every
// matrix present exactly once, in inventory order, same kind/layer/k/n, same
// segment roles/names/rows/offsets, same format/group, sizes consistent with the
// geometry and every payload size equal to what the geometry implies. Never
// touches the filesystem or a device.
[[nodiscard]] Status validate_projection_pack_manifest(
    const ProjectionPackManifest &manifest,
    luisa::span<const QwenMergedMatrixSpec> inventory) noexcept;

struct ProjectionPackReport {
    uint64_t matrices{0};
    uint64_t segments{0};
    uint64_t word_bytes{0};
    uint64_t scale_bytes{0};
    uint64_t source_bytes_read{0};
    uint64_t source_hash{0};
    double seconds{0.0};
};

// Offline packer: walk the canonical merged-projection inventory of a validated
// BF16/F32 (bf16 or fp8 format) or Q8_0 (q8_0 format) checkpoint, write one
// payload pair per merged matrix and finish with the manifest. Never touches a
// device; streams in bounded chunks; the whole checkpoint is never held in RAM.
[[nodiscard]] Status pack_projection_pack_from_gguf(const GgufReader &reader,
                                                    const Qwen35Model &model,
                                                    luisa::string_view source_name,
                                                    const luisa::filesystem::path &dir,
                                                    const ProjectionPackOptions &options,
                                                    ProjectionPackReport *report = nullptr) noexcept;

}// namespace luisa::qwen
