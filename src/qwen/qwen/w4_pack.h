#pragma once
// Qwen3.5-4B -- OFFLINE W4A16 pack (opt_plan/05 Q2 + section 5).
//
// Why a pack exists: the deterministic per-group clipping search of
// common.h::pack_w4_matrix costs minutes over a 4 B-parameter model, so it must
// never run inside an inference request. This module produces and consumes a
// self-describing pack:
//
//   <dir>/manifest.txt             one line-oriented, order-stable text file
//   <dir>/tensors/<index>.f32.bin  dense F32 payload   (norms/ssm_a/dt/conv)
//   <dir>/tensors/<index>.w4.bin   8 signed int4 codes per uint32 word
//   <dir>/tensors/<index>.w4s.bin  one FP32 scale per group of `group` K
//   <dir>/tensors/<index>.fp8.bin  the SAME matrix as FP8 E4M3 quads
//   <dir>/tensors/<index>.fp8s.bin its FP32 scales (one per 64 K)
//
// The manifest records, per tensor: name, role, layer, ne-ordered shape, the
// scale group, every payload's byte size and 64-bit FNV-1a hash, and the
// SOURCE tensor's byte size + hash (the GGUF payload the pack was derived
// from), plus the clip-search configuration and the segment/format order of the
// file itself. The loader verifies the version, the geometry, the sizes AND
// every hash and fails closed on any mismatch: a truncated, stale or foreign
// pack is never partially loaded.
//
// Both payloads exist in the DEFAULT pack (plan option (c)): W4 for decode and
// FP8 for prefill + the LM head. `W4PackOptions::emit_fp8_payloads = false`
// produces the single-copy pack used by the explicit option-(b) A/B arm.
//
// Nothing here quantizes inside a request, and the pack is NEVER a substitute
// for a checkpoint the CLI was asked to load: `example_qwen --version dsl-w4`
// requires `--w4-pack <dir>` and refuses any other checkpoint format.
//
// The hash is FNV-1a 64 (offset basis 1469598103934665603, prime 1099511628211)
// over the raw payload bytes in file order. It is a corruption/staleness
// detector, NOT a cryptographic digest, and the manifest says so.

#include <cstddef>
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "qwen/common.h"
#include "qwen/gguf.h"
#include "qwen/model.h"
#include "qwen/w4_importance.h"
#include "qwen/weights.h"

namespace luisa::qwen {

inline constexpr uint32_t kW4PackVersion = 1u;
inline constexpr uint64_t kFnv1aOffsetBasis = 1469598103934665603ull;
inline constexpr uint64_t kFnv1aPrime = 1099511628211ull;

[[nodiscard]] uint64_t qwen_fnv1a64(luisa::span<const std::byte> bytes, uint64_t seed = kFnv1aOffsetBasis) noexcept;

[[nodiscard]] luisa::string qwen_pack_hash_string(uint64_t hash);

struct W4PackOptions {
    W4ClipSearchConfig clip{};
    // Emit the FP8 payloads (plan option (c), the default). False produces the
    // single-copy pack of the explicit option-(b) A/B arm.
    bool emit_fp8_payloads{true};
    // IMPORTANCE-AWARE clipping (qwen/w4_importance.h). When non-null every W4
    // matrix of the inventory MUST have a statistics entry with the same NAME and
    // the same K, otherwise the pack fails closed and names the tensor. The
    // statistics must also describe the SAME checkpoint (size + rolling payload
    // digest), so a stats file from another run is stale and refused. nullptr is
    // the measured clipping-only arm and is recorded as such in the manifest.
    const W4Importance *importance{nullptr};
    // Host read/write chunk in bytes (bounded streaming; clamped to
    // [64 KiB, 128 MiB] and rounded down to a multiple of 16).
    uint64_t chunk_bytes{16ull * 1024ull * 1024ull};
    // True when the inventory handed to pack_w4_from_gguf IS the checkpoint's
    // complete inventory: only then is the pack's rolling source digest the
    // CHECKPOINT digest, so only then is the importance staleness digest check
    // exact. The model overload sets this automatically; a filtered/subset pack
    // (tooling, tests, smokes) leaves it false and states in the report that its
    // digest identifies the subset.
    bool full_inventory{false};
};

// One tensor record of a pack. Sizes are exactly the payload byte counts.
struct W4PackTensorRecord {
    luisa::string name;
    QwenTensorRole role{QwenTensorRole::token_embd};
    uint64_t layer{UINT64_MAX};
    uint64_t k{0};
    uint64_t n{0};
    // Storage of the DECODE tensor: f32_dense (no w4 payload) or w4_packed.
    WeightStorage storage{WeightStorage::w4_packed};
    uint32_t group{0};
    uint64_t source_bytes{0};
    uint64_t source_hash{0};
    uint64_t w4_bytes{0};
    uint64_t w4_hash{0};
    uint64_t w4s_bytes{0};
    uint64_t w4s_hash{0};
    uint64_t f32_bytes{0};
    uint64_t f32_hash{0};
    uint64_t fp8_bytes{0};
    uint64_t fp8_hash{0};
    uint64_t fp8s_bytes{0};
    uint64_t fp8s_hash{0};
};

struct W4PackManifest {
    uint32_t version{0};
    luisa::string source_name;
    uint64_t source_size{0};
    uint64_t source_hash{0};
    W4ClipSearchConfig clip{};
    bool has_fp8_payloads{false};
    // "w4,fp8" or "w4" -- the segment/format order the files follow.
    luisa::string format_order;
    // IMPORTANCE-AWARE packs only (qwen/w4_importance.h): the calibration that
    // weighted the clipping objective. `has_importance == false` is the
    // clipping-only arm from the section 3 measurement (see the docs), and the
    // loader rejects the pack if this flag disagrees with a missing/extra line.
    bool has_importance{false};
    uint64_t importance_hash{0};    // FNV-1a 64 of the calibration payload
    uint64_t importance_source_hash{0};// checkpoint digest the stats came from
    uint64_t importance_tokens{0};  // activation tokens behind the statistics
    luisa::string importance_name;  // calibration source name (provenance only)
    luisa::vector<W4PackTensorRecord> tensors;
};

// Write the manifest of an existing pack directory (the packer writes it last,
// so a pack without a manifest is an incomplete pack).
[[nodiscard]] Status write_w4_pack_manifest(const luisa::filesystem::path &dir,
                                            const W4PackManifest &manifest) noexcept;
[[nodiscard]] Result<W4PackManifest> read_w4_pack_manifest(const luisa::filesystem::path &dir) noexcept;

// Offline packer: walk every tensor of a validated BF16/F32 checkpoint, run the
// deterministic clipping search for every matrix, write the W4 payload and (when
// requested) the FP8 payload, and finish with the manifest. `report` receives
// the aggregate statistics (the same ones the CLI prints). Never touches a
// device.
struct W4PackReport {
    uint64_t tensors{0};
    uint64_t w4_matrices{0};
    uint64_t dense_tensors{0};
    uint64_t w4_bytes{0};
    uint64_t w4s_bytes{0};
    uint64_t fp8_bytes{0};
    uint64_t fp8s_bytes{0};
    uint64_t f32_bytes{0};
    // ---- aggregate statistics, normalized ONCE at the end of the pack ----
    // Everything below is a MODEL-WIDE quantity, not a per-tensor leftover:
    //   mse / rtn_mse      = sum over every packed element of the squared
    //                        dequantization error / w4_elements (an element MSE)
    //   weighted_mse / ... = sum over every group of the (importance-weighted)
    //                        group objective / groups
    //   mean_ratio         = group-weighted mean chosen clipping ratio
    // A packer run resets the report first, so a caller-supplied report never
    // inherits stale numbers from an earlier run.
    uint64_t groups{0};       // W4 scale groups packed
    uint64_t w4_elements{0};  // W4 elements packed (the MSE denominator)
    uint64_t clipped_groups{0};
    double mse{0.0};
    double rtn_mse{0.0};
    double weighted_mse{0.0};
    double rtn_weighted_mse{0.0};
    float mean_ratio{1.0f};
    // Importance-aware coverage: matrices packed with measured statistics vs
    // matrices packed with the clipping-only search (must be 0 for a strict
    // importance-aware pack; the packer refuses when statistics are required).
    uint64_t importance_matrices{0};
    uint64_t unweighted_matrices{0};
    double seconds{0.0};
    // 100 * (1 - mse/rtn_mse): the measured clipping gain over plain RTN, in
    // percent. 0 when the RTN reference is degenerate (all-zero pack).
    [[nodiscard]] double mse_improvement() const noexcept {
        return rtn_mse > 0.0 ? 100.0 * (1.0 - mse / rtn_mse) : 0.0;
    }
    // 100 * (1 - weighted_mse/rtn_weighted_mse) for an importance-aware pack.
    [[nodiscard]] double weighted_improvement() const noexcept {
        return rtn_weighted_mse > 0.0 ? 100.0 * (1.0 - weighted_mse / rtn_weighted_mse) : 0.0;
    }
};
// ---------------------------------------------------------------------------
// Source identity (opt_plan/05 Q2 + phase 8b review A1).
//
// The manifest's `source hash=` is a ROLLING digest of the checkpoint payloads
// the pack was derived from: for EVERY tensor of the inventory, in inventory
// order, the raw stored bytes of that tensor (the exact GGUF payload, never a
// decoded/normalized copy) are hashed with FNV-1a 64, and each per-tensor hash is
// folded into the checkpoint digest as its eight little-endian bytes. The packer
// computes the per-tensor halves while it streams the payloads, so these helpers
// reproduce the same digest without touching the pack: the runner compares the
// digest of the checkpoint it was told to load against the pack, which rejects a
// SAME-SIZE / different-payload checkpoint (a size comparison cannot).
// ---------------------------------------------------------------------------
inline constexpr uint64_t kW4SourceHashChunkBytes = 16ull * 1024ull * 1024ull;

// Fold one per-tensor payload hash into the checkpoint digest. This is the ONLY
// place the rolling rule lives; the packer and every verifier call it.
[[nodiscard]] uint64_t w4_roll_source_hash(uint64_t state, uint64_t tensor_hash) noexcept;

// FNV-1a 64 of one GGUF tensor's raw payload, streamed in `chunk_bytes` chunks
// (clamped to [64 KiB, 128 MiB] like the packer). Fails closed on a short read.
[[nodiscard]] Status w4_hash_source_tensor(GgufReader &reader, const GgufTensorInfo &info,
                                           uint64_t chunk_bytes, uint64_t &hash_out) noexcept;

// Recompute the checkpoint digest of an inventory (production: model.tensors(),
// which is what the packer walks). `w4_source_digest` is the primitive the tests
// and tooling use on an explicit inventory.
[[nodiscard]] Result<uint64_t> w4_source_digest(GgufReader &reader,
                                                luisa::span<const QwenTensorRef> inventory,
                                                uint64_t chunk_bytes = kW4SourceHashChunkBytes) noexcept;

// Convenience overload over the model's own inventory.
[[nodiscard]] Result<uint64_t> compute_w4_source_hash(const GgufReader &reader,
                                                      const Qwen35Model &model,
                                                      uint64_t chunk_bytes = kW4SourceHashChunkBytes) noexcept;

// Verify a pack's recorded source identity against the checkpoint the runner was
// told to load: the file SIZE and the recomputed rolling payload DIGEST must both
// match. Fails closed (unvalidated) with the pack's and the checkpoint's values,
// so a same-size checkpoint with different weights is refused before any upload.
[[nodiscard]] Status verify_w4_pack_source(GgufReader &reader,
                                           luisa::span<const QwenTensorRef> inventory,
                                           const W4PackManifest &manifest,
                                           uint64_t chunk_bytes = kW4SourceHashChunkBytes) noexcept;
[[nodiscard]] Status verify_w4_pack_source(const GgufReader &reader, const Qwen35Model &model,
                                           const W4PackManifest &manifest,
                                           uint64_t chunk_bytes = kW4SourceHashChunkBytes) noexcept;

// Offline packer. `source_name` is recorded in the manifest (the CLI passes the
// checkpoint's file name); `source_size` is taken from the reader itself and
// `source_hash` is the rolling digest of the inventory computed here.
//
// The INVENTORY overload packs exactly the tensors it is given, in that order:
// `model.tensors()` is the production inventory (all 426 tensors of a frozen
// Qwen3.5-4B checkpoint) and is what the model overload passes; tests and tooling
// use an explicit subset. A subset pack is a valid, self-describing pack (a
// consumer that needs the missing tensors fails closed at its own lookup), but
// it is NOT a runnable dsl-w4 checkpoint. The manifest is written LAST, so a pack
// directory without a manifest is an incomplete pack.
[[nodiscard]] Status pack_w4_from_gguf(const GgufReader &reader,
                                       luisa::span<const QwenTensorRef> inventory,
                                       luisa::string_view source_name,
                                       const luisa::filesystem::path &dir,
                                       const W4PackOptions &options,
                                       W4PackReport *report = nullptr) noexcept;
[[nodiscard]] Status pack_w4_from_gguf(const GgufReader &reader, const Qwen35Model &model,
                                       luisa::string_view source_name,
                                       const luisa::filesystem::path &dir,
                                       const W4PackOptions &options,
                                       W4PackReport *report = nullptr) noexcept;

// Load a pack into device buffers. Every checksum, size and geometry is
// verified; the result holds one tensor per packed matrix (w4_packed, name =
// manifest name) plus, when the pack carries them, its FP8 partner named
// "<name>.fp8", and one dense F32 tensor per sensitive tensor. Fails closed on
// any mismatch (nothing is returned, every buffer created is released).
[[nodiscard]] Result<QwenDeviceWeights> upload_w4_pack(compute::Device &device, compute::Stream &stream,
                                                       const luisa::filesystem::path &dir,
                                                       uint64_t chunk_bytes = 16ull * 1024ull * 1024ull,
                                                       QwenWeightUploadReport *report = nullptr);

}// namespace luisa::qwen
