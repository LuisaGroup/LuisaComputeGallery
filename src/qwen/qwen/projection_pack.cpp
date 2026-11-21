// Qwen3.5-4B -- OFFLINE merged input-projection pack (see qwen/projection_pack.h).
// Offline writer: streams the merged SOURCE tensors out of the checkpoint and
// writes one row-concatenated payload pair per merged matrix, then the manifest.
// Runtime consumer: verifies every payload's size/hash while streaming it into the
// device buffers (qwen/weights.h append_merged_projections) and hands the engine a
// first-class segment descriptor instead of a bag of views (opt_plan/03 D5).
// Recoverable Status everywhere; no exceptions, no RTTI, no device in the packer.
#include "qwen/projection_pack.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>

#include "qwen/weights.h"
#include "qwen/w4_pack.h"

namespace luisa::qwen {
namespace {

[[nodiscard]] Status io_error(luisa::string msg, luisa::string detail = {}) {
    return Status::error(StatusCode::io_error, std::move(msg), std::move(detail));
}

[[nodiscard]] Status invalid(luisa::string msg, luisa::string detail = {}) {
    return Status::error(StatusCode::invalid_argument, std::move(msg), std::move(detail));
}

[[nodiscard]] uint64_t clamp_chunk(uint64_t requested) noexcept {
    const uint64_t c = std::clamp<uint64_t>(requested, kProjectionPackMinChunkBytes,
                                            kProjectionPackMaxChunkBytes);
    return c & ~15ull;
}

// Hash (and optionally write) one payload buffer in bounded chunks. The hash is
// FNV-1a 64 over the raw payload bytes in FILE order, the same rule the W4 pack
// uses (qwen/w4_pack.h).
struct HashWriter {
    std::ofstream *out{nullptr};
    uint64_t hash{kFnv1aOffsetBasis};
    uint64_t bytes{0};
    [[nodiscard]] bool write(luisa::span<const std::byte> data) noexcept {
        hash = qwen_fnv1a64(data, hash);
        bytes += static_cast<uint64_t>(data.size());
        if (out != nullptr && out->is_open()) {
            out->write(reinterpret_cast<const char *>(data.data()),
                       static_cast<std::streamsize>(data.size()));
            return static_cast<bool>(*out);
        }
        return true;
    }
};

[[nodiscard]] Status open_out(const luisa::filesystem::path &path, std::ofstream &out) {
    out.open(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return io_error("cannot open projection-pack payload for writing", luisa::to_string(path));
    }
    return Status::success();
}

[[nodiscard]] Status open_in(const luisa::filesystem::path &path, std::ifstream &in) {
    in.open(path, std::ios::binary);
    if (!in.is_open()) {
        return io_error("cannot open projection-pack payload for reading", luisa::to_string(path));
    }
    return Status::success();
}

[[nodiscard]] Status read_exact(std::ifstream &in, luisa::span<std::byte> dst,
                                const luisa::filesystem::path &path) {
    if (!dst.empty()) {
        in.read(reinterpret_cast<char *>(dst.data()), static_cast<std::streamsize>(dst.size()));
        if (in.gcount() != static_cast<std::streamsize>(dst.size())) {
            return io_error("projection-pack payload is truncated", luisa::to_string(path));
        }
    }
    return Status::success();
}

[[nodiscard]] Status write_manifest_text(const luisa::filesystem::path &path,
                                         luisa::string_view text) {
    std::ofstream out;
    out.open(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return io_error("cannot open the projection-pack manifest for writing", luisa::to_string(path));
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!out) { return io_error("projection-pack manifest write failed", luisa::to_string(path)); }
    return Status::success();
}

[[nodiscard]] bool parse_u64(luisa::string_view tok, uint64_t &out) noexcept {
    if (tok.empty()) { return false; }
    uint64_t value = 0;
    for (char c : tok) {
        if (c < '0' || c > '9') { return false; }
        const uint64_t d = static_cast<uint64_t>(c - '0');
        if (value > (UINT64_MAX - d) / 10ull) { return false; }
        value = value * 10ull + d;
    }
    out = value;
    return true;
}

// 16-hex-digit hash (the manifest's canonical form).
[[nodiscard]] bool parse_hex64(luisa::string_view hex, uint64_t &out) noexcept {
    if (hex.size() != 16u) { return false; }
    uint64_t h = 0;
    for (char c : hex) {
        uint32_t d;
        if (c >= '0' && c <= '9') {
            d = static_cast<uint32_t>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            d = static_cast<uint32_t>(c - 'a') + 10u;
        } else {
            return false;
        }
        h = (h << 4u) | d;
    }
    out = h;
    return true;
}

struct KeyValues {
    luisa::string_view key;
    luisa::string_view value;
};

// Tokenize one manifest line of `key=value` fields (the first token may be a bare
// tag with no '=').
[[nodiscard]] luisa::vector<KeyValues> split_fields(luisa::string_view line) {
    luisa::vector<KeyValues> fields;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) { i++; }
        const size_t begin = i;
        while (i < line.size() && line[i] != ' ' && line[i] != '\t') { i++; }
        if (i == begin) { break; }
        const auto tok = line.substr(begin, i - begin);
        const size_t eq = tok.find('=');
        if (eq == luisa::string_view::npos) {
            fields.push_back({luisa::string_view{}, tok});
        } else {
            fields.push_back({tok.substr(0, eq), tok.substr(eq + 1u)});
        }
    }
    return fields;
}

// "bytes:hash" field.
[[nodiscard]] bool parse_pair(luisa::string_view v, uint64_t &bytes, uint64_t &hash) noexcept {
    const size_t colon = v.find(':');
    if (colon == luisa::string_view::npos) { return false; }
    return parse_u64(v.substr(0, colon), bytes) && parse_hex64(v.substr(colon + 1u), hash);
}

// Role name -> role. QwenTensorRole is a small dense enum, so a linear scan of the
// canonical names is exact and needs no extra table.
[[nodiscard]] bool parse_role(luisa::string_view name, QwenTensorRole &out) noexcept {
    for (uint32_t i = 0; i <= static_cast<uint32_t>(QwenTensorRole::gdn_norm); ++i) {
        const auto role = static_cast<QwenTensorRole>(i);
        if (qwen_tensor_role_name(role) == name) {
            out = role;
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool parse_storage(luisa::string_view name, WeightStorage &out) noexcept {
    for (auto s : {WeightStorage::bf16_packed, WeightStorage::fp8_packed,
                   WeightStorage::q8_0_packed, WeightStorage::w4_packed,
                   WeightStorage::f32_dense}) {
        if (weight_storage_name(s) == name) {
            out = s;
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool parse_kind(luisa::string_view name, QwenMergeKind &out) noexcept {
    for (auto k : {QwenMergeKind::gdn_qkv_z_ab, QwenMergeKind::attn_qkv}) {
        if (qwen_merge_kind_name(k) == name) {
            out = k;
            return true;
        }
    }
    return false;
}

[[nodiscard]] uint64_t source_row_bytes(WeightStorage storage, uint32_t gguf_type,
                                        uint64_t k) noexcept {
    if (storage == WeightStorage::q8_0_packed) {
        return (k / kQ8_0BlockSize) * kQ8_0BlockBytes;
    }
    return gguf_type == static_cast<uint32_t>(GgufTensorType::f32) ? k * 4u : k * 2u;
}

}// namespace

// ---------------------------------------------------------------------------
// Names and geometry
// ---------------------------------------------------------------------------
[[nodiscard]] luisa::string projection_pack_matrix_name(QwenMergeKind kind, uint64_t layer) {
    return luisa::string{"blk."} + luisa::string{std::to_string(layer).c_str()} + "." +
           luisa::string{qwen_merge_kind_name(kind)} + ".merged";
}

[[nodiscard]] Status projection_pack_payload_paths(const luisa::filesystem::path &dir,
                                                   uint64_t index,
                                                   luisa::filesystem::path &words_path,
                                                   luisa::filesystem::path &scales_path) noexcept {
    const auto stem = luisa::string{std::to_string(index).c_str()};
    words_path = dir / "matrices" / luisa::filesystem::path{stem + ".bin"};
    scales_path = dir / "matrices" / luisa::filesystem::path{stem + ".scales.bin"};
    return Status::success();
}

[[nodiscard]] Result<uint32_t> projection_pack_elements_per_word(WeightStorage storage) noexcept {
    switch (storage) {
        case WeightStorage::bf16_packed: return Result<uint32_t>::success(2u);
        case WeightStorage::fp8_packed:
        case WeightStorage::q8_0_packed: return Result<uint32_t>::success(4u);
        case WeightStorage::w4_packed:
            return Result<uint32_t>::failure(
                Status::error(StatusCode::unsupported,
                    "W4 is not mergeable here: the merged projection pack has no 4-bit "
                    "realization (opt_plan/03 section 3 measured the BF16/FP8/Q8 decode "
                    "GEMV only); keep W4 on its own dual-copy pack"));
        case WeightStorage::f32_dense:
            return Result<uint32_t>::failure(
                invalid("a projection-pack matrix must be a packed format (bf16/fp8/q8_0)"));
    }
    return Result<uint32_t>::failure(invalid("unknown projection-pack storage"));
}

[[nodiscard]] Result<uint64_t> projection_pack_scales_per_row(WeightStorage storage,
                                                              uint32_t group, uint64_t k) noexcept {
    switch (storage) {
        case WeightStorage::bf16_packed:
            return Result<uint64_t>::success(0ull);
        case WeightStorage::fp8_packed:
            if (group != kFp8GroupSize) {
                return Result<uint64_t>::failure(invalid(
                    "the merged fp8_e4m3 projection pack requires the canonical scale group 64",
                    luisa::string{std::to_string(group).c_str()}));
            }
            if (k == 0u || k % 4u != 0u) {
                return Result<uint64_t>::failure(invalid(
                    "the merged fp8_e4m3 projection pack requires K % 4 == 0 (whole quad words "
                    "per row)"));
            }
            return Result<uint64_t>::success(fp8_group_count(k));
        case WeightStorage::q8_0_packed:
            if (group != 0u && group != kQ8_0BlockSize) {
                return Result<uint64_t>::failure(invalid(
                    "the merged q8_0 projection pack requires the GGUF block geometry (32)"));
            }
            if (k == 0u || k % kQ8_0BlockSize != 0u) {
                return Result<uint64_t>::failure(invalid(
                    "the merged q8_0 projection pack requires K % 32 == 0 (whole blocks per row)"));
            }
            return Result<uint64_t>::success(q8_0_group_count(k));
        default: break;
    }
    return Result<uint64_t>::failure(invalid("unsupported projection-pack storage"));
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------
[[nodiscard]] Status write_projection_pack_manifest(const luisa::filesystem::path &dir,
                                                    const ProjectionPackManifest &manifest) noexcept {
    if (manifest.matrices.empty()) {
        return invalid("refusing to write an empty projection-pack manifest");
    }
    luisa::string text;
    text.reserve(2048u * (manifest.matrices.size() + 4u));
    const auto num = [](uint64_t v) { return luisa::string{std::to_string(v).c_str()}; };
    const auto add = [&text](luisa::string_view s) { text.append(s); };
    add("qwen-projection-pack ");
    add(num(manifest.version));
    add("\n");
    add("source name=");
    add(manifest.source_name);
    add(" size=");
    add(num(manifest.source_size));
    add(" hash=");
    add(qwen_pack_hash_string(manifest.source_hash));
    add(" hash_algorithm=fnv1a64\n");
    add("format ");
    add(weight_storage_name(manifest.storage));
    add(" group=");
    add(num(manifest.group));
    add(" elements_per_word=");
    add(num(manifest.elements_per_word));
    add(" scales_per_row=");
    add(num(manifest.scales_per_row));
    add(" matrices=");
    add(num(manifest.matrices.size()));
    add(" order=");
    add(manifest.format_order);
    add("\n");
    for (size_t i = 0; i < manifest.matrices.size(); ++i) {
        const auto &m = manifest.matrices[i];
        add("matrix idx=");
        add(num(i));
        add(" name=");
        add(m.name);
        add(" kind=");
        add(qwen_merge_kind_name(m.kind));
        add(" layer=");
        add(num(m.layer));
        add(" k=");
        add(num(m.k));
        add(" n=");
        add(num(m.n));
        add(" storage=");
        add(weight_storage_name(m.storage));
        add(" group=");
        add(num(m.group));
        add(" scales_per_row=");
        add(num(m.scales_per_row));
        add(" words=");
        add(num(m.word_bytes));
        add(":");
        add(qwen_pack_hash_string(m.word_hash));
        add(" scales=");
        add(num(m.scale_bytes));
        add(":");
        add(qwen_pack_hash_string(m.scale_hash));
        add(" segments=");
        add(num(m.segments.size()));
        add("\n");
        uint64_t row = 0;
        for (size_t s = 0; s < m.segments.size(); ++s) {
            const auto &seg = m.segments[s];
            if (seg.row_offset != row) {
                return invalid("refusing to write a manifest with a non-contiguous segment",
                               m.name);
            }
            row += seg.rows;
            add("segment idx=");
            add(num(s));
            add(" name=");
            add(seg.name);
            add(" role=");
            add(qwen_tensor_role_name(seg.role));
            add(" rows=");
            add(num(seg.rows));
            add(" row_offset=");
            add(num(seg.row_offset));
            add(" src=");
            add(num(seg.source_bytes));
            add(":");
            add(qwen_pack_hash_string(seg.source_hash));
            add("\n");
        }
        if (row != m.n) {
            return invalid("refusing to write a manifest whose segments do not cover N", m.name);
        }
    }
    return write_manifest_text(dir / "manifest.txt", text);
}

[[nodiscard]] Result<ProjectionPackManifest> read_projection_pack_manifest(
    const luisa::filesystem::path &dir) noexcept {
    const auto path = dir / "manifest.txt";
    std::ifstream in;
    if (auto s = open_in(path, in); !s.ok()) {
        return Result<ProjectionPackManifest>::failure(std::move(s));
    }
    luisa::vector<luisa::string> lines;
    {
        luisa::string line;
        while (std::getline(in, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) { line.pop_back(); }
            lines.emplace_back(std::move(line));
        }
    }
    ProjectionPackManifest m;
    size_t li = 0;
    bool have_header = false, have_source = false, have_format = false;
    size_t declared_matrices = SIZE_MAX;
    const auto fail = [&path](luisa::string msg) {
        return Result<ProjectionPackManifest>::failure(invalid(std::move(msg), luisa::to_string(path)));
    };
    for (; li < lines.size(); ++li) {
        const auto fields = split_fields(luisa::string_view{lines[li]});
        if (fields.empty()) { continue; }
        const auto tag = fields[0].key.empty() ? fields[0].value : fields[0].key;
        if (!have_header) {
            uint64_t version = 0;
            if (tag != "qwen-projection-pack" || fields.size() != 2u ||
                !parse_u64(fields[1].value, version) || version != kProjectionPackVersion) {
                return fail("not a qwen projection-pack manifest of a supported version");
            }
            m.version = static_cast<uint32_t>(version);
            have_header = true;
            continue;
        }
        if (tag == "source") {
            if (have_source) { return fail("duplicate source line"); }
            for (const auto &f : fields) {
                if (f.key == "name") {
                    m.source_name = luisa::string{f.value};
                } else if (f.key == "size") {
                    if (!parse_u64(f.value, m.source_size)) { return fail("bad source size"); }
                } else if (f.key == "hash") {
                    if (!parse_hex64(f.value, m.source_hash)) { return fail("bad source hash"); }
                } else if (f.key == "hash_algorithm") {
                    if (f.value != "fnv1a64") { return fail("only fnv1a64 is accepted"); }
                }
            }
            have_source = true;
            continue;
        }
        if (tag == "format") {
            if (have_format) { return fail("duplicate format line"); }
            uint64_t group = 0, epw = 0, spr = 0, matrices = 0;
            for (const auto &f : fields) {
                if (f.key == "group") {
                    if (!parse_u64(f.value, group)) { return fail("bad group"); }
                } else if (f.key == "elements_per_word") {
                    if (!parse_u64(f.value, epw)) { return fail("bad elements_per_word"); }
                } else if (f.key == "scales_per_row") {
                    if (!parse_u64(f.value, spr)) { return fail("bad scales_per_row"); }
                } else if (f.key == "matrices") {
                    if (!parse_u64(f.value, matrices)) { return fail("bad matrix count"); }
                } else if (f.key == "order") {
                    m.format_order = luisa::string{f.value};
                }
            }
            // The format name is the one bare token of this line (fields[0] is the
            // "format" tag itself).
            if (fields.size() >= 2u && fields[1].key.empty()) {
                if (!parse_storage(fields[1].value, m.storage)) { return fail("unknown pack format"); }
            } else {
                return fail("format line is missing its format name");
            }
            m.group = static_cast<uint32_t>(group);
            m.elements_per_word = static_cast<uint32_t>(epw);
            m.scales_per_row = spr;
            declared_matrices = static_cast<size_t>(matrices);
            if (m.storage != WeightStorage::bf16_packed && m.storage != WeightStorage::fp8_packed &&
                m.storage != WeightStorage::q8_0_packed) {
                return fail("the pack format must be bf16_packed, fp8_packed or q8_0_packed");
            }
            have_format = true;
            continue;
        }
        if (tag == "matrix") {
            if (!have_format) { return fail("matrix line before the format line"); }
            ProjectionPackMatrixRecord rec;
            uint64_t idx = 0, stored_segments = 0;
            bool have_segments = false;
            for (const auto &f : fields) {
                if (f.key == "idx") {
                    if (!parse_u64(f.value, idx)) { return fail("bad matrix index"); }
                } else if (f.key == "name") {
                    rec.name = luisa::string{f.value};
                } else if (f.key == "kind") {
                    if (!parse_kind(f.value, rec.kind)) { return fail("unknown merge kind"); }
                } else if (f.key == "layer") {
                    if (!parse_u64(f.value, rec.layer)) { return fail("bad layer"); }
                } else if (f.key == "k") {
                    if (!parse_u64(f.value, rec.k)) { return fail("bad k"); }
                } else if (f.key == "n") {
                    if (!parse_u64(f.value, rec.n)) { return fail("bad n"); }
                } else if (f.key == "storage") {
                    if (!parse_storage(f.value, rec.storage)) { return fail("unknown matrix storage"); }
                } else if (f.key == "group") {
                    uint64_t v = 0;
                    if (!parse_u64(f.value, v)) { return fail("bad matrix group"); }
                    rec.group = static_cast<uint32_t>(v);
                } else if (f.key == "scales_per_row") {
                    if (!parse_u64(f.value, rec.scales_per_row)) { return fail("bad scales_per_row"); }
                } else if (f.key == "words") {
                    if (!parse_pair(f.value, rec.word_bytes, rec.word_hash)) { return fail("bad words pair"); }
                } else if (f.key == "scales") {
                    if (!parse_pair(f.value, rec.scale_bytes, rec.scale_hash)) { return fail("bad scales pair"); }
                } else if (f.key == "segments") {
                    if (!parse_u64(f.value, stored_segments)) { return fail("bad segment count"); }
                    have_segments = true;
                }
            }
            if (!have_segments) { return fail("matrix line without a segment count"); }
            if (idx != m.matrices.size()) { return fail("matrix records are out of order"); }
            if (rec.storage != m.storage) { return fail("matrix storage differs from the pack format"); }
            if (rec.group != m.group || rec.scales_per_row != m.scales_per_row) {
                return fail("matrix scale geometry differs from the pack format line");
            }
            if (rec.name.empty() || rec.k == 0u || rec.n == 0u) { return fail("incomplete matrix record"); }
            for (uint64_t s = 0; s < stored_segments; ++s) {
                if (++li >= lines.size()) { return fail("missing segment line"); }
                const auto sf = split_fields(luisa::string_view{lines[li]});
                if (sf.empty() || !sf[0].key.empty() || sf[0].value != "segment") {
                    return fail("expected a segment line");
                }
                QwenTensorSegment seg;
                uint64_t idx_s = 0;
                for (const auto &f : sf) {
                    if (f.key == "idx") {
                        if (!parse_u64(f.value, idx_s)) { return fail("bad segment index"); }
                    } else if (f.key == "name") {
                        seg.name = luisa::string{f.value};
                    } else if (f.key == "role") {
                        if (!parse_role(f.value, seg.role)) { return fail("unknown segment role"); }
                    } else if (f.key == "rows") {
                        if (!parse_u64(f.value, seg.rows)) { return fail("bad segment rows"); }
                    } else if (f.key == "row_offset") {
                        if (!parse_u64(f.value, seg.row_offset)) { return fail("bad segment row offset"); }
                    } else if (f.key == "src") {
                        if (!parse_pair(f.value, seg.source_bytes, seg.source_hash)) {
                            return fail("bad segment source pair");
                        }
                    }
                }
                if (idx_s != s) {
                    return fail("segment records are out of order");
                }
                if (seg.name.empty() || seg.rows == 0u) { return fail("incomplete segment record"); }
                rec.segments.emplace_back(std::move(seg));
            }
            m.matrices.emplace_back(std::move(rec));
            continue;
        }
        return fail("unexpected manifest line: " + luisa::string{tag});
    }
    if (!have_header || !have_source || !have_format) {
        return fail("manifest is missing its header/source/format line");
    }
    if (declared_matrices != m.matrices.size()) {
        return fail("matrix count does not match the format line");
    }
    return Result<ProjectionPackManifest>::success(std::move(m));
}

// ---------------------------------------------------------------------------
// Structural validation against the canonical inventory (never touches a device
// or the filesystem: this is the fail-closed gate every consumer runs first).
// ---------------------------------------------------------------------------
[[nodiscard]] Status validate_projection_pack_manifest(
    const ProjectionPackManifest &manifest,
    luisa::span<const QwenMergedMatrixSpec> inventory) noexcept {
    if (manifest.version != kProjectionPackVersion) {
        return invalid("unsupported projection-pack version");
    }
    auto epw = projection_pack_elements_per_word(manifest.storage);
    if (!epw.ok()) { return epw.status; }
    if (manifest.elements_per_word != *epw.value) {
        return invalid("pack elements_per_word does not match its format");
    }
    if (manifest.matrices.size() != inventory.size()) {
        return invalid("pack matrix count does not match the model inventory",
                       luisa::string{std::to_string(manifest.matrices.size())} + " vs " +
                           luisa::string{std::to_string(inventory.size())});
    }
    for (size_t i = 0; i < inventory.size(); ++i) {
        const auto &want = inventory[i];
        const auto &got = manifest.matrices[i];
        const auto name = projection_pack_matrix_name(want.kind, want.layer);
        if (got.name != name) {
            return invalid("merged matrix name/order mismatch", got.name);
        }
        if (got.kind != want.kind || got.layer != want.layer || got.k != want.k || got.n != want.n) {
            return invalid("merged matrix geometry does not match the model inventory", got.name);
        }
        if (got.storage != manifest.storage) {
            return invalid("merged matrix format differs from the pack format", got.name);
        }
        auto spr = projection_pack_scales_per_row(got.storage, got.group, got.k);
        if (!spr.ok()) { return spr.status; }
        if (got.scales_per_row != *spr.value || got.scales_per_row != manifest.scales_per_row) {
            return invalid("merged matrix scale geometry mismatch", got.name);
        }
        uint64_t words_per_row = 0, word_bytes = 0, scale_bytes = 0;
        if (!checked_mul(got.k, got.n, words_per_row)) {
            return invalid("merged matrix element count overflow", got.name);
        }
        if (!checked_mul(words_per_row / manifest.elements_per_word, 4u, word_bytes) ||
            !checked_mul(got.n * got.scales_per_row, 4u, scale_bytes)) {
            return invalid("merged matrix payload size overflow", got.name);
        }
        if (got.word_bytes != word_bytes) {
            return invalid("merged matrix word payload size mismatch", got.name);
        }
        if (got.scale_bytes != scale_bytes) {
            return invalid("merged matrix scale payload size mismatch", got.name);
        }
        if (got.segments.size() != want.segments.size()) {
            return invalid("merged matrix segment count mismatch", got.name);
        }
        uint64_t row = 0;
        for (size_t s = 0; s < want.segments.size(); ++s) {
            const auto &ws = want.segments[s];
            const auto &gs = got.segments[s];
            if (gs.role != ws.role || gs.name != ws.name || gs.rows != ws.rows ||
                gs.row_offset != ws.row_offset) {
                return invalid("merged matrix segment does not match the canonical inventory",
                               got.name + "." + ws.name);
            }
            if (gs.source_bytes == 0u || gs.source_hash == 0u || gs.row_offset != row) {
                return invalid("merged matrix segment source identity/order is missing",
                               got.name + "." + ws.name);
            }
            row += gs.rows;
        }
        if (row != got.n) {
            return invalid("merged matrix segments do not cover N", got.name);
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Source identity
// ---------------------------------------------------------------------------
namespace {

[[nodiscard]] Result<uint64_t> source_digest_impl(GgufReader &reader,
                                                  luisa::span<const QwenMergedMatrixSpec> inventory,
                                                  uint64_t chunk_bytes) {
    const uint64_t chunk = clamp_chunk(chunk_bytes);
    uint64_t state = kFnv1aOffsetBasis;
    for (const auto &spec : inventory) {
        for (const auto &seg : spec.segments) {
            const auto *info = reader.find(seg.name);
            if (info == nullptr) {
                return Result<uint64_t>::failure(Status::error(
                    StatusCode::not_found,
                    "merged projection source tensor is missing from the checkpoint", seg.name));
            }
            uint64_t hash = 0;
            if (auto s = w4_hash_source_tensor(reader, *info, chunk, hash); !s.ok()) {
                return Result<uint64_t>::failure(std::move(s));
            }
            state = w4_roll_source_hash(state, hash);
        }
    }
    return Result<uint64_t>::success(state);
}

}// namespace

[[nodiscard]] Result<uint64_t> projection_pack_source_digest(
    GgufReader &reader, luisa::span<const QwenMergedMatrixSpec> inventory,
    uint64_t chunk_bytes) noexcept {
    return source_digest_impl(reader, inventory, chunk_bytes);
}

[[nodiscard]] Result<uint64_t> projection_pack_source_digest(
    GgufReader &reader, const Qwen35Model &model, uint64_t chunk_bytes) noexcept {
    auto inventory = qwen35_merged_projection_inventory(model);
    if (!inventory.ok()) { return Result<uint64_t>::failure(std::move(inventory.status)); }
    return source_digest_impl(reader, *inventory.value, chunk_bytes);
}

[[nodiscard]] Status verify_projection_pack_source(
    GgufReader &reader, const Qwen35Model &model, const ProjectionPackManifest &manifest,
    uint64_t chunk_bytes) noexcept {
    const uint64_t file_size = reader.file_size();
    if (manifest.source_size != file_size) {
        return Status::error(
            StatusCode::unvalidated,
            "the projection pack was built from a checkpoint of a different byte size",
            luisa::string{std::to_string(manifest.source_size)} + " vs " +
                luisa::string{std::to_string(file_size)});
    }
    auto digest = projection_pack_source_digest(reader, model, chunk_bytes);
    if (!digest.ok()) { return digest.status; }
    if (*digest.value != manifest.source_hash) {
        return Status::error(
            StatusCode::unvalidated,
            "the projection pack was built from different weights (same checkpoint size, "
            "different merged source payloads)",
            qwen_pack_hash_string(manifest.source_hash) + " vs " +
                qwen_pack_hash_string(*digest.value));
    }
    return Status::success();
}

[[nodiscard]] Status verify_projection_pack_payloads(
    const luisa::filesystem::path &dir, const ProjectionPackManifest &manifest,
    uint64_t chunk_bytes) noexcept {
    const uint64_t chunk = std::min<uint64_t>(clamp_chunk(chunk_bytes), 16ull * 1024ull * 1024ull);
    luisa::vector<std::byte> staging(static_cast<size_t>(chunk));
    for (size_t i = 0; i < manifest.matrices.size(); ++i) {
        const auto &rec = manifest.matrices[i];
        luisa::filesystem::path words_path, scales_path;
        if (auto s = projection_pack_payload_paths(dir, i, words_path, scales_path); !s.ok()) { return s; }
        const auto verify_one = [&](const luisa::filesystem::path &path, uint64_t bytes,
                                    uint64_t expected) -> Status {
            if (bytes == 0u) {
                return Status::error(StatusCode::unvalidated,
                                     "projection-pack payload is missing from the manifest",
                                     luisa::to_string(path));
            }
            std::ifstream in;
            if (auto s = open_in(path, in); !s.ok()) { return s; }
            HashWriter hw{nullptr};
            uint64_t left = bytes;
            while (left > 0u) {
                const uint64_t n = std::min<uint64_t>(staging.size(), left);
                const auto span = luisa::span<std::byte>{staging.data(), static_cast<size_t>(n)};
                if (auto s = read_exact(in, span, path); !s.ok()) { return s; }
                hw.write(span);
                left -= n;
            }
            // A payload file larger than its record is also corrupt.
            char extra = 0;
            if (in.read(&extra, 1) && in.gcount() == 1) {
                return Status::error(StatusCode::unvalidated,
                                     "projection-pack payload is longer than its manifest record",
                                     luisa::to_string(path));
            }
            if (hw.hash != expected) {
                return Status::error(StatusCode::unvalidated,
                                     "projection-pack payload hash mismatch (stale or corrupt pack)",
                                     luisa::to_string(path));
            }
            return Status::success();
        };
        if (auto s = verify_one(words_path, rec.word_bytes, rec.word_hash); !s.ok()) { return s; }
        if (rec.scale_bytes != 0u) {
            if (auto s = verify_one(scales_path, rec.scale_bytes, rec.scale_hash); !s.ok()) { return s; }
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Offline packer
// ---------------------------------------------------------------------------
namespace {

// Fail-closed source-format gate of one segment.
[[nodiscard]] Status check_segment_source(WeightStorage storage, const GgufTensorInfo &info,
                                          uint64_t k, uint64_t rows) {
    if (info.dims.size() != 2u || info.dims[0] != k || info.dims[1] != rows) {
        return invalid("merged projection source tensor does not have the [K, rows] shape",
                       info.name);
    }
    switch (storage) {
        case WeightStorage::bf16_packed:
            if (info.type != static_cast<uint32_t>(GgufTensorType::bf16)) {
                return invalid("a bf16 projection pack requires BF16 source matrices "
                               "(no load-time conversion is recorded for this pack)",
                               info.name);
            }
            if (k % 2u != 0u) {
                return invalid("a merged bf16 matrix requires an even K (whole payload words per row)",
                               info.name);
            }
            break;
        case WeightStorage::fp8_packed:
            if (info.type != static_cast<uint32_t>(GgufTensorType::bf16) &&
                info.type != static_cast<uint32_t>(GgufTensorType::f32)) {
                return invalid("an fp8_e4m3 projection pack requires BF16/F32 source matrices",
                               info.name);
            }
            if (k % 4u != 0u) {
                return invalid("a merged fp8_e4m3 matrix requires K % 4 == 0 (whole quad words)",
                               info.name);
            }
            break;
        case WeightStorage::q8_0_packed:
            if (info.type != static_cast<uint32_t>(GgufTensorType::q8_0)) {
                return invalid("a q8_0 projection pack requires a native Q8_0 checkpoint "
                               "(the block codes are copied, never re-quantized)",
                               info.name);
            }
            if (k % kQ8_0BlockSize != 0u) {
                return invalid("a merged q8_0 matrix requires K % 32 == 0 (whole blocks)", info.name);
            }
            break;
        default:
            return invalid("unsupported projection-pack format");
    }
    return Status::success();
}

}// namespace

[[nodiscard]] Status pack_projection_pack_from_gguf(const GgufReader &reader_in,
                                                    const Qwen35Model &model,
                                                    luisa::string_view source_name,
                                                    const luisa::filesystem::path &dir,
                                                    const ProjectionPackOptions &options,
                                                    ProjectionPackReport *report) noexcept {
    auto &reader = const_cast<GgufReader &>(reader_in);
    auto inventory = qwen35_merged_projection_inventory(model);
    if (!inventory.ok()) { return inventory.status; }
    auto epw = projection_pack_elements_per_word(options.format);
    if (!epw.ok()) { return epw.status; }
    const uint64_t chunk = clamp_chunk(options.chunk_bytes);
    const uint64_t source_size = reader.file_size();
    // The canonical scale geometry of every matrix follows from the format and K,
    // and every matrix of one model shares K, so one probe is enough; validate each
    // matrix anyway (a heterogeneous model fails closed instead of mixing groups).
    ProjectionPackManifest manifest;
    manifest.version = kProjectionPackVersion;
    manifest.source_name = luisa::string{source_name};
    manifest.source_size = source_size;
    manifest.storage = options.format;
    manifest.elements_per_word = *epw.value;
    manifest.format_order = luisa::string{weight_storage_name(options.format)};
    {
        auto spr = projection_pack_scales_per_row(options.format,
                                                  options.format == WeightStorage::q8_0_packed
                                                      ? kQ8_0BlockSize
                                                      : options.group,
                                                  inventory.value->empty() ? 0u : inventory.value->front().k);
        if (!spr.ok()) { return spr.status; }
        manifest.scales_per_row = *spr.value;
        manifest.group = options.format == WeightStorage::bf16_packed
                             ? 0u
                             : (options.format == WeightStorage::q8_0_packed ? kQ8_0BlockSize
                                                                            : options.group);
    }
    std::error_code ec;
    luisa::filesystem::create_directories(dir / "matrices", ec);
    if (ec) {
        return io_error("cannot create the projection-pack directory", luisa::to_string(dir / "matrices"));
    }
    ProjectionPackReport local_report;
    auto &rep = report != nullptr ? *report : local_report;
    rep = ProjectionPackReport{};
    const auto t_begin = std::chrono::steady_clock::now();
    // Per-tensor source hashes, folded into the checkpoint digest in inventory order.
    uint64_t digest_state = kFnv1aOffsetBasis;
    luisa::vector<uint32_t> word_staging;
    luisa::vector<float> scale_staging;
    luisa::vector<uint8_t> code_staging;
    luisa::vector<float> value_staging;
    luisa::vector<std::byte> raw_staging;
    for (size_t mi = 0; mi < inventory.value->size(); ++mi) {
        const auto &spec = (*inventory.value)[mi];
        ProjectionPackMatrixRecord rec;
        rec.name = projection_pack_matrix_name(spec.kind, spec.layer);
        rec.kind = spec.kind;
        rec.layer = spec.layer;
        rec.k = spec.k;
        rec.n = spec.n;
        rec.storage = manifest.storage;
        rec.group = manifest.group;
        rec.scales_per_row = manifest.scales_per_row;
        luisa::filesystem::path words_path, scales_path;
        if (auto s = projection_pack_payload_paths(dir, mi, words_path, scales_path); !s.ok()) { return s; }
        std::ofstream words_out, scales_out;
        if (auto s = open_out(words_path, words_out); !s.ok()) { return s; }
        if (rec.scales_per_row != 0u) {
            if (auto s = open_out(scales_path, scales_out); !s.ok()) { return s; }
        }
        HashWriter words_hw{&words_out};
        HashWriter scales_hw{rec.scales_per_row != 0u ? &scales_out : nullptr};
        const uint64_t words_per_row = spec.k / manifest.elements_per_word;
        for (const auto &want_seg : spec.segments) {
            const auto *info = reader.find(want_seg.name);
            if (info == nullptr) {
                return Status::error(StatusCode::not_found,
                                     "merged projection source tensor is missing from the checkpoint",
                                     want_seg.name);
            }
            if (auto s = check_segment_source(manifest.storage, *info, spec.k, want_seg.rows); !s.ok()) {
                return s;
            }
            QwenTensorSegment seg = want_seg;
            seg.source_bytes = info->data_size;
            const uint64_t row_bytes = source_row_bytes(manifest.storage, info->type, spec.k);
            const uint64_t rows_per_chunk =
                std::max<uint64_t>(1u, chunk / std::max<uint64_t>(row_bytes, 1u));
            HashWriter source_hw{nullptr};
            uint64_t row = 0;
            while (row < want_seg.rows) {
                const uint64_t rows = std::min<uint64_t>(rows_per_chunk, want_seg.rows - row);
                const uint64_t elems = rows * spec.k;
                raw_staging.resize(static_cast<size_t>(rows * row_bytes));
                if (auto s = reader.read_tensor_bytes_at(*info, row * row_bytes, raw_staging); !s.ok()) {
                    return s;
                }
                source_hw.write(raw_staging);
                if (manifest.storage == WeightStorage::q8_0_packed) {
                    word_staging.resize(static_cast<size_t>(elems / 4u));
                    scale_staging.resize(static_cast<size_t>(elems / kQ8_0BlockSize));
                    if (auto s = pack_q8_0_blocks(raw_staging, elems, word_staging, scale_staging);
                        !s.ok()) {
                        return s;
                    }
                } else if (manifest.storage == WeightStorage::bf16_packed) {
                    word_staging.resize(static_cast<size_t>((elems + 1u) / 2u));
                    if (auto s = pack_bf16_tensor(raw_staging, elems, word_staging); !s.ok()) { return s; }
                } else {
                    // fp8_e4m3: decode the source rows, quantize them row by row with
                    // the SAME host reference the uploader uses, then quad-pack.
                    value_staging.resize(static_cast<size_t>(elems));
                    if (info->type == static_cast<uint32_t>(GgufTensorType::f32)) {
                        if (raw_staging.size() != elems * 4u) { return invalid("f32 source size mismatch", info->name); }
                        std::memcpy(value_staging.data(), raw_staging.data(),
                                    static_cast<size_t>(elems * 4u));
                    } else {
                        if (raw_staging.size() != elems * 2u) { return invalid("bf16 source size mismatch", info->name); }
                        for (uint64_t i = 0; i < elems; ++i) {
                            const auto lo = static_cast<uint32_t>(
                                static_cast<uint8_t>(raw_staging[2u * i]));
                            const auto hi = static_cast<uint32_t>(
                                static_cast<uint8_t>(raw_staging[2u * i + 1u]));
                            value_staging[i] = bf16_to_f32(static_cast<uint16_t>(lo | (hi << 8u)));
                        }
                    }
                    code_staging.resize(static_cast<size_t>(elems));
                    scale_staging.resize(static_cast<size_t>(rows * rec.scales_per_row));
                    for (uint64_t r = 0; r < rows; ++r) {
                        if (auto s = quantize_fp8_row(
                                luisa::span<const float>{value_staging.data() + r * spec.k,
                                                         static_cast<size_t>(spec.k)},
                                luisa::span<uint8_t>{code_staging.data() + r * spec.k,
                                                     static_cast<size_t>(spec.k)},
                                luisa::span<float>{scale_staging.data() + r * rec.scales_per_row,
                                                   static_cast<size_t>(rec.scales_per_row)});
                            !s.ok()) {
                            return s;
                        }
                    }
                    word_staging.resize(static_cast<size_t>((elems + 3u) / 4u));
                    if (auto s = pack_fp8_quads(code_staging, word_staging); !s.ok()) { return s; }
                }
                const auto word_bytes = luisa::span<const std::byte>{
                    reinterpret_cast<const std::byte *>(word_staging.data()),
                    word_staging.size() * 4u};
                if (!words_hw.write(word_bytes)) {
                    return io_error("projection-pack word payload write failed", luisa::to_string(words_path));
                }
                if (rec.scales_per_row != 0u) {
                    const auto scale_bytes = luisa::span<const std::byte>{
                        reinterpret_cast<const std::byte *>(scale_staging.data()),
                        scale_staging.size() * 4u};
                    if (!scales_hw.write(scale_bytes)) {
                        return io_error("projection-pack scale payload write failed",
                                        luisa::to_string(scales_path));
                    }
                }
                row += rows;
            }
            if (source_hw.bytes != info->data_size) {
                return io_error("merged projection source tensor byte count mismatch",
                                info->name);
            }
            seg.source_hash = source_hw.hash;
            rep.source_bytes_read += info->data_size;
            digest_state = w4_roll_source_hash(digest_state, seg.source_hash);
            rec.segments.emplace_back(std::move(seg));
        }
        words_out.close();
        if (!words_out) {
            return io_error("projection-pack word payload write failed", luisa::to_string(words_path));
        }
        if (rec.scales_per_row != 0u) {
            scales_out.close();
            if (!scales_out) {
                return io_error("projection-pack scale payload write failed", luisa::to_string(scales_path));
            }
        }
        rec.word_bytes = words_hw.bytes;
        rec.word_hash = words_hw.hash;
        rec.scale_bytes = scales_hw.bytes;
        rec.scale_hash = rec.scale_bytes != 0u ? scales_hw.hash : 0u;
        const uint64_t expect_words = spec.n * words_per_row * 4u;
        const uint64_t expect_scales = spec.n * rec.scales_per_row * 4u;
        if (rec.word_bytes != expect_words || rec.scale_bytes != expect_scales) {
            return invalid("projection-pack payload size does not match the geometry", rec.name);
        }
        rep.word_bytes += rec.word_bytes;
        rep.scale_bytes += rec.scale_bytes;
        rep.segments += rec.segments.size();
        manifest.matrices.emplace_back(std::move(rec));
    }
    manifest.source_hash = digest_state;
    rep.matrices = manifest.matrices.size();
    rep.source_hash = manifest.source_hash;
    rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_begin).count();
    if (auto s = validate_projection_pack_manifest(manifest, *inventory.value); !s.ok()) { return s; }
    return write_projection_pack_manifest(dir, manifest);
}

// ---------------------------------------------------------------------------
// Device consumer: append the merged matrices + their segment aliases.
// ---------------------------------------------------------------------------
namespace {

// Stream one payload file into `dst`, verifying its byte size and hash. `T` is the
// buffer element type (the payload ABI is 32-bit words for the codes and FP32 for the
// scales, so both instantiations are plain 4-byte copies).
template<typename T>
[[nodiscard]] Status upload_payload(compute::Stream &stream, const luisa::filesystem::path &path,
                                    compute::Buffer<T> &dst, uint64_t bytes,
                                    uint64_t expect_hash, uint64_t chunk) {
    if (!dst || dst.size() * sizeof(T) != bytes) {
        return invalid("projection-pack payload size does not match the allocated buffer",
                       luisa::to_string(path));
    }
    std::ifstream in;
    if (auto s = open_in(path, in); !s.ok()) { return s; }
    luisa::vector<T> staging(static_cast<size_t>(std::max<uint64_t>(1u, chunk / sizeof(T))));
    HashWriter hw{nullptr};
    uint64_t done = 0;
    const uint64_t words = bytes / sizeof(T);
    while (done < words) {
        const uint64_t n = std::min<uint64_t>(staging.size(), words - done);
        const auto span = luisa::span<std::byte>{reinterpret_cast<std::byte *>(staging.data()),
                                                 static_cast<size_t>(n * sizeof(T))};
        if (auto s = read_exact(in, span, path); !s.ok()) { return s; }
        hw.write(span);
        stream << dst.view(static_cast<size_t>(done), static_cast<size_t>(n))
                      .copy_from(luisa::span<const T>{staging.data(), static_cast<size_t>(n)});
        done += n;
    }
    if (hw.bytes != bytes) {
        return io_error("projection-pack payload byte count mismatch", luisa::to_string(path));
    }
    if (hw.hash != expect_hash) {
        return Status::error(StatusCode::unvalidated,
                             "projection-pack payload hash mismatch (stale or corrupt pack)",
                             luisa::to_string(path));
    }
    return Status::success();
}

}// namespace

[[nodiscard]] bool QwenDeviceWeights::projection_pack_covers(const ProjectionPackManifest &manifest,
                                                             QwenTensorRole role,
                                                             uint64_t layer) noexcept {
    for (const auto &m : manifest.matrices) {
        if (m.layer != layer) { continue; }
        for (const auto &seg : m.segments) {
            if (seg.role == role) { return true; }
        }
    }
    return false;
}

[[nodiscard]] Status QwenDeviceWeights::append_merged_projections(
    compute::Device &device, compute::Stream &stream, const luisa::filesystem::path &dir,
    const ProjectionPackManifest &manifest, luisa::vector<QwenDeviceTensor> &tensors,
    uint64_t chunk_bytes, QwenWeightUploadReport *report) noexcept {
    const uint64_t chunk = clamp_chunk(chunk_bytes);
    const uint64_t base = tensors.size();
    luisa::vector<QwenDeviceTensor> local;
    local.reserve(manifest.matrices.size() * 5u);
    for (size_t mi = 0; mi < manifest.matrices.size(); ++mi) {
        const auto &rec = manifest.matrices[mi];
        auto epw = projection_pack_elements_per_word(rec.storage);
        if (!epw.ok()) { return epw.status; }
        auto spr = projection_pack_scales_per_row(rec.storage, rec.group, rec.k);
        if (!spr.ok()) { return spr.status; }
        if (rec.segments.empty()) { return invalid("merged matrix has no segments", rec.name); }
        uint64_t words_per_row = rec.k / *epw.value;
        if (words_per_row == 0u) {
            return invalid("merged matrix word stride overflow", rec.name);
        }
        uint64_t words = 0, scales = 0;
        if (!checked_mul(words_per_row, rec.n, words) || !checked_mul(rec.scales_per_row, rec.n, scales)) {
            return invalid("merged matrix element count overflow", rec.name);
        }
        QwenDeviceTensor parent;
        parent.role = rec.segments.front().role;
        parent.layer = rec.layer;
        parent.name = rec.name;
        parent.storage = rec.storage;
        parent.dims = {rec.k, rec.n};
        parent.element_count = rec.k * rec.n;
        parent.groups_per_row = rec.scales_per_row;
        parent.group = rec.group;
        parent.merge_kind = rec.kind;
        parent.segments = rec.segments;
        parent.packed = device.create_buffer<uint32_t>(static_cast<size_t>(words));
        if (!parent.packed) {
            return Status::error(StatusCode::out_of_memory,
                                 "device buffer allocation failed", rec.name);
        }
        parent.packed.set_name(rec.name.c_str());
        if (scales != 0u) {
            parent.scales = device.create_buffer<float>(static_cast<size_t>(scales));
            if (!parent.scales) {
                return Status::error(StatusCode::out_of_memory,
                                     "device buffer allocation failed", rec.name + ".scales");
            }
            parent.scales.set_name((rec.name + ".scales").c_str());
        }
        luisa::filesystem::path words_path, scales_path;
        if (auto s = projection_pack_payload_paths(dir, mi, words_path, scales_path); !s.ok()) { return s; }
        if (auto s = upload_payload(stream, words_path, parent.packed, rec.word_bytes,
                                    rec.word_hash, chunk);
            !s.ok()) {
            return s;
        }
        if (scales != 0u && rec.scale_bytes != 0u) {
            if (auto s = upload_payload(stream, scales_path, parent.scales, rec.scale_bytes,
                                        rec.scale_hash, chunk);
                !s.ok()) {
                return s;
            }
        }
        if (report != nullptr) {
            report->projection_pack_matrices++;
            report->projection_pack_bytes += rec.word_bytes + rec.scale_bytes;
            report->projection_pack_segments += rec.segments.size();
            // The merged matrix is real device residency exactly like an uploaded
            // tensor; the skip path already kept the source bytes out of the count, so
            // device_bytes stays the HONEST total (equal to the unmerged upload).
            report->device_bytes += rec.word_bytes + rec.scale_bytes;
        }
        const uint64_t parent_index = base + static_cast<uint64_t>(local.size());
        local.emplace_back(std::move(parent));
        // Copy the buffer handles out before appending aliases: emplace_back may
        // reallocate `local`, which would invalidate a reference to the parent.
        const auto merged_packed = local.back().packed;
        const auto merged_scales = local.back().scales;
        for (const auto &seg : rec.segments) {
            QwenDeviceTensor alias;
            alias.role = seg.role;
            alias.layer = rec.layer;
            alias.name = seg.name;
            alias.storage = rec.storage;
            alias.dims = {rec.k, seg.rows};
            alias.element_count = rec.k * seg.rows;
            alias.groups_per_row = rec.scales_per_row;
            alias.group = rec.group;
            alias.merge_parent_index = parent_index;
            alias.merge_row_offset = seg.row_offset;
            alias.packed = merged_packed;
            alias.packed_view = merged_packed.view(
                static_cast<size_t>(seg.row_offset * words_per_row),
                static_cast<size_t>(seg.rows * words_per_row));
            if (merged_scales) {
                alias.scales = merged_scales;
                alias.scales_view = merged_scales.view(
                    static_cast<size_t>(seg.row_offset * rec.scales_per_row),
                    static_cast<size_t>(seg.rows * rec.scales_per_row));
            }
            if (!alias.packed_view || (merged_scales && !alias.scales_view)) {
                return invalid("merged segment view is empty", alias.name);
            }
            local.emplace_back(std::move(alias));
        }
    }
    // Success only: splice the whole set in one go (no partial merge is observable).
    for (auto &t : local) { tensors.emplace_back(std::move(t)); }
    return Status::success();
}

}// namespace luisa::qwen
