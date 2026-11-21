// Qwen3.5-4B -- OFFLINE W4A16 pack: manifest, deterministic packer, fail-closed
// device loader. Host code only for the packer; the loader needs a Device.
#include "qwen/w4_pack.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>

namespace luisa::qwen {
namespace {

constexpr uint64_t kMinChunkBytes = 64ull * 1024ull;
constexpr uint64_t kMaxChunkBytes = 128ull * 1024ull * 1024ull;

[[nodiscard]] uint64_t clamp_chunk(uint64_t requested) noexcept {
    const uint64_t c = std::clamp<uint64_t>(requested, kMinChunkBytes, kMaxChunkBytes);
    return c & ~15ull;
}

[[nodiscard]] Status io_error(luisa::string msg, luisa::string detail = {}) {
    return Status::error(StatusCode::io_error, std::move(msg), std::move(detail));
}
[[nodiscard]] Status invalid(luisa::string msg, luisa::string detail = {}) {
    return Status::error(StatusCode::invalid_argument, std::move(msg), std::move(detail));
}

[[nodiscard]] luisa::filesystem::path shard_name(uint64_t index, const char *suffix) {
    return luisa::filesystem::path{"tensors"} /
           luisa::filesystem::path{std::string{std::to_string(index)} + suffix};
}

// Hash + write (or just hash) one buffer in bounded chunks.
struct HashWriter {
    std::ofstream *out{nullptr};
    uint64_t hash{kFnv1aOffsetBasis};
    uint64_t bytes{0};
    [[nodiscard]] bool write(luisa::span<const std::byte> data) noexcept {
        hash = qwen_fnv1a64(data, hash);
        bytes += data.size();
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
        return io_error("cannot open pack payload for writing", luisa::to_string(path));
    }
    return Status::success();
}

[[nodiscard]] Status open_in(const luisa::filesystem::path &path, std::ifstream &in) {
    in.open(path, std::ios::binary);
    if (!in.is_open()) {
        return io_error("cannot open pack payload for reading", luisa::to_string(path));
    }
    return Status::success();
}

// Read exactly `bytes` bytes at the current stream position.
[[nodiscard]] Status read_exact(std::ifstream &in, luisa::span<std::byte> dst,
                                const luisa::filesystem::path &path) {
    in.read(reinterpret_cast<char *>(dst.data()), static_cast<std::streamsize>(dst.size()));
    if (in.gcount() != static_cast<std::streamsize>(dst.size())) {
        return io_error("pack payload is truncated", luisa::to_string(path));
    }
    return Status::success();
}

[[nodiscard]] Status write_manifest_text(const luisa::filesystem::path &path,
                                         luisa::string_view text) {
    std::ofstream out;
    out.open(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) { return io_error("cannot open the pack manifest for writing", luisa::to_string(path)); }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!out) { return io_error("pack manifest write failed", luisa::to_string(path)); }
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

[[nodiscard]] bool parse_f32(luisa::string_view tok, float &out) noexcept {
    luisa::string s{tok};
    char *end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0') { return false; }
    out = static_cast<float>(v);
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

// Tokenize one manifest line of `key=value` fields (the first token may be a
// bare tag with no '=').
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
            fields.push_back({tok.substr(0, eq), tok.substr(eq + 1)});
        }
    }
    return fields;
}

}// namespace

[[nodiscard]] uint64_t qwen_fnv1a64(luisa::span<const std::byte> bytes, uint64_t seed) noexcept {
    uint64_t h = seed;
    for (auto b : bytes) {
        h ^= static_cast<uint64_t>(static_cast<uint8_t>(b));
        h *= kFnv1aPrime;
    }
    return h;
}

[[nodiscard]] luisa::string qwen_pack_hash_string(uint64_t hash) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(hash));
    return luisa::string{buf};
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------
[[nodiscard]] Status write_w4_pack_manifest(const luisa::filesystem::path &dir,
                                            const W4PackManifest &manifest) noexcept {
    luisa::string text;
    text.reserve(256u * (manifest.tensors.size() + 4u));
    const auto add = [&text](luisa::string_view s) { text.append(s); };
    add("qwen-w4-pack ");
    add(luisa::string{std::to_string(manifest.version).c_str()});
    add("\n");
    add("source name=");
    add(manifest.source_name);
    add(" size=");
    add(luisa::string{std::to_string(manifest.source_size).c_str()});
    add(" hash=");
    add(qwen_pack_hash_string(manifest.source_hash));
    add(" hash_algorithm=fnv1a64\n");
    add("clip group=");
    add(luisa::string{std::to_string(manifest.clip.group).c_str()});
    add(" min=");
    add(luisa::string{std::to_string(manifest.clip.min_ratio).c_str()});
    add(" max=");
    add(luisa::string{std::to_string(manifest.clip.max_ratio).c_str()});
    add(" steps=");
    add(luisa::string{std::to_string(manifest.clip.steps).c_str()});
    add("\n");
    add("order ");
    add(manifest.format_order);
    add("\n");
    if (manifest.has_importance) {
        // Importance-aware pack: the calibration that weighted the clipping
        // objective, pinned by payload hash, source digest and token count.
        add("importance name=");
        add(manifest.importance_name);
        add(" payload=");
        add(qwen_pack_hash_string(manifest.importance_hash));
        add(" source=");
        add(qwen_pack_hash_string(manifest.importance_source_hash));
        add(" tokens=");
        add(luisa::string{std::to_string(manifest.importance_tokens).c_str()});
        add(" hash_algorithm=fnv1a64\n");
    }
    for (const auto &t : manifest.tensors) {
        add("tensor ");
        add(t.name);
        add(" role=");
        add(luisa::string{std::to_string(static_cast<uint32_t>(t.role)).c_str()});
        add(" layer=");
        add(t.layer == UINT64_MAX ? luisa::string{"none"}
                                  : luisa::string{std::to_string(t.layer).c_str()});
        add(" k=");
        add(luisa::string{std::to_string(t.k).c_str()});
        add(" n=");
        add(luisa::string{std::to_string(t.n).c_str()});
        add(" storage=");
        add(weight_storage_name(t.storage));
        add(" group=");
        add(luisa::string{std::to_string(t.group).c_str()});
        const auto pair = [&add](const char *key, uint64_t bytes, uint64_t hash) {
            add(" ");
            add(key);
            add("=");
            add(luisa::string{std::to_string(bytes).c_str()});
            add(":");
            add(qwen_pack_hash_string(hash));
        };
        pair("src", t.source_bytes, t.source_hash);
        if (t.storage == WeightStorage::w4_packed) {
            pair("w4", t.w4_bytes, t.w4_hash);
            pair("w4s", t.w4s_bytes, t.w4s_hash);
            if (t.fp8_bytes != 0u) { pair("fp8", t.fp8_bytes, t.fp8_hash); }
            if (t.fp8s_bytes != 0u) { pair("fp8s", t.fp8s_bytes, t.fp8s_hash); }
        } else {
            pair("f32", t.f32_bytes, t.f32_hash);
        }
        add("\n");
    }
    return write_manifest_text(dir / "manifest.txt", text);
}

namespace {

// "bytes:hash" field.
[[nodiscard]] bool parse_pair(luisa::string_view v, uint64_t &bytes, uint64_t &hash) noexcept {
    const size_t colon = v.find(':');
    if (colon == luisa::string_view::npos) { return false; }
    return parse_u64(v.substr(0, colon), bytes) && parse_hex64(v.substr(colon + 1), hash);
}

}// namespace

[[nodiscard]] Result<W4PackManifest> read_w4_pack_manifest(const luisa::filesystem::path &dir) noexcept {
    const auto path = dir / "manifest.txt";
    std::ifstream in;
    if (auto s = open_in(path, in); !s.ok()) {
        return Result<W4PackManifest>::failure(std::move(s));
    }
    luisa::string line;
    W4PackManifest m;
    bool have_header = false, have_source = false, have_clip = false;
    uint64_t line_no = 0;
    while (std::getline(in, line)) {
        line_no++;
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) { line.pop_back(); }
        auto fields = split_fields(luisa::string_view{line});
        if (fields.empty()) { continue; }
        if (!have_header) {
            if (fields[0].key.size() != 0u || fields[0].value != "qwen-w4-pack" ||
                fields.size() != 2u) {
                return Result<W4PackManifest>::failure(
                    invalid("not a qwen W4 pack manifest", luisa::to_string(path)));
            }
            uint64_t version = 0;
            if (!parse_u64(fields[1].value, version) || version != kW4PackVersion) {
                return Result<W4PackManifest>::failure(invalid(
                    "unsupported W4 pack version (only " +
                        luisa::string{std::to_string(kW4PackVersion).c_str()} + " is accepted)",
                    luisa::to_string(path)));
            }
            m.version = static_cast<uint32_t>(version);
            have_header = true;
            continue;
        }
        if (fields[0].key.size() == 0u && fields[0].value == "source") {
            for (size_t i = 1u; i < fields.size(); ++i) {
                if (fields[i].key == "name") {
                    m.source_name = luisa::string{fields[i].value};
                } else if (fields[i].key == "size") {
                    if (!parse_u64(fields[i].value, m.source_size)) {
                        return Result<W4PackManifest>::failure(invalid("malformed manifest source size"));
                    }
                } else if (fields[i].key == "hash") {
                    if (!parse_hex64(fields[i].value, m.source_hash)) {
                        return Result<W4PackManifest>::failure(invalid("malformed manifest source hash"));
                    }
                }
            }
            have_source = true;
            continue;
        }
        if (fields[0].key.size() == 0u && fields[0].value == "clip") {
            for (size_t i = 1u; i < fields.size(); ++i) {
                uint64_t u = 0;
                if (fields[i].key == "group") {
                    if (!parse_u64(fields[i].value, u)) { return Result<W4PackManifest>::failure(invalid("malformed clip group")); }
                    m.clip.group = static_cast<uint32_t>(u);
                } else if (fields[i].key == "min") {
                    if (!parse_f32(fields[i].value, m.clip.min_ratio)) { return Result<W4PackManifest>::failure(invalid("malformed clip min")); }
                } else if (fields[i].key == "max") {
                    if (!parse_f32(fields[i].value, m.clip.max_ratio)) { return Result<W4PackManifest>::failure(invalid("malformed clip max")); }
                } else if (fields[i].key == "steps") {
                    if (!parse_u64(fields[i].value, u)) { return Result<W4PackManifest>::failure(invalid("malformed clip steps")); }
                    m.clip.steps = static_cast<uint32_t>(u);
                }
            }
            if (auto s = m.clip.validate(); !s.ok()) {
                return Result<W4PackManifest>::failure(std::move(s));
            }
            have_clip = true;
            continue;
        }
        if (fields[0].key.size() == 0u && fields[0].value == "order") {
            m.format_order.clear();
            for (size_t i = 1u; i < fields.size(); ++i) {
                if (fields[i].key.size() != 0u) { continue; }
                if (!m.format_order.empty()) { m.format_order.push_back(','); }
                m.format_order.append(fields[i].value);
            }
            m.has_fp8_payloads = m.format_order.find("fp8") != luisa::string::npos;
            continue;
        }
        if (fields[0].key.size() == 0u && fields[0].value == "importance") {
            if (m.has_importance) {
                return Result<W4PackManifest>::failure(
                    invalid("duplicate importance line in the W4 pack manifest"));
            }
            m.has_importance = true;
            for (size_t i = 1u; i < fields.size(); ++i) {
                const auto &f = fields[i];
                if (f.key == "name") {
                    m.importance_name = luisa::string{f.value};
                } else if (f.key == "payload") {
                    if (!parse_hex64(f.value, m.importance_hash)) {
                        return Result<W4PackManifest>::failure(invalid("malformed importance payload hash"));
                    }
                } else if (f.key == "source") {
                    if (!parse_hex64(f.value, m.importance_source_hash)) {
                        return Result<W4PackManifest>::failure(invalid("malformed importance source hash"));
                    }
                } else if (f.key == "tokens") {
                    if (!parse_u64(f.value, m.importance_tokens)) {
                        return Result<W4PackManifest>::failure(invalid("malformed importance token count"));
                    }
                } else if (f.key == "hash_algorithm") {
                    if (f.value != "fnv1a64") {
                        return Result<W4PackManifest>::failure(invalid("unsupported importance hash algorithm"));
                    }
                } else {
                    return Result<W4PackManifest>::failure(
                        invalid("unknown importance manifest field", luisa::string{f.key}));
                }
            }
            if (m.importance_tokens == 0u) {
                return Result<W4PackManifest>::failure(
                    invalid("importance-aware pack records 0 calibration tokens"));
            }
            continue;
        }
        if (fields[0].key.size() == 0u && fields[0].value == "tensor") {
            W4PackTensorRecord t;
            if (fields.size() < 2u) { return Result<W4PackManifest>::failure(invalid("tensor record without a name")); }
            t.name = luisa::string{fields[1].value};
            bool have_storage = false;
            for (size_t i = 2u; i < fields.size(); ++i) {
                const auto &f = fields[i];
                uint64_t u = 0;
                if (f.key == "role") {
                    if (!parse_u64(f.value, u)) { return Result<W4PackManifest>::failure(invalid("malformed tensor role")); }
                    t.role = static_cast<QwenTensorRole>(u);
                } else if (f.key == "layer") {
                    if (f.value == "none") {
                        t.layer = UINT64_MAX;
                    } else {
                        if (!parse_u64(f.value, u)) { return Result<W4PackManifest>::failure(invalid("malformed tensor layer")); }
                        t.layer = u;
                    }
                } else if (f.key == "k") {
                    if (!parse_u64(f.value, t.k)) { return Result<W4PackManifest>::failure(invalid("malformed tensor k")); }
                } else if (f.key == "n") {
                    if (!parse_u64(f.value, t.n)) { return Result<W4PackManifest>::failure(invalid("malformed tensor n")); }
                } else if (f.key == "group") {
                    if (!parse_u64(f.value, u)) { return Result<W4PackManifest>::failure(invalid("malformed tensor group")); }
                    t.group = static_cast<uint32_t>(u);
                } else if (f.key == "storage") {
                    if (f.value == "w4_packed") {
                        t.storage = WeightStorage::w4_packed;
                    } else if (f.value == "f32_dense") {
                        t.storage = WeightStorage::f32_dense;
                    } else {
                        return Result<W4PackManifest>::failure(
                            invalid("unsupported storage in a W4 pack", luisa::string{f.value}));
                    }
                    have_storage = true;
                } else if (f.key == "src") {
                    if (!parse_pair(f.value, t.source_bytes, t.source_hash)) { return Result<W4PackManifest>::failure(invalid("malformed src field")); }
                } else if (f.key == "w4") {
                    if (!parse_pair(f.value, t.w4_bytes, t.w4_hash)) { return Result<W4PackManifest>::failure(invalid("malformed w4 field")); }
                } else if (f.key == "w4s") {
                    if (!parse_pair(f.value, t.w4s_bytes, t.w4s_hash)) { return Result<W4PackManifest>::failure(invalid("malformed w4s field")); }
                } else if (f.key == "f32") {
                    if (!parse_pair(f.value, t.f32_bytes, t.f32_hash)) { return Result<W4PackManifest>::failure(invalid("malformed f32 field")); }
                } else if (f.key == "fp8") {
                    if (!parse_pair(f.value, t.fp8_bytes, t.fp8_hash)) { return Result<W4PackManifest>::failure(invalid("malformed fp8 field")); }
                } else if (f.key == "fp8s") {
                    if (!parse_pair(f.value, t.fp8s_bytes, t.fp8s_hash)) { return Result<W4PackManifest>::failure(invalid("malformed fp8s field")); }
                } else {
                    return Result<W4PackManifest>::failure(
                        invalid("unknown manifest field", luisa::string{f.key}));
                }
            }
            if (!have_storage) {
                return Result<W4PackManifest>::failure(invalid("tensor record without a storage", t.name));
            }
            if (t.storage == WeightStorage::w4_packed) {
                if (auto s = validate_w4_geometry(t.k, t.group); !s.ok()) {
                    return Result<W4PackManifest>::failure(std::move(s));
                }
            }
            m.tensors.push_back(std::move(t));
            continue;
        }
        return Result<W4PackManifest>::failure(
            invalid("unexpected line in the W4 pack manifest",
                    luisa::string{"line "} + luisa::string{std::to_string(line_no).c_str()}));
    }
    if (!have_header) {
        return Result<W4PackManifest>::failure(invalid("empty W4 pack manifest", luisa::to_string(path)));
    }
    if (!have_source || !have_clip || m.tensors.empty()) {
        return Result<W4PackManifest>::failure(invalid("incomplete W4 pack manifest", luisa::to_string(path)));
    }
    return Result<W4PackManifest>::success(std::move(m));
}

// ---------------------------------------------------------------------------
// Source identity (see w4_pack.h): the rolling payload digest of a checkpoint.
// ---------------------------------------------------------------------------
[[nodiscard]] uint64_t w4_roll_source_hash(uint64_t state, uint64_t tensor_hash) noexcept {
    std::byte le[8];
    for (uint32_t b = 0; b < 8u; ++b) {
        le[b] = static_cast<std::byte>((tensor_hash >> (8u * b)) & 0xFFu);
    }
    return qwen_fnv1a64(luisa::span<const std::byte>{le, 8u}, state);
}

[[nodiscard]] Status w4_hash_source_tensor(GgufReader &reader, const GgufTensorInfo &info,
                                           uint64_t chunk_bytes, uint64_t &hash_out) noexcept {
    const uint64_t chunk = clamp_chunk(chunk_bytes);
    luisa::vector<std::byte> staging(static_cast<size_t>(std::min(chunk, std::max<uint64_t>(1u, info.data_size))));
    HashWriter hw{nullptr};
    uint64_t done = 0;
    while (done < info.data_size) {
        const uint64_t n = std::min<uint64_t>(staging.size(), info.data_size - done);
        const auto span = luisa::span<std::byte>{staging.data(), static_cast<size_t>(n)};
        if (auto s = reader.read_tensor_bytes_at(info, done, span); !s.ok()) { return s; }
        hw.write(span);
        done += n;
    }
    if (hw.bytes != info.data_size) {
        return io_error("source tensor payload size changed while hashing", info.name);
    }
    hash_out = hw.hash;
    return Status::success();
}

// Same inventory walk + rolling rule the packer uses while it streams payloads.
[[nodiscard]] Status w4_source_digest_impl(GgufReader &reader,
                                           luisa::span<const QwenTensorRef> inventory,
                                           uint64_t chunk_bytes, uint64_t &digest_out) noexcept {
    if (inventory.empty()) {
        return invalid("the source-identity inventory is empty");
    }
    uint64_t digest = kFnv1aOffsetBasis;
    for (const auto &ref : inventory) {
        if (ref.info == nullptr) {
            return invalid("the source-identity inventory holds a null tensor");
        }
        uint64_t hash = 0;
        if (auto s = w4_hash_source_tensor(reader, *ref.info, chunk_bytes, hash); !s.ok()) {
            return s;
        }
        digest = w4_roll_source_hash(digest, hash);
    }
    digest_out = digest;
    return Status::success();
}

[[nodiscard]] Result<uint64_t> w4_source_digest(GgufReader &reader,
                                                luisa::span<const QwenTensorRef> inventory,
                                                uint64_t chunk_bytes) noexcept {
    uint64_t digest = 0;
    if (auto s = w4_source_digest_impl(reader, inventory, chunk_bytes, digest); !s.ok()) {
        return Result<uint64_t>::failure(std::move(s));
    }
    return Result<uint64_t>::success(digest);
}

[[nodiscard]] Result<uint64_t> compute_w4_source_hash(const GgufReader &reader,
                                                      const Qwen35Model &model,
                                                      uint64_t chunk_bytes) noexcept {
    auto &r = const_cast<GgufReader &>(reader);
    return w4_source_digest(r, model.tensors(), chunk_bytes);
}

[[nodiscard]] Status verify_w4_pack_source(GgufReader &reader,
                                           luisa::span<const QwenTensorRef> inventory,
                                           const W4PackManifest &manifest,
                                           uint64_t chunk_bytes) noexcept {
    const uint64_t file_size = reader.file_size();
    if (manifest.source_size != file_size) {
        return Status::error(
            StatusCode::unvalidated,
            "W4 pack was built from a different checkpoint (source size mismatch)",
            "pack=" + luisa::string{std::to_string(manifest.source_size).c_str()} +
                " checkpoint=" + luisa::string{std::to_string(file_size).c_str()});
    }
    uint64_t digest = 0;
    if (auto s = w4_source_digest_impl(reader, inventory, chunk_bytes, digest); !s.ok()) {
        return s;
    }
    if (digest != manifest.source_hash) {
        // Same byte count, different payload: exactly the case a size check cannot
        // catch, so it is refused rather than reported.
        return Status::error(
            StatusCode::unvalidated,
            "W4 pack was built from a different checkpoint with the SAME file size "
            "(rolling source payload digest mismatch)",
            "pack=" + qwen_pack_hash_string(manifest.source_hash) +
                " checkpoint=" + qwen_pack_hash_string(digest));
    }
    return Status::success();
}

[[nodiscard]] Status verify_w4_pack_source(const GgufReader &reader, const Qwen35Model &model,
                                           const W4PackManifest &manifest,
                                           uint64_t chunk_bytes) noexcept {
    auto &r = const_cast<GgufReader &>(reader);
    return verify_w4_pack_source(r, model.tensors(), manifest, chunk_bytes);
}

// ---------------------------------------------------------------------------
// Offline packer
// ---------------------------------------------------------------------------
namespace {

// Streaming reader of one GGUF tensor payload in F32 (decode BF16 exactly).
struct SourceReader {
    GgufReader &reader;
    const GgufTensorInfo &info;
    uint32_t element_bytes{0};// 2 (BF16) or 4 (F32)

    [[nodiscard]] Status read(luisa::span<std::byte> dst, uint64_t element_offset,
                              uint64_t element_count, HashWriter &hash) {
        const uint64_t byte_offset = element_offset * element_bytes;
        const uint64_t byte_count = element_count * element_bytes;
        if (dst.size() != byte_count) { return invalid("source read span size mismatch", info.name); }
        if (auto s = reader.read_tensor_bytes_at(info, byte_offset, dst); !s.ok()) { return s; }
        hash.write(dst);
        return Status::success();
    }
};

}// namespace

namespace {

// The packer proper. `full_inventory` says the span IS the checkpoint's complete
// inventory, which is what makes the rolling digest the checkpoint digest (and
// therefore what the importance staleness check compares against).
[[nodiscard]] Status pack_w4_from_gguf_impl(const GgufReader &reader_in,
                                            luisa::span<const QwenTensorRef> inventory,
                                            luisa::string_view source_name,
                                            const luisa::filesystem::path &dir,
                                            const W4PackOptions &options,
                                            bool full_inventory,
                                            W4PackReport *report) noexcept {
    if (auto s = options.clip.validate(); !s.ok()) { return s; }
    if (inventory.empty()) { return invalid("the W4 pack inventory is empty"); }
    auto &reader = const_cast<GgufReader &>(reader_in);
    const uint64_t file_size = reader_in.file_size();
    // ---- importance preflight: fail closed BEFORE any payload is written -----
    if (options.importance != nullptr) {
        const auto &imp = *options.importance;
        if (imp.version != kW4ImportanceVersion) {
            return Status::error(StatusCode::unvalidated,
                                 "unsupported importance statistics version",
                                 luisa::string{std::to_string(imp.version).c_str()});
        }
        if (imp.source.size != file_size) {
            return Status::error(StatusCode::unvalidated,
                                 "importance statistics were collected on a different checkpoint "
                                 "(source size mismatch)",
                                 luisa::string{"stats="} +
                                     luisa::string{std::to_string(imp.source.size).c_str()} +
                                     " checkpoint=" + luisa::string{std::to_string(file_size).c_str()});
        }
        if (imp.source.hash == 0u) {
            return Status::error(StatusCode::unvalidated,
                                 "importance statistics do not record the checkpoint payload digest "
                                 "(cannot be proven fresh)");
        }
        // Every tensor this pack will quantize must carry statistics for the SAME
        // K, and no statistics may be silently unused-by-accident.
        for (const auto &ref : inventory) {
            if (ref.info == nullptr) { return invalid("the W4 pack inventory holds a null tensor"); }
            const auto &info = *ref.info;
            const auto storage = classify_w4_pack_storage(info.name, info.dims, options.clip.group);
            if (!storage.ok()) { return storage.status; }
            if (*storage.value != WeightStorage::w4_packed) { continue; }
            if (auto st = imp.status_for(info.name, info.dims[0]); !st.ok()) { return st; }
        }
    }
    std::error_code ec;
    luisa::filesystem::create_directories(dir / "tensors", ec);
    if (ec) { return io_error("cannot create the pack directory", luisa::to_string(dir)); }
    const uint64_t chunk = clamp_chunk(options.chunk_bytes);
    W4PackManifest manifest;
    manifest.source_hash = kFnv1aOffsetBasis;
    manifest.version = kW4PackVersion;
    manifest.source_name = luisa::string{source_name};
    manifest.source_size = file_size;
    manifest.clip = options.clip;
    manifest.has_fp8_payloads = options.emit_fp8_payloads;
    manifest.format_order = options.emit_fp8_payloads ? "w4,fp8" : "w4";
    if (options.importance != nullptr) {
        manifest.has_importance = true;
        manifest.importance_name = options.importance->source.name;
        manifest.importance_source_hash = options.importance->source.hash;
        manifest.importance_tokens = options.importance->source.tokens;
        manifest.importance_hash = qwen_fnv1a64(luisa::span<const std::byte>{
            reinterpret_cast<const std::byte *>(options.importance->values.data()),
            options.importance->values.size() * sizeof(float)});
    }
    W4PackReport local{};
    auto &rep = report != nullptr ? *report : local;
    // Deterministic reset: a caller-supplied report never inherits numbers.
    rep = W4PackReport{};
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t fp8_group = kFp8GroupSize;
    // Model-wide accumulators. They are normalized ONCE at the end (see
    // W4PackReport): mse/rtn_mse are element-weighted, the weighted objectives and
    // the clipping ratio are group-weighted.
    double mse_sse = 0.0, rtn_sse = 0.0, weighted_sum = 0.0, rtn_weighted_sum = 0.0,
           ratio_sum = 0.0;
    uint64_t w4_elements = 0, groups_total = 0;
    uint64_t index = 0;
    for (const auto &ref : inventory) {
        if (ref.info == nullptr) { return invalid("the W4 pack inventory holds a null tensor"); }
        const auto &info = *ref.info;
        const uint32_t type = info.type;
        if (type != static_cast<uint32_t>(GgufTensorType::f32) &&
            type != static_cast<uint32_t>(GgufTensorType::bf16)) {
            return Status::error(StatusCode::unsupported,
                                 "a W4 pack must be produced from an F32/BF16 checkpoint "
                                 "(Q8_0 would be double-quantized)",
                                 luisa::string{info.name});
        }
        const auto storage = classify_w4_pack_storage(info.name, info.dims, options.clip.group);
        if (!storage.ok()) { return storage.status; }
        W4PackTensorRecord rec;
        rec.name = info.name;
        rec.role = ref.role;
        rec.layer = ref.layer;
        rec.storage = *storage.value;
        rec.group = rec.storage == WeightStorage::w4_packed ? options.clip.group : 0u;
        const uint32_t element_bytes = type == static_cast<uint32_t>(GgufTensorType::bf16) ? 2u : 4u;
        rec.source_bytes = info.element_count * element_bytes;
        SourceReader src{reader, info, element_bytes};
        HashWriter base_hash{nullptr};
        if (rec.storage == WeightStorage::f32_dense) {
            if (info.dims.size() != 1u && !info.name.ends_with(".ssm_conv1d.weight") &&
                info.dims.size() != 2u) {
                return invalid("dense F32 selection for a non-1-D tensor", luisa::string{info.name});
            }
            // Dense tensors (1-D norms / ssm_a / ssm_dt.bias, and the depthwise
            // conv) are stored as a flat F32 vector: k = element count, n = 1.
            rec.k = info.element_count;
            rec.n = 1u;
            std::ofstream out;
            if (auto s = open_out(dir / shard_name(index, ".f32.bin"), out); !s.ok()) { return s; }
            HashWriter hw{&out};
            const uint64_t elements = info.element_count;
            luisa::vector<std::byte> raw(static_cast<size_t>(std::min(chunk, elements * element_bytes)));
            luisa::vector<float> values;
            uint64_t done = 0;
            while (done < elements) {
                const uint64_t n = std::min<uint64_t>(raw.size() / element_bytes, elements - done);
                const auto span = luisa::span<std::byte>{raw.data(), static_cast<size_t>(n * element_bytes)};
                if (auto s = src.read(span, done, n, base_hash); !s.ok()) { return s; }
                if (element_bytes == 4u) {
                    hw.write(span);
                } else {
                    values.resize(static_cast<size_t>(n));
                    for (uint64_t i = 0; i < n; ++i) {
                        const auto lo = static_cast<uint32_t>(static_cast<uint8_t>(span[2u * i]));
                        const auto hi = static_cast<uint32_t>(static_cast<uint8_t>(span[2u * i + 1u]));
                        values[static_cast<size_t>(i)] =
                            bf16_to_f32(static_cast<uint16_t>(lo | (hi << 8u)));
                    }
                    hw.write(luisa::span<const std::byte>{
                        reinterpret_cast<const std::byte *>(values.data()),
                        values.size() * sizeof(float)});
                }
                done += n;
            }
            if (!out) { return io_error("f32 payload write failed", luisa::to_string(dir)); }
            out.close();
            rec.f32_bytes = hw.bytes;
            rec.f32_hash = hw.hash;
            rec.source_hash = base_hash.hash;
            rep.dense_tensors++;
            rep.f32_bytes += hw.bytes;
        } else {
            if (info.dims.size() != 2u) {
                return invalid("w4 selection for a non-matrix tensor", luisa::string{info.name});
            }
            const uint64_t k = info.dims[0];
            const uint64_t n = info.dims[1];
            rec.k = k;
            rec.n = n;
            const uint64_t gpr = k / options.clip.group;
            const uint64_t fp8_gpr = k / fp8_group;
            luisa::span<const float> importance;
            if (options.importance != nullptr) {
                importance = options.importance->channels(info.name, k);
                if (importance.empty()) {
                    // The preflight proved the entry exists; an empty span here is a
                    // real inconsistency, never a silent fallback.
                    return Status::error(StatusCode::unvalidated,
                                         "importance statistics vanished between preflight and packing",
                                         luisa::string{info.name});
                }
                rep.importance_matrices++;
            } else {
                rep.unweighted_matrices++;
            }
            // Row blocks so the host staging stays bounded (a row is 5-18 KiB).
            const uint64_t row_bytes = k * element_bytes;
            const uint64_t rows_per_chunk = std::max<uint64_t>(1u, chunk / std::max<uint64_t>(1u, row_bytes));
            std::ofstream w4_out, w4s_out, fp8_out, fp8s_out;
            if (auto s = open_out(dir / shard_name(index, ".w4.bin"), w4_out); !s.ok()) { return s; }
            if (auto s = open_out(dir / shard_name(index, ".w4s.bin"), w4s_out); !s.ok()) { return s; }
            if (options.emit_fp8_payloads) {
                if (auto s = open_out(dir / shard_name(index, ".fp8.bin"), fp8_out); !s.ok()) { return s; }
                if (auto s = open_out(dir / shard_name(index, ".fp8s.bin"), fp8s_out); !s.ok()) { return s; }
            }
            HashWriter w4h{&w4_out}, w4sh{&w4s_out}, fp8h{&fp8_out}, fp8sh{&fp8s_out};
            luisa::vector<std::byte> raw(static_cast<size_t>(rows_per_chunk * row_bytes));
            luisa::vector<float> values(static_cast<size_t>(rows_per_chunk * k));
            luisa::vector<uint32_t> w4_words;
            luisa::vector<float> w4_scales;
            luisa::vector<uint32_t> fp8_words;
            luisa::vector<float> fp8_scales;
            for (uint64_t row0 = 0; row0 < n; row0 += rows_per_chunk) {
                const uint64_t rows = std::min<uint64_t>(rows_per_chunk, n - row0);
                const uint64_t elements = rows * k;
                const auto span = luisa::span<std::byte>{raw.data(), static_cast<size_t>(elements * element_bytes)};
                if (auto s = src.read(span, row0 * k, elements, base_hash); !s.ok()) { return s; }
                for (uint64_t i = 0; i < elements; ++i) {
                    if (element_bytes == 4u) {
                        std::memcpy(&values[static_cast<size_t>(i)], span.data() + 4u * i, 4u);
                    } else {
                        const auto lo = static_cast<uint32_t>(static_cast<uint8_t>(span[2u * i]));
                        const auto hi = static_cast<uint32_t>(static_cast<uint8_t>(span[2u * i + 1u]));
                        values[static_cast<size_t>(i)] =
                            bf16_to_f32(static_cast<uint16_t>(lo | (hi << 8u)));
                    }
                }
                // The ONE deterministic clipping search of the tree
                // (common.h::pack_w4_matrix / pack_w4_matrix_weighted);
                // dsl::host_pack_w4_matrix is the device-layer mirror of the
                // UNWEIGHTED search and a host test proves they agree bit for bit.
                W4PackStats stats{};
                // pack_w4_matrix writes through caller-sized spans: size them for
                // this row block (elements/8 words, rows * k/group scales).
                w4_words.assign(static_cast<size_t>(elements / kW4CodesPerWord), 0u);
                w4_scales.assign(static_cast<size_t>(rows * gpr), 0.0f);
                const auto values_span = luisa::span<const float>{values.data(), static_cast<size_t>(elements)};
                const auto words_span = luisa::span<uint32_t>{w4_words.data(), w4_words.size()};
                const auto scales_span = luisa::span<float>{w4_scales.data(), w4_scales.size()};
                auto status = options.importance != nullptr
                                  ? pack_w4_matrix_weighted(values_span, importance, k, rows,
                                                            options.clip, words_span, scales_span, &stats)
                                  : pack_w4_matrix(values_span, k, rows, options.clip,
                                                   words_span, scales_span, &stats);
                if (!status.ok()) { return status; }
                const uint64_t block_groups = rows * gpr;
                rep.clipped_groups += stats.clipped_groups;
                mse_sse += stats.mse * static_cast<double>(elements);
                rtn_sse += stats.rtn_mse * static_cast<double>(elements);
                weighted_sum += stats.weighted_mse * static_cast<double>(block_groups);
                rtn_weighted_sum += stats.rtn_weighted_mse * static_cast<double>(block_groups);
                ratio_sum += static_cast<double>(stats.mean_ratio) * static_cast<double>(block_groups);
                groups_total += block_groups;
                w4h.write(luisa::span<const std::byte>{
                    reinterpret_cast<const std::byte *>(w4_words.data()),
                    w4_words.size() * sizeof(uint32_t)});
                w4sh.write(luisa::span<const std::byte>{
                    reinterpret_cast<const std::byte *>(w4_scales.data()),
                    w4_scales.size() * sizeof(float)});
                if (options.emit_fp8_payloads) {
                    const uint64_t words = (elements + 3u) / 4u;
                    fp8_words.assign(static_cast<size_t>(words), 0u);
                    uint64_t scale_count = 0;
                    if (!checked_mul(fp8_gpr, rows, scale_count)) {
                        return invalid("fp8 scale count overflow", luisa::string{info.name});
                    }
                    fp8_scales.assign(static_cast<size_t>(scale_count), 0.0f);
                    if (auto s = quantize_fp8_matrix(values_span, k, rows,
                                                     luisa::span<uint32_t>{fp8_words.data(), fp8_words.size()},
                                                     luisa::span<float>{fp8_scales.data(), fp8_scales.size()});
                        !s.ok()) {
                        return s;
                    }
                    fp8h.write(luisa::span<const std::byte>{
                        reinterpret_cast<const std::byte *>(fp8_words.data()),
                        fp8_words.size() * sizeof(uint32_t)});
                    fp8sh.write(luisa::span<const std::byte>{
                        reinterpret_cast<const std::byte *>(fp8_scales.data()),
                        fp8_scales.size() * sizeof(float)});
                }
            }
            if (!w4_out || !w4s_out || (options.emit_fp8_payloads && (!fp8_out || !fp8s_out))) {
                return io_error("pack payload write failed", luisa::to_string(dir));
            }
            w4_out.close();
            w4s_out.close();
            fp8_out.close();
            fp8s_out.close();
            rec.source_hash = base_hash.hash;
            rec.w4_bytes = w4h.bytes;
            rec.w4_hash = w4h.hash;
            rec.w4s_bytes = w4sh.bytes;
            rec.w4s_hash = w4sh.hash;
            if (options.emit_fp8_payloads) {
                rec.fp8_bytes = fp8h.bytes;
                rec.fp8_hash = fp8h.hash;
                rec.fp8s_bytes = fp8sh.bytes;
                rec.fp8s_hash = fp8sh.hash;
            }
            rep.w4_matrices++;
            rep.w4_bytes += rec.w4_bytes;
            rep.w4s_bytes += rec.w4s_bytes;
            rep.fp8_bytes += rec.fp8_bytes;
            rep.fp8s_bytes += rec.fp8s_bytes;
            w4_elements += k * n;
        }
        // Roll the per-tensor source hash into the manifest's source digest.
        manifest.source_hash = w4_roll_source_hash(manifest.source_hash, rec.source_hash);
        rep.tensors++;
        manifest.tensors.push_back(std::move(rec));
        index++;
    }
    // Normalize the aggregates exactly once, over the whole model.
    rep.groups = groups_total;
    rep.w4_elements = w4_elements;
    rep.mse = w4_elements != 0u ? mse_sse / static_cast<double>(w4_elements) : 0.0;
    rep.rtn_mse = w4_elements != 0u ? rtn_sse / static_cast<double>(w4_elements) : 0.0;
    rep.weighted_mse = groups_total != 0u ? weighted_sum / static_cast<double>(groups_total) : 0.0;
    rep.rtn_weighted_mse = groups_total != 0u ? rtn_weighted_sum / static_cast<double>(groups_total) : 0.0;
    rep.mean_ratio = groups_total != 0u ? static_cast<float>(ratio_sum / static_cast<double>(groups_total))
                                        : 1.0f;
    // Importance statistics must describe the SAME payload the pack was built from
    // (a same-size/different-payload checkpoint is stale, not "close enough").
    if (options.importance != nullptr && full_inventory &&
        manifest.source_hash != options.importance->source.hash) {
        return Status::error(StatusCode::unvalidated,
                             "importance statistics were collected on a different checkpoint with "
                             "the SAME file size (rolling source payload digest mismatch)",
                             "stats=" + qwen_pack_hash_string(options.importance->source.hash) +
                                 " checkpoint=" + qwen_pack_hash_string(manifest.source_hash));
    }
    // The manifest goes LAST: a pack without a manifest is an incomplete pack.
    const auto t1 = std::chrono::steady_clock::now();
    rep.seconds = std::chrono::duration<double>(t1 - t0).count();
    if (auto s = write_w4_pack_manifest(dir, manifest); !s.ok()) { return s; }
    return Status::success();
}

}// namespace

[[nodiscard]] Status pack_w4_from_gguf(const GgufReader &reader,
                                       luisa::span<const QwenTensorRef> inventory,
                                       luisa::string_view source_name,
                                       const luisa::filesystem::path &dir,
                                       const W4PackOptions &options,
                                       W4PackReport *report) noexcept {
    // A subset inventory is a tooling/test pack: the rolling digest then identifies
    // the SUBSET, so it cannot be compared with a calibration of the full model.
    return pack_w4_from_gguf_impl(reader, inventory, source_name, dir, options,
                                  options.full_inventory, report);
}

[[nodiscard]] Status pack_w4_from_gguf(const GgufReader &reader, const Qwen35Model &model,
                                       luisa::string_view source_name,
                                       const luisa::filesystem::path &dir,
                                       const W4PackOptions &options,
                                       W4PackReport *report) noexcept {
    // The model inventory IS the checkpoint's full inventory: the digest is the
    // checkpoint digest and the importance staleness check is active.
    auto full = options;
    full.full_inventory = true;
    return pack_w4_from_gguf_impl(reader, model.tensors(), source_name, dir, full, true, report);
}

// ---------------------------------------------------------------------------
// Device upload
// ---------------------------------------------------------------------------
namespace {

struct SyncGuard {
    compute::Stream &stream;
    bool done{false};
    ~SyncGuard() noexcept {
        if (!done) { stream.synchronize(); }
    }
};

// Stream one payload file into `dst`, verifying its byte size and hash.
[[nodiscard]] Status upload_words(compute::Stream &stream, const luisa::filesystem::path &path,
                                  compute::Buffer<uint32_t> &dst, uint64_t bytes,
                                  uint64_t expect_hash, uint64_t chunk, uint64_t &hash_out) {
    if (!dst || dst.size() * 4u != bytes) {
        return invalid("pack payload size does not match the allocated buffer", luisa::to_string(path));
    }
    std::ifstream in;
    if (auto s = open_in(path, in); !s.ok()) { return s; }
    luisa::vector<uint32_t> staging(static_cast<size_t>(std::max<uint64_t>(1u, chunk / 4u)));
    HashWriter hw{nullptr};
    uint64_t done = 0;
    const uint64_t words = bytes / 4u;
    while (done < words) {
        const uint64_t n = std::min<uint64_t>(staging.size(), words - done);
        const auto span = luisa::span<std::byte>{reinterpret_cast<std::byte *>(staging.data()),
                                                 static_cast<size_t>(n * 4u)};
        if (auto s = read_exact(in, span, path); !s.ok()) { return s; }
        hw.write(span);
        stream << dst.view(static_cast<size_t>(done), static_cast<size_t>(n))
                      .copy_from(luisa::span<const uint32_t>{staging.data(), static_cast<size_t>(n)});
        done += n;
    }
    hash_out = hw.hash;
    if (hw.bytes != bytes) {
        return io_error("pack payload byte count mismatch", luisa::to_string(path));
    }
    if (hw.hash != expect_hash) {
        return Status::error(StatusCode::unvalidated,
                             "pack payload hash mismatch (stale or corrupt pack)",
                             luisa::to_string(path));
    }
    return Status::success();
}

[[nodiscard]] Status upload_scales(compute::Stream &stream, const luisa::filesystem::path &path,
                                   compute::Buffer<float> &dst, uint64_t bytes, uint64_t expect_hash,
                                   uint64_t chunk) {
    if (!dst || dst.size() * 4u != bytes) {
        return invalid("pack scale size does not match the allocated buffer", luisa::to_string(path));
    }
    std::ifstream in;
    if (auto s = open_in(path, in); !s.ok()) { return s; }
    luisa::vector<float> staging(static_cast<size_t>(std::max<uint64_t>(1u, chunk / 4u)));
    HashWriter hw{nullptr};
    uint64_t done = 0;
    const uint64_t count = bytes / 4u;
    while (done < count) {
        const uint64_t n = std::min<uint64_t>(staging.size(), count - done);
        const auto span = luisa::span<std::byte>{reinterpret_cast<std::byte *>(staging.data()),
                                                 static_cast<size_t>(n * 4u)};
        if (auto s = read_exact(in, span, path); !s.ok()) { return s; }
        hw.write(span);
        stream << dst.view(static_cast<size_t>(done), static_cast<size_t>(n))
                      .copy_from(luisa::span<const float>{staging.data(), static_cast<size_t>(n)});
        done += n;
    }
    if (hw.hash != expect_hash) {
        return Status::error(StatusCode::unvalidated,
                             "pack scale hash mismatch (stale or corrupt pack)",
                             luisa::to_string(path));
    }
    return Status::success();
}

}// namespace

[[nodiscard]] Result<QwenDeviceWeights> upload_w4_pack(compute::Device &device, compute::Stream &stream,
                                                       const luisa::filesystem::path &dir,
                                                       uint64_t chunk_bytes,
                                                       QwenWeightUploadReport *report) {
    auto manifest = read_w4_pack_manifest(dir);
    if (!manifest.ok()) { return Result<QwenDeviceWeights>::failure(std::move(manifest.status)); }
    QwenWeightUploadReport local{};
    auto &rep = report != nullptr ? *report : local;
    rep = QwenWeightUploadReport{};
    const uint64_t chunk = clamp_chunk(chunk_bytes);
    SyncGuard guard{stream};
    luisa::vector<QwenDeviceTensor> tensors;
    tensors.reserve(manifest.value->tensors.size() * 2u);
    uint64_t index = 0;
    for (const auto &rec : manifest.value->tensors) {
        if (rec.storage == WeightStorage::f32_dense) {
            const uint64_t elements = rec.k;
            if (rec.f32_bytes != elements * 4u) {
                return Result<QwenDeviceWeights>::failure(
                    invalid("dense payload size mismatch", rec.name));
            }
            QwenDeviceTensor t;
            t.role = rec.role;
            t.layer = rec.layer;
            t.name = rec.name;
            t.storage = WeightStorage::f32_dense;
            t.dims = {rec.k};
            t.element_count = elements;
            t.f32 = device.create_buffer<float>(static_cast<size_t>(elements));
            if (!t.f32) {
                return Result<QwenDeviceWeights>::failure(StatusCode::out_of_memory,
                                                          "device buffer allocation failed", rec.name);
            }
            t.f32.set_name(rec.name.c_str());
            luisa::vector<float> staging(static_cast<size_t>(std::max<uint64_t>(1u, chunk / 4u)));
            std::ifstream in;
            if (auto s = open_in(dir / shard_name(index, ".f32.bin"), in); !s.ok()) {
                return Result<QwenDeviceWeights>::failure(std::move(s));
            }
            HashWriter hw{nullptr};
            uint64_t done = 0;
            while (done < elements) {
                const uint64_t n = std::min<uint64_t>(staging.size(), elements - done);
                const auto span = luisa::span<std::byte>{reinterpret_cast<std::byte *>(staging.data()),
                                                         static_cast<size_t>(n * 4u)};
                if (auto s = read_exact(in, span, dir); !s.ok()) {
                    return Result<QwenDeviceWeights>::failure(std::move(s));
                }
                hw.write(span);
                stream << t.f32.view(static_cast<size_t>(done), static_cast<size_t>(n))
                              .copy_from(luisa::span<const float>{staging.data(), static_cast<size_t>(n)});
                done += n;
            }
            if (hw.hash != rec.f32_hash) {
                return Result<QwenDeviceWeights>::failure(Status::error(
                    StatusCode::unvalidated, "dense payload hash mismatch", rec.name));
            }
            rep.f32_count++;
            rep.device_bytes += elements * 4u;
            tensors.push_back(std::move(t));
        } else {
            const uint64_t elements = rec.k * rec.n;
            const uint64_t words = elements / kW4CodesPerWord;
            const uint64_t scales = rec.n * (rec.k / rec.group);
            if (rec.w4_bytes != words * 4u || rec.w4s_bytes != scales * 4u) {
                return Result<QwenDeviceWeights>::failure(
                    invalid("w4 payload size does not match the manifest geometry", rec.name));
            }
            QwenDeviceTensor t;
            t.role = rec.role;
            t.layer = rec.layer;
            t.name = rec.name;
            t.storage = WeightStorage::w4_packed;
            t.dims = {rec.k, rec.n};
            t.element_count = elements;
            t.group = rec.group;
            t.groups_per_row = rec.k / rec.group;
            t.packed = device.create_buffer<uint32_t>(static_cast<size_t>(words));
            t.scales = device.create_buffer<float>(static_cast<size_t>(scales));
            if (!t.packed || !t.scales) {
                return Result<QwenDeviceWeights>::failure(StatusCode::out_of_memory,
                                                          "device buffer allocation failed", rec.name);
            }
            t.packed.set_name(rec.name.c_str());
            t.scales.set_name(rec.name.c_str());
            uint64_t hash = 0;
            if (auto s = upload_words(stream, dir / shard_name(index, ".w4.bin"), t.packed,
                                      rec.w4_bytes, rec.w4_hash, chunk, hash);
                !s.ok()) {
                return Result<QwenDeviceWeights>::failure(std::move(s));
            }
            if (auto s = upload_scales(stream, dir / shard_name(index, ".w4s.bin"), t.scales,
                                       rec.w4s_bytes, rec.w4s_hash, chunk);
                !s.ok()) {
                return Result<QwenDeviceWeights>::failure(std::move(s));
            }
            rep.w4_count++;
            rep.device_bytes += rec.w4_bytes + rec.w4s_bytes;
            // The FP8 payload becomes a SEPARATE tensor named "<name>.fp8": the
            // engine reads it for prefill and for the LM head (plan option (c)).
            if (rec.fp8_bytes != 0u || rec.fp8s_bytes != 0u) {
                const uint64_t fp8_words = (elements + 3u) / 4u;
                const uint64_t fp8_scales = rec.n * (rec.k / kFp8GroupSize);
                if (rec.fp8_bytes != fp8_words * 4u || rec.fp8s_bytes != fp8_scales * 4u) {
                    return Result<QwenDeviceWeights>::failure(
                        invalid("fp8 payload size does not match the manifest geometry", rec.name));
                }
                QwenDeviceTensor p;
                p.role = rec.role;
                p.layer = rec.layer;
                p.name = luisa::string{rec.name} + ".fp8";
                p.storage = WeightStorage::fp8_packed;
                p.dims = {rec.k, rec.n};
                p.element_count = elements;
                p.group = kFp8GroupSize;
                p.groups_per_row = rec.k / kFp8GroupSize;
                p.packed = device.create_buffer<uint32_t>(static_cast<size_t>(fp8_words));
                p.scales = device.create_buffer<float>(static_cast<size_t>(fp8_scales));
                if (!p.packed || !p.scales) {
                    return Result<QwenDeviceWeights>::failure(StatusCode::out_of_memory,
                                                              "device buffer allocation failed", p.name);
                }
                p.packed.set_name(p.name.c_str());
                p.scales.set_name(p.name.c_str());
                if (auto s = upload_words(stream, dir / shard_name(index, ".fp8.bin"), p.packed,
                                          rec.fp8_bytes, rec.fp8_hash, chunk, hash);
                    !s.ok()) {
                    return Result<QwenDeviceWeights>::failure(std::move(s));
                }
                if (auto s = upload_scales(stream, dir / shard_name(index, ".fp8s.bin"), p.scales,
                                           rec.fp8s_bytes, rec.fp8s_hash, chunk);
                    !s.ok()) {
                    return Result<QwenDeviceWeights>::failure(std::move(s));
                }
                rep.fp8_count++;
                rep.device_bytes += rec.fp8_bytes + rec.fp8s_bytes;
                // Push the DECODE tensor first: role lookups return the first
                // match, and the engine's format detection reads it. The partner is
                // found by its "<name>.fp8" name, never by role.
                tensors.push_back(std::move(t));
                tensors.push_back(std::move(p));
            } else {
                tensors.push_back(std::move(t));
            }
        }
        rep.manifest_records++;
        index++;
    }
    guard.done = true;
    // The report must be internally consistent: tensor_count counts the DEVICE
    // tensors this upload created (one W4 record can create two of them), never
    // the manifest records it walked.
    rep.tensor_count = rep.f32_count + rep.bf16_count + rep.fp8_count + rep.q8_0_count +
                       rep.w4_count;
    rep.diagnostics.push_back(Diagnostic{
        DiagnosticSeverity::info, "w4_pack_loaded",
        luisa::string{"source="} + manifest.value->source_name + " tensors=" +
            luisa::string{std::to_string(rep.tensor_count).c_str()} + " (records=" +
            luisa::string{std::to_string(rep.manifest_records).c_str()} + ") w4=" +
            luisa::string{std::to_string(rep.w4_count).c_str()} + " fp8=" +
            luisa::string{std::to_string(rep.fp8_count).c_str()} + " f32=" +
            luisa::string{std::to_string(rep.f32_count).c_str()}});
    if (manifest.value->has_importance) {
        rep.diagnostics.push_back(Diagnostic{
            DiagnosticSeverity::info, "w4_pack_importance_aware",
            luisa::string{"importance=tokens:"} +
                luisa::string{std::to_string(manifest.value->importance_tokens).c_str()} +
                " source=" + qwen_pack_hash_string(manifest.value->importance_source_hash) +
                " payload=" + qwen_pack_hash_string(manifest.value->importance_hash)});
    }
    return Result<QwenDeviceWeights>::success(
        QwenDeviceWeights::from_device_tensors(std::move(tensors)));
}

}// namespace luisa::qwen
