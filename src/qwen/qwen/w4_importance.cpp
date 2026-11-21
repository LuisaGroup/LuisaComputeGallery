// Qwen3.5-4B -- W4 importance calibration statistics: checked writer/reader.
// Host code only; nothing here touches a device or a GGUF reader.
#include "qwen/w4_importance.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>

#include "qwen/w4_pack.h"

namespace luisa::qwen {
namespace {

[[nodiscard]] Status io_error(luisa::string msg, luisa::string detail = {}) {
    return Status::error(StatusCode::io_error, std::move(msg), std::move(detail));
}
[[nodiscard]] Status invalid(luisa::string msg, luisa::string detail = {}) {
    return Status::error(StatusCode::invalid_argument, std::move(msg), std::move(detail));
}
[[nodiscard]] Status stale(luisa::string msg, luisa::string detail = {}) {
    return Status::error(StatusCode::unvalidated, std::move(msg), std::move(detail));
}

struct Field {
    luisa::string_view key;
    luisa::string_view value;
};

// Tokenize one line of `key=value` fields; the first token may be a bare tag.
[[nodiscard]] luisa::vector<Field> split_fields(luisa::string_view line) {
    luisa::vector<Field> fields;
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
    if (end == s.c_str() || *end != '\0' || !std::isfinite(v)) { return false; }
    out = static_cast<float>(v);
    return true;
}

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

[[nodiscard]] luisa::string fmt_f32(float v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
    return luisa::string{buf};
}
[[nodiscard]] luisa::string fmt_u64(uint64_t v) {
    return luisa::string{std::to_string(v).c_str()};
}

// Domain check shared by the writer and the reader: every statistic must be a
// finite, non-negative second moment and the tensor must carry activation energy.
[[nodiscard]] Status validate_values(luisa::string_view name, uint64_t k, uint64_t tokens,
                                     luisa::span<const float> values, float &mean_out,
                                     float &min_out, float &max_out) noexcept {
    if (k == 0u) { return invalid("importance tensor has K == 0", luisa::string{name}); }
    if (tokens == 0u) {
        return invalid("importance tensor was never exercised by the calibration run "
                       "(no activation rows); refusing to invent statistics",
                       luisa::string{name});
    }
    if (static_cast<uint64_t>(values.size()) != k) {
        return invalid("importance channel count does not match K",
                       luisa::string{name} + " k=" + fmt_u64(k) +
                           " got=" + fmt_u64(static_cast<uint64_t>(values.size())));
    }
    double sum = 0.0;
    float lo = std::numeric_limits<float>::infinity();
    float hi = 0.0f;
    for (uint64_t j = 0; j < k; ++j) {
        const float v = values[static_cast<size_t>(j)];
        if (!std::isfinite(v)) {
            return invalid("importance statistic is not finite (NaN/inf)", luisa::string{name});
        }
        if (v < 0.0f) {
            return invalid("importance statistic is negative", luisa::string{name});
        }
        sum += static_cast<double>(v);
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    const double mean = sum / static_cast<double>(k);
    if (!(mean > 0.0) || !std::isfinite(mean)) {
        return invalid("importance statistics are all zero (no activation energy); "
                       "the calibration run captured nothing usable",
                       luisa::string{name});
    }
    mean_out = static_cast<float>(mean);
    min_out = lo;
    max_out = hi;
    return Status::success();
}

}// namespace

luisa::span<const float> W4Importance::channels(luisa::string_view name, uint64_t k) const noexcept {
    uint64_t offset = 0;
    for (const auto &t : tensors) {
        if (t.name == name) {
            if (t.k != k) { return {}; }
            return luisa::span<const float>{values.data() + offset, static_cast<size_t>(t.k)};
        }
        offset += t.k;
    }
    return {};
}

[[nodiscard]] Status W4Importance::status_for(luisa::string_view name, uint64_t k) const noexcept {
    uint64_t offset = 0;
    for (const auto &t : tensors) {
        if (t.name == name) {
            if (t.k != k) {
                return stale("importance statistics were collected for a different K "
                             "(the pack geometry does not match the calibration)",
                             luisa::string{name} + " stats_k=" + fmt_u64(t.k) + " pack_k=" + fmt_u64(k));
            }
            (void)offset;
            return Status::success();
        }
        offset += t.k;
    }
    return Status::error(StatusCode::not_found,
                         "no importance statistics for this tensor (the calibration run "
                         "never calibrated it); refusing to pack it importance-aware",
                         luisa::string{name});
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------
[[nodiscard]] Status write_w4_importance(const luisa::filesystem::path &dir,
                                         const W4ImportanceSource &source,
                                         const QwenCalibrationReport &report) noexcept {
    if (source.name.empty()) { return invalid("importance source name is empty"); }
    if (source.size == 0u) { return invalid("importance source size is 0"); }
    if (source.prompts == 0u || source.prompt_len == 0u) {
        return invalid("importance calibration prompt set is empty (prompts/prompt_len must be > 0)");
    }
    if (report.observed.empty()) { return invalid("importance calibration observed no tensors"); }
    if (report.projections == 0u) {
        return invalid("importance calibration recorded no projection dispatch");
    }
    std::error_code ec;
    luisa::filesystem::create_directories(dir, ec);
    if (ec) { return io_error("cannot create the importance directory", luisa::to_string(dir)); }

    // Build the records (validating every tensor) and the concatenated payload.
    W4Importance stats;
    stats.version = kW4ImportanceVersion;
    stats.source = source;
    uint64_t total_bytes = 0;
    for (const auto &t : report.observed) {
        W4ImportanceTensorRecord rec;
        rec.name = t.name;
        rec.k = t.k;
        rec.tokens = t.tokens;
        if (auto s = validate_values(rec.name, rec.k, rec.tokens,
                                     luisa::span<const float>{t.second_moment.data(),
                                                              t.second_moment.size()},
                                     rec.mean, rec.min, rec.max);
            !s.ok()) {
            return s;
        }
        for (const auto &prev : stats.tensors) {
            if (prev.name == rec.name) {
                return invalid("duplicate importance tensor record", rec.name);
            }
        }
        rec.bytes = rec.k * sizeof(float);
        rec.hash = qwen_fnv1a64(luisa::span<const std::byte>{
            reinterpret_cast<const std::byte *>(t.second_moment.data()),
            t.second_moment.size() * sizeof(float)});
        total_bytes += rec.bytes;
        stats.values.insert(stats.values.end(), t.second_moment.begin(), t.second_moment.end());
        stats.tensors.push_back(std::move(rec));
    }

    // Payload FIRST, manifest LAST: a directory without a manifest is incomplete.
    const auto payload_path = dir / kW4ImportancePayloadName;
    {
        std::ofstream out{payload_path, std::ios::binary | std::ios::trunc};
        if (!out.is_open()) {
            return io_error("cannot open the importance payload for writing",
                            luisa::to_string(payload_path));
        }
        out.write(reinterpret_cast<const char *>(stats.values.data()),
                  static_cast<std::streamsize>(stats.values.size() * sizeof(float)));
        if (!out) {
            return io_error("importance payload write failed", luisa::to_string(payload_path));
        }
    }

    luisa::string text;
    text.reserve(256u * (stats.tensors.size() + 4u));
    const auto add = [&text](luisa::string_view s) { text.append(s); };
    add("qwen-w4-importance ");
    add(fmt_u64(kW4ImportanceVersion));
    add("\n");
    add("source name=");
    add(source.name);
    add(" size=");
    add(fmt_u64(source.size));
    add(" hash=");
    add(qwen_pack_hash_string(source.hash));
    add(" hash_algorithm=fnv1a64\n");
    add("calibration prompts=");
    add(fmt_u64(source.prompts));
    add(" prompt_len=");
    add(fmt_u64(source.prompt_len));
    add(" decode_tokens=");
    add(fmt_u64(source.decode_tokens));
    add(" tokens=");
    add(fmt_u64(source.tokens));
    add(" statistic=mean_of_squares_per_input_channel precision=f32\n");
    for (const auto &t : stats.tensors) {
        add("tensor ");
        add(t.name);
        add(" k=");
        add(fmt_u64(t.k));
        add(" tokens=");
        add(fmt_u64(t.tokens));
        add(" mean=");
        add(fmt_f32(t.mean));
        add(" min=");
        add(fmt_f32(t.min));
        add(" max=");
        add(fmt_f32(t.max));
        add(" bytes=");
        add(fmt_u64(t.bytes));
        add(" hash=");
        add(qwen_pack_hash_string(t.hash));
        add("\n");
    }
    const auto manifest_path = dir / kW4ImportanceManifestName;
    {
        std::ofstream out{manifest_path, std::ios::binary | std::ios::trunc};
        if (!out.is_open()) {
            return io_error("cannot open the importance manifest for writing",
                            luisa::to_string(manifest_path));
        }
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!out) {
            return io_error("importance manifest write failed", luisa::to_string(manifest_path));
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------
[[nodiscard]] Result<W4Importance> read_w4_importance(const luisa::filesystem::path &dir) noexcept {
    const auto manifest_path = dir / kW4ImportanceManifestName;
    std::ifstream in{manifest_path, std::ios::binary};
    if (!in.is_open()) {
        return Result<W4Importance>::failure(
            Status::error(StatusCode::not_found, "importance manifest is missing",
                          luisa::to_string(manifest_path)));
    }
    W4Importance stats;
    bool have_header = false, have_source = false, have_calibration = false;
    luisa::string line;
    uint64_t line_no = 0;
    while (std::getline(in, line)) {
        line_no++;
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) { line.pop_back(); }
        auto fields = split_fields(luisa::string_view{line});
        if (fields.empty()) { continue; }
        if (!have_header) {
            if (fields[0].key.size() != 0u || fields[0].value != "qwen-w4-importance" ||
                fields.size() != 2u) {
                return Result<W4Importance>::failure(
                    invalid("not a qwen W4 importance manifest", luisa::to_string(manifest_path)));
            }
            uint64_t version = 0;
            if (!parse_u64(fields[1].value, version) || version != kW4ImportanceVersion) {
                return Result<W4Importance>::failure(invalid(
                    "unsupported W4 importance version (only " +
                        fmt_u64(kW4ImportanceVersion) + " is accepted)",
                    luisa::to_string(manifest_path)));
            }
            stats.version = static_cast<uint32_t>(version);
            have_header = true;
            continue;
        }
        if (fields[0].key.size() == 0u && fields[0].value == "source") {
            for (size_t i = 1u; i < fields.size(); ++i) {
                const auto &f = fields[i];
                if (f.key == "name") {
                    stats.source.name = luisa::string{f.value};
                } else if (f.key == "size") {
                    if (!parse_u64(f.value, stats.source.size)) {
                        return Result<W4Importance>::failure(invalid("malformed importance source size"));
                    }
                } else if (f.key == "hash") {
                    if (!parse_hex64(f.value, stats.source.hash)) {
                        return Result<W4Importance>::failure(invalid("malformed importance source hash"));
                    }
                }
            }
            have_source = true;
            continue;
        }
        if (fields[0].key.size() == 0u && fields[0].value == "calibration") {
            uint64_t u = 0;
            for (size_t i = 1u; i < fields.size(); ++i) {
                const auto &f = fields[i];
                if (f.key == "prompts") {
                    if (!parse_u64(f.value, u)) { return Result<W4Importance>::failure(invalid("malformed calibration prompts")); }
                    stats.source.prompts = static_cast<uint32_t>(u);
                } else if (f.key == "prompt_len") {
                    if (!parse_u64(f.value, u)) { return Result<W4Importance>::failure(invalid("malformed calibration prompt_len")); }
                    stats.source.prompt_len = static_cast<uint32_t>(u);
                } else if (f.key == "decode_tokens") {
                    if (!parse_u64(f.value, u)) { return Result<W4Importance>::failure(invalid("malformed calibration decode_tokens")); }
                    stats.source.decode_tokens = static_cast<uint32_t>(u);
                } else if (f.key == "tokens") {
                    if (!parse_u64(f.value, stats.source.tokens)) { return Result<W4Importance>::failure(invalid("malformed calibration tokens")); }
                } else if (f.key == "statistic") {
                    if (f.value != "mean_of_squares_per_input_channel") {
                        return Result<W4Importance>::failure(invalid(
                            "unsupported importance statistic (only the measured input-channel "
                            "second moment is accepted)", luisa::string{f.value}));
                    }
                } else if (f.key == "precision") {
                    if (f.value != "f32") {
                        return Result<W4Importance>::failure(invalid("unsupported importance precision"));
                    }
                }
            }
            have_calibration = true;
            continue;
        }
        if (fields[0].key.size() == 0u && fields[0].value == "tensor") {
            if (fields.size() < 2u) {
                return Result<W4Importance>::failure(invalid("importance tensor record without a name"));
            }
            W4ImportanceTensorRecord rec;
            rec.name = luisa::string{fields[1].value};
            for (size_t i = 2u; i < fields.size(); ++i) {
                const auto &f = fields[i];
                if (f.key == "k") {
                    if (!parse_u64(f.value, rec.k)) { return Result<W4Importance>::failure(invalid("malformed importance k", rec.name)); }
                } else if (f.key == "tokens") {
                    if (!parse_u64(f.value, rec.tokens)) { return Result<W4Importance>::failure(invalid("malformed importance tokens", rec.name)); }
                } else if (f.key == "mean") {
                    if (!parse_f32(f.value, rec.mean)) { return Result<W4Importance>::failure(invalid("malformed importance mean", rec.name)); }
                } else if (f.key == "min") {
                    if (!parse_f32(f.value, rec.min)) { return Result<W4Importance>::failure(invalid("malformed importance min", rec.name)); }
                } else if (f.key == "max") {
                    if (!parse_f32(f.value, rec.max)) { return Result<W4Importance>::failure(invalid("malformed importance max", rec.name)); }
                } else if (f.key == "bytes") {
                    if (!parse_u64(f.value, rec.bytes)) { return Result<W4Importance>::failure(invalid("malformed importance bytes", rec.name)); }
                } else if (f.key == "hash") {
                    if (!parse_hex64(f.value, rec.hash)) { return Result<W4Importance>::failure(invalid("malformed importance hash", rec.name)); }
                } else {
                    return Result<W4Importance>::failure(
                        invalid("unknown importance manifest field", luisa::string{f.key}));
                }
            }
            if (rec.k == 0u || rec.bytes != rec.k * sizeof(float)) {
                return Result<W4Importance>::failure(
                    invalid("importance tensor geometry is inconsistent", rec.name));
            }
            for (const auto &prev : stats.tensors) {
                if (prev.name == rec.name) {
                    return Result<W4Importance>::failure(
                        invalid("duplicate importance tensor record", rec.name));
                }
            }
            stats.tensors.push_back(std::move(rec));
            continue;
        }
        return Result<W4Importance>::failure(invalid(
            "unexpected line in the W4 importance manifest",
            luisa::string{"line "} + fmt_u64(line_no)));
    }
    if (!have_header || !have_source || !have_calibration || stats.tensors.empty()) {
        return Result<W4Importance>::failure(
            invalid("incomplete W4 importance manifest", luisa::to_string(manifest_path)));
    }
    if (stats.source.name.empty() || stats.source.size == 0u ||
        stats.source.prompts == 0u || stats.source.prompt_len == 0u || stats.source.tokens == 0u) {
        return Result<W4Importance>::failure(
            invalid("importance manifest records an incomplete calibration identity",
                    luisa::to_string(manifest_path)));
    }

    // Payload: exact concatenated size, then per-tensor hash + domain.
    uint64_t total_bytes = 0;
    for (const auto &t : stats.tensors) { total_bytes += t.bytes; }
    const auto payload_path = dir / kW4ImportancePayloadName;
    std::ifstream payload{payload_path, std::ios::binary};
    if (!payload.is_open()) {
        return Result<W4Importance>::failure(
            Status::error(StatusCode::not_found, "importance payload is missing",
                          luisa::to_string(payload_path)));
    }
    if (total_bytes % sizeof(float) != 0u) {
        return Result<W4Importance>::failure(
            invalid("importance payload size is not a whole number of F32 values",
                    luisa::to_string(payload_path)));
    }
    stats.values.resize(static_cast<size_t>(total_bytes / sizeof(float)));
    payload.read(reinterpret_cast<char *>(stats.values.data()),
                 static_cast<std::streamsize>(total_bytes));
    if (payload.gcount() != static_cast<std::streamsize>(total_bytes)) {
        return Result<W4Importance>::failure(
            io_error("importance payload is truncated", luisa::to_string(payload_path)));
    }
    {
        char extra = 0;
        payload.read(&extra, 1);
        if (payload.gcount() != 0) {
            return Result<W4Importance>::failure(
                io_error("importance payload is larger than the manifest declares",
                         luisa::to_string(payload_path)));
        }
    }
    uint64_t offset = 0;
    for (auto &t : stats.tensors) {
        const auto values = luisa::span<const float>{stats.values.data() + offset,
                                                     static_cast<size_t>(t.k)};
        const uint64_t hash = qwen_fnv1a64(luisa::span<const std::byte>{
            reinterpret_cast<const std::byte *>(values.data()),
            values.size() * sizeof(float)});
        if (hash != t.hash) {
            return Result<W4Importance>::failure(
                stale("importance payload hash mismatch (stale or corrupt calibration)", t.name));
        }
        float mean = 0.0f, lo = 0.0f, hi = 0.0f;
        if (auto s = validate_values(t.name, t.k, t.tokens, values, mean, lo, hi); !s.ok()) {
            return Result<W4Importance>::failure(std::move(s));
        }
        // The recorded summary must describe the payload it claims to describe.
        const auto close = [](float a, float b) {
            const double d = std::fabs(static_cast<double>(a) - static_cast<double>(b));
            return d <= 1e-6 * std::max(1.0, std::fabs(static_cast<double>(b)));
        };
        if (!close(mean, t.mean) || !close(lo, t.min) || !close(hi, t.max)) {
            return Result<W4Importance>::failure(stale(
                "importance summary does not match its payload (stale or edited manifest)", t.name));
        }
        offset += t.k;
    }
    return Result<W4Importance>::success(std::move(stats));
}

}// namespace luisa::qwen
