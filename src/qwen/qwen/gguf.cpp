// Qwen3.5-4B shared engine layer -- GGUF v3 reader implementation.
// Strict fail-closed parsing; payload bytes are read on demand. No exceptions.
#include "qwen/gguf.h"

#include <algorithm>
#include <cstring>
#include <utility>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace luisa::qwen {
namespace {

[[nodiscard]] Result<uint64_t> file_size_of(std::FILE *file) {
#ifdef _WIN32
    const auto size = _filelengthi64(_fileno(file));
    if (size < 0) {
        return Result<uint64_t>::failure(StatusCode::io_error, "cannot query GGUF file size", "file");
    }
    return Result<uint64_t>::success(static_cast<uint64_t>(size));
#else
    const int fd = fileno(file);
    const auto size = lseek(fd, 0, SEEK_END);
    if (size < 0 || lseek(fd, 0, SEEK_SET) < 0) {
        return Result<uint64_t>::failure(StatusCode::io_error, "cannot query GGUF file size", "file");
    }
    return Result<uint64_t>::success(static_cast<uint64_t>(size));
#endif
}

[[nodiscard]] Status seek_to(std::FILE *file, uint64_t offset) {
#ifdef _WIN32
    const int rc = _fseeki64(file, static_cast<int64_t>(offset), SEEK_SET);
#else
    const int rc = fseeko(file, static_cast<off_t>(offset), SEEK_SET);
#endif
    if (rc != 0) {
        return Status::error(StatusCode::io_error, "cannot seek in GGUF file", "file");
    }
    return Status::success();
}

// Sequential cursor over the header region with checked arithmetic. Every
// read is bounds-checked against the real file size before touching IO.
struct Cursor {
    std::FILE *file{nullptr};
    uint64_t file_size{0};
    uint64_t pos{0};

    [[nodiscard]] Status read(luisa::span<std::byte> out) {
        uint64_t end = 0;
        if (!checked_add(pos, out.size(), end) || end > file_size) {
            return Status::error(StatusCode::io_error, "GGUF header truncated", "file");
        }
        if (out.empty()) { return Status::success(); }
        if (std::fread(out.data(), 1u, out.size(), file) != out.size()) {
            return Status::error(StatusCode::io_error, "GGUF header read failed", "file");
        }
        pos = end;
        return Status::success();
    }
    template<typename T>
    [[nodiscard]] Result<T> scalar() {
        std::byte buf[sizeof(T)];
        if (auto s = read(buf); !s.ok()) { return Result<T>::failure(std::move(s)); }
        T v;
        std::memcpy(&v, buf, sizeof(T));
        return Result<T>::success(v);
    }
    [[nodiscard]] Result<luisa::string> string(uint64_t max_bytes) {
        auto len = scalar<uint64_t>();
        if (!len.ok()) { return Result<luisa::string>::failure(std::move(len.status)); }
        if (*len.value > max_bytes) {
            return Result<luisa::string>::failure(StatusCode::io_error,
                                                  "GGUF string exceeds size limit", "string");
        }
        uint64_t end = 0;
        if (!checked_add(pos, *len.value, end) || end > file_size) {
            return Result<luisa::string>::failure(StatusCode::io_error,
                                                  "GGUF string truncated", "string");
        }
        luisa::string s;
        s.resize(static_cast<size_t>(*len.value));
        if (*len.value != 0) {
            if (auto st = read(luisa::span<std::byte>{reinterpret_cast<std::byte *>(s.data()), s.size()});
                !st.ok()) {
                return Result<luisa::string>::failure(std::move(st));
            }
        }
        return Result<luisa::string>::success(std::move(s));
    }
};

[[nodiscard]] bool meta_type_valid(uint32_t t) noexcept { return t <= 12u; }

[[nodiscard]] Result<GgufMetaValue> parse_meta_value(Cursor &cur, uint32_t type, uint32_t depth);

[[nodiscard]] Result<GgufMetaValue> parse_meta_array(Cursor &cur, uint32_t depth) {
    if (depth >= kGgufMaxArrayDepth) {
        return Result<GgufMetaValue>::failure(StatusCode::io_error,
                                              "GGUF metadata array nesting too deep", "array");
    }
    auto elem_type = cur.scalar<uint32_t>();
    if (!elem_type.ok()) { return Result<GgufMetaValue>::failure(std::move(elem_type.status)); }
    if (!meta_type_valid(*elem_type.value)) {
        return Result<GgufMetaValue>::failure(StatusCode::io_error,
                                              "GGUF metadata array has invalid element type", "array");
    }
    auto count = cur.scalar<uint64_t>();
    if (!count.ok()) { return Result<GgufMetaValue>::failure(std::move(count.status)); }
    if (*count.value > kGgufMaxArrayElements) {
        return Result<GgufMetaValue>::failure(StatusCode::io_error,
                                              "GGUF metadata array exceeds element limit", "array");
    }
    GgufMetaValue v;
    v.type = GgufMetaType::array;
    v.element_type = static_cast<GgufMetaType>(*elem_type.value);
    v.array.reserve(static_cast<size_t>(*count.value));
    for (uint64_t i = 0; i < *count.value; ++i) {
        auto elem = parse_meta_value(cur, *elem_type.value, depth + 1u);
        if (!elem.ok()) { return Result<GgufMetaValue>::failure(std::move(elem.status)); }
        v.array.emplace_back(std::move(*elem.value));
    }
    return Result<GgufMetaValue>::success(std::move(v));
}

[[nodiscard]] Result<GgufMetaValue> parse_meta_value(Cursor &cur, uint32_t type, uint32_t depth) {
    if (!meta_type_valid(type)) {
        return Result<GgufMetaValue>::failure(StatusCode::io_error,
                                              "GGUF metadata value has invalid type", "value");
    }
    GgufMetaValue v;
    v.type = static_cast<GgufMetaType>(type);
    switch (static_cast<GgufMetaType>(type)) {
        case GgufMetaType::u8: {
            auto r = cur.scalar<uint8_t>();
            if (!r.ok()) { return Result<GgufMetaValue>::failure(std::move(r.status)); }
            v.u64 = *r.value;
            break;
        }
        case GgufMetaType::i8: {
            auto r = cur.scalar<int8_t>();
            if (!r.ok()) { return Result<GgufMetaValue>::failure(std::move(r.status)); }
            v.i64 = *r.value;
            break;
        }
        case GgufMetaType::u16: {
            auto r = cur.scalar<uint16_t>();
            if (!r.ok()) { return Result<GgufMetaValue>::failure(std::move(r.status)); }
            v.u64 = *r.value;
            break;
        }
        case GgufMetaType::i16: {
            auto r = cur.scalar<int16_t>();
            if (!r.ok()) { return Result<GgufMetaValue>::failure(std::move(r.status)); }
            v.i64 = *r.value;
            break;
        }
        case GgufMetaType::u32: {
            auto r = cur.scalar<uint32_t>();
            if (!r.ok()) { return Result<GgufMetaValue>::failure(std::move(r.status)); }
            v.u64 = *r.value;
            break;
        }
        case GgufMetaType::i32: {
            auto r = cur.scalar<int32_t>();
            if (!r.ok()) { return Result<GgufMetaValue>::failure(std::move(r.status)); }
            v.i64 = *r.value;
            break;
        }
        case GgufMetaType::f32: {
            auto r = cur.scalar<uint32_t>();
            if (!r.ok()) { return Result<GgufMetaValue>::failure(std::move(r.status)); }
            float f;
            const uint32_t bits = *r.value;
            std::memcpy(&f, &bits, sizeof(f));
            v.f64 = static_cast<double>(f);
            break;
        }
        case GgufMetaType::boolean: {
            auto r = cur.scalar<uint8_t>();
            if (!r.ok()) { return Result<GgufMetaValue>::failure(std::move(r.status)); }
            if (*r.value > 1u) {
                return Result<GgufMetaValue>::failure(StatusCode::io_error,
                                                      "GGUF bool metadata is not 0/1", "bool");
            }
            v.u64 = *r.value;
            break;
        }
        case GgufMetaType::string: {
            auto r = cur.string(kGgufMaxStringBytes);
            if (!r.ok()) { return Result<GgufMetaValue>::failure(std::move(r.status)); }
            v.str = std::move(*r.value);
            break;
        }
        case GgufMetaType::array: return parse_meta_array(cur, depth);
        case GgufMetaType::u64: {
            auto r = cur.scalar<uint64_t>();
            if (!r.ok()) { return Result<GgufMetaValue>::failure(std::move(r.status)); }
            v.u64 = *r.value;
            break;
        }
        case GgufMetaType::i64: {
            auto r = cur.scalar<int64_t>();
            if (!r.ok()) { return Result<GgufMetaValue>::failure(std::move(r.status)); }
            v.i64 = *r.value;
            break;
        }
        case GgufMetaType::f64: {
            auto r = cur.scalar<double>();
            if (!r.ok()) { return Result<GgufMetaValue>::failure(std::move(r.status)); }
            v.f64 = *r.value;
            break;
        }
    }
    return Result<GgufMetaValue>::success(std::move(v));
}

// Byte size of one tensor payload, checked. For Q8_0 the element count must
// be a multiple of the block size and the payload is blocks * 34 bytes.
[[nodiscard]] Result<uint64_t> tensor_payload_size(uint32_t type, uint64_t elements) {
    switch (type) {
        case static_cast<uint32_t>(GgufTensorType::f32):
        case static_cast<uint32_t>(GgufTensorType::bf16): {
            const uint64_t unit = type == static_cast<uint32_t>(GgufTensorType::f32) ? 4u : 2u;
            uint64_t bytes = 0;
            if (!checked_mul(elements, unit, bytes)) {
                return Result<uint64_t>::failure(StatusCode::io_error,
                                                 "GGUF tensor payload size overflow", "tensor");
            }
            return Result<uint64_t>::success(bytes);
        }
        case static_cast<uint32_t>(GgufTensorType::q8_0): {
            if (elements % kQ8_0BlockSize != 0u) {
                return Result<uint64_t>::failure(StatusCode::io_error,
                                                 "Q8_0 tensor element count is not a multiple of 32",
                                                 "tensor");
            }
            uint64_t bytes = 0;
            if (!checked_mul(elements / kQ8_0BlockSize, kQ8_0BlockBytes, bytes)) {
                return Result<uint64_t>::failure(StatusCode::io_error,
                                                 "GGUF tensor payload size overflow", "tensor");
            }
            return Result<uint64_t>::success(bytes);
        }
        default:
            return Result<uint64_t>::failure(StatusCode::unsupported,
                                             "unsupported GGUF tensor type", "tensor");
    }
}

}// namespace

[[nodiscard]] luisa::string_view gguf_tensor_type_name(uint32_t type) noexcept {
    switch (static_cast<GgufTensorType>(type)) {
        case GgufTensorType::f32: return "F32";
        case GgufTensorType::f16: return "F16";
        case GgufTensorType::q4_0: return "Q4_0";
        case GgufTensorType::q4_1: return "Q4_1";
        case GgufTensorType::q5_0: return "Q5_0";
        case GgufTensorType::q5_1: return "Q5_1";
        case GgufTensorType::q8_0: return "Q8_0";
        case GgufTensorType::q8_1: return "Q8_1";
        case GgufTensorType::q2_k: return "Q2_K";
        case GgufTensorType::q3_k: return "Q3_K";
        case GgufTensorType::q4_k: return "Q4_K";
        case GgufTensorType::q5_k: return "Q5_K";
        case GgufTensorType::q6_k: return "Q6_K";
        case GgufTensorType::q8_k: return "Q8_K";
        case GgufTensorType::i8: return "I8";
        case GgufTensorType::i16: return "I16";
        case GgufTensorType::i32: return "I32";
        case GgufTensorType::i64: return "I64";
        case GgufTensorType::f64: return "F64";
        case GgufTensorType::bf16: return "BF16";
        default: return "unknown";
    }
}

[[nodiscard]] bool gguf_tensor_type_supported(uint32_t type) noexcept {
    return type == static_cast<uint32_t>(GgufTensorType::f32) ||
           type == static_cast<uint32_t>(GgufTensorType::bf16) ||
           type == static_cast<uint32_t>(GgufTensorType::q8_0);
}

void GgufReaderDeleter::operator()(GgufReader *reader) const noexcept { delete reader; }

GgufReader::~GgufReader() noexcept {
    if (_file != nullptr) { std::fclose(_file); }
}

[[nodiscard]] Status GgufReader::read_exact(uint64_t offset, luisa::span<std::byte> out) {
    uint64_t end = 0;
    if (!checked_add(offset, out.size(), end) || end > _file_size) {
        return Status::error(StatusCode::io_error, "GGUF payload read out of bounds", "file");
    }
    if (out.empty()) { return Status::success(); }
    if (auto s = seek_to(_file, offset); !s.ok()) { return s; }
    if (std::fread(out.data(), 1u, out.size(), _file) != out.size()) {
        return Status::error(StatusCode::io_error, "GGUF payload read failed", "file");
    }
    return Status::success();
}

[[nodiscard]] Result<luisa::unique_ptr<GgufReader, GgufReaderDeleter>>
GgufReader::open(const luisa::filesystem::path &path) {
    using ReaderPtr = luisa::unique_ptr<GgufReader, GgufReaderDeleter>;
    std::FILE *file = nullptr;
#ifdef _WIN32
    file = _wfopen(path.c_str(), L"rb");
#else
    file = std::fopen(path.c_str(), "rb");
#endif
    if (file == nullptr) {
        return Result<ReaderPtr>::failure(StatusCode::io_error, "cannot open GGUF file",
                                          luisa::string{path.string()});
    }
    auto size = file_size_of(file);
    if (!size.ok()) {
        std::fclose(file);
        return Result<ReaderPtr>::failure(std::move(size.status));
    }
    ReaderPtr reader{new GgufReader{}};
    reader->_file = file;
    reader->_file_size = *size.value;

    Cursor cur{file, *size.value, 0};
    // Magic "GGUF" + version.
    auto magic = cur.scalar<uint32_t>();
    if (!magic.ok()) { return Result<ReaderPtr>::failure(std::move(magic.status)); }
    if (*magic.value != 0x46554747u) {
        return Result<ReaderPtr>::failure(StatusCode::io_error, "not a GGUF file (bad magic)",
                                          luisa::string{path.string()});
    }
    auto version = cur.scalar<uint32_t>();
    if (!version.ok()) { return Result<ReaderPtr>::failure(std::move(version.status)); }
    if (*version.value != 3u) {
        return Result<ReaderPtr>::failure(StatusCode::unsupported,
                                          "only GGUF v3 is supported", luisa::string{path.string()});
    }
    reader->_version = *version.value;
    auto tensor_count = cur.scalar<uint64_t>();
    if (!tensor_count.ok()) { return Result<ReaderPtr>::failure(std::move(tensor_count.status)); }
    if (*tensor_count.value == 0u || *tensor_count.value > kGgufMaxTensorCount) {
        return Result<ReaderPtr>::failure(StatusCode::io_error,
                                          "GGUF tensor count out of range", "tensor_count");
    }
    auto kv_count = cur.scalar<uint64_t>();
    if (!kv_count.ok()) { return Result<ReaderPtr>::failure(std::move(kv_count.status)); }
    if (*kv_count.value > kGgufMaxArrayElements) {
        return Result<ReaderPtr>::failure(StatusCode::io_error,
                                          "GGUF metadata count out of range", "kv_count");
    }
    // Metadata key/value pairs.
    for (uint64_t i = 0; i < *kv_count.value; ++i) {
        auto key = cur.string(kGgufMaxStringBytes);
        if (!key.ok()) { return Result<ReaderPtr>::failure(std::move(key.status)); }
        if ((*key.value).empty()) {
            return Result<ReaderPtr>::failure(StatusCode::io_error,
                                              "GGUF metadata key is empty", "key");
        }
        if (reader->_metadata_index.contains(*key.value)) {
            return Result<ReaderPtr>::failure(StatusCode::io_error,
                                              "duplicate GGUF metadata key", *key.value);
        }
        if (cur.pos > kGgufMaxMetadataBytes) {
            return Result<ReaderPtr>::failure(StatusCode::io_error,
                                              "GGUF metadata exceeds size limit", "metadata");
        }
        auto type = cur.scalar<uint32_t>();
        if (!type.ok()) { return Result<ReaderPtr>::failure(std::move(type.status)); }
        auto value = parse_meta_value(cur, *type.value, 0u);
        if (!value.ok()) { return Result<ReaderPtr>::failure(std::move(value.status)); }
        reader->_metadata_index.emplace(*key.value, reader->_metadata_entries.size());
        reader->_metadata_entries.emplace_back(std::move(*key.value), std::move(*value.value));
    }
    if (cur.pos > kGgufMaxMetadataBytes) {
        return Result<ReaderPtr>::failure(StatusCode::io_error,
                                          "GGUF metadata exceeds size limit", "metadata");
    }
    // Alignment override (rare); default 32.
    if (auto *align = reader->metadata("general.alignment"); align != nullptr) {
        const auto a = reader->meta_u64("general.alignment");
        if (!a.has_value() || *a == 0u || (*a & (*a - 1u)) != 0u ||
            *a > 4096u) {
            return Result<ReaderPtr>::failure(StatusCode::io_error,
                                              "invalid general.alignment", "general.alignment");
        }
        reader->_alignment = *a;
    }
    // Tensor info table.
    reader->_tensors.reserve(static_cast<size_t>(*tensor_count.value));
    for (uint64_t i = 0; i < *tensor_count.value; ++i) {
        auto name = cur.string(kGgufMaxStringBytes);
        if (!name.ok()) { return Result<ReaderPtr>::failure(std::move(name.status)); }
        if ((*name.value).empty()) {
            return Result<ReaderPtr>::failure(StatusCode::io_error,
                                              "GGUF tensor name is empty", "tensor");
        }
        if (reader->_tensor_index.contains(*name.value)) {
            return Result<ReaderPtr>::failure(StatusCode::io_error,
                                              "duplicate GGUF tensor name", *name.value);
        }
        auto nd = cur.scalar<uint32_t>();
        if (!nd.ok()) { return Result<ReaderPtr>::failure(std::move(nd.status)); }
        if (*nd.value == 0u || *nd.value > kGgufMaxRank) {
            return Result<ReaderPtr>::failure(StatusCode::io_error,
                                              "GGUF tensor rank out of range", *name.value);
        }
        GgufTensorInfo info;
        info.name = std::move(*name.value);
        info.dims.resize(*nd.value);
        for (uint32_t d = 0; d < *nd.value; ++d) {
            auto dim = cur.scalar<uint64_t>();
            if (!dim.ok()) { return Result<ReaderPtr>::failure(std::move(dim.status)); }
            if (*dim.value == 0u) {
                return Result<ReaderPtr>::failure(StatusCode::io_error,
                                                  "GGUF tensor dimension is zero", info.name);
            }
            info.dims[d] = *dim.value;
        }
        if (!checked_element_count(info.dims, info.element_count)) {
            return Result<ReaderPtr>::failure(StatusCode::io_error,
                                              "GGUF tensor element count overflow", info.name);
        }
        auto type = cur.scalar<uint32_t>();
        if (!type.ok()) { return Result<ReaderPtr>::failure(std::move(type.status)); }
        if (!gguf_tensor_type_supported(*type.value)) {
            return Result<ReaderPtr>::failure(
                StatusCode::unsupported,
                luisa::string{"unsupported GGUF tensor type "} +
                    luisa::string{gguf_tensor_type_name(*type.value)},
                info.name);
        }
        info.type = *type.value;
        auto offset = cur.scalar<uint64_t>();
        if (!offset.ok()) { return Result<ReaderPtr>::failure(std::move(offset.status)); }
        auto payload = tensor_payload_size(info.type, info.element_count);
        if (!payload.ok()) { return Result<ReaderPtr>::failure(std::move(payload.status)); }
        info.data_size = *payload.value;
        // Payload offsets are relative to the (aligned) data section start,
        // which is only known after the whole info table has been read.
        info.data_offset = *offset.value;// provisional: relative offset
        reader->_tensor_index.emplace(info.name, reader->_tensors.size());
        reader->_tensors.emplace_back(std::move(info));
    }
    // Data section starts at the next alignment boundary after the header.
    uint64_t data_start = 0;
    if (!checked_align_up(cur.pos, reader->_alignment, data_start) || data_start > *size.value) {
        return Result<ReaderPtr>::failure(StatusCode::io_error,
                                          "GGUF data section start out of bounds", "alignment");
    }
    reader->_data_start = data_start;
    // Resolve absolute offsets and validate ranges + non-overlap.
    for (auto &t : reader->_tensors) {
        uint64_t abs = 0, end = 0;
        if (!checked_add(data_start, t.data_offset, abs) ||
            !checked_add(abs, t.data_size, end) || end > *size.value) {
            return Result<ReaderPtr>::failure(StatusCode::io_error,
                                              "GGUF tensor payload out of file bounds", t.name);
        }
        t.data_offset = abs;
    }
    {
        luisa::vector<size_t> order(reader->_tensors.size());
        for (size_t i = 0; i < order.size(); ++i) { order[i] = i; }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return reader->_tensors[a].data_offset < reader->_tensors[b].data_offset;
        });
        uint64_t prev_end = data_start;
        for (auto idx : order) {
            const auto &t = reader->_tensors[idx];
            if (t.data_offset < prev_end) {
                return Result<ReaderPtr>::failure(StatusCode::io_error,
                                                  "GGUF tensor payloads overlap", t.name);
            }
            prev_end = t.data_offset + t.data_size;
        }
    }
    return Result<ReaderPtr>::success(std::move(reader));
}

// ---------------------------------------------------------------------------
// Metadata accessors
// ---------------------------------------------------------------------------
[[nodiscard]] const GgufMetaValue *GgufReader::metadata(luisa::string_view key) const noexcept {
    const auto it = _metadata_index.find(key);
    if (it == _metadata_index.end()) { return nullptr; }
    return &_metadata_entries[it->second].second;
}

[[nodiscard]] luisa::optional<luisa::string_view> GgufReader::meta_string(luisa::string_view key) const noexcept {
    const auto *v = metadata(key);
    if (v == nullptr || v->type != GgufMetaType::string) { return luisa::nullopt; }
    return luisa::string_view{v->str};
}

[[nodiscard]] luisa::optional<uint64_t> GgufReader::meta_u64(luisa::string_view key) const noexcept {
    const auto *v = metadata(key);
    if (v == nullptr) { return luisa::nullopt; }
    switch (v->type) {
        case GgufMetaType::u8:
        case GgufMetaType::u16:
        case GgufMetaType::u32:
        case GgufMetaType::u64:
        case GgufMetaType::boolean: return v->u64;
        case GgufMetaType::i8:
        case GgufMetaType::i16:
        case GgufMetaType::i32:
        case GgufMetaType::i64:
            if (v->i64 < 0) { return luisa::nullopt; }
            return static_cast<uint64_t>(v->i64);
        default: return luisa::nullopt;
    }
}

[[nodiscard]] luisa::optional<int64_t> GgufReader::meta_i64(luisa::string_view key) const noexcept {
    const auto *v = metadata(key);
    if (v == nullptr) { return luisa::nullopt; }
    switch (v->type) {
        case GgufMetaType::i8:
        case GgufMetaType::i16:
        case GgufMetaType::i32:
        case GgufMetaType::i64: return v->i64;
        case GgufMetaType::u8:
        case GgufMetaType::u16:
        case GgufMetaType::u32:
        case GgufMetaType::boolean: return static_cast<int64_t>(v->u64);
        case GgufMetaType::u64:
            if (v->u64 > static_cast<uint64_t>(INT64_MAX)) { return luisa::nullopt; }
            return static_cast<int64_t>(v->u64);
        default: return luisa::nullopt;
    }
}

[[nodiscard]] luisa::optional<double> GgufReader::meta_f64(luisa::string_view key) const noexcept {
    const auto *v = metadata(key);
    if (v == nullptr) { return luisa::nullopt; }
    if (v->type == GgufMetaType::f32 || v->type == GgufMetaType::f64) { return v->f64; }
    return luisa::nullopt;
}

[[nodiscard]] luisa::optional<bool> GgufReader::meta_bool(luisa::string_view key) const noexcept {
    const auto *v = metadata(key);
    if (v == nullptr || v->type != GgufMetaType::boolean) { return luisa::nullopt; }
    return v->u64 != 0u;
}

[[nodiscard]] luisa::optional<luisa::span<const GgufMetaValue>>
GgufReader::meta_array(luisa::string_view key) const noexcept {
    const auto *v = metadata(key);
    if (v == nullptr || v->type != GgufMetaType::array) { return luisa::nullopt; }
    return luisa::span<const GgufMetaValue>{v->array};
}

[[nodiscard]] Result<luisa::vector<luisa::string>>
GgufReader::meta_string_array(luisa::string_view key) const {
    const auto *v = metadata(key);
    if (v == nullptr || v->type != GgufMetaType::array ||
        v->element_type != GgufMetaType::string) {
        return Result<luisa::vector<luisa::string>>::failure(
            StatusCode::not_found, "metadata key is not a string array", luisa::string{key});
    }
    luisa::vector<luisa::string> out;
    out.reserve(v->array.size());
    for (const auto &e : v->array) { out.emplace_back(e.str); }
    return Result<luisa::vector<luisa::string>>::success(std::move(out));
}

// ---------------------------------------------------------------------------
// Tensor access
// ---------------------------------------------------------------------------
[[nodiscard]] const GgufTensorInfo *GgufReader::find(luisa::string_view name) const noexcept {
    const auto it = _tensor_index.find(name);
    if (it == _tensor_index.end()) { return nullptr; }
    return &_tensors[it->second];
}

[[nodiscard]] Status GgufReader::read_tensor_bytes(const GgufTensorInfo &tensor,
                                                   luisa::span<std::byte> out) {
    if (static_cast<uint64_t>(out.size()) != tensor.data_size) {
        return Status::error(StatusCode::invalid_argument,
                             "GGUF tensor byte read size mismatch", tensor.name);
    }
    return read_exact(tensor.data_offset, out);
}

[[nodiscard]] Status GgufReader::read_tensor_bytes_at(const GgufTensorInfo &tensor, uint64_t offset,
                                                      luisa::span<std::byte> out) {
    uint64_t end = 0;
    if (!checked_add(offset, out.size(), end) || end > tensor.data_size) {
        return Status::error(StatusCode::invalid_argument,
                             "GGUF tensor byte sub-range out of bounds", tensor.name);
    }
    return read_exact(tensor.data_offset + offset, out);
}

[[nodiscard]] Status GgufReader::read_tensor_f32(const GgufTensorInfo &tensor, luisa::span<float> out) {
    if (static_cast<uint64_t>(out.size()) != tensor.element_count) {
        return Status::error(StatusCode::invalid_argument,
                             "GGUF tensor f32 read size mismatch", tensor.name);
    }
    if (tensor.type == static_cast<uint32_t>(GgufTensorType::f32)) {
        return read_exact(tensor.data_offset,
                          luisa::span<std::byte>{reinterpret_cast<std::byte *>(out.data()),
                                                 out.size_bytes()});
    }
    if (tensor.type == static_cast<uint32_t>(GgufTensorType::bf16)) {
        // Stream in bounded chunks so a huge tensor never needs a huge host
        // staging buffer.
        constexpr size_t kChunkElements = 1u << 20u;// 1 Mi elements = 2 MiB staging
        luisa::vector<uint16_t> staging;
        staging.resize(std::min<uint64_t>(kChunkElements, tensor.element_count));
        uint64_t done = 0;
        while (done < tensor.element_count) {
            const uint64_t n = std::min<uint64_t>(staging.size(), tensor.element_count - done);
            if (auto s = read_exact(tensor.data_offset + done * 2u,
                                    luisa::span<std::byte>{reinterpret_cast<std::byte *>(staging.data()),
                                                           static_cast<size_t>(n * 2u)});
                !s.ok()) {
                return s;
            }
            for (uint64_t i = 0; i < n; ++i) { out[done + i] = bf16_to_f32(staging[i]); }
            done += n;
        }
        return Status::success();
    }
    return Status::error(StatusCode::unsupported,
                         "tensor is not F32/BF16; use the explicit Q8_0 dequantizing read",
                         tensor.name);
}

[[nodiscard]] Status GgufReader::read_tensor_q8_0_f32(const GgufTensorInfo &tensor, luisa::span<float> out) {
    if (tensor.type != static_cast<uint32_t>(GgufTensorType::q8_0)) {
        return Status::error(StatusCode::unsupported, "tensor is not Q8_0", tensor.name);
    }
    if (static_cast<uint64_t>(out.size()) != tensor.element_count) {
        return Status::error(StatusCode::invalid_argument,
                             "GGUF tensor Q8_0 read size mismatch", tensor.name);
    }
    // One block at a time: 2-byte FP16 scale + 32 int8 codes.
    const uint64_t blocks = tensor.element_count / kQ8_0BlockSize;
    std::byte block[kQ8_0BlockBytes];
    for (uint64_t b = 0; b < blocks; ++b) {
        if (auto s = read_exact(tensor.data_offset + b * kQ8_0BlockBytes, block); !s.ok()) {
            return s;
        }
        uint16_t scale_bits;
        std::memcpy(&scale_bits, block, sizeof(scale_bits));
        // Explicit FP16 -> FP32 (bit-exact, no hardware rounding).
        const uint32_t sign = static_cast<uint32_t>(scale_bits & 0x8000u) << 16u;
        const uint32_t exp = (scale_bits >> 10u) & 0x1fu;
        const uint32_t mant = scale_bits & 0x03ffu;
        uint32_t bits;
        if (exp == 0u) {
            if (mant == 0u) {
                bits = sign;
            } else {
                uint32_t m = mant;
                int e = -14;
                while ((m & 0x0400u) == 0u) {
                    m <<= 1u;
                    --e;
                }
                m &= 0x03ffu;
                bits = sign | (static_cast<uint32_t>(e + 127) << 23u) | (m << 13u);
            }
        } else if (exp == 0x1fu) {
            bits = sign | 0x7f800000u | (mant << 13u);
        } else {
            bits = sign | ((exp + 112u) << 23u) | (mant << 13u);
        }
        float scale;
        std::memcpy(&scale, &bits, sizeof(scale));
        for (uint32_t i = 0; i < kQ8_0BlockSize; ++i) {
            const auto code = static_cast<int8_t>(block[2u + i]);
            out[b * kQ8_0BlockSize + i] = scale * static_cast<float>(code);
        }
    }
    return Status::success();
}

}// namespace luisa::qwen
