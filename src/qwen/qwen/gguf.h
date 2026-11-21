#pragma once
// Qwen3.5-4B shared engine layer -- checked host-side GGUF v3 reader.
// Host-only: no DSL/runtime device includes.
//
// Strict fail-closed parsing with bounds/overflow validation everywhere:
// magic/version, metadata value tree (depth-limited), tensor info table,
// duplicate names, alignment, payload ranges vs. the real file size. Data
// bytes are read ON DEMAND, one tensor (or one chunk of a tensor) at a time;
// opening a multi-GB GGUF never copies the whole file into host memory.
//
// Tensor dtypes: F32 (0) and BF16 (30) payloads can be read raw or decoded to
// FP32. Q8_0 (8) is explicitly DETECTED and its block geometry validated, and
// an explicit dequantizing read is provided -- Q8_0 is INT8 block
// quantization and is never silently treated as FP8. All other GGUF types
// are rejected at open time (fail closed).
//
// Metadata is kept in a typed tree; scalar/array/string accessors cover what
// the Qwen config and tokenizer metadata need (including the full
// tokenizer.ggml.tokens / token_type / merges arrays and chat template).

#include <cstdint>
#include <cstdio>

#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/variant.h>
#include <luisa/core/stl/vector.h>

#include "qwen/common.h"

namespace luisa::qwen {

// GGUF metadata value types (gguf spec v3).
enum class GgufMetaType : uint32_t {
    u8 = 0,
    i8 = 1,
    u16 = 2,
    i16 = 3,
    u32 = 4,
    i32 = 5,
    f32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    u64 = 10,
    i64 = 11,
    f64 = 12,
};

// GGUF tensor types. Only f32/bf16/q8_0 are accepted by this reader; the
// numeric values of the others are kept for diagnostics.
enum class GgufTensorType : uint32_t {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
};

[[nodiscard]] luisa::string_view gguf_tensor_type_name(uint32_t type) noexcept;
// True for the types this reader can open (F32, BF16, Q8_0).
[[nodiscard]] bool gguf_tensor_type_supported(uint32_t type) noexcept;

inline constexpr uint32_t kGgufMaxRank = 4u;
inline constexpr uint64_t kGgufMaxMetadataBytes = 256ull * 1024ull * 1024ull;
inline constexpr uint64_t kGgufMaxStringBytes = 64ull * 1024ull * 1024ull;
inline constexpr uint64_t kGgufMaxArrayElements = 16ull * 1024ull * 1024ull;
inline constexpr uint32_t kGgufMaxArrayDepth = 8u;
inline constexpr uint64_t kGgufMaxTensorCount = 1ull * 1024ull * 1024ull;
inline constexpr uint64_t kGgufDefaultAlignment = 32u;
// Q8_0 block geometry (2-byte FP16 scale + 32 signed int8 codes) lives in
// qwen/common.h next to the storage ABI: kQ8_0BlockSize / kQ8_0BlockBytes.

struct GgufMetaValue {
    GgufMetaType type{GgufMetaType::u8};
    // Scalar payload (bool stored as 0/1 in u64; f32/f64 bit patterns in
    // u64/f64 respectively). string holds GgufMetaType::string values.
    uint64_t u64{0};
    int64_t i64{0};
    double f64{0};
    luisa::string str;
    // Array payload; element_type is the (single) GGUF type of all elements.
    GgufMetaType element_type{GgufMetaType::u8};
    luisa::vector<GgufMetaValue> array;
};

struct GgufTensorInfo {
    luisa::string name;
    uint32_t type{0};// raw GgufTensorType value
    luisa::vector<uint64_t> dims;// ne order: matrix [K, N], conv [kernel, channels]
    uint64_t element_count{0};
    uint64_t data_offset{0};// absolute byte offset in the file
    uint64_t data_size{0};  // exact stored byte size (incl. quant block padding)
};

class GgufReader;
struct GgufReaderDeleter {
    void operator()(GgufReader *reader) const noexcept;
};

class GgufReader {
public:
    [[nodiscard]] static Result<luisa::unique_ptr<GgufReader, GgufReaderDeleter>>
    open(const luisa::filesystem::path &path);
    ~GgufReader() noexcept;
    GgufReader(const GgufReader &) = delete;
    GgufReader &operator=(const GgufReader &) = delete;

    [[nodiscard]] uint32_t version() const noexcept { return _version; }
    [[nodiscard]] uint64_t file_size() const noexcept { return _file_size; }
    [[nodiscard]] uint64_t alignment() const noexcept { return _alignment; }
    [[nodiscard]] uint64_t data_start() const noexcept { return _data_start; }

    // ---- metadata -------------------------------------------------------
    [[nodiscard]] const GgufMetaValue *metadata(luisa::string_view key) const noexcept;
    [[nodiscard]] luisa::optional<luisa::string_view> meta_string(luisa::string_view key) const noexcept;
    // Integer access accepts every integer metadata type (u8..u64/i8..i64)
    // and bool; out-of-range or non-integer values yield nullopt.
    [[nodiscard]] luisa::optional<uint64_t> meta_u64(luisa::string_view key) const noexcept;
    [[nodiscard]] luisa::optional<int64_t> meta_i64(luisa::string_view key) const noexcept;
    [[nodiscard]] luisa::optional<double> meta_f64(luisa::string_view key) const noexcept;// f32/f64
    [[nodiscard]] luisa::optional<bool> meta_bool(luisa::string_view key) const noexcept;
    // Array of strings (tokenizer tokens/merges, tags, ...).
    [[nodiscard]] luisa::optional<luisa::span<const GgufMetaValue>> meta_array(luisa::string_view key) const noexcept;
    // Convenience: copy a string array out as plain strings.
    [[nodiscard]] Result<luisa::vector<luisa::string>> meta_string_array(luisa::string_view key) const;
    // All metadata keys in file order (for diagnostics / tooling).
    [[nodiscard]] luisa::span<const std::pair<luisa::string, GgufMetaValue>> metadata_entries() const noexcept {
        return _metadata_entries;
    }

    // ---- tensors ----------------------------------------------------------
    [[nodiscard]] luisa::span<const GgufTensorInfo> tensors() const noexcept { return _tensors; }
    [[nodiscard]] const GgufTensorInfo *find(luisa::string_view name) const noexcept;

    // Raw stored bytes of one tensor (no conversion). out.size() must equal
    // tensor.data_size exactly.
    [[nodiscard]] Status read_tensor_bytes(const GgufTensorInfo &tensor, luisa::span<std::byte> out);
    // Raw byte sub-range of one tensor (streaming). [offset, offset+out.size())
    // must lie inside the tensor payload.
    [[nodiscard]] Status read_tensor_bytes_at(const GgufTensorInfo &tensor, uint64_t offset,
                                              luisa::span<std::byte> out);
    // Decode F32/BF16 tensors to FP32 (explicit conversion, no
    // reinterpretation). Q8_0 and other types are rejected here.
    [[nodiscard]] Status read_tensor_f32(const GgufTensorInfo &tensor, luisa::span<float> out);
    // Explicit Q8_0 dequantization: value = fp16(scale) * int8(code), blocks
    // of 32 along the flattened row-major element order. Fails for non-Q8_0.
    [[nodiscard]] Status read_tensor_q8_0_f32(const GgufTensorInfo &tensor, luisa::span<float> out);

private:
    GgufReader() noexcept = default;
    [[nodiscard]] Status read_exact(uint64_t offset, luisa::span<std::byte> out);

    std::FILE *_file{nullptr};
    uint64_t _file_size{0};
    uint32_t _version{0};
    uint64_t _alignment{kGgufDefaultAlignment};
    uint64_t _data_start{0};
    luisa::vector<std::pair<luisa::string, GgufMetaValue>> _metadata_entries;
    luisa::unordered_map<luisa::string, size_t> _metadata_index;
    luisa::vector<GgufTensorInfo> _tensors;
    luisa::unordered_map<luisa::string, size_t> _tensor_index;
};

}// namespace luisa::qwen
