#pragma once
/**
 * @file image_io.h
 * @brief Image writers shared by the gallery examples (stb_image_write).
 *
 * Every example that produces a picture writes it through this header, so the
 * output format follows the *file extension* of the requested path:
 *
 *     image_io::write("out.png",  pixels, w, h, 4);   // lossless
 *     image_io::write("out.jpg",  pixels, w, h, 4);   // quality 95, JPEG
 *     image_io::write("out.jpeg", pixels, w, h, 4);   // alias of .jpg
 *     image_io::write("out.bmp",  pixels, w, h, 4);   // uncompressed
 *     image_io::write("out.tga",  pixels, w, h, 4);   // uncompressed
 *
 * The examples expose the destination through `--capture <path>` (see
 * common/reference_compare.h and common/headless.h), which is what makes
 * `example_xxx dx --capture render.jpg` work headless.
 *
 * Unknown/absent extensions fall back to PNG. Missing parent directories are
 * created, because `--capture docs/img/x.png` should just work.
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <luisa/core/logging.h>

#include <stb/stb_image_write.h>

namespace luisa::gallery::image_io {

/// Output container chosen from the file extension.
enum class Format : uint32_t {
    png,
    jpeg,
    bmp,
    tga,
};

/// JPEG quality used for `.jpg`/`.jpeg` output (0-100).
inline constexpr int JPEG_QUALITY = 95;

[[nodiscard]] inline std::string lower_extension(std::string_view path) noexcept {
    auto dot = path.find_last_of('.');
    if (dot == std::string_view::npos) { return {}; }
    auto slash = path.find_last_of("/\\");
    if (slash != std::string_view::npos && slash > dot) { return {}; }
    std::string extension;
    extension.reserve(path.size() - dot);
    for (auto i = dot + 1u; i < path.size(); ++i) {
        auto c = path[i];
        if (c >= 'A' && c <= 'Z') { c = static_cast<char>(c - 'A' + 'a'); }
        extension.push_back(c);
    }
    return extension;
}

/// Format implied by the extension of `path`; PNG when it is unknown.
[[nodiscard]] inline Format format_of(std::string_view path) noexcept {
    auto extension = lower_extension(path);
    if (extension == "jpg" || extension == "jpeg") { return Format::jpeg; }
    if (extension == "bmp") { return Format::bmp; }
    if (extension == "tga") { return Format::tga; }
    return Format::png;
}

/// True for the three extensions the gallery documents: png, jpg, jpeg.
[[nodiscard]] inline bool is_documented_extension(std::string_view path) noexcept {
    auto extension = lower_extension(path);
    return extension.empty() || extension == "png" || extension == "jpg" ||
           extension == "jpeg" || extension == "bmp" || extension == "tga";
}

namespace detail {

inline void create_parent_directory(const std::filesystem::path &path) noexcept {
    std::error_code error;
    auto parent = path.parent_path();
    if (!parent.empty() && !std::filesystem::exists(parent, error)) {
        std::filesystem::create_directories(parent, error);
    }
}

}// namespace detail

/// Write `data` (width * height * channels bytes, 8 bit per channel) to `path`.
/// Returns false (and logs a warning) when stb rejected the data or the file
/// could not be opened; the caller turns that into its non-zero exit code.
[[nodiscard]] inline bool write(const std::filesystem::path &path,
                                const void *data,
                                uint32_t width, uint32_t height,
                                uint32_t channels) noexcept {
    if (path.empty() || data == nullptr || width == 0u || height == 0u ||
        channels == 0u || channels > 4u) {
        LUISA_WARNING("image_io::write: invalid image ({}x{}, {} channel(s), path '{}').",
                      width, height, channels, path.string());
        return false;
    }
    auto file = path.string();
    detail::create_parent_directory(path);
    auto w = static_cast<int>(width);
    auto h = static_cast<int>(height);
    auto c = static_cast<int>(channels);
    auto result = 0;
    switch (format_of(file)) {
        case Format::jpeg:
            result = stbi_write_jpg(file.c_str(), w, h, c, data, JPEG_QUALITY);
            break;
        case Format::bmp:
            result = stbi_write_bmp(file.c_str(), w, h, c, data);
            break;
        case Format::tga:
            result = stbi_write_tga(file.c_str(), w, h, c, data);
            break;
        case Format::png:
        default:
            result = stbi_write_png(file.c_str(), w, h, c, data, 0);
            break;
    }
    if (result == 0) {
        LUISA_WARNING("image_io::write: failed to write '{}'.", file);
        return false;
    }
    return true;
}

/// Drop-in replacement for `stbi_write_png`, except that the container comes
/// from the extension of `path` (png/jpg/jpeg/bmp/tga) instead of being fixed to
/// PNG. `stride` is the byte distance between two rows, 0 meaning
/// `width * channels`. Returns 1 on success and 0 on failure, exactly like the
/// stb writer, so the existing `if (write_image(...) == 0)` call sites keep
/// working.
[[nodiscard]] inline int write_image(const char *path,
                                     int width, int height, int channels,
                                     const void *data, int stride) noexcept {
    if (path == nullptr || data == nullptr || width <= 0 || height <= 0 ||
        channels <= 0 || channels > 4) {
        return 0;
    }
    auto path_string = std::string_view{path};
    auto format = format_of(path_string);
    auto packed_stride = width * channels;
    if (format == Format::png || stride == 0 || stride == packed_stride) {
        // stb's PNG writer honours the stride itself; the other writers want the
        // rows packed, which they are when the stride already matches.
        auto file = std::filesystem::path{std::string{path_string}};
        if (format == Format::png) {
            detail::create_parent_directory(file);
            return stbi_write_png(path, width, height, channels, data, stride);
        }
        return write(file, data, static_cast<uint32_t>(width),
                     static_cast<uint32_t>(height),
                     static_cast<uint32_t>(channels))
                   ? 1
                   : 0;
    }
    // Uncommon: a padded stride for a non-PNG container. Repack the rows.
    auto row_bytes = static_cast<size_t>(packed_stride);
    auto source = static_cast<const std::uint8_t *>(data);
    std::vector<std::uint8_t> packed(row_bytes * static_cast<size_t>(height));
    for (auto y = 0; y < height; ++y) {
        std::memcpy(packed.data() + static_cast<size_t>(y) * row_bytes,
                    source + static_cast<size_t>(y) * static_cast<size_t>(stride),
                    row_bytes);
    }
    return write(std::filesystem::path{std::string{path_string}}, packed.data(),
                 static_cast<uint32_t>(width), static_cast<uint32_t>(height),
                 static_cast<uint32_t>(channels))
               ? 1
               : 0;
}

}// namespace luisa::gallery::image_io
