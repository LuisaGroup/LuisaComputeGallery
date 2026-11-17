#pragma once
/**
 * @file headless.h
 * @brief Headless (command line / CI / agent) entry point shared by the gallery
 *        examples.
 *
 * Every example in this repository must be runnable without a window: an agent,
 * a CI job or a script has no desktop to click on. The convention is
 *
 *     example_xxx <backend> [--headless] [--frames N] [--output-dir DIR] [more options]
 *
 * `--headless` (alias of the older `--offline` where an example already had one)
 * means: do not create a window or swapchain, run a bounded amount of work, write
 * any image/report under `--output-dir`, then exit with status 0 on success and
 * non-zero on failure. `--frames` and `--output-dir` are optional refinements.
 *
 * Two ways to use this header:
 *
 *   // 1. examples with their own argument parsing: consume the generic flags and
 *   //    keep parsing the rest
 *   auto headless = luisa::gallery::headless::parse(argc, argv);
 *   ...
 *   if (headless.enabled()) { ... bounded, window-less path ... }
 *
 *   // 2. examples that only need the yes/no answer
 *   if (luisa::gallery::headless::wants_headless(argc, argv)) { ... }
 *
 * Examples that parse luisa::ref::ExampleOptions (rendering/ and simulation/)
 * get the same behaviour from that parser and do not need this header.
 */
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace luisa::gallery::headless {

struct Options {
    /// `--headless` was requested.
    bool headless{false};
    /// `--frames N`: how many frames/iterations to run. 0 means "example default".
    uint32_t frames{0u};
    /// `--output-dir DIR`: where images/reports go. Defaults to the working dir.
    std::string output_dir{"."};
    /// `--capture <path>` (alias `--screenshot <path>`): write the main result
    /// image to exactly this file. The container follows the extension
    /// (.png/.jpg/.jpeg/.bmp/.tga, see common/image_io.h), and asking for a
    /// capture implies the headless path.
    std::optional<std::string> capture_path;
    /// Set once path() handed out the capture target.
    bool capture_consumed{false};

    [[nodiscard]] bool enabled() const noexcept { return headless; }

    /// Path of an output file inside `output_dir`.
    ///
    /// With `--capture <file>` the *first* call returns that file verbatim (the
    /// picture the user asked for); any further call - a multi-image example
    /// writing its intermediates - keeps using `output_dir`.
    [[nodiscard]] std::string path(std::string_view file_name) {
        if (capture_path && !capture_consumed) {
            capture_consumed = true;
            return *capture_path;
        }
        if (output_dir.empty() || output_dir == ".") { return std::string{file_name}; }
        auto dir = output_dir;
        while (!dir.empty() && (dir.back() == '/' || dir.back() == '\\')) { dir.pop_back(); }
        return dir + "/" + std::string{file_name};
    }
};

namespace detail {

[[nodiscard]] inline bool is_flag(int argc, char *argv[], int index, std::string_view flag) noexcept {
    return index < argc && argv[index] != nullptr && flag == std::string_view{argv[index]};
}

}// namespace detail

/// Consume `--headless`, `--frames N`, `--output-dir DIR` and
/// `--capture|--screenshot PATH` from argv. Every other argument is kept, in
/// order, so the caller can continue parsing. `--offline` is *not* consumed:
/// examples that understand it (ExampleOptions) must still see it.
[[nodiscard]] inline Options parse(int &argc, char *argv[]) noexcept {
    Options options;
    std::vector<char *> kept;
    kept.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; i++) {
        if (argv[i] == nullptr) { break; }
        std::string_view arg{argv[i]};
        if (arg == "--headless" || arg == "--cli") {
            options.headless = true;
            continue;
        }
        if (arg == "--frames" && i + 1 < argc && argv[i + 1] != nullptr) {
            options.frames = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            continue;
        }
        if (arg == "--output-dir" && i + 1 < argc && argv[i + 1] != nullptr) {
            options.output_dir = argv[++i];
            continue;
        }
        if ((arg == "--capture" || arg == "--screenshot") &&
            i + 1 < argc && argv[i + 1] != nullptr) {
            options.capture_path = std::string{argv[++i]};
            options.headless = true;
            continue;
        }
        kept.push_back(argv[i]);
    }
    auto kept_count = static_cast<int>(kept.size());
    for (int i = 0; i < kept_count; i++) { argv[i] = kept[i]; }
    if (argc > 0) { argv[kept_count] = nullptr; }
    argc = kept_count;
    return options;
}

/// Convenience for examples that only need the boolean.
[[nodiscard]] inline bool wants_headless(int argc, char *argv[]) noexcept {
    for (int i = 1; i < argc; i++) {
        if (argv[i] == nullptr) { break; }
        std::string_view arg{argv[i]};
        if (arg == "--headless" || arg == "--cli" || arg == "--offline") {
            return true;
        }
    }
    return false;
}

}// namespace luisa::gallery::headless
