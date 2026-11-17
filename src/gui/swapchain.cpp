// Multi-Window Swapchain Test
// Demonstrates creating multiple windows and swapchains from a single
// device and rendering to all of them simultaneously.
//
// Features demonstrated:
// - Multiple window creation
// - Multiple swapchain management
// - Synchronized rendering to multiple outputs
// - Framerate measurement

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/syntax.h>
#include <luisa/gui/window.h>
#include <luisa/gui/framerate.h>
#include <stb/stb_image_write.h>

#include "common/image_io.h"
#include "common/headless.h"

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    Context context{argv[0]};

    // Consume --headless/--cli, --frames N and --output-dir DIR; every other
    // argument stays in argv for the example itself.
    auto headless = luisa::gallery::headless::parse(argc, argv);

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--headless] [--frames N] [--output-dir DIR]. <backend>: cuda, dx, metal, vk, hip, fallback", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    static constexpr auto width = 2048u;
    static constexpr auto height = 1024u;
    static constexpr auto resolution = make_uint2(width, height);

    // Animated background shader
    auto draw = device.compile<2>([](ImageFloat image, Float time) noexcept {
        auto p = dispatch_id().xy();
        auto uv = make_float2(p) / make_float2(resolution) * 2.0f - 1.0f;
        auto color = def(make_float4());
        Constant<float> scales{pi, luisa::exp(1.f), luisa::sqrt(2.f)};
        for (auto i = 0u; i < 3u; i++) {
            color[i] = cos(time * scales[i] + uv.y * 11.f +
                           sin(-time * scales[2u - i] + uv.x * 7.f) * 4.f) *
                           .5f +
                       .5f;
        }
        color[3] = 1.0f;
        image.write(p, color);
    });

    Stream stream = device.create_stream(headless.enabled() ? StreamTag::COMPUTE : StreamTag::GRAPHICS);
    auto image = device.create_image<float>(PixelStorage::BYTE4, resolution);

    // Headless: no window and no swapchain -- render `frames` frames of the
    // animated background offscreen and write the result out.
    if (headless.enabled()) {
        auto frame_count = headless.frames == 0u ? 1u : headless.frames;
        Clock clock;
        for (auto frame = 0u; frame < frame_count; frame++) {
            stream << draw(image, static_cast<float>(clock.toc() * 1e-3))
                          .dispatch(resolution);
        }
        luisa::vector<uint8_t> host_image(static_cast<size_t>(width) * height * 4u);
        stream << image.copy_to(luisa::span{host_image}) << synchronize();
        auto file = headless.path("gui_swapchain.png");
        if (luisa::gallery::image_io::write_image(file.c_str(), static_cast<int>(width), static_cast<int>(height),
                           4, host_image.data(), 0) == 0) {
            LUISA_WARNING("Failed to write '{}'.", file);
            return 1;
        }
        LUISA_INFO("Headless swapchain: {} frame(s) at {}x{} rendered in {:.1f} ms (no window, no swapchain), wrote '{}'.",
                   frame_count, width, height, clock.toc(), file);
        return 0;
    }

    // Package for window + swapchain
    struct PackagedWindow {
        Window window;
        Swapchain swapchain;
    };

    // Create multiple windows
    static constexpr auto window_count = 4u;
    luisa::vector<PackagedWindow> windows;
    windows.reserve(window_count);
    for (auto i = 0u; i < window_count; i++) {
        Window window{luisa::format("Window #{}", i), resolution >> i};
        auto swpachain = device.create_swapchain(stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = resolution,
            .wants_hdr = false,
            .wants_vsync = false,
            .back_buffer_count = 3,
        });
        windows.emplace_back(PackagedWindow{
            std::move(window),
            std::move(swpachain)});
    }

    // Render to all windows
    Clock clk;
    Framerate framerate;
    while (std::all_of(windows.cbegin(), windows.cend(), [](auto &&w) noexcept {
        return !w.window.should_close();
    })) {
        stream << draw(image, static_cast<float>(clk.toc() * 1e-3))
                      .dispatch(resolution);
        for (auto &&w : windows) { stream << w.swapchain.present(image); }
        framerate.record(1u);
        LUISA_INFO("FPS: {}", framerate.report());
        for (auto &&w : windows) { w.window.poll_events(); }
    }
}
