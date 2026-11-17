// Static Image Swapchain Test
// Demonstrates displaying a static image using the swapchain.
// Loads an image from file and presents it continuously.
//
// Features demonstrated:
// - Loading images with stb_image
// - Creating swapchains for image display
// - Framerate measurement
// - Static content presentation

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/gui/window.h>
#include <luisa/gui/framerate.h>
#include <luisa/runtime/swapchain.h>

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
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream(headless.enabled() ? StreamTag::COMPUTE : StreamTag::GRAPHICS);

    // Load image from file using stb_image
    auto width = 0;
    auto height = 0;
    auto channels = 0;
    auto pixels = stbi_load("src/tests/logo.png", &width, &height, &channels, 4);
    auto loaded_from_file = pixels != nullptr;
    // The logo does not ship with every checkout: the headless path keeps going
    // on a generated stand-in image so a script still gets a deterministic run.
    luisa::vector<uint8_t> stand_in_pixels;
    if (pixels == nullptr && headless.enabled()) {
        LUISA_WARNING("Cannot load 'src/tests/logo.png'; generating a stand-in image.");
        width = 256;
        height = 256;
        stand_in_pixels.resize(static_cast<size_t>(width) * height * 4u);
        for (auto y = 0; y < height; y++) {
            for (auto x = 0; x < width; x++) {
                auto offset = (static_cast<size_t>(y) * width + x) * 4u;
                auto checker = ((x / 32 + y / 32) % 2 == 0) ? 200.0f : 40.0f;
                stand_in_pixels[offset + 0u] = static_cast<uint8_t>(checker);
                stand_in_pixels[offset + 1u] = static_cast<uint8_t>(x * 255 / (width - 1));
                stand_in_pixels[offset + 2u] = static_cast<uint8_t>(y * 255 / (height - 1));
                stand_in_pixels[offset + 3u] = 255u;
            }
        }
        pixels = stand_in_pixels.data();
    }
    auto resolution = make_uint2(width, height);
    auto image = device.create_image<float>(PixelStorage::BYTE4, resolution);
    stream << image.copy_from(luisa::span{pixels, static_cast<size_t>(width * height * 4)}) << synchronize();
    if (loaded_from_file) { stbi_image_free(pixels); }

    // Headless: no window and no swapchain -- "present" the static image to host
    // memory `frames` times and write the last frame out.
    if (headless.enabled()) {
        auto frame_count = headless.frames == 0u ? 1u : headless.frames;
        luisa::vector<uint8_t> host_image(static_cast<size_t>(width) * height * 4u);
        Clock clock;
        for (auto frame = 0u; frame < frame_count; frame++) {
            stream << image.copy_to(luisa::span{host_image}) << synchronize();
        }
        auto file = headless.path("gui_swapchain_static.png");
        if (luisa::gallery::image_io::write_image(file.c_str(), width, height, 4, host_image.data(), 0) == 0) {
            LUISA_WARNING("Failed to write '{}'.", file);
            return 1;
        }
        LUISA_INFO("Headless swapchain_static: {} frame(s) of {}x{} in {:.1f} ms (no window, no swapchain), wrote '{}'.",
                   frame_count, width, height, clock.toc(), file);
        return 0;
    }

    // Create window and swapchain
    Window window{"Display", resolution};
    SwapchainOption sc_options{
        .display = window.native_display(),
        .window = window.native_handle(),
        .size = resolution,
        .wants_hdr = false,
        .wants_vsync = true,
        .back_buffer_count = 8,
    };
    auto swapchain = device.create_swapchain(stream, sc_options);

    // Display loop with FPS measurement
    Clock clk;
    Framerate framerate;
    while (!window.should_close()) {
        stream << swapchain.present(image);
        framerate.record(1u);
        LUISA_INFO("FPS: {}", framerate.report());
        window.poll_events();
    }
}
