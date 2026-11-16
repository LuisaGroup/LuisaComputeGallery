#include <algorithm>
#include <cstdint>

#include <luisa/core/clock.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>
#include <luisa/gui/window.h>
#include <luisa/dsl/sugar.h>
#include <luisa/backends/ext/dx_hdr_ext.hpp>
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include "common/headless.h"

using namespace luisa;
using namespace luisa::compute;
int main(int argc, char *argv[]) {
    // Consume --headless/--cli, --frames N and --output-dir DIR; every other
    // argument stays in argv for the example itself.
    auto headless = luisa::gallery::headless::parse(argc, argv);
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend> [--headless] [--frames N] [--output-dir DIR]. The interactive path always uses dx; --headless accepts any backend.", argv[0]);
        exit(1);
    }
    Context context{argv[0]};
    // The interactive path always drives the DX HDR extension (that is what the
    // example demonstrates); the headless path is backend agnostic, so it honours
    // <backend> like every other example.
    auto backend = headless.enabled() ? luisa::string_view{argv[1]} : luisa::string_view{"dx"};
    Device device = context.create_device(backend);
    auto image_width = 0;
    auto image_height = 0;
    auto image_channels = 0;
    auto image_pixels = stbi_load("genshin_start.jpg", &image_width, &image_height, &image_channels, 4);
    // The sample image does not ship with the repository: the headless path keeps
    // going on a generated stand-in so the example can run unattended.
    luisa::vector<uint8_t> stand_in_pixels;
    if (image_pixels == nullptr && headless.enabled()) {
        LUISA_WARNING("Cannot load 'genshin_start.jpg'; generating a stand-in image.");
        image_width = 320;
        image_height = 180;
        stand_in_pixels.resize(static_cast<size_t>(image_width) * image_height * 4u);
        for (auto y = 0; y < image_height; y++) {
            for (auto x = 0; x < image_width; x++) {
                auto offset = (static_cast<size_t>(y) * image_width + x) * 4u;
                stand_in_pixels[offset + 0u] = static_cast<uint8_t>(x * 255 / (image_width - 1));
                stand_in_pixels[offset + 1u] = static_cast<uint8_t>(y * 255 / (image_height - 1));
                stand_in_pixels[offset + 2u] = static_cast<uint8_t>(128.0f + 96.0f * ((x / 32 + y / 32) % 2));
                stand_in_pixels[offset + 3u] = 255u;
            }
        }
        image_pixels = stand_in_pixels.data();
    }

    auto texture = device.create_image<float>(PixelStorage::BYTE4, uint2(image_width, image_height), 6u);
    Stream stream = device.create_stream(headless.enabled() ? StreamTag::COMPUTE : StreamTag::GRAPHICS);
    const uint2 resolution{(uint)image_width, (uint)image_height};
    float white_point{1.0f};
    float scale = 1.0f;
    // The window, the HDR extension and the swapchain only exist in the
    // interactive path: with --headless the composite goes to a plain BYTE4
    // image instead.
    luisa::unique_ptr<Window> window;
    Swapchain swap_chain;
    auto output_storage = PixelStorage::BYTE4;
    int mode = 0;
    if (!headless.enabled()) {
        window = luisa::make_unique<Window>("path tracing", resolution);
        auto dx_hdr_ext = device.extension<DXHDRExt>();
        bool use_hdr10 = true;
        if (dx_hdr_ext->device_support_hdr()) {
            auto display_data = dx_hdr_ext->get_display_data(window->native_handle());
            LUISA_INFO("\nDisplay data: \nbits_per_color: {}\ncolor_space: {}\nred_primary: {}\ngreen_primary: {}\nblue_primary: {}\nwhite_point: {}\nmin_luminance: {}\nmax_luminance: {}\nmax_full_frame_luminance: {}", display_data.bits_per_color, luisa::to_string(display_data.color_space), display_data.red_primary, display_data.green_primary, display_data.blue_primary, display_data.white_point, display_data.min_luminance, display_data.max_luminance, display_data.max_full_frame_luminance);
            white_point = display_data.max_luminance / 80.0f;
            use_hdr10 = display_data.bits_per_color <= 10;
        }
        const PixelStorage hdr_storage = use_hdr10 ? PixelStorage::R10G10B10A2 : PixelStorage::HALF4;
        const DXHDRExt::ColorSpace hdr_color_space = use_hdr10 ? DXHDRExt::ColorSpace::RGB_FULL_G2084_NONE_P2020 : DXHDRExt::ColorSpace::RGB_FULL_G10_NONE_P709;
        swap_chain = dx_hdr_ext->create_swapchain(
            stream,
            DXHDRExt::DXSwapchainOption{
                .window = window->native_handle(),
                .size = make_uint2(resolution),
                .storage = dx_hdr_ext->device_support_hdr() ? hdr_storage : PixelStorage::BYTE4,
                .wants_vsync = false,
            });
        dx_hdr_ext->set_color_space(swap_chain, dx_hdr_ext->device_support_hdr() ? hdr_color_space : DXHDRExt::ColorSpace::RGB_FULL_G22_NONE_P709);
        // The kernel encodes for whatever the swapchain stores as its back buffer.
        output_storage = swap_chain.backend_storage();
        if (output_storage == PixelStorage::R10G10B10A2) {
            mode = 1;
        } else if (output_storage == PixelStorage::HALF4) {
            mode = 2;
        }
    }

    Callable rec709_to_rec2020 = [](Float3 color) {
        float3x3 conversion =
            transpose(make_float3x3(
                0.627402, 0.329292, 0.043306,
                0.069095, 0.919544, 0.011360,
                0.016394, 0.088028, 0.895578));
        return conversion * color;
    };

    Callable linear_to_st2084 = [](Float3 color) noexcept {
        Float m1 = 2610.f / 4096.f / 4.f;
        Float m2 = 2523.f / 4096.f * 128.f;
        Float c1 = 3424.f / 4096.f;
        Float c2 = 2413.f / 4096.f * 32.f;
        Float c3 = 2392.f / 4096.f * 32.f;
        Float3 cp = pow(abs(color), m1);
        return pow((c1 + c2 * cp) / (1.0f + c3 * cp), m2);
    };
    Callable linear_to_srgb = [&](Var<float3> x) noexcept {
        return saturate(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                               12.92f * x,
                               x <= 0.00031308f));
    };

    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale, Float inter_val, Int mode) noexcept {
        UInt2 coord = dispatch_id().xy();
        Float4 hdr = lerp(float4(1.f), hdr_image.read(coord), inter_val);
        Float3 ldr = hdr.xyz() * scale;
        $switch (mode) {
            // sRGB
            $case (0) {
                ldr = linear_to_srgb(ldr);
            };
            // 10-bit
            $case (1) {

                const float st2084max = 10000.0;
                const float hdrScalar = 80.0f / st2084max;

                // The HDR scene is in Rec.709, but the display is Rec.2020
                ldr = rec709_to_rec2020(ldr);

                // Apply the ST.2084 curve to the scene.
                ldr = linear_to_st2084(ldr * hdrScalar);
            };
            // 16-bit
            $case (2) {
                // LINEAR
            };
        };
        ldr_image.write(coord, make_float4(ldr, 1.f));
    };
    auto hdr2ldr_shader = device.compile(hdr2ldr_kernel);
    Image<float> ldr_image = device.create_image<float>(output_storage, resolution);
    stream << texture.copy_from(luisa::span{image_pixels, static_cast<size_t>(image_width * image_height * 4)}) << synchronize();
    Clock clk;
    float inter_val = 0.0f;
    float light_scale = 0.f;
    // Headless: no window and no swapchain -- run `frames` composite frames and
    // write the last one to <output-dir>/gui_win_hdr.png.
    if (headless.enabled()) {
        auto frame_count = headless.frames == 0u ? 1u : headless.frames;
        for (auto frame = 0u; frame < frame_count; frame++) {
            // Walk the same fade-in the interactive loop animates.
            auto time = static_cast<double>(frame + 1u) * 4.0 / static_cast<double>(frame_count);
            light_scale = std::clamp<double>(time / 3, 0, 1);
            inter_val = std::clamp<double>(time - 3.0, 0, 1);
            stream << hdr2ldr_shader(texture, ldr_image, white_point * light_scale, inter_val, mode).dispatch(resolution);
        }
        luisa::vector<uint8_t> host_image(static_cast<size_t>(resolution.x) * resolution.y * 4u);
        stream << ldr_image.copy_to(luisa::span{host_image}) << synchronize();
        auto file = headless.path("gui_win_hdr.png");
        if (stbi_write_png(file.c_str(), static_cast<int>(resolution.x), static_cast<int>(resolution.y),
                           4, host_image.data(), 0) == 0) {
            LUISA_WARNING("Failed to write '{}'.", file);
            return 1;
        }
        LUISA_INFO("Headless win_hdr: {} frame(s) of {}x{} sRGB composite in {:.1f} ms (no window, no HDR swapchain), wrote '{}'.",
                   frame_count, resolution.x, resolution.y, clk.toc(), file);
        return 0;
    }
    while (!window->should_close()) {
        window->poll_events();
        auto time = (clk.toc() * 1e-3) - 0.1;
        light_scale = std::clamp<double>(time / 3, 0, 1);
        inter_val = std::clamp<double>(time - 3.0, 0, 1);
        stream << hdr2ldr_shader(texture, ldr_image, white_point * light_scale, inter_val, mode).dispatch(resolution)
               << swap_chain.present(ldr_image);
    }
    stream << synchronize();
}
