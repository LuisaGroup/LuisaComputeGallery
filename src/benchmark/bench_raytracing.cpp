// Ray tracing benchmarks.
//
// Four independent GPU kernels exercise the ray tracing side of the backend:
//
//   1. `rt_cornell_path_tracing` - hardware ray query path tracing over the
//      Cornell Box triangle scene (NEE + MIS + Russian roulette) with
//      `reorder_shader_execution()` right after the primary ray query, 8 spp
//      per iteration.
//   2. `rt_nested_callable_pt` - the same scene traced through nested
//      `$lambda` / `Callable` closures that keep intermediate shading values in
//      `luisa::optional` capture cells and never reorder execution. This is the
//      shape of code that stresses DXC inlining, SROA and aggregate copy
//      elimination, so it is timed separately from (1).
//   3. `rt_sdf_raymarch` - analytic signed distance field sphere tracing with
//      3-tap finite difference normals, an analytic disk light and 6 bounces,
//      16 samples per pixel per iteration.
//   4. `rt_voxel_dda` - 3D DDA traversal of a 64^3 voxel grid with 256 steps,
//      ambient occlusion and directional lighting, 4 frames per iteration.
//
// Scene construction, shader compilation and image allocation all happen
// outside the timed region: `run_timed_bench` only measures the recorded
// dispatch. Every benchmark then re-reads its render target once and checks
// that the image is finite, non-degenerate and sensibly exposed.

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/optional.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numbers>
#include <vector>

#include "bench_common.h"

#include "cornell_box.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace luisa;
using namespace luisa::compute;

// ---------------------------------------------------------------------------
// Shading basis of the Cornell Box path tracers.
// ---------------------------------------------------------------------------

// Registered as a DSL structure, so a kernel can build one with
// `def<Onb>(tangent, binormal, normal)` and read it back through `Var<Onb>`.
struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

LUISA_STRUCT(Onb, tangent, binormal, normal) {
    // Transform a direction from the local shading frame to world space.
    [[nodiscard]] Float3 to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};

// Host-side basis of the nested-callable path tracer: it stores DSL
// expressions (not concrete vectors) and is deliberately not registered, which
// is exactly how `path_tracing_nested_callable.cpp` keeps it.
struct NestedOnb {
    Float3 tangent;
    Float3 binormal;
    Float3 normal;
    [[nodiscard]] Float3 to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};

namespace luisa::bench {
namespace {

// ---------------------------------------------------------------------------
// Host-side image verification.
// ---------------------------------------------------------------------------

struct LumaStats {
    double mean{0.0};
    double min_value{0.0};
    double max_value{0.0};
};

// Same transfer function the display kernels use, evaluated on the host.
[[nodiscard]] float srgb_encode(float x) noexcept {
    x = std::clamp(x, 0.0f, 1.0f);
    return x <= 0.00031308f ?
               12.92f * x :
               1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

// Count NaN/Inf channels of an RGBA32F image.
[[nodiscard]] uint64_t count_non_finite_rgba(luisa::span<const float> rgba) noexcept {
    uint64_t count = 0u;
    for (auto v : rgba) { count += std::isfinite(v) ? 0u : 1u; }
    return count;
}

// Display-referred luma of an RGBA8 image (the `hdr2ldr` output).
[[nodiscard]] LumaStats luma_from_rgba8(luisa::span<const uint8_t> rgba) noexcept {
    LUISA_ASSERT(rgba.size() % 4u == 0u,
                 "RGBA8 image data must be 4 bytes per pixel, got {}.",
                 rgba.size());
    LumaStats stats{.mean = 0.0, .min_value = 1.0, .max_value = 0.0};
    auto pixel_count = rgba.size() / 4u;
    if (pixel_count == 0u) {
        stats.min_value = 0.0;
        return stats;
    }
    double sum = 0.0;
    for (size_t i = 0u; i < pixel_count; ++i) {
        auto r = static_cast<double>(rgba[4u * i + 0u]) * (1.0 / 255.0);
        auto g = static_cast<double>(rgba[4u * i + 1u]) * (1.0 / 255.0);
        auto b = static_cast<double>(rgba[4u * i + 2u]) * (1.0 / 255.0);
        auto luma = 0.212671 * r + 0.715160 * g + 0.072169 * b;
        sum += luma;
        stats.min_value = std::min(stats.min_value, luma);
        stats.max_value = std::max(stats.max_value, luma);
    }
    stats.mean = sum / static_cast<double>(pixel_count);
    return stats;
}

// Display-referred luma of a linear RGBA32F render target: the stored radiance
// is clamped to the display range and gamma encoded before the luma weights are
// applied, so all four benchmarks report the same kind of number.
[[nodiscard]] LumaStats luma_from_rgba32f(luisa::span<const float> rgba) noexcept {
    LUISA_ASSERT(rgba.size() % 4u == 0u,
                 "RGBA32F image data must be 4 floats per pixel, got {}.",
                 rgba.size());
    LumaStats stats{.mean = 0.0, .min_value = 1.0, .max_value = 0.0};
    auto pixel_count = rgba.size() / 4u;
    double sum = 0.0;
    size_t valid_pixels = 0u;
    for (size_t i = 0u; i < pixel_count; ++i) {
        auto r = rgba[4u * i + 0u];
        auto g = rgba[4u * i + 1u];
        auto b = rgba[4u * i + 2u];
        if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b)) { continue; }
        auto luma = static_cast<double>(
            srgb_encode(0.212671f * r + 0.715160f * g + 0.072169f * b));
        sum += luma;
        ++valid_pixels;
        stats.min_value = std::min(stats.min_value, luma);
        stats.max_value = std::max(stats.max_value, luma);
    }
    if (valid_pixels == 0u) {
        stats.min_value = 0.0;
        stats.max_value = 0.0;
        return stats;
    }
    stats.mean = sum / static_cast<double>(valid_pixels);
    return stats;
}

// Fold the readback of a render target into the benchmark result: the image has
// to be finite, exposed inside (luma_lower, luma_upper) and actually varied.
bool apply_image_check(
    const BenchConfig &cfg,
    BenchResult &result,
    uint64_t non_finite_channels,
    const LumaStats &luma,
    double luma_lower,
    double luma_upper) noexcept {
    if (!cfg.verify) {
        result.verified = true;
        result.check_note = "verification disabled";
        return true;
    }
    auto finite = non_finite_channels == 0u;
    auto exposed = luma.mean > luma_lower && luma.mean < luma_upper;
    auto varied = luma.max_value - luma.min_value > 1.0e-4;
    auto ok = finite && exposed && varied;
    result.verified = ok;
    result.check_note = luisa::format(
        "mean_luma={:.4f} (want {:.3f}..{:.3f}), display_range=[{:.4f}, {:.4f}], "
        "nonfinite_channels={}",
        luma.mean, luma_lower, luma_upper,
        luma.min_value, luma.max_value, non_finite_channels);
    if (!ok) {
        LUISA_WARNING_WITH_LOCATION(
            "Ray tracing benchmark '{}' failed its image check: {}.",
            result.name, result.check_note.c_str());
    }
    return ok;
}

void emit(std::vector<BenchResult> &results, BenchResult &&result) noexcept {
    LUISA_INFO("raytracing/{}: median {:.3f} ms, {:.3f} {} | verified={} | {}",
               result.name, result.median_ms, result.throughput,
               result.throughput_unit, result.verified ? "yes" : "no",
               result.check_note.c_str());
    results.emplace_back(std::move(result));
}

// ---------------------------------------------------------------------------
// Cornell Box scene shared by the two path tracing benchmarks.
// ---------------------------------------------------------------------------

struct CornellBoxScene {
    Buffer<float3> vertices;
    Buffer<float3> materials;
    luisa::vector<Buffer<Triangle>> triangle_buffers;
    luisa::vector<Mesh> meshes;
    BindlessArray heap;
    Accel accel;
};

// The Cornell Box has eight shapes (floor, ceiling, three walls, two boxes and
// the emissive light); the light is always the last one, which lets both path
// tracers identify a light hit by its instance index.
static constexpr size_t cornell_box_shape_count = 8u;
static constexpr size_t cornell_box_light_index = cornell_box_shape_count - 1u;

[[nodiscard]] CornellBoxScene load_cornell_box(compute::Device &device,
                                               compute::Stream &stream) {
    tinyobj::ObjReaderConfig obj_reader_config;
    obj_reader_config.triangulate = true;
    obj_reader_config.vertex_color = false;
    tinyobj::ObjReader obj_reader;
    auto loaded = obj_reader.ParseFromString(obj_string, "", obj_reader_config);
    LUISA_ASSERT(loaded, "Failed to parse the embedded Cornell Box OBJ: {}.",
                 obj_reader.Error());
    if (!obj_reader.Warning().empty()) {
        LUISA_WARNING_WITH_LOCATION("{}", obj_reader.Warning());
    }

    auto &&positions = obj_reader.GetAttrib().vertices;
    luisa::vector<float3> vertices;
    vertices.reserve(positions.size() / 3u);
    for (size_t i = 0u; i + 2u < positions.size(); i += 3u) {
        vertices.emplace_back(make_float3(
            positions[i + 0u], positions[i + 1u], positions[i + 2u]));
    }
    LUISA_ASSERT(!vertices.empty(), "The embedded Cornell Box has no vertices.");
    LUISA_ASSERT(obj_reader.GetShapes().size() == cornell_box_shape_count,
                 "The embedded Cornell Box must produce {} shapes, got {}.",
                 cornell_box_shape_count, obj_reader.GetShapes().size());

    CornellBoxScene scene;
    scene.vertices = device.create_buffer<float3>(vertices.size());
    stream << scene.vertices.copy_from(luisa::span{vertices});

    scene.heap = device.create_bindless_array(cornell_box_shape_count);
    scene.triangle_buffers.reserve(cornell_box_shape_count);
    scene.meshes.reserve(cornell_box_shape_count);
    size_t total_triangles = 0u;
    for (auto &&shape : obj_reader.GetShapes()) {
        auto instance = static_cast<uint32_t>(scene.meshes.size());
        auto &&shape_indices = shape.mesh.indices;
        LUISA_ASSERT(!shape_indices.empty() && shape_indices.size() % 3u == 0u,
                     "Shape '{}' is not a triangle list.", shape.name);
        auto triangle_count = shape_indices.size() / 3u;
        total_triangles += triangle_count;
        luisa::vector<uint32_t> vertex_indices;
        vertex_indices.reserve(shape_indices.size());
        for (auto index : shape_indices) {
            vertex_indices.emplace_back(static_cast<uint32_t>(index.vertex_index));
        }
        auto &triangle_buffer = scene.triangle_buffers.emplace_back(
            device.create_buffer<Triangle>(triangle_count));
        auto &mesh = scene.meshes.emplace_back(
            device.create_mesh(scene.vertices, triangle_buffer));
        scene.heap.emplace_on_update(instance, triangle_buffer);
        stream << triangle_buffer.copy_from(luisa::span{vertex_indices})
               << mesh.build();
    }

    // Diffuse albedos of the eight Cornell Box materials; the light itself is
    // handled analytically by both kernels.
    float3 albedos[] = {
        make_float3(0.725f, 0.710f, 0.680f),// floor
        make_float3(0.725f, 0.710f, 0.680f),// ceiling
        make_float3(0.725f, 0.710f, 0.680f),// back wall
        make_float3(0.140f, 0.450f, 0.091f),// right wall (green)
        make_float3(0.630f, 0.065f, 0.050f),// left wall (red)
        make_float3(0.725f, 0.710f, 0.680f),// short box
        make_float3(0.725f, 0.710f, 0.680f),// tall box
        make_float3(0.000f, 0.000f, 0.000f),// light (emissive)
    };
    scene.materials = device.create_buffer<float3>(std::size(albedos));

    scene.accel = device.create_accel({});
    for (auto &mesh : scene.meshes) {
        scene.accel.emplace_back(mesh, make_float4x4(1.0f));
    }
    stream << scene.heap.update()
           << scene.accel.build()
           << scene.materials.copy_from(luisa::span{albedos, std::size(albedos)})
           << synchronize();
    LUISA_INFO("Cornell Box uploaded: {} vertices, {} shapes, {} triangles.",
               vertices.size(), scene.meshes.size(), total_triangles);
    return scene;
}

// ---------------------------------------------------------------------------
// 1. Hardware ray query path tracing on the Cornell Box.
// ---------------------------------------------------------------------------

void bench_cornell_path_tracing(compute::Device &device,
                                compute::Stream &stream,
                                const BenchConfig &cfg,
                                std::vector<BenchResult> &results) {

    static constexpr auto name = "rt_cornell_path_tracing";
    if (!should_run(cfg, "raytracing", name)) { return; }
    static constexpr uint32_t width = 1024u;
    static constexpr uint32_t height = 1024u;
    static constexpr uint32_t spp = 8u;
    static constexpr uint32_t max_depth = 10u;
    auto resolution = make_uint2(width, height);
    double compile_ms = 0.0;

    auto scene = load_cornell_box(device, stream);

    // TEA (Tiny Encryption Algorithm) seeds one random stream per pixel.
    Callable tea = [](UInt v0, UInt v1) noexcept {
        UInt s0 = def(0u);
        for (uint n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4u) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4u) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    Kernel2D make_sampler_kernel = [&](ImageUInt seed_image) noexcept {
        set_name("bench_rt_cornell_make_sampler");
        UInt2 p = dispatch_id().xy();
        seed_image.write(p, make_uint4(tea(p.x, p.y)));
    };

    // Linear congruential generator for the per-pixel sample stream.
    Callable lcg = [](UInt &state) noexcept {
        constexpr uint lcg_a = 1664525u;
        constexpr uint lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };

    Callable make_onb = [](const Float3 &normal) noexcept {
        Float3 binormal = normalize(ite(
            abs(normal.x) > abs(normal.z),
            make_float3(-normal.y, normal.x, 0.0f),
            make_float3(0.0f, -normal.z, normal.y)));
        Float3 tangent = normalize(cross(binormal, normal));
        return def<Onb>(tangent, binormal, normal);
    };

    Callable generate_ray = [](Float2 p) noexcept {
        static constexpr float fov = radians(27.8f);
        static constexpr float3 origin = make_float3(-0.01f, 0.995f, 5.0f);
        Float3 pixel = origin + make_float3(p * tan(0.5f * fov), -1.0f);
        return make_ray(origin, normalize(pixel - origin));
    };

    Callable cosine_sample_hemisphere = [](Float2 u) noexcept {
        Float r = sqrt(u.x);
        Float phi = 2.0f * constants::pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
    };

    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return pdf_a / max(pdf_a + pdf_b, 1e-4f);
    };

    // Unidirectional path tracer: NEE for direct lighting, MIS on the light
    // surface, cosine-weighted BSDF sampling and Russian roulette termination.
    Kernel2D raytracing_kernel = [&](ImageFloat image, ImageUInt seed_image,
                                     AccelVar accel, UInt2 resolution,
                                     UInt dispatch_spp) noexcept {
        set_name("bench_rt_cornell_path_tracing");
        set_block_size(16u, 16u, 1u);
        UInt2 coord = dispatch_id().xy();
        Float frame_size = min(resolution.x, resolution.y).cast<float>();
        UInt state = seed_image.read(coord).x;
        Float3 radiance = def(make_float3(0.0f));

        constexpr float3 light_position = make_float3(-0.24f, 1.98f, 0.16f);
        constexpr float3 light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
        constexpr float3 light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
        constexpr float3 light_emission = make_float3(17.0f, 12.0f, 4.0f);
        Float light_area = length(cross(light_u, light_v));
        Float3 light_normal = normalize(cross(light_u, light_v));

        $for (i, dispatch_spp) {
            Float rx = lcg(state);
            Float ry = lcg(state);
            Float2 pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
            Var<Ray> ray = generate_ray(pixel * make_float2(1.0f, -1.0f));
            Float3 beta = def(make_float3(1.0f));// path throughput
            Float pdf_bsdf = def(0.0f);          // BSDF pdf for MIS

            $for (depth, max_depth) {
                Var<TriangleHit> hit = accel.intersect(ray, {});
                // Coalesce the lanes that survived the ray query before they
                // diverge on shading.
                reorder_shader_execution();
                $if (hit->miss()) { $break; };
                Var<Triangle> triangle = scene.heap->buffer<Triangle>(hit.inst).read(hit.prim);
                Float3 p0 = scene.vertices->read(triangle.i0);
                Float3 p1 = scene.vertices->read(triangle.i1);
                Float3 p2 = scene.vertices->read(triangle.i2);
                Float3 p = triangle_interpolate(hit.bary, p0, p1, p2);
                Float3 n = normalize(cross(p1 - p0, p2 - p0));

                Float cos_wo = dot(-ray->direction(), n);
                $if (cos_wo < 1e-4f) { $break; };

                // Direct hit on the light: full emission on the primary ray,
                // MIS weighted contribution on all secondary ones.
                $if (hit.inst == static_cast<uint>(cornell_box_light_index)) {
                    $if (depth == 0u) {
                        radiance += light_emission;
                    }
                    $else {
                        Float pdf_light = length_squared(p - ray->origin()) / (light_area * cos_wo);
                        Float mis_weight = balanced_heuristic(pdf_bsdf, pdf_light);
                        radiance += mis_weight * beta * light_emission;
                    };
                    $break;
                };

                // Next event estimation: sample the area light directly.
                Float ux_light = lcg(state);
                Float uy_light = lcg(state);
                Float3 p_light = light_position + ux_light * light_u + uy_light * light_v;
                Float3 pp = offset_ray_origin(p, n);
                Float3 pp_light = offset_ray_origin(p_light, light_normal);
                Float d_light = distance(pp, pp_light);
                Float3 wi_light = normalize(pp_light - pp);
                Var<Ray> shadow_ray = make_ray(offset_ray_origin(pp, n), wi_light, 0.f, d_light);
                Bool occluded = accel.intersect_any(shadow_ray, {});
                Float cos_wi_light = dot(wi_light, n);
                Float cos_light = -dot(light_normal, wi_light);
                Float3 albedo = scene.materials->read(hit.inst);
                $if (!occluded & cos_wi_light > 1e-4f & cos_light > 1e-4f) {
                    Float pdf_light = (d_light * d_light) / (light_area * cos_light);
                    Float pdf_bsdf_nee = cos_wi_light * inv_pi;
                    Float mis_weight = balanced_heuristic(pdf_light, pdf_bsdf_nee);
                    Float3 bsdf = albedo * inv_pi * cos_wi_light;
                    radiance += beta * bsdf * mis_weight * light_emission / max(pdf_light, 1e-4f);
                };

                // Cosine-weighted BSDF sampling for the next path segment.
                Var<Onb> onb = make_onb(n);
                Float ux = lcg(state);
                Float uy = lcg(state);
                Float3 wi_local = cosine_sample_hemisphere(make_float2(ux, uy));
                Float cos_wi = abs(wi_local.z);
                Float3 new_direction = onb->to_world(wi_local);
                ray = make_ray(pp, new_direction);
                pdf_bsdf = cos_wi * inv_pi;
                beta *= albedo;// * cos_wi * inv_pi / pdf_bsdf == 1

                // Russian roulette.
                Float l = dot(make_float3(0.212671f, 0.715160f, 0.072169f), beta);
                $if (l == 0.0f) { $break; };
                Float q = max(l, 0.05f);
                Float r = lcg(state);
                $if (r >= q) { $break; };
                beta *= 1.0f / q;
            };
        };

        radiance /= dispatch_spp.cast<float>();
        seed_image.write(coord, make_uint4(state));
        $if (any(dsl::isnan(radiance))) { radiance = make_float3(0.0f); };
        image.write(coord, make_float4(clamp(radiance, 0.0f, 30.0f), dispatch_spp.cast<float>()));
    };

    // Progressive accumulation: `w` carries the per-pixel sample count.
    Kernel2D accumulate_kernel = [](ImageFloat accum_image, ImageFloat curr_image) noexcept {
        UInt2 p = dispatch_id().xy();
        Float4 accum = accum_image.read(p);
        Float4 curr = curr_image.read(p);
        accum_image.write(p, accum + make_float4(curr.xyz() * curr.w, curr.w));
    };

    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.f));
    };

    Callable linear_to_srgb = [](Var<float3> x) noexcept {
        return saturate(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                               12.92f * x,
                               x <= 0.00031308f));
    };

    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale) noexcept {
        UInt2 coord = dispatch_id().xy();
        Float4 hdr = hdr_image.read(coord);
        Float3 ldr = linear_to_srgb(clamp(
            hdr.xyz() / max(hdr.w, 1.0e-6f) * scale, 0.f, 1.f));
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    };

    auto make_sampler_shader = compile_uncached(
        device, make_sampler_kernel, "bench_rt_cornell_make_sampler", cfg, compile_ms);
    auto raytracing_shader = compile_uncached(
        device, raytracing_kernel, "bench_rt_cornell_path_tracing", cfg, compile_ms);
    auto accumulate_shader = compile_uncached(
        device, accumulate_kernel, "bench_rt_cornell_accumulate", cfg, compile_ms);
    auto clear_shader = compile_uncached(
        device, clear_kernel, "bench_rt_cornell_clear", cfg, compile_ms);
    auto hdr2ldr_shader = compile_uncached(
        device, hdr2ldr_kernel, "bench_rt_cornell_hdr2ldr", cfg, compile_ms);

    Image<float> framebuffer = device.create_image<float>(PixelStorage::HALF4, resolution);
    Image<float> accum_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    Image<float> ldr_image = device.create_image<float>(PixelStorage::BYTE4, resolution);
    Image<uint> seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);
    luisa::vector<float> accum_host(size_t{width} * size_t{height} * 4u);
    luisa::vector<uint8_t> ldr_host(size_t{width} * size_t{height} * 4u);

    stream << clear_shader(accum_image).dispatch(resolution)
           << make_sampler_shader(seed_image).dispatch(resolution);

    static constexpr double work_units = double(1024ull * 1024ull * 8ull);
    auto result = run_timed_bench(
        "raytracing", name, stream, cfg, compile_ms, work_units, "ray_spp/s",
        [&] {
            stream << raytracing_shader(framebuffer, seed_image, scene.accel, resolution, spp)
                          .dispatch(resolution)
                   << accumulate_shader(accum_image, framebuffer)
                          .dispatch(resolution);
        });

    // Verify the final, fully accumulated image: the raw float accumulation
    // must be finite and the tone mapped display image must be exposed.
    stream << hdr2ldr_shader(accum_image, ldr_image, 2.f).dispatch(resolution)
           << accum_image.copy_to(luisa::span{accum_host})
           << ldr_image.copy_to(luisa::span{ldr_host})
           << synchronize();
    auto non_finite = count_non_finite_rgba(luisa::span<const float>{accum_host});
    auto luma = luma_from_rgba8(luisa::span<const uint8_t>{ldr_host});
    apply_image_check(cfg, result, non_finite, luma, 0.05, 0.95);

    emit(results, std::move(result));
}

// ---------------------------------------------------------------------------
// 2. Cornell Box path tracing without `reorder_shader_execution()`, built out of
//    nested `$lambda` closures and `luisa::optional` capture cells.
// ---------------------------------------------------------------------------

void bench_nested_callable_path_tracing(compute::Device &device,
                                        compute::Stream &stream,
                                        const BenchConfig &cfg,
                                        std::vector<BenchResult> &results) {

    static constexpr auto name = "rt_nested_callable_pt";
    if (!should_run(cfg, "raytracing", name)) { return; }
    static constexpr uint32_t width = 1024u;
    static constexpr uint32_t height = 1024u;
    static constexpr uint32_t spp = 8u;
    static constexpr uint32_t max_depth = 10u;
    auto resolution = make_uint2(width, height);
    double compile_ms = 0.0;

    auto scene = load_cornell_box(device, stream);

    Callable tea = [](UInt v0, UInt v1) noexcept {
        UInt s0 = def(0u);
        for (uint n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4u) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4u) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    Kernel2D make_sampler_kernel = [&](ImageUInt seed_image) noexcept {
        set_name("bench_rt_nested_make_sampler");
        UInt2 p = dispatch_id().xy();
        seed_image.write(p, make_uint4(tea(p.x, p.y)));
    };

    Callable generate_ray = [](Float2 p) noexcept {
        static constexpr float fov = radians(27.8f);
        static constexpr float3 origin = make_float3(-0.01f, 0.995f, 5.0f);
        Float3 pixel = origin + make_float3(p * tan(0.5f * fov), -1.0f);
        return make_ray(origin, normalize(pixel - origin));
    };

    Callable cosine_sample_hemisphere = [](Float2 u) noexcept {
        Float r = sqrt(u.x);
        Float phi = 2.0f * constants::pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
    };

    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return pdf_a / max(pdf_a + pdf_b, 1e-4f);
    };

    // The nested form deliberately has no `reorder_shader_execution()`: every
    // closure body is inlined into one straight-line block, and the values that
    // cross closure boundaries travel through `luisa::optional` capture cells
    // instead of being returned, which is the pattern DXC has to fold.
    Kernel2D raytracing_kernel = [&](ImageFloat image, ImageUInt seed_image,
                                     AccelVar accel, UInt2 resolution,
                                     UInt dispatch_spp) noexcept {
        set_name("bench_rt_nested_callable_pt");
        set_block_size(16u, 16u, 1u);
        UInt2 coord = dispatch_id().xy();
        Float frame_size = min(resolution.x, resolution.y).cast<float>();
        UInt state = seed_image.read(coord).x;

        Lambda lcg = [&state] {
            constexpr uint lcg_a = 1664525u;
            constexpr uint lcg_c = 1013904223u;
            state = lcg_a * state + lcg_c;
            return cast<float>(state & 0x00ffffffu) *
                   (1.0f / static_cast<float>(0x01000000u));
        };

        auto make_onb = [](const Float3 &normal) noexcept {
            Float3 binormal = normalize(ite(
                abs(normal.x) > abs(normal.z),
                make_float3(-normal.y, normal.x, 0.0f),
                make_float3(0.0f, -normal.z, normal.y)));
            Float3 tangent = normalize(cross(binormal, normal));
            return NestedOnb{tangent, binormal, normal};
        };

        constexpr float3 light_position = make_float3(-0.24f, 1.98f, 0.16f);
        constexpr float3 light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
        constexpr float3 light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
        constexpr float3 light_emission = make_float3(17.0f, 12.0f, 4.0f);
        Float light_area = length(cross(light_u, light_v));
        Float3 light_normal = normalize(cross(light_u, light_v));

        Float3 radiance = def(make_float3(0.0f));
        $for (i, dispatch_spp) {
            Float rx = lcg();
            Float ry = lcg();
            Float2 pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
            Var<Ray> ray = generate_ray(pixel * make_float2(1.0f, -1.0f));
            Float3 beta = def(make_float3(1.0f));
            Float pdf_bsdf = def(0.0f);
            $for (depth, max_depth) {
                Var<TriangleHit> hit = accel.intersect(ray, {});
                $if (hit->miss()) { $break; };
                Var<Triangle> triangle = scene.heap->buffer<Triangle>(hit.inst).read(hit.prim);
                Float3 p0 = scene.vertices->read(triangle.i0);
                Float3 p1 = scene.vertices->read(triangle.i1);
                Float3 p2 = scene.vertices->read(triangle.i2);
                Float3 p = triangle_interpolate(hit.bary, p0, p1, p2);
                Float3 n = normalize(cross(p1 - p0, p2 - p0));
                Float cos_wo = dot(-ray->direction(), n);
                $if (cos_wo < 1e-4f) { $break; };

                // Light hit: full emission on the camera ray, MIS elsewhere.
                $if (hit.inst == static_cast<uint>(cornell_box_light_index)) {
                    $if (depth == 0u) {
                        radiance += light_emission;
                    }
                    $else {
                        Float pdf_light = length_squared(p - ray->origin()) / (light_area * cos_wo);
                        Float mis_weight = balanced_heuristic(pdf_bsdf, pdf_light);
                        radiance += mis_weight * beta * light_emission;
                    };
                    $break;
                };

                // Sample the light through a nested closure that publishes the
                // shading point and the albedo into optional capture cells.
                luisa::optional<Float3> pp;
                luisa::optional<Float3> albedo;
                $lambda({
                    Float ux_light = lcg();
                    Float uy_light = lcg();
                    Float3 p_light = light_position + ux_light * light_u + uy_light * light_v;
                    Float3 pp_light = offset_ray_origin(p_light, light_normal);
                    pp.emplace(offset_ray_origin(p, n));
                    albedo.emplace(scene.materials->read(hit.inst));
                    Float d_light = distance(*pp, pp_light);
                    Float3 wi_light = normalize(pp_light - *pp);
                    $lambda({
                        Var<Ray> shadow_ray = make_ray(offset_ray_origin(*pp, n), wi_light, 0.f, d_light);
                        Bool occluded = accel.intersect_any(shadow_ray, {});
                        Float cos_wi_light = dot(wi_light, n);
                        Float cos_light = -dot(light_normal, wi_light);
                        $if (!occluded & cos_wi_light > 1e-4f & cos_light > 1e-4f) {
                            Float pdf_light = (d_light * d_light) / (light_area * cos_light);
                            Float pdf_bsdf_nee = cos_wi_light * inv_pi;
                            Float mis_weight = balanced_heuristic(pdf_light, pdf_bsdf_nee);
                            Float3 bsdf = *albedo * inv_pi * cos_wi_light;
                            radiance += beta * bsdf * mis_weight * light_emission / max(pdf_light, 1e-4f);
                        };
                    })();
                })();

                // Sample the BSDF through a parameterized closure that writes
                // back into the enclosing path state.
                auto sample_f = $lambda((Float2 u) {
                    NestedOnb onb = make_onb(n);
                    Float3 wi_local = cosine_sample_hemisphere(make_float2(u.x, u.y));
                    Float cos_wi = abs(wi_local.z);
                    Float3 new_direction = onb.to_world(wi_local);
                    ray = make_ray(*pp, new_direction);
                    pdf_bsdf = cos_wi * inv_pi;
                    beta *= *albedo;// * cos_wi * inv_pi / pdf_bsdf == 1
                });

                $lambda({
                    Float ux = lcg();
                    Float uy = lcg();
                    sample_f(make_float2(ux, uy));
                })();

                // Russian roulette.
                Float l = dot(make_float3(0.212671f, 0.715160f, 0.072169f), beta);
                $if (l == 0.0f) { $break; };
                Float q = max(l, 0.05f);
                Float r = lcg();
                $if (r >= q) { $break; };
                beta *= 1.0f / q;
            };
        };

        radiance /= dispatch_spp.cast<float>();
        seed_image.write(coord, make_uint4(state));
        $if (any(dsl::isnan(radiance))) { radiance = make_float3(0.0f); };
        image.write(coord, make_float4(clamp(radiance, 0.0f, 30.0f), dispatch_spp.cast<float>()));
    };

    Kernel2D accumulate_kernel = [](ImageFloat accum_image, ImageFloat curr_image) noexcept {
        UInt2 p = dispatch_id().xy();
        Float4 accum = accum_image.read(p);
        Float4 curr = curr_image.read(p);
        accum_image.write(p, accum + make_float4(curr.xyz() * curr.w, curr.w));
    };

    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.f));
    };

    Callable linear_to_srgb = [](Var<float3> x) noexcept {
        return saturate(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                               12.92f * x,
                               x <= 0.00031308f));
    };

    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale) noexcept {
        UInt2 coord = dispatch_id().xy();
        Float4 hdr = hdr_image.read(coord);
        Float3 ldr = linear_to_srgb(clamp(
            hdr.xyz() / max(hdr.w, 1.0e-6f) * scale, 0.f, 1.f));
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    };

    auto make_sampler_shader = compile_uncached(
        device, make_sampler_kernel, "bench_rt_nested_make_sampler", cfg, compile_ms);
    auto raytracing_shader = compile_uncached(
        device, raytracing_kernel, "bench_rt_nested_callable_pt", cfg, compile_ms);
    auto accumulate_shader = compile_uncached(
        device, accumulate_kernel, "bench_rt_nested_accumulate", cfg, compile_ms);
    auto clear_shader = compile_uncached(
        device, clear_kernel, "bench_rt_nested_clear", cfg, compile_ms);
    auto hdr2ldr_shader = compile_uncached(
        device, hdr2ldr_kernel, "bench_rt_nested_hdr2ldr", cfg, compile_ms);

    Image<float> framebuffer = device.create_image<float>(PixelStorage::HALF4, resolution);
    Image<float> accum_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    Image<float> ldr_image = device.create_image<float>(PixelStorage::BYTE4, resolution);
    Image<uint> seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);
    luisa::vector<float> accum_host(size_t{width} * size_t{height} * 4u);
    luisa::vector<uint8_t> ldr_host(size_t{width} * size_t{height} * 4u);

    stream << clear_shader(accum_image).dispatch(resolution)
           << make_sampler_shader(seed_image).dispatch(resolution);

    static constexpr double work_units = double(1024ull * 1024ull * 8ull);
    auto result = run_timed_bench(
        "raytracing", name, stream, cfg, compile_ms, work_units, "ray_spp/s",
        [&] {
            stream << raytracing_shader(framebuffer, seed_image, scene.accel, resolution, spp)
                          .dispatch(resolution)
                   << accumulate_shader(accum_image, framebuffer)
                          .dispatch(resolution);
        });

    stream << hdr2ldr_shader(accum_image, ldr_image, 2.f).dispatch(resolution)
           << accum_image.copy_to(luisa::span{accum_host})
           << ldr_image.copy_to(luisa::span{ldr_host})
           << synchronize();
    auto non_finite = count_non_finite_rgba(luisa::span<const float>{accum_host});
    auto luma = luma_from_rgba8(luisa::span<const uint8_t>{ldr_host});
    apply_image_check(cfg, result, non_finite, luma, 0.05, 0.95);

    emit(results, std::move(result));
}

// ---------------------------------------------------------------------------
// 3. Analytic SDF sphere tracing with finite difference normals.
// ---------------------------------------------------------------------------

void bench_sdf_raymarch(compute::Device &device,
                        compute::Stream &stream,
                        const BenchConfig &cfg,
                        std::vector<BenchResult> &results) {

    static constexpr auto name = "rt_sdf_raymarch";
    if (!should_run(cfg, "raytracing", name)) { return; }
    static constexpr uint32_t width = 1280u;
    static constexpr uint32_t height = 720u;
    static constexpr uint32_t spp = 16u;
    static constexpr uint32_t max_ray_depth = 6u;
    static constexpr uint32_t max_march_steps = 100u;
    static constexpr float eps = 1e-4f;
    static constexpr float inf = 1e10f;
    static constexpr float fov = 0.23f;
    static constexpr float dist_limit = 100.0f;
    static constexpr float3 camera_pos = make_float3(0.0f, 0.32f, 3.7f);
    static constexpr float3 light_pos = make_float3(-1.5f, 0.6f, 0.3f);
    static constexpr float3 light_normal = make_float3(1.0f, 0.0f, 0.0f);
    static constexpr float light_radius = 2.0f;
    auto resolution = make_uint2(width, height);
    double compile_ms = 0.0;

    Callable tea = [](UInt v0, UInt v1) noexcept {
        UInt s0 = def(0u);
        for (uint n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4u) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4u) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    Kernel2D make_sampler_kernel = [&](ImageUInt seed_image) noexcept {
        set_name("bench_rt_sdf_make_sampler");
        UInt2 p = dispatch_id().xy();
        seed_image.write(p, make_uint4(tea(p.x, p.y)));
    };

    // Analytic disk light: returns the distance along the ray or `inf`.
    Callable intersect_light = [](Float3 pos, Float3 d) noexcept {
        Float cos_w = dot(-d, light_normal);
        Float dist = dot(d, light_pos - pos);
        Float D = dist / cos_w;
        Float dist_to_center = distance_squared(light_pos, pos + D * d);
        Bool valid = cos_w > 0.0f & dist > 0.0f & dist_to_center < light_radius * light_radius;
        return ite(valid, D, inf);
    };

    Callable rand = [](UInt &state) noexcept {
        constexpr uint lcg_a = 1664525u;
        constexpr uint lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state) / cast<float>(std::numeric_limits<uint>::max());
    };

    // Cosine distributed direction around `n`.
    Callable out_dir = [&rand](Float3 n, UInt &seed) noexcept {
        Float3 u = ite(
            abs(n.y) < 1.0f - eps,
            normalize(cross(n, make_float3(0.0f, 1.0f, 0.0f))),
            make_float3(1.f, 0.f, 0.f));
        Float3 v = cross(n, u);
        Float phi = 2.0f * std::numbers::pi_v<float> * rand(seed);
        Float ay = sqrt(rand(seed));
        Float ax = sqrt(1.0f - ay * ay);
        return ax * (cos(phi) * u + sin(phi) * v) + ay * n;
    };

    Callable make_nested = [](Float f) noexcept {
        static constexpr float freq = 40.0f;
        f *= freq;
        f = ite(f < 0.f, ite(cast<int>(f) % 2 == 0, 1.f - fract(f), fract(f)), f);
        return (f - 0.2f) * (1.0f / freq);
    };

    // Sphere, box, cylinder and a wavy wall.
    Callable sdf = [&make_nested](Float3 o) noexcept {
        Float wall = min(o.y + 0.1f, o.z + 0.4f);
        Float sphere = distance(o, make_float3(0.0f, 0.35f, 0.0f)) - 0.36f;
        Float3 q = abs(o - make_float3(0.8f, 0.3f, 0.0f)) - 0.3f;
        Float box = length(max(q, 0.0f)) + min(max(max(q.x, q.y), q.z), 0.0f);
        Float3 O = o - make_float3(-0.8f, 0.3f, 0.0f);
        Float2 d = make_float2(length(make_float2(O.x, O.z)) - 0.3f, abs(O.y) - 0.3f);
        Float cylinder = min(max(d.x, d.y), 0.0f) + length(max(d, 0.0f));
        Float geometry = make_nested(min(min(sphere, box), cylinder));
        Float g = max(geometry, -(0.32f - (o.y * 0.6f + o.z * 0.8f)));
        return min(wall, g);
    };

    Callable ray_march = [&sdf](Float3 p, Float3 d) noexcept {
        Float dist = def(0.0f);
        $for (j, max_march_steps) {
            Float s = sdf(p + dist * d);
            $if (s <= 1e-6f | dist >= inf) { $break; };
            dist += s;
        };
        return min(dist, inf);
    };

    // 3-tap forward difference normal.
    Callable sdf_normal = [&sdf](Float3 p) noexcept {
        static constexpr float d = 1e-3f;
        Float3 n = def(make_float3());
        Float sdf_center = sdf(p);
        for (uint i = 0u; i < 3u; i++) {
            Float3 inc = p;
            inc[i] += d;
            n[i] = (1.0f / d) * (sdf(inc) - sdf_center);
        }
        return normalize(n);
    };

    Callable next_hit = [&ray_march, &sdf_normal](Float &closest, Float3 &normal,
                                                  Float3 &c, Float3 pos, Float3 d) noexcept {
        closest = inf;
        normal = make_float3();
        c = make_float3();
        Float ray_march_dist = ray_march(pos, d);
        $if (ray_march_dist < min(dist_limit, closest)) {
            closest = ray_march_dist;
            Float3 hit_pos = pos + d * closest;
            normal = sdf_normal(hit_pos);
            Int t = cast<int>((hit_pos.x + 10.0f) * 1.1f + 0.5f) % 3;
            c = make_float3(0.4f) + make_float3(0.3f, 0.2f, 0.3f) * ite(t == make_int3(0, 1, 2), 1.0f, 0.0f);
        };
    };

    // One dispatch renders `spp` samples per pixel: the sample loop lives inside
    // the kernel so the timed region is a single dispatch.
    Kernel2D render_kernel = [&](ImageUInt seed_image, ImageFloat output_image,
                                 UInt samples_per_pixel) noexcept {
        set_name("bench_rt_sdf_raymarch");
        set_block_size(16u, 8u, 1u);
        UInt2 coord = dispatch_id().xy();
        UInt2 grid = dispatch_size().xy();
        Float2 resolution_f = make_float2(grid);
        Float aspect_ratio = resolution_f.x / resolution_f.y;
        UInt seed = seed_image.read(coord).x;
        Float3 sum_color = def(make_float3(0.0f));

        $for (sample, samples_per_pixel) {
            Float ux = rand(seed);
            Float uy = rand(seed);
            Float2 uv = make_float2(cast<float>(coord.x) + ux,
                                    cast<float>(grid.y - 1u - coord.y) + uy);
            Float3 d = normalize(make_float3(
                2.0f * fov * uv / resolution_f.y -
                    fov * make_float2(aspect_ratio, 1.0f) - 1e-5f,
                -1.0f));
            Float3 pos = def(camera_pos);
            Float3 throughput = def(make_float3(1.0f));
            Float hit_light = def(0.0f);
            $for (depth, max_ray_depth) {
                Float closest = def(0.0f);
                Float3 normal = def(make_float3());
                Float3 c = def(make_float3());
                next_hit(closest, normal, c, pos, d);
                Float dist_to_light = intersect_light(pos, d);
                $if (dist_to_light < closest) {
                    hit_light = 1.0f;
                    $break;
                };
                $if (length_squared(normal) == 0.0f) { $break; };
                Float3 hit_pos = pos + closest * d;
                d = out_dir(normal, seed);
                pos = hit_pos + 1e-4f * d;
                throughput *= c;
            };
            sum_color += throughput * hit_light;
        };

        seed_image.write(coord, make_uint4(seed));
        output_image.write(coord, make_float4(sum_color / samples_per_pixel.cast<float>(), 1.0f));
    };

    auto make_sampler_shader = compile_uncached(
        device, make_sampler_kernel, "bench_rt_sdf_make_sampler", cfg, compile_ms);
    auto render_shader = compile_uncached(
        device, render_kernel, "bench_rt_sdf_raymarch", cfg, compile_ms);

    Image<uint> seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);
    Image<float> output_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    luisa::vector<float> output_host(size_t{width} * size_t{height} * 4u);

    stream << make_sampler_shader(seed_image).dispatch(resolution);

    static constexpr double work_units = double(1280ull * 720ull * 16ull);
    auto result = run_timed_bench(
        "raytracing", name, stream, cfg, compile_ms, work_units, "spp_px/s",
        [&] {
            stream << render_shader(seed_image, output_image, spp).dispatch(resolution);
        });

    stream << output_image.copy_to(luisa::span{output_host}) << synchronize();
    auto non_finite = count_non_finite_rgba(luisa::span<const float>{output_host});
    auto luma = luma_from_rgba32f(luisa::span<const float>{output_host});
    apply_image_check(cfg, result, non_finite, luma, 0.02, 0.98);

    emit(results, std::move(result));
}

// ---------------------------------------------------------------------------
// 4. 64^3 voxel grid traversal with a 3D DDA.
// ---------------------------------------------------------------------------

void bench_voxel_dda(compute::Device &device,
                     compute::Stream &stream,
                     const BenchConfig &cfg,
                     std::vector<BenchResult> &results) {

    static constexpr auto name = "rt_voxel_dda";
    if (!should_run(cfg, "raytracing", name)) { return; }
    static constexpr uint32_t width = 1024u;
    static constexpr uint32_t height = 1024u;
    static constexpr uint32_t grid_size = 64u;
    static constexpr uint32_t total_voxels = grid_size * grid_size * grid_size;
    static constexpr uint32_t max_steps = 256u;
    static constexpr uint32_t frames_per_iteration = 4u;
    static constexpr uint2 resolution = make_uint2(width, height);
    double compile_ms = 0.0;

    // ---- procedural voxel scene (identical to the interactive example) ----
    luisa::vector<uint32_t> host_voxels(total_voxels, 0u);
    auto grid_dims = static_cast<int>(grid_size);
    auto set_voxel = [&](int x, int y, int z, uint32_t type) noexcept {
        if (x < 0 || y < 0 || z < 0 || x >= grid_dims || y >= grid_dims || z >= grid_dims) { return; }
        auto index = (static_cast<size_t>(z) * grid_size + static_cast<size_t>(y)) * grid_size +
                     static_cast<size_t>(x);
        host_voxels[index] = type;
    };
    for (int x = 0; x < grid_dims; x++) {
        for (int z = 0; z < grid_dims; z++) {
            // Base terrain from two sine waves plus a low frequency term.
            auto terrain_height = 10 + static_cast<int>(
                                           std::sin(static_cast<float>(x) * 0.15f) * 4.0f +
                                           std::cos(static_cast<float>(z) * 0.15f) * 4.0f +
                                           std::sin(static_cast<float>(x) * 0.05f +
                                                    static_cast<float>(z) * 0.05f) * 8.0f);
            terrain_height = std::min(std::max(terrain_height, 2), grid_dims - 2);
            for (int y = 0; y < terrain_height; y++) {
                uint32_t type = 1u;// dirt
                if (y >= terrain_height - 2) { type = 2u; }// grass
                if (y < 5) { type = 3u; }                  // stone
                set_voxel(x, y, z, type);
            }
            // Deterministically scattered trees.
            if (((x * 7 + z * 13) % 47) == 0 && terrain_height > 15) {
                auto tree_height = 5 + (x + z) % 4;
                for (int h = 0; h < tree_height; h++) {
                    set_voxel(x, terrain_height + h, z, 4u);// wood
                }
                for (int lx = -2; lx <= 2; lx++) {
                    for (int ly = -1; ly <= 2; ly++) {
                        for (int lz = -2; lz <= 2; lz++) {
                            auto manhattan = (lx < 0 ? -lx : lx) + (ly < 0 ? -ly : ly) + (lz < 0 ? -lz : lz);
                            if (manhattan <= 3) {
                                set_voxel(x + lx, terrain_height + tree_height + ly, z + lz, 5u);// leaves
                            }
                        }
                    }
                }
            }
        }
    }
    // A floating "magic orb" in the middle of the grid.
    int sphere_center = 32;
    int sphere_radius = 8;
    for (int x = -sphere_radius; x <= sphere_radius; x++) {
        for (int y = -sphere_radius; y <= sphere_radius; y++) {
            for (int z = -sphere_radius; z <= sphere_radius; z++) {
                if (x * x + y * y + z * z <= sphere_radius * sphere_radius) {
                    set_voxel(sphere_center + x, sphere_center + 13 + y, sphere_center + z, 6u);
                }
            }
        }
    }

    Buffer<uint32_t> voxel_grid = device.create_buffer<uint32_t>(total_voxels);
    stream << voxel_grid.copy_from(luisa::span{host_voxels}) << synchronize();

    // ---- DDA traversal kernel ----
    Kernel2D render_kernel = [&](ImageFloat image, Float3 cam_pos,
                                 Float2 cam_rot, BufferVar<uint32_t> voxels) noexcept {
        set_name("bench_rt_voxel_dda");
        set_block_size(16u, 16u, 1u);
        UInt2 uv = dispatch_id().xy();
        UInt2 size = dispatch_size().xy();

        // Normalized device coordinates (Y flipped so +Y is up on screen).
        Float2 ndc = (make_float2(uv) / make_float2(size)) * 2.0f - 1.0f;
        ndc.y = -ndc.y;

        Float aspect = cast<float>(size.x) / cast<float>(size.y);
        constexpr float fov = 0.8f;
        Float3 ray_dir_cam = normalize(make_float3(
            ndc.x * aspect * fov, ndc.y * fov, -1.0f));

        // Apply the camera rotation (yaw around Y, then pitch around X).
        Float cos_y = cos(cam_rot.x);
        Float sin_y = sin(cam_rot.x);
        Float cos_x = cos(cam_rot.y);
        Float sin_x = sin(cam_rot.y);
        Float x1 = ray_dir_cam.x * cos_y - ray_dir_cam.z * sin_y;
        Float z1 = ray_dir_cam.x * sin_y + ray_dir_cam.z * cos_y;
        Float y2 = ray_dir_cam.y * cos_x - z1 * sin_x;
        Float z2 = ray_dir_cam.y * sin_x + z1 * cos_x;
        Float3 ray_dir = normalize(make_float3(x1, y2, z2));
        Float3 ray_origin = cam_pos;

        // Grid bounds and slab intersection.
        Float3 grid_min = make_float3(0.0f);
        Float3 grid_max = make_float3(cast<float>(grid_size));
        Float3 tmin = (grid_min - ray_origin) / ray_dir;
        Float3 tmax = (grid_max - ray_origin) / ray_dir;
        Float3 tmin_new = min(tmin, tmax);
        Float3 tmax_new = max(tmin, tmax);
        Float t_enter = max(max(tmin_new.x, tmin_new.y), tmin_new.z);
        Float t_exit = min(min(tmax_new.x, tmax_new.y), tmax_new.z);

        Float3 sky_color = lerp(
            make_float3(0.5f, 0.7f, 1.0f),
            make_float3(0.1f, 0.2f, 0.4f),
            ray_dir.y * 0.5f + 0.5f);
        Float3 final_color = sky_color;

        $if (t_enter <= t_exit & t_exit > 0.0f) {
            // Start just inside the grid to avoid self-intersection.
            Float t = ite(t_enter > 0.0f, t_enter, 0.0f);
            Float3 p = ray_origin + ray_dir * (t + 0.001f);
            Float vx = floor(p.x);
            Float vy = floor(p.y);
            Float vz = floor(p.z);
            Int step_x = ite(ray_dir.x > 0.0f, 1, -1);
            Int step_y = ite(ray_dir.y > 0.0f, 1, -1);
            Int step_z = ite(ray_dir.z > 0.0f, 1, -1);
            Float delta_tx = abs(1.0f / ray_dir.x);
            Float delta_ty = abs(1.0f / ray_dir.y);
            Float delta_tz = abs(1.0f / ray_dir.z);
            Float next_tx = ite(ray_dir.x > 0.0f,
                                (vx + 1.0f - p.x) / ray_dir.x,
                                (vx - p.x) / ray_dir.x);
            Float next_ty = ite(ray_dir.y > 0.0f,
                                (vy + 1.0f - p.y) / ray_dir.y,
                                (vy - p.y) / ray_dir.y);
            Float next_tz = ite(ray_dir.z > 0.0f,
                                (vz + 1.0f - p.z) / ray_dir.z,
                                (vz - p.z) / ray_dir.z);

            Bool hit = def(false);
            Float3 hit_color = def(make_float3(1.0f));
            UInt hit_step = def(0u);

            $for (step, max_steps) {
                Int ix = cast<int>(vx);
                Int iy = cast<int>(vy);
                Int iz = cast<int>(vz);
                $if ((ix < 0) | (ix >= static_cast<int>(grid_size)) |
                     (iy < 0) | (iy >= static_cast<int>(grid_size)) |
                     (iz < 0) | (iz >= static_cast<int>(grid_size))) {
                    $break;
                };
                UInt voxel_index = cast<uint>((iz * static_cast<int>(grid_size) + iy) *
                                                  static_cast<int>(grid_size) + ix);
                UInt voxel_type = voxels.read(voxel_index);
                $if (voxel_type > 0u) {
                    // Material palette by voxel type.
                    $if (voxel_type == 1u) {
                        hit_color = make_float3(0.6f, 0.4f, 0.3f);// dirt
                    }
                    $elif (voxel_type == 2u) {
                        hit_color = make_float3(0.2f, 0.7f, 0.2f);// grass
                    }
                    $elif (voxel_type == 3u) {
                        hit_color = make_float3(0.5f, 0.5f, 0.5f);// stone
                    }
                    $elif (voxel_type == 4u) {
                        hit_color = make_float3(0.4f, 0.25f, 0.1f);// wood
                    }
                    $elif (voxel_type == 5u) {
                        hit_color = make_float3(0.1f, 0.5f, 0.1f);// leaves
                    }
                    $else {
                        hit_color = make_float3(0.8f, 0.3f, 0.9f);// magic orb
                    };
                    hit_step = step;
                    hit = true;
                    $break;
                };
                // Advance to the next voxel boundary.
                $if (next_tx < next_ty & next_tx < next_tz) {
                    vx = vx + cast<float>(step_x);
                    next_tx = next_tx + delta_tx;
                }
                $else {
                    $if (next_ty < next_tz) {
                        vy = vy + cast<float>(step_y);
                        next_ty = next_ty + delta_ty;
                    }
                    $else {
                        vz = vz + cast<float>(step_z);
                        next_tz = next_tz + delta_tz;
                    };
                };
                $if (t > 200.0f) { $break; };
            };

            $if (hit) {
                // Cheaper ambient occlusion the further the ray walked.
                Float ao = 1.0f - (cast<float>(hit_step) / cast<float>(max_steps)) * 0.5f;
                Float3 light_dir = normalize(make_float3(0.5f, 1.0f, 0.3f));
                Float3 normal = normalize(make_float3(
                    ite(next_tx < next_ty & next_tx < next_tz, cast<float>(-step_x), 0.0f),
                    ite((next_ty <= next_tx | next_tx >= next_tz) & next_ty < next_tz, cast<float>(-step_y), 0.0f),
                    ite(next_tz <= next_ty & (next_tx >= next_ty | next_tz <= next_tx), cast<float>(-step_z), 0.0f)));
                Float diff = max(dot(normal, light_dir), 0.0f);
                final_color = hit_color * (0.3f + 0.7f * diff) * ao;
            };
        };

        image.write(uv, make_float4(final_color, 1.0f));
    };

    auto render_shader = compile_uncached(
        device, render_kernel, "bench_rt_voxel_dda", cfg, compile_ms);

    Image<float> display = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    luisa::vector<float> display_host(size_t{width} * size_t{height} * 4u);

    float3 cam_pos = make_float3(32.0f, 40.0f, 80.0f);
    static constexpr float2 base_cam_rot = make_float2(0.0f, -0.3f);

    static constexpr double work_units = double(1024ull * 1024ull * 4ull);
    auto result = run_timed_bench(
        "raytracing", name, stream, cfg, compile_ms, work_units, "ray/s",
        [&] {
            CommandList command_list = CommandList::create(frames_per_iteration);
            for (auto frame = 0u; frame < frames_per_iteration; frame++) {
                // Nudge the camera every frame, like the interactive viewer.
                auto cam_rot = make_float2(
                    base_cam_rot.x + 0.01f * static_cast<float>(frame), base_cam_rot.y);
                command_list << render_shader(display, cam_pos, cam_rot, voxel_grid)
                                    .dispatch(resolution);
            }
            stream << command_list.commit();
        });

    stream << display.copy_to(luisa::span{display_host}) << synchronize();
    auto non_finite = count_non_finite_rgba(luisa::span<const float>{display_host});
    auto luma = luma_from_rgba32f(luisa::span<const float>{display_host});
    apply_image_check(cfg, result, non_finite, luma, 0.02, 0.98);

    emit(results, std::move(result));
}

}// namespace

// Entry point of the "raytracing" category. Each benchmark gates itself with
// `should_run(cfg, "raytracing", name)`, so filtered runs skip both the scene
// setup and the shader compilation of the entries they do not select.
void run_raytracing_benchmarks(
    compute::Device &device,
    compute::Stream &stream,
    const BenchConfig &cfg,
    std::vector<BenchResult> &results) {
    bench_cornell_path_tracing(device, stream, cfg, results);
    bench_nested_callable_path_tracing(device, stream, cfg, results);
    bench_sdf_raymarch(device, stream, cfg, results);
    bench_voxel_dda(device, stream, cfg, results);
}

}// namespace luisa::bench
