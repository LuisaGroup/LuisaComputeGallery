// Physics benchmark group of the gallery benchmark suite.
//
// Three device-resident physics workloads, all measured through the shared
// harness in bench_common.h (uncached compile, one recorded batch per timed
// iteration, host-observed wall time around one submit plus one fence):
//
// * "physics_mpm3d": 3D MLS-MPM (Neo-Hookean) with a 64^3 grid and 65536
//   particles, 10 substeps per timed iteration. One substep is the classic
//   four-dispatch pipeline: clear_grid -> point_to_grid (27-cell quadratic
//   B-spline scatter through FP32 atomics) -> simulate_grid -> grid_to_point
//   (27-cell gather plus the float3x3 outer product that updates the affine
//   momentum C). The particles are re-seeded before the warmup so the timed
//   run never depends on the verification run. Work unit: one particle
//   substep.
// * "physics_nbody": all-pairs gravitational N-Body with 8192 particles,
//   shared-memory tiled (256 particles per tile, two block barriers per
//   tile) with Euler integration and a deterministic depth-resolved 5x5
//   splat: an unsigned atomic-min winner buffer is cleared, rasterized and
//   then resolved onto a 1024^2 BYTE4 image. Two simulation+render steps per
//   timed iteration. Work unit: one particle-particle interaction.
// * "physics_wave_eq": 2D finite-difference wave equation on a 1024^2
//   FLOAT1 ping-pong pair seeded with deterministic Gaussian droplets, 16
//   substeps per timed iteration plus the gradient/caustic render pass into
//   a FLOAT4 image. Work unit: one cell update.
//
// Conventions: no exceptions (LUISA_ASSERT for invariants, verification results
// are *reported* instead of aborting the suite), luisa::span for every
// copy_from/copy_to, LUISA_STRUCT at namespace scope, `&`/`|` for DSL Bools and
// every kernel compiled through compile_uncached() so the compile cost lands in
// BenchResult::compile_ms.

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "bench_common.h"

// `float3` in the struct below needs the luisa namespace: the using
// declarations must precede the LUISA_STRUCT reflection (same layout as the
// simulation examples).
using namespace luisa;
using namespace luisa::compute;

// LUISA_STRUCT expands `namespace luisa::compute { ... }` blocks of its own, so
// it must be used at namespace scope.  The N-Body particle therefore lives at
// global scope, with a bench-specific name: a unity build merges the benchmark
// translation units, so a generic name such as `Particle` could collide with a
// sibling file.
struct PhysicsBenchParticle {
    float3 position;
    float3 velocity;
    float mass;
    float pad[3];
};
LUISA_STRUCT(PhysicsBenchParticle, position, velocity, mass, pad) {};

namespace luisa::bench {
// Benchmark-private helpers live in a named namespace with an anonymous
// namespace inside it: a unity build concatenates the benchmark TUs, and
// generic helper names (sqr, is_finite, ...) would otherwise collide with a
// sibling file.
namespace physics_bench {
namespace {

using NBodyParticle = PhysicsBenchParticle;

/// Outcome of the host-side check of one benchmark.
struct PhysicsCheck {
    bool passed{true};
    luisa::string note;
};

[[nodiscard]] inline bool is_finite(const luisa::float3 &v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

template<typename T>
[[nodiscard]] inline auto sqr(T &&x) noexcept {
    return x * x;
}

// Vector conversions are spelled component-wise: the native CUDA backend maps
// make_int3(int, int, int) to lc_make_int3, while a vector-to-vector
// cast<T>(vector) becomes a static_cast between lc_* vector types that have no
// cross-type constructor.
[[nodiscard]] inline auto to_int3(Float3 v) noexcept {
    return make_int3(cast<int>(v.x), cast<int>(v.y), cast<int>(v.z));
}

[[nodiscard]] inline auto to_float3(Int3 v) noexcept {
    return make_float3(cast<float>(v.x), cast<float>(v.y), cast<float>(v.z));
}

// ---------------------------------------------------------------------------
// 1. MLS-MPM in 3D
// ---------------------------------------------------------------------------
void bench_mpm3d(compute::Device &device, compute::Stream &stream,
                 const BenchConfig &cfg, std::vector<BenchResult> &results) {

    // Simulation parameters, identical to the interactive 3D MPM example.
    static constexpr uint kGridSize = 64u;
    static constexpr uint kGridCells = kGridSize * kGridSize * kGridSize;
    static constexpr uint kParticleCount = 65536u;
    static constexpr uint kSubsteps = 10u;
    static constexpr float kDx = 1.0f / static_cast<float>(kGridSize);
    static constexpr float kDt = 8.0e-5f;
    static constexpr float kParticleVolume = (0.5f * kDx) * (0.5f * kDx) * (0.5f * kDx);
    // Particle density is 1, so the particle mass is its volume.
    static constexpr float kParticleMass = kParticleVolume;
    static constexpr float kGravity = 9.8f;
    static constexpr float kYoungsModulus = 400.0f;
    static constexpr int kBoundary = 3;

    // Linear index of a grid cell.  Dispatch coordinates are always inside the
    // grid, so they only need the flat-index arithmetic; particle-derived cells
    // are clamped to the boundary cell, exactly like the interactive example,
    // so the atomic target is always valid.  Both forms stay scalar: a
    // vector-to-vector cast<int3>(uint3) is rejected by the native CUDA backend,
    // whose lc_* vectors have no cross-type constructors.
    auto dispatch_cell_index = [](UInt3 cell) noexcept {
        return cell.x + cell.y * kGridSize + cell.z * (kGridSize * kGridSize);
    };

    auto particle_cell_index = [](Int3 cell) noexcept {
        Int3 p = clamp(cell, 0, static_cast<int>(kGridSize) - 1);
        return cast<uint>(p.x + p.y * static_cast<int>(kGridSize) +
                          p.z * static_cast<int>(kGridSize * kGridSize));
    };

    // outer(a, b) = a b^T (the affine momentum update of MLS-MPM).
    auto outer_product = [](Float3 a, Float3 b) noexcept {
        return make_float3x3(
            make_float3(a[0] * b[0], a[1] * b[0], a[2] * b[0]),
            make_float3(a[0] * b[1], a[1] * b[1], a[2] * b[1]),
            make_float3(a[0] * b[2], a[1] * b[2], a[2] * b[2]));
    };

    auto trace_of = [](Float3x3 m) noexcept {
        return m[0][0] + m[1][1] + m[2][2];
    };

    // Grid state: (vx, vy, vz, mass) per cell.
    Kernel3D clear_grid = [&](BufferFloat grid) noexcept {
        set_block_size(8u, 8u, 1u);
        UInt cell = dispatch_cell_index(dispatch_id().xyz());
        for (uint c = 0u; c < 4u; ++c) {
            grid.write(cell * 4u + c, 0.0f);
        }
    };

    // Particle-to-grid transfer: 27-cell quadratic B-spline scatter with FP32
    // atomic accumulation.
    Kernel1D point_to_grid = [&](BufferFloat3 x, BufferFloat3 v,
                                 BufferFloat3x3 affine, BufferFloat jvp,
                                 BufferFloat grid) noexcept {
        set_block_size(64u, 1u, 1u);
        UInt p = dispatch_x();
        Float3 grid_position = x.read(p) / kDx;
        Int3 base = to_int3(grid_position - 0.5f);
        Float3 fx = grid_position - to_float3(base);

        // Quadratic B-spline weights, one per axis.
        std::array<Float3, 3u> weights{
            0.5f * sqr(1.5f - fx),
            0.75f - sqr(fx - 1.0f),
            0.5f * sqr(fx - 0.5f)};

        // Neo-Hookean stress plus the affine momentum of the particle.
        Float stress = -4.0f * kDt * kYoungsModulus * kParticleVolume *
                       (jvp.read(p) - 1.0f) / (kDx * kDx);
        Float3x3 affine_term =
            make_float3x3(stress, 0.0f, 0.0f,
                          0.0f, stress, 0.0f,
                          0.0f, 0.0f, stress) +
            kParticleMass * affine.read(p);
        Float3 particle_velocity = v.read(p);

        $for (neighbor, 27u) {
            UInt i = neighbor % 3u;
            UInt j = (neighbor / 3u) % 3u;
            UInt k = neighbor / 9u;
            Int3 offset = make_int3(cast<int>(i), cast<int>(j), cast<int>(k));
            Float3 dpos = (to_float3(offset) - fx) * kDx;
            Float wi = ite(i == 0u, weights[0u].x,
                           ite(i == 1u, weights[1u].x, weights[2u].x));
            Float wj = ite(j == 0u, weights[0u].y,
                           ite(j == 1u, weights[1u].y, weights[2u].y));
            Float wk = ite(k == 0u, weights[0u].z,
                           ite(k == 1u, weights[1u].z, weights[2u].z));
            Float weight = wi * wj * wk;
            Float3 momentum =
                weight * (kParticleMass * particle_velocity + affine_term * dpos);
            UInt cell = particle_cell_index(base + offset);
            for (uint c = 0u; c < 3u; ++c) {
                grid.atomic(cell * 4u + c).fetch_add(momentum[c]);
            }
            grid.atomic(cell * 4u + 3u).fetch_add(weight * kParticleMass);
        };
    };

    // Grid update: normalize momentum by mass, apply gravity and make the six
    // boundary layers sticky.
    Kernel3D simulate_grid = [&](BufferFloat grid) noexcept {
        set_block_size(8u, 8u, 1u);
        UInt3 coord = dispatch_id().xyz();
        UInt cell = dispatch_cell_index(coord);
        Float4 velocity_and_mass = make_float4(grid.read(cell * 4u),
                                               grid.read(cell * 4u + 1u),
                                               grid.read(cell * 4u + 2u),
                                               grid.read(cell * 4u + 3u));
        Float3 velocity = velocity_and_mass.xyz();
        Float mass = velocity_and_mass.w;
        velocity = ite(mass > 0.0f, velocity / mass, velocity);
        velocity.y -= kDt * kGravity;
        // A velocity component that points into a wall cell is cancelled.
        auto sticky = [](UInt c, Float value) noexcept {
            Bool into_low = (c < static_cast<uint>(kBoundary)) & (value < 0.0f);
            Bool into_high = (c > kGridSize - static_cast<uint>(kBoundary)) & (value > 0.0f);
            return ite(into_low | into_high, 0.0f, value);
        };
        velocity = make_float3(sticky(coord.x, velocity.x),
                               sticky(coord.y, velocity.y),
                               sticky(coord.z, velocity.z));
        grid.write(cell * 4u, velocity.x);
        grid.write(cell * 4u + 1u, velocity.y);
        grid.write(cell * 4u + 2u, velocity.z);
        grid.write(cell * 4u + 3u, mass);
    };

    // Grid-to-particle transfer: 27-cell gather plus the affine update.
    Kernel1D grid_to_point = [&](BufferFloat3 x, BufferFloat3 v,
                                 BufferFloat3x3 affine, BufferFloat jvp,
                                 BufferFloat grid) noexcept {
        set_block_size(64u, 1u, 1u);
        UInt p = dispatch_x();
        Float3 grid_position = x.read(p) / kDx;
        Int3 base = to_int3(grid_position - 0.5f);
        Float3 fx = grid_position - to_float3(base);
        std::array<Float3, 3u> weights{
            0.5f * sqr(1.5f - fx),
            0.75f - sqr(fx - 1.0f),
            0.5f * sqr(fx - 0.5f)};
        Float3 new_velocity = def(make_float3(0.0f));
        Float3x3 new_affine = def(make_float3x3(0.0f));
        // The 27-cell stencil is unrolled at trace time, so the weight lookup
        // stays a plain array index instead of an ite chain.
        for (uint n = 0u; n < 27u; ++n) {
            // Host-side int3 keeps the offsets compile-time constants.
            int3 offset = make_int3(static_cast<int>(n % 3u),
                                    static_cast<int>((n / 3u) % 3u),
                                    static_cast<int>(n / 9u));
            Float3 dpos = (make_float3(offset) - fx) * kDx;
            Float weight = weights[static_cast<size_t>(offset.x)].x *
                           weights[static_cast<size_t>(offset.y)].y *
                           weights[static_cast<size_t>(offset.z)].z;
            UInt cell = particle_cell_index(base + offset);
            Float3 cell_velocity = make_float3(grid.read(cell * 4u),
                                               grid.read(cell * 4u + 1u),
                                               grid.read(cell * 4u + 2u));
            new_velocity += weight * cell_velocity;
            new_affine = new_affine + (4.0f * weight / (kDx * kDx)) *
                                          outer_product(cell_velocity, dpos);
        }
        v.write(p, new_velocity);
        x.write(p, x.read(p) + new_velocity * kDt);
        jvp.write(p, jvp.read(p) * (1.0f + kDt * trace_of(new_affine)));
        affine.write(p, new_affine);
    };

    double compile_ms = 0.0;
    auto clear_grid_shader = compile_uncached(
        device, clear_grid, "bench_physics_mpm3d_clear_grid", cfg, compile_ms);
    auto point_to_grid_shader = compile_uncached(
        device, point_to_grid, "bench_physics_mpm3d_point_to_grid", cfg, compile_ms);
    auto simulate_grid_shader = compile_uncached(
        device, simulate_grid, "bench_physics_mpm3d_simulate_grid", cfg, compile_ms);
    auto grid_to_point_shader = compile_uncached(
        device, grid_to_point, "bench_physics_mpm3d_grid_to_point", cfg, compile_ms);

    // Device state.
    Buffer<float3> x = device.create_buffer<float3>(kParticleCount);
    Buffer<float3> v = device.create_buffer<float3>(kParticleCount);
    Buffer<float3x3> affine = device.create_buffer<float3x3>(kParticleCount);
    Buffer<float> jvp = device.create_buffer<float>(kParticleCount);
    Buffer<float> grid = device.create_buffer<float>(kGridCells * 4u);

    // Deterministic initial state: a 0.4-wide cube in the middle of the domain.
    luisa::vector<float3> host_x(kParticleCount);
    luisa::vector<float3> host_v(kParticleCount, make_float3(0.0f));
    luisa::vector<float3x3> host_affine(kParticleCount, make_float3x3(0.0f));
    luisa::vector<float> host_jvp(kParticleCount, 1.0f);
    uint64_t seed = 0x0123456789abcdefull;
    for (uint i = 0u; i < kParticleCount; ++i) {
        host_x[i] = make_float3(prng_f32(seed, 0.2f, 0.6f),
                                prng_f32(seed, 0.2f, 0.6f),
                                prng_f32(seed, 0.2f, 0.6f));
    }

    auto reset_particles = [&]() noexcept {
        stream << x.copy_from(luisa::span{host_x})
               << v.copy_from(luisa::span{host_v})
               << affine.copy_from(luisa::span{host_affine})
               << jvp.copy_from(luisa::span{host_jvp})
               << synchronize();
    };

    // One timed iteration: ten MLS-MPM substeps recorded into a single
    // CommandList.
    auto dispatch_fn = [&]() noexcept {
        CommandList cmd_list;
        for (uint substep = 0u; substep < kSubsteps; ++substep) {
            cmd_list << clear_grid_shader(grid).dispatch(kGridSize, kGridSize, kGridSize)
                     << point_to_grid_shader(x, v, affine, jvp, grid).dispatch(kParticleCount)
                     << simulate_grid_shader(grid).dispatch(kGridSize, kGridSize, kGridSize)
                     << grid_to_point_shader(x, v, affine, jvp, grid).dispatch(kParticleCount);
        }
        stream << cmd_list.commit();
    };

    // --- verification: particles must stay finite and inside the unit cube ---
    PhysicsCheck check;
    if (cfg.verify) {
        reset_particles();
        for (uint i = 0u; i < 2u; ++i) {
            dispatch_fn();
        }
        stream << synchronize();
        luisa::vector<float3> checked_x(kParticleCount);
        luisa::vector<float3> checked_v(kParticleCount);
        stream << x.copy_to(luisa::span{checked_x})
               << v.copy_to(luisa::span{checked_v})
               << synchronize();
        size_t bad = 0u;
        size_t escaped = 0u;
        size_t first_bad = kParticleCount;
        float max_speed = 0.0f;
        for (uint i = 0u; i < kParticleCount; ++i) {
            const auto &pi = checked_x[i];
            const auto &vi = checked_v[i];
            bool finite = is_finite(pi) && is_finite(vi);
            bool inside = pi.x >= 0.0f && pi.x <= 1.0f &&
                          pi.y >= 0.0f && pi.y <= 1.0f &&
                          pi.z >= 0.0f && pi.z <= 1.0f;
            if (!finite) {
                bad++;
            }
            if (!inside) {
                escaped++;
            }
            if ((!finite || !inside) && first_bad == kParticleCount) {
                first_bad = i;
            }
            if (finite) {
                max_speed = std::max(max_speed, std::sqrt(vi.x * vi.x +
                                                          vi.y * vi.y +
                                                          vi.z * vi.z));
            }
        }
        check.passed = bad == 0u && escaped == 0u;
        if (!check.passed) {
            check.note = luisa::format(
                "{} non-finite and {} out-of-[0,1]^3 particle states out of {} "
                "(first bad index {}, max |v| {:.6e})",
                bad, escaped, kParticleCount,
                first_bad == kParticleCount ? 0u : first_bad, max_speed);
            LUISA_WARNING("physics_mpm3d: {}", check.note);
        } else {
            check.note = luisa::format(
                "{} particle positions/velocities finite, all inside [0,1]^3 "
                "(max |v| {:.6e})",
                kParticleCount, max_speed);
        }
        // The verification run advanced the state; re-seed so the timed run
        // starts from the same bounded cube as a fresh process would.
        reset_particles();
    } else {
        reset_particles();
        check.note = "verification skipped (verify disabled)";
    }

    auto result = run_timed_bench(
        "physics", "physics_mpm3d", stream, cfg, compile_ms,
        static_cast<double>(kParticleCount) * static_cast<double>(kSubsteps),
        "pt_step/s", dispatch_fn);
    result.verified = check.passed;
    result.check_note = std::move(check.note);
    results.emplace_back(std::move(result));
}

// ---------------------------------------------------------------------------
// 2. All-pairs gravitational N-Body with depth-resolved splatting
// ---------------------------------------------------------------------------
void bench_nbody(compute::Device &device, compute::Stream &stream,
                 const BenchConfig &cfg, std::vector<BenchResult> &results) {

    static constexpr uint kParticleCount = 8192u;
    static constexpr uint kTileSize = 256u;  // must be a power of two
    static constexpr uint kTileCount = kParticleCount / kTileSize;
    static constexpr uint kStepsPerIteration = 2u;
    static constexpr uint kImageSize = 1024u;
    static constexpr uint kPixelCount = kImageSize * kImageSize;
    static constexpr float kDt = 5.0e-4f;
    static constexpr float kSoftening = 0.05f;
    static constexpr float kGravity = 0.5f;
    static constexpr float kDamping = 0.999f;
    static constexpr float kCentralMass = 500.0f;
    static constexpr int kGlowRadius = 2;    // 5x5 splat footprint
    // kParticleCount is a power of two, so the index mask is a plain mask.
    static constexpr uint kPixelIndexMask = kParticleCount - 1u;
    static constexpr uint kDepthMask = ~kPixelIndexMask;
    static constexpr uint kInvalidWinner = 0xffffffffu;
    static constexpr float kMinVisibleDistance = 0.01f;
    static constexpr float kViewDistance = 4.0f;
    static constexpr float kProjectionScale = 1.5f;
    static constexpr float kRotX = 0.3f;
    static constexpr float kRotY = 0.0f; // fixed camera, no per-frame input
    static_assert(kTileCount * kTileSize == kParticleCount,
                  "the particle count must be a multiple of the tile size");
    static_assert((kParticleCount & (kParticleCount - 1u)) == 0u,
                  "the winner encoding needs a power-of-two particle count");

    // Camera rotation is folded into host constants, so the kernels only
    // multiply by literals.
    static const float kCosRotX = std::cos(kRotX);
    static const float kSinRotX = std::sin(kRotX);
    static const float kCosRotY = std::cos(kRotY);
    static const float kSinRotY = std::sin(kRotY);

    // Fixed camera: rotate around Y then X and return (screen xy, view depth).
    auto camera_space = [](Float3 position) noexcept {
        Float x1 = position.x * kCosRotY - position.z * kSinRotY;
        Float z1 = position.x * kSinRotY + position.z * kCosRotY;
        Float y1 = position.y;
        Float y2 = y1 * kCosRotX - z1 * kSinRotX;
        Float z2 = y1 * kSinRotX + z1 * kCosRotX;
        return compose(make_float2(x1, y2), kViewDistance + z2);
    };

    // Positive float bit patterns keep their ordering when read as unsigned
    // integers, so the low bits of the distance can carry the particle index:
    // atomic-min is then deterministic even for coincident splats.
    auto encode_winner = [](Float distance, UInt index) noexcept {
        return (as<uint>(distance) & kDepthMask) | index;
    };

    Kernel1D clear_winners = [](BufferUInt winners) noexcept {
        set_block_size(256u, 1u, 1u);
        winners.write(dispatch_x(), kInvalidWinner);
    };

    // Force accumulation with shared-memory tiling: each block stages 256
    // (position, mass) records, then every thread tests its own particle
    // against the whole tile.  A pair that lands on the same particle
    // contributes exactly zero because the softening keeps the distance
    // non-zero while the displacement is zero.
    Kernel1D nbody = [&](BufferVar<NBodyParticle> read_buf,
                         BufferVar<NBodyParticle> write_buf) noexcept {
        set_block_size(kTileSize, 1u, 1u);
        Shared<float4> tile{kTileSize};
        UInt idx = dispatch_x();
        UInt lane = thread_x();
        Var<NBodyParticle> self = read_buf.read(idx);
        Float3 self_position = self.position;
        Float self_mass = self.mass;
        Float3 force = def(make_float3(0.0f));
        for (uint tile_index = 0u; tile_index < kTileCount; ++tile_index) {
            auto other = read_buf.read(tile_index * kTileSize + lane);
            tile[lane] = make_float4(other.position, other.mass);
            sync_block();
            $for (j, kTileSize) {
                Float4 staged = tile[j];
                Float3 delta = staged.xyz() - self_position;
                Float distance_sq = dot(delta, delta) + kSoftening * kSoftening;
                Float distance = sqrt(distance_sq);
                Float attraction = kGravity * self_mass * staged.w / distance_sq;
                force += (attraction / distance) * delta;
            };
            sync_block();
        }
        // Euler integration with a small damping factor for stability.
        Float3 new_velocity = self.velocity + force / self_mass * kDt;
        new_velocity = new_velocity * kDamping;
        Float3 new_position = self_position + new_velocity * kDt;
        write_buf.write(idx, def<NBodyParticle>(new_position, new_velocity,
                                                self_mass, make_float3(0.0f)));
    };

    // Depth-resolved splat: the only contended operation is an unsigned
    // atomic-min over the 5x5 footprint of the projected particle.
    Kernel1D rasterize = [&](BufferVar<NBodyParticle> particles,
                             BufferUInt winners) noexcept {
        set_block_size(256u, 1u, 1u);
        UInt idx = dispatch_x();
        Var<NBodyParticle> particle = particles.read(idx);
        Var camera = camera_space(particle.position);
        auto camera_xy = camera.get<0>();
        Float distance = camera.get<1>();
        Bool finite_projection = !luisa::compute::isnan(distance) &
                                 !luisa::compute::isinf(distance) &
                                 !luisa::compute::isnan(camera_xy.x) &
                                 !luisa::compute::isinf(camera_xy.x) &
                                 !luisa::compute::isnan(camera_xy.y) &
                                 !luisa::compute::isinf(camera_xy.y);
        $if ((distance > kMinVisibleDistance) & finite_projection) {
            Float scale = kProjectionScale / distance;
            Float screen_x = camera_xy.x * scale *
                                 static_cast<float>(kImageSize) * 0.5f +
                             static_cast<float>(kImageSize) * 0.5f;
            Float screen_y = camera_xy.y * scale *
                                 static_cast<float>(kImageSize) * 0.5f +
                             static_cast<float>(kImageSize) * 0.5f;
            Bool finite_screen = !luisa::compute::isnan(screen_x) &
                                 !luisa::compute::isinf(screen_x) &
                                 !luisa::compute::isnan(screen_y) &
                                 !luisa::compute::isinf(screen_y);
            Bool center_in_bounds = (screen_x > -1.0f) &
                                    (screen_x < static_cast<float>(kImageSize)) &
                                    (screen_y > -1.0f) &
                                    (screen_y < static_cast<float>(kImageSize));
            $if (finite_screen & center_in_bounds) {
                Int2 center = make_int2(cast<int>(screen_x), cast<int>(screen_y));
                UInt winner = encode_winner(distance, idx);
                for (int dy = -kGlowRadius; dy <= kGlowRadius; ++dy) {
                    for (int dx = -kGlowRadius; dx <= kGlowRadius; ++dx) {
                        Int2 pixel = center + make_int2(dx, dy);
                        $if ((pixel.x >= 0) & (pixel.x < static_cast<int>(kImageSize)) &
                             (pixel.y >= 0) & (pixel.y < static_cast<int>(kImageSize))) {
                            winners.atomic(cast<uint>(pixel.y) * kImageSize +
                                           cast<uint>(pixel.x))
                                .fetch_min(winner);
                        };
                    }
                }
            };
        };
    };

    // Color the winning splat of every pixel in a separate image-parallel pass.
    Kernel2D resolve = [&](BufferVar<NBodyParticle> particles,
                           BufferUInt winners, ImageFloat image) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 pixel = dispatch_id().xy();
        UInt winner = winners.read(pixel.y * kImageSize + pixel.x);
        Float3 color = make_float3(0.02f, 0.02f, 0.05f);
        $if (winner != kInvalidWinner) {
            UInt idx = winner & kPixelIndexMask;
            Var<NBodyParticle> particle = particles.read(idx);
            Var camera = camera_space(particle.position);
            auto camera_xy = camera.get<0>();
            Float distance = camera.get<1>();
            Float scale = kProjectionScale / max(distance, kMinVisibleDistance);
            Float screen_x = camera_xy.x * scale *
                                 static_cast<float>(kImageSize) * 0.5f +
                             static_cast<float>(kImageSize) * 0.5f;
            Float screen_y = camera_xy.y * scale *
                                 static_cast<float>(kImageSize) * 0.5f +
                             static_cast<float>(kImageSize) * 0.5f;
            Float delta_x = cast<float>(pixel.x) - screen_x;
            Float delta_y = cast<float>(pixel.y) - screen_y;
            Float footprint = sqrt(delta_x * delta_x + delta_y * delta_y);
            Float depth_factor = 1.0f / (1.0f + distance * 0.1f);
            Float r = 0.5f + 0.5f * sin(cast<float>(idx) * 0.1f);
            Float g = 0.5f + 0.5f * sin(cast<float>(idx) * 0.13f + 2.0f);
            Float b = 0.8f + 0.2f * sin(cast<float>(idx) * 0.07f + 4.0f);
            Float intensity = exp(-footprint * 0.8f) * 0.9f;
            color = make_float3(r, g, b) * depth_factor * intensity;
        };
        image.write(pixel, make_float4(color, 1.0f));
    };

    double compile_ms = 0.0;
    auto clear_winners_shader = compile_uncached(
        device, clear_winners, "bench_physics_nbody_clear_winners", cfg, compile_ms);
    auto nbody_shader = compile_uncached(
        device, nbody, "bench_physics_nbody_forces", cfg, compile_ms);
    auto rasterize_shader = compile_uncached(
        device, rasterize, "bench_physics_nbody_rasterize", cfg, compile_ms);
    auto resolve_shader = compile_uncached(
        device, resolve, "bench_physics_nbody_resolve", cfg, compile_ms);

    // Double-buffered particle state: the force kernel reads one buffer and
    // writes the other, and the host swaps the pair after recording the step.
    Buffer<NBodyParticle> particles_a = device.create_buffer<NBodyParticle>(kParticleCount);
    Buffer<NBodyParticle> particles_b = device.create_buffer<NBodyParticle>(kParticleCount);
    Buffer<uint> winners = device.create_buffer<uint>(kPixelCount);
    Image<float> display = device.create_image<float>(PixelStorage::BYTE4, kImageSize, kImageSize);
    bool a_is_read_buffer = true;

    auto read_buffer = [&]() noexcept -> Buffer<NBodyParticle> & {
        return a_is_read_buffer ? particles_a : particles_b;
    };
    auto write_buffer = [&]() noexcept -> Buffer<NBodyParticle> & {
        return a_is_read_buffer ? particles_b : particles_a;
    };

    // Deterministic galaxy-like disk: tangential speed balanced against the
    // central mass plus a small random perturbation.
    luisa::vector<NBodyParticle> host_particles(kParticleCount);
    {
        uint64_t seed = 0xfedcba9876543210ull;
        static constexpr float kTwoPi = 6.28318530717958647692f;
        for (uint i = 0u; i < kParticleCount; ++i) {
            float radius = prng_f32(seed, 0.2f, 1.5f);
            float angle = prng_f32(seed, 0.0f, kTwoPi);
            float height = prng_f32(seed, -0.075f, 0.075f);
            float orbital_speed = std::sqrt(kGravity * kCentralMass / radius);
            float3 position = make_float3(radius * std::cos(angle),
                                          height,
                                          radius * std::sin(angle));
            float3 velocity = make_float3(
                -orbital_speed * std::sin(angle) + prng_f32(seed, -0.15f, 0.15f),
                prng_f32(seed, -0.05f, 0.05f),
                orbital_speed * std::cos(angle) + prng_f32(seed, -0.15f, 0.15f));
            host_particles[i] = NBodyParticle{
                .position = position,
                .velocity = velocity,
                .mass = prng_f32(seed, 0.5f, 2.0f),
                .pad = {0.0f, 0.0f, 0.0f}};
        }
    }

    auto reset_particles = [&]() noexcept {
        // Both buffers get the seed state, so the first step works no matter
        // which one the previous run left as the read buffer.
        stream << particles_a.copy_from(luisa::span{host_particles})
               << particles_b.copy_from(luisa::span{host_particles})
               << synchronize();
        a_is_read_buffer = true;
    };

    // One timed iteration: two simulation+render steps in a single CommandList.
    auto dispatch_fn = [&]() noexcept {
        CommandList cmd_list;
        for (uint step = 0u; step < kStepsPerIteration; ++step) {
            Buffer<NBodyParticle> &src = read_buffer();
            Buffer<NBodyParticle> &dst = write_buffer();
            cmd_list << clear_winners_shader(winners).dispatch(kPixelCount)
                     << nbody_shader(src, dst).dispatch(kParticleCount)
                     << rasterize_shader(src, winners).dispatch(kParticleCount)
                     << resolve_shader(src, winners, display).dispatch(kImageSize, kImageSize);
            a_is_read_buffer = !a_is_read_buffer;
        }
        stream << cmd_list.commit();
    };

    // --- verification: finite particle states and a non-empty splat ----------
    PhysicsCheck check;
    reset_particles();
    if (cfg.verify) {
        for (uint i = 0u; i < 2u; ++i) {
            dispatch_fn();
        }
        luisa::vector<NBodyParticle> checked(kParticleCount);
        luisa::vector<uint> checked_winners(kPixelCount);
        luisa::vector<uint8_t> checked_pixels(static_cast<size_t>(kPixelCount) * 4u);
        stream << read_buffer().copy_to(luisa::span{checked})
               << winners.copy_to(luisa::span{checked_winners})
               << display.copy_to(luisa::span{checked_pixels})
               << synchronize();
        size_t bad = 0u;
        float max_speed = 0.0f;
        for (uint i = 0u; i < kParticleCount; ++i) {
            const auto &particle = checked[i];
            if (!is_finite(particle.position) || !is_finite(particle.velocity) ||
                !std::isfinite(particle.mass)) {
                bad++;
                continue;
            }
            max_speed = std::max(max_speed,
                                 std::sqrt(particle.velocity.x * particle.velocity.x +
                                           particle.velocity.y * particle.velocity.y +
                                           particle.velocity.z * particle.velocity.z));
        }
        size_t active_pixels = 0u;
        for (uint i = 0u; i < kPixelCount; ++i) {
            active_pixels += checked_winners[i] != kInvalidWinner ? 1u : 0u;
        }
        size_t lit_pixels = 0u;
        for (uint i = 0u; i < kPixelCount; ++i) {
            auto offset = static_cast<size_t>(i) * 4u;
            auto sum = static_cast<uint32_t>(checked_pixels[offset + 0u]) +
                       static_cast<uint32_t>(checked_pixels[offset + 1u]) +
                       static_cast<uint32_t>(checked_pixels[offset + 2u]);
            lit_pixels += sum > 40u ? 1u : 0u;
        }
        check.passed = bad == 0u && active_pixels != 0u && lit_pixels != 0u;
        if (!check.passed) {
            check.note = luisa::format(
                "{} of {} particle states are non-finite, {} active splat "
                "pixels, {} lit pixels on the {}x{} image",
                bad, kParticleCount, active_pixels, lit_pixels, kImageSize, kImageSize);
            LUISA_WARNING("physics_nbody: {}", check.note);
        } else {
            check.note = luisa::format(
                "{} particle states finite (max |v| {:.6e}), {} active splat "
                "pixels, {} lit pixels on the {}x{} image",
                kParticleCount, max_speed, active_pixels, lit_pixels,
                kImageSize, kImageSize);
        }
        reset_particles();
    } else {
        check.note = "verification skipped (verify disabled)";
    }

    auto result = run_timed_bench(
        "physics", "physics_nbody", stream, cfg, compile_ms,
        static_cast<double>(kParticleCount) * static_cast<double>(kParticleCount) *
            static_cast<double>(kStepsPerIteration),
        "pair/s", dispatch_fn);
    result.verified = check.passed;
    result.check_note = std::move(check.note);
    results.emplace_back(std::move(result));
}

// ---------------------------------------------------------------------------
// 3. 2D finite-difference wave equation
// ---------------------------------------------------------------------------
void bench_wave_equation(compute::Device &device, compute::Stream &stream,
                         const BenchConfig &cfg, std::vector<BenchResult> &results) {

    static constexpr uint kSize = 1024u;
    static constexpr uint kSubsteps = 16u;
    // c^2 dt^2 = 0.04 keeps the 2D explicit scheme far inside its CFL limit.
    static constexpr float kWaveSpeed = 0.2f;
    static constexpr float kDt = 1.0f;
    static constexpr float kDamping = 0.999f;
    static constexpr uint kDropletCount = 8u;
    static constexpr float kDropletStrength = 1.5f;
    static constexpr float kDropletRadius = 15.0f;

    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        set_block_size(16u, 16u, 1u);
        image.write(dispatch_id().xy(), make_float4(0.0f));
    };

    // Deterministic Gaussian droplet: seeded on the host, applied by the device.
    Kernel2D drop_kernel = [&](ImageFloat height, UInt2 center, Float strength) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 uv = dispatch_id().xy();
        Float dx = cast<float>(cast<int>(uv.x) - cast<int>(center.x));
        Float dy = cast<float>(cast<int>(uv.y) - cast<int>(center.y));
        Float distance_sq = dx * dx + dy * dy;
        Float gaussian = exp(-distance_sq / (kDropletRadius * kDropletRadius * 0.5f));
        Float current = height.read(uv).x;
        height.write(uv, make_float4(current + strength * gaussian, 0.0f, 0.0f, 1.0f));
    };

    // One explicit finite-difference step with clamped (fixed) borders.
    Kernel2D wave_step = [&](ImageFloat prev, ImageFloat curr, ImageFloat next) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 uv = dispatch_id().xy();
        UInt2 size = dispatch_size().xy();
        auto sample = [&](Int2 offset) noexcept {
            Int2 p = make_int2(cast<int>(uv.x), cast<int>(uv.y)) + offset;
            p.x = clamp(p.x, 0, cast<int>(size.x) - 1);
            p.y = clamp(p.y, 0, cast<int>(size.y) - 1);
            return curr.read(make_uint2(cast<uint>(p.x), cast<uint>(p.y))).x;
        };
        Float u_curr = curr.read(uv).x;
        Float u_prev = prev.read(uv).x;
        Float laplacian = sample(make_int2(-1, 0)) + sample(make_int2(1, 0)) +
                          sample(make_int2(0, -1)) + sample(make_int2(0, 1)) -
                          4.0f * u_curr;
        Float u_next = 2.0f * u_curr - u_prev +
                       (kWaveSpeed * kWaveSpeed * kDt * kDt) * laplacian;
        u_next = u_next * kDamping;
        next.write(uv, make_float4(u_next, 0.0f, 0.0f, 1.0f));
    };

    // Gradient/caustic shading of the current height field into a FLOAT4 image.
    Kernel2D render_kernel = [&](ImageFloat height, ImageFloat output, Float time) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 uv = dispatch_id().xy();
        UInt2 size = dispatch_size().xy();
        auto sample = [&](Int2 offset) noexcept {
            Int2 p = make_int2(cast<int>(uv.x), cast<int>(uv.y)) + offset;
            p.x = clamp(p.x, 0, cast<int>(size.x) - 1);
            p.y = clamp(p.y, 0, cast<int>(size.y) - 1);
            return height.read(make_uint2(cast<uint>(p.x), cast<uint>(p.y))).x;
        };
        Float h = height.read(uv).x;
        Float slope_x = abs(sample(make_int2(2, 0)) - sample(make_int2(-2, 0)));
        Float slope_y = abs(sample(make_int2(0, 2)) - sample(make_int2(0, -2)));
        Float slope = sqrt(slope_x * slope_x + slope_y * slope_y);
        Float3 deep_color = make_float3(0.0f, 0.1f, 0.3f);
        Float3 shallow_color = make_float3(0.0f, 0.4f, 0.6f);
        Float3 foam_color = make_float3(0.9f, 0.95f, 1.0f);
        Float3 highlight_color = make_float3(0.6f, 0.8f, 1.0f);
        Float depth_t = clamp(h * 0.5f + 0.5f, 0.0f, 1.0f);
        Float3 color = lerp(deep_color, shallow_color, depth_t);
        Float caustics = smoothstep(0.1f, 0.5f, slope);
        color = lerp(color, highlight_color, caustics * 0.5f);
        Float foam = smoothstep(0.5f, 1.0f, h);
        color = lerp(color, foam_color, foam * 0.7f);
        Float shimmer = sin(cast<float>(uv.x) * 0.1f + time * 2.0f) *
                        sin(cast<float>(uv.y) * 0.1f + time * 1.5f);
        color = color * (0.9f + 0.1f * ((shimmer + 1.0f) * 0.5f));
        output.write(uv, make_float4(color, 1.0f));
    };

    double compile_ms = 0.0;
    auto clear_shader = compile_uncached(
        device, clear_kernel, "bench_physics_wave_eq_clear", cfg, compile_ms);
    auto drop_shader = compile_uncached(
        device, drop_kernel, "bench_physics_wave_eq_drop", cfg, compile_ms);
    auto wave_shader = compile_uncached(
        device, wave_step, "bench_physics_wave_eq_step", cfg, compile_ms);
    auto render_shader = compile_uncached(
        device, render_kernel, "bench_physics_wave_eq_render", cfg, compile_ms);

    // Three FLOAT1 height fields form the ping-pong pair (prev, curr, next);
    // the render target is a separate FLOAT4 image.
    Image<float> height_prev = device.create_image<float>(PixelStorage::FLOAT1, kSize, kSize);
    Image<float> height_curr = device.create_image<float>(PixelStorage::FLOAT1, kSize, kSize);
    Image<float> height_next = device.create_image<float>(PixelStorage::FLOAT1, kSize, kSize);
    Image<float> display = device.create_image<float>(PixelStorage::FLOAT4, kSize, kSize);

    // Deterministic droplet centers, well inside the domain so the clamped
    // borders never dominate the result.
    luisa::vector<uint2> droplets(kDropletCount);
    {
        uint64_t seed = 0x5deece66dull;
        static constexpr uint kMargin = 96u;
        for (uint i = 0u; i < kDropletCount; ++i) {
            droplets[i] = make_uint2(kMargin + prng_u32(seed) % (kSize - 2u * kMargin),
                                     kMargin + prng_u32(seed) % (kSize - 2u * kMargin));
        }
    }

    // Seed prev and curr with the same field so the initial velocity is zero.
    auto seed_waves = [&]() noexcept {
        stream << clear_shader(height_prev).dispatch(kSize, kSize)
               << clear_shader(height_curr).dispatch(kSize, kSize)
               << clear_shader(height_next).dispatch(kSize, kSize);
        for (auto center : droplets) {
            stream << drop_shader(height_prev, center, kDropletStrength).dispatch(kSize, kSize)
                   << drop_shader(height_curr, center, kDropletStrength).dispatch(kSize, kSize);
        }
        stream << synchronize();
    };

    float render_time = 0.0f;
    // One timed iteration: sixteen substeps plus the render pass, recorded into
    // a single CommandList.  The buffer rotation happens on the host while the
    // list is recorded, exactly like the interactive example.
    auto dispatch_fn = [&]() noexcept {
        CommandList cmd_list;
        for (uint substep = 0u; substep < kSubsteps; ++substep) {
            cmd_list << wave_shader(height_prev, height_curr, height_next)
                            .dispatch(kSize, kSize);
            luisa::swap(height_prev, height_curr);
            luisa::swap(height_curr, height_next);
        }
        render_time += 0.016f;
        cmd_list << render_shader(height_curr, display, render_time).dispatch(kSize, kSize);
        stream << cmd_list.commit();
    };

    // --- verification: finite heights and non-zero energy ---------------------
    PhysicsCheck check;
    seed_waves();
    if (cfg.verify) {
        for (uint i = 0u; i < 2u; ++i) {
            dispatch_fn();
        }
        stream << synchronize();
        luisa::vector<float> checked(kSize * kSize);
        stream << height_curr.copy_to(luisa::span{checked}) << synchronize();
        size_t bad = 0u;
        double energy = 0.0;
        float max_abs = 0.0f;
        for (size_t i = 0u; i < checked.size(); ++i) {
            float h = checked[i];
            if (!std::isfinite(h)) {
                bad++;
                continue;
            }
            energy += static_cast<double>(h) * static_cast<double>(h);
            max_abs = std::max(max_abs, std::abs(h));
        }
        check.passed = bad == 0u && std::isfinite(energy) && energy > 0.0;
        if (!check.passed) {
            check.note = luisa::format(
                "{}x{} height field: {} non-finite cells, energy {:.6e} "
                "(max |h| {:.6e})",
                kSize, kSize, bad, energy, max_abs);
            LUISA_WARNING("physics_wave_eq: {}", check.note);
        } else {
            check.note = luisa::format(
                "{}x{} height field finite, energy {:.6e} (max |h| {:.6e})",
                kSize, kSize, energy, max_abs);
        }
        // Re-seed so the timed run starts from the same droplets.
        seed_waves();
        render_time = 0.0f;
    } else {
        check.note = "verification skipped (verify disabled)";
    }

    auto result = run_timed_bench(
        "physics", "physics_wave_eq", stream, cfg, compile_ms,
        static_cast<double>(kSize) * static_cast<double>(kSize) *
            static_cast<double>(kSubsteps),
        "cell_step/s", dispatch_fn);
    result.verified = check.passed;
    result.check_note = std::move(check.note);
    results.emplace_back(std::move(result));
}

}// namespace
}// namespace physics_bench

void run_physics_benchmarks(compute::Device &device,
                            compute::Stream &stream,
                            const BenchConfig &cfg,
                            std::vector<BenchResult> &results) {
    LUISA_ASSERT(cfg.timed_iters > 0u,
                 "the physics benchmarks need at least one timed iteration");
    if (should_run(cfg, "physics", "physics_mpm3d")) {
        physics_bench::bench_mpm3d(device, stream, cfg, results);
    }
    if (should_run(cfg, "physics", "physics_nbody")) {
        physics_bench::bench_nbody(device, stream, cfg, results);
    }
    if (should_run(cfg, "physics", "physics_wave_eq")) {
        physics_bench::bench_wave_equation(device, stream, cfg, results);
    }
}

}// namespace luisa::bench
