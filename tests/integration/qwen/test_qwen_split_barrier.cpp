// Phase 9c LCV-3 — VK split-barrier experiment. STANDALONE MEASUREMENT, not an
// engine speedup (nothing here is reachable from QwenDslEngine).
//
// opt_plan/04 sections 3/8 ask for exactly one thing before any split-barrier use:
// "enabled path and sync_block() path produce identical results; DX never selects
// it". This file provides that differential on a pipeline with the shape of the
// tiled prefill GEMM K-loop:
//
//   producer: cooperatively stage the NEXT K chunk of the A and B tiles into a
//             3-stage shared-memory ring;
//   consumer: accumulate the CURRENT chunk into a per-thread FP32 dot.
//
// The two arms differ ONLY in the barrier between the two halves:
//   split : split_barrier_arrive() [publish the current stage]
//           -> stage the next chunk -> split_barrier_wait() [acquire the current]
//   sync  : sync_block() at the same point (publish + acquire in one op)
// Both arms therefore issue exactly ONE barrier per K iteration; the split arm
// simply does the next chunk's shared-memory stores inside the barrier window
// instead of after it. Three stages are required for correctness: an arrive/wait
// pair lets one invocation run one iteration ahead of the others, so the buffer
// being written is never the buffer being read (see the comment in the kernel).
//
// Usage: test_qwen_split_barrier <backend>
//   vk   -> runs the differential + the timing when VK_EXT_shader_split_barrier is
//           enabled on a device the test creates with an EXPLICIT config
//   elsewhere -> prints why the split arm cannot be selected and runs only the
//           sync_block() arm (never a silent fallback: the reason is printed and
//           the policy checks below prove DX/CUDA can never select the split path)
//
// Exit code: 0 all checks passed / correct skip, 1 wrong result or a policy
// violation, 2 usage/device error.

#include "qwen_device_test_utils.h"

#include "qwen/caps.h"

#include <luisa/backends/ext/vk_config_ext.h>
#include <luisa/backends/ext/vk_extension_ext.h>
#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string_view>

using namespace luisa;
using namespace luisa::compute;
using namespace qwen_test;
using luisa::qwen::qwen_probe_backend_caps;

namespace {

// Tiled prefill-GEMM shape: a 16x16 output tile per block, 256 threads, one
// output element per thread, K chunked by 32 with a 3-stage shared-memory ring
// (3 * 32 * 16 * 4 B * 2 operands = 12 KiB of shared memory).
constexpr uint32_t bm = 16u;
constexpr uint32_t bn = 16u;
constexpr uint32_t bk = 32u;
constexpr uint32_t stages = 3u;
// NOTE on dispatch(): a 1D `Shader1D::dispatch(n)` launches n THREADS. The block
// count is ceil(n / the kernel's set_block_size()) and the DSL guards the body by
// "linear thread id < n". Every dispatch below therefore multiplies the block
// count by `block`; passing the block count silently launched just that many
// threads (measured while developing this file: 4 of 1024 output slots written),
// which is the first thing to check when a tiled kernel leaves zeros behind.
constexpr uint32_t block = bm * bn;// 256 threads per block
constexpr uint32_t a_tile_elems = bk * bm;
constexpr uint32_t b_tile_elems = bk * bn;
static_assert(block % 32u == 0u);
static_assert(a_tile_elems % block == 0u && b_tile_elems % block == 0u);

struct GemmPlan {
    uint32_t m{0u}, n{0u}, k{0u};
};

// Geometry contract, checked on the HOST and refused loudly (no silent
// fallback): the kernel has no bounds checks by design, so a shape that does not
// divide the tile is a planning failure, exactly like the qwen GEMV plans.
[[nodiscard]] bool plan_ok(GemmPlan p, luisa::string &why) noexcept {
    if (p.m == 0u || p.n == 0u || p.k == 0u) {
        why = "empty shape";
        return false;
    }
    if (p.m % bm != 0u || p.n % bn != 0u) {
        why = luisa::format("M({}) and N({}) must be multiples of the {}x{} tile",
                            p.m, p.n, bm, bn);
        return false;
    }
    if (p.k % bk != 0u) {
        why = luisa::format("K({}) must be a multiple of the {} K chunk", p.k, bk);
        return false;
    }
    if (p.k / bk < 1u) {
        why = luisa::format("K({}) needs at least one {} K chunk", p.k, bk);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// The selection policy this experiment pins: ONLY a Vulkan device whose
// VK_EXT_shader_split_barrier is enabled may take the split path. Everything else
// -- including a Vulkan device that merely SUPPORTS the extension -- takes
// sync_block(), and DX/CUDA can never choose it, which is what opt_plan/04
// section 2 requires (the HLSL codegen raises a hard error on the ops, so an
// unguarded emission would be a build failure, not a slowdown).
// ---------------------------------------------------------------------------
[[nodiscard]] bool may_select_split_barrier(std::string_view backend,
                                            bool split_barrier_enabled) noexcept {
    return backend == "vk" && split_barrier_enabled;
}

template<bool split>
[[nodiscard]] auto make_tiled_gemm_kernel() noexcept {
    return [](BufferFloat a, BufferFloat b, BufferFloat c,
              UInt m, UInt n, UInt k) noexcept {
        set_block_size(block, 1u, 1u);
        set_warp_size(32u);
        set_name(split ? luisa::string{"qwen_split_barrier_gemm"}
                       : luisa::string{"qwen_sync_block_gemm"});
        Shared<float> a_sh{stages * a_tile_elems};
        Shared<float> b_sh{stages * b_tile_elems};
        // Luisa semantics for a 1D kernel: `dispatch_x()` is the GLOBAL linear
        // thread id (0 .. dispatch_size-1) and `thread_x()` the index inside the
        // block, so the block index is the quotient. Using dispatch_x() directly as
        // a block index is the mistake that produced 4 live threads in 1024.
        const auto tid = thread_x();
        const auto bid = dispatch_x() / block;
        const auto tiles_n = n / bn;
        const auto m0 = (bid / tiles_n) * bm;
        const auto n0 = (bid % tiles_n) * bn;
        const auto i = tid % bm;
        const auto j = tid / bm;
        const auto iters = k / bk;
        // Stage one K chunk of both tiles into `stage`. Called either before the
        // loop (prologue), inside the barrier window (split), or right after the
        // barrier (sync) -- in every case the destination stage is one no other
        // invocation can be reading yet.
        auto load_tile = [&](Expr<uint> it, Expr<uint> stage) noexcept {
            const auto k0 = it * bk;
            for (auto r = 0u; r < a_tile_elems / block; ++r) {
                const auto idx = tid + r * block;
                const auto kk = idx / bm;
                const auto ii = idx % bm;
                a_sh[(stage * bk + kk) * bm + ii] = a.read((k0 + kk) * m + m0 + ii);
            }
            for (auto r = 0u; r < b_tile_elems / block; ++r) {
                const auto idx = tid + r * block;
                const auto kk = idx / bn;
                const auto jj = idx % bn;
                b_sh[(stage * bk + kk) * bn + jj] = b.read((k0 + kk) * n + n0 + jj);
            }
        };
        load_tile(0u, 0u);
        Var<float> acc = def(0.0f);
        // The ONLY difference between the two arms is this barrier pair.
        // Correctness argument for the 3-stage ring: an arrive/wait pair is a full
        // workgroup rendezvous at the `wait`, so two invocations can be at most one
        // iteration apart. Iteration `it` writes stage (it+1)%3 inside the barrier
        // window while another invocation may still be accumulating iteration
        // it-1 on stage (it-1)%3 = (it+2)%3 -- a different stage. A 2-stage ring
        // WOULD race here, which is why this uses 3.
        $for (it, 0u, iters) {
            if constexpr (split) {
                split_barrier_arrive();
            } else {
                sync_block();
            }
            $if (it + 1u < iters) {
                load_tile(it + 1u, (it + 1u) % stages);
            };
            if constexpr (split) {
                split_barrier_wait();
            }
            const auto stage = it % stages;
            $for (kk, 0u, bk) {
                acc = acc + a_sh[(stage * bk + kk) * bm + i] *
                                b_sh[(stage * bk + kk) * bn + j];
            };
        };
        c.write((m0 + i) * n + n0 + j, acc);
    };
}

// Independent CPU reference (small shapes only: it is O(M*N*K)).
[[nodiscard]] luisa::vector<float> gemm_reference(luisa::span<const float> a,
                                                  luisa::span<const float> b,
                                                  GemmPlan p) noexcept {
    luisa::vector<float> c(size_t(p.m) * p.n, 0.0f);
    for (auto i = 0u; i < p.m; ++i) {
        for (auto j = 0u; j < p.n; ++j) {
            float acc = 0.0f;
            for (auto kk = 0u; kk < p.k; ++kk) {
                acc += a[size_t(kk) * p.m + i] * b[size_t(kk) * p.n + j];
            }
            c[size_t(i) * p.n + j] = acc;
        }
    }
    return c;
}

[[nodiscard]] double now_ms() noexcept {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch())
        .count();
}

} // namespace

int main(int argc, char **argv) {
    unbuffered_output();
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <backend>   (e.g. vk, dx, cuda)\n", argv[0]);
        return 2;
    }
    const std::string_view backend{argv[1]};
    Context context{argv[0]};

    // ------------------------------------------------------------------
    // Device creation. The split-barrier arm needs an EXPLICIT config opt-in: a
    // device created without VK_EXT_shader_split_barrier must never be able to
    // select it, which is checked below by creating a second, opted-OUT device.
    // ------------------------------------------------------------------
    struct SplitConfig final : public VulkanDeviceConfigExt {
        bool enable{true};
        [[nodiscard]] bool enable_shader_split_barrier() const noexcept override {
            return enable;
        }
    };
    // The Vulkan backend contract allows at most ONE backend Device alive in a
    // process (Volk's dispatch tables are process-global), so the opted-OUT probe
    // is a separate run of this binary (--opt-out-probe) instead of a second device.
    const bool opt_out_probe =
        argc > 2 && std::string_view{argv[2]} == "--opt-out-probe";
    Device device;
    if (backend == "vk") {
        DeviceConfig config;
        auto split_config = luisa::make_unique<SplitConfig>();
        split_config->enable = !opt_out_probe;
        config.extension = std::move(split_config);
        device = context.create_device("vk", &config);
    } else {
        device = context.create_device(backend.data());
    }
    if (!device) {
        std::fprintf(stderr, "cannot create a '%s' device\n", backend.data());
        return 2;
    }
    Stream stream = device.create_stream();
    const auto caps = qwen_probe_backend_caps(device);
    qwen_log_backend_caps(caps);

    bool ext_enabled = false;
    if (backend == "vk") {
        auto *status = device.extension<VulkanExtensionStatusExt>();
        ext_enabled = status != nullptr &&
                      status->is_enabled("VK_EXT_shader_split_barrier");
    }
    const bool split_selected =
        may_select_split_barrier(backend, ext_enabled) && caps.backend_name == backend;
    if (opt_out_probe) {
        // The explicit opt-out: the extension must NOT be enabled, and the policy
        // must refuse the split path for this device. Nothing else runs here.
        check(!ext_enabled,
              "policy: a device created with enable_shader_split_barrier()=false "
              "does not enable VK_EXT_shader_split_barrier");
        check(!split_selected,
              "policy: that device therefore cannot select the split arm");
        std::printf("VERDICT: the split-barrier opt-out is honored (extension "
                    "enabled=%d). No kernel was compiled on this device.\n",
                    ext_enabled ? 1 : 0);
        return summarize("qwen split barrier (opt-out probe)", argv[1]);
    }
    std::printf("\n== arm selection ==\n");
    std::printf("backend '%s': VK_EXT_shader_split_barrier enabled=%d -> split arm "
                "%s; the sync_block() arm always runs\n",
                caps.backend_name.c_str(), ext_enabled ? 1 : 0,
                split_selected ? "SELECTED" : "NOT SELECTED");

    // Policy checks: the gate can never open on a non-Vulkan backend, and never
    // without the extension.
    check(!may_select_split_barrier("dx", false),
          "policy: DX can never select the split-barrier path");
    check(!may_select_split_barrier("dx", true),
          "policy: DX cannot select it even if a capability bit were wrong");
    check(!may_select_split_barrier("cuda", true),
          "policy: CUDA can never select the split-barrier path");
    check(!may_select_split_barrier("vk", false),
          "policy: a Vulkan device WITHOUT the extension selects sync_block()");

    // ------------------------------------------------------------------
    // Correctness differential. Small shape, CPU reference, both arms.
    // ------------------------------------------------------------------
    std::printf("\n== differential: tiled GEMM K-loop, split vs sync_block ==\n");
    const GemmPlan small{32u, 32u, 128u};
    luisa::vector<float> a_small(size_t(small.m) * small.k);
    luisa::vector<float> b_small(size_t(small.n) * small.k);
    for (auto kk = 0u; kk < small.k; ++kk) {
        for (auto i = 0u; i < small.m; ++i) {
            a_small[size_t(kk) * small.m + i] =
                0.01f * static_cast<float>((kk * 7u + i * 13u) % 97u) - 0.5f;
        }
        for (auto j = 0u; j < small.n; ++j) {
            b_small[size_t(kk) * small.n + j] =
                0.01f * static_cast<float>((kk * 11u + j * 5u) % 89u) - 0.4f;
        }
    }
    // Diagnostic: verify the argument passing and the thread/block mapping the
    // kernels below rely on, before blaming the tiling.
    {
        auto dbg = device.create_buffer<float>(16u);
        luisa::vector<float> dbg_zero(16u, -1.0f);
        auto dbg_shader = device.compile(Kernel1D{
            [](BufferFloat a, BufferFloat b, BufferFloat c,
               UInt m, UInt n, UInt k) noexcept {
                set_block_size(block, 1u, 1u);
                const auto tid = thread_x();
                c.write(0u, cast<float>(m));
                c.write(1u, cast<float>(n));
                c.write(2u, cast<float>(k));
                c.write(3u, cast<float>(tid));
                c.write(4u, cast<float>(dispatch_x()));
                c.write(5u, cast<float>(block_size().x));
                c.write(6u, cast<float>(dispatch_size().x));
                c.write(7u, cast<float>(tid % bm));
                c.write(8u, cast<float>(tid / bm));
                c.write(9u, a.read(0u));
                c.write(10u, b.read(0u));
                c.write(11u, a.read(1u));
                c.write(12u, a.read(31u));
                c.write(13u, a.read(32u));
            }});
        // Occupancy probe in its own buffer: 4 * 256 = 1024 threads must all run, one
    // slot per global thread id, the thread's own index as the payload.
        auto occ = device.create_buffer<float>(1024u);
        luisa::vector<float> occ_zero(1024u, -1.0f);
        auto occ_shader = device.compile(Kernel1D{
            [](BufferFloat c) noexcept {
                set_block_size(block, 1u, 1u);
                c.write(dispatch_x(), cast<float>(thread_x()));
            }});
        luisa::vector<float> host_a(size_t(32u) * 128u), host_b(size_t(32u) * 128u);
        for (auto kk = 0u; kk < 128u; ++kk) {
            for (auto i = 0u; i < 32u; ++i) {
                host_a[size_t(kk) * 32u + i] = 0.01f * static_cast<float>((kk * 7u + i * 13u) % 97u) - 0.5f;
                host_b[size_t(kk) * 32u + i] = 0.01f * static_cast<float>((kk * 11u + i * 5u) % 89u) - 0.4f;
            }
        }
        auto ab = device.create_buffer<float>(host_a.size());
        auto bb = device.create_buffer<float>(host_b.size());
        stream << ab.copy_from(luisa::span{host_a})
               << bb.copy_from(luisa::span{host_b})
               << dbg.copy_from(luisa::span{dbg_zero})
               << synchronize();
        stream << dbg_shader(ab, bb, dbg, 32u, 32u, 128u).dispatch(4u * block) << synchronize();
        luisa::vector<float> dbg_out(16u);
        stream << dbg.copy_to(luisa::span{dbg_out}) << synchronize();
        stream << occ.copy_from(luisa::span{occ_zero}) << synchronize();
        stream << occ_shader(occ).dispatch(4u * block) << synchronize();
        luisa::vector<float> occ_out(1024u);
        stream << occ.copy_to(luisa::span{occ_out}) << synchronize();
        size_t live = 0;
        float occ_max_tid = -1.0f;
        for (auto v : occ_out) {
            if (v >= 0.0f) {
                ++live;
                occ_max_tid = std::max(occ_max_tid, v);
            }
        }
        std::printf("occupancy: %zu of 1024 slots written, max tid=%g (expect 1024 and 255)\n",
                    live, static_cast<double>(occ_max_tid));
        std::printf("debug mapping: m=%g n=%g k=%g tid=%g bid=%g block=%g grid=%g i=%g j=%g "
                    "a0=%g b0=%g a1=%g a31=%g a32=%g (host a0=%g a31=%g a32=%g)\n",
                    static_cast<double>(dbg_out[0]), static_cast<double>(dbg_out[1]),
                    static_cast<double>(dbg_out[2]), static_cast<double>(dbg_out[3]),
                    static_cast<double>(dbg_out[4]), static_cast<double>(dbg_out[5]),
                    static_cast<double>(dbg_out[6]), static_cast<double>(dbg_out[7]),
                    static_cast<double>(dbg_out[8]), static_cast<double>(dbg_out[9]),
                    static_cast<double>(dbg_out[10]), static_cast<double>(dbg_out[11]),
                    static_cast<double>(dbg_out[12]), static_cast<double>(dbg_out[13]),
                    static_cast<double>(host_a[0]), static_cast<double>(host_a[31]),
                    static_cast<double>(host_a[32]));
    }

    // Diagnostic sweep over the number of K chunks: a mismatch that depends on
    // the chunk count points at the stage ring, one that is already wrong at one
    // chunk points at the tile load/index math.
    for (auto chunks : {1u, 2u, 3u, 4u}) {
        const GemmPlan probe{32u, 32u, chunks * bk};
        luisa::vector<float> pa(size_t(probe.m) * probe.k);
        luisa::vector<float> pb(size_t(probe.n) * probe.k);
        for (auto kk = 0u; kk < probe.k; ++kk) {
            for (auto i = 0u; i < probe.m; ++i) {
                pa[size_t(kk) * probe.m + i] =
                    0.01f * static_cast<float>((kk * 7u + i * 13u) % 97u) - 0.5f;
            }
            for (auto j = 0u; j < probe.n; ++j) {
                pb[size_t(kk) * probe.n + j] =
                    0.01f * static_cast<float>((kk * 11u + j * 5u) % 89u) - 0.4f;
            }
        }
        auto pab = device.create_buffer<float>(pa.size());
        auto pbb = device.create_buffer<float>(pb.size());
        auto pcb = device.create_buffer<float>(size_t(probe.m) * probe.n);
        luisa::vector<float> zeros(size_t(probe.m) * probe.n, 0.0f);
        // No-shared-memory baseline with the SAME thread->output mapping: if this
        // one is right, the defect is in the shared-memory staging; if it is also
        // wrong, it is the launch/thread mapping.
        auto baseline_shader = device.compile(Kernel1D{
            [](BufferFloat a, BufferFloat b, BufferFloat c,
               UInt m, UInt n, UInt k) noexcept {
                set_block_size(block, 1u, 1u);
                const auto tid = thread_x();
                const auto bid = dispatch_x() / block;
                const auto tiles_n = n / bn;
                const auto m0 = (bid / tiles_n) * bm;
                const auto n0 = (bid % tiles_n) * bn;
                const auto i = tid % bm;
                const auto j = tid / bm;
                Var<float> acc = def(0.0f);
                $for (kk, 0u, k) {
                    acc = acc + a.read(kk * m + m0 + i) * b.read(kk * n + n0 + j);
                };
                c.write((m0 + i) * n + n0 + j, acc);
            }});
        auto probe_shader = device.compile(
            Kernel1D{make_tiled_gemm_kernel<false>()});
        stream << pab.copy_from(luisa::span{pa})
               << pbb.copy_from(luisa::span{pb})
               << pcb.copy_from(luisa::span{zeros})
               << synchronize();
        stream << probe_shader(pab, pbb, pcb, probe.m, probe.n, probe.k)
                      .dispatch((probe.m / bm) * (probe.n / bn) * block)
               << synchronize();
        luisa::vector<float> got(size_t(probe.m) * probe.n);
        stream << pcb.copy_to(luisa::span{got}) << synchronize();
        const auto ref = gemm_reference(luisa::span<const float>{pa},
                                       luisa::span<const float>{pb}, probe);
        float worst = 0.0f;
        size_t worst_at = 0;
        for (size_t t = 0; t < got.size(); ++t) {
            const auto d = std::fabs(got[t] - ref[t]);
            if (d > worst) { worst = d; worst_at = t; }
        }
        // Same shape through the no-shared baseline.
        luisa::vector<float> base_got(size_t(probe.m) * probe.n, -1.0f);
        stream << pcb.copy_from(luisa::span{zeros}) << synchronize();
        stream << baseline_shader(pab, pbb, pcb, probe.m, probe.n, probe.k)
                      .dispatch((probe.m / bm) * (probe.n / bn) * block)
               << synchronize();
        stream << pcb.copy_to(luisa::span{base_got}) << synchronize();
        float base_worst = 0.0f;
        for (size_t t = 0; t < base_got.size(); ++t) {
            base_worst = std::max(base_worst, std::fabs(base_got[t] - ref[t]));
        }
        std::printf("sweep: chunks=%u worst|diff|=%.6g at %zu (got %.6g want %.6g) "
                    "| baseline worst|diff|=%.6g\n",
                    chunks, static_cast<double>(worst), worst_at,
                    static_cast<double>(got[worst_at]),
                    static_cast<double>(ref[worst_at]),
                    static_cast<double>(base_worst));
    }
    const auto cpu = gemm_reference(a_small, b_small, small);
    const size_t small_elems = size_t(small.m) * small.n;
    luisa::vector<float> zeros_small(small_elems, 0.0f);
    auto a_small_buf = device.create_buffer<float>(a_small.size());
    auto b_small_buf = device.create_buffer<float>(b_small.size());
    auto c_small_sync = device.create_buffer<float>(small_elems);
    auto c_small_split = device.create_buffer<float>(small_elems);
    stream << a_small_buf.copy_from(luisa::span{a_small})
           << b_small_buf.copy_from(luisa::span{b_small})
           << c_small_sync.copy_from(luisa::span{zeros_small})
           << c_small_split.copy_from(luisa::span{zeros_small})
           << synchronize();
    const auto small_blocks = (small.m / bm) * (small.n / bn);
    // NOTE the CTAD forms: `Kernel1D{lambda}` maps the DSL parameter types
    // (Var<Buffer<float>> / Var<uint>) back to the prototype types the shader
    // invocation takes (Buffer<float> / uint32_t). Writing the alias by hand with
    // the DSL types produces a shader that cannot be invoked with a Buffer.
    auto sync_kernel = Kernel1D{make_tiled_gemm_kernel<false>()};
    auto sync_shader = device.compile(sync_kernel);
    stream << sync_shader(a_small_buf, b_small_buf, c_small_sync,
                          small.m, small.n, small.k)
                    .dispatch(small_blocks * block)
           << synchronize();
    luisa::vector<float> got_sync(small_elems);
    stream << c_small_sync.copy_to(luisa::span{got_sync}) << synchronize();
    check_close(luisa::span<const float>{got_sync}, luisa::span<const float>{cpu},
                1e-3f, 1e-4f,
                "split-barrier: sync_block() arm matches the CPU reference");

    if (!split_selected) {
        std::printf("SKIP: the split arm cannot be selected on '%s' (extension "
                    "enabled=%d), so no differential is possible there. The sync "
                    "arm above still ran and matched the reference.\n",
                    caps.backend_name.c_str(), ext_enabled ? 1 : 0);
        std::printf("VERDICT: split barrier NOT measured on this backend; nothing was "
                    "emitted and nothing fell back silently.\n");
        return summarize("qwen split barrier", argv[1]);
    }
    auto split_kernel = Kernel1D{make_tiled_gemm_kernel<true>()};
    auto split_shader = device.compile(split_kernel);
    stream << split_shader(a_small_buf, b_small_buf, c_small_split,
                           small.m, small.n, small.k)
                    .dispatch(small_blocks * block)
           << synchronize();
    luisa::vector<float> got_split(small_elems);
    stream << c_small_split.copy_to(luisa::span{got_split}) << synchronize();
    check_close(luisa::span<const float>{got_split}, luisa::span<const float>{cpu},
                1e-3f, 1e-4f, "split-barrier: split arm matches the CPU reference");
    check(std::memcmp(got_split.data(), got_sync.data(),
                      got_split.size() * sizeof(float)) == 0,
          "split-barrier: split and sync_block() arms are BIT-IDENTICAL");

    // ------------------------------------------------------------------
    // Timing: cold-ish caches. A 96 MiB flush kernel (larger than the 64 MiB L2 of
    // the development GPU) runs before every timed launch; the flush cost is
    // measured separately and subtracted. >= 30 reps, median.
    // ------------------------------------------------------------------
    std::printf("\n== timing (standalone experiment, cold-ish caches) ==\n");
    const std::array<GemmPlan, 2> workloads{
        GemmPlan{1024u, 1024u, 2560u}, GemmPlan{512u, 512u, 5120u}};
    constexpr uint32_t reps = 30u;
    constexpr uint32_t launches_per_rep = 4u;
    constexpr size_t flush_floats = 96ull * 1024ull * 1024ull / sizeof(float);
    auto a_big = device.create_buffer<float>(size_t(1024u) * 5120u);
    auto b_big = device.create_buffer<float>(size_t(1024u) * 5120u);
    auto c_big_sync = device.create_buffer<float>(size_t(1024u) * 1024u);
    auto c_big_split = device.create_buffer<float>(size_t(1024u) * 1024u);
    auto flush_buf = device.create_buffer<float>(flush_floats);
    {
        luisa::vector<float> host_a(size_t(1024u) * 5120u, 0.5f);
        luisa::vector<float> host_b(size_t(1024u) * 5120u, 0.25f);
        stream << a_big.copy_from(luisa::span{host_a})
               << b_big.copy_from(luisa::span{host_b})
               << synchronize();
    }
    auto flush_shader = device.compile(Kernel1D{[&](BufferFloat buf, UInt n) noexcept {
        set_block_size(256u, 1u, 1u);
        buf.write(dispatch_x(), 1.0f);
    }});
    auto flush_launch = [&]() {
        return flush_shader(flush_buf, static_cast<uint32_t>(flush_floats))
            .dispatch(static_cast<uint32_t>(flush_floats));
    };

    // Flush-only cost (median of `reps` launches, timed the same way).
    luisa::vector<double> flush_times;
    for (auto r = 0u; r < reps; ++r) {
        const auto t0 = now_ms();
        stream << flush_launch() << synchronize();
        flush_times.emplace_back(now_ms() - t0);
    }
    std::sort(flush_times.begin(), flush_times.end());
    const auto flush_ms = flush_times[flush_times.size() / 2u];

    // `c` is the RUNTIME buffer type (Buffer<float>); BufferFloat is the DSL
    // parameter type and is only valid inside a kernel lambda.
    auto time_arm = [&](auto &shader, const Buffer<float> &c, GemmPlan p) {
        const auto blocks = (p.m / bm) * (p.n / bn);
        luisa::vector<double> cold;
        luisa::vector<double> hot;
        for (auto r = 0u; r < reps; ++r) {
            // Cold-ish: flush the L2 before every launch; the flush itself is
            // timed in the same loop and subtracted.
            {
                const auto t0 = now_ms();
                for (auto l = 0u; l < launches_per_rep; ++l) {
                    stream << flush_launch()
                           << shader(a_big, b_big, c, p.m, p.n, p.k).dispatch(blocks * block);
                }
                stream << synchronize();
                const auto total = now_ms() - t0;
                cold.emplace_back((total - flush_ms * launches_per_rep) /
                                  static_cast<double>(launches_per_rep));
            }
            // Hot: back-to-back launches, the array stays in L2.
            {
                const auto t0 = now_ms();
                for (auto l = 0u; l < launches_per_rep; ++l) {
                    stream << shader(a_big, b_big, c, p.m, p.n, p.k).dispatch(blocks * block);
                }
                stream << synchronize();
                hot.emplace_back((now_ms() - t0) /
                                 static_cast<double>(launches_per_rep));
            }
        }
        std::sort(cold.begin(), cold.end());
        std::sort(hot.begin(), hot.end());
        return std::array<double, 2>{cold[cold.size() / 2u], hot[hot.size() / 2u]};
    };

    for (auto const &p : workloads) {
        luisa::string why;
        if (!plan_ok(p, why)) {
            check(false, "timing: the workload geometry is invalid", why);
            continue;
        }
        const double flops = 2.0 * double(p.m) * double(p.n) * double(p.k);
        // Same buffers, same data, same launch count for both arms.
        auto s = time_arm(sync_shader, c_big_sync, p);
        auto f = time_arm(split_shader, c_big_split, p);
        // Differential at scale: run one launch of each and compare bitwise.
        const auto blocks = (p.m / bm) * (p.n / bn);
        stream << sync_shader(a_big, b_big, c_big_sync, p.m, p.n, p.k).dispatch(blocks * block)
               << split_shader(a_big, b_big, c_big_split, p.m, p.n, p.k).dispatch(blocks * block)
               << synchronize();
        luisa::vector<float> y_sync(size_t(p.m) * p.n);
        luisa::vector<float> y_split(size_t(p.m) * p.n);
        stream << c_big_sync.copy_to(luisa::span{y_sync})
               << c_big_split.copy_to(luisa::span{y_split})
               << synchronize();
        const bool identical =
            std::memcmp(y_sync.data(), y_split.data(), y_sync.size() * sizeof(float)) == 0;
        const char *p_name = p.k == 2560u ? "1024x1024x2560" : "512x512x5120";
        std::printf("timing[%s]: sync_block cold %.4f ms hot %.4f ms (%.1f / %.1f GFLOP/s) "
                    "| split cold %.4f ms hot %.4f ms (%.1f / %.1f GFLOP/s) "
                    "| cold delta %+.2f%% hot delta %+.2f%% | identical=%d | flush %.4f ms\n",
                    p_name, s[0], s[1], flops / (s[0] * 1.0e6), flops / (s[1] * 1.0e6),
                    f[0], f[1], flops / (f[0] * 1.0e6), flops / (f[1] * 1.0e6),
                    (f[0] / s[0] - 1.0) * 100.0, (f[1] / s[1] - 1.0) * 100.0,
                    identical ? 1 : 0, flush_ms);
        check(identical,
              "split-barrier: both arms produce bit-identical results at the timed "
              "shape",
              p_name);
        // Keep/drop: the split arm is kept only when it is measurably faster
        // (>= 1 %) AND correct. Anything else is a DROP, recorded, never assumed.
        const bool keep = identical && f[0] < s[0] * 0.99;
        std::printf("timing[%s]: decision %s%s\n", p_name,
                    keep ? "KEEP" : "DROP",
                    keep ? "" : " (engine keeps sync_block(); opt_plan/04 section 3 "
                                "requires a measured win, not a theoretical one)");
    }

    std::printf("\nVERDICT: this is a STANDALONE measurement of the tiled GEMM K-loop; "
                "no engine path emits split barriers, so no end-to-end speedup is "
                "claimed here (see the phase-9c report for the recorded keep/drop).\n");
    return summarize("qwen split barrier", argv[1]);
}
