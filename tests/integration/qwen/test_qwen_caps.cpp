// Qwen backend capability record + Vulkan cooperative-extension reporting.
// (opt_plan/09 V7, opt_plan/04 sections 1/3/7; phase 9c LCV-1.)
//
// Usage: test_qwen_caps <backend>          e.g. test_qwen_caps vk
//
// WHY A DEVICE TEST. The fields pinned here are answers the backend gives about a
// *logical* device: which optional extensions it enabled, and which
// cooperative-vector configurations the driver enumerated. A host-only test can
// only check the false/0 defaults (that half lives in tests/unit/qwen/
// test_qwen_engine.cpp); the interesting contract is "the record equals the
// negotiation", which needs a real device.
//
// WHAT IT PINS
//  * `qwen_probe_backend_caps()` reports exactly what the Vulkan backend recorded
//    (VulkanExtensionStatusExt::cooperative_status()), so the numbers are READ
//    from a real read-only API and never guessed from the backend name or the
//    device string.
//  * "enabled" and "usable" stay separate answers: `cooperative_vector_usable ==
//    enabled && fp32 config`, and the FLOAT32/FLOAT32/FLOAT32/FLOAT32 with
//    transpose=false configuration is the only one the current SPIR-V codegen can
//    emit. Before phase 9c both fields were hardwired false on this driver because
//    the extension is negotiated outside the optional-extension registry.
//  * Cooperative matrix reports NEGOTIATED ONLY: `is_active()` stays false for
//    every cooperative-matrix name, because nothing in this tree lowers to a
//    cooperative-matrix instruction. "enabled" must never be read as "emitted".
//  * Every non-Vulkan backend reports the whole group false and 0 (fail-closed),
//    and never claims a usable cooperative-vector path.
//
// Exit code: 0 all checks passed, 1 at least one failed, 2 usage/device error.

#include "qwen_device_test_utils.h"

#include "qwen/caps.h"
#include "qwen/dsl_ops.h"

#include <luisa/backends/ext/vk_extension_ext.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>

#include <cmath>
#include <cstdio>
#include <string_view>

using namespace luisa;
using namespace luisa::compute;
using namespace qwen_test;
using luisa::qwen::QwenBackendCaps;
using luisa::qwen::qwen_decode_attn_blocks_per_sm;
using luisa::qwen::qwen_log_backend_caps;
using luisa::qwen::qwen_probe_backend_caps;

namespace {

// One grep-able line per run: the phase log quotes it instead of re-reading the
// multi-line caps block.
void print_machine_line(const QwenBackendCaps &caps,
                        const VulkanCooperativeStatus *coop) noexcept {
    std::printf("caps-report: backend=%s cooperative_vector=%d fp32_config=%d "
                "usable=%d configs=%u matrix_enabled=%d matrix_extension='%s' "
                "matrix_workgroup_scope=%d split_barrier_enabled=%d "
                "vector_physical_support=%d vector_memory_model=%d "
                "vector_replicated_composites=%d\n",
                caps.backend_name.c_str(),
                caps.cooperative_vector ? 1 : 0,
                caps.cooperative_vector_fp32_config ? 1 : 0,
                caps.cooperative_vector_usable ? 1 : 0,
                caps.cooperative_vector_config_count,
                caps.cooperative_matrix ? 1 : 0,
                caps.cooperative_matrix_extension.c_str(),
                coop != nullptr && coop->matrix_workgroup_scope ? 1 : 0,
                caps.split_barrier_enabled ? 1 : 0,
                coop != nullptr && coop->vector_physical_support ? 1 : 0,
                coop != nullptr && coop->vector_memory_model ? 1 : 0,
                coop != nullptr && coop->vector_replicated_composites ? 1 : 0);
}

// The device-independent part of the contract: these four answers can never
// contradict each other, on any backend.
void check_invariants(const QwenBackendCaps &caps) {
    check(!caps.cooperative_vector_usable ||
              (caps.cooperative_vector && caps.cooperative_vector_fp32_config),
          "caps: usable cooperative vector implies BOTH enabled and an FP32 config",
          luisa::format("enabled={} fp32={} usable={}", caps.cooperative_vector,
                        caps.cooperative_vector_fp32_config,
                        caps.cooperative_vector_usable));
    check(!caps.cooperative_vector_fp32_config || caps.cooperative_vector,
          "caps: an FP32 configuration cannot be reported without the extension "
          "being enabled");
    check(!caps.cooperative_matrix || !caps.cooperative_matrix_extension.empty(),
          "caps: a negotiated cooperative matrix reports its extension name");
    check(!caps.split_barrier || caps.backend_name == "vk",
          "caps: split barriers are a Vulkan-only capability",
          luisa::format("backend={}", caps.backend_name));
    check(!caps.split_barrier || caps.split_barrier_enabled,
          "caps: a used split barrier implies the extension is enabled");
    check(!caps.async_copy_native || caps.backend_name == "cuda",
          "caps: native cp.async is a CUDA-only capability",
          luisa::format("backend={}", caps.backend_name));
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
    Device device;
    try {
        device = context.create_device(backend.data());
    } catch (...) {
        std::fprintf(stderr, "cannot create a '%s' device\n", backend.data());
        return 2;
    }
    if (!device) {
        std::fprintf(stderr, "cannot create a '%s' device\n", backend.data());
        return 2;
    }

    // ------------------------------------------------------------------
    // The record itself. This is the artifact opt_plan/04 section 7 asks for:
    // printed once, before any timing, with every unqueryable value named.
    // ------------------------------------------------------------------
    const auto caps = qwen_probe_backend_caps(device);
    qwen_log_backend_caps(caps);

    check(!caps.backend_name.empty() &&
              luisa::string_view{caps.backend_name} == luisa::string_view{device.backend_name()},
          "caps: backend name is the device's own", caps.backend_name);
    check(caps.warp_size != 0u, "caps: warp size is probed",
          luisa::format("warp={}", caps.warp_size));
    // compute_max_shared_memory_size() is implemented by the CUDA and Vulkan
    // backends only; the DX backend inherits DeviceInterface's 0 ("not
    // queryable"), which is the honest answer and not a claim about the hardware.
    if (luisa::string_view{caps.backend_name} == "dx") {
        check(caps.max_shared_memory_bytes == 0u,
              "caps: DX reports max shared memory as unqueryable (0), never a guess",
              luisa::format("shared={} B", caps.max_shared_memory_bytes));
    } else {
        check(caps.max_shared_memory_bytes != 0u,
              "caps: max shared memory is probed",
              luisa::format("shared={} B", caps.max_shared_memory_bytes));
    }
    check(caps.memory_granularity_bytes != 0u,
          "caps: memory granularity is probed",
          luisa::format("granularity={} B", caps.memory_granularity_bytes));
    check_invariants(caps);

    // ------------------------------------------------------------------
    // Parallelism (gate 11): the SM count is either a REAL query answer or 0.
    //
    // The CUDA backend answers DeviceInterface::query("multi_processor_count")
    // with cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) inside
    // with_handle (patches/luisa-cuda-multiprocessor-query.patch: decimal text, an
    // empty answer and no warning when the driver call fails). DX and VK implement
    // no such key, log one "Unknown device property" line and answer nothing. Both
    // halves are pinned here, because a count inferred from the device name, the
    // tier or the compute capability would silently turn the block-count report into
    // a fabricated occupancy claim (opt_plan/03 section 4.2, 08 section 6 D6).
    // ------------------------------------------------------------------
    {
        const bool is_cuda = luisa::string_view{caps.backend_name} == "cuda";
        if (is_cuda) {
            check(caps.sm_count != 0u,
                  "caps: CUDA reports a real SM count (> 0), not the 0 default",
                  luisa::format("sm_count={} source='{}'", caps.sm_count,
                                caps.sm_count_source));
            check(caps.sm_count_source == "device_query:multi_processor_count",
                  "caps: the CUDA SM count carries the device-query source label",
                  caps.sm_count_source);
            check(caps.notes.find("exposes no SM-count query") == luisa::string::npos,
                  "caps: a cuda device never claims its SM count is unqueryable",
                  caps.notes);
        } else {
            check(caps.sm_count == 0u,
                  "caps: a backend without the key reports no SM count (0), never a guess",
                  luisa::format("backend={} sm_count={}", caps.backend_name,
                                caps.sm_count));
            check(caps.sm_count_source == "unqueryable",
                  "caps: an unavailable SM count is labelled 'unqueryable'",
                  caps.sm_count_source);
            check(caps.notes.find("exposes no SM-count query") != luisa::string::npos,
                  "caps: the unavailable SM count is explained in the notes",
                  caps.notes);
        }
        // The gate-11 ratio itself, through the exact helper the engine stores in
        // QwenEngineInfo::decode_attn_blocks_per_sm (caps.h):
        //   176 blocks = the split+dedup grid above the threshold
        //                (kv_heads 4 * max_splits 40 + q_heads 16), i.e. 2.2 at 80 SMs
        //    16 blocks = the legacy per-q_head grid, i.e. 0.2 at 80 SMs
        // A 0 SM count must stay 0 ("occupancy evidence unavailable"), never inf/NaN.
        const auto split_blocks_per_sm =
            qwen_decode_attn_blocks_per_sm(176u, caps.sm_count);
        const auto legacy_blocks_per_sm =
            qwen_decode_attn_blocks_per_sm(16u, caps.sm_count);
        check(qwen_decode_attn_blocks_per_sm(176u, 0u) == 0.0f &&
                  qwen_decode_attn_blocks_per_sm(0u, 80u) == 0.0f,
              "caps: blocks/SM is 0 (unavailable) when there is no SM count, never "
              "inf/NaN");
        const auto split_bound = luisa::qwen::dsl::qwen_attn_decode_split_bound(caps.sm_count, 4u);
        std::printf("caps-parallelism: sm_count=%u source='%s' split_blocks=176 "
                    "split_blocks_per_sm=%.4f legacy_blocks=16 "
                    "legacy_blocks_per_sm=%.4f split_bound=%u default_bound=%u\n",
                    caps.sm_count, caps.sm_count_source.c_str(), split_blocks_per_sm,
                    legacy_blocks_per_sm, split_bound,
                    luisa::qwen::dsl::qwen_attn_decode_split_default_bound);
        if (caps.sm_count != 0u) {
            // 2 * 80 / 4 = 40 = qwen_attn_decode_split_cap on the 80-SM target box, so
            // the newly queryable count reproduces the documented default grid instead
            // of changing it: the query cannot move a block count, a partial buffer,
            // a generated id or a captured graph.
            check(split_bound == luisa::qwen::dsl::qwen_attn_decode_split_default_bound,
                  "caps: the SM-count-derived split bound equals the documented "
                  "default 40 on this part (no grid change)",
                  luisa::format("bound={} default={}", split_bound,
                                luisa::qwen::dsl::qwen_attn_decode_split_default_bound));
            if (caps.sm_count == 80u) {
                // This box (RTX 4080 SUPER, AD103) reports 80 SMs: pin the two
                // numbers the report quotes instead of letting them drift.
                check(std::fabs(split_blocks_per_sm - 2.2f) < 1e-6f,
                      "caps: 176 split blocks / 80 SMs = 2.2 blocks/SM",
                      luisa::format("{}", split_blocks_per_sm));
                check(std::fabs(legacy_blocks_per_sm - 0.2f) < 1e-6f,
                      "caps: 16 legacy blocks / 80 SMs = 0.2 blocks/SM",
                      luisa::format("{}", legacy_blocks_per_sm));
            }
        }
    }

    // ------------------------------------------------------------------
    // Vulkan half: the record must equal the negotiation.
    // ------------------------------------------------------------------
    VulkanCooperativeStatus coop{};
    const auto *vk_status = device.extension<VulkanExtensionStatusExt>();
    const bool is_vk = luisa::string_view{caps.backend_name} == "vk";
    check(is_vk == (vk_status != nullptr),
          "caps: VulkanExtensionStatusExt exists exactly on the Vulkan backend",
          luisa::format("backend={} status_ext={}", caps.backend_name,
                        vk_status != nullptr ? 1 : 0));

    if (vk_status != nullptr) {
        coop = vk_status->cooperative_status();
        print_machine_line(caps, &coop);

        // The cooperative extensions are negotiated OUTSIDE the optional-extension
        // registry; the whole point of phase 9c LCV-1 is that these two fields are
        // now read from the backend instead of staying false.
        check(caps.cooperative_vector == coop.vector_enabled,
              "caps: cooperative vector 'enabled' equals the backend's record",
              luisa::format("caps={} backend={}", caps.cooperative_vector,
                            coop.vector_enabled));
        check(caps.cooperative_vector_fp32_config == coop.vector_fp32_config,
              "caps: cooperative vector 'fp32 config' equals the backend's record",
              luisa::format("caps={} backend={}",
                            caps.cooperative_vector_fp32_config,
                            coop.vector_fp32_config));
        check(caps.cooperative_vector_usable ==
                  (coop.vector_enabled && coop.vector_fp32_config),
              "caps: cooperative vector 'usable' equals enabled && fp32 config");
        check(caps.cooperative_vector_config_count == coop.vector_config_count,
              "caps: cooperative vector configuration count is the driver's",
              luisa::format("caps={} backend={}",
                            caps.cooperative_vector_config_count,
                            coop.vector_config_count));
        check(caps.cooperative_matrix == coop.matrix_enabled,
              "caps: cooperative matrix 'enabled' equals the backend's record");
        check(caps.cooperative_matrix_extension == coop.matrix_extension,
              "caps: cooperative matrix extension name equals the backend's record",
              luisa::format("caps='{}' backend='{}'",
                            caps.cooperative_matrix_extension,
                            coop.matrix_extension));

        // Prerequisites: an enabled cooperative-vector extension implies both
        // SPIR-V prerequisites held, and the physical support it was probed from.
        check(!coop.vector_enabled ||
                  (coop.vector_physical_support && coop.vector_memory_model &&
                   coop.vector_replicated_composites),
              "caps: enabled cooperative vector implies support + both SPIR-V "
              "prerequisites");
        check(!coop.vector_fp32_config || coop.vector_enabled,
              "caps: an FP32 cooperative-vector config implies the extension was "
              "enabled");
        // A device that supports the extension but did not enable it must say so,
        // and the record has to explain why instead of reporting a bare false.
        if (coop.vector_physical_support && !coop.vector_enabled) {
            check(caps.notes.find("was not enabled") != luisa::string::npos,
                  "caps: 'supported but not enabled' is explained in the notes",
                  caps.notes);
        }
        // The same 'enabled but unusable' shape this box actually has.
        if (coop.vector_enabled && !coop.vector_fp32_config) {
            check(caps.notes.find("not usable") != luisa::string::npos,
                  "caps: 'enabled but no FP32 config' is explained in the notes",
                  caps.notes);
        }

        // Per-name queries must agree with the detail struct for the two names
        // that are not in the registry (phase 9c LCV-1).
        check(vk_status->is_enabled("VK_NV_cooperative_vector") ==
                  coop.vector_enabled,
              "caps: is_enabled(VK_NV_cooperative_vector) equals the record");
        check(vk_status->is_supported("VK_NV_cooperative_vector") ==
                  coop.vector_physical_support,
              "caps: is_supported(VK_NV_cooperative_vector) equals the record");
        check(vk_status->is_active("VK_NV_cooperative_vector") ==
                  coop.vector_enabled,
              "caps: is_active(VK_NV_cooperative_vector) follows the codegen "
              "routing (enabled == the target feature is on)");
        check(vk_status->is_enabled("VK_KHR_cooperative_matrix") ==
                  coop.matrix_enabled,
              "caps: is_enabled(VK_KHR_cooperative_matrix) equals the record");
        check(!vk_status->is_active("VK_KHR_cooperative_matrix") &&
                  !vk_status->is_active("VK_NV_cooperative_matrix") &&
                  !vk_status->is_active("VK_NV_cooperative_matrix2"),
              "caps: NO cooperative-matrix name is active: enabled is never emitted");
        // Registry invariants must still hold for every listed extension (the
        // cooperative names are deliberately not in this list).
        const auto extensions = vk_status->extensions();
        check(!extensions.empty(), "caps: the optional-extension list is populated",
              luisa::format("{} entries", extensions.size()));
        for (auto const &status : extensions) {
            check(!status.enabled || (status.supported && status.requested),
                  "caps: registry invariant enabled -> supported && requested",
                  status.name);
            check(!status.active || status.enabled,
                  "caps: registry invariant active -> enabled", status.name);
            const luisa::string_view name{status.name};
            check(name != luisa::string_view{"VK_NV_cooperative_vector"} &&
                      name != luisa::string_view{"VK_KHR_cooperative_matrix"},
                  "caps: the cooperative extensions stay out of the registry list",
                  status.name);
        }
        check(vk_status->is_active("VK_EXT_shader_split_barrier") ==
                  caps.split_barrier,
              "caps: 'split barrier used' answer is the status extension's active flag");
        check(vk_status->is_enabled("VK_EXT_shader_split_barrier") ==
                  caps.split_barrier_enabled,
              "caps: 'split barrier enabled' answer is the status extension's "
              "enabled flag");
        // "used" can never exceed "enabled": a routed shader needs the extension.
        check(!caps.split_barrier || caps.split_barrier_enabled,
              "caps: a used split barrier implies the extension is enabled");
        if (!caps.split_barrier_enabled) {
            check(caps.notes.find("is not enabled on this device") !=
                      luisa::string::npos,
                  "caps: an unavailable split barrier is explained in the notes",
                  caps.notes);
        }
        // This box: the extension is enabled whenever the backend advertises it,
        // and no qwen kernel declares SPV_EXT_split_barrier yet, so "used" stays
        // false while "enabled" is true. That pair is the whole point of the two
        // fields and must be reported, not collapsed.
        LUISA_INFO("split barrier: enabled={} used={}", caps.split_barrier_enabled,
                   caps.split_barrier);
    } else {
        print_machine_line(caps, nullptr);
        // Fail-closed on every other backend: no cooperative extension, no
        // cooperative-vector usability, no split barrier.
        check(!caps.cooperative_vector,
              "caps: no cooperative vector on a non-Vulkan backend");
        check(!caps.cooperative_vector_fp32_config &&
                  !caps.cooperative_vector_usable,
              "caps: no cooperative-vector configuration on a non-Vulkan backend");
        check(caps.cooperative_vector_config_count == 0u,
              "caps: no cooperative-vector configuration count on a non-Vulkan "
              "backend");
        check(!caps.cooperative_matrix && caps.cooperative_matrix_extension.empty(),
              "caps: no cooperative matrix on a non-Vulkan backend");
        check(!caps.split_barrier && !caps.split_barrier_enabled,
              "caps: no split barrier on a non-Vulkan backend");
        check(!caps.cuda_graph || caps.backend_name == "cuda",
              "caps: CUDA graph is a CUDA-only capability");
        check(!caps.pdl || caps.backend_name == "cuda",
              "caps: PDL is a CUDA-only capability");
    }

    return summarize("qwen caps", argv[1]);
}
