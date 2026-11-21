// Qwen3.5-4B backend capability probe. See qwen/caps.h for the contract.
//
// Everything here reads already-cached backend state (no device work is issued
// except the one-element sparse-buffer probe the KV allocator also performs).
// Every field answers "is the fast path actually available on THIS device?",
// because the alternative -- inferring it from the backend name -- is how an
// accelerator silently never engages.

#include "qwen/caps.h"

#include <algorithm>

#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <luisa/backends/ext/cuda/cuda_pdl_ext.h>
#include <luisa/backends/ext/vk_extension_ext.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/sparse_buffer.h>

#include "qwen/kv_cache_allocator.h"

namespace luisa::qwen {

namespace {

// Capability values are reported in 32-bit fields; a driver limit above 4 GiB
// saturates instead of wrapping (no consumer compares against such a value).
[[nodiscard]] uint32_t saturate_u32(uint64_t value) noexcept {
    return static_cast<uint32_t>(std::min<uint64_t>(value, 0xFFFFFFFFull));
}

// Append one caveat without dropping the previous ones.
void add_note(luisa::string &notes, luisa::string_view note) noexcept {
    if (!notes.empty()) { notes += "; "; }
    notes += note;
}

// Parse a decimal answer of device.query() into a u64. The pinned backends
// answer numeric properties as plain decimal text and an unknown property as an
// empty string, so "empty" and "not a number" both yield false instead of 0
// being mistaken for a measured value.
[[nodiscard]] bool query_u64(compute::Device &device, luisa::string_view key,
                             uint64_t &out) noexcept {
    const auto answer = luisa::string{device.query(key)};
    if (answer.empty()) { return false; }
    uint64_t value = 0u;
    for (auto c : answer) {
        if (c < '0' || c > '9') { return false; }
        if (value > (UINT64_MAX - static_cast<uint64_t>(c - '0')) / 10u) { return false; }
        value = value * 10u + static_cast<uint64_t>(c - '0');
    }
    out = value;
    return true;
}

}// namespace

QwenBackendCaps qwen_probe_backend_caps(compute::Device &device) noexcept {
    QwenBackendCaps caps;
    if (!device) {
        // Host-only call site (device-free unit test, or a failed create). Every
        // field stays false/0, which IS the "unknown" answer the callers gate on.
        caps.notes = "no device: all capabilities unknown (false/0)";
        return caps;
    }
    caps.backend_name = luisa::string{device.backend_name()};
    caps.warp_size = static_cast<uint32_t>(device.compute_warp_size());
    caps.max_shared_memory_bytes = saturate_u32(device.compute_max_shared_memory_size());
    caps.memory_granularity_bytes = saturate_u32(device.memory_granularity());

    // ------------------------------------------------------------------
    // Architecture identity, L2 size, clusters, VK allocation count.
    //
    // Every one of these is recorded through a REAL query/extension API or
    // explicitly reported as unavailable. The pinned LuisaCompute implements
    // exactly these DeviceInterface::query() keys:
    //   cuda (cuda_device.cpp:1586-1634): device_name, shader_bfloat16,
    //       total_memory, free_memory, multi_processor_count
    //       (multi_processor_count is added by
    //       patches/luisa-cuda-multiprocessor-query.patch: a real
    //       cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) answer
    //       in decimal text, empty on driver failure and without a warning)
    //   dx   (LCDevice.cpp:1002-1026):    device_name,
    //       dx_cluster_acceleration_structure, buffer_device_address,
    //       shader_bfloat16
    //   vk   (device.cpp:3354-3411):      shader_untyped_pointers,
    //       opacity_micromap*, vk_cluster_acceleration_structure,
    //       shader_device_clock, buffer_device_address, motion_blur,
    //       shader_int64, shader_bfloat16,
    //       timeline_semaphore_max_value_difference
    // and an UNKNOWN key logs "Unknown device property '...'" on CUDA and DX.
    // There is therefore no key for a CUDA architecture/toolkit version, none
    // for the L2 size, none for thread-block clusters / cluster launch control,
    // no SM count on DX/VK, and no VK maxMemoryAllocationCount (the value lives
    // in the VK backend's VkPhysicalDeviceProperties and in neither
    // VulkanExtensionStatusExt nor any query). Probing an invented key would emit
    // a warning and return nothing, so those fields stay 0/false and the notes
    // say exactly why: a reader must never conclude "the hardware lacks it" from
    // them.
    // ------------------------------------------------------------------
    if (luisa::string_view{caps.backend_name} == "cuda") {
        if (query_u64(device, "total_memory", caps.total_memory_bytes)) {
            // 15.0 GiB usable on T16 / 31.0 GiB on T32 (08 section 1): the only
            // runtime tier signal this API exposes.
            add_note(caps.notes,
                     "total_memory is the one tier signal the CUDA query exposes");
        }
    }
    {
        // Nothing here can be answered on any pinned backend; say it once.
        add_note(caps.notes,
                 "cuda_arch/cuda_toolkit_version, l2_cache_bytes, cluster_launch/"
                 "cluster_launch_control and vk_max_memory_allocation_count have no "
                 "query or extension in the pinned LuisaCompute (CUDA implements "
                 "device_name/shader_bfloat16/total_memory/free_memory and, since "
                 "patches/luisa-cuda-multiprocessor-query.patch, "
                 "multi_processor_count); 0/false means "
                 "'not queryable', never 'the device lacks it' -- the benchmark harness "
                 "records CC/toolkit from nvidia-smi and the tier explicitly (08 section 7)");
    }

    // Sparse-buffer support and the mapping granularity. This is the same probe
    // QwenKvCacheAllocator::create() performs; doing it here as well keeps the
    // capability record honest even when nobody creates an engine allocator
    // (example_qwen, benchmark harnesses). Failure is a normal answer here: the
    // allocator fails closed on its own.
    {
        auto probe = device.create_sparse_buffer<float>(1u);
        if (probe) {
            caps.sparse_buffer = true;
            caps.sparse_tile_bytes = saturate_u32(probe.tile_size_bytes());
        } else {
            add_note(caps.notes,
                     "no sparse-buffer support: the KV allocator will fail closed");
        }
        probe = compute::SparseBuffer<float>{};
    }

    // CUDA graph. device.extension<T>() returns nullptr when the running
    // backend does not implement the extension, so this single check is also the
    // "wrong backend" answer -- do not parse the backend name for it.
    if (auto *graph = device.extension<compute::CudaGraphExt>()) {
        caps.cuda_graph = true;
        const auto gpu = graph->capabilities();
        caps.cuda_graph_conditional_nodes = gpu.conditional_nodes;
        caps.cuda_graph_device_launch = gpu.device_launch;
        caps.cuda_graph_programmatic_edges = gpu.programmatic_edges;
        if (!gpu.conditional_nodes) {
            // Replay is otherwise still available; only the whole-step loop
            // shape that needs a conditional node is unavailable.
            add_note(caps.notes,
                     "CUDA graph without conditional nodes: dynamic replay loops "
                     "must stay host-driven");
        }
    }

    // Programmatic dependent launch. pdl_supported() is false on any device
    // below sm_90, and opt_plan/04 requires the PDL path to be SKIPPED and
    // logged in that case, never asserted.
    if (auto *pdl = device.extension<compute::CUDAPdlStreamExt>()) {
        caps.pdl = pdl->pdl_supported();
    }

    // cp.async: a static property of the backend implementation (see caps.h).
    caps.async_copy_native = luisa::string_view{caps.backend_name} == "cuda";

    // Vulkan optional-extension negotiation. Same null-pointer guard: on DX and
    // CUDA this extension does not exist and every field stays false, which is
    // the correct "not available" answer for those backends too.
    if (auto *vk_status = device.extension<compute::VulkanExtensionStatusExt>()) {
        // "enabled" (the extension is on the logical device, so a split-barrier
        // kernel compiles and runs) vs "active" (a compiled shader already declared
        // SPV_EXT_split_barrier). Phase 9c LCV-3 needs the first answer to decide
        // whether the split path may be attempted at all.
        caps.split_barrier_enabled =
            vk_status->is_enabled("VK_EXT_shader_split_barrier");
        caps.split_barrier = vk_status->is_active("VK_EXT_shader_split_barrier");
        // VK_KHR_cooperative_matrix and VK_NV_cooperative_vector are negotiated
        // inside the Vulkan backend (it picks KHR -> NV2 -> NV, and requires the
        // Vulkan memory model + replicated composites for vectors), OUTSIDE the
        // optional-extension registry. They are therefore read through
        // VulkanExtensionStatusExt::cooperative_status(), which reports the
        // recorded negotiation instead of the registry's "unknown name" answer
        // (phase 9c LCV-1; before that patch these two fields were always false
        // on a device that had both extensions enabled).
        const auto cooperative = vk_status->cooperative_status();
        caps.cooperative_vector = cooperative.vector_enabled;
        caps.cooperative_vector_fp32_config = cooperative.vector_fp32_config;
        caps.cooperative_vector_usable =
            cooperative.vector_enabled && cooperative.vector_fp32_config;
        caps.cooperative_vector_config_count = cooperative.vector_config_count;
        caps.cooperative_matrix = cooperative.matrix_enabled;
        caps.cooperative_matrix_extension = cooperative.matrix_extension;
        if (cooperative.vector_enabled && !cooperative.vector_fp32_config) {
            add_note(caps.notes,
                     "VK_NV_cooperative_vector is ENABLED but the driver enumerates no "
                     "FLOAT32/FLOAT32/FLOAT32/FLOAT32 transpose=false configuration, which is "
                     "the only one the SPIR-V codegen can emit: the extension is negotiated, "
                     "not usable (cooperative_vector_usable=false)");
        }
        if (cooperative.vector_physical_support && !cooperative.vector_enabled) {
            add_note(caps.notes,
                     "VK_NV_cooperative_vector is supported by the device but was not "
                     "enabled: SPV_NV_cooperative_vector needs vulkanMemoryModel + "
                     "vulkanMemoryModelDeviceScope and SPV_EXT_replicated_composites");
        }
        if (!caps.cooperative_matrix) {
            add_note(caps.notes,
                     "no cooperative-matrix extension was enabled on this device "
                     "(no qwen kernel emits one either way)");
        }
        if (!caps.split_barrier_enabled) {
            add_note(caps.notes,
                     "VK_EXT_shader_split_barrier is not enabled on this device: a "
                     "split-barrier kernel cannot compile, stage boundaries keep the full "
                     "sync_block() path");
        }
    }
    // Parallelism: ask for a real SM count once and record the ANSWER, including
    // "there is none". The CUDA backend answers this key with the driver attribute
    // (patches/luisa-cuda-multiprocessor-query.patch), so a cuda device reports its
    // real execution-unit count; DX and VK implement no such key, log ONE
    // "Unknown device property 'multi_processor_count'" line here and answer
    // nothing, which is recorded as sm_count = 0 / "unqueryable". That warning is
    // the documented, expected outcome of an honest probe -- it is not an error, and
    // the split-KV bound then falls back to its documented default
    // (dsl_ops.h qwen_attn_decode_split_bound) instead of guessing a device
    // property from the device name, the tier or the compute capability.
    {
        uint64_t sm = 0u;
        if (query_u64(device, "multi_processor_count", sm) && sm != 0u) {
            caps.sm_count = saturate_u32(sm);
            caps.sm_count_source = "device_query:multi_processor_count";
        } else {
            caps.sm_count = 0u;
            caps.sm_count_source = "unqueryable";
            add_note(caps.notes,
                     "this backend exposes no SM-count query "
                     "(DeviceInterface::query(\"multi_processor_count\")): sm_count is 0 "
                     "(unqueryable), so blocks/SM is reported as unavailable and the "
                     "split-KV bound uses its documented default instead of a guess");
        }
    }

    return caps;
}

void qwen_fill_kv_page_caps(QwenBackendCaps &caps,
                            const QwenKvCacheAllocator &kv) noexcept {
    caps.kv_page_bytes = kv.page_size_bytes();
    caps.kv_tiles_per_page = kv.tiles_per_page();
    caps.kv_pages_per_region = kv.pages_per_region();
    caps.kv_heap_slots = kv.heap_slot_count();
    caps.kv_elements_per_word = kv.elements_per_word();
    caps.kv_region_words = kv.region_words();
    caps.kv_payload_bytes = kv.payload_bytes();
    caps.kv_min_commit_pages = kv.min_commit_pages();
}

void qwen_log_backend_caps(const QwenBackendCaps &caps) noexcept {
    // An unavailable value is printed as the word "unknown" / "n/a" instead of a
    // bare 0, because a 0 in a benchmark log reads as a measured zero.
    const auto u32_or = [](uint32_t value) {
        return value == 0u ? luisa::string{"unknown"} : luisa::format("{}", value);
    };
    const auto u64_or = [](uint64_t value) {
        return value == 0ull ? luisa::string{"unknown"} : luisa::format("{}", value);
    };
    LUISA_INFO("Qwen backend caps:\n"
               "  device: backend '{}', warp {}, max shared {} B, memory granularity {} B, "
               "total memory {} B\n"
               "  architecture: cuda arch '{}', cuda toolkit {}, L2 {} B, "
               "cluster launch {}, cluster launch control {} (0/false = not queryable)\n"
                 "  parallelism: sm count {}, source '{}' (0 = unqueryable, never a guess)\n"
               "  sparse: supported {}, tile {} B -> kv page {} B, {} tiles/page, "
               "{} pages/region, {} heap slots, vk maxMemoryAllocationCount {} "
               "(0 = unqueryable, the allocator keeps its 4096 hint)\n"
               "  cuda graph: available {}, conditional nodes {}, device launch {}, "
               "programmatic edges {}\n"
               "  accelerators: pdl {}, native async copy {}, split barrier used {} (enabled {}), "
               "cooperative vector {} (fp32 config {}, usable {}, {} configs), "
               "cooperative matrix {} (extension '{}', enabled never emitted)\n"
               "  notes: {}",
               caps.backend_name, caps.warp_size, caps.max_shared_memory_bytes,
               caps.memory_granularity_bytes, u64_or(caps.total_memory_bytes),
               caps.cuda_arch.empty() ? luisa::string_view{"unknown"} : luisa::string_view{caps.cuda_arch},
               u32_or(caps.cuda_toolkit_version), u32_or(caps.l2_cache_bytes),
               caps.cluster_launch, caps.cluster_launch_control,
                 u32_or(caps.sm_count),
                 caps.sm_count_source.empty() ? luisa::string_view{"unqueryable"}
                                              : luisa::string_view{caps.sm_count_source},
               caps.sparse_buffer, caps.sparse_tile_bytes,
               caps.kv_page_bytes, caps.kv_tiles_per_page, caps.kv_pages_per_region,
               caps.kv_heap_slots, u32_or(caps.vk_max_memory_allocation_count),
               caps.cuda_graph, caps.cuda_graph_conditional_nodes,
               caps.cuda_graph_device_launch, caps.cuda_graph_programmatic_edges, caps.pdl,
               caps.async_copy_native, caps.split_barrier, caps.split_barrier_enabled,
                 caps.cooperative_vector,
               caps.cooperative_vector_fp32_config, caps.cooperative_vector_usable,
               caps.cooperative_vector_config_count, caps.cooperative_matrix,
               caps.cooperative_matrix_extension.empty()
                   ? luisa::string_view{"none"}
                   : luisa::string_view{caps.cooperative_matrix_extension},
               caps.notes.empty() ? luisa::string_view{"none"} : luisa::string_view{caps.notes});
}

}// namespace luisa::qwen
