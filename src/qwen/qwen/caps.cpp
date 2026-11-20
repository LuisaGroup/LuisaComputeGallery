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
        caps.split_barrier = vk_status->is_active("VK_EXT_shader_split_barrier");
        caps.cooperative_vector = vk_status->is_active("VK_NV_cooperative_vector");
        caps.cooperative_matrix = vk_status->is_enabled("VK_KHR_cooperative_matrix");
        // VK_KHR_cooperative_matrix and VK_NV_cooperative_vector are negotiated
        // conditionally inside the Vulkan backend (it picks KHR -> NV2 -> NV and
        // requires the Vulkan memory model for vectors); they are not part of
        // the optional-extension registry that VulkanExtensionStatusExt
        // exposes. A false here therefore means "not visible through the status
        // extension", not "the device lacks it" -- stated in the notes because a
        // benchmark reader would otherwise draw the wrong conclusion.
        if (!caps.cooperative_matrix || !caps.cooperative_vector) {
            add_note(caps.notes,
                     "cooperative matrix/vector are negotiated inside the VK backend and "
                     "not visible through VulkanExtensionStatusExt; false means 'not "
                     "reported', and no qwen kernel emits either instruction");
        }
        if (!caps.split_barrier) {
            add_note(caps.notes,
                     "split barrier inactive: stage boundaries keep the full "
                     "sync_block() path");
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
}

void qwen_log_backend_caps(const QwenBackendCaps &caps) noexcept {
    LUISA_INFO("Qwen backend caps:\n"
               "  device: backend '{}', warp {}, max shared {} B, memory granularity {} B\n"
               "  sparse: supported {}, tile {} B -> kv page {} B, {} tiles/page, "
               "{} pages/region, {} heap slots\n"
               "  cuda graph: available {}, conditional nodes {}, device launch {}, "
               "programmatic edges {}\n"
               "  accelerators: pdl {}, native async copy {}, split barrier {}, "
               "cooperative vector {}, cooperative matrix {}\n"
               "  notes: {}",
               caps.backend_name, caps.warp_size, caps.max_shared_memory_bytes,
               caps.memory_granularity_bytes, caps.sparse_buffer, caps.sparse_tile_bytes,
               caps.kv_page_bytes, caps.kv_tiles_per_page, caps.kv_pages_per_region,
               caps.kv_heap_slots, caps.cuda_graph, caps.cuda_graph_conditional_nodes,
               caps.cuda_graph_device_launch, caps.cuda_graph_programmatic_edges, caps.pdl,
               caps.async_copy_native, caps.split_barrier, caps.cooperative_vector,
               caps.cooperative_matrix,
               caps.notes.empty() ? luisa::string_view{"none"} : luisa::string_view{caps.notes});
}

}// namespace luisa::qwen
