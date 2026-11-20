#pragma once
// Qwen3.5-4B backend capability probe (host-only).
//
// opt_plan/04-backend-accelerators.md section 7: every accelerator in that plan
// (Vulkan split barriers, cooperative vectors/matrices, PDL, CUDA graph
// conditionals, cp.async pipelining) is gated on a device capability the engine
// would otherwise have to guess. This struct is filled ONCE at engine creation,
// logged once at Info level, and carried in QwenEngineInfo so a benchmark log
// states which fast paths the run could have taken -- "the accelerator did not
// help" is otherwise indistinguishable from "the accelerator never engaged".
//
// Rules:
//  - The probe NEVER throws and never fails. A capability that cannot be
//    queried (wrong backend, extension absent, device handle invalid) reports
//    false/0, i.e. fail-closed: the caller must skip the fast path and log it,
//    never assert.
//  - Only the non-kv_* fields are filled by qwen_probe_backend_caps(). The
//    kv_* fields describe the allocator's page geometry and are completed by
//    the engine right after its QwenKvCacheAllocator is created, because the
//    requested page size is an engine option and the effective one depends on
//    the probed tile size.
//
// Host-only: no DSL types, no RTTI, no exceptions.

#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/runtime/device.h>

namespace luisa::qwen {

class QwenKvCacheAllocator;

struct QwenBackendCaps {
    // ---- device queries ----------------------------------------------------
    luisa::string backend_name{};          // device.backend_name() ("cuda"/"dx"/"vk")
    uint32_t warp_size{0};                 // device.compute_warp_size()
    uint32_t max_shared_memory_bytes{0};   // device.compute_max_shared_memory_size()
    uint32_t memory_granularity_bytes{0};  // device.memory_granularity()
    // ---- sparse KV backing -------------------------------------------------
    bool sparse_buffer{false};     // a probe SparseBuffer creation succeeded
    uint32_t sparse_tile_bytes{0}; // probe.tile_size_bytes() (mapping granularity)
    // ---- effective KV page geometry (filled by the engines) ----------------
    uint32_t kv_page_bytes{0};        // allocation unit; 0 until the allocator exists
    uint32_t kv_tiles_per_page{0};    // kv_page_bytes / sparse_tile_bytes
    uint32_t kv_pages_per_region{0};  // ceil(logical bytes / kv_page_bytes)
    uint32_t kv_heap_slots{0};        // region_count * kv_pages_per_region
    // ---- CUDA graph --------------------------------------------------------
    bool cuda_graph{false};                    // device.extension<CudaGraphExt>() != nullptr
    bool cuda_graph_conditional_nodes{false};  // Capabilities::conditional_nodes
    bool cuda_graph_device_launch{false};      // Capabilities::device_launch
    bool cuda_graph_programmatic_edges{false}; // Capabilities::programmatic_edges
    // ---- other accelerators ------------------------------------------------
    bool pdl{false}; // CUDAPdlStreamExt::pdl_supported() (sm_90+ programmatic dependent launch)
    // CUDA only: the backend lowers an asynchronous copy to shared memory
    // natively (inline PTX cp.async, cuda_codegen_ast.cpp CallOp::ASYNC_COPY).
    // DX never has it; VK only reaches it through the HLSL user-compute fallback
    // route. This is a static property of the backend implementation, not a
    // runtime probe, because Luisa exposes no "cp.async lowerable" query.
    bool async_copy_native{false};
    // VK_EXT_shader_split_barrier actually routed (not merely enabled).
    bool split_barrier{false};
    // VK_NV_cooperative_vector actually routed (memory model + replicated
    // composites must both be negotiable, see the VK backend).
    bool cooperative_vector{false};
    // VK_KHR_cooperative_matrix enabled on the logical device. NOTE: "enabled,
    // never emitted" -- no qwen kernel uses cooperative-matrix instructions, so
    // this field documents availability only and must not be read as "the
    // engine will use tensor cores".
    bool cooperative_matrix{false};
    // Short human-readable caveats (empty when nothing needs explaining).
    luisa::string notes{};
};

// Fill everything except the kv_* geometry. Never throws, never fails; a
// device-free or unsupported configuration yields the false/0 defaults.
[[nodiscard]] QwenBackendCaps qwen_probe_backend_caps(compute::Device &device) noexcept;

// Complete the kv_* geometry from a successfully created allocator. Kept here
// (instead of open-coding four assignments at every engine create site) so the
// reported page geometry can never drift from the allocator that actually backs
// the cache.
void qwen_fill_kv_page_caps(QwenBackendCaps &caps,
                            const QwenKvCacheAllocator &kv) noexcept;

// ONE Info-level multi-line block (no per-field logging: a benchmark log must
// stay greppable and the values are only meaningful together).
void qwen_log_backend_caps(const QwenBackendCaps &caps) noexcept;

}// namespace luisa::qwen
