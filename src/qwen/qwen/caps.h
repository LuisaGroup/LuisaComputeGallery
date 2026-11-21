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
    luisa::string backend_name{}; // device.backend_name() ("cuda"/"dx"/"vk")
    uint32_t warp_size{0}; // device.compute_warp_size()
    uint32_t max_shared_memory_bytes{0}; // device.compute_max_shared_memory_size()
    uint32_t memory_granularity_bytes{0};  // device.memory_granularity()
    // device.query("total_memory") in bytes: one of the four keys the pinned
    // CUDA backend implements (cuda_device.cpp:1586-1614), and the only runtime
    // signal that distinguishes a 16 GiB from a 32 GiB tier without trusting a
    // file name. 0 = the running backend has no such key (VK/DX) or the query
    // failed; it is never a guess.
    uint64_t total_memory_bytes{0};

    // ---- parallelism --------------------------------------------------------
    // Multiprocessor (execution unit) count: the ONE quantity that turns a kernel's
    // block count into "blocks per SM" evidence. qwen_probe_backend_caps() ASKS for
    // "multi_processor_count" exactly once and records what came back:
    //   cuda  -> answered by patches/luisa-cuda-multiprocessor-query.patch: the
    //            backend calls cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    //            inside with_handle and returns the count as decimal text; a driver
    //            failure is an EMPTY answer (no warning), which leaves the field at 0.
    //   dx/vk -> this tree implements no such key there, so query() logs one
    //            "Unknown device property 'multi_processor_count'" line and answers
    //            nothing; sm_count stays 0 with sm_count_source = "unqueryable".
    // sm_count == 0 therefore means "not queryable on this backend/driver" (the
    // fail-closed answer), never "0 SMs". Nothing is inferred from the device name,
    // the tier or the compute capability: a guessed SM count would silently turn a
    // block-count report into a fabricated occupancy claim (opt_plan/03 section 4.2,
    // 08 section 6 D6). Derive the ratio with qwen_decode_attn_blocks_per_sm() so a
    // missing count can never become inf/NaN.
    uint32_t sm_count{0};
    luisa::string sm_count_source{};// "device_query:multi_processor_count" or "unqueryable"

    // ---- architecture identity (08 section 1/§7 step 1) --------------------
    // CUDA architecture ("sm_89" / "sm_120") and toolkit version. NEITHER is
    // queryable through the pinned LuisaCompute: the CUDA backend's
    // DeviceInterface::query() answers only device_name, shader_bfloat16,
    // total_memory and free_memory, and no extension exposes
    // cuDeviceGetAttribute / cudaRuntimeGetVersion. Empty/0 therefore means
    // "not exposed by this API", NOT "no architecture" -- the benchmark harness
    // records the compute capability and toolkit from nvidia-smi/nvcc, and 08
    // section 1 requires the tier and card in every run log. Adding a real
    // query key is an upstream LuisaCompute change.
    luisa::string cuda_arch{};
    uint32_t cuda_toolkit_version{0};// e.g. 12080 for CUDA 12.8; 0 = unknown

    // ---- cache -----------------------------------------------------------------
    // L2 cache size in bytes. `device.query("l2_cache_size")` is implemented by
    // NO backend in the pinned tree (CUDA/DX would additionally log "Unknown
    // device property"), so this stays 0 with a note. 02 section 1.2's prefill
    // GEMM expectation depends on it, so the number must come from a real query
    // once one exists -- never from a spec sheet.
    uint32_t l2_cache_bytes{0};

    // ---- clusters ----------------------------------------------------------
    // CUDA thread-block clusters / cluster launch control. The pinned tree has
    // no probe (no DeviceInterface method, no extension, no query key), so both
    // stay false and `notes` says so. Read this as "unqueryable", never as
    // "the hardware has no clusters" (08 section 6 explains why a consumer
    // Blackwell part reports a 1x1x1 cluster shape; that is a hardware fact,
    // not something this struct measured).
    bool cluster_launch{false};
    bool cluster_launch_control{false};
    // VK: VkPhysicalDeviceLimits::maxMemoryAllocationCount when the backend can
    // surface it. It cannot today (the value is only reachable inside the VK
    // backend via vk_mem_alloc / VkPhysicalDeviceProperties, and neither
    // VulkanExtensionStatusExt nor DeviceInterface::query exposes it), so this
    // is 0 and QwenKvCacheAllocator keeps its documented 4096-clamp hint
    // (kv_cache_allocator.cpp:180-207). 0 = "not queryable", not "zero
    // allocations allowed".
    uint32_t vk_max_memory_allocation_count{0};

    // ---- sparse KV backing -------------------------------------------------
    bool sparse_buffer{false}; // a probe SparseBuffer creation succeeded
    uint32_t sparse_tile_bytes{0}; // probe.tile_size_bytes() (mapping granularity)

    // ---- effective KV page geometry (filled by the engines) ----------------
    uint32_t kv_page_bytes{0};        // allocation unit; 0 until the allocator exists
    uint32_t kv_tiles_per_page{0};    // kv_page_bytes / sparse_tile_bytes
    uint32_t kv_pages_per_region{0};  // ceil(logical bytes / kv_page_bytes)
    uint32_t kv_heap_slots{0};        // region_count * kv_pages_per_region
    // KV dtype/layout (opt_plan/05 Q3). `kv_elements_per_word` is 1 for FP32 and 2
    // for BF16 (two RNE codes per 32-bit word); `kv_region_words` is the addressable
    // size of ONE K or V region in words (64-position block padding included) and
    // `kv_payload_bytes` the same payload WITHOUT that padding, i.e. the number the
    // tier capacity model must use. `kv_min_commit_pages` is the page count one
    // region needs for a SINGLE token: 1 at the FP32 position-major layout and one
    // per kv_head at the BF16 blocked layout, which is the granularity cost of the
    // head-major block layout stated as a number instead of hidden.
    uint32_t kv_elements_per_word{0};
    uint64_t kv_region_words{0};
    uint64_t kv_payload_bytes{0};
    uint32_t kv_min_commit_pages{0};
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
    // VK_EXT_shader_split_barrier is ENABLED on the logical device, i.e. a
    // split_barrier_arrive()/wait() kernel can compile and run. Reported by
    // VulkanExtensionStatusExt::is_enabled(); a device without the feature fails
    // closed at codegen instead of degrading to sync_block().
    bool split_barrier_enabled{false};
    // ... and it has actually been routed: at least one compiled shader declared
    // SPV_EXT_split_barrier (VulkanExtensionStatusExt::is_active(), which the backend
    // only sets when the codegen emits the instructions). Before any such kernel both
    // this and split_barrier_enabled are separate answers, and "enabled but not yet
    // used" must never be read as "unavailable".
    bool split_barrier{false};
    // VK_NV_cooperative_vector enabled on the logical device, i.e. the physical
    // device exposes it with shader's compute-stage support AND both SPIR-V
    // prerequisites (vulkanMemoryModel/vulkanMemoryModelDeviceScope +
    // shaderReplicatedComposites) are negotiable, so the codegen may emit the NV
    // instructions. Read through VulkanExtensionStatusExt::cooperative_status()
    // (phase 9c LCV-1); before that the field was always false because the VK
    // backend negotiates this extension outside the optional-extension registry.
    bool cooperative_vector{false};
    // A FLOAT32/FLOAT32/FLOAT32/FLOAT32, transpose=false configuration was
    // enumerated by the driver -- the only one the SPIR-V codegen can emit
    // (spirv_codegen/instruction.cpp hardcodes the FP32 component types and
    // transpose=false). `cooperative_vector && !cooperative_vector_fp32_config`
    // is a device that NEGOTIATED the extension but cannot run the operations this
    // tree emits; that distinction is why the field exists (measured on the
    // phase-9c box: enabled=1, fp32 config=0).
    bool cooperative_vector_fp32_config{false};
    // The conjunction the gate has to read: the extension is enabled AND the FP32
    // configuration exists, so a cooperative-vector kernel could actually run.
    // Nothing in the engine emits one (phase 9c rejected adoption: see the report).
    bool cooperative_vector_usable{false};
    // How many VkCooperativeVectorPropertiesNV entries the driver enumerated
    // (0 = extension absent/not queried). Includes the packed/quantized configs
    // the backend's sanity guard hides from its Info log, so it is the only honest
    // "how many configurations are there" number in the record.
    uint32_t cooperative_vector_config_count{0};
    // VK_KHR_cooperative_matrix enabled on the logical device, with the enabled
    // twin in cooperative_matrix_extension ("" = none; the backend prefers
    // VK_KHR_cooperative_matrix, optionally + VK_NV_cooperative_matrix2, and falls
    // back to VK_NV_cooperative_matrix). NOTE: "enabled, never emitted" -- no qwen
    // kernel uses cooperative-matrix instructions, so this pair documents
    // availability only and must not be read as "the engine will use tensor
    // cores".
    bool cooperative_matrix{false};
    luisa::string cooperative_matrix_extension{};
    // Short human-readable caveats (empty when nothing needs explaining).
    luisa::string notes{};
};

// The blocks-per-SM evidence gate 11 asks for, derived from a PROBED SM count.
// 0 when either input is 0: a backend that cannot report an SM count must make the
// caller report "occupancy evidence unavailable" -- never inf/NaN, and never a
// ratio computed from a device property that was guessed instead of queried.
[[nodiscard]] inline float
qwen_decode_attn_blocks_per_sm(uint32_t blocks, uint32_t sm_count) noexcept {
    if (blocks == 0u || sm_count == 0u) { return 0.0f; }
    return static_cast<float>(blocks) / static_cast<float>(sm_count);
}

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
