// Qwen3.5-4B full-attention KV cache -- sparse virtual-buffer allocator.
// See kv_cache_allocator.h for the contract; this file only implements the
// host-side residency/lifetime choreography.

#include "qwen/kv_cache_allocator.h"

#include <algorithm>

#include <luisa/core/logging.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/sparse_command_list.h>

namespace luisa::qwen {

using compute::CommandList;
using compute::SparseBufferHeap;
using compute::SparseCommandList;

namespace {
// Page floor. Vulkan's maxMemoryAllocationCount is commonly 4096 and DX costs
// one D3D12MA allocation per heap, so the allocation unit cannot be the backend
// sparse tile (64 KiB on DX/VK) at a multi-thousand-token capacity.
constexpr uint64_t kMinPageBytes = 2ull * 1024ull * 1024ull;
// Commonly reported VkPhysicalDeviceLimits::maxMemoryAllocationCount. Luisa
// exposes no query for it, so it is only used for the R3 diagnostic below.
constexpr uint64_t kCommonVkAllocationLimit = 4096ull;
}// namespace

Status QwenKvCacheAllocator::create(compute::Device &device, compute::Stream &stream,
                                    uint32_t max_context, uint32_t values_per_token,
                                    uint32_t layer_count, uint64_t page_bytes_hint) noexcept {
    if (valid()) [[unlikely]] {
        return Status::error(StatusCode::invalid_argument,
                             "QwenKvCacheAllocator::create on an already-initialized allocator");
    }
    if (!device || !stream) [[unlikely]] {
        return Status::error(StatusCode::invalid_argument,
                             "KV cache creation requires a valid device and stream");
    }
    if (max_context == 0u || values_per_token == 0u || layer_count == 0u) [[unlikely]] {
        return Status::error(StatusCode::invalid_argument,
                             "KV cache geometry must be nonzero",
                             "max_context/values_per_token/layer_count");
    }

    // Probe backend sparse-buffer support and the device tile granularity;
    // both are only known after creating a sparse buffer. The probe is
    // destroyed before the real virtual buffer is created (no mappings).
    auto probe = device.create_sparse_buffer<float>(1u);
    if (!probe) [[unlikely]] {
        return Status::error(
            StatusCode::unsupported,
            "backend has no sparse-buffer support; refusing a dense KV fallback",
            "create_sparse_buffer<float>");
    }
    const uint64_t tile_bytes = probe.tile_size_bytes();
    probe = compute::SparseBuffer<float>{};
    if (tile_bytes == 0u || tile_bytes % sizeof(float) != 0u ||
        tile_bytes > 0xFFFFFFFFull) [[unlikely]] {
        return Status::error(StatusCode::unsupported,
                             "sparse tile size is not a usable whole number of float elements",
                             "tile_size_bytes");
    }

    // Allocation unit (page). A caller hint must be an exact tile multiple:
    // silently rounding a residency budget the caller asked for is worse than
    // refusing it, and the alignment can never be finer than a backend tile.
    if (page_bytes_hint != 0u && page_bytes_hint % tile_bytes != 0u) [[unlikely]] {
        return Status::error(StatusCode::invalid_argument,
                             "kv_page_bytes must be an exact multiple of the backend sparse tile size",
                             "page_bytes_hint");
    }
    const uint64_t desired_page_bytes =
        std::max(std::max(kMinPageBytes, tile_bytes), page_bytes_hint);
    // ceil-div then multiply: page_bytes is always exactly tiles_per_page *
    // tile_bytes (risk R2 -- a page smaller than a tile would break VK's
    // required_binding_size <= heap_byte_size invariant). A device reporting a
    // tile larger than the floor simply gets tiles_per_page == 1.
    const uint64_t page_tiles =
        desired_page_bytes / tile_bytes + (desired_page_bytes % tile_bytes != 0u ? 1u : 0u);
    uint64_t page_bytes = 0;
    if (!checked_mul(page_tiles, tile_bytes, page_bytes) ||
        page_bytes > 0xFFFFFFFFull) [[unlikely]] {
        return Status::error(StatusCode::invalid_argument,
                             "KV cache page size is not representable", "kv_page_bytes");
    }

    // Per-region logical length, logical payload bytes, and the page-aligned
    // element stride (ceil-div by the page element count; valid for any page
    // size, power-of-two or not).
    uint64_t logical_elements = 0;
    if (!checked_mul(static_cast<uint64_t>(max_context),
                     static_cast<uint64_t>(values_per_token), logical_elements)) [[unlikely]] {
        return Status::error(StatusCode::invalid_argument,
                             "KV cache element count overflows 64 bits",
                             "max_context * values_per_token");
    }
    uint64_t logical_bytes = 0;
    if (!checked_mul(logical_elements, sizeof(float), logical_bytes)) [[unlikely]] {
        return Status::error(StatusCode::invalid_argument,
                             "KV cache byte count overflows 64 bits", "logical_bytes");
    }
    const uint64_t pages_per_region =
        logical_bytes / page_bytes + (logical_bytes % page_bytes != 0u ? 1u : 0u);
    if (pages_per_region == 0u ||
        pages_per_region > 0xFFFFFFFFull) [[unlikely]] {
        return Status::error(StatusCode::out_of_memory,
                             "KV cache page count is out of range", "pages_per_region");
    }
    const uint64_t page_elements = page_bytes / sizeof(float);
    uint64_t stride_elements = 0;
    if (!checked_mul(pages_per_region, page_elements, stride_elements)) [[unlikely]] {
        return Status::error(StatusCode::out_of_memory,
                             "KV cache region stride overflows", "cache_stride_elements");
    }
    uint64_t tiles_per_region = 0;
    if (!checked_mul(pages_per_region, page_tiles, tiles_per_region)) [[unlikely]] {
        return Status::error(StatusCode::out_of_memory,
                             "KV cache tile count overflows", "tiles_per_region");
    }

    uint64_t region_count = 0;
    if (!checked_mul(static_cast<uint64_t>(layer_count), 2ull, region_count)) [[unlikely]] {
        return Status::error(StatusCode::invalid_argument,
                             "KV cache region count overflows", "layer_count * 2");
    }
    uint64_t total_tiles = 0;
    if (!checked_mul(tiles_per_region, region_count, total_tiles) ||
        total_tiles > 0xFFFFFFFFull) [[unlikely]] {
        // map/unmap take uint32 tile indices: this is the addressable envelope.
        return Status::error(StatusCode::out_of_memory,
                             "KV cache tile count exceeds the uint32 sparse envelope",
                             "total_tiles");
    }
    uint64_t total_elements = 0;
    if (!checked_mul(total_tiles, tile_bytes / sizeof(float), total_elements)) [[unlikely]] {
        return Status::error(StatusCode::out_of_memory,
                             "KV cache virtual element count overflows", "total_elements");
    }
    uint64_t total_bytes = 0;
    if (!checked_mul(total_tiles, tile_bytes, total_bytes)) [[unlikely]] {
        return Status::error(StatusCode::out_of_memory,
                             "KV cache virtual byte count overflows", "reserved_bytes");
    }
    // One heap slot per (region, page): the number of allocations a fully
    // resident cache needs, and the count Vulkan limits.
    uint64_t heap_slots = 0;
    if (!checked_mul(pages_per_region, region_count, heap_slots) ||
        heap_slots > 0xFFFFFFFFull) [[unlikely]] {
        return Status::error(StatusCode::out_of_memory,
                             "KV cache heap slot count is out of range", "heap_slots");
    }

    auto buffer = device.create_sparse_buffer<float>(static_cast<size_t>(total_elements));
    if (!buffer) [[unlikely]] {
        return Status::error(StatusCode::out_of_memory,
                             "virtual sparse KV buffer creation failed", "create_sparse_buffer");
    }

    _device = device;
    _stream = stream;
    _buffer = std::move(buffer);
    _heaps.resize(static_cast<size_t>(heap_slots));// empty handles, one slot per (region, page)
    _stride_elements = stride_elements;
    _logical_elements = logical_elements;
    _max_context = max_context;
    _values_per_token = values_per_token;
    _layer_count = layer_count;
    _tile_size_bytes = static_cast<uint32_t>(tile_bytes);
    _page_size_bytes = static_cast<uint32_t>(page_bytes);
    _tiles_per_page = static_cast<uint32_t>(page_tiles);
    _tiles_per_region = static_cast<uint32_t>(tiles_per_region);
    _pages_per_region = static_cast<uint32_t>(pages_per_region);
    _total_tiles = static_cast<uint32_t>(total_tiles);
    _region_count = static_cast<uint32_t>(region_count);
    _mapped_heap_count = 0u;
    _committed_tokens = 0u;

    // Risk R3 diagnostic (opt_plan/01-kv-cache-pages.md section 6): Vulkan's
    // maxMemoryAllocationCount is commonly 4096 and nothing in Luisa can query
    // it, so report the exact hint that would fit instead of clamping or
    // failing. The page model already reduces the default geometry 32x.
    if (heap_slots > kCommonVkAllocationLimit) {
        // Smallest page size whose per-region page count fits the limit:
        // ceil(logical / max_pages) rounded up to a whole tile.
        const uint64_t max_pages_per_region =
            std::max<uint64_t>(kCommonVkAllocationLimit / region_count, 1ull);
        const uint64_t min_pages_bytes =
            logical_bytes / max_pages_per_region +
            (logical_bytes % max_pages_per_region != 0u ? 1u : 0u);
        const uint64_t suggested_tiles =
            min_pages_bytes / tile_bytes + (min_pages_bytes % tile_bytes != 0u ? 1u : 0u);
        const uint64_t suggested_page_bytes = suggested_tiles * tile_bytes;
        LUISA_WARNING("Qwen KV cache needs {} heap allocations ({} regions x {} pages/region), "
                   "above the common Vulkan maxMemoryAllocationCount of {}; raise "
                   "kv_page_bytes to at least {} B ({} pages/region) to fit",
                   heap_slots, region_count, pages_per_region, kCommonVkAllocationLimit,
                   suggested_page_bytes,
                   logical_bytes / suggested_page_bytes +
                       (logical_bytes % suggested_page_bytes != 0u ? 1u : 0u));
    }

    LUISA_INFO("Qwen KV cache (sparse): {} layers, {} tokens/layer, {} values/token, "
               "tile {} B, page {} B, {} tiles/page, {} pages/region, {} heaps, "
               "stride {} elems, {} regions, {} tiles virtual ({:.2f} MiB)",
               layer_count, max_context, values_per_token, tile_bytes, page_bytes,
               page_tiles, pages_per_region, heap_slots, stride_elements, region_count,
               total_tiles, static_cast<double>(total_bytes) / 1048576.0);
    return Status::success();
}

compute::BufferView<float> QwenKvCacheAllocator::k_view(uint32_t layer) const noexcept {
    LUISA_ASSERT(valid() && layer < _layer_count,
                 "QwenKvCacheAllocator::k_view({}) on an allocator with {} layers",
                 layer, _layer_count);
    // Region order: K then V for each layer.
    return _buffer.view(region_base_elements(layer * 2u), _logical_elements);
}

compute::BufferView<float> QwenKvCacheAllocator::v_view(uint32_t layer) const noexcept {
    LUISA_ASSERT(valid() && layer < _layer_count,
                 "QwenKvCacheAllocator::v_view({}) on an allocator with {} layers",
                 layer, _layer_count);
    return _buffer.view(region_base_elements(layer * 2u + 1u), _logical_elements);
}

uint32_t QwenKvCacheAllocator::committed_pages_per_region() const noexcept {
    if (_page_size_bytes == 0u) { return 0u; }
    const uint64_t pages = pages_for_tokens(_committed_tokens, _values_per_token,
                                            _page_size_bytes);
    return static_cast<uint32_t>(std::min<uint64_t>(pages, _pages_per_region));
}

Status QwenKvCacheAllocator::ensure_capacity(compute::Stream &stream, uint32_t tokens) noexcept {
    if (!valid()) [[unlikely]] {
        return Status::error(StatusCode::invalid_argument,
                             "QwenKvCacheAllocator::ensure_capacity on an uninitialized allocator");
    }
    if (!stream || stream.handle() != _stream.handle() ||
        stream.device() != _stream.device()) [[unlikely]] {
        return Status::error(StatusCode::invalid_argument,
                             "KV cache growth must use the stream supplied at create");
    }
    if (tokens == 0u || tokens <= _committed_tokens) { return Status::success(); }
    if (tokens > _max_context) [[unlikely]] {
        return Status::error(StatusCode::invalid_argument,
                             "requested tokens exceed the configured max_context",
                             "ensure_capacity");
    }
    // Residency is page-granular: the request is rounded up to whole pages,
    // which is up to one page (512 tokens per region at the default geometry)
    // more than the token watermark reported by committed_tokens().
    const uint64_t required = std::min<uint64_t>(
        pages_for_tokens(tokens, _values_per_token, _page_size_bytes), _pages_per_region);
    const uint32_t committed = committed_pages_per_region();
    LUISA_ASSERT(required <= _pages_per_region && committed <= _pages_per_region,
                 "QwenKvCacheAllocator page bookkeeping out of range");
    if (required == committed) {
        // The committed pages already cover the request.
        _committed_tokens = tokens;
        return Status::success();
    }

    // Allocate EVERY newly required page heap before mapping anything, so a
    // partial heap-allocation failure leaves the previous mapping state
    // untouched (the fresh heaps are destroyed; nothing was committed).
    luisa::vector<uint32_t> fresh;
    fresh.reserve(static_cast<size_t>(required - committed) * _region_count);
    for (uint32_t r = 0u; r < _region_count; ++r) {
        const uint32_t base = r * _pages_per_region;
        for (uint64_t p = committed; p < required; ++p) {
            const uint32_t slot = base + static_cast<uint32_t>(p);
            // Risk R1: exactly one heap per (region, page) slot between
            // allocation and retirement. A second heap in the same slot would
            // leak the first and, on DX, alias two mappings onto one RangeStart.
            LUISA_ASSERT(!_heaps[slot],
                         "QwenKvCacheAllocator page slot {} is already mapped", slot);
            _heaps[slot] = _device.allocate_sparse_buffer_heap(_page_size_bytes);
            if (!_heaps[slot]) [[unlikely]] {
                for (const auto idx : fresh) { _heaps[idx] = SparseBufferHeap{}; }
                return Status::error(
                    StatusCode::out_of_memory,
                    "sparse heap allocation failed while growing the KV cache",
                    "allocate_sparse_buffer_heap");
            }
            fresh.emplace_back(slot);
        }
    }

    // One map batch for all regions, one map per page (ONE heap == ONE page:
    // DX computes RangeStart from the heap base, so a heap must never be mapped
    // twice). Existing mappings and their data are untouched; the same-stream
    // commit orders the maps before any kernel that follows (and CUDA's map is
    // host-immediate anyway). No host sync.
    SparseCommandList maps;
    for (uint32_t r = 0u; r < _region_count; ++r) {
        const uint32_t base = r * _pages_per_region;
        for (uint64_t p = committed; p < required; ++p) {
            maps << _buffer.map_tile(static_cast<uint32_t>(region_base_tile(r) +
                                                           p * _tiles_per_page),
                                     _tiles_per_page, _heaps[base + static_cast<uint32_t>(p)]);
        }
    }
    stream << maps.commit();
    _mapped_heap_count += static_cast<uint32_t>(fresh.size());
    _committed_tokens = tokens;
    return Status::success();
}

Status QwenKvCacheAllocator::reset(compute::Stream &stream) noexcept {
    if (!valid()) { return Status::success(); }
    if (!stream || stream.handle() != _stream.handle() ||
        stream.device() != _stream.device()) [[unlikely]] {
        return Status::error(StatusCode::invalid_argument,
                             "KV cache reset must use the stream supplied at create");
    }
    // CUDA's sparse unmap is host-immediate: drain every in-flight dispatch
    // that may still read or write committed tiles BEFORE unmapping.
    stream.synchronize();
    const bool had_residency = _mapped_heap_count != 0u;
    const uint32_t committed_pages = committed_pages_per_region();
    if (committed_pages > 0u) {
        // Mapped tiles are a contiguous [0, pages*tiles_per_page) run per
        // region because every page maps tiles_per_page whole tiles.
        const uint32_t committed_tiles = committed_pages * _tiles_per_page;
        SparseCommandList unmaps;
        for (uint32_t r = 0u; r < _region_count; ++r) {
            unmaps << _buffer.unmap_tile(static_cast<uint32_t>(region_base_tile(r)),
                                         committed_tiles);
        }
        stream << unmaps.commit();
    }
    // Retire every heap through a completion callback so destruction happens
    // after the unmap and all preceding stream work. The callback owns the
    // handles (never `this`), runs on the backend thread, and neither submits
    // nor synchronizes. The sparse buffer itself stays alive for reuse.
    if (had_residency) {
        luisa::vector<SparseBufferHeap> retired;
        retired.swap(_heaps);
        _heaps.resize(heap_slot_count());// fresh empty page slots for the next session
        CommandList retire = CommandList::create(0u, 1u);
        retire.add_callback([heaps = std::move(retired)]() mutable noexcept {
            heaps.clear();// heap destruction, deferred to stream completion
        });
        stream << retire.commit();
    }
    _mapped_heap_count = 0u;
    _committed_tokens = 0u;
    // Match the public engine reset contract: unmapping and heap retirement
    // are complete when this returns. With no prior residency, the entry drain
    // is sufficient and no second empty-stream synchronization is needed.
    if (had_residency) { stream.synchronize(); }
    return Status::success();
}

QwenKvCacheAllocator::~QwenKvCacheAllocator() noexcept {
    if (!valid()) {
        _heaps.clear();
        return;
    }
    // Retained device/stream handles remain valid through teardown. Drain
    // in-flight work before the host-immediate unmap path, then unmap every committed
    // page range (coalesced per region) and retire the buffer + heaps with a
    // completion callback so no resource dies while the GPU can still touch
    // it. No callback captures `this`; the final synchronize keeps teardown
    // deterministic.
    _stream.synchronize();
    const uint32_t committed_pages = committed_pages_per_region();
    if (committed_pages > 0u) {
        const uint32_t committed_tiles = committed_pages * _tiles_per_page;
        SparseCommandList unmaps;
        for (uint32_t r = 0u; r < _region_count; ++r) {
            unmaps << _buffer.unmap_tile(static_cast<uint32_t>(region_base_tile(r)),
                                         committed_tiles);
        }
        _stream << unmaps.commit();
    }
    auto buffer = std::move(_buffer);
    auto heaps = std::move(_heaps);
    _mapped_heap_count = 0u;
    _committed_tokens = 0u;
    CommandList retire = CommandList::create(0u, 1u);
    retire.add_callback([buffer = std::move(buffer),
                         heaps = std::move(heaps)]() mutable noexcept {
        heaps.clear();// destroy the heaps first, then release the buffer
        buffer = compute::SparseBuffer<float>{};
    });
    _stream << retire.commit() << compute::synchronize();
}

}// namespace luisa::qwen
