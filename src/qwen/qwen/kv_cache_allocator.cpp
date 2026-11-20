// Qwen3.5-4B full-attention KV cache -- sparse virtual-buffer allocator.
// See kv_cache_allocator.h for the contract; this file only implements the
// host-side residency/lifetime choreography.

#include "qwen/kv_cache_allocator.h"

#include <luisa/core/logging.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/sparse_command_list.h>

namespace luisa::qwen {

using compute::CommandList;
using compute::SparseBufferHeap;
using compute::SparseCommandList;

Status QwenKvCacheAllocator::create(compute::Device &device, compute::Stream &stream,
                                    uint32_t max_context, uint32_t values_per_token,
                                    uint32_t layer_count) noexcept {
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
    const uint64_t tile_elems = tile_bytes / sizeof(float);

    // Per-region logical length and tile-aligned element stride (ceil-div by
    // the tile element count; valid for any tile size, power-of-two or not).
    uint64_t logical_elements = 0;
    if (!checked_mul(static_cast<uint64_t>(max_context),
                     static_cast<uint64_t>(values_per_token), logical_elements)) [[unlikely]] {
        return Status::error(StatusCode::invalid_argument,
                             "KV cache element count overflows 64 bits",
                             "max_context * values_per_token");
    }
    uint64_t padded_logical = 0;
    if (!checked_add(logical_elements, tile_elems - 1u, padded_logical)) [[unlikely]] {
        return Status::error(StatusCode::out_of_memory,
                             "KV cache region stride overflows", "cache_stride_elements");
    }
    const uint64_t tiles_per_region = padded_logical / tile_elems;
    uint64_t stride_elements = 0;
    if (!checked_mul(tiles_per_region, tile_elems, stride_elements)) [[unlikely]] {
        return Status::error(StatusCode::out_of_memory,
                             "KV cache region stride overflows", "cache_stride_elements");
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
    if (!checked_mul(total_tiles, tile_elems, total_elements)) [[unlikely]] {
        return Status::error(StatusCode::out_of_memory,
                             "KV cache virtual element count overflows", "total_elements");
    }
    uint64_t total_bytes = 0;
    if (!checked_mul(total_tiles, tile_bytes, total_bytes)) [[unlikely]] {
        return Status::error(StatusCode::out_of_memory,
                             "KV cache virtual byte count overflows", "reserved_bytes");
    }

    auto buffer = device.create_sparse_buffer<float>(static_cast<size_t>(total_elements));
    if (!buffer) [[unlikely]] {
        return Status::error(StatusCode::out_of_memory,
                             "virtual sparse KV buffer creation failed", "create_sparse_buffer");
    }

    _device = device;
    _stream = stream;
    _buffer = std::move(buffer);
    _heaps.resize(static_cast<size_t>(total_tiles));// empty heap handles, one slot per tile
    _stride_elements = stride_elements;
    _logical_elements = logical_elements;
    _max_context = max_context;
    _values_per_token = values_per_token;
    _layer_count = layer_count;
    _tile_size_bytes = static_cast<uint32_t>(tile_bytes);
    _tiles_per_region = static_cast<uint32_t>(tiles_per_region);
    _total_tiles = static_cast<uint32_t>(total_tiles);
    _region_count = static_cast<uint32_t>(region_count);
    _mapped_heap_count = 0u;
    _committed_tokens = 0u;

    LUISA_INFO("Qwen KV cache (sparse): {} layers, {} tokens/layer, {} values/token, "
               "tile {} B, stride {} elems, {} regions, {} tiles virtual ({:.2f} MiB)",
               layer_count, max_context, values_per_token, tile_bytes,
               stride_elements, region_count, total_tiles,
               static_cast<double>(total_bytes) / 1048576.0);
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
    const uint64_t required = tiles_for_tokens(tokens, _values_per_token, tile_elements());
    const uint64_t committed = committed_tiles_per_region();
    LUISA_ASSERT(required <= _tiles_per_region && committed <= _tiles_per_region,
                 "QwenKvCacheAllocator tile bookkeeping out of range");
    if (required == committed) {
        // Same committed tiles already cover the request.
        _committed_tokens = tokens;
        return Status::success();
    }

    // Allocate EVERY newly required heap before mapping anything, so a
    // partial heap-allocation failure leaves the previous mapping state
    // untouched (the fresh heaps are destroyed; nothing was committed).
    luisa::vector<uint32_t> fresh;
    fresh.reserve(static_cast<size_t>(required - committed) * _region_count);
    for (uint32_t r = 0u; r < _region_count; ++r) {
        const uint32_t base = r * _tiles_per_region;
        for (uint64_t t = committed; t < required; ++t) {
            const uint32_t g = base + static_cast<uint32_t>(t);
            _heaps[g] = _device.allocate_sparse_buffer_heap(_tile_size_bytes);
            if (!_heaps[g]) [[unlikely]] {
                for (const auto idx : fresh) { _heaps[idx] = SparseBufferHeap{}; }
                return Status::error(
                    StatusCode::out_of_memory,
                    "sparse heap allocation failed while growing the KV cache",
                    "allocate_sparse_buffer_heap");
            }
            fresh.emplace_back(g);
        }
    }

    // One map batch for all regions. Existing mappings and their data are
    // untouched; the same-stream commit orders the maps before any kernel
    // that follows (and CUDA's map is host-immediate anyway). No host sync.
    SparseCommandList maps;
    for (uint32_t r = 0u; r < _region_count; ++r) {
        const uint32_t base = r * _tiles_per_region;
        for (uint64_t t = committed; t < required; ++t) {
            const uint32_t g = base + static_cast<uint32_t>(t);
            maps << _buffer.map_tile(g, 1u, _heaps[g]);
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
    const uint64_t committed = committed_tiles_per_region();
    if (committed > 0u) {
        // Mapped tiles are a contiguous [0, committed) run per region.
        SparseCommandList unmaps;
        for (uint32_t r = 0u; r < _region_count; ++r) {
            unmaps << _buffer.unmap_tile(r * _tiles_per_region,
                                         static_cast<uint32_t>(committed));
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
        _heaps.resize(_total_tiles);// fresh empty heap slots for the next session
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
    // tile range (coalesced per region) and retire the buffer + heaps with a
    // completion callback so no resource dies while the GPU can still touch
    // it. No callback captures `this`; the final synchronize keeps teardown
    // deterministic.
    _stream.synchronize();
    const uint64_t committed = committed_tiles_per_region();
    if (committed > 0u) {
        SparseCommandList unmaps;
        for (uint32_t r = 0u; r < _region_count; ++r) {
            unmaps << _buffer.unmap_tile(r * _tiles_per_region,
                                         static_cast<uint32_t>(committed));
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
