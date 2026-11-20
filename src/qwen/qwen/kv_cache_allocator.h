#pragma once
// Qwen3.5-4B full-attention KV cache -- sparse virtual-buffer allocator.
//
// Replaces the per-layer dense `Buffer<float>` K/V pairs (8 trunk layers for
// the DSL/Tile engines, 1 draft pair for NextN) with ONE virtual
// `SparseBuffer<float>` holding 2*layer_count regions -- per layer a K region
// then a V region. Every region is a tile-aligned slice of the virtual buffer
// whose logical view is exactly [max_context, values_per_token] elements, so
// the existing kernels keep their dense addressing (`p * values_per_token +
// channel`) through a plain BufferView; there is no block table.
//
// Allocation unit: one "page" per heap. Two sizes are involved and they are NOT
// the same thing:
//  - tile: the backend mapping granularity (`SparseBufferCreationInfo::
//    tile_size_bytes`, discovered from a one-element probe buffer). 2 MiB on
//    CUDA, 64 KiB on DX and VK. It is never changed.
//  - page: the ALLOCATION unit. page_bytes = ceil_div(max(2 MiB, tile_bytes,
//    hint), tile_bytes) * tile_bytes, so a page is always an exact whole number
//    of tiles and at least 2 MiB; tiles_per_page = page_bytes / tile_bytes.
//    The 2 MiB floor is a Vulkan hard limit, not a tuning knob: the default
//    geometry (max_context 4096, values_per_token 1024, 16 regions) would
//    otherwise need 4096 heap allocations, and VkPhysicalDeviceLimits::
//    maxMemoryAllocationCount is commonly 4096 -- one heap per tile is a latent
//    out_of_memory on VK and 4096 D3D12MA allocations on DX. CUDA already
//    reports a 2 MiB tile, so there it degenerates to tiles_per_page == 1.
//
// ONE HEAP == ONE PAGE, mapped by exactly ONE map_tile call with
// tile_count = tiles_per_page. Mapping several tile ranges out of one heap is
// NOT possible portably: DX computes RangeStart from the heap base (two mappings
// of one heap would alias), VK binds at the allocation base and CUDA maps from
// handle offset 0. The no-alias invariant is therefore "one heap appears in
// exactly one mapping", and it is what the allocator's regression test probes.
//
// Region geometry: the per-region stride is padded to a whole number of pages,
// so every region starts page-aligned and every page is full (no partial page at
// a region end, which is what would let one region's last page overlap the next
// region's first tile).
//
// Residency grows on demand: ensure_capacity() allocates and page-maps the heaps
// covering the number of tokens actually in use. Growth touches only previously
// unmapped pages, so committed capacity and data stay stable and in-flight work
// on the same stream is undisturbed (map-before-kernel ordering is plain
// same-stream ordering; on CUDA the map is host-immediate).
//
// Documented trade-off (see opt_plan/01-kv-cache-pages.md section 2.6): growth
// granularity becomes one page = up to 512 tokens of context per region on
// DX/VK (2 MiB / (1024 floats per token * 4 B)), instead of one 64 KiB tile =
// 16 tokens. A 100-token session therefore commits 1 page * 16 regions = 32 MiB
// instead of 7 tiles * 16 = 7 MiB, and it never exceeds the old worst case at any
// context >= 512 tokens. Pass page_bytes_hint = 256 KiB (with the knowledge that
// the caller is trading allocation count for granularity) if a future variant
// needs many short sessions; partial page mapping is not implemented because it
// cannot be expressed portably (see above).
//
// Lifetime edges follow the in-tree sparse contract (see
// LuisaCompute/src/tests/integration/ml/paged_attention.{h,cpp}):
//  - reset()/teardown synchronize BEFORE unmapping (CUDA's sparse unmap is
//    host-immediate) and retire heap handles through a
//    CommandList::add_callback completion capture submitted on the same
//    stream, so destruction happens only after the unmap and every preceding
//    command. No callback captures `this`; no callback submits or
//    synchronizes.
//  - The allocator retains Device and Stream handles. Every public operation
//    must use the creation stream so teardown drains all possible last uses.
//
// Fail-closed: no sparse-buffer support, unusable geometry, or size/tile
// overflow are recoverable Status failures (unsupported / invalid_argument /
// out_of_memory). A dense-buffer fallback is never silently substituted.
//
// Host-only class: no DSL types, no RTTI, no exceptions.

#include <cstdint>

#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/sparse_buffer.h>
#include <luisa/runtime/sparse_heap.h>
#include <luisa/runtime/stream.h>

#include "qwen/common.h"

namespace luisa::qwen {

class QwenKvCacheAllocator {
public:
    QwenKvCacheAllocator() noexcept = default;
    ~QwenKvCacheAllocator() noexcept;
    QwenKvCacheAllocator(const QwenKvCacheAllocator &) = delete;
    QwenKvCacheAllocator &operator=(const QwenKvCacheAllocator &) = delete;
    // A live sparse mapping needs an explicit ordered teardown. Implicit move
    // assignment would destroy the destination buffer before that teardown.
    QwenKvCacheAllocator(QwenKvCacheAllocator &&) = delete;
    QwenKvCacheAllocator &operator=(QwenKvCacheAllocator &&) = delete;

    // Validate the geometry, probe backend sparse-buffer support and the
    // device tile granularity, then create the empty virtual buffer sized
    // 2*layer_count page-aligned regions (K then V per layer). No heap is
    // committed yet: committed_bytes()/mapped_heap_count() are zero on
    // success and the allocator is ready for ensure_capacity().
    //
    // `page_bytes_hint` is the desired allocation unit in bytes; 0 keeps the
    // 2 MiB default. A nonzero hint must be an exact multiple of the probed
    // tile size (the alignment cannot be finer than a backend tile), otherwise
    // create() fails closed with invalid_argument rather than silently rounding
    // a caller-supplied residency budget upwards.
    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream,
                                uint32_t max_context, uint32_t values_per_token,
                                uint32_t layer_count,
                                uint64_t page_bytes_hint = 0u) noexcept;

    // Commit residency for `tokens` context positions on EVERY K and V
    // region. tokens == 0 and tokens <= committed_tokens() are no-ops;
    // tokens > max_context() is invalid_argument. Newly required pages are
    // heap-allocated first, then mapped with one SparseCommandList committed
    // on `stream` (one map_tile per page, tile_count = tiles_per_page; no host
    // synchronize on the normal growth path -- the map is ordered before any
    // kernel on the same stream). On a heap-allocation failure nothing is
    // mapped: the fresh heaps are destroyed, the previous committed_tokens()
    // state is preserved, and out_of_memory is returned.
    [[nodiscard]] Status ensure_capacity(compute::Stream &stream, uint32_t tokens) noexcept;

    // Drop all residency so a new session can remap and reuse the (still
    // alive) virtual buffer. Safe after an in-flight dispatch: the stream is
    // synchronized BEFORE any unmap (CUDA's unmap is host-immediate), every
    // committed page range is unmapped (one coalesced run of whole pages per
    // region), and the retired heap handles are destroyed by a completion
    // callback submitted behind the unmap on the same stream. Bookkeeping
    // returns to the freshly-created state. No data is zeroed: attention only
    // reads positions appended in the new session. Any unmapping/retirement is
    // complete on return, matching the public engine reset contract.
    [[nodiscard]] Status reset(compute::Stream &stream) noexcept;

    // All operations must use the stream supplied to create(). This makes the
    // destructor's drain cover every dispatch that can reference the cache.
    // True after a successful create().
    [[nodiscard]] bool valid() const noexcept { return static_cast<bool>(_buffer); }
    [[nodiscard]] uint32_t max_context() const noexcept { return _max_context; }
    [[nodiscard]] uint32_t values_per_token() const noexcept { return _values_per_token; }
    [[nodiscard]] uint32_t layer_count() const noexcept { return _layer_count; }
    // Device sparse tile (mapping granularity) in bytes (0 before create()).
    [[nodiscard]] uint32_t tile_size_bytes() const noexcept { return _tile_size_bytes; }
    // Allocation unit in bytes: an exact multiple of tile_size_bytes() and at
    // least 2 MiB (0 before create()).
    [[nodiscard]] uint32_t page_size_bytes() const noexcept { return _page_size_bytes; }
    // Backend tiles per page: page_size_bytes() / tile_size_bytes(), >= 1.
    [[nodiscard]] uint32_t tiles_per_page() const noexcept { return _tiles_per_page; }
    // Pages per K (or V) region: ceil(logical bytes / page_size_bytes()).
    [[nodiscard]] uint32_t pages_per_region() const noexcept { return _pages_per_region; }
    // Pages currently resident in one region: ceil(committed tokens *
    // values_per_token * 4 / page_size_bytes()), clamped to pages_per_region().
    [[nodiscard]] uint32_t committed_pages_per_region() const noexcept;
    // Size of the heap-slot table: region_count * pages_per_region(), i.e. the
    // total number of page allocations a fully resident cache would hold. This
    // is the number Vulkan's maxMemoryAllocationCount compares against.
    [[nodiscard]] uint32_t heap_slot_count() const noexcept {
        return static_cast<uint32_t>(static_cast<uint64_t>(_region_count) * _pages_per_region);
    }
    // Element stride of one K (or V) region inside the virtual buffer:
    // pages_per_region * page_elements, i.e. a whole number of pages.
    [[nodiscard]] uint64_t cache_stride_elements() const noexcept { return _stride_elements; }
    // Exact configured K/V payload across all 2*layer_count regions,
    // excluding page alignment padding.
    [[nodiscard]] uint64_t logical_bytes() const noexcept {
        return static_cast<uint64_t>(_region_count) * _logical_elements * sizeof(float);
    }
    // Virtual padded size of the whole sparse buffer (an exact multiple of
    // the tile size, and of the page size); an upper bound of the physically
    // committable bytes.
    [[nodiscard]] uint64_t reserved_bytes() const noexcept {
        return static_cast<uint64_t>(_total_tiles) * _tile_size_bytes;
    }
    // Bytes physically resident right now: one page per mapped heap.
    [[nodiscard]] uint64_t committed_bytes() const noexcept {
        return static_cast<uint64_t>(_mapped_heap_count) * _page_size_bytes;
    }
    [[nodiscard]] uint32_t mapped_heap_count() const noexcept { return _mapped_heap_count; }
    // High-water token request covered by the current committed pages. The
    // final physical page may cover additional positions beyond this watermark
    // (up to a whole page, i.e. 512 tokens per region at the default geometry).
    [[nodiscard]] uint32_t committed_tokens() const noexcept { return _committed_tokens; }

    // Dense-addressing views over one layer's K / V region. The logical size
    // is exactly max_context*values_per_token elements; the underlying
    // offset is page-aligned and never changes across growth or reset, so a
    // view captured by an engine stays valid for the allocator's lifetime.
    // `layer` must be < layer_count().
    [[nodiscard]] compute::BufferView<float> k_view(uint32_t layer) const noexcept;
    [[nodiscard]] compute::BufferView<float> v_view(uint32_t layer) const noexcept;

private:
    // Pages required to cover `tokens` positions in one region.
    [[nodiscard]] static uint64_t pages_for_tokens(uint64_t tokens,
                                                   uint64_t values_per_token,
                                                   uint64_t page_bytes) noexcept {
        const uint64_t bytes = tokens * values_per_token * sizeof(float);
        return bytes / page_bytes + (bytes % page_bytes != 0u ? 1u : 0u);
    }
    [[nodiscard]] uint64_t tile_elements() const noexcept {
        return static_cast<uint64_t>(_tile_size_bytes) / sizeof(float);
    }
    // First backend tile index of one region, i.e. the region stride expressed
    // in tiles. Every region occupies a whole number of pages, so region bases
    // are page-aligned by construction and page p of a region starts at
    // region_base_tile(r) + p * tiles_per_page.
    [[nodiscard]] uint64_t region_base_tile(uint32_t region) const noexcept {
        return static_cast<uint64_t>(region) * _tiles_per_region;
    }
    [[nodiscard]] uint64_t region_base_elements(uint32_t region) const noexcept {
        return static_cast<uint64_t>(region) * _stride_elements;
    }

    compute::Device _device{};
    compute::Stream _stream{};
    compute::SparseBuffer<float> _buffer{};
    luisa::vector<compute::SparseBufferHeap> _heaps{};// one slot per (region, page)
    uint64_t _stride_elements{0};   // per-region element stride (page-aligned)
    uint64_t _logical_elements{0};  // per-region logical length
    uint32_t _max_context{0};
    uint32_t _values_per_token{0};
    uint32_t _layer_count{0};
    uint32_t _tile_size_bytes{0};
    uint32_t _page_size_bytes{0};
    uint32_t _tiles_per_page{0};
    uint32_t _tiles_per_region{0};// = pages_per_region * tiles_per_page (region stride in tiles)
    uint32_t _pages_per_region{0};
    uint32_t _total_tiles{0};
    uint32_t _region_count{0};
    uint32_t _mapped_heap_count{0};
    uint32_t _committed_tokens{0};
};

}// namespace luisa::qwen
