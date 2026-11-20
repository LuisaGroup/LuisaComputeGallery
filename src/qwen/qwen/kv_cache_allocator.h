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
// Residency grows on demand: one SparseBufferHeap per virtual tile, allocated
// and tile-mapped by ensure_capacity() up to the number of tokens actually in
// use. Growth touches only previously unmapped tiles, so committed capacity
// and data stay stable and in-flight work on the same stream is undisturbed
// (map-before-kernel ordering is plain same-stream ordering; on CUDA the map
// is host-immediate).
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
    // 2*layer_count tile-aligned regions (K then V per layer). No heap is
    // committed yet: committed_bytes()/mapped_heap_count() are zero on
    // success and the allocator is ready for ensure_capacity().
    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream,
                                uint32_t max_context, uint32_t values_per_token,
                                uint32_t layer_count) noexcept;

    // Commit residency for `tokens` context positions on EVERY K and V
    // region. tokens == 0 and tokens <= committed_tokens() are no-ops;
    // tokens > max_context() is invalid_argument. Newly required tiles are
    // heap-allocated first, then mapped with one SparseCommandList committed
    // on `stream` (no host synchronize on the normal growth path; the map is
    // ordered before any kernel on the same stream). On a heap-allocation
    // failure nothing is mapped: the fresh heaps are destroyed, the previous
    // committed_tokens() state is preserved, and out_of_memory is returned.
    [[nodiscard]] Status ensure_capacity(compute::Stream &stream, uint32_t tokens) noexcept;

    // Drop all residency so a new session can remap and reuse the (still
    // alive) virtual buffer. Safe after an in-flight dispatch: the stream is
    // synchronized BEFORE any unmap (CUDA's unmap is host-immediate), every
    // mapped tile range is unmapped (one coalesced run per region), and the
    // retired heap handles are destroyed by a completion callback submitted
    // behind the unmap on the same stream. Bookkeeping returns to the
    // freshly-created state. No data is zeroed: attention only reads
    // positions appended in the new session. Any unmapping/retirement is
    // complete on return, matching the public engine reset contract.
    [[nodiscard]] Status reset(compute::Stream &stream) noexcept;

    // All operations must use the stream supplied to create(). This makes the
    // destructor's drain cover every dispatch that can reference the cache.
    // True after a successful create().
    [[nodiscard]] bool valid() const noexcept { return static_cast<bool>(_buffer); }
    [[nodiscard]] uint32_t max_context() const noexcept { return _max_context; }
    [[nodiscard]] uint32_t values_per_token() const noexcept { return _values_per_token; }
    [[nodiscard]] uint32_t layer_count() const noexcept { return _layer_count; }
    // Device sparse tile granularity in bytes (0 before create()).
    [[nodiscard]] uint32_t tile_size_bytes() const noexcept { return _tile_size_bytes; }
    // Element stride of one K (or V) region inside the virtual buffer:
    // max_context*values_per_token aligned up to a whole number of tiles.
    [[nodiscard]] uint64_t cache_stride_elements() const noexcept { return _stride_elements; }
    // Exact configured K/V payload across all 2*layer_count regions,
    // excluding tile alignment padding.
    [[nodiscard]] uint64_t logical_bytes() const noexcept {
        return static_cast<uint64_t>(_region_count) * _logical_elements * sizeof(float);
    }
    // Virtual padded size of the whole sparse buffer (an exact multiple of
    // the tile size); an upper bound of the physically committable bytes.
    [[nodiscard]] uint64_t reserved_bytes() const noexcept {
        return static_cast<uint64_t>(_total_tiles) * _tile_size_bytes;
    }
    // Bytes physically resident right now: one tile per mapped heap.
    [[nodiscard]] uint64_t committed_bytes() const noexcept {
        return static_cast<uint64_t>(_mapped_heap_count) * _tile_size_bytes;
    }
    [[nodiscard]] uint32_t mapped_heap_count() const noexcept { return _mapped_heap_count; }
    // High-water token request covered by the current committed tiles. The
    // final physical tile may cover additional positions beyond this watermark.
    [[nodiscard]] uint32_t committed_tokens() const noexcept { return _committed_tokens; }

    // Dense-addressing views over one layer's K / V region. The logical size
    // is exactly max_context*values_per_token elements; the underlying
    // offset is tile-aligned and never changes across growth or reset, so a
    // view captured by an engine stays valid for the allocator's lifetime.
    // `layer` must be < layer_count().
    [[nodiscard]] compute::BufferView<float> k_view(uint32_t layer) const noexcept;
    [[nodiscard]] compute::BufferView<float> v_view(uint32_t layer) const noexcept;

private:
    // Tiles required to cover `tokens` positions in one region.
    [[nodiscard]] static uint64_t tiles_for_tokens(uint64_t tokens,
                                                   uint64_t values_per_token,
                                                   uint64_t tile_elements) noexcept {
        const uint64_t elements = tokens * values_per_token;
        return elements / tile_elements + (elements % tile_elements != 0u ? 1u : 0u);
    }
    [[nodiscard]] uint64_t committed_tiles_per_region() const noexcept {
        return tiles_for_tokens(_committed_tokens, _values_per_token, tile_elements());
    }
    [[nodiscard]] uint64_t tile_elements() const noexcept {
        return static_cast<uint64_t>(_tile_size_bytes) / sizeof(float);
    }
    [[nodiscard]] uint64_t region_base_elements(uint32_t region) const noexcept {
        return static_cast<uint64_t>(region) * _stride_elements;
    }

    compute::Device _device{};
    compute::Stream _stream{};
    compute::SparseBuffer<float> _buffer{};
    luisa::vector<compute::SparseBufferHeap> _heaps{};// one slot per virtual tile
    uint64_t _stride_elements{0};   // per-region element stride (tile-aligned)
    uint64_t _logical_elements{0};  // per-region logical length
    uint32_t _max_context{0};
    uint32_t _values_per_token{0};
    uint32_t _layer_count{0};
    uint32_t _tile_size_bytes{0};
    uint32_t _tiles_per_region{0};
    uint32_t _total_tiles{0};
    uint32_t _region_count{0};
    uint32_t _mapped_heap_count{0};
    uint32_t _committed_tokens{0};
};

}// namespace luisa::qwen
