#pragma once
// Qwen3.5-4B full-attention KV cache -- sparse virtual-buffer allocator.
//
// Replaces the per-layer dense `Buffer<float>` K/V pairs (8 trunk layers for
// the DSL/Tile engines, 1 draft pair for NextN) with ONE virtual
// `SparseBuffer<float>` holding 2*layer_count regions -- per layer a K region
// then a V region. Every region is a tile-aligned slice of the virtual buffer
// whose logical view is exactly one region of the configured dtype/layout
// (qwen/kv_cache_layout.h), so the kernels address it through a plain BufferView;
// there is no block table.
//
// DTYPE + LAYOUT (opt_plan/05-quantization.md section 4 item Q3, 03 section 4.3):
// the allocator stores 32-bit WORDS and every element index is decided by
// `QwenKvLayoutDesc` (qwen/kv_cache_layout.h), which is the single contract the
// allocator, the engine, the attention kernels and the tests share:
//   f32  + position_major -> legacy [position][kv_head][channel], one FP32 code
//                            per word. This stays the reference/A-B path and its
//                            geometry is byte-identical to the pre-Q3 allocator.
//   bf16 + blocked64      -> [kv_head][position_block 64][position%64][channel],
//                            two RNE BF16 codes per word, low code first.
//   fp8_e4m3 + blocked64  -> the same blocked addressing, four E4M3FN codes per
//                            word, PLUS one dense FP32 scale per (kv_head,
//                            position_block): value = code * scale. The scale
//                            arrays are separate DENSE buffers owned by the
//                            allocator (one per region, kv_heads*position_blocks
//                            floats, allocated at create and cleared at reset), so
//                            a scale read can never fall into an unmapped sparse
//                            page. See qwen/kv_cache_layout.h for the scale rule.
// Both cross combinations of an UNSCALED dtype are legal (the dtype and the
// layout are independent axes) so the two effects can be measured apart;
// fp8_e4m3 is only legal with blocked64 because its scale is block-indexed.
// `dtype()`/`layout()` report what was built and nothing ever substitutes one for
// the other.
//
// The BF16 region is HALF the FP32 bytes per stored value, so a 2 MiB page covers
// twice the tokens: at the model geometry (kv_heads 4, head_dim 256, 1024 values
// per token) a region is 8 MiB of BF16 codes against 16 MiB of FP32, and a 2 MiB
// page covers 1024 context positions instead of 512 -- in the LEGACY layout. The
// blocked layout changes that in a way that must be stated honestly (see the
// residency note below), which is exactly why `positions_per_page_per_head()` and
// the create() log line exist.
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
// Residency: ensure_capacity() maps every page that INTERSECTS the payload the
// requested token count touches. Two cases:
//  * position_major (FP32 legacy): the payload is a contiguous prefix, so the
//    resident pages are a prefix -- exactly the pre-Q3 behavior.
//  * blocked64: each head owns a contiguous block array, so the touched range is
//    ONE RUN PER HEAD (head_base + ceil(tokens/64) whole 64-position blocks). The
//    union of those runs is NOT a page prefix in general (at max_context 8192 with
//    a 2 MiB page, head 0 needs pages {0}, head 1 needs {2}, ...), so residency is
//    a page BITMAP, not a count. Mapping a linear prefix instead would leave a
//    head's first block unmapped the moment the layout interleaves heads across
//    pages -- that is the bug this design exists to prevent.
//  Growth only ever ADDS pages (there is no eviction), so committed capacity and
//  existing data stay stable and in-flight work on the same stream is undisturbed
//  (map-before-kernel ordering is plain same-stream ordering; on CUDA the map is
//  host-immediate). Because the first page of every head must be mapped before a
//  single token can be attended, the minimum committing request maps
//  `min_commit_pages()` pages per region: 1 at f32 (a 512-token prefix) and, at
//  the default model geometry, one page per kv_head at bf16 (2 MiB each), i.e.
//  8 MiB instead of 2 MiB per region for a short session. That is the honest
//  granularity cost of the head-major block layout and it is reported by
//  create() rather than hidden.
//
// Documented trade-off (see opt_plan/01-kv-cache-pages.md section 2.6/2.7): growth
// granularity becomes one page, i.e. up to 512 tokens of context per region at
// FP32 (2 MiB / (1024 floats per token * 4 B)) or 1024 at BF16 in the legacy
// layout. A 100-token FP32 session therefore commits 1 page * 16 regions = 32 MiB
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
// Fail-closed: no sparse-buffer support, unusable geometry (including a BF16
// request with an odd head_dim, where a code pair would straddle a position row),
// or size/tile overflow are recoverable Status failures (unsupported /
// invalid_argument / out_of_memory). A dense-buffer fallback or a silent dtype
// substitution is never substituted.
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
#include "qwen/kv_cache_layout.h"

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
    //
    // `dtype` selects the stored code width and `layout` the addressing
    // (qwen/kv_cache_layout.h). `kv_heads` is required whenever the layout is
    // blocked64 (the region is head-major) and must divide values_per_token;
    // head_dim = values_per_token / kv_heads. For position_major it may be 0
    // (the legacy FP32 geometry does not need it) but is recorded when given.
    // A dtype/layout combination the contract rejects, an odd head_dim at BF16,
    // or a kv_heads that does not divide values_per_token all fail closed with
    // invalid_argument -- the dtype is never downgraded to FP32 implicitly.
    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream,
                                uint32_t max_context, uint32_t values_per_token,
                                uint32_t layer_count,
                                uint64_t page_bytes_hint = 0u,
                                QwenKvDType dtype = QwenKvDType::f32,
                                uint32_t kv_heads = 0u,
                                QwenKvLayout layout = QwenKvLayout::position_major) noexcept;

    // Commit residency for `tokens` context positions on EVERY K and V
    // region, i.e. every page intersecting those positions in all heads.
    // tokens == 0 and tokens <= committed_tokens() are no-ops;
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
    // resident page range is unmapped (coalesced runs of whole pages per
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
    // kv_heads * head_dim (the legacy position-major values per token).
    [[nodiscard]] uint32_t values_per_token() const noexcept { return _values_per_token; }
    [[nodiscard]] uint32_t layer_count() const noexcept { return _layer_count; }
    // Stored code width / addressing of this cache.
    [[nodiscard]] QwenKvDType kv_dtype() const noexcept { return _desc.dtype; }
    [[nodiscard]] QwenKvLayout kv_layout() const noexcept { return _desc.layout; }
    // Stored codes per 32-bit word (1 at f32, 2 at bf16).
    [[nodiscard]] uint32_t elements_per_word() const noexcept { return _desc.elements_per_word(); }
    [[nodiscard]] uint32_t kv_heads() const noexcept { return _desc.kv_heads; }
    [[nodiscard]] uint32_t head_dim() const noexcept { return _desc.head_dim; }
    // blocked64: ceil(max_context/64); 0 at position_major.
    [[nodiscard]] uint32_t position_blocks() const noexcept { return _desc.position_blocks; }
    // fp8_e4m3: kv_heads * position_blocks FP32 scales per region (0 otherwise).
    [[nodiscard]] uint32_t scales_per_region() const noexcept { return _desc.scale_count; }
    // Bytes of ALL dense scale arrays (2*layer_count regions); 0 for f32/bf16.
    [[nodiscard]] uint64_t scale_bytes() const noexcept {
        return static_cast<uint64_t>(_region_count) * _desc.scale_bytes();
    }
    // The full contract value (index arithmetic, padding, byte counts).
    [[nodiscard]] const QwenKvLayoutDesc &layout_desc() const noexcept { return _desc; }
    // Words (32-bit) addressable in ONE K or V region, padding included. The
    // views are exactly this long.
    [[nodiscard]] uint64_t region_words() const noexcept { return _region_words; }
    // Device sparse tile (mapping granularity) in bytes (0 before create()).
    [[nodiscard]] uint32_t tile_size_bytes() const noexcept { return _tile_size_bytes; }
    // Allocation unit in bytes: an exact multiple of tile_size_bytes() and at
    // least 2 MiB (0 before create()).
    [[nodiscard]] uint32_t page_size_bytes() const noexcept { return _page_size_bytes; }
    // Backend tiles per page: page_size_bytes() / tile_size_bytes(), >= 1.
    [[nodiscard]] uint32_t tiles_per_page() const noexcept { return _tiles_per_page; }
    // Pages per K (or V) region: ceil(region bytes / page_size_bytes()).
    [[nodiscard]] uint32_t pages_per_region() const noexcept { return _pages_per_region; }
    // Pages resident in one region (the same set in every region, so a count is
    // sufficient): a prefix at position_major, the union of the per-head block
    // runs at blocked64. See page_resident().
    [[nodiscard]] uint32_t committed_pages_per_region() const noexcept { return _committed_pages; }
    // Is page `page` of every region mapped right now?
    [[nodiscard]] bool page_resident(uint32_t page) const noexcept {
        return page < _page_flags.size() && _page_flags[page] != 0u;
    }
    // Number of pages one region must commit to cover ONE position at f32 /
    // position_major (a 1-page prefix) or at blocked64 (the first block of every
    // head). This is the minimum residency cost of the configured layout.
    [[nodiscard]] uint32_t min_commit_pages() const noexcept;
    // Size of the heap-slot table: region_count * pages_per_region(), i.e. the
    // total number of page allocations a fully resident cache would hold. This
    // is the number Vulkan's maxMemoryAllocationCount compares against.
    [[nodiscard]] uint32_t heap_slot_count() const noexcept {
        return static_cast<uint32_t>(static_cast<uint64_t>(_region_count) * _pages_per_region);
    }
    // Element stride of one K (or V) region inside the virtual buffer:
    // pages_per_region * page_words, i.e. a whole number of pages. "Element" is
    // one 32-bit word for both dtypes (see elements_per_word()).
    [[nodiscard]] uint64_t cache_stride_elements() const noexcept { return _stride_words; }
    // Exact configured payload across all 2*layer_count regions, excluding page
    // alignment padding. For bf16/blocked64 this INCLUDES the 64-position block
    // padding of the last block of every head (at most 63 positions per head,
    // i.e. <= 1.5 % at max_context 4096); payload_bytes() excludes it.
    // Includes the dense per-region scale arrays of a scaled dtype: they are part
    // of what the cache actually keeps resident (16 KiB at max_context 4096, 1 MiB
    // at 262144) and hiding them would flatter the FP8 capacity number.
    [[nodiscard]] uint64_t logical_bytes() const noexcept {
        return static_cast<uint64_t>(_region_count) * _region_words * sizeof(float) +
               scale_bytes();
    }
    // The same payload without the block padding, i.e. exactly
    // region_count * max_context * values_per_token * element_bytes. This is the
    // number to quote against the tier capacity model.
    [[nodiscard]] uint64_t payload_bytes() const noexcept {
        return static_cast<uint64_t>(_region_count) * _desc.payload_bytes();
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
    // final physical page may cover additional positions beyond this watermark.
    [[nodiscard]] uint32_t committed_tokens() const noexcept { return _committed_tokens; }

    // Views over one layer's K / V region, one element per 32-bit WORD:
    //  f32  -> `word_view()` is the FP32 payload itself (read it as float);
    //  bf16 -> `word_view()` is a BufferView<uint32_t> of two-code words and
    //          `k_view()` is the same memory typed as float, which is only
    //          meaningful through qwen_kv_read_word()/the device kernels (never
    //          interpret a BF16 word as a float value).
    // The logical size is exactly region_words(); the underlying offset is
    // page-aligned and never changes across growth or reset, so a view captured
    // by an engine stays valid for the allocator's lifetime. `layer` must be
    // < layer_count().
    [[nodiscard]] compute::BufferView<float> k_view(uint32_t layer) const noexcept;
    [[nodiscard]] compute::BufferView<float> v_view(uint32_t layer) const noexcept;
    [[nodiscard]] compute::BufferView<uint32_t> k_word_view(uint32_t layer) const noexcept;
    [[nodiscard]] compute::BufferView<uint32_t> v_word_view(uint32_t layer) const noexcept;
    // fp8_e4m3 only: the DENSE FP32 scale array [kv_heads][position_blocks] of one
    // region (K or V). Allocated at create(), never resized and never remapped, so
    // the view an engine captures stays valid for the allocator's lifetime. Always
    // zero-initialized (a block that was never written has scale 0) and returned to
    // zero by reset(). An assert fires for the unscaled dtypes: binding these to a
    // kernel that reads fp8 codes would be a silent layout error.
    [[nodiscard]] compute::BufferView<float> k_scale_view(uint32_t layer) const noexcept;
    [[nodiscard]] compute::BufferView<float> v_scale_view(uint32_t layer) const noexcept;

private:
    // Fill `flags` (resized to pages_per_region) with the residency the requested
    // token count needs and return how many pages that is. Pure: ensure_capacity()
    // builds the TARGET set here first and only mutates state once every new heap
    // is allocated, so a failed growth cannot leave a half-applied bitmap.
    [[nodiscard]] uint32_t residency_for(uint32_t tokens,
                                         luisa::vector<uint8_t> &flags) const noexcept;
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
        return static_cast<uint64_t>(region) * _stride_words;
    }
    // Unmap every resident page of every region (coalesced runs) and retire the
    // heaps; shared by reset() and the destructor.
    void unmap_and_retire(compute::Stream &stream) noexcept;

    compute::Device _device{};
    compute::Stream _stream{};
    compute::SparseBuffer<float> _buffer{};
    // Dense per-region fp8 scales ([kv_heads][position_blocks] each). Empty for the
    // unscaled dtypes. `_scale_zero` is the persistent host zero image both create()
    // and reset() upload from (the upload command borrows its pointer and the stream
    // may execute it after this call returns).
    luisa::vector<compute::Buffer<float>> _k_scales{};
    luisa::vector<compute::Buffer<float>> _v_scales{};
    luisa::vector<float> _scale_zero{};
    luisa::vector<compute::SparseBufferHeap> _heaps{};// one slot per (region, page)
    luisa::vector<uint8_t> _page_flags{};            // one slot per page of one region
    QwenKvLayoutDesc _desc{};
    uint64_t _stride_words{0};   // per-region word stride (page-aligned)
    uint64_t _region_words{0};   // per-region addressable words (block padding included)
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
    uint32_t _committed_pages{0};
    uint32_t _committed_tokens{0};
};

}// namespace luisa::qwen
