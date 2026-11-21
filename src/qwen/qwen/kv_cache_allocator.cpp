// Qwen3.5-4B full-attention KV cache -- sparse virtual-buffer allocator.
// See kv_cache_allocator.h for the contract; this file only implements the
// host-side residency/lifetime choreography and the dtype/layout-aware page set.

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
                                    uint32_t layer_count, uint64_t page_bytes_hint,
                                    QwenKvDType dtype, uint32_t kv_heads,
                                    QwenKvLayout layout) noexcept {
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
    // Head geometry and the dtype/layout contract. `kv_heads` is mandatory for the
    // head-major blocked layout: deriving it from values_per_token would mean
    // guessing the attention width, and a wrong guess would silently mis-address
    // every K/V (the exact failure mode this contract exists to prevent).
    uint32_t heads = kv_heads;
    if (heads == 0u) {
        if (layout == QwenKvLayout::blocked64) {
            return Status::error(StatusCode::invalid_argument,
                                 "the blocked64 KV layout addresses heads explicitly, so kv_heads is required",
                                 "kv_heads");
        }
        heads = 1u;// unused by the legacy position-major geometry
    }
    if (values_per_token % heads != 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "kv_heads must divide values_per_token (= kv_heads * head_dim)",
                             "kv_heads");
    }
    const uint32_t head_dim = values_per_token / heads;
    QwenKvLayoutDesc desc{};
    if (auto s = qwen_kv_layout_desc(dtype, layout, max_context, heads, head_dim, desc);
        !s.ok()) [[unlikely]] {
        return s;
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

    // Per-region payload in WORDS (one word is 4 B for both dtypes), the region
    // byte count, and the page-aligned word stride (ceil-div by the page word
    // count; valid for any page size, power-of-two or not).
    const uint64_t region_words = desc.region_words;
    uint64_t logical_bytes = 0;
    if (!checked_mul(region_words, sizeof(float), logical_bytes)) [[unlikely]] {
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
    const uint64_t page_words = page_bytes / sizeof(float);
    uint64_t stride_words = 0;
    if (!checked_mul(pages_per_region, page_words, stride_words)) [[unlikely]] {
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
    _page_flags.assign(static_cast<size_t>(pages_per_region), 0u);
    _desc = desc;
    _stride_words = stride_words;
    _region_words = region_words;
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
    _committed_pages = 0u;
    _committed_tokens = 0u;

    // Dense per-region FP8 scales (only for the scaled dtype). They live OUTSIDE
    // the sparse buffer, so reading the scale of any addressable block is always a
    // mapped read; zero at create() is the documented "this block was never
    // written" sentinel (qwen/kv_cache_layout.h). The upload borrows `_scale_zero`,
    // a member, so it outlives the command even if the stream runs it later.
    if (desc.scaled()) {
        _scale_zero.assign(static_cast<size_t>(desc.scale_count), 0.0f);
        const auto zero = luisa::span<const float>{_scale_zero};
        _k_scales.reserve(layer_count);
        _v_scales.reserve(layer_count);
        for (uint32_t layer = 0u; layer < layer_count; ++layer) {
            _k_scales.emplace_back(_device.create_buffer<float>(desc.scale_count));
            _v_scales.emplace_back(_device.create_buffer<float>(desc.scale_count));
            if (!_k_scales.back() || !_v_scales.back()) [[unlikely]] {
                return Status::error(StatusCode::out_of_memory,
                                     "KV cache fp8 scale buffer allocation failed",
                                     "create_buffer");
            }
            stream << _k_scales.back().view().copy_from(zero);
            stream << _v_scales.back().view().copy_from(zero);
        }
    }
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
        LUISA_WARNING("Qwen KV cache ({} KV) needs {} heap allocations ({} regions x {} pages/region), "
                   "above the common Vulkan maxMemoryAllocationCount of {}; raise "
                   "kv_page_bytes to at least {} B ({} pages/region) to fit",
                   qwen_kv_dtype_name(dtype), heap_slots, region_count, pages_per_region,
                   kCommonVkAllocationLimit, suggested_page_bytes,
                   logical_bytes / suggested_page_bytes +
                       (logical_bytes % suggested_page_bytes != 0u ? 1u : 0u));
    }

    // Honest residency report: the payload geometry, the block padding and what
    // one page covers in TOKEN terms for this dtype/layout. At f32/position_major
    // a 2 MiB page covers page_words / values_per_token positions (512 at the model
    // geometry). At bf16/blocked64 that arithmetic still describes the BYTES per
    // token (1024 positions at 2 KiB/token) but NOT the commit granularity, which
    // is one page per head at minimum because each head owns a separate block
    // array -- so the minimum for one token is printed too instead of claiming the
    // flattering per-page number.
    const uint64_t positions_per_page = desc.positions_per_page_per_head(page_words);
    const uint64_t first_commit_pages = min_commit_pages();
    const double first_commit_mib =
        static_cast<double>(first_commit_pages) * static_cast<double>(page_bytes) /
        (1024.0 * 1024.0);
    LUISA_INFO("Qwen KV cache (sparse): {} layers, {} tokens/layer, {} values/token "
               "({} kv heads x {} dim), dtype {}, layout {}, {} code(s)/word, "
               "tile {} B, page {} B ({} positions/head at this geometry), {} tiles/page, "
               "{} pages/region, {} heaps, {} words/region ({} B payload, {} B with block "
               "padding), stride {} words, {} regions, {} tiles virtual ({:.2f} MiB); "
               "first-token residency {} page(s)/region = {:.2f} MiB",
               layer_count, max_context, values_per_token, desc.kv_heads, desc.head_dim,
               qwen_kv_dtype_name(dtype), qwen_kv_layout_name(layout),
               desc.elements_per_word(), tile_bytes, page_bytes, positions_per_page,
               page_tiles, pages_per_region, heap_slots, region_words,
               desc.payload_bytes(), desc.region_bytes(), stride_words,
               region_count, total_tiles, static_cast<double>(total_bytes) / 1048576.0,
               first_commit_pages, first_commit_mib);
if (desc.scaled()) {
        LUISA_INFO("Qwen KV cache (sparse): {} dense FP32 scales/region "
                   "([{} kv heads][{} position blocks of {} positions], {} B/region, "
                   "{} B total), outside the sparse mapping -- value = code * scale",
                   desc.scale_count, desc.kv_heads, desc.position_blocks,
                   qwen_kv_position_block, desc.scale_bytes(), scale_bytes());
}
    return Status::success();
}

uint32_t QwenKvCacheAllocator::min_commit_pages() const noexcept {
    if (_pages_per_region == 0u) { return 0u; }
    luisa::vector<uint8_t> flags;
    return residency_for(1u, flags);
}

uint32_t QwenKvCacheAllocator::residency_for(uint32_t tokens,
                                             luisa::vector<uint8_t> &flags) const noexcept {
    flags.assign(static_cast<size_t>(_pages_per_region), 0u);
    if (_pages_per_region == 0u || tokens == 0u) { return 0u; }
    const uint64_t page_words = static_cast<uint64_t>(_page_size_bytes) / sizeof(float);
    const uint64_t clamped = std::min<uint64_t>(tokens, _max_context);
    if (_desc.layout == QwenKvLayout::position_major) {
        // Contiguous prefix: legacy behavior, the resident set is pages [0, n).
        const uint64_t codes = _desc.prefix_codes(static_cast<uint32_t>(clamped));
        const uint64_t words = codes / _desc.elements_per_word();
        const uint64_t pages = words / page_words + (words % page_words != 0u ? 1u : 0u);
        const uint64_t capped = std::min<uint64_t>(pages, _pages_per_region);
        for (uint64_t p = 0u; p < capped; ++p) { flags[static_cast<size_t>(p)] = 1u; }
        return static_cast<uint32_t>(capped);
    }
    // blocked64: one contiguous run per head; mark every page it intersects.
    // A run is a whole number of 64-position blocks (the unit that must stay
    // contiguous), so a page that only partially covers a run is still mapped
    // whole -- a page is the allocation unit and there is no partial mapping.
    for (uint32_t head = 0u; head < _desc.kv_heads; ++head) {
        const uint64_t run = _desc.head_run_codes(head, static_cast<uint32_t>(clamped));
        if (run == 0u) { continue; }
        const uint64_t base_code = _desc.head_base_codes(head);
        const uint64_t base_word = _desc.word_index(base_code);
        const uint64_t run_words = run / _desc.elements_per_word();
        const uint64_t first = base_word / page_words;
        const uint64_t last = (base_word + run_words - 1u) / page_words;
        for (uint64_t p = first; p <= last && p < _pages_per_region; ++p) {
            flags[static_cast<size_t>(p)] = 1u;
        }
    }
    uint32_t count = 0u;
    for (const auto f : flags) { count += f != 0u ? 1u : 0u; }
    return count;
}

compute::BufferView<float> QwenKvCacheAllocator::k_view(uint32_t layer) const noexcept {
    LUISA_ASSERT(valid() && layer < _layer_count,
                 "QwenKvCacheAllocator::k_view({}) on an allocator with {} layers",
                 layer, _layer_count);
    // Region order: K then V for each layer.
    return _buffer.view(region_base_elements(layer * 2u), _region_words);
}

compute::BufferView<float> QwenKvCacheAllocator::v_view(uint32_t layer) const noexcept {
    LUISA_ASSERT(valid() && layer < _layer_count,
                 "QwenKvCacheAllocator::v_view({}) on an allocator with {} layers",
                 layer, _layer_count);
    return _buffer.view(region_base_elements(layer * 2u + 1u), _region_words);
}

compute::BufferView<uint32_t> QwenKvCacheAllocator::k_word_view(uint32_t layer) const noexcept {
    return k_view(layer).as<uint32_t>();
}

compute::BufferView<uint32_t> QwenKvCacheAllocator::v_word_view(uint32_t layer) const noexcept {
    return v_view(layer).as<uint32_t>();
}

compute::BufferView<float> QwenKvCacheAllocator::k_scale_view(uint32_t layer) const noexcept {
    LUISA_ASSERT(valid() && _desc.scaled() && layer < _layer_count,
                 "QwenKvCacheAllocator::k_scale_view({}) on a {} allocator with {} layers",
                 layer, qwen_kv_dtype_name(_desc.dtype), _layer_count);
    return _k_scales[layer].view();
}

compute::BufferView<float> QwenKvCacheAllocator::v_scale_view(uint32_t layer) const noexcept {
    LUISA_ASSERT(valid() && _desc.scaled() && layer < _layer_count,
                 "QwenKvCacheAllocator::v_scale_view({}) on a {} allocator with {} layers",
                 layer, qwen_kv_dtype_name(_desc.dtype), _layer_count);
    return _v_scales[layer].view();
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
    // Residency is page-granular and layout-dependent (see residency_for()): the
    // request is rounded up to whole pages / whole 64-position blocks, which is
    // up to one page beyond the token watermark reported by committed_tokens().
    luisa::vector<uint8_t> target;
    const uint32_t required = residency_for(tokens, target);
    LUISA_ASSERT(required <= _pages_per_region && target.size() == _pages_per_region,
                 "QwenKvCacheAllocator page bookkeeping out of range");
    // Pages to add: every page the target set needs that is not resident yet.
    // Residency is monotone (residency_for() grows with the token count), so this
    // is normally a superset test; counting the actual additions keeps the growth
    // path correct even if a future layout made the sets non-nested.
    uint32_t add_count = 0u;
    for (uint32_t p = 0u; p < _pages_per_region; ++p) {
        if (target[p] != 0u && _page_flags[p] == 0u) { ++add_count; }
    }
    if (add_count == 0u) {
        // The committed pages already cover the request.
        _committed_pages = required;
        _committed_tokens = tokens;
        return Status::success();
    }

    // Allocate EVERY newly required page heap before mapping anything, so a
    // partial heap-allocation failure leaves the previous mapping state
    // untouched (the fresh heaps are destroyed; nothing was committed).
    luisa::vector<uint32_t> added;
    for (uint32_t r = 0u; r < _region_count; ++r) {
        const uint32_t base = r * _pages_per_region;
        for (uint32_t p = 0u; p < _pages_per_region; ++p) {
            if (target[p] == 0u || _page_flags[p] != 0u) { continue; }
            const uint32_t slot = base + p;
            // Risk R1: exactly one heap per (region, page) slot between
            // allocation and retirement. A second heap in the same slot would
            // leak the first and, on DX, alias two mappings onto one RangeStart.
            LUISA_ASSERT(!_heaps[slot],
                         "QwenKvCacheAllocator page slot {} is already mapped", slot);
            _heaps[slot] = _device.allocate_sparse_buffer_heap(_page_size_bytes);
            if (!_heaps[slot]) [[unlikely]] {
                for (const auto idx : added) { _heaps[idx] = SparseBufferHeap{}; }
                return Status::error(
                    StatusCode::out_of_memory,
                    "sparse heap allocation failed while growing the KV cache",
                    "allocate_sparse_buffer_heap");
            }
            added.emplace_back(slot);
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
        for (uint32_t p = 0u; p < _pages_per_region; ++p) {
            if (target[p] == 0u || _page_flags[p] != 0u) { continue; }
            maps << _buffer.map_tile(static_cast<uint32_t>(region_base_tile(r) +
                                                           static_cast<uint64_t>(p) * _tiles_per_page),
                                     _tiles_per_page, _heaps[base + p]);
        }
    }
    stream << maps.commit();
    // Commit the bookkeeping only after the map batch was submitted.
    _page_flags = std::move(target);
    _committed_pages = required;
    _mapped_heap_count += static_cast<uint32_t>(added.size());
    _committed_tokens = tokens;
    return Status::success();
}

void QwenKvCacheAllocator::unmap_and_retire(compute::Stream &stream) noexcept {
    const bool had_residency = _mapped_heap_count != 0u || _committed_pages != 0u;
    if (had_residency) {
        // Mapped pages are whole runs of tiles_per_page tiles; coalesce adjacent
        // page indices into one unmap_tile per run per region (at position_major
        // that is exactly the single prefix run of the pre-Q3 allocator; at
        // blocked64 it is one run per contiguous page group, so the per-head
        // interleave costs at most kv_heads records per region instead of one).
        SparseCommandList unmaps;
        for (uint32_t r = 0u; r < _region_count; ++r) {
            uint32_t p = 0u;
            while (p < _pages_per_region) {
                if (_page_flags[p] == 0u) {
                    ++p;
                    continue;
                }
                const uint32_t begin = p;
                while (p < _pages_per_region && _page_flags[p] != 0u) { ++p; }
                const uint32_t count = p - begin;
                unmaps << _buffer.unmap_tile(
                    static_cast<uint32_t>(region_base_tile(r) +
                                          static_cast<uint64_t>(begin) * _tiles_per_page),
                    count * _tiles_per_page);
            }
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
    _page_flags.assign(static_cast<size_t>(_pages_per_region), 0u);
    _mapped_heap_count = 0u;
    _committed_pages = 0u;
    _committed_tokens = 0u;
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
    const bool had_residency = _mapped_heap_count != 0u || _committed_pages != 0u;
    unmap_and_retire(stream);
    // Match the public engine reset contract: unmapping and heap retirement
    // are complete when this returns. With no prior residency, the entry drain
    // is sufficient and no second empty-stream synchronization is needed.
    // The dense fp8 scales are reset to the "never written" sentinel (0) as well:
    // the next session derives every block scale from its own values, exactly as a
    // freshly created cache would. The upload is ordered on the same stream behind
    // the unmap, and `_scale_zero` outlives it (member, same lifetime contract as
    // the allocator).
    if (!_k_scales.empty()) {
        const auto zero = luisa::span<const float>{_scale_zero};
        for (auto &buffer : _k_scales) { stream << buffer.view().copy_from(zero); }
        for (auto &buffer : _v_scales) { stream << buffer.view().copy_from(zero); }
    }
    if (had_residency) { stream.synchronize(); }
    return Status::success();
}

QwenKvCacheAllocator::~QwenKvCacheAllocator() noexcept {
    if (!valid()) {
        _heaps.clear();
        return;
    }
    // Retained device/stream handles remain valid through teardown. Drain
    // in-flight work before the host-immediate unmap path, then unmap every
    // committed page range and retire the buffer + heaps with a completion
    // callback so no resource dies while the GPU can still touch it. No callback
    // captures `this`; the final synchronize keeps teardown deterministic.
    _stream.synchronize();
    unmap_and_retire(_stream);
    auto buffer = std::move(_buffer);
    auto heaps = std::move(_heaps);
    CommandList retire = CommandList::create(0u, 1u);
    retire.add_callback([buffer = std::move(buffer),
                         heaps = std::move(heaps)]() mutable noexcept {
        heaps.clear();// destroy the heaps first, then release the buffer
        buffer = compute::SparseBuffer<float>{};
    });
    _stream << retire.commit() << compute::synchronize();
}

}// namespace luisa::qwen
