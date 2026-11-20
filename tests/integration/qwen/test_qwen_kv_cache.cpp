// Qwen sparse KV-cache allocator device regression test (tests/integration/qwen).
//
// Usage: test_qwen_kv_cache <backend>       e.g. test_qwen_kv_cache cuda
// The backend is EXPLICIT argv[1]; there is no default and no silent skip.
// Requires no checkpoint: the allocator is driven with synthetic data only.
// Exits nonzero on any failure. The ONLY allowed skip is create() returning
// StatusCode::unsupported because the backend has no sparse-buffer support;
// every other create/ensure/reset failure is a failure.
//
// Vocabulary: tile = backend mapping granularity (never changed, 2 MiB on CUDA
// and 64 KiB on DX/VK); page = ALLOCATION unit = a whole number of tiles and at
// least 2 MiB; one heap per page, mapped by exactly one map_tile call. Every
// token count and every expected residency below is derived from the PROBED tile
// size -- nothing is assumed about a specific backend.
//
// Coverage:
//  - probe-stage allocator: tile-size discovery, on-demand commit, and
//    destruction while a dispatched write may still be in flight.
//  - main allocator: zero residency after create, view sizes/distinct
//    page-aligned region bases, idempotent ensure_capacity, growth across a
//    real PAGE boundary (the page size and the token count that crosses one are
//    derived from the probed tile size, never assumed), exact-pattern writes
//    through sparse k_view/v_view on two layer pairs, data preservation across
//    growth, reset after dispatched work (bookkeeping back to zero, buffer
//    reusable), and deferred teardown after an in-flight dispatch.
//  - page-model invariants: page_size_bytes() >= 2 MiB and an exact multiple of
//    tile_size_bytes(), tiles_per_page() == page/tile, pages_per_region() ==
//    ceil(logical bytes / page bytes), mapped_heap_count() ==
//    committed_pages_per_region() * region_count, committed_bytes() ==
//    mapped_heap_count() * page_size_bytes().
//  - NO-ALIAS PROBE (the important one): every (region, page) gets a
//    region-and-page-distinct FP32 pattern written into the FIRST and LAST tile
//    of that page and read back. Two pages ever mapped from one heap base (the
//    DX RangeStart trap) collide on the first element compared.
//  - one contiguous write straddling a page boundary inside one region.
//  - growth that crosses a page boundary preserves every earlier byte.
//  - reset() then re-ensure_capacity() to the same and to a larger token count
//    reuses the same virtual addresses (views stay valid) and remaps the right
//    page count.
//  - envelope guards fail closed: tokens > max_context -> invalid_argument,
//    total-tile overflow -> out_of_memory, unusable page hint ->
//    invalid_argument.
//  - kv_page_bytes == tile_bytes (legacy one-tile-page configuration) is
//    byte-identical to the default paged configuration for the same tokens.
#include "qwen_device_test_utils.h"

#include "qwen/kv_cache_allocator.h"

#include <luisa/core/stl/memory.h>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::qwen;
using namespace qwen_test;

namespace {

// Exact-pattern writer: out[i] = scale * i. Powers-of-two scales keep every
// product of the small indices used here bit-exact in FP32, so device and
// host patterns compare exactly.
using WriteKvKernel = Kernel1D<void(Buffer<float>, Buffer<float>,
                                    uint32_t, float, float)>;

WriteKvKernel make_write_kv_kernel() noexcept {
    return [](BufferFloat k, BufferFloat v, Var<uint> count,
              Var<float> k_scale, Var<float> v_scale) noexcept {
        set_block_size(256u, 1u, 1u);
        const auto i = dispatch_x();
        $if (i < count) {
            k.write(i, k_scale * cast<float>(i));
            v.write(i, v_scale * cast<float>(i));
        };
    };
}

// Window writer for the no-alias probe: out[base + i] = tag + i. One dispatch
// per (region, page, page end) gives every probed 32-element window a unique,
// host-reproducible tag, so a page that aliases another page is detected on the
// first compared element.
using WriteWindowKernel = Kernel1D<void(Buffer<float>, uint32_t, uint32_t, float)>;

WriteWindowKernel make_write_window_kernel() noexcept {
    return [](BufferFloat buf, Var<uint> base, Var<uint> count, Var<float> tag) noexcept {
        set_block_size(64u, 1u, 1u);
        const auto i = dispatch_x();
        $if (i < count) { buf.write(base + i, tag + cast<float>(i)); };
    };
}

[[nodiscard]] luisa::vector<float> pattern(uint32_t count, float scale) {
    luisa::vector<float> host(count);
    for (uint32_t i = 0u; i < count; ++i) {
        host[i] = scale * static_cast<float>(i);
    }
    return host;
}

[[nodiscard]] luisa::vector<float> window_pattern(float tag, uint32_t count) {
    luisa::vector<float> host(count);
    for (uint32_t i = 0u; i < count; ++i) {
        host[i] = tag + static_cast<float>(i);
    }
    return host;
}

// Spec-level ceil-div, used only to state the EXPECTED page geometry
// independently of the allocator under test.
[[nodiscard]] uint64_t ceil_div(uint64_t a, uint64_t b) noexcept {
    return a / b + (a % b != 0u ? 1u : 0u);
}

template<typename Shader>
void dispatch_write(Shader &shader, Stream &stream, BufferView<float> k, BufferView<float> v,
                    uint32_t count, float k_scale, float v_scale) {
    stream << shader(k, v, count, k_scale, v_scale).dispatch(count);
}

template<typename Shader>
void dispatch_window(Shader &shader, Stream &stream, BufferView<float> view,
                     uint32_t base, uint32_t count, float tag) {
    stream << shader(view, base, count, tag).dispatch(count);
}

[[nodiscard]] luisa::vector<float> read_back(Stream &stream, BufferView<float> view) {
    luisa::vector<float> host(view.size());
    stream << view.copy_to(luisa::span{host}) << synchronize();
    return host;
}

}// namespace

int main(int argc, char **argv) {
    unbuffered_output();
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <backend>\n", argv[0]);
        return 2;
    }
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    if (!device) {
        std::fprintf(stderr, "failed to create device for backend '%s'\n", argv[1]);
        return 2;
    }
    auto stream = device.create_stream();
    auto shader = device.compile(make_write_kv_kernel());
    auto window_shader = device.compile(make_write_window_kernel());

    // ------------------------------------------------------------------
    // Stage 1: probe the device tile granularity and cover destruction
    // after a dispatched write (no synchronize before the destructor).
    // ------------------------------------------------------------------
    constexpr uint64_t kMinPageBytes = 2ull * 1024ull * 1024ull;
    uint32_t tile_bytes = 0u;
    {
        QwenKvCacheAllocator probe;
        auto st = probe.create(device, stream, 8u, 4u, 1u);
        if (!st.ok()) {
            if (st.code == StatusCode::unsupported) {
                std::printf("SKIP: backend '%s' has no sparse-buffer support (%s)\n",
                            argv[1], st.message.c_str());
                return 0;
            }
            check(false, "probe allocator create", st.message);
            return summarize("qwen kv cache", argv[1]);
        }
        check(true, "probe allocator create");
        tile_bytes = probe.tile_size_bytes();
        check(tile_bytes >= sizeof(float) && tile_bytes % sizeof(float) == 0u,
              "probe tile is a whole number of floats");
        check(probe.committed_bytes() == 0u && probe.mapped_heap_count() == 0u &&
                  probe.committed_tokens() == 0u,
              "probe starts with zero residency");
        check(probe.ensure_capacity(stream, 2u).ok(), "probe ensure_capacity");
        // The page model allocates ONE heap per page and maps tiles_per_page
        // tiles out of it, so residency is counted in pages, not in tiles. Two
        // tokens of four floats fit in the first page of every region.
        const uint32_t probe_pages = probe.committed_pages_per_region();
        check(probe.committed_tokens() == 2u && probe_pages == 1u &&
                  probe.mapped_heap_count() == 2u * probe_pages &&
                  probe.committed_bytes() ==
                      static_cast<uint64_t>(2u * probe_pages) * probe.page_size_bytes(),
              "probe commits exactly the required page heaps");
        const uint32_t probe_write = std::min(8u, 2u * 4u);
        dispatch_write(shader, stream, probe.k_view(0), probe.v_view(0), probe_write, 0.25f, 0.125f);
        // probe destroyed here, in flight, before any synchronize.
    }
    stream.synchronize();
    check(true, "destructor after in-flight dispatch (probe)");

    // ------------------------------------------------------------------
    // Stage 2: main allocator. Every number is derived from the probed
    // tile size, so the test crosses a real PAGE boundary on any backend:
    // page_bytes follows the documented formula, and one page holds
    // page_elems/vpt tokens of `vpt` values per token.
    // ------------------------------------------------------------------
    const uint32_t vpt = 12u;  // ragged: 48 B/token vs power-of-two tiles
    const uint32_t layers = 3u;// three K/V pairs
    const uint32_t tile_elems = tile_bytes / sizeof(float);
    const uint64_t page_bytes = ceil_div(std::max(kMinPageBytes,
                                                  static_cast<uint64_t>(tile_bytes)),
                                         tile_bytes) *
                                tile_bytes;
    const uint32_t page_elems = static_cast<uint32_t>(page_bytes / sizeof(float));
    // Tokens covered by one page inside a region (floor: the page boundary falls
    // inside a token for ragged strides, which is exactly the interesting case).
    const uint32_t tokens_per_page = std::max(1u, page_elems / vpt);
    const uint32_t cross_tokens = tokens_per_page + 3u;      // crosses ONE page boundary
    const uint32_t max_context = cross_tokens + tokens_per_page / 2u;// + a partial page
    const uint32_t region_count = 2u * layers;
    // Expected residency of the configuration under test, expressed in pages.
    const auto expected_pages = [&](uint32_t tokens) {
        const uint64_t bytes = static_cast<uint64_t>(tokens) * vpt * sizeof(float);
        return static_cast<uint32_t>(std::min<uint64_t>(ceil_div(bytes, page_bytes),
                                                       ceil_div(static_cast<uint64_t>(max_context) * vpt * sizeof(float),
                                                                page_bytes)));
    };
    const auto expected_heaps = [&](uint32_t tokens) {
        return region_count * expected_pages(tokens);
    };

    auto alloc = luisa::make_unique<QwenKvCacheAllocator>();
    {
        auto st = alloc->create(device, stream, max_context, vpt, layers);
        check(st.ok(), "allocator create", st.message);
        if (!st.ok()) { return summarize("qwen kv cache", argv[1]); }
    }
    check(alloc->valid(), "allocator valid after create");
    check(alloc->max_context() == max_context && alloc->values_per_token() == vpt &&
              alloc->layer_count() == layers,
          "configured geometry accessors");
    check(alloc->committed_bytes() == 0u && alloc->mapped_heap_count() == 0u &&
              alloc->committed_tokens() == 0u,
          "zero residency after create");
    check(alloc->reset(stream).ok() && alloc->valid() &&
              alloc->committed_bytes() == 0u && alloc->mapped_heap_count() == 0u,
          "reset with no committed tiles preserves the empty buffer");
    check(alloc->logical_bytes() ==
              static_cast<uint64_t>(region_count) * max_context * vpt * sizeof(float),
          "logical_bytes is the exact configured payload");
    check(alloc->reserved_bytes() >= alloc->logical_bytes() &&
              alloc->reserved_bytes() % tile_bytes == 0u,
          "reserved_bytes is the tile-padded virtual size");
    check(alloc->cache_stride_elements() >= static_cast<uint64_t>(max_context) * vpt &&
              alloc->cache_stride_elements() % page_elems == 0u,
          "region stride is page-aligned");

    // ---- page-model invariants (spec checks 1 and 2) --------------------
    {
        const uint32_t page_size = alloc->page_size_bytes();
        const uint32_t tiles_per_page = alloc->tiles_per_page();
        const uint32_t pages_per_region = alloc->pages_per_region();
        const uint64_t logical_elems = static_cast<uint64_t>(max_context) * vpt;
        check(page_size >= kMinPageBytes && page_size % tile_bytes == 0u,
              "page_size_bytes() is >= 2 MiB and an exact multiple of the tile");
        check(tiles_per_page == page_size / tile_bytes && tiles_per_page >= 1u,
              "tiles_per_page() == page_size_bytes()/tile_size_bytes()");
        check(pages_per_region == ceil_div(logical_elems * sizeof(float), page_size),
              "pages_per_region() == ceil(logical bytes / page bytes)");
        check(alloc->cache_stride_elements() ==
                  static_cast<uint64_t>(pages_per_region) * (page_size / sizeof(float)),
              "cache_stride_elements() == pages_per_region * page elements");
        check(alloc->heap_slot_count() == region_count * pages_per_region,
              "heap_slot_count() == region_count * pages_per_region");
        check(alloc->reserved_bytes() ==
                  static_cast<uint64_t>(alloc->heap_slot_count()) * page_size,
              "reserved_bytes() == heap slots * page size");
        check(expected_pages(max_context) == pages_per_region,
              "spec page formula reproduces the allocator geometry");
    }

    // View geometry: exact dense sizes, distinct page-aligned region bases.
    {
        bool sizes_ok = true, bases_ok = true, aligned_ok = true;
        luisa::vector<uint64_t> bases;
        for (uint32_t l = 0u; l < layers; ++l) {
            const auto k = alloc->k_view(l);
            const auto v = alloc->v_view(l);
            sizes_ok = sizes_ok && k.size() == max_context * vpt && v.size() == max_context * vpt;
            bases.emplace_back(k.offset());
            bases.emplace_back(v.offset());
            aligned_ok = aligned_ok && k.offset() % page_elems == 0u &&
                         v.offset() % page_elems == 0u;
        }
        std::sort(bases.begin(), bases.end());
        bases_ok = std::adjacent_find(bases.begin(), bases.end()) == bases.end();
        check(sizes_ok, "k/v views have the exact dense logical size");
        check(bases_ok, "region bases are pairwise distinct");
        check(aligned_ok, "region bases are page-aligned");
    }

    // On-demand commit, idempotence, and argument validation.
    check(alloc->ensure_capacity(stream, 0u).ok(), "ensure_capacity(0) is a no-op");
    check(alloc->ensure_capacity(stream, 3u).ok(), "first ensure_capacity");
    const uint64_t small_commit = alloc->committed_bytes();
    check(small_commit > 0u && alloc->mapped_heap_count() == expected_heaps(3u) &&
              alloc->committed_pages_per_region() == expected_pages(3u) &&
              alloc->committed_tokens() == 3u,
          "ensure_capacity commits exactly the required page heaps");
    check(alloc->ensure_capacity(stream, 3u).ok() &&
              alloc->ensure_capacity(stream, 0u).ok() &&
              alloc->committed_bytes() == small_commit &&
              alloc->mapped_heap_count() == expected_heaps(3u),
          "ensure_capacity is idempotent below the committed level");
    {
        auto st = alloc->ensure_capacity(stream, max_context + 1u);
        check(!st.ok() && st.code == StatusCode::invalid_argument,
              "ensure_capacity rejects tokens > max_context");
        check(alloc->committed_bytes() == small_commit,
              "rejected ensure_capacity changes nothing");
    }
    {
        auto other_stream = device.create_stream();
        auto grow = alloc->ensure_capacity(other_stream, 4u);
        check(!grow.ok() && grow.code == StatusCode::invalid_argument &&
                  alloc->committed_bytes() == small_commit,
              "ensure_capacity rejects a non-creation stream");
        auto reset = alloc->reset(other_stream);
        check(!reset.ok() && reset.code == StatusCode::invalid_argument &&
                  alloc->valid() && alloc->committed_bytes() == small_commit,
              "reset rejects a non-creation stream without unmapping");
    }

    // Early data on layers 0 and 2, then growth that CROSSES a page boundary
    // (page 0 -> page 1) and therefore adds one heap per region.
    const uint32_t t_small = 3u;
    dispatch_write(shader, stream, alloc->k_view(0), alloc->v_view(0),
                   t_small * vpt, 0.25f, 0.125f);
    dispatch_write(shader, stream, alloc->k_view(2), alloc->v_view(2),
                   t_small * vpt, -0.5f, 2.0f);
    stream.synchronize();
    {
        const auto k0 = read_back(stream, alloc->k_view(0).subview(0u, t_small * vpt));
        const auto v0 = read_back(stream, alloc->v_view(0).subview(0u, t_small * vpt));
        const auto k2 = read_back(stream, alloc->k_view(2).subview(0u, t_small * vpt));
        const auto v2 = read_back(stream, alloc->v_view(2).subview(0u, t_small * vpt));
        check_exact(k0, pattern(t_small * vpt, 0.25f), "layer 0 K pattern (pre-growth)");
        check_exact(v0, pattern(t_small * vpt, 0.125f), "layer 0 V pattern (pre-growth)");
        check_exact(k2, pattern(t_small * vpt, -0.5f), "layer 2 K pattern (pre-growth)");
        check_exact(v2, pattern(t_small * vpt, 2.0f), "layer 2 V pattern (pre-growth)");
    }

    const auto k0_offset_before = alloc->k_view(0).offset();
    check(alloc->ensure_capacity(stream, cross_tokens).ok(),
          "growth across a page boundary");
    check(alloc->committed_tokens() == cross_tokens &&
              alloc->committed_pages_per_region() == expected_pages(cross_tokens) &&
              alloc->mapped_heap_count() == expected_heaps(cross_tokens) &&
              alloc->committed_bytes() >
                  static_cast<uint64_t>(expected_heaps(3u)) * alloc->page_size_bytes() &&
              alloc->committed_bytes() ==
                  static_cast<uint64_t>(alloc->mapped_heap_count()) * alloc->page_size_bytes(),
          "growth increases residency by one page per region");
    check(alloc->k_view(0).offset() == k0_offset_before &&
              alloc->k_view(0).size() == max_context * vpt &&
              alloc->v_view(2).offset() > alloc->k_view(2).offset(),
          "views are unchanged by growth");
    {
        // Early data must survive the new page mappings untouched: the first
        // page keeps its heap and its contents across the growth step.
        const auto k0 = read_back(stream, alloc->k_view(0).subview(0u, t_small * vpt));
        const auto k2 = read_back(stream, alloc->k_view(2).subview(0u, t_small * vpt));
        check_exact(k0, pattern(t_small * vpt, 0.25f), "layer 0 K preserved after growth");
        check_exact(k2, pattern(t_small * vpt, -0.5f), "layer 2 K preserved after growth");
    }

    // Full-span rewrite on layers 0 and 1 covering positions that cross the
    // page boundary inside a region; exact readback of the whole span.
    dispatch_write(shader, stream, alloc->k_view(0), alloc->v_view(0),
                   cross_tokens * vpt, 0.25f, 0.125f);
    dispatch_write(shader, stream, alloc->k_view(1), alloc->v_view(1),
                   cross_tokens * vpt, 4.0f, -0.0625f);
    stream.synchronize();
    {
        const auto k0 = read_back(stream, alloc->k_view(0).subview(0u, cross_tokens * vpt));
        const auto v0 = read_back(stream, alloc->v_view(0).subview(0u, cross_tokens * vpt));
        const auto k1 = read_back(stream, alloc->k_view(1).subview(0u, cross_tokens * vpt));
        const auto v1 = read_back(stream, alloc->v_view(1).subview(0u, cross_tokens * vpt));
        check_exact(k0, pattern(cross_tokens * vpt, 0.25f), "layer 0 K full span across pages");
        check_exact(v0, pattern(cross_tokens * vpt, 0.125f), "layer 0 V full span across pages");
        check_exact(k1, pattern(cross_tokens * vpt, 4.0f), "layer 1 K full span across pages");
        check_exact(v1, pattern(cross_tokens * vpt, -0.0625f), "layer 1 V full span across pages");
    }

    // Commit the full configured capacity and confirm further growth keeps
    // the already-written data stable.
    check(alloc->ensure_capacity(stream, max_context).ok(), "commit full capacity");
    check(alloc->committed_tokens() == max_context &&
              alloc->committed_pages_per_region() == alloc->pages_per_region() &&
              alloc->mapped_heap_count() == expected_heaps(max_context) &&
              alloc->committed_bytes() ==
                  static_cast<uint64_t>(alloc->mapped_heap_count()) * alloc->page_size_bytes(),
          "full-capacity residency accounting");
    {
        const auto k1 = read_back(stream, alloc->k_view(1).subview(0u, cross_tokens * vpt));
        check_exact(k1, pattern(cross_tokens * vpt, 4.0f), "data stable at full capacity");
    }
    // Growth from one page to ALL pages must not disturb either page: write a
    // page-1 marker and re-read it after the last growth step above.
    {
        const uint32_t marker_elems = std::min<uint32_t>(32u, vpt * 2u);
        const uint32_t marker_base = static_cast<uint32_t>(page_elems);
        dispatch_window(window_shader, stream, alloc->k_view(1),
                        marker_base, marker_elems, 4096.0f);
        stream.synchronize();
        const auto got = read_back(stream, alloc->k_view(1).subview(marker_base, marker_elems));
        check_exact(got, window_pattern(4096.0f, marker_elems),
                    "page-crossing growth preserved page-1 bytes");
    }

    // ------------------------------------------------------------------
    // Stage 3: NO-ALIAS PROBE. For every region and every committed page,
    // write a (region, page)-distinct pattern into the FIRST tile of the
    // page and another into its LAST tile, then read both back. A page
    // mapped twice out of one heap base (the DX RangeStart trap) makes two
    // windows land on the same bytes and the very first comparison fails.
    // Windows are 32 elements so the probe stays cheap while still touching
    // both page ends.
    // ------------------------------------------------------------------
    {
        const uint32_t committed_pages = alloc->committed_pages_per_region();
        const uint32_t pages_per_region = alloc->pages_per_region();
        const uint32_t window = 32u;
        const uint64_t logical_elems = static_cast<uint64_t>(max_context) * vpt;
        // First element of page p, and the last element of page p that still
        // lives inside the region's logical view (the final page of a region may
        // be partially used; its tail is mapped but not addressable through the
        // dense view, so the "last tile" probe is clamped into the view).
        const auto page_first = [&](uint32_t p) {
            return static_cast<uint32_t>(static_cast<uint64_t>(p) * page_elems);
        };
        const auto page_last_window = [&](uint32_t p) {
            const uint64_t end = std::min<uint64_t>(static_cast<uint64_t>(p + 1u) * page_elems,
                                                    logical_elems);
            return static_cast<uint32_t>(end - window);
        };
        const auto tag_for = [&](uint32_t r, uint32_t p, uint32_t end) {
            // Unique per (region, page, end) and small enough to stay exact in
            // FP32; the +i inside a window can never collide across tags.
            return static_cast<float>(1000u * (r * pages_per_region + p) + (end == 0u ? 7u : 507u));
        };
        bool windows_written = true;
        for (uint32_t r = 0u; r < region_count; ++r) {
            // Region order is [K0,V0,K1,V1,...]; resolving the view through the
            // public accessors keeps the probe backend-agnostic.
            const auto view = (r % 2u == 0u) ? alloc->k_view(r / 2u) : alloc->v_view(r / 2u);
            for (uint32_t p = 0u; p < committed_pages; ++p) {
                windows_written = windows_written && page_first(p) + window <= logical_elems &&
                                  page_last_window(p) >= page_first(p) + window;
                dispatch_window(window_shader, stream, view, page_first(p), window,
                                tag_for(r, p, 0u));
                dispatch_window(window_shader, stream, view, page_last_window(p), window,
                                tag_for(r, p, 1u));
            }
        }
        stream.synchronize();
        check(windows_written, "no-alias probe windows fit inside every page");
        bool first_ok = true, last_ok = true;
        for (uint32_t r = 0u; r < region_count; ++r) {
            const auto view = (r % 2u == 0u) ? alloc->k_view(r / 2u) : alloc->v_view(r / 2u);
            for (uint32_t p = 0u; p < committed_pages; ++p) {
                const auto got_first = read_back(stream, view.subview(page_first(p), window));
                const auto got_last = read_back(stream, view.subview(page_last_window(p), window));
                first_ok = first_ok && std::equal(got_first.begin(), got_first.end(),
                                                  window_pattern(tag_for(r, p, 0u), window).begin());
                last_ok = last_ok && std::equal(got_last.begin(), got_last.end(),
                                                window_pattern(tag_for(r, p, 1u), window).begin());
            }
        }
        check(first_ok, "no-alias probe: first-tile window of every (region, page) is distinct");
        check(last_ok, "no-alias probe: last-tile window of every (region, page) is distinct");

        // ------------------------------------------------------------------
        // Stage 4: one contiguous write straddling a page boundary inside a
        // region reads back intact (the write itself is a single dispatch over
        // a subview that spans two pages).
        // ------------------------------------------------------------------
        const uint32_t straddle_tokens = 4u;
        const uint32_t straddle_start_token = tokens_per_page - 1u;
        const uint64_t straddle_start =
            static_cast<uint64_t>(straddle_start_token) * vpt;
        const bool fits = straddle_start + straddle_tokens * vpt <= logical_elems;
        if (fits) {
            const auto start = static_cast<uint32_t>(straddle_start);
            const auto count = straddle_tokens * vpt;
            // The span starts one token before the page boundary, so it crosses
            // page 0 -> page 1 inside the region by construction.
            const bool crosses = straddle_start < page_elems &&
                                 straddle_start + count > page_elems;
            dispatch_write(shader, stream, alloc->k_view(2).subview(start, count),
                           alloc->v_view(2).subview(start, count), count, 0.5f, -1.0f);
            stream.synchronize();
            const auto got_k = read_back(stream, alloc->k_view(2).subview(start, count));
            const auto got_v = read_back(stream, alloc->v_view(2).subview(start, count));
            check(crosses && std::equal(got_k.begin(), got_k.end(), pattern(count, 0.5f).begin()),
                  "page-straddling write spans a boundary and reads back intact (K)");
            check(std::equal(got_v.begin(), got_v.end(), pattern(count, -1.0f).begin()),
                  "page-straddling write reads back intact (V)");
        } else {
            // Cannot happen for the geometry above (region >= 2 pages), but never
            // silently skip a check.
            check(false, "page-straddling write geometry", "straddle span exceeds the region view");
        }
    }

    // Reset after a dispatched write (the write is still in flight when
    // reset is entered; reset itself synchronizes before unmapping).
    dispatch_write(shader, stream, alloc->k_view(0), alloc->v_view(0), 16u, 8.0f, 8.0f);
    {
        auto st = alloc->reset(stream);
        check(st.ok(), "reset after dispatched work", st.message);
    }
    check(alloc->committed_bytes() == 0u && alloc->mapped_heap_count() == 0u &&
              alloc->committed_tokens() == 0u && alloc->committed_pages_per_region() == 0u,
          "reset drops all residency bookkeeping");
    check(alloc->valid() && alloc->reserved_bytes() % tile_bytes == 0u &&
              alloc->k_view(0).size() == max_context * vpt &&
              alloc->k_view(0).offset() == k0_offset_before,
          "buffer and views survive reset");

    // Remap and reuse: fresh page heaps back the same views; a new pattern reads
    // back exactly (content after reset is undefined until rewritten).
    check(alloc->ensure_capacity(stream, 4u).ok(), "remap after reset");
    check(alloc->mapped_heap_count() == expected_heaps(4u) &&
              alloc->committed_pages_per_region() == expected_pages(4u) &&
              alloc->committed_tokens() == 4u,
          "post-reset residency accounting");
    // Spec check 6: the same token count must remap the same page count and keep
    // every virtual address, then a larger count must add pages without moving
    // a single view.
    {
        const auto k_before = alloc->k_view(0).offset();
        const auto v_before = alloc->v_view(2).offset();
        const uint32_t pages_before = alloc->committed_pages_per_region();
        const bool same_again = alloc->ensure_capacity(stream, 4u).ok() &&
                                alloc->committed_pages_per_region() == pages_before &&
                                alloc->mapped_heap_count() == expected_heaps(4u);
        check(same_again && alloc->k_view(0).offset() == k_before &&
                  alloc->v_view(2).offset() == v_before,
              "re-ensure to the same token count keeps addresses and page count");
        const bool bigger = alloc->ensure_capacity(stream, cross_tokens).ok();
        check(bigger && alloc->committed_pages_per_region() == expected_pages(cross_tokens) &&
                  alloc->committed_pages_per_region() > pages_before &&
                  alloc->mapped_heap_count() == expected_heaps(cross_tokens) &&
                  alloc->k_view(0).offset() == k_before &&
                  alloc->v_view(2).offset() == v_before,
              "re-ensure to a larger token count remaps more pages at the same addresses");
    }
    dispatch_write(shader, stream, alloc->k_view(0), alloc->v_view(0),
                   4u * vpt, 0.25f, 0.125f);
    dispatch_write(shader, stream, alloc->k_view(2), alloc->v_view(2),
                   4u * vpt, -0.5f, 2.0f);
    stream.synchronize();
    {
        const auto k0 = read_back(stream, alloc->k_view(0).subview(0u, 4u * vpt));
        const auto v2 = read_back(stream, alloc->v_view(2).subview(0u, 4u * vpt));
        check_exact(k0, pattern(4u * vpt, 0.25f), "layer 0 K reused after reset");
        check_exact(v2, pattern(4u * vpt, 2.0f), "layer 2 V reused after reset");
    }

    // Deferred teardown coverage: dispatch without a synchronize and destroy
    // the allocator while the write may still be in flight; the destructor
    // must drain, unmap, and retire through its completion callback.
    dispatch_write(shader, stream, alloc->k_view(2), alloc->v_view(2), 4u * vpt, 1.0f, 1.0f);
    alloc = nullptr;
    stream.synchronize();
    check(true, "destructor after in-flight dispatch (main)");

    // ------------------------------------------------------------------
    // Stage 5: envelope guards and configuration equivalence.
    // ------------------------------------------------------------------
    // A page hint that is not a whole number of tiles is unusable and must fail
    // closed instead of being silently rounded (a caller-supplied residency
    // budget must never be reinterpreted).
    {
        QwenKvCacheAllocator bad;
        auto st = bad.create(device, stream, 16u, 8u, 1u,
                             static_cast<uint64_t>(tile_bytes) + 1u);
        check(!st.ok() && st.code == StatusCode::invalid_argument,
              "kv_page_bytes that is not a tile multiple fails closed");
        check(!bad.valid() && bad.heap_slot_count() == 0u,
              "rejected page hint leaves no allocator state");
    }
    // A virtual tile count above the uint32 sparse envelope is out_of_memory.
    // layer_count is derived from tiles_per_page so the envelope is exceeded on
    // every backend without allocating anything.
    {
        const uint64_t tiles_per_page = page_bytes / tile_bytes;
        const uint64_t overflow_layers = 0xFFFFFFFFull / (2ull * tiles_per_page) + 2ull;
        QwenKvCacheAllocator big;
        auto st = big.create(device, stream, 8u, 1u, static_cast<uint32_t>(overflow_layers));
        check(!st.ok() && st.code == StatusCode::out_of_memory && !big.valid(),
              "total-tile overflow fails closed with out_of_memory");
    }
    // Spec check 8: kv_page_bytes == tile_bytes (legacy one-tile pages where the
    // backend tile allows it -- the 2 MiB floor still applies on DX/VK) must be
    // functionally interchangeable with the default paged configuration: the
    // same token sequence produces byte-identical contents.
    {
        const uint32_t lvpt = 12u;
        const uint32_t llayers = 2u;
        const uint32_t ltokens = 16u;
        const uint32_t lmax = 64u;
        auto paged = luisa::make_unique<QwenKvCacheAllocator>();
        auto legacy = luisa::make_unique<QwenKvCacheAllocator>();
        auto st_paged = paged->create(device, stream, lmax, lvpt, llayers);
        auto st_legacy = legacy->create(device, stream, lmax, lvpt, llayers,
                                        static_cast<uint64_t>(tile_bytes));
        check(st_paged.ok() && st_legacy.ok(),
              "paged and one-tile-page configurations create",
              st_paged.ok() ? st_legacy.message : st_paged.message);
        check(legacy->page_size_bytes() ==
                  ceil_div(std::max(kMinPageBytes, static_cast<uint64_t>(tile_bytes)),
                           tile_bytes) *
                      tile_bytes,
              "one-tile-page hint still honors the 2 MiB page floor");
        if (st_paged.ok() && st_legacy.ok()) {
            const uint32_t n = ltokens * lvpt;
            const bool grown = paged->ensure_capacity(stream, ltokens).ok() &&
                               legacy->ensure_capacity(stream, ltokens).ok();
            check(grown, "one-tile-page equivalence growth",
                  grown ? luisa::string{} : luisa::string{"ensure_capacity failed"});
            dispatch_write(shader, stream, paged->k_view(0), paged->v_view(1),
                           n, 0.25f, -0.125f);
            dispatch_write(shader, stream, legacy->k_view(0), legacy->v_view(1),
                           n, 0.25f, -0.125f);
            stream.synchronize();
            const auto pk = read_back(stream, paged->k_view(0).subview(0u, n));
            const auto lk = read_back(stream, legacy->k_view(0).subview(0u, n));
            check(std::equal(pk.begin(), pk.end(), lk.begin()) &&
                      std::equal(pk.begin(), pk.end(), pattern(n, 0.25f).begin()),
                  "one-tile pages and paged config are byte-identical (K)");
            const auto pv = read_back(stream, paged->v_view(1).subview(0u, n));
            const auto lv = read_back(stream, legacy->v_view(1).subview(0u, n));
            check(std::equal(pv.begin(), pv.end(), lv.begin()) &&
                      std::equal(pv.begin(), pv.end(), pattern(n, -0.125f).begin()),
                  "one-tile pages and paged config are byte-identical (V)");
        } else {
            check(false, "one-tile-page equivalence", "create failed");
        }
        paged = nullptr;
        legacy = nullptr;
        stream.synchronize();
    }

    // ------------------------------------------------------------------
    // Stage 6: the REAL engine geometry (max_context 4096, 1024 values per
    // token, 8 full-attention layers = 16 regions) -- exactly the
    // configuration opt_plan/01-kv-cache-pages.md section 2.6 is about.
    // Residency is MEASURED, not assumed: a short session commits one page per
    // region (16 x 2 MiB = 32 MiB) instead of the old 7 tiles per region, and a
    // full 4096-token context needs 128 heaps instead of 4096.
    // ------------------------------------------------------------------
    {
        const uint32_t engine_vpt = 1024u;
        const uint32_t engine_layers = 8u;// ->16 regions (K and V per layer)
        const uint32_t engine_max = 4096u;
        const uint64_t region_logical = static_cast<uint64_t>(engine_max) * engine_vpt * sizeof(float);
        QwenKvCacheAllocator kv;
        auto st = kv.create(device, stream, engine_max, engine_vpt, engine_layers);
        check(st.ok(), "engine-geometry allocator create", st.message);
        if (st.ok()) {
            const uint64_t page = kv.page_size_bytes();
            const uint64_t regions = 2ull * engine_layers;
            uint64_t bytes[3] = {};
            uint32_t heaps[3] = {};
            const uint32_t tokens[3] = {128u, 512u, engine_max};
            bool grown = true;
            for (uint32_t i = 0u; i < 3u; ++i) {
                grown = grown && kv.ensure_capacity(stream, tokens[i]).ok();
                bytes[i] = kv.committed_bytes();
                heaps[i] = kv.mapped_heap_count();
            }
            // Page model expected residency, derived (never hardcoded).
            const auto expect_pages = [&](uint32_t t) {
                return static_cast<uint32_t>(
                    ceil_div(static_cast<uint64_t>(t) * engine_vpt * sizeof(float), page));
            };
            std::printf("engine geometry: page %u B, %u tiles/page, %u pages/region, "
                        "%u heap slots (maxMemoryAllocationCount 4096); committed "
                        "128 tokens = %llu B (%u heaps), 512 tokens = %llu B (%u heaps), "
                        "4096 tokens = %llu B (%u heaps)\n",
                        kv.page_size_bytes(), kv.tiles_per_page(), kv.pages_per_region(),
                        kv.heap_slot_count(),
                        static_cast<unsigned long long>(bytes[0]), heaps[0],
                        static_cast<unsigned long long>(bytes[1]), heaps[1],
                        static_cast<unsigned long long>(bytes[2]), heaps[2]);
            check(grown && kv.pages_per_region() == ceil_div(region_logical, page) &&
                      kv.heap_slot_count() == regions * kv.pages_per_region(),
                  "engine-geometry page count matches the spec formula");
            check(bytes[0] == regions * expect_pages(128u) * page &&
                      heaps[0] == regions * expect_pages(128u) &&
                      bytes[1] == regions * expect_pages(512u) * page &&
                      heaps[1] == regions * expect_pages(512u) &&
                      bytes[2] == regions * expect_pages(engine_max) * page &&
                      heaps[2] == kv.heap_slot_count(),
                  "engine-geometry committed bytes == heaps * page at 128/512/4096 tokens");
            check(kv.reset(stream).ok() && kv.committed_bytes() == 0u &&
                      kv.mapped_heap_count() == 0u,
                  "engine-geometry allocator reset");
        }
    }

    // ------------------------------------------------------------------
    // Stage 7: the long-context geometry of the plan's risk R3. At
    // max_context 262144 the default 2 MiB page still needs
    // 512 pages/region x 16 regions = 8192 heap slots, above the commonly
    // reported Vulkan maxMemoryAllocationCount of 4096. The allocator must NOT
    // clamp or fail: it reserves the 16 GiB VIRTUAL range (no heap is
    // committed) and warns with the kv_page_bytes value that would fit. Only
    // the geometry is asserted here; the warning itself goes to the log.
    // ------------------------------------------------------------------
    {
        const uint32_t long_max = 262144u;
        const uint32_t long_vpt = 1024u;
        const uint32_t long_layers = 8u;
        QwenKvCacheAllocator kv;
        auto st = kv.create(device, stream, long_max, long_vpt, long_layers);
        check(st.ok(), "long-context geometry create (reserves virtual only)", st.message);
        if (st.ok()) {
            const uint32_t page_slots = static_cast<uint32_t>(
                ceil_div(static_cast<uint64_t>(long_max) * long_vpt * sizeof(float),
                         kv.page_size_bytes()));
            check(kv.pages_per_region() == page_slots &&
                      kv.heap_slot_count() == 2ull * long_layers * page_slots &&
                      kv.heap_slot_count() > 4096u &&
                      kv.committed_bytes() == 0u && kv.mapped_heap_count() == 0u,
                  "long-context geometry exceeds 4096 heap slots without committing");
            std::printf("long-context geometry: max_context %u -> %u pages/region, "
                        "%u heap slots, %llu MiB virtual, 0 committed (R3 warning above "
                        "names the kv_page_bytes that would fit)\n",
                        long_max, kv.pages_per_region(), kv.heap_slot_count(),
                        static_cast<unsigned long long>(kv.reserved_bytes() / (1024ull * 1024ull)));
        }
    }

    return summarize("qwen kv cache", argv[1]);
}
