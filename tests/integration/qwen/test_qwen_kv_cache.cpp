// Qwen sparse KV-cache allocator device regression test (tests/integration/qwen).
//
// Usage: test_qwen_kv_cache <backend>       e.g. test_qwen_kv_cache cuda
// The backend is EXPLICIT argv[1]; there is no default and no silent skip.
// Requires no checkpoint: the allocator is driven with synthetic data only.
// Exits nonzero on any failure. The ONLY allowed skip is create() returning
// StatusCode::unsupported because the backend has no sparse-buffer support;
// every other create/ensure/reset failure is a failure.
//
// Coverage:
//  - probe-stage allocator: tile-size discovery, on-demand commit, and
//    destruction while a dispatched write may still be in flight.
//  - main allocator: zero residency after create, view sizes/distinct
//    tile-aligned region bases, idempotent ensure_capacity, growth across a
//    real tile boundary (token count derived from tile_size_bytes, never
//    assumed), exact-pattern writes through sparse k_view/v_view on two layer
//    pairs, data preservation across growth, reset after dispatched work
//    (bookkeeping back to zero, buffer reusable), and deferred teardown after
//    an in-flight dispatch.
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

[[nodiscard]] luisa::vector<float> pattern(uint32_t count, float scale) {
    luisa::vector<float> host(count);
    for (uint32_t i = 0u; i < count; ++i) {
        host[i] = scale * static_cast<float>(i);
    }
    return host;
}

template<typename Shader>
void dispatch_write(Shader &shader, Stream &stream, BufferView<float> k, BufferView<float> v,
                    uint32_t count, float k_scale, float v_scale) {
    stream << shader(k, v, count, k_scale, v_scale).dispatch(count);
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

    // ------------------------------------------------------------------
    // Stage 1: probe the device tile granularity and cover destruction
    // after a dispatched write (no synchronize before the destructor).
    // ------------------------------------------------------------------
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
        const uint32_t tile_elems = tile_bytes / sizeof(float);
        check(tile_bytes >= sizeof(float) && tile_bytes % sizeof(float) == 0u,
              "probe tile is a whole number of floats");
        check(probe.committed_bytes() == 0u && probe.mapped_heap_count() == 0u &&
                  probe.committed_tokens() == 0u,
              "probe starts with zero residency");
        check(probe.ensure_capacity(stream, 2u).ok(), "probe ensure_capacity");
        const uint32_t probe_region_tiles =
            (2u * 4u + tile_elems - 1u) / tile_elems;// tiles covering 2 tokens of 4 floats
        check(probe.committed_tokens() == 2u &&
                  probe.mapped_heap_count() == 2u * probe_region_tiles &&
                  probe.committed_bytes() == 2ull * probe_region_tiles * tile_bytes,
              "probe commits exactly the required tiles");
        const uint32_t probe_write = std::min(8u, probe_region_tiles * tile_elems);
        dispatch_write(shader, stream, probe.k_view(0), probe.v_view(0), probe_write, 0.25f, 0.125f);
        // probe destroyed here, in flight, before any synchronize.
    }
    stream.synchronize();
    check(true, "destructor after in-flight dispatch (probe)");

    // ------------------------------------------------------------------
    // Stage 2: main allocator. Geometry is derived from the probed tile
    // size so the test crosses a real tile boundary on any backend.
    // ------------------------------------------------------------------
    const uint32_t vpt = 12u;                 // ragged: 48 B/token vs power-of-two tiles
    const uint32_t layers = 3u;               // three K/V pairs
    const uint32_t tile_elems = tile_bytes / sizeof(float);
    const uint32_t cross_tokens = tile_elems / vpt + 3u;// crosses >= 1 tile boundary
    const uint32_t max_context = cross_tokens + 8u;
    const auto expected_tiles = [&](uint32_t tokens) {
        return 2u * layers * ((static_cast<uint64_t>(tokens) * vpt + tile_elems - 1u) / tile_elems);
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
              static_cast<uint64_t>(2u * layers) * max_context * vpt * sizeof(float),
          "logical_bytes is the exact configured payload");
    check(alloc->reserved_bytes() >= alloc->logical_bytes() &&
              alloc->reserved_bytes() % tile_bytes == 0u,
          "reserved_bytes is the tile-padded virtual size");
    check(alloc->cache_stride_elements() >= static_cast<uint64_t>(max_context) * vpt &&
              alloc->cache_stride_elements() % tile_elems == 0u,
          "region stride is tile-aligned");

    // View geometry: exact dense sizes, distinct tile-aligned region bases.
    {
        bool sizes_ok = true, bases_ok = true, aligned_ok = true;
        luisa::vector<uint64_t> bases;
        for (uint32_t l = 0u; l < layers; ++l) {
            const auto k = alloc->k_view(l);
            const auto v = alloc->v_view(l);
            sizes_ok = sizes_ok && k.size() == max_context * vpt && v.size() == max_context * vpt;
            bases.emplace_back(k.offset());
            bases.emplace_back(v.offset());
            aligned_ok = aligned_ok && k.offset() % tile_elems == 0u &&
                         v.offset() % tile_elems == 0u;
        }
        std::sort(bases.begin(), bases.end());
        bases_ok = std::adjacent_find(bases.begin(), bases.end()) == bases.end();
        check(sizes_ok, "k/v views have the exact dense logical size");
        check(bases_ok, "region bases are pairwise distinct");
        check(aligned_ok, "region bases are tile-aligned");
    }

    // On-demand commit, idempotence, and argument validation.
    check(alloc->ensure_capacity(stream, 0u).ok(), "ensure_capacity(0) is a no-op");
    check(alloc->ensure_capacity(stream, 3u).ok(), "first ensure_capacity");
    const uint64_t small_commit = alloc->committed_bytes();
    check(small_commit > 0u && alloc->mapped_heap_count() == expected_tiles(3u) &&
              alloc->committed_tokens() == 3u,
          "ensure_capacity commits exactly the required tiles");
    check(alloc->ensure_capacity(stream, 3u).ok() &&
              alloc->ensure_capacity(stream, 0u).ok() &&
              alloc->committed_bytes() == small_commit &&
              alloc->mapped_heap_count() == expected_tiles(3u),
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

    // Early data on layers 0 and 2, then growth across a real tile boundary.
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
          "growth across a tile boundary");
    check(alloc->committed_tokens() == cross_tokens &&
              alloc->mapped_heap_count() == expected_tiles(cross_tokens) &&
              alloc->committed_bytes() > small_commit,
          "growth increases residency");
    check(alloc->k_view(0).offset() == k0_offset_before &&
              alloc->k_view(0).size() == max_context * vpt &&
              alloc->v_view(2).offset() > alloc->k_view(2).offset(),
          "views are unchanged by growth");
    {
        // Early data must survive the new tile mappings untouched.
        const auto k0 = read_back(stream, alloc->k_view(0).subview(0u, t_small * vpt));
        const auto k2 = read_back(stream, alloc->k_view(2).subview(0u, t_small * vpt));
        check_exact(k0, pattern(t_small * vpt, 0.25f), "layer 0 K preserved after growth");
        check_exact(k2, pattern(t_small * vpt, -0.5f), "layer 2 K preserved after growth");
    }

    // Full-span rewrite on layers 0 and 1 covering positions that cross the
    // tile boundary inside a region; exact readback of the whole span.
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
        check_exact(k0, pattern(cross_tokens * vpt, 0.25f), "layer 0 K full span across tiles");
        check_exact(v0, pattern(cross_tokens * vpt, 0.125f), "layer 0 V full span across tiles");
        check_exact(k1, pattern(cross_tokens * vpt, 4.0f), "layer 1 K full span across tiles");
        check_exact(v1, pattern(cross_tokens * vpt, -0.0625f), "layer 1 V full span across tiles");
    }

    // Commit the full configured capacity and confirm further growth keeps
    // the already-written data stable.
    check(alloc->ensure_capacity(stream, max_context).ok(), "commit full capacity");
    check(alloc->committed_tokens() == max_context &&
              alloc->mapped_heap_count() == expected_tiles(max_context),
          "full-capacity residency accounting");
    {
        const auto k1 = read_back(stream, alloc->k_view(1).subview(0u, cross_tokens * vpt));
        check_exact(k1, pattern(cross_tokens * vpt, 4.0f), "data stable at full capacity");
    }

    // Reset after a dispatched write (the write is still in flight when
    // reset is entered; reset itself synchronizes before unmapping).
    dispatch_write(shader, stream, alloc->k_view(0), alloc->v_view(0), 16u, 8.0f, 8.0f);
    {
        auto st = alloc->reset(stream);
        check(st.ok(), "reset after dispatched work", st.message);
    }
    check(alloc->committed_bytes() == 0u && alloc->mapped_heap_count() == 0u &&
              alloc->committed_tokens() == 0u,
          "reset drops all residency bookkeeping");
    check(alloc->valid() && alloc->reserved_bytes() % tile_bytes == 0u &&
              alloc->k_view(0).size() == max_context * vpt &&
              alloc->k_view(0).offset() == k0_offset_before,
          "buffer and views survive reset");

    // Remap and reuse: fresh heaps back the same views; a new pattern reads
    // back exactly (content after reset is undefined until rewritten).
    check(alloc->ensure_capacity(stream, 4u).ok(), "remap after reset");
    check(alloc->mapped_heap_count() == expected_tiles(4u) && alloc->committed_tokens() == 4u,
          "post-reset residency accounting");
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

    return summarize("qwen kv cache", argv[1]);
}
