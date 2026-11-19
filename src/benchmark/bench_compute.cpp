// ---------------------------------------------------------------------------
// bench_compute.cpp - the four "compute" benchmarks of the gallery benchmark
// suite (see bench_common.h for the shared harness):
//
//   compute_radix_sort    a 4-pass 8-bit LSD radix sort of N = 2^22 `(code,
//                         slot)` pairs, block-parallel: per-block per-warp
//                         shared histogram, exclusive scan of the
//                         (block, digit) matrix, scan of the scan groups, and a
//                         warp-ballot ranked stable scatter.  The pipeline is
//                         the multi-block one of src/compute/lbvh/lbvh_sort.cpp.
//   compute_lbvh_build    a full Karras 2012 LBVH over N = 2^20 synthetic
//                         triangles: triangle AABBs, 30-bit Morton codes, the
//                         radix sort above, leaves, the `delta` /
//                         `determineRange` / `findSplit` searches, the block
//                         AABBs of the two-level reduction, and the internal
//                         node AABBs.  See src/compute/lbvh/lbvh_storage.cpp.
//   compute_ngram_hash    an open-addressing FNV-1a-64 n-gram index probe plus
//                         the shared-memory `atomic_min` corpus scan of
//                         src/compute/tokenize/ngram_kernels.cpp.
//   compute_image_pipeline a 2048x2048 FLOAT4 pipeline: procedural pattern,
//                         separable 9-tap Gaussian blur (H + V), 3x3 Sobel
//                         edges, unsharp + vignette composite.  See
//                         src/compute/image_processing.cpp.
//
// Conventions of the DSL are the ones of the reference files above: C++23, no
// exceptions (`LUISA_ASSERT`), `luisa::span` for every host transfer, `&` / `|`
// for DSL booleans (the DSL `&&` / `||` are vector-only), `$break` instead of
// `break`, and every kernel compiled through `compile_uncached()`.
//
// Every timed dispatch is self-contained: the radix sort restores its unsorted
// input from a pristine buffer *inside* the timed command list, the LBVH
// re-derives its primitive AABBs procedurally, the n-gram index is built once on
// the host (deterministically) and only probed on the device, and the image
// pipeline starts from its own pattern kernel.  Every iteration therefore does
// exactly the same work, which is what makes the samples comparable.
// ---------------------------------------------------------------------------

#include "bench_common.h"

#include <luisa/core/stl/vector.h>
#include <luisa/dsl/sugar.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>

// ---------------------------------------------------------------------------
// Device-side records.
//
// `LUISA_STRUCT` opens `namespace luisa::compute` to specialize the DSL types of
// the struct, so these definitions must stay at global scope (exactly like
// `LbvhPrim` / `LbvhNode` in src/compute/lbvh/lbvh_common.h).
//
// Both records are two `float4`, i.e. 32 bytes: one AABB is one 16-byte vector
// load, and the 32-bit handle of the LBVH node is *bit-cast* into the spare
// fourth lane instead of costing a third vector.
// ---------------------------------------------------------------------------

// xyz = AABB lo, w = primitive id (bit-cast); xyz = AABB hi, w = unused.
struct BenchPrim {
    luisa::float4 lo;
    luisa::float4 hi;
};
LUISA_STRUCT(BenchPrim, lo, hi) {};

// Internal node: packed_lo.w = left child, packed_hi.w = right child.
// Leaf:         packed_lo.w = `kInvalidHandle`, packed_hi.w = primitive id.
struct BenchNode {
    luisa::float4 packed_lo;
    luisa::float4 packed_hi;
};
LUISA_STRUCT(BenchNode, packed_lo, packed_hi) {};

namespace luisa::bench {

namespace {

using namespace luisa::compute;

// ---------------------------------------------------------------------------
// Constants shared by the kernels.
// ---------------------------------------------------------------------------

// LSD radix sort: one work-group ranks `kSortChunk` elements per block.
constexpr uint kRadixBlock = 256u;                 // threads per work-group
constexpr uint kRadixBits = 8u;                    // bits per pass
constexpr uint kRadixBins = 256u;                  // 2^kRadixBits digits
constexpr uint kRadixWarpLanes = 32u;              // ballots are 32 lanes wide
constexpr uint kRadixWarps = kRadixBlock / kRadixWarpLanes;
constexpr uint kRadixWarpDigits = kRadixBins + 1u; // + one "outside the range" slot
constexpr uint kSortItems = 16u;                   // elements per lane per block
constexpr uint kSortChunk = kRadixBlock * kSortItems;
constexpr uint kSortScanGroupLimit = 64u;          // scan groups per sort
constexpr uint kSortScanWalk = 256u;               // longest per-thread matrix walk
constexpr uint kSortPasses = 4u;                   // 4 x 8 bit = 32 bit codes
constexpr uint kMaxGroupsX = 65535u;               // DX12 groups per dimension

// LBVH.
constexpr uint kBuildBlock = 256u;    // threads per work-group of the build
constexpr uint kBuildWarps = kBuildBlock / 32u; // warps per work-group of the build
constexpr uint kReductionBlock = 32u; // leaves per block AABB of the two-level reduction
constexpr uint kInvalidHandle = 0xFFFFFFFFu;
constexpr float kSceneHalfExtent = 1.0f;

// Image pipeline.
constexpr uint kImageBlock = 16u;
constexpr uint kImageSize = 2048u;

// n-gram index: corpus, queries, n-gram length range, draft length.
constexpr uint kCorpusTokens = 2097152u; // 2^21
constexpr uint kQueryCount = 1024u;
constexpr uint kQueryStride = 8u;  // tokens per query
constexpr uint kQueryTokens = 8u;  // significant tokens per query
constexpr uint kNgramMin = 2u;
constexpr uint kNgramMax = 4u;
constexpr uint kNgramK = 4u;           // draft tokens per query
constexpr uint kNgramVocab = 64u;      // corpus vocabulary (tokens [0, 64))
constexpr uint kNgramSentinel = 1000000u; // never present in the corpus
constexpr uint kNgramScanThreads = 256u;

// ---------------------------------------------------------------------------
// Small host helpers.
// ---------------------------------------------------------------------------

[[nodiscard]] constexpr uint ceil_div_u(uint x, uint y) noexcept { return (x + y - 1u) / y; }

// Number of work-groups one sort of `count` elements needs.
[[nodiscard]] constexpr uint sort_blocks(uint count) noexcept {
    return ceil_div_u(count, kSortChunk);
}

// Blocks one scan group walks: at most `kSortScanGroupLimit` groups, and no
// thread walks more than `kSortScanWalk` rows of the matrix.
[[nodiscard]] constexpr uint sort_blocks_per_group(uint blocks) noexcept {
    auto value = ceil_div_u(blocks, kSortScanGroupLimit);
    return std::clamp(value, 1u, kSortScanWalk);
}

[[nodiscard]] constexpr uint sort_groups(uint blocks, uint blocks_per_group) noexcept {
    return ceil_div_u(blocks, blocks_per_group);
}

// ---------------------------------------------------------------------------
// Field access of `BenchNode` / `BenchPrim`.
//
// The templates accept both a `Var<T>` and the `Expr<T>` a `Buffer::read()`
// returns, and they are the only place that knows how the records pack their
// AABBs and handles.
// ---------------------------------------------------------------------------

template<typename N>
[[nodiscard]] inline Float3 aabb_lo(N &&node) noexcept {
    return make_float3(node.packed_lo.x, node.packed_lo.y, node.packed_lo.z);
}

template<typename N>
[[nodiscard]] inline Float3 aabb_hi(N &&node) noexcept {
    return make_float3(node.packed_hi.x, node.packed_hi.y, node.packed_hi.z);
}

// Left child handle of an internal node, `kInvalidHandle` for a leaf.
template<typename N>
[[nodiscard]] inline UInt child_left(N &&node) noexcept {
    return node.packed_lo.w.template bitcast<uint>();
}

// Right child handle of an internal node, the primitive id of a leaf.
template<typename N>
[[nodiscard]] inline UInt child_right(N &&node) noexcept {
    return node.packed_hi.w.template bitcast<uint>();
}

template<typename P>
[[nodiscard]] inline Float3 prim_lo(P &&prim) noexcept {
    return make_float3(prim.lo.x, prim.lo.y, prim.lo.z);
}

template<typename P>
[[nodiscard]] inline Float3 prim_hi(P &&prim) noexcept {
    return make_float3(prim.hi.x, prim.hi.y, prim.hi.z);
}

// Handles are bit-cast, not value-cast, so the full 32-bit range (including the
// `kInvalidHandle` leaf marker) survives the float lane.
[[nodiscard]] inline Float pack_handle(UInt handle) noexcept {
    return handle.bitcast<float>();
}

[[nodiscard]] inline Float pack_invalid_handle() noexcept {
    return def(kInvalidHandle).bitcast<float>();
}

[[nodiscard]] inline Float4 pack_node_plane(Float3 plane, UInt handle) noexcept {
    return make_float4(plane.x, plane.y, plane.z, pack_handle(handle));
}

// Host-side reads of the same lanes, for the structural self-check (which walks
// the read-back node array outside the DSL).
[[nodiscard]] inline uint host_child_left(const BenchNode &node) noexcept {
    return luisa::bit_cast<uint>(node.packed_lo.w);
}

[[nodiscard]] inline uint host_child_right(const BenchNode &node) noexcept {
    return luisa::bit_cast<uint>(node.packed_hi.w);
}

[[nodiscard]] inline float3 host_aabb_lo(const BenchNode &node) noexcept {
    return make_float3(node.packed_lo.x, node.packed_lo.y, node.packed_lo.z);
}

[[nodiscard]] inline float3 host_aabb_hi(const BenchNode &node) noexcept {
    return make_float3(node.packed_hi.x, node.packed_hi.y, node.packed_hi.z);
}

// ---------------------------------------------------------------------------
// Host-side scratch of the radix sort: the (block, digit) count matrix, the
// per-digit totals of every scan group, and the base of every digit's output
// range.  The three buffers are exactly the ones `lbvh_sort.cpp` allocates.
// ---------------------------------------------------------------------------

struct RadixSortBuffers {
    Buffer<uint> matrix;
    Buffer<uint> group_totals;
    Buffer<uint> digit_base;
};

[[nodiscard]] RadixSortBuffers create_radix_sort_buffers(compute::Device &device,
                                                         uint blocks, uint groups) noexcept {
    return RadixSortBuffers{
        .matrix = device.create_buffer<uint>(static_cast<size_t>(blocks) * kRadixBins),
        .group_totals = device.create_buffer<uint>(static_cast<size_t>(groups) * kRadixBins),
        .digit_base = device.create_buffer<uint>(kRadixBins),
    };
}

// ---------------------------------------------------------------------------
// The four kernels of one LSD radix-sort pass.
// ---------------------------------------------------------------------------

// Pass 1/4: the digit histogram of every block.
//
// The histogram is privatized per warp (`kRadixWarps * kRadixBins` shared
// counters, 8 KiB) so the shared atomics of the 256 lanes never contend for the
// same address, and the per-warp replicas are reduced by one thread per digit
// into the global (block, digit) matrix.
[[nodiscard]] inline auto make_radix_hist_kernel() noexcept {
    return Kernel1D{[](BufferUInt2 keys_in, BufferUInt matrix,
                       UInt count, UInt shift) noexcept {
        set_block_size(kRadixBlock);
        Shared<uint> hist{kRadixWarps * kRadixBins};
        UInt tid = thread_x();
        UInt warp = tid / kRadixWarpLanes;
        $for (i, ceil_div_u(kRadixWarps * kRadixBins, kRadixBlock)) {
            UInt slot = i * kRadixBlock + tid;
            $if (slot < kRadixWarps * kRadixBins) { hist[slot] = 0u; };
        };
        sync_block();
        UInt first = block_id().x * kSortChunk;
        $for (t, kSortItems) {
            UInt index = first + t * kRadixBlock + tid;
            $if (index < count) {
                UInt code = keys_in.read(index).x;
                hist.atomic(warp * kRadixBins + ((code >> shift) & (kRadixBins - 1u)))
                    .fetch_add(1u);
            };
        };
        sync_block();
        $if (tid < kRadixBins) {
            UInt sum = def(0u);
            $for (w, kRadixWarps) { sum = sum + hist[w * kRadixBins + tid]; };
            matrix.write(block_id().x * kRadixBins + tid, sum);
        };
    }};
}

// Pass 2/4: the exclusive scan of the (block, digit) matrix.
//
// One work-group per scan group, one thread per digit, walking the blocks of the
// group in order: `matrix[block][digit]` becomes the number of elements of that
// digit in the preceding blocks *of the same group*, and the group's total is
// written to `group_totals`.
[[nodiscard]] inline auto make_radix_scan_blocks_kernel() noexcept {
    return Kernel1D{[](BufferUInt matrix, BufferUInt group_totals,
                       UInt blocks, UInt blocks_per_group) noexcept {
        set_block_size(kRadixBins);
        UInt digit = thread_x();
        UInt begin = block_id().x * blocks_per_group;
        UInt end = min(begin + blocks_per_group, blocks);
        UInt sum = def(0u);
        $for (block, begin, end) {
            UInt value = matrix.read(block * kRadixBins + digit);
            matrix.write(block * kRadixBins + digit, sum);
            sum = sum + value;
        };
        group_totals.write(block_id().x * kRadixBins + digit, sum);
    }};
}

// Pass 3/4: the exclusive scan of the scan groups' totals, plus the base of every
// digit's output range.
//
// `group_totals` is scanned in place, so a block's digit `d` starts at
// `digit_base[d] + <counts of d in the preceding blocks>`.  The first term makes
// the whole output range of `d` contiguous, the second keeps the blocks of `d`
// in input order - i.e. exactly the LSD order a stable sort needs.
[[nodiscard]] inline auto make_radix_scan_groups_kernel() noexcept {
    return Kernel1D{[](BufferUInt group_totals, BufferUInt digit_base,
                       UInt groups) noexcept {
        set_block_size(kRadixBins);
        Shared<uint> total{kRadixBins};
        UInt digit = thread_x();
        UInt sum = def(0u);
        $for (group, groups) {
            UInt value = group_totals.read(group * kRadixBins + digit);
            group_totals.write(group * kRadixBins + digit, sum);
            sum = sum + value;
        };
        total[digit] = sum;
        sync_block();
        // one sequential 256-entry scan is cheaper than a block scan here
        $if (digit == 0u) {
            UInt prefix = def(0u);
            $for (d, kRadixBins) {
                UInt value = total[d];
                digit_base.write(d, prefix);
                prefix = prefix + value;
            };
        };
    }};
}

// Pass 4/4: the ranked stable scatter, one warp per sub-chunk.
//
// A lane keeps its `kSortItems` keys in registers, so the scatter never reads the
// block's range a second time.  The lanes of the same digit are found with one
// warp-wide bit-or per digit bit: `matched` shrinks to exactly the lanes whose
// digit equals this lane's digit, so the lane's rank inside its warp sub-chunk is
// the popcount of the preceding matching lanes plus the warp's running count.
// Cross-warp order comes from the exclusive scan of `warp_hist`, and the block's
// output offset (computed by passes 1-3) is the scan's starting value - so the
// output is byte-identical to the single-work-group reference.
[[nodiscard]] inline auto make_radix_scatter_kernel() noexcept {
    return Kernel1D{[](BufferUInt2 keys_in, BufferUInt2 keys_out, BufferUInt matrix,
                       BufferUInt group_totals, BufferUInt digit_base,
                       UInt count, UInt shift, UInt blocks_per_group) noexcept {
        set_block_size(kRadixBlock);
        set_warp_size(static_cast<uint8_t>(kRadixWarpLanes));
        Shared<uint> warp_hist{kRadixWarps * kRadixWarpDigits};
        UInt tid = thread_x();
        UInt lane = warp_lane_id();
        UInt warp = tid / kRadixWarpLanes;
        UInt block = block_id().x;
        // warp `w` owns the contiguous span [warp_first, warp_first + items * 32)
        UInt warp_first = block * kSortChunk + warp * (kSortItems * kRadixWarpLanes);
        $for (i, ceil_div_u(kRadixWarps * kRadixWarpDigits, kRadixBlock)) {
            UInt slot = i * kRadixBlock + tid;
            $if (slot < kRadixWarps * kRadixWarpDigits) { warp_hist[slot] = 0u; };
        };
        sync_block();
        Local<uint> rank{kSortItems};
        Local<uint> code_of{kSortItems};
        Local<uint> slot_of{kSortItems};
        $for (i, kSortItems) {
            UInt index = warp_first + i * kRadixWarpLanes + lane;
            UInt code = def(0u);
            UInt key_slot = def(0u);
            // lanes past the end are ranked in the spare digit `kRadixBins`, so
            // they can never contribute to a real digit's cursor
            UInt bin = def(kRadixBins);
            $if (index < count) {
                auto key = keys_in.read(index);
                code = key.x;
                key_slot = key.y;
                bin = (code >> shift) & (kRadixBins - 1u);
            };
            code_of[i] = code;
            slot_of[i] = key_slot;
            UInt matched = def(0xffffffffu);
            $for (b, kRadixBits) {
                UInt x = (bin >> b) & 1u;
                UInt y = warp_active_bit_or(x << lane);
                matched = matched & (y ^ select(0u, 0xffffffffu, x == 0u));
            };
            UInt prefix = popcount(matched & ((1u << lane) - 1u));
            UInt total = popcount(matched);
            UInt warp_pre = warp_hist[warp * kRadixWarpDigits + bin];
            // the first lane of a digit publishes the warp's new cursor; every
            // lane of the warp reads the old value above, which keeps the ranks
            // deterministic
            $if (prefix == 0u) {
                warp_hist[warp * kRadixWarpDigits + bin] = warp_pre + total;
            };
            rank[i] = warp_pre + prefix;
        };
        sync_block();
        // per-warp exclusive scan of the digit counts, offset by this block's own
        // output offset: the digit's global base, the counts of the preceding
        // blocks of its scan group, and the groups before that
        UInt running = def(0u);
        running = digit_base.read(tid) +
                  matrix.read(block * kRadixBins + tid) +
                  group_totals.read((block / blocks_per_group) * kRadixBins + tid);
        $for (w, kRadixWarps) {
            UInt count_w = warp_hist[w * kRadixWarpDigits + tid];
            warp_hist[w * kRadixWarpDigits + tid] = running;
            running = running + count_w;
        };
        sync_block();
        $for (i, kSortItems) {
            UInt index = warp_first + i * kRadixWarpLanes + lane;
            $if (index < count) {
                UInt code = code_of[i];
                UInt bin = (code >> shift) & (kRadixBins - 1u);
                UInt pos = rank[i] + warp_hist[warp * kRadixWarpDigits + bin];
                keys_out.write(pos, make_uint2(code, slot_of[i]));
            };
        };
    }};
}

// One pass = histogram, scan of the blocks, scan of the groups, ranked scatter.
template<typename KHist, typename KScanBlocks, typename KScanGroups, typename KScatter>
void encode_radix_pass(CommandList &cmds,
                       const KHist &hist, const KScanBlocks &scan_blocks,
                       const KScanGroups &scan_groups, const KScatter &scatter,
                       const Buffer<uint2> &keys_in, const Buffer<uint2> &keys_out,
                       const RadixSortBuffers &scratch,
                       uint count, uint shift, uint blocks, uint blocks_per_group) noexcept {
    LUISA_ASSERT(blocks > 0u && blocks_per_group > 0u,
                 "empty radix-sort pass.");
    auto groups = sort_groups(blocks, blocks_per_group);
    cmds << hist(keys_in, scratch.matrix, count, shift)
                .dispatch(blocks * kRadixBlock);
    cmds << scan_blocks(scratch.matrix, scratch.group_totals, blocks, blocks_per_group)
                .dispatch(groups * kRadixBins);
    cmds << scan_groups(scratch.group_totals, scratch.digit_base, groups)
                .dispatch(kRadixBins);
    cmds << scatter(keys_in, keys_out, scratch.matrix, scratch.group_totals,
                    scratch.digit_base, count, shift, blocks_per_group)
                .dispatch(blocks * kRadixBlock);
}

// Four 8-bit passes: `keys_a -> keys_b -> keys_a -> keys_b -> keys_a`, so the
// sorted keys are left in `keys_a` (exactly like `LbvhRadixSort::sort`).
template<typename KHist, typename KScanBlocks, typename KScanGroups, typename KScatter>
void encode_radix_sort(CommandList &cmds,
                       const KHist &hist, const KScanBlocks &scan_blocks,
                       const KScanGroups &scan_groups, const KScatter &scatter,
                       const Buffer<uint2> &keys_a, const Buffer<uint2> &keys_b,
                       const RadixSortBuffers &scratch, uint count) noexcept {
    auto blocks = sort_blocks(count);
    auto blocks_per_group = sort_blocks_per_group(blocks);
    LUISA_ASSERT(blocks_per_group <= kSortScanWalk,
                 "radix sort scan walk {} exceeds {}.", blocks_per_group, kSortScanWalk);
    const Buffer<uint2> *in = &keys_a;
    const Buffer<uint2> *out = &keys_b;
    for (uint pass = 0u; pass < kSortPasses; ++pass) {
        encode_radix_pass(cmds, hist, scan_blocks, scan_groups, scatter,
                          *in, *out, scratch, count, pass * kRadixBits,
                          blocks, blocks_per_group);
        std::swap(in, out);
    }
}

// ---------------------------------------------------------------------------
// Benchmark 1: LSD radix sort.
// ---------------------------------------------------------------------------

void bench_compute_radix_sort(compute::Device &device, compute::Stream &stream,
                              const BenchConfig &cfg,
                              std::vector<BenchResult> &results) noexcept {
    constexpr uint kCount = 4194304u;// 2^22 pairs = 32 MiB of keys
    auto blocks = sort_blocks(kCount);
    auto blocks_per_group = sort_blocks_per_group(blocks);
    auto groups = sort_groups(blocks, blocks_per_group);

    // `keys_src` stays pristine: every timed iteration copies it into `keys_a`,
    // so the sort always starts from the exact same unsorted input.
    auto keys_src = device.create_buffer<uint2>(kCount);
    auto keys_a = device.create_buffer<uint2>(kCount);
    auto keys_b = device.create_buffer<uint2>(kCount);
    auto scratch = create_radix_sort_buffers(device, blocks, groups);

    // 22-bit codes: every one of the four passes sees all 256 digits, which is
    // what makes the histogram and the scatter do real work.  The slots are the
    // input order, so a stable sort is verifiable through the slot lane.
    luisa::vector<uint2> host_keys(kCount);
    uint64_t state = 0x5DEECE66Dull;
    uint32_t expected_slot_xor = 0u;
    for (uint i = 0u; i < kCount; ++i) {
        auto code = prng_u32(state) >> 10u;// 22 significant bits
        host_keys[i] = uint2{code, i};
        expected_slot_xor ^= i;
    }
    stream << keys_src.copy_from(luisa::span{host_keys}) << compute::synchronize();

    double compile_ms = 0.0;
    auto hist = compile_uncached(device, make_radix_hist_kernel(),
                                 "bench_compute_radix_sort_hist", cfg, compile_ms);
    auto scan_blocks = compile_uncached(device, make_radix_scan_blocks_kernel(),
                                        "bench_compute_radix_sort_scan_blocks", cfg, compile_ms);
    auto scan_groups = compile_uncached(device, make_radix_scan_groups_kernel(),
                                        "bench_compute_radix_sort_scan_groups", cfg, compile_ms);
    auto scatter = compile_uncached(device, make_radix_scatter_kernel(),
                                    "bench_compute_radix_sort_scatter", cfg, compile_ms);

    auto dispatch = [&] {
        // 1 device->device copy + 4 passes x 4 commands
        CommandList cmds = CommandList::create(2u + kSortPasses * 4u);
        cmds << keys_src.copy_to(keys_a);
        encode_radix_sort(cmds, hist, scan_blocks, scan_groups, scatter,
                          keys_a, keys_b, scratch, kCount);
        stream << cmds.commit();
    };

    auto result = run_timed_bench("compute", "compute_radix_sort", stream, cfg,
                                  compile_ms, static_cast<double>(kCount), "elem/s",
                                  dispatch);

    if (!cfg.verify) {
        result.verified = false;
        result.check_note = "verification disabled";
        results.push_back(std::move(result));
        return;
    }
    if (cfg.warmup_iters == 0u && cfg.timed_iters == 0u) {
        dispatch();
        stream << compute::synchronize();
    }

    // The four passes leave the sorted keys in `keys_a`.  The order check proves
    // the sort; the slot xor and sum prove that the output is a permutation of the
    // input (a lost or duplicated key changes both).
    constexpr uint64_t expected_slot_sum =
        static_cast<uint64_t>(kCount) * (static_cast<uint64_t>(kCount) - 1ull) / 2ull;
    luisa::vector<uint2> sorted(kCount);
    stream << keys_a.copy_to(luisa::span{sorted}) << compute::synchronize();
    bool ordered = true;
    bool stable = true;
    uint32_t slot_xor = 0u;
    uint64_t slot_sum = 0u;
    for (uint i = 0u; i < kCount; ++i) {
        auto key = sorted[i];
        slot_xor ^= key.y;
        slot_sum += key.y;
        if (i > 0u) {
            auto prev = sorted[i - 1u];
            if (key.x < prev.x) { ordered = false; }
            // LSD radix sort is stable, and the input is in index order, so equal
            // codes must come out with increasing slots.
            if (key.x == prev.x && key.y < prev.y) { stable = false; }
        }
    }
    bool permutation = slot_xor == expected_slot_xor && slot_sum == expected_slot_sum;
    bool ok = ordered && stable && permutation;
    result.verified = ok;
    result.check_note = luisa::format(
        "sorted={}, stable={}, permutation={} (slot xor 0x{:08x}, sum {}), "
        "blocks={}, blocks/group={}, groups={}",
        ordered, stable, permutation, slot_xor, slot_sum, blocks, blocks_per_group, groups);
    if (!ok) { LUISA_WARNING("{}: {}", result.name, result.check_note); }
    results.push_back(std::move(result));
}

// ---------------------------------------------------------------------------
// Benchmark 2: the Karras 2012 LBVH build.
// ---------------------------------------------------------------------------

// Arithmetic mixing so the synthetic triangles are procedurally reproducible
// without a random-number buffer.
[[nodiscard]] inline UInt mix_bits(UInt x) noexcept {
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    return x ^ (x >> 16u);
}

// A mixed word as a uniform float in [0, 1).
[[nodiscard]] inline Float unit_float(UInt h) noexcept {
    return cast<float>(h & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

// Pass 1 of the build: one triangle AABB per primitive.
//
// Three vertices are derived from the index alone (deterministic, so the timed
// dispatch needs no input buffer to restore) and the AABB is their min/max,
// which is exactly what a real triangle input pass does.
[[nodiscard]] inline auto make_lbvh_prim_kernel() noexcept {
    return Kernel1D{[](BufferVar<BenchPrim> prims, UInt count) noexcept {
        set_block_size(kBuildBlock);
        UInt i = dispatch_id().x;
        $if (i < count) {
            UInt h = mix_bits(i);
            Float3 base = (make_float3(unit_float(mix_bits(h ^ 0x9E3779B9u)),
                                       unit_float(mix_bits(h ^ 0x85EBCA6Bu)),
                                       unit_float(mix_bits(h ^ 0xC2B2AE35u))) *
                            (2.0f * kSceneHalfExtent * 0.9f)) -
                          make_float3(kSceneHalfExtent * 0.9f);
            Float3 d1 = (make_float3(unit_float(mix_bits(h ^ 0x27D4EB2Fu)),
                                     unit_float(mix_bits(h ^ 0x165667B1u)),
                                     unit_float(mix_bits(h ^ 0xD3A2646Cu))) -
                         0.5f) *
                        0.1f;
            Float3 d2 = (make_float3(unit_float(mix_bits(h ^ 0xFD7046C5u)),
                                     unit_float(mix_bits(h ^ 0xB55A4F09u)),
                                     unit_float(mix_bits(h ^ 0x9E3779B1u))) -
                         0.5f) *
                        0.1f;
            Float3 v0 = base;
            Float3 v1 = base + d1;
            Float3 v2 = base + d2;
            Float3 lo = min(min(v0, v1), v2);
            Float3 hi = max(max(v0, v1), v2);
            Var<BenchPrim> prim;
            prim.lo = make_float4(lo.x, lo.y, lo.z, pack_handle(i));
            prim.hi = make_float4(hi.x, hi.y, hi.z, 0.0f);
            prims.write(i, prim);
        };
    }};
}

// Pass 2 of the build: the 30-bit Morton code of every primitive.
[[nodiscard]] inline auto make_lbvh_morton_kernel() noexcept {
    return Kernel1D{[](BufferVar<BenchPrim> prims, BufferUInt2 keys, UInt count,
                       Float3 scene_min, Float3 scene_inv_extent) noexcept {
        set_block_size(kBuildBlock);
        // "expandBits" of Karras / VkLBVH: spread 10 bits to 30.
        auto expand_bits = [](UInt v) noexcept {
            v = (v * 0x00010001u) & 0xFF0000FFu;
            v = (v * 0x00000101u) & 0x0F00F00Fu;
            v = (v * 0x00000011u) & 0xC30C30C3u;
            v = (v * 0x00000005u) & 0x49249249u;
            return v;
        };
        UInt i = dispatch_id().x;
        $if (i < count) {
            auto prim = prims.read(i);
            auto center = (prim_lo(prim) + prim_hi(prim)) * 0.5f;
            auto mapped = clamp((center - scene_min) * scene_inv_extent,
                                make_float3(0.0f), make_float3(1.0f)) *
                          1024.0f;
            UInt x = cast<uint>(min(mapped.x, 1023.0f));
            UInt y = cast<uint>(min(mapped.y, 1023.0f));
            UInt z = cast<uint>(min(mapped.z, 1023.0f));
            UInt code = expand_bits(x) * 4u + expand_bits(y) * 2u + expand_bits(z);
            keys.write(i, make_uint2(code, i));
        };
    }};
}

// Pass 3 of the build (after the sort): the leaves.
//
// Leaf `i` is the primitive at sorted position `i`; the leaves live at
// [count - 1, 2 * count - 2] of the node array.
[[nodiscard]] inline auto make_lbvh_leaf_kernel() noexcept {
    return Kernel1D{[](BufferUInt2 keys, BufferVar<BenchPrim> prims,
                       BufferVar<BenchNode> nodes, UInt count) noexcept {
        set_block_size(kBuildBlock);
        UInt i = dispatch_id().x;
        $if (i < count) {
            UInt slot = keys.read(i).y;
            auto prim = prims.read(slot);
            Var<BenchNode> leaf;
            leaf.packed_lo = pack_node_plane(prim_lo(prim), def(kInvalidHandle));
            leaf.packed_hi = pack_node_plane(prim_hi(prim), slot);
            nodes.write(count - 1u + i, leaf);
        };
    }};
}

// Pass 4 of the build: the structure of the internal nodes.
//
// One lane per node: `determine_range()` and `find_split()` are chains of
// dependent key reads, so a whole warp per node would compute the same chain 32
// times over while a warp of one-lane-per-node keeps 32 independent chains (32
// loads) in flight.  The result is one `uint4` per node, (first, last, child_a,
// child_b), which the AABB pass reads back with a single 16-byte load.
[[nodiscard]] inline auto make_lbvh_plan_kernel() noexcept {
    return Kernel1D{[](BufferUInt2 keys, BufferUInt4 plans, UInt count) noexcept {
        set_block_size(kBuildBlock);
        // -------- the two searches of Karras 2012 --------
        // delta(): the number of leading bits shared by two keys (i.e. clz of
        // their xor).  Equal Morton codes are ordered by their sorted slot, which
        // keeps the keys strictly increasing and the tree well formed; such pairs
        // rank above every pair of distinct codes, hence the +32.
        Callable delta = [](BufferUInt2 keys, UInt n, Int i, UInt code_i, Int j) noexcept {
            Int result = def(-1);
            $if (j >= 0 & j < cast<int>(n)) {
                auto code_j = keys.read(cast<uint>(j)).x;
                $if (code_i == code_j) {
                    result = 32 + cast<int>(clz(cast<uint>(i) ^ cast<uint>(j)));
                }
                $else {
                    result = cast<int>(clz(code_i ^ code_j));
                };
            };
            return result;
        };
        // determineRange(): the leaf range covered by internal node `idx`.
        Callable determine_range = [&delta](BufferUInt2 keys, UInt n, Int idx) noexcept {
            auto code = keys.read(cast<uint>(idx)).x;
            auto delta_left = delta(keys, n, idx, code, idx - 1);
            auto delta_right = delta(keys, n, idx, code, idx + 1);
            auto direction = select(-1, 1, delta_right >= delta_left);
            auto delta_min = min(delta_left, delta_right);
            auto l_max = def(2);
            $while (delta(keys, n, idx, code, idx + l_max * direction) > delta_min &
                    l_max < cast<int>(n)) {
                l_max = l_max * 2;
            };
            auto l = def(0);
            auto stride = l_max / 2;
            $while (stride > 0) {
                $if (delta(keys, n, idx, code, idx + (l + stride) * direction) > delta_min) {
                    l = l + stride;
                };
                stride = stride / 2;
            };
            auto other = idx + l * direction;
            return make_uint2(cast<uint>(min(idx, other)), cast<uint>(max(idx, other)));
        };
        // findSplit(): where a range is split between its two children.
        Callable find_split = [&delta](BufferUInt2 keys, UInt n, UInt first, UInt last) noexcept {
            auto first_code = keys.read(first).x;
            auto common_prefix = delta(keys, n, cast<int>(first), first_code, cast<int>(last));
            auto split = def(cast<int>(first));
            auto stride = cast<int>(last - first);
            $loop {
                stride = (stride + 1) / 2;
                auto next = split + stride;
                $if (next < cast<int>(last)) {
                    $if (delta(keys, n, cast<int>(first), first_code, next) > common_prefix) {
                        split = next;
                    };
                };
                $if (stride <= 1) { $break; };
            };
            return cast<uint>(split);
        };
        // internal nodes: [0, count - 2], leaves: [count - 1, 2 * count - 2]
        UInt i = dispatch_id().x;
        $if (i + 1u < count) {
            auto range = determine_range(keys, count, cast<int>(i));
            auto split = find_split(keys, count, range.x, range.y);
            // a child covering a single element is a leaf
            auto child_a = select(split, count - 1u + split, split == range.x);
            auto child_b = select(split + 1u, count - 1u + split + 1u, split + 1u == range.y);
            plans.write(i, make_uint4(range.x, range.y, child_a, child_b));
        };
    }};
}

// Pass 5 of the build: the block AABBs of the two-level reduction.
//
// Block `b` is the AABB of the leaf slots [b * kReductionBlock, ...), so the
// internal-node pass reads O(range / kReductionBlock) block records instead of
// O(range) leaves for the long ranges.
[[nodiscard]] inline auto make_lbvh_block_kernel() noexcept {
    return Kernel1D{[](BufferVar<BenchNode> nodes, BufferVar<BenchNode> blocks,
                       UInt leaf_begin, UInt leaf_end, UInt block_begin) noexcept {
        set_block_size(kBuildBlock);
        set_warp_size(static_cast<uint8_t>(kReductionBlock));
        UInt lane_count = warp_lane_count();
        UInt lane = warp_lane_id();
        UInt block = block_begin + (block_id().x * kBuildBlock + thread_x()) / lane_count;
        // the span is clamped to the tree's own leaves; a block the internal-node
        // pass reads is strictly inside a node's range, hence entirely inside it
        UInt block_first = max(block * kReductionBlock, leaf_begin);
        UInt block_last = min((block + 1u) * kReductionBlock, leaf_end);
        $if (block_first < block_last) {
            Float3 lo = def(make_float3(1.0e30f));
            Float3 hi = def(make_float3(-1.0e30f));
            // the lanes of the warp all hold the same `block`, so the whole warp
            // is active inside the guard and the lanes past the block contribute
            // the identity to the reduction
            UInt j = def(block_first + lane);
            $while (j < block_last) {
                auto leaf = nodes.read(j);
                lo = min(lo, aabb_lo(leaf));
                hi = max(hi, aabb_hi(leaf));
                j = j + lane_count;
            };
            lo = warp_active_min(lo);
            hi = warp_active_max(hi);
            $if (lane == 0u) {
                // only the AABB planes of a block are ever read back
                Var<BenchNode> bounds;
                bounds.packed_lo = make_float4(lo.x, lo.y, lo.z, 0.0f);
                bounds.packed_hi = make_float4(hi.x, hi.y, hi.z, 0.0f);
                blocks.write(block, bounds);
            };
        };
    }};
}

// Pass 6 of the build: the AABBs of the internal nodes, one warp per node.
//
// Internal node `i`'s AABB is the union of the leaf AABBs of its range
// [plan.x, plan.y].  The range is covered by three disjoint pieces - the leaves
// up to the next block boundary, the whole blocks strictly inside the range, and
// the leaves from the last block boundary on - so a node whose range stays inside
// one block (the overwhelming majority) takes the first loop only and costs
// exactly what a direct reduction costs.  The union is bit-identical to the
// direct one: min/max are exact and every element of the range is read once.
[[nodiscard]] inline auto make_lbvh_build_kernel() noexcept {
    return Kernel2D{[](BufferVar<BenchNode> nodes, BufferVar<BenchNode> blocks,
                       BufferUInt4 plans, UInt count, UInt row_stride) noexcept {
        set_block_size(kBuildBlock, 1u);
        set_warp_size(static_cast<uint8_t>(kReductionBlock));
        UInt lane_count = warp_lane_count();
        UInt lane = warp_lane_id();
        // one warp per internal node; the rows are laid out `row_stride` threads
        // apart so that no grid dimension exceeds kMaxGroupsX
        UInt linear = block_id().y * row_stride +
                      block_id().x * kBuildBlock + thread_id().x;
        UInt i = linear / lane_count;
        $if (i + 1u < count) {
            auto plan = plans.read(i);
            UInt leaf_base = count - 1u;
            UInt first = leaf_base + plan.x;
            UInt last = leaf_base + plan.y;
            Float3 lo = def(make_float3(1.0e30f));
            Float3 hi = def(make_float3(-1.0e30f));
            UInt first_block = first / kReductionBlock;
            UInt last_block = last / kReductionBlock;
            // `left_end` is exclusive; the last piece is empty when the range
            // stays inside one block
            UInt left_end = min(last + 1u, (first_block + 1u) * kReductionBlock);
            UInt right_begin = max(left_end, last_block * kReductionBlock);
            UInt j = def(first + lane);
            $while (j < left_end) {
                auto leaf = nodes.read(j);
                lo = min(lo, aabb_lo(leaf));
                hi = max(hi, aabb_hi(leaf));
                j = j + lane_count;
            };
            j = right_begin + lane;
            $while (j <= last) {
                auto leaf = nodes.read(j);
                lo = min(lo, aabb_lo(leaf));
                hi = max(hi, aabb_hi(leaf));
                j = j + lane_count;
            };
            j = first_block + 1u + lane;
            $while (j < last_block) {
                auto block = blocks.read(j);
                lo = min(lo, aabb_lo(block));
                hi = max(hi, aabb_hi(block));
                j = j + lane_count;
            };
            lo = warp_active_min(lo);
            hi = warp_active_max(hi);
            $if (lane == 0u) {
                Var<BenchNode> node;
                node.packed_lo = pack_node_plane(lo, plan.z);
                node.packed_hi = pack_node_plane(hi, plan.w);
                nodes.write(i, node);
            };
        };
    }};
}

// Verification only: the exact AABB of the triangle set, reduced on the device so
// the host does not have to mirror the procedural triangle generator.
//
// One work-group, one warp reduction, one shared-memory reduction - deliberately
// no atomics: floating-point atomic min/max needs
// SPV_EXT_shader_atomic_float_min_max, which a portable benchmark cannot assume
// (the Vulkan backend reports VK_ERROR_DEVICE_LOST without it).  The result is
// also bit-exactly deterministic, which a chain of float atomics is not required
// to be.
[[nodiscard]] inline auto make_lbvh_scene_bounds_kernel() noexcept {
    return Kernel1D{[](BufferVar<BenchPrim> prims, BufferFloat bounds, UInt count) noexcept {
        set_block_size(kBuildBlock);
        set_warp_size(static_cast<uint8_t>(kReductionBlock));
        UInt tid = thread_x();
        UInt lane = warp_lane_id();
        UInt warp = tid / kReductionBlock;
        // one `(lo.xyz, hi.xyz)` triple per warp
        Shared<float> per_warp{2u * kBuildWarps * 3u};
        Float3 lo = def(make_float3(1.0e30f));
        Float3 hi = def(make_float3(-1.0e30f));
        UInt i = def(tid);
        $while (i < count) {
            auto prim = prims.read(i);
            lo = min(lo, prim_lo(prim));
            hi = max(hi, prim_hi(prim));
            i = i + kBuildBlock;
        };
        lo = warp_active_min(lo);
        hi = warp_active_max(hi);
        $if (lane == 0u) {
            per_warp[warp * 3u + 0u] = lo.x;
            per_warp[warp * 3u + 1u] = lo.y;
            per_warp[warp * 3u + 2u] = lo.z;
            per_warp[kBuildWarps * 3u + warp * 3u + 0u] = hi.x;
            per_warp[kBuildWarps * 3u + warp * 3u + 1u] = hi.y;
            per_warp[kBuildWarps * 3u + warp * 3u + 2u] = hi.z;
        };
        sync_block();
        $if (tid == 0u) {
            Float3 total_lo = def(make_float3(1.0e30f));
            Float3 total_hi = def(make_float3(-1.0e30f));
            $for (w, kBuildWarps) {
                total_lo = min(total_lo, make_float3(per_warp[w * 3u + 0u],
                                                     per_warp[w * 3u + 1u],
                                                     per_warp[w * 3u + 2u]));
                total_hi = max(total_hi, make_float3(per_warp[kBuildWarps * 3u + w * 3u + 0u],
                                                     per_warp[kBuildWarps * 3u + w * 3u + 1u],
                                                     per_warp[kBuildWarps * 3u + w * 3u + 2u]));
            };
            bounds.write(0u, total_lo.x);
            bounds.write(1u, total_lo.y);
            bounds.write(2u, total_lo.z);
            bounds.write(3u, total_hi.x);
            bounds.write(4u, total_hi.y);
            bounds.write(5u, total_hi.z);
        };
    }};
}

void bench_compute_lbvh_build(compute::Device &device, compute::Stream &stream,
                              const BenchConfig &cfg,
                              std::vector<BenchResult> &results) noexcept {
    constexpr uint kCount = 1048576u;// 2^20 triangles
    constexpr uint kNodeCount = kCount * 2u - 1u;
    constexpr uint kBlockCapacity = (kNodeCount + kReductionBlock - 1u) / kReductionBlock;

    auto blocks = sort_blocks(kCount);
    auto blocks_per_group = sort_blocks_per_group(blocks);
    auto groups = sort_groups(blocks, blocks_per_group);

    auto prims = device.create_buffer<BenchPrim>(kCount);
    auto keys_a = device.create_buffer<uint2>(kCount);
    auto keys_b = device.create_buffer<uint2>(kCount);
    auto nodes = device.create_buffer<BenchNode>(kNodeCount);
    auto node_blocks = device.create_buffer<BenchNode>(kBlockCapacity + 1u);
    auto plans = device.create_buffer<uint4>(kCount);
    auto scene_bounds = device.create_buffer<float>(6u);
    auto scratch = create_radix_sort_buffers(device, blocks, groups);

    double compile_ms = 0.0;
    auto prim_kernel = compile_uncached(device, make_lbvh_prim_kernel(),
                                        "bench_compute_lbvh_prim", cfg, compile_ms);
    auto morton_kernel = compile_uncached(device, make_lbvh_morton_kernel(),
                                          "bench_compute_lbvh_morton", cfg, compile_ms);
    auto hist = compile_uncached(device, make_radix_hist_kernel(),
                                 "bench_compute_lbvh_hist", cfg, compile_ms);
    auto scan_blocks = compile_uncached(device, make_radix_scan_blocks_kernel(),
                                        "bench_compute_lbvh_scan_blocks", cfg, compile_ms);
    auto scan_groups = compile_uncached(device, make_radix_scan_groups_kernel(),
                                        "bench_compute_lbvh_scan_groups", cfg, compile_ms);
    auto scatter = compile_uncached(device, make_radix_scatter_kernel(),
                                    "bench_compute_lbvh_scatter", cfg, compile_ms);
    auto leaf_kernel = compile_uncached(device, make_lbvh_leaf_kernel(),
                                        "bench_compute_lbvh_leaf", cfg, compile_ms);
    auto plan_kernel = compile_uncached(device, make_lbvh_plan_kernel(),
                                        "bench_compute_lbvh_plan", cfg, compile_ms);
    auto block_kernel = compile_uncached(device, make_lbvh_block_kernel(),
                                         "bench_compute_lbvh_block", cfg, compile_ms);
    auto build_kernel = compile_uncached(device, make_lbvh_build_kernel(),
                                         "bench_compute_lbvh_build", cfg, compile_ms);
    // compiled here (not in the verification block) so that the reported compile
    // time covers every kernel of the benchmark
    auto scene_bounds_kernel = compile_uncached(device, make_lbvh_scene_bounds_kernel(),
                                                "bench_compute_lbvh_scene_bounds", cfg,
                                                compile_ms);

    // The scene the Morton codes are normalized against, and the grids of the two
    // warp-per-node passes (fixed 32-lane warps, see `set_warp_size`).
    float3 scene_min = make_float3(-kSceneHalfExtent);
    float3 scene_inv_extent = make_float3(1.0f / (2.0f * kSceneHalfExtent));

    constexpr uint leaf_begin = kCount - 1u;
    constexpr uint leaf_end = leaf_begin + kCount;
    constexpr uint block_begin = leaf_begin / kReductionBlock;
    constexpr uint block_end = (leaf_end - 1u) / kReductionBlock;
    constexpr uint block_count = block_end - block_begin + 1u;
    constexpr uint block_threads = block_count * kReductionBlock;
    constexpr uint block_groups = ceil_div_u(block_threads, kBuildBlock);

    constexpr uint build_threads = kCount * kReductionBlock;
    constexpr uint build_groups = ceil_div_u(build_threads, kBuildBlock);
    constexpr uint build_groups_x = std::min(build_groups, kMaxGroupsX);
    constexpr uint build_rows = ceil_div_u(build_groups, build_groups_x);
    constexpr uint build_row_stride = build_groups_x * kBuildBlock;

    constexpr uint prim_dispatch = ceil_div_u(kCount, kBuildBlock) * kBuildBlock;
    constexpr uint plan_dispatch = ceil_div_u(kCount, kBuildBlock) * kBuildBlock;

    auto dispatch = [&] {
        CommandList cmds = CommandList::create(8u + kSortPasses * 4u);
        cmds << prim_kernel(prims, kCount).dispatch(prim_dispatch);
        cmds << morton_kernel(prims, keys_a, kCount, scene_min, scene_inv_extent)
                    .dispatch(prim_dispatch);
        encode_radix_sort(cmds, hist, scan_blocks, scan_groups, scatter,
                          keys_a, keys_b, scratch, kCount);
        // the four passes leave the sorted keys in `keys_a`
        cmds << leaf_kernel(keys_a, prims, nodes, kCount).dispatch(prim_dispatch);
        cmds << plan_kernel(keys_a, plans, kCount).dispatch(plan_dispatch);
        cmds << block_kernel(nodes, node_blocks, leaf_begin, leaf_end, block_begin)
                    .dispatch(block_groups * kBuildBlock);
        cmds << build_kernel(nodes, node_blocks, plans, kCount, build_row_stride)
                    .dispatch(build_groups_x * kBuildBlock, build_rows);
        stream << cmds.commit();
    };

    auto result = run_timed_bench("compute", "compute_lbvh_build", stream, cfg,
                                  compile_ms, static_cast<double>(kCount), "prim/s",
                                  dispatch);

    if (!cfg.verify) {
        result.verified = false;
        result.check_note = "verification disabled";
        results.push_back(std::move(result));
        return;
    }
    if (cfg.warmup_iters == 0u && cfg.timed_iters == 0u) {
        dispatch();
        stream << compute::synchronize();
    }

    // ---- structural self-check -------------------------------------------
    // (1) the root AABB must be finite and enclose the exact scene AABB, which a
    //     one-off device reduction measures; (2) every child handle must be a
    //     valid node index and every leaf must carry the leaf marker; (3) the tree
    //     must be a single connected tree of exactly 2n-1 nodes whose leaves are a
    //     permutation of the primitives; (4) every internal node's AABB must be
    //     the exact union of its children's, which validates the plan ranges, the
    //     block AABBs and the two-level reduction.
    luisa::vector<float> host_bounds{1.0e30f, 1.0e30f, 1.0e30f,
                                     -1.0e30f, -1.0e30f, -1.0e30f};
    stream << scene_bounds.copy_from(luisa::span{host_bounds})
           << scene_bounds_kernel(prims, scene_bounds, kCount).dispatch(kBuildBlock)
           << scene_bounds.copy_to(luisa::span{host_bounds})
           << compute::synchronize();

    luisa::vector<BenchNode> host_nodes(kNodeCount);
    stream << nodes.copy_to(luisa::span{host_nodes}) << compute::synchronize();

    auto root_lo = host_aabb_lo(host_nodes[0u]);
    auto root_hi = host_aabb_hi(host_nodes[0u]);
    bool root_finite = std::isfinite(root_lo.x) && std::isfinite(root_lo.y) &&
                       std::isfinite(root_lo.z) && std::isfinite(root_hi.x) &&
                       std::isfinite(root_hi.y) && std::isfinite(root_hi.z);
    bool root_encloses = root_lo.x <= host_bounds[0u] && root_lo.y <= host_bounds[1u] &&
                         root_lo.z <= host_bounds[2u] && root_hi.x >= host_bounds[3u] &&
                         root_hi.y >= host_bounds[4u] && root_hi.z >= host_bounds[5u];

    // (2) handles
    size_t bad_handles = 0u;
    for (uint i = 0u; i < kNodeCount; ++i) {
        auto left = host_child_left(host_nodes[i]);
        if (i + 1u < kCount) {// internal node
            auto right = host_child_right(host_nodes[i]);
            if (left >= kNodeCount || right >= kNodeCount) { bad_handles++; }
        } else if (left != kInvalidHandle) {// leaf must carry the marker
            bad_handles++;
        }
    }

    // (3) reachability + (4) the AABB union, in one depth-first walk.
    std::vector<bool> visited(kNodeCount, false);
    std::vector<uint8_t> prim_seen(kCount, 0u);
    std::vector<uint> stack;
    stack.reserve(64u);
    stack.push_back(0u);
    size_t visited_nodes = 0u;
    size_t leaves = 0u;
    size_t bad_unions = 0u;
    size_t bad_prims = 0u;
    bool structure_ok = true;
    while (!stack.empty() && structure_ok) {
        auto index = stack.back();
        stack.pop_back();
        if (index >= kNodeCount || visited[index]) {
            structure_ok = false;
            break;
        }
        visited[index] = true;
        visited_nodes++;
        auto left = host_child_left(host_nodes[index]);
        if (left == kInvalidHandle) {
            leaves++;
            auto prim = host_child_right(host_nodes[index]);
            if (prim >= kCount || prim_seen[prim] != 0u) {
                bad_prims++;
            } else {
                prim_seen[prim] = 1u;
            }
            continue;
        }
        auto right = host_child_right(host_nodes[index]);
        if (right >= kNodeCount) {
            structure_ok = false;
            break;
        }
        // the AABB of the node must be the exact union of its two children's
        auto node_lo = host_aabb_lo(host_nodes[index]);
        auto node_hi = host_aabb_hi(host_nodes[index]);
        auto lo_a = host_aabb_lo(host_nodes[left]);
        auto hi_a = host_aabb_hi(host_nodes[left]);
        auto lo_b = host_aabb_lo(host_nodes[right]);
        auto hi_b = host_aabb_hi(host_nodes[right]);
        auto expect_lo = make_float3(std::min(lo_a.x, lo_b.x),
                                     std::min(lo_a.y, lo_b.y),
                                     std::min(lo_a.z, lo_b.z));
        auto expect_hi = make_float3(std::max(hi_a.x, hi_b.x),
                                     std::max(hi_a.y, hi_b.y),
                                     std::max(hi_a.z, hi_b.z));
        if (node_lo.x != expect_lo.x || node_lo.y != expect_lo.y ||
            node_lo.z != expect_lo.z || node_hi.x != expect_hi.x ||
            node_hi.y != expect_hi.y || node_hi.z != expect_hi.z) {
            bad_unions++;
        }
        stack.push_back(left);
        stack.push_back(right);
    }

    bool ok = root_finite && root_encloses && bad_handles == 0u && structure_ok &&
              visited_nodes == kNodeCount && leaves == kCount && bad_prims == 0u &&
              bad_unions == 0u;
    result.verified = ok;
    result.check_note = luisa::format(
        "root_finite={}, encloses_scene={}, nodes={}/{}, leaves={}/{}, "
        "bad_handles={}, bad_unions={}, bad_prims={}",
        root_finite, root_encloses, visited_nodes, kNodeCount, leaves, kCount,
        bad_handles, bad_unions, bad_prims);
    if (!ok) { LUISA_WARNING("{}: {}", result.name, result.check_note); }
    results.push_back(std::move(result));
}

// ---------------------------------------------------------------------------
// Benchmark 3: the n-gram hash index.
// ---------------------------------------------------------------------------

// FNV-1a-64 over the little-endian bytes of `n` 32-bit tokens - byte-identical to
// the DSL loop below (and to the host-side index builder of
// src/compute/tokenize/ngram_kernels.cpp).
[[nodiscard]] inline uint64_t host_fnvon64(const uint *tokens, uint n) noexcept {
    uint64_t key = 14695981039346656037ull;
    for (uint j = 0u; j < n; ++j) {
        auto token = static_cast<uint64_t>(tokens[j]);
        for (uint b = 0u; b < 4u; ++b) {
            key = (key ^ ((token >> (b * 8u)) & 0xFFull)) * 1099511628211ull;
        }
    }
    return key == 0ull ? 1ull : key;
}

// A host-side open-addressing table with the device's exact probe sequence.
// `tags` pack (n, position) into one word: position < 2^24 by construction.
struct NgramHostIndex {
    luisa::vector<uint64_t> keys;
    luisa::vector<uint> tags;
    uint cap_log2{0u};
    uint64_t collisions{0u};
    uint64_t inserted{0u};

    [[nodiscard]] uint32_t mask() const noexcept { return (1u << cap_log2) - 1u; }

    [[nodiscard]] uint32_t slot_of(uint64_t key) const noexcept {
        return static_cast<uint32_t>((key * 0x9E3779B97F4A7C15ull) >> (64u - cap_log2));
    }
};

[[nodiscard]] inline bool same_ngram(const luisa::vector<uint> &corpus,
                                     uint pos_a, uint pos_b, uint n) noexcept {
    for (uint j = 0u; j < n; ++j) {
        if (corpus[pos_a + j] != corpus[pos_b + j]) { return false; }
    }
    return true;
}

// Inserts every n-gram of the corpus; the first occurrence of an n-gram wins, so
// the stored position is the earliest one (which is what the device scan finds).
void host_index_insert(NgramHostIndex &index, const luisa::vector<uint> &corpus,
                       uint n, uint pos) noexcept {
    auto key = host_fnvon64(corpus.data() + pos, n);
    auto slot = index.slot_of(key);
    for (;;) {
        if (index.keys[slot] == 0ull) {
            index.keys[slot] = key;
            index.tags[slot] = (n << 24u) | pos;
            index.inserted++;
            return;
        }
        if (index.keys[slot] == key) {
            auto tag = index.tags[slot];
            auto n_old = tag >> 24u;
            auto pos_old = tag & 0x00FFFFFFu;
            if (n_old == n && same_ngram(corpus, pos_old, pos, n)) {
                return;// the same n-gram: the earliest position is already there
            }
            // a genuine 64-bit key collision of two different n-grams: the
            // fail-closed probe of the device kernel cannot represent it
            index.collisions++;
            return;
        }
        slot = (slot + 1u) & index.mask();
    }
}

// The device's probe, on the host: the earliest position of `n` tokens of the
// query, if the corpus contains that n-gram.
[[nodiscard]] bool host_index_lookup(const NgramHostIndex &index,
                                     const luisa::vector<uint> &corpus,
                                     const uint *query, uint n, uint &pos_out) noexcept {
    auto key = host_fnvon64(query, n);
    if (key == 0ull) { key = 1ull; }
    auto slot = index.slot_of(key);
    for (;;) {
        auto stored = index.keys[slot];
        if (stored == 0ull) { return false; }
        if (stored == key) {
            auto tag = index.tags[slot];
            auto n_old = tag >> 24u;
            auto pos_old = tag & 0x00FFFFFFu;
            if (n_old == n) {
                bool equal = true;
                for (uint j = 0u; j < n; ++j) {
                    if (corpus[pos_old + j] != query[j]) {
                        equal = false;
                        break;
                    }
                }
                if (equal) {
                    pos_out = pos_old;
                    return true;
                }
            }
        }
        slot = (slot + 1u) & index.mask();
    }
}

// The device kernel of src/compute/tokenize/ngram_kernels.cpp: the FNV-1a-64
// open-addressing probe, longest n first, content-verified against the corpus.
[[nodiscard]] inline auto make_ngram_hash_kernel(uint32_t cap_log2) noexcept {
    return Kernel1D{[cap_log2](BufferUInt corpus, UInt corpus_len,
                               BufferUInt queries, BufferUInt query_len, UInt query_stride,
                               BufferVar<uint64_t> keys, BufferUInt tags,
                               BufferUInt out_n, BufferUInt out_pos,
                               BufferUInt out_draft, BufferUInt out_draft_len) noexcept {
        set_block_size(kNgramScanThreads);
        UInt qid = dispatch_id().x;
        UInt qlen = query_len.read(qid);
        // every draft slot starts invalid; every query gets a count
        $for (i, kNgramK) { out_draft.write(qid * kNgramK + i, kInvalidHandle); };
        out_draft_len.write(qid, 0u);
        out_n.write(qid, 0u);
        out_pos.write(qid, 0u);
        UInt best_n = def(0u);
        UInt best_pos = def(0u);
        UInt done = def(0u);
        // longest n first: the first length present in the index wins
        $for (nn, kNgramMax - kNgramMin + 1u) {
            UInt n = kNgramMax - nn;
            $if (done == 0u & n <= qlen) {
                // FNV-1a 64 over the query suffix, byte-identical to the host
                // index builder
                Var<uint64_t> key = 14695981039346656037ull;
                $for (j, n) {
                    Var<uint64_t> token = cast<uint64_t>(
                        queries.read(qid * query_stride + qlen - n + j));
                    $for (b, 4u) {
                        key = (key ^ ((token >> (b * 8u)) & 0xFFull)) * 1099511628211ull;
                    };
                };
                $if (key == 0ull) { key = 1ull; };// 0 is the empty-slot marker
                // linear-probe the index; content-verify key collisions
                UInt found = def(0u);
                UInt found_pos = def(0u);
                Var<uint64_t> slot = (key * 0x9E3779B97F4A7C15ull) >> (64u - cap_log2);
                // the host guarantees a load factor <= 0.5, so an empty slot is
                // always found; the bound only guards against a corrupt index
                UInt probes = def(0u);
                $while (found == 0u & probes < (1u << cap_log2)) {
                    UInt probe = cast<uint>(slot);
                    Var<uint64_t> stored = keys.read(probe);
                    $if (stored == key) {
                        auto tag = tags.read(probe);
                        UInt probe_n = tag >> 24u;
                        UInt probe_pos = tag & 0x00FFFFFFu;
                        Bool same = probe_n == n;
                        $for (j, kNgramMax) {
                            $if (j < n & same) {
                                same = corpus.read(probe_pos + j) ==
                                       queries.read(qid * query_stride + qlen - n + j);
                            };
                        };
                        $if (same) {
                            found = 1u;
                            found_pos = probe_pos;
                        };
                    };
                    $if (found == 0u) {
                        $if (stored == 0ull) { found = 2u; };// empty slot: absent
                    };
                    $if (found == 0u) {
                        slot = (slot + 1ull) & ((1ull << cap_log2) - 1ull);
                    };
                    probes = probes + 1u;
                };
                $if (found == 1u) {
                    done = 1u;
                    best_n = n;
                    best_pos = found_pos;
                };
            };
        };
        out_n.write(qid, best_n);
        out_pos.write(qid, best_pos);
        $if (done != 0u) {
            UInt start = best_pos + best_n;
            UInt avail = corpus_len - start;// the index only stores positions with pos + n <= len
            UInt cnt = min(kNgramK, avail);
            $for (i, kNgramK) {
                $if (i < cnt) {
                    out_draft.write(qid * kNgramK + i, corpus.read(start + i));
                };
            };
            out_draft_len.write(qid, cnt);
        };
    }};
}

// The parallel shared-memory `atomic_min` scan of the same reference: one
// work-group per query, the corpus walked with a strided loop, a first-token
// filter so nearly all positions cost a single global read, and the earliest
// match of the block reduced through one shared counter.
[[nodiscard]] inline auto make_ngram_scan_kernel() noexcept {
    return Kernel1D{[](BufferUInt corpus, UInt corpus_len,
                       BufferUInt queries, BufferUInt query_len, UInt query_stride,
                       BufferUInt out_n, BufferUInt out_pos,
                       BufferUInt out_draft, BufferUInt out_draft_len) noexcept {
        set_block_size(kNgramScanThreads);
        UInt qid = block_id().x;
        UInt lane = thread_x();
        UInt qlen = query_len.read(qid);
        Shared<uint> suffix{kNgramMax};
        Shared<uint> earliest{1u};
        UInt best_n = def(0u);
        UInt best_pos = def(0u);
        UInt done = def(0u);
        $if (lane == 0u) {
            $for (i, kNgramK) { out_draft.write(qid * kNgramK + i, kInvalidHandle); };
            out_draft_len.write(qid, 0u);
            out_n.write(qid, 0u);
            out_pos.write(qid, 0u);
        };
        sync_block();
        // longest n first: the first length with a match wins
        $for (nn, kNgramMax - kNgramMin + 1u) {
            UInt n = kNgramMax - nn;
            $if (done == 0u & n <= qlen & n <= corpus_len) {
                // stage the query suffix q[qlen - n, qlen) into shared memory
                $for (idx, lane, n, kNgramScanThreads) {
                    suffix.write(idx, queries.read(qid * query_stride + qlen - n + idx));
                };
                $if (lane == 0u) { earliest.write(0u, kInvalidHandle); };
                sync_block();
                // strided scan: thread `lane` checks positions lane, lane + bs, ...
                Bool mine = def(false);
                UInt my_pos = def(0u);
                $for (p, lane, corpus_len - n + 1u, kNgramScanThreads) {
                    $if (mine == false) {
                        // first-token filter: nearly all corpus positions fail
                        // here, costing a single global read instead of a full
                        // n-token compare
                        $if (corpus.read(p) == suffix.read(0u)) {
                            Bool match = def(true);
                            $for (j, kNgramMax) {
                                $if (j < n & match) {
                                    match = corpus.read(p + j) == suffix.read(j);
                                };
                            };
                            $if (match) {
                                mine = true;
                                my_pos = p;
                            };
                        };
                    };
                };
                // earliest match across the block wins
                $if (mine) { earliest.atomic(0u).fetch_min(my_pos); };
                sync_block();
                UInt pos = earliest.read(0u);
                $if (pos != kInvalidHandle) {
                    done = 1u;
                    best_n = n;
                    best_pos = pos;
                };
            };
        };
        $if (lane == 0u) {
            out_n.write(qid, best_n);
            out_pos.write(qid, best_pos);
            $if (done != 0u) {
                UInt start = best_pos + best_n;
                UInt avail = corpus_len - start;
                UInt cnt = min(kNgramK, avail);
                $for (i, kNgramK) {
                    $if (i < cnt) {
                        out_draft.write(qid * kNgramK + i, corpus.read(start + i));
                    };
                };
                out_draft_len.write(qid, cnt);
            };
        };
    }};
}

// One expected answer of a query, computed on the host from the same index.
struct NgramExpectation {
    uint n{0u};
    uint pos{0u};
    uint draft_len{0u};
    std::array<uint, kNgramK> draft{};
};

void bench_compute_ngram_hash(compute::Device &device, compute::Stream &stream,
                              const BenchConfig &cfg,
                              std::vector<BenchResult> &results) noexcept {
    // ---- corpus and queries (deterministic) -------------------------------
    luisa::vector<uint> corpus(kCorpusTokens);
    uint64_t state = 0xC0FFEE1234ull;
    for (uint i = 0u; i < kCorpusTokens; ++i) {
        corpus[i] = prng_u32(state) % kNgramVocab;
    }
    luisa::vector<uint> queries(kQueryCount * kQueryStride, 0u);
    luisa::vector<uint> query_len(kQueryCount, kQueryTokens);
    for (uint q = 0u; q < kQueryCount; ++q) {
        auto qstate = splitmix64(0x9E3779B97F4A7C15ull * (static_cast<uint64_t>(q) + 1ull));
        auto start = prng_u32(qstate) % (kCorpusTokens - kQueryStride);
        for (uint j = 0u; j < kQueryTokens; ++j) {
            queries[q * kQueryStride + j] = corpus[start + j];
        }
        // 50% of the queries end in a present 4-gram, 25% end in a present
        // 3-gram (the 4-gram is broken), 25% match nothing at all
        if (q % 4u == 2u) {
            queries[q * kQueryStride + kQueryTokens - 1u] = kNgramSentinel;
        } else if (q % 4u == 3u) {
            queries[q * kQueryStride + kQueryTokens - kNgramMax] = kNgramSentinel;
        }
    }

    // ---- host index, load factor <= 0.5 ----------------------------------
    // Capacity: the next power of two that holds twice the largest possible number
    // of *distinct* n-grams (n-grams of length n cannot outnumber the corpus
    // positions, nor the vocabulary^n possible tuples).
    uint64_t bound = 0u;
    for (uint n = kNgramMin; n <= kNgramMax; ++n) {
        uint64_t tuples = 1u;
        for (uint j = 0u; j < n; ++j) { tuples *= kNgramVocab; }
        bound += std::min<uint64_t>(tuples, kCorpusTokens);
    }
    uint cap_log2 = 1u;
    while ((uint64_t{1u} << cap_log2) < 2u * bound) { cap_log2++; }
    LUISA_ASSERT(cap_log2 < 31u, "n-gram index capacity overflow.");

    NgramHostIndex index;
    index.cap_log2 = cap_log2;
    index.keys.resize(size_t{1u} << cap_log2, 0ull);
    index.tags.resize(size_t{1u} << cap_log2, 0u);
    for (uint n = kNgramMin; n <= kNgramMax; ++n) {
        for (uint p = 0u; p + n <= kCorpusTokens; ++p) {
            host_index_insert(index, corpus, n, p);
        }
    }
    LUISA_ASSERT(index.collisions == 0u,
                 "the n-gram index hit {} genuine 64-bit key collision(s); the "
                 "device probe cannot represent them.",
                 index.collisions);
    double load_factor = static_cast<double>(index.inserted) /
                         static_cast<double>(size_t{1u} << cap_log2);
    LUISA_ASSERT(load_factor <= 0.5,
                 "n-gram index load factor {} exceeds 0.5.", load_factor);

    // ---- the expected answer of every query ------------------------------
    luisa::vector<NgramExpectation> expected(kQueryCount);
    uint matched = 0u;
    for (uint q = 0u; q < kQueryCount; ++q) {
        auto *tokens = queries.data() + q * kQueryStride;
        auto qlen = query_len[q];
        auto &exp = expected[q];
        for (uint n = kNgramMax; n >= kNgramMin; --n) {
            if (n > qlen) { continue; }
            uint pos = 0u;
            if (host_index_lookup(index, corpus, tokens + qlen - n, n, pos)) {
                exp.n = n;
                exp.pos = pos;
                break;
            }
        }
        if (exp.n != 0u) {
            matched++;
            auto begin = exp.pos + exp.n;
            exp.draft_len = std::min<uint>(kNgramK, kCorpusTokens - begin);
            for (uint i = 0u; i < exp.draft_len; ++i) { exp.draft[i] = corpus[begin + i]; }
        }
    }

    // ---- device buffers ---------------------------------------------------
    auto corpus_buf = device.create_buffer<uint>(kCorpusTokens);
    auto query_buf = device.create_buffer<uint>(kQueryCount * kQueryStride);
    auto query_len_buf = device.create_buffer<uint>(kQueryCount);
    auto ngram_keys = device.create_buffer<uint64_t>(size_t{1u} << cap_log2);
    auto ngram_tags = device.create_buffer<uint>(size_t{1u} << cap_log2);
    auto hash_n = device.create_buffer<uint>(kQueryCount);
    auto hash_pos = device.create_buffer<uint>(kQueryCount);
    auto hash_draft = device.create_buffer<uint>(kQueryCount * kNgramK);
    auto hash_draft_len = device.create_buffer<uint>(kQueryCount);
    auto scan_n = device.create_buffer<uint>(kQueryCount);
    auto scan_pos = device.create_buffer<uint>(kQueryCount);
    auto scan_draft = device.create_buffer<uint>(kQueryCount * kNgramK);
    auto scan_draft_len = device.create_buffer<uint>(kQueryCount);

    stream << corpus_buf.copy_from(luisa::span{corpus})
           << query_buf.copy_from(luisa::span{queries})
           << query_len_buf.copy_from(luisa::span{query_len})
           << ngram_keys.copy_from(luisa::span{index.keys})
           << ngram_tags.copy_from(luisa::span{index.tags})
           << compute::synchronize();

    double compile_ms = 0.0;
    auto hash_kernel = compile_uncached(device, make_ngram_hash_kernel(cap_log2),
                                        "bench_compute_ngram_hash_probe", cfg, compile_ms);
    auto scan_kernel = compile_uncached(device, make_ngram_scan_kernel(),
                                        "bench_compute_ngram_scan", cfg, compile_ms);

    auto dispatch = [&] {
        CommandList cmds = CommandList::create(2u);
        cmds << hash_kernel(corpus_buf, kCorpusTokens, query_buf, query_len_buf,
                            kQueryStride, ngram_keys, ngram_tags,
                            hash_n, hash_pos, hash_draft, hash_draft_len)
                    .dispatch(kQueryCount);
        cmds << scan_kernel(corpus_buf, kCorpusTokens, query_buf, query_len_buf,
                            kQueryStride, scan_n, scan_pos, scan_draft, scan_draft_len)
                    .dispatch(kQueryCount * kNgramScanThreads);
        stream << cmds.commit();
    };

    auto result = run_timed_bench("compute", "compute_ngram_hash", stream, cfg,
                                  compile_ms, static_cast<double>(kQueryCount), "query/s",
                                  dispatch);

    if (!cfg.verify) {
        result.verified = false;
        result.check_note = "verification disabled";
        results.push_back(std::move(result));
        return;
    }
    if (cfg.warmup_iters == 0u && cfg.timed_iters == 0u) {
        dispatch();
        stream << compute::synchronize();
    }

    luisa::vector<uint> host_hash_n(kQueryCount), host_hash_pos(kQueryCount);
    luisa::vector<uint> host_hash_draft(kQueryCount * kNgramK), host_hash_len(kQueryCount);
    luisa::vector<uint> host_scan_n(kQueryCount), host_scan_pos(kQueryCount);
    luisa::vector<uint> host_scan_draft(kQueryCount * kNgramK), host_scan_len(kQueryCount);
    stream << hash_n.copy_to(luisa::span{host_hash_n})
           << hash_pos.copy_to(luisa::span{host_hash_pos})
           << hash_draft.copy_to(luisa::span{host_hash_draft})
           << hash_draft_len.copy_to(luisa::span{host_hash_len})
           << scan_n.copy_to(luisa::span{host_scan_n})
           << scan_pos.copy_to(luisa::span{host_scan_pos})
           << scan_draft.copy_to(luisa::span{host_scan_draft})
           << scan_draft_len.copy_to(luisa::span{host_scan_len})
           << compute::synchronize();

    auto compare = [&](const luisa::vector<uint> &n, const luisa::vector<uint> &pos,
                       const luisa::vector<uint> &draft, const luisa::vector<uint> &len) noexcept {
        size_t bad = 0u;
        size_t matched_count = 0u;
        for (uint q = 0u; q < kQueryCount; ++q) {
            const auto &exp = expected[q];
            bool ok = n[q] == exp.n && pos[q] == exp.pos && len[q] == exp.draft_len;
            if (exp.n != 0u) { matched_count++; }
            for (uint i = 0u; ok && i < exp.draft_len; ++i) {
                if (draft[q * kNgramK + i] != exp.draft[i]) { ok = false; }
            }
            if (!ok) { bad++; }
        }
        return std::pair{bad, matched_count};
    };
    auto [hash_bad, hash_matched] = compare(host_hash_n, host_hash_pos, host_hash_draft,
                                            host_hash_len);
    auto [scan_bad, scan_matched] = compare(host_scan_n, host_scan_pos, host_scan_draft,
                                            host_scan_len);
    bool ok = hash_bad == 0u && scan_bad == 0u &&
              hash_matched == matched && scan_matched == matched;
    result.verified = ok;
    result.check_note = luisa::format(
        "matched {}/{} (expected {}), hash_mismatch={}, scan_mismatch={}, "
        "cap_log2={}, load={:.3f}",
        hash_matched, scan_matched, matched, hash_bad, scan_bad, cap_log2, load_factor);
    if (!ok) { LUISA_WARNING("{}: {}", result.name, result.check_note); }
    results.push_back(std::move(result));
}

// ---------------------------------------------------------------------------
// Benchmark 4: the image pipeline.
// ---------------------------------------------------------------------------

// Stage 1: a procedural "Lenna-like" pattern (checkerboard, concentric circles,
// diagonal gradient, two "eyes" and a "mouth", tinted per channel).
[[nodiscard]] inline auto make_image_pattern_kernel() noexcept {
    return Kernel2D{[](ImageFloat image) noexcept {
        set_block_size(kImageBlock, kImageBlock, 1u);
        Float2 uv = make_float2(dispatch_id().xy()) / make_float2(dispatch_size().xy());
        Float checker = sin(uv.x * 20.0f) * sin(uv.y * 20.0f);
        Float2 center = uv - make_float2(0.5f);
        Float radius = length(center);
        Float circles = sin(radius * 50.0f);
        Float diagonal = sin((uv.x + uv.y) * 10.0f);
        Float pattern = checker * 0.3f + circles * 0.4f + diagonal * 0.3f;
        Float eye1 = exp(-length(uv - make_float2(0.35f, 0.4f)) * 50.0f);
        Float eye2 = exp(-length(uv - make_float2(0.65f, 0.4f)) * 50.0f);
        Float mouth = exp(-pow(uv.y - 0.6f - 0.1f * pow(uv.x - 0.5f, 2.0f), 2.0f) * 200.0f);
        Float value = pattern * 0.5f + 0.5f + eye1 * 0.3f + eye2 * 0.3f + mouth * 0.2f;
        Float3 tint = make_float3(0.5f + 0.5f * sin(uv.x * 6.2831853f),
                                  0.5f + 0.5f * sin(uv.y * 6.2831853f + 2.0943951f),
                                  0.5f + 0.5f * sin((uv.x + uv.y) * 6.2831853f + 4.1887902f));
        image.write(dispatch_id().xy(), make_float4(value * tint, 1.0f));
    }};
}

// Stage 2/3: the separable 9-tap Gaussian blur (sigma = 2), horizontal and
// vertical.  The taps are unrolled at trace time, so the weights are plain host
// constants.
template<bool Horizontal>
[[nodiscard]] inline auto make_image_blur_kernel() noexcept {
    return Kernel2D{[](ImageFloat input, ImageFloat output) noexcept {
        set_block_size(kImageBlock, kImageBlock, 1u);
        UInt2 uv = dispatch_id().xy();
        UInt2 size = dispatch_size().xy();
        // Gaussian weights (normalized, sigma = 2.0)
        Float weights[9] = {0.006f, 0.028f, 0.084f, 0.168f, 0.224f,
                            0.168f, 0.084f, 0.028f, 0.006f};
        Int offsets[9] = {-4, -3, -2, -1, 0, 1, 2, 3, 4};
        Float3 sum = def(make_float3(0.0f));
        for (uint i = 0u; i < 9u; ++i) {
            UInt2 sample_uv = [&] {
                if constexpr (Horizontal) {
                    return make_uint2(clamp(cast<uint>(cast<int>(uv.x) + offsets[i]),
                                            0u, size.x - 1u),
                                      uv.y);
                } else {
                    return make_uint2(uv.x,
                                      clamp(cast<uint>(cast<int>(uv.y) + offsets[i]),
                                            0u, size.y - 1u));
                }
            }();
            sum = sum + input.read(sample_uv).xyz() * weights[i];
        }
        output.write(uv, make_float4(sum, 1.0f));
    }};
}

// Stage 4: 3x3 Sobel edge detection (gradient magnitude in R, direction in G).
[[nodiscard]] inline auto make_image_sobel_kernel() noexcept {
    return Kernel2D{[](ImageFloat input, ImageFloat output) noexcept {
        set_block_size(kImageBlock, kImageBlock, 1u);
        UInt2 uv = dispatch_id().xy();
        UInt2 size = dispatch_size().xy();
        auto sample = [&](Int2 offset) noexcept {
            UInt x = clamp(cast<uint>(cast<int>(uv.x) + offset.x), 0u, size.x - 1u);
            UInt y = clamp(cast<uint>(cast<int>(uv.y) + offset.y), 0u, size.y - 1u);
            return input.read(make_uint2(x, y)).x;// luminance
        };
        // Sobel X: [-1 0 1; -2 0 2; -1 0 1]
        Float gx = sample(make_int2(-1, -1)) * -1.0f +
                   sample(make_int2(1, -1)) * 1.0f +
                   sample(make_int2(-1, 0)) * -2.0f +
                   sample(make_int2(1, 0)) * 2.0f +
                   sample(make_int2(-1, 1)) * -1.0f +
                   sample(make_int2(1, 1)) * 1.0f;
        // Sobel Y: [-1 -2 -1; 0 0 0; 1 2 1]
        Float gy = sample(make_int2(-1, -1)) * -1.0f +
                   sample(make_int2(0, -1)) * -2.0f +
                   sample(make_int2(1, -1)) * -1.0f +
                   sample(make_int2(-1, 1)) * 1.0f +
                   sample(make_int2(0, 1)) * 2.0f +
                   sample(make_int2(1, 1)) * 1.0f;
        Float magnitude = sqrt(gx * gx + gy * gy);
        Float angle = atan2(gy, gx) / 3.14159f * 0.5f + 0.5f;
        output.write(uv, make_float4(magnitude, angle, 0.0f, 1.0f));
    }};
}

// Stage 5: unsharp mask + edge overlay + vignette.
[[nodiscard]] inline auto make_image_composite_kernel() noexcept {
    return Kernel2D{[](ImageFloat original, ImageFloat blurred, ImageFloat edges,
                       ImageFloat output, Float edge_intensity) noexcept {
        set_block_size(kImageBlock, kImageBlock, 1u);
        UInt2 uv = dispatch_id().xy();
        Float4 orig = original.read(uv);
        Float4 blur = blurred.read(uv);
        Float4 edge = edges.read(uv);
        // unsharp mask: sharpened = original + (original - blurred) * amount
        Float sharpen_amount = 1.5f;
        Float3 sharpened = orig.xyz() + (orig.xyz() - blur.xyz()) * sharpen_amount;
        sharpened = clamp(sharpened, 0.0f, 1.0f);
        // edge overlay with color
        Float3 edge_color = make_float3(0.0f, 0.8f, 1.0f) * edge.x * edge_intensity;
        Float3 final_color = lerp(sharpened, edge_color, edge.x * 0.5f);
        // vignette
        Float2 coord = make_float2(uv) / make_float2(dispatch_size().xy());
        Float center_dist = length(coord - make_float2(0.5f));
        Float vignette = 1.0f - center_dist * 0.5f;
        output.write(uv, make_float4(final_color * vignette, 1.0f));
    }};
}

void bench_compute_image_pipeline(compute::Device &device, compute::Stream &stream,
                                  const BenchConfig &cfg,
                                  std::vector<BenchResult> &results) noexcept {
    auto source_image = device.create_image<float>(PixelStorage::FLOAT4, kImageSize, kImageSize);
    auto temp_image = device.create_image<float>(PixelStorage::FLOAT4, kImageSize, kImageSize);
    auto blurred_image = device.create_image<float>(PixelStorage::FLOAT4, kImageSize, kImageSize);
    auto edge_image = device.create_image<float>(PixelStorage::FLOAT4, kImageSize, kImageSize);
    auto output_image = device.create_image<float>(PixelStorage::FLOAT4, kImageSize, kImageSize);

    double compile_ms = 0.0;
    auto pattern_kernel = compile_uncached(device, make_image_pattern_kernel(),
                                           "bench_compute_image_pattern", cfg, compile_ms);
    auto blur_h_kernel = compile_uncached(device, make_image_blur_kernel<true>(),
                                          "bench_compute_image_blur_h", cfg, compile_ms);
    auto blur_v_kernel = compile_uncached(device, make_image_blur_kernel<false>(),
                                          "bench_compute_image_blur_v", cfg, compile_ms);
    auto sobel_kernel = compile_uncached(device, make_image_sobel_kernel(),
                                         "bench_compute_image_sobel", cfg, compile_ms);
    auto composite_kernel = compile_uncached(device, make_image_composite_kernel(),
                                             "bench_compute_image_composite", cfg, compile_ms);

    auto dispatch = [&] {
        CommandList cmds = CommandList::create(5u);
        cmds << pattern_kernel(source_image).dispatch(kImageSize, kImageSize);
        cmds << blur_h_kernel(source_image, temp_image).dispatch(kImageSize, kImageSize);
        cmds << blur_v_kernel(temp_image, blurred_image).dispatch(kImageSize, kImageSize);
        cmds << sobel_kernel(source_image, edge_image).dispatch(kImageSize, kImageSize);
        cmds << composite_kernel(source_image, blurred_image, edge_image, output_image, 0.75f)
                    .dispatch(kImageSize, kImageSize);
        stream << cmds.commit();
    };

    auto result = run_timed_bench("compute", "compute_image_pipeline", stream, cfg,
                                  compile_ms, static_cast<double>(kImageSize) * kImageSize,
                                  "px/s", dispatch);

    if (!cfg.verify) {
        result.verified = false;
        result.check_note = "verification disabled";
        results.push_back(std::move(result));
        return;
    }
    if (cfg.warmup_iters == 0u && cfg.timed_iters == 0u) {
        dispatch();
        stream << compute::synchronize();
    }

    constexpr size_t pixel_count = static_cast<size_t>(kImageSize) * kImageSize;
    luisa::vector<float> host_pixels(pixel_count * 4u);
    stream << output_image.copy_to(luisa::span{host_pixels}) << compute::synchronize();
    size_t non_finite = 0u;
    double checksum = 0.0;
    float lo = std::numeric_limits<float>::max();
    float hi = std::numeric_limits<float>::lowest();
    for (size_t i = 0u; i < pixel_count * 4u; ++i) {
        auto value = host_pixels[i];
        if (!std::isfinite(value)) { non_finite++; }
        checksum += static_cast<double>(value);
        lo = std::min(lo, value);
        hi = std::max(hi, value);
    }
    bool ok = non_finite == 0u && std::isfinite(checksum) && checksum != 0.0 &&
              std::abs(checksum) > 1.0;
    result.verified = ok;
    result.check_note = luisa::format(
        "{}x{} FLOAT4, non_finite={}, checksum={:.6f}, min={:.6f}, max={:.6f}",
        kImageSize, kImageSize, non_finite, checksum, lo, hi);
    if (!ok) { LUISA_WARNING("{}: {}", result.name, result.check_note); }
    results.push_back(std::move(result));
}

}// namespace

// ---------------------------------------------------------------------------
// The "compute" suite.
// ---------------------------------------------------------------------------

void run_compute_benchmarks(compute::Device &device, compute::Stream &stream,
                            const BenchConfig &cfg,
                            std::vector<BenchResult> &results) {
    if (should_run(cfg, "compute", "compute_radix_sort")) {
        bench_compute_radix_sort(device, stream, cfg, results);
    }
    if (should_run(cfg, "compute", "compute_lbvh_build")) {
        bench_compute_lbvh_build(device, stream, cfg, results);
    }
    if (should_run(cfg, "compute", "compute_ngram_hash")) {
        bench_compute_ngram_hash(device, stream, cfg, results);
    }
    if (should_run(cfg, "compute", "compute_image_pipeline")) {
        bench_compute_image_pipeline(device, stream, cfg, results);
    }
}

}// namespace luisa::bench
