#pragma once
// Persistent PINNED download staging for the Qwen DSL executor
// (opt_plan/03 section 6, opt_plan/09 N4.3).
//
// What it changes. Before this, every logits download and every bounded greedy
// token download was a `BufferView::copy_to(host span)`, i.e. a CUDA
// BufferDownloadCommand: on this backend that is a device->host DMA into the
// STREAM'S 32 MiB download pool followed by a stream-ordered host->host memcpy into
// the caller's span (cuda_command_encoder.cpp, BufferDownloadCommand). The pinned
// path instead owns a page-locked, device-mapped host buffer for the engine's whole
// lifetime and downloads with a device->device copy into it (one DMA into
// host-mapped memory), synchronizes the stream, and then memcpys from that host
// buffer into the caller's span. Note the honest accounting: BOTH realizations move
// the bytes twice (DMA + one host memcpy); the pinned one replaces the driver's
// bounce/staging copy with a direct DMA into page-locked memory that the engine owns
// and reuses, and it does not depend on the download pool having a free block (the
// pool warns and falls back to ad-hoc allocation when it does not). The second
// memcpy is NOT eliminated and no log claims otherwise.
//
// Lifetime contract. The pinned buffers are engine members, so they outlive every
// command this module records; a queued copy is always followed by a stream
// synchronize before the host reads the pinned memory, so no command can still be
// writing it when the caller's span is filled. `reset()` only drops the accounting:
// the buffers (and their pinning) are reused for the whole session.
//
// Fail-closed selection. `PinnedMemoryExt` is optional: on a backend that does not
// expose it (or when the caller opts out) the engine keeps the previous pageable
// path unchanged and reports why (QwenEngineInfo::pinned_*).

#include <cstdint>

#include <luisa/backends/ext/pinned_memory_ext.hpp>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "qwen/common.h"

namespace luisa::qwen {

// Why the pinned path is or is not in use. Every field is a measured fact (an
// extension lookup, an allocation result), never a tier/name inference.
struct QwenPinnedSelection {
    bool requested{false};   // QwenEngineOptions::dsl_pinned_transfers
    bool extension{false};   // device.extension<PinnedMemoryExt>() != nullptr
    bool tokens{false};      // the token ring was allocated
    bool logits{false};      // the logits staging buffer was allocated
    bool selected{false};    // BOTH allocations succeeded and the path is in use
    uint32_t ring_tokens{0}; // token ring capacity
    uint64_t logits_bytes{0};
    luisa::string reason;    // "selected" / the refusal, with the measured cause
    [[nodiscard]] luisa::string summary() const noexcept;
};

// Accounting of the pinned path, reported through QwenEngineInfo so a log can prove
// the realization engaged instead of assuming it (and so the A/B arm shows 0).
struct QwenPinnedStats {
    uint64_t downloads{0};  // completed downloads (logits and token ranges)
    uint64_t slices{0};     // ring slices used (a range larger than the ring loops)
    uint64_t bytes{0};      // device->pinned bytes moved
    uint64_t host_copies{0};// host memcpys into the caller's span (NOT eliminated)
};

// Persistent pinned staging: one uint32 ring for bounded token history downloads and
// one float buffer for logits. Both are device-visible buffers allocated through
// PinnedMemoryExt::allocate_pinned_memory, so their `native_handle()` is the
// page-locked host address the CPU reads after the synchronization.
class QwenPinnedRing {
public:
    QwenPinnedRing() noexcept = default;
    // Never throws. A failed allocation leaves the object invalid and every
    // download call refuses instead of silently copying through pageable memory.
    [[nodiscard]] static QwenPinnedRing create(compute::Device &device,
                                               compute::PinnedMemoryExt *ext,
                                               uint32_t ring_tokens,
                                               uint64_t logits_elements) noexcept;

    [[nodiscard]] bool valid() const noexcept { return _tokens && _logits; }
    [[nodiscard]] uint32_t capacity() const noexcept { return _ring_tokens; }
    [[nodiscard]] uint64_t logits_elements() const noexcept { return _logits.size(); }

    // `out[i]` = src[src_first + i] for every i, for an ARBITRARY range: while the
    // remaining range exceeds the ring it is downloaded in ring-sized slices, each
    // stream-ordered copy being synchronized before the host memcpy (so reuse of the
    // ring can never race a pending copy).
    [[nodiscard]] Status download_tokens(compute::Stream &stream,
                                         compute::BufferView<uint32_t> src,
                                         uint32_t src_first,
                                         luisa::span<uint32_t> out,
                                         QwenPinnedStats &stats) noexcept;
    // Same shape for FP32 logits (and any other float range).
    [[nodiscard]] Status download_floats(compute::Stream &stream,
                                         compute::BufferView<float> src,
                                         luisa::span<float> out,
                                         QwenPinnedStats &stats) noexcept;

    // The caller-visible host addresses (null when not pinned). Exposed for tests
    // that verify the round trip without going through an engine.
    [[nodiscard]] uint32_t *host_tokens() noexcept;
    [[nodiscard]] float *host_logits() noexcept;
    [[nodiscard]] compute::Buffer<uint32_t> &token_buffer() noexcept { return _tokens; }
    [[nodiscard]] compute::Buffer<float> &logits_buffer() noexcept { return _logits; }

private:
    compute::Buffer<uint32_t> _tokens;
    compute::Buffer<float> _logits;
    uint32_t _ring_tokens{0};
};

}// namespace luisa::qwen
