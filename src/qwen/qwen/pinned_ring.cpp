// Persistent pinned download staging -- see qwen/pinned_ring.h for the contract and
// for the honest copy accounting (the pinned path does NOT remove the second copy).

#include "qwen/pinned_ring.h"

#include <algorithm>
#include <cstring>

#include <luisa/core/logging.h>

namespace luisa::qwen {

[[nodiscard]] luisa::string QwenPinnedSelection::summary() const noexcept {
    return luisa::format("requested={} extension={} ring={} logits={} selected={} "
                         "ring-tokens={} logits-bytes={} reason='{}'",
                         requested ? 1 : 0, extension ? 1 : 0, tokens ? 1 : 0,
                         logits ? 1 : 0, selected ? 1 : 0, ring_tokens, logits_bytes, reason);
}

[[nodiscard]] QwenPinnedRing QwenPinnedRing::create(compute::Device &device,
                                                    compute::PinnedMemoryExt *ext,
                                                    uint32_t ring_tokens,
                                                    uint64_t logits_elements) noexcept {
    QwenPinnedRing ring;
    if (ext == nullptr) { return ring; }
    if (ring_tokens == 0u || logits_elements == 0u) { return ring; }
    // write_combined = false: the engine READS these buffers on the host, and a
    // write-combined mapping makes host reads several times slower.
    compute::PinnedMemoryOption option;
    option.write_combined = false;
    ring._tokens = ext->allocate_pinned_memory<uint32_t>(ring_tokens, option);
    ring._logits = ext->allocate_pinned_memory<float>(static_cast<size_t>(logits_elements), option);
    if (!ring._tokens || !ring._logits) {
        // Fail closed: release whatever was allocated so a partial pinning never
        // looks like a usable path.
        ring._tokens = {};
        ring._logits = {};
        return ring;
    }
    ring._ring_tokens = ring_tokens;
    (void)device;// the buffers own their pinning; the device only supplied the ext
    return ring;
}

[[nodiscard]] uint32_t *QwenPinnedRing::host_tokens() noexcept {
    return _tokens ? static_cast<uint32_t *>(_tokens.native_handle()) : nullptr;
}

[[nodiscard]] float *QwenPinnedRing::host_logits() noexcept {
    return _logits ? static_cast<float *>(_logits.native_handle()) : nullptr;
}

[[nodiscard]] Status QwenPinnedRing::download_tokens(compute::Stream &stream,
                                                      compute::BufferView<uint32_t> src,
                                                      uint32_t src_first,
                                                      luisa::span<uint32_t> out,
                                                      QwenPinnedStats &stats) noexcept {
    if (out.empty()) { return Status::success(); }
    if (!valid()) {
        return Status::error(StatusCode::internal_error,
                             "pinned token download on an unallocated ring");
    }
    auto *host = host_tokens();
    if (host == nullptr) {
        return Status::error(StatusCode::internal_error,
                             "pinned token ring has no host address");
    }
    if (src_first > src.size() || out.size() > src.size() - src_first) {
        return Status::error(StatusCode::invalid_argument,
                             "pinned token download range exceeds the source buffer");
    }
    const auto total = static_cast<uint32_t>(out.size());
    uint32_t done = 0u;
    while (done < total) {
        const auto slice = std::min(total - done, _ring_tokens);
        // (1) stream-ordered device->device copy into the pinned buffer (one DMA
        // into host-mapped memory), (2) synchronize, (3) host memcpy into the
        // caller's span. The ring is reused only after the synchronization, so a
        // later slice can never overwrite a copy that is still pending.
        stream << src.subview(static_cast<size_t>(src_first) + done, slice)
                      .copy_to(_tokens.view(0u, slice));
        stream << compute::synchronize();
        std::memcpy(out.data() + done, host, static_cast<size_t>(slice) * sizeof(uint32_t));
        stats.slices++;
        stats.bytes += static_cast<uint64_t>(slice) * sizeof(uint32_t);
        stats.host_copies++;
        done += slice;
    }
    stats.downloads++;
    return Status::success();
}

[[nodiscard]] Status QwenPinnedRing::download_floats(compute::Stream &stream,
                                                      compute::BufferView<float> src,
                                                      luisa::span<float> out,
                                                      QwenPinnedStats &stats) noexcept {
    if (out.empty()) { return Status::success(); }
    if (!valid()) {
        return Status::error(StatusCode::internal_error,
                             "pinned logits download on an unallocated staging buffer");
    }
    if (out.size() > src.size() || out.size() > _logits.size()) {
        return Status::error(StatusCode::invalid_argument,
                             "pinned logits download range exceeds the staging buffer");
    }
    auto *host = host_logits();
    if (host == nullptr) {
        return Status::error(StatusCode::internal_error,
                             "pinned logits staging buffer has no host address");
    }
    const auto n = static_cast<uint32_t>(out.size());
    stream << src.subview(0u, n).copy_to(_logits.view(0u, n));
    stream << compute::synchronize();
    std::memcpy(out.data(), host, static_cast<size_t>(n) * sizeof(float));
    stats.slices++;
    stats.bytes += static_cast<uint64_t>(n) * sizeof(float);
    stats.host_copies++;
    stats.downloads++;
    return Status::success();
}

}// namespace luisa::qwen
