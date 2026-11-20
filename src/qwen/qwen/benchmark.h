#pragma once
// Opt-in Qwen command-level benchmark instrumentation.
//
// Normal execution uses QwenCommandBatch as a plain CommandList wrapper: every
// command is appended and submitted at the executor's existing batch boundary.
// With QwenEngineOptions::benchmark=true, each appended command is instead
// committed on the caller's single stream and followed by synchronize(). The
// recorded submit_sync_ms is therefore a HOST WALL interval that includes
// submission, execution, synchronization and host scheduling. It is not a
// hardware GPU timestamp and synchronized samples must not be summed and
// presented as normal-mode GPU time.

#include <cstdint>
#include <utility>

#include <luisa/core/clock.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/stream.h>

#include "qwen/common.h"

namespace luisa::qwen {

enum class QwenBenchmarkPhase : uint32_t {
    reset = 0,
    prefill,
    decode,
    mtp,
};
[[nodiscard]] luisa::string_view qwen_benchmark_phase_name(QwenBenchmarkPhase phase) noexcept;

struct QwenBenchmarkSample {
    uint64_t ordinal{0};
    QwenBenchmarkPhase phase{QwenBenchmarkPhase::decode};
    uint32_t position{0};
    uint32_t rows{0};
    uint32_t layer{UINT32_MAX};// UINT32_MAX = global/non-layer command
    luisa::string name;
    double record_ms{0.0};     // host append/encode interval
    double submit_sync_ms{0.0};// commit through completed synchronize
    double total_ms{0.0};      // record + submit/sync + instrumentation
};

struct QwenBenchmarkReport {
    luisa::vector<QwenBenchmarkSample> samples;
    uint64_t command_count{0};
    uint64_t submit_count{0};
    uint64_t synchronize_count{0};
    double record_ms{0.0};
    double submit_sync_ms{0.0};
    double total_ms{0.0};
    void clear() noexcept;
};

// Noncopyable recorder used by both DSL and Tile executors. annotate() applies
// to the next and subsequent commands until changed; layer() updates the
// current decoder-layer attribution. Empty flush() calls are no-ops.
class QwenCommandBatch : concepts::Noncopyable {
public:
    explicit QwenCommandBatch(compute::Stream &stream,
                              QwenBenchmarkReport *report = nullptr,
                              QwenBenchmarkPhase phase = QwenBenchmarkPhase::decode,
                              uint32_t position = 0u, uint32_t rows = 0u,
                              size_t reserve_commands = 0u) noexcept;
    // Drops a partially recorded list on a recoverable error path instead of
    // triggering CommandList's 'forgot to commit' assertion in debug builds.
    ~QwenCommandBatch() noexcept;

    QwenCommandBatch &operator<<(luisa::unique_ptr<compute::Command> &&command) noexcept;
    QwenCommandBatch &annotate(luisa::string_view name) noexcept;
    QwenCommandBatch &layer(uint32_t layer) noexcept;
    QwenCommandBatch &set_context(QwenBenchmarkPhase phase, uint32_t position, uint32_t rows) noexcept;

    // Commit any pending commands, optionally waiting for completion. In
    // per-command mode this is normally empty because operator<< already did
    // both. It never creates an extra empty submission/synchronization.
    void flush(bool synchronize = true) noexcept;
    void clear() noexcept;
    // Transfer a normal-mode list to a consumer such as CudaGraphExt. The
    // caller takes ownership of command submission and any consumed-list error
    // path. Per-command benchmark mode is empty by the time this is called.
    [[nodiscard]] compute::CommandList take_commands() noexcept {
        return std::move(_commands);
    }
    [[nodiscard]] bool empty() const noexcept { return _commands.empty(); }
    [[nodiscard]] bool instrumented() const noexcept { return _report != nullptr; }

private:
    compute::Stream *_stream{nullptr};
    QwenBenchmarkReport *_report{nullptr};
    compute::CommandList _commands;
    QwenBenchmarkPhase _phase{QwenBenchmarkPhase::decode};
    uint32_t _position{0};
    uint32_t _rows{0};
    uint32_t _layer{UINT32_MAX};
    luisa::string_view _name{"unlabeled"};
    luisa::string _owned_name;// profile-only; avoids borrowed-view lifetime traps
    size_t _reserve_commands{0};
};

// Aggregate synchronized command samples by exact operator name, retaining
// first-seen order. Counts and wall durations are sums within one report.
struct QwenBenchmarkAggregate {
    luisa::string name;
    uint64_t count{0};
    double record_ms{0.0};
    double submit_sync_ms{0.0};
    double total_ms{0.0};
};
[[nodiscard]] luisa::vector<QwenBenchmarkAggregate> qwen_benchmark_aggregate(
    const QwenBenchmarkReport &report, bool per_layer = false);

// Append one report as UTF-8 CSV. Creates the file and writes a header when
// `append` is false or the existing file is empty. Values are raw samples;
// callers remain responsible for recording run configuration separately.
[[nodiscard]] Status qwen_benchmark_append_csv(
    const luisa::filesystem::path &path, const QwenBenchmarkReport &report,
    bool append, luisa::string_view variant, luisa::string_view backend,
    uint64_t iteration) noexcept;

}// namespace luisa::qwen
