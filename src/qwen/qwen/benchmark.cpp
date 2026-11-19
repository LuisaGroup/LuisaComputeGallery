#include "qwen/benchmark.h"

#include <algorithm>
#include <fstream>
#include <iomanip>

namespace luisa::qwen {

[[nodiscard]] luisa::string_view qwen_benchmark_phase_name(QwenBenchmarkPhase phase) noexcept {
    switch (phase) {
        case QwenBenchmarkPhase::reset: return "reset";
        case QwenBenchmarkPhase::prefill: return "prefill";
        case QwenBenchmarkPhase::decode: return "decode";
        case QwenBenchmarkPhase::mtp: return "mtp";
    }
    return "unknown";
}

void QwenBenchmarkReport::clear() noexcept {
    samples.clear();
    command_count = 0u;
    submit_count = 0u;
    synchronize_count = 0u;
    record_ms = 0.0;
    submit_sync_ms = 0.0;
    total_ms = 0.0;
}

QwenCommandBatch::QwenCommandBatch(compute::Stream &stream, QwenBenchmarkReport *report,
                                   QwenBenchmarkPhase phase, uint32_t position,
                                   uint32_t rows, size_t reserve_commands) noexcept
    : _stream{&stream}, _report{report}, _commands{compute::CommandList::create(reserve_commands)},
      _phase{phase}, _position{position}, _rows{rows}, _reserve_commands{reserve_commands} {}

QwenCommandBatch::~QwenCommandBatch() noexcept {
    _commands.clear();
}

QwenCommandBatch &QwenCommandBatch::operator<<(luisa::unique_ptr<compute::Command> &&command) noexcept {
    if (_report == nullptr) {
        _commands << std::move(command);
        return *this;
    }
    Clock total_clock;
    Clock clock;
    total_clock.tic();
    clock.tic();
    _commands << std::move(command);
    const double record_ms = clock.toc();
    clock.tic();
    *_stream << _commands.commit() << compute::synchronize();
    const double submit_sync_ms = clock.toc();
    _commands.clear();
    if (_reserve_commands != 0u) { _commands.reserve(_reserve_commands, 0u); }

    QwenBenchmarkSample sample;
    sample.ordinal = _report->samples.size();
    sample.phase = _phase;
    sample.position = _position;
    sample.rows = _rows;
    sample.layer = _layer;
    sample.name = _name.empty() ? luisa::string{"unlabeled"} : luisa::string{_name};
    sample.record_ms = record_ms;
    sample.submit_sync_ms = submit_sync_ms;
    sample.total_ms = total_clock.toc();
    _report->samples.emplace_back(std::move(sample));
    _report->command_count++;
    _report->submit_count++;
    _report->synchronize_count++;
    _report->record_ms += record_ms;
    _report->submit_sync_ms += submit_sync_ms;
    _report->total_ms += _report->samples.back().total_ms;
    return *this;
}

QwenCommandBatch &QwenCommandBatch::annotate(luisa::string_view name) noexcept {
    if (_report != nullptr) {
        _owned_name = luisa::string{name};
        _name = _owned_name;
    }
    return *this;
}

QwenCommandBatch &QwenCommandBatch::layer(uint32_t layer) noexcept {
    _layer = layer;
    return *this;
}

QwenCommandBatch &QwenCommandBatch::set_context(QwenBenchmarkPhase phase,
                                                uint32_t position, uint32_t rows) noexcept {
    _phase = phase;
    _position = position;
    _rows = rows;
    return *this;
}

void QwenCommandBatch::flush(bool synchronize) noexcept {
    if (_commands.empty()) { return; }
    if (synchronize) {
        *_stream << _commands.commit() << compute::synchronize();
    } else {
        *_stream << _commands.commit();
    }
    _commands.clear();
    if (_reserve_commands != 0u) { _commands.reserve(_reserve_commands, 0u); }
}

void QwenCommandBatch::clear() noexcept {
    _commands.clear();
    if (_reserve_commands != 0u) { _commands.reserve(_reserve_commands, 0u); }
}

[[nodiscard]] luisa::vector<QwenBenchmarkAggregate> qwen_benchmark_aggregate(
    const QwenBenchmarkReport &report, bool per_layer) {
    luisa::vector<QwenBenchmarkAggregate> out;
    for (const auto &sample : report.samples) {
        luisa::string key;
        if (per_layer && sample.layer != UINT32_MAX) {
            key = luisa::string{"layer."} + luisa::string{std::to_string(sample.layer)} + ".";
        }
        key += sample.name;
        auto it = std::find_if(out.begin(), out.end(), [&](const auto &a) { return a.name == key; });
        if (it == out.end()) {
            QwenBenchmarkAggregate a;
            a.name = std::move(key);
            out.emplace_back(std::move(a));
            it = out.end() - 1;
        }
        it->count++;
        it->record_ms += sample.record_ms;
        it->submit_sync_ms += sample.submit_sync_ms;
        it->total_ms += sample.total_ms;
    }
    return out;
}

namespace {
void csv_string(std::ostream &out, luisa::string_view value) {
    out << '"';
    for (const char c : value) {
        if (c == '"') { out << "\"\""; } else { out << c; }
    }
    out << '"';
}
}// namespace

[[nodiscard]] Status qwen_benchmark_append_csv(
    const luisa::filesystem::path &path, const QwenBenchmarkReport &report,
    bool append, luisa::string_view variant, luisa::string_view backend,
    uint64_t iteration) noexcept {
    std::error_code ec;
    bool write_header = !append;
    if (append) {
        const bool exists = luisa::filesystem::exists(path, ec);
        if (ec) {
            return Status::error(StatusCode::io_error, "cannot inspect Qwen benchmark CSV",
                                 luisa::string{path.string()});
        }
        if (!exists) {
            write_header = true;
        } else {
            const auto size = luisa::filesystem::file_size(path, ec);
            if (ec) {
                return Status::error(StatusCode::io_error, "cannot inspect Qwen benchmark CSV size",
                                     luisa::string{path.string()});
            }
            write_header = size == 0u;
        }
    }
    std::ofstream out{path, std::ios::binary | (append ? std::ios::app : std::ios::trunc)};
    if (!out) {
        return Status::error(StatusCode::io_error, "cannot open Qwen benchmark CSV", luisa::string{path.string()});
    }
    out << std::setprecision(12);
    if (write_header) {
        out << "variant,backend,iteration,ordinal,phase,position,rows,layer,name,record_ms,submit_sync_ms,total_ms\n";
    }
    for (const auto &s : report.samples) {
        csv_string(out, variant);
        out << ',';
        csv_string(out, backend);
        out << ',' << iteration << ',' << s.ordinal << ','
            << qwen_benchmark_phase_name(s.phase) << ',' << s.position << ',' << s.rows << ',';
        if (s.layer == UINT32_MAX) { out << "global"; } else { out << s.layer; }
        out << ',';
        csv_string(out, s.name);
        out << ',' << s.record_ms << ',' << s.submit_sync_ms << ',' << s.total_ms << '\n';
    }
    out.flush();
    if (!out) {
        return Status::error(StatusCode::io_error, "failed while writing Qwen benchmark CSV", luisa::string{path.string()});
    }
    return Status::success();
}

}// namespace luisa::qwen
