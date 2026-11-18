#pragma once
// Shared harness for the Qwen device regression tests (tests/integration/qwen).
//
// Both test binaries (test_qwen_ops_dsl, test_qwen_ops_tile) take the backend
// EXPLICITLY as argv[1] (cuda / dx / vk / ...). There is no silent fallback:
// missing or unknown backends fail with a usage message and a nonzero exit.
// Every comparison prints an actionable, located error and increments the
// failure counter; main() returns nonzero when any check failed.
//
// No real checkpoint is needed: all data is synthetic (deterministic RNG /
// trigonometric fills) with ragged shapes; expectations come from the
// independent CPU oracles in qwen/dsl_ops.h and qwen/tile_ops.h, plus six
// hand-computed external-contract fixtures per operator family set.
#include <luisa/core/logging.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <cmath>
#include <cstdio>
#include <type_traits>

namespace qwen_test {

using namespace luisa;

// Deterministic xorshift64* RNG: identical data on every backend/run.
struct Rng {
    uint64_t s{0x123456789ABCDEFull};
    uint32_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return static_cast<uint32_t>((s * 0x2545F4914F6CDD1Dull) >> 32);
    }
    float u(float lo, float hi) { return lo + (hi - lo) * (float(next() >> 8) / 16777216.0f); }
    void fill(luisa::vector<float> &v, float lo, float hi) {
        for (auto &x : v) { x = u(lo, hi); }
    }
};

inline int failures = 0;
inline int passes = 0;

// Unbuffered stdout/stderr: a backend crash must not swallow the check log
// (redirected stdout is fully buffered by default and the buffer is lost on
// segfault). Call once at the top of main().
inline void unbuffered_output() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
}

inline void check(bool ok, const char *name, const luisa::string &extra = {}) {
    std::printf("[%s] %s%s%s\n", ok ? "ok" : "FAIL", name,
                extra.empty() ? "" : " : ", extra.c_str());
    if (ok) {
        passes++;
    } else {
        failures++;
    }
}

inline bool close(float got, float want, float atol, float rtol) {
    return std::isfinite(got) && std::isfinite(want) &&
           std::fabs(got - want) <= atol + rtol * std::fabs(want);
}

// Elementwise comparison with located diagnostics: prints the first mismatch
// (index, got, want, tolerance) so a failure is actionable without a debugger.
inline bool check_close(luisa::span<const float> got, luisa::span<const float> want,
                        float atol, float rtol, const char *name) {
    if (got.size() != want.size()) {
        check(false, name,
              luisa::string{"size mismatch: got "} + std::to_string(got.size()).c_str() +
                  " want " + std::to_string(want.size()).c_str());
        return false;
    }
    for (size_t i = 0; i < got.size(); i++) {
        if (!close(got[i], want[i], atol, rtol)) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "index %zu: device=%.9g expected=%.9g (atol=%g rtol=%g); "
                          "check the kernel's decode/reduction path for this operator",
                          i, double(got[i]), double(want[i]), double(atol), double(rtol));
            check(false, name, buf);
            return false;
        }
    }
    check(true, name);
    return true;
}

// Bitwise-exact comparison (raw copies, gates that are pure passthrough).
inline bool check_exact(luisa::span<const float> got, luisa::span<const float> want,
                        const char *name) {
    if (got.size() != want.size()) {
        check(false, name, luisa::string{"size mismatch"});
        return false;
    }
    for (size_t i = 0; i < got.size(); i++) {
        if (got[i] != want[i]) {
            char buf[192];
            std::snprintf(buf, sizeof(buf), "index %zu: device=%.9g expected=%.9g (exact)",
                          i, double(got[i]), double(want[i]));
            check(false, name, buf);
            return false;
        }
    }
    check(true, name);
    return true;
}

// LIFETIME CONTRACT: copy_from/copy_to record a BORROWED host pointer that
// the backend dereferences only when the command list is encoded/executed
// (DX/VK encode at submit; the readback memcpy runs after GPU completion).
// A function-local staging vector would dangle. Both helpers therefore
// SYNCHRONIZE before returning, so the caller's vector is provably still
// alive at the moment the host memory is read/written, and the returned
// data is complete. Do not remove the synchronize() calls.
template<typename T>
inline compute::Buffer<T> upload(compute::Device &dev, compute::Stream &stream,
                                 const luisa::vector<T> &data) {
    auto buf = dev.create_buffer<T>(data.empty() ? 1 : data.size());
    if (!data.empty()) {
        stream << buf.copy_from(luisa::span{data}) << compute::synchronize();
    }
    return buf;
}

template<typename T>
inline luisa::vector<T> download(compute::Stream &stream, compute::Buffer<T> &buf, size_t n) {
    // Poison: unwritten elements fail comparisons instead of reading as zero.
    luisa::vector<T> out(n);
    if constexpr (std::is_floating_point_v<T>) {
        for (auto &v : out) { v = static_cast<T>(-12345); }
    } else {
        for (auto &v : out) { v = static_cast<T>(0xDEADBEEFu); }
    }
    if (n > 0) {
        stream << buf.view(0, n).copy_to(luisa::span{out}) << compute::synchronize();
    }
    return out;
}

inline int summarize(const char *suite, const char *backend) {
    std::printf("%s on %s: %s (%d passed, %d failed)\n", suite, backend,
                failures == 0 ? "ALL PASSED" : "FAILED", passes, failures);
    if (failures != 0) {
        std::printf("Re-run with the failing backend as argv[1]; every FAIL line above "
                    "names the operator and the first mismatching element.\n");
    }
    return failures == 0 ? 0 : 1;
}

}// namespace qwen_test
