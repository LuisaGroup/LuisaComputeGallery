// Phase 9c LCV-2 — VK cooperative-vector spike. STANDALONE EXPERIMENT (throwaway).
//
// opt_plan/04 section 3 asks three questions before any adoption of
// VK_NV_cooperative_vector for the Qwen GEMV:
//   1. does the extension's *store/layout* contract let us write a matrix we own
//      into a ByteBuffer and read it back with the same layout the codegen
//      assumes (InferencingOptimalNV, transpose = false)?
//   2. does `cooperative_mat_mul` then produce the CPU reference for a known
//      K x M matrix and a known activation vector?
//   3. only if (2) holds: is it >= 1.5x the existing DSL GEMV on representative
//      shapes with cold-ish caches over >= 30 reps?
//
// This file answers (1) and (2) on the running device and prints the exact stage
// at which the answer stops. It is NOT engine integration: nothing here is
// reachable from QwenDslEngine, the recurrence is untouched, and a failure is
// reported as a rejection (never as a pass).
//
// Usage: test_qwen_coopvec_spike <backend>
//   vk    -> runs the stages when the device negotiated VK_NV_cooperative_vector
//   dx/cuda/... -> prints "SKIP: stage 0 ..." and exits 0 (fail-closed no-op)
//
// Exit code: 0 when every attempted stage passed OR the spike correctly skipped,
// 1 when a stage that could run produced a WRONG result (a real defect),
// 2 usage/device error.

#include "qwen_device_test_utils.h"

#include "qwen/caps.h"

#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string_view>

using namespace luisa;
using namespace luisa::compute;
using namespace qwen_test;
using luisa::qwen::QwenBackendCaps;
using luisa::qwen::qwen_probe_backend_caps;

namespace {

// Vertex-free progress log: every stage announces itself so a device-lost or a
// driver-level failure still leaves the exact stage in the log.
void stage(int index, const char *what) noexcept {
    std::printf("\n== stage %d: %s ==\n", index, what);
}

enum class Outcome {
    passed,
    skipped,
    wrong_result,
};

// ---------------------------------------------------------------------------
// Stage 1: the vector store/load layout contract.
//
// The vector half of the extension has no component-type configuration: a
// cooperative vector of N FP32 values is stored/loaded at a byte offset. If the
// bytes a cooperative_vector_store writes are NOT the N contiguous FP32 words we
// would upload from the host, then no weight matrix can be prepared offline and
// the whole feature is unusable for weights -- exactly the "stop" condition
// opt_plan/04 section 3 step 1 defines.
// ---------------------------------------------------------------------------
struct VectorLayoutReport {
    Outcome outcome{Outcome::skipped};
    bool store_contiguous{false};
    bool load_round_trip{false};
    bool offset_honored{false};
    float store_max_diff{std::numeric_limits<float>::quiet_NaN()};
    float load_max_diff{std::numeric_limits<float>::quiet_NaN()};
};

[[nodiscard]] VectorLayoutReport stage1_vector_layout(Device &device, Stream &stream) {
    constexpr uint32_t n = 16u;
    constexpr size_t elem = sizeof(float);
    constexpr size_t vector_bytes = n * elem;
    // Two vectors back to back: the second one is written at byte offset
    // `vector_bytes`, so a driver that ignores the offset is caught.
    constexpr size_t total_bytes = 2u * vector_bytes;
    VectorLayoutReport report;

    auto buffer = device.create_byte_buffer(total_bytes);
    auto out = device.create_buffer<float>(n);
    luisa::vector<std::byte> zeros(total_bytes, std::byte{0});

    // Store [1..n] at offset 0 and again at offset vector_bytes.
    // No set_block_size(): a bare Kernel1D dispatch runs one invocation, which is
    // what the in-tree cooperative-vector tests do (set_block_size requires a
    // multiple of 32, so 1 is only expressible as "unset").
    auto store_shader = device.compile(Kernel1D{[&](ByteBufferVar buf) noexcept {
        CoopVectorRef ref{CoopRefVecType::FLOAT32, n};
        ref.set_byte_offset(0u);
        CoopVector<float> v{n};
        for (auto i = 0u; i < n; ++i) { v[i] = static_cast<float>(i + 1u); }
        cooperative_vector_store(buf, ref, Expr<CoopVector<float>>{v});
        ref.set_byte_offset(static_cast<uint32_t>(vector_bytes));
        CoopVector<float> w{n};
        for (auto i = 0u; i < n; ++i) { w[i] = 100.0f + static_cast<float>(i + 1u); }
        cooperative_vector_store(buf, ref, Expr<CoopVector<float>>{w});
    }});
    // Load the vector at offset 0 through the cooperative path.
    auto load_shader = device.compile(Kernel1D{[&](ByteBufferVar buf, BufferFloat dst) noexcept {
        CoopVectorRef ref{CoopRefVecType::FLOAT32, n};
        ref.set_byte_offset(0u);
        auto v = cooperative_vector_load<float>(buf, ref);
        for (auto i = 0u; i < n; ++i) { dst.write(i, v[i]); }
    }});

    luisa::vector<std::byte> raw(total_bytes);
    luisa::vector<float> loaded(n, -12345.0f);
    stream << buffer.copy_from(zeros.data())
           << store_shader(buffer).dispatch(1u)
           << buffer.copy_to(raw.data())
           << load_shader(buffer, out).dispatch(1u)
           << out.copy_to(luisa::span{loaded})
           << synchronize();

    // (a) does the store write plain contiguous FP32?
    auto *words = reinterpret_cast<const float *>(raw.data());
    float store_diff = 0.0f;
    bool contiguous = true;
    for (auto i = 0u; i < n; ++i) {
        auto want = static_cast<float>(i + 1u);
        if (words[i] != want) { contiguous = false; }
        store_diff = std::max(store_diff, std::fabs(words[i] - want));
    }
    float second_diff = 0.0f;
    for (auto i = 0u; i < n; ++i) {
        second_diff = std::max(second_diff,
                                std::fabs(words[n + i] - (100.0f + static_cast<float>(i + 1u))));
    }
    report.store_contiguous = contiguous;
    report.store_max_diff = store_diff;
    report.offset_honored = second_diff == 0.0f;
    std::printf("stage1: raw bytes after cooperative_vector_store (FP32 words):");
    for (auto i = 0u; i < 2u * n; ++i) {
        std::printf(" %g", static_cast<double>(words[i]));
    }
    std::printf("\n");
    std::printf("stage1: store contiguous=%d max|diff|=%.9g ; second vector @+%zu B max|diff|=%.9g\n",
                contiguous ? 1 : 0, static_cast<double>(store_diff), vector_bytes,
                static_cast<double>(second_diff));

    // (b) does the cooperative load return what the store wrote?
    float load_diff = 0.0f;
    for (auto i = 0u; i < n; ++i) {
        load_diff = std::max(load_diff,
                              std::fabs(loaded[i] - static_cast<float>(i + 1u)));
    }
    report.load_max_diff = load_diff;
    report.load_round_trip = load_diff == 0.0f;
    std::printf("stage1: cooperative_vector_load round-trip max|diff|=%.9g\n",
                static_cast<double>(load_diff));

    check(contiguous, "coopvec stage1: cooperative_vector_store writes contiguous FP32",
          luisa::format("max|diff|={}", store_diff));
    check(report.offset_honored, "coopvec stage1: the byte offset is honored",
          luisa::format("max|diff|={}", second_diff));
    check(report.load_round_trip, "coopvec stage1: load returns the stored vector",
          luisa::format("max|diff|={}", load_diff));
    report.outcome = (contiguous && report.offset_honored && report.load_round_trip)
                         ? Outcome::passed
                         : Outcome::wrong_result;
    return report;
}

// ---------------------------------------------------------------------------
// Stage 2: `cooperative_mat_mul` against a CPU reference.
//
// K x M FP32 matrix in a ByteBuffer, uploaded from the host (that is how weights
// would arrive), probed with a known activation vector. Two candidate readings of
// the same bytes are compared, because the "InferencingOptimalNV" memory layout is
// implementation-defined and the codegen hardcodes it:
//   A: element (k, m) at byte 4*(k*M + m)   -> y[m] = sum_k W[k*M+m] * x[k]
//   B: element (k, m) at byte 4*(m*K + k)   -> y[m] = sum_k W[m*K+k] * x[k]
// A driver that implements neither (or no FP32 configuration at all) shows up as
// a large error against both, which is a REJECTION, not a pass.
// ---------------------------------------------------------------------------
struct MatMulReport {
    Outcome outcome{Outcome::skipped};
    float err_layout_a{std::numeric_limits<float>::quiet_NaN()};
    float err_layout_b{std::numeric_limits<float>::quiet_NaN()};
    float err_zero{std::numeric_limits<float>::quiet_NaN()};
    bool layout_a{false};
    bool layout_b{false};
};

[[nodiscard]] MatMulReport stage2_matmul(Device &device, Stream &stream) {
    constexpr uint32_t k_dim = 8u;
    constexpr uint32_t m_dim = 16u;
    constexpr float tol = 1e-4f;
    MatMulReport report;

    // Deterministic data: w[k][m] = 0.25*k - 0.5*m + 1, x[k] = 0.5*k + 1.
    luisa::vector<float> host_matrix(size_t(k_dim) * m_dim);
    luisa::vector<float> host_x(k_dim);
    for (auto k = 0u; k < k_dim; ++k) {
        host_x[k] = 0.5f * static_cast<float>(k) + 1.0f;
        for (auto m = 0u; m < m_dim; ++m) {
            host_matrix[size_t(k) * m_dim + m] =
                0.25f * static_cast<float>(k) - 0.5f * static_cast<float>(m) + 1.0f;
        }
    }
    // Reference A: the buffer is read K-major (m contiguous).
    luisa::vector<float> ref_a(m_dim, 0.0f);
    for (auto m = 0u; m < m_dim; ++m) {
        float acc = 0.0f;
        for (auto k = 0u; k < k_dim; ++k) {
            acc += host_matrix[size_t(k) * m_dim + m] * host_x[k];
        }
        ref_a[m] = acc;
    }
    // Reference B: the same bytes read M-major (k contiguous).
    luisa::vector<float> ref_b(m_dim, 0.0f);
    for (auto m = 0u; m < m_dim; ++m) {
        float acc = 0.0f;
        for (auto k = 0u; k < k_dim; ++k) {
            acc += host_matrix[size_t(m) * k_dim + k] * host_x[k];
        }
        ref_b[m] = acc;
    }

    auto matrix = device.create_byte_buffer(host_matrix.size() * sizeof(float));
    auto x_buf = device.create_buffer<float>(k_dim);
    auto y_buf = device.create_buffer<float>(m_dim);
    // One host-side branch: with a single invocation the whole cooperative
    // vector belongs to it; with a warp only lane 0 writes, because every lane
    // holds the full slice in this spike (the second variant is the check for a
    // driver that expects a DISTRIBUTED value instead).
    auto run = [&](uint32_t block_size, const char *label) {
        luisa::vector<float> y(m_dim, -12345.0f);
        luisa::vector<float> zeros(m_dim, 0.0f);
        // The compile step is where the unsupported configuration was measured to
        // kill the process (after the SPIR-V module was produced, inside the Vulkan
        // driver's pipeline creation), so it gets its own marker.
        std::printf("stage2[%s]: compile\n", label);
        auto shader = device.compile(Kernel1D{[&](ByteBufferVar mat, BufferFloat x,
                                                  BufferFloat y) noexcept {
            if (block_size > 1u) { set_block_size(block_size); }
            CoopMatrixRef mref{CoopRefVecType::FLOAT32, k_dim, m_dim};
            mref.set_byte_offset(0u);
            CoopVector<float> xv{k_dim};
            for (auto i = 0u; i < k_dim; ++i) {
                xv[i] = x.read(i);
            }
            auto yv = cooperative_mat_mul<float, float>(
                mat, mref, Expr<CoopVector<float>>{xv});
            if (block_size == 1u) {
                for (auto i = 0u; i < m_dim; ++i) { y.write(i, yv[i]); }
            } else {
                $if (warp_lane_id() == 0u) {
                    for (auto i = 0u; i < m_dim; ++i) { y.write(i, yv[i]); }
                };
            }
        }});
        // Markers between the four stream operations: the measured failure mode
        // of an unsupported FP32 configuration is a PROCESS-LEVEL crash inside the
        // driver, so the last marker printed is the only localization available.
        std::printf("stage2[%s]: upload\n", label);
        stream << matrix.copy_from(host_matrix.data())
               << x_buf.copy_from(luisa::span{host_x})
               << y_buf.copy_from(luisa::span{zeros})
               << synchronize();
        std::printf("stage2[%s]: dispatch\n", label);
        stream << shader(matrix, x_buf, y_buf).dispatch(1u) << synchronize();
        std::printf("stage2[%s]: download\n", label);
        stream << y_buf.copy_to(luisa::span{y}) << synchronize();
        std::printf("stage2[%s]: done\n", label);
        float ea = 0.0f, eb = 0.0f, ez = 0.0f;
        for (auto i = 0u; i < m_dim; ++i) {
            ea = std::max(ea, std::fabs(y[i] - ref_a[i]));
            eb = std::max(eb, std::fabs(y[i] - ref_b[i]));
            ez = std::max(ez, std::fabs(y[i]));
        }
        std::printf("stage2[%s]: y =", label);
        for (auto i = 0u; i < m_dim; ++i) { std::printf(" %.6g", static_cast<double>(y[i])); }
        std::printf("\nstage2[%s]: max|y-refA|=%.9g max|y-refB|=%.9g max|y|=%.9g\n",
                    label, static_cast<double>(ea), static_cast<double>(eb),
                    static_cast<double>(ez));
        return std::array<float, 3>{ea, eb, ez};
    };

    // Variant 1: one invocation (what the in-tree unit tests dispatch).
    auto one = run(1u, "1 invocation");
    // Variant 2: a full warp, one writer (the shape a real kernel would use).
    auto warp = run(32u, "warp of 32");
    report.err_layout_a = std::min(one[0], warp[0]);
    report.err_layout_b = std::min(one[1], warp[1]);
    report.err_zero = std::min(one[2], warp[2]);
    report.layout_a = one[0] <= tol || warp[0] <= tol;
    report.layout_b = one[1] <= tol || warp[1] <= tol;

    check(report.layout_a || report.layout_b,
          "coopvec stage2: cooperative_mat_mul reproduces a CPU reference layout",
          luisa::format("errA={} errB={} (tol={})", report.err_layout_a,
                        report.err_layout_b, tol));
    report.outcome = (report.layout_a || report.layout_b) ? Outcome::passed
                                                          : Outcome::wrong_result;
    return report;
}

} // namespace

int main(int argc, char **argv) {
    unbuffered_output();
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <backend>   (e.g. vk, dx, cuda)\n", argv[0]);
        return 2;
    }
    const std::string_view backend{argv[1]};
    Context context{argv[0]};
    Device device;
    try {
        device = context.create_device(backend.data());
    } catch (...) {
        std::fprintf(stderr, "cannot create a '%s' device\n", backend.data());
        return 2;
    }
    if (!device) {
        std::fprintf(stderr, "cannot create a '%s' device\n", backend.data());
        return 2;
    }
    Stream stream = device.create_stream();

    // ------------------------------------------------------------------
    // Stage 0: the gate. Everything below is skipped -- not silently, with the
    // reason printed -- when the device did not negotiate the extension or the
    // driver has no configuration the codegen can emit.
    // ------------------------------------------------------------------
    stage(0, "capability gate");
    const auto caps = qwen_probe_backend_caps(device);
    qwen_log_backend_caps(caps);
    std::printf("stage0: backend '%s', cooperative_vector enabled=%d usable=%d "
                "fp32_config=%d configs=%u matrix='%s'\n",
                caps.backend_name.c_str(), caps.cooperative_vector ? 1 : 0,
                caps.cooperative_vector_usable ? 1 : 0,
                caps.cooperative_vector_fp32_config ? 1 : 0,
                caps.cooperative_vector_config_count,
                caps.cooperative_matrix_extension.c_str());
    if (luisa::string_view{caps.backend_name} != "vk") {
        std::printf("SKIP: stage 0: backend '%s' has no cooperative-vector path "
                    "(VK_NV_cooperative_vector is Vulkan-only; dx lowers it to scalar "
                    "emulation and cuda has no AST route). Nothing attempted.\n",
                    caps.backend_name.c_str());
        return 0;
    }
    if (!caps.cooperative_vector) {
        std::printf("SKIP: stage 0: VK_NV_cooperative_vector is not enabled on this "
                    "device (see the caps notes for which prerequisite is missing). "
                    "Nothing attempted.\n");
        return 0;
    }
    std::printf("stage0: extension is enabled -> the SPIR-V codegen may emit the NV "
                "instructions; probing the store/layout contract next.\n");

    // ------------------------------------------------------------------
    // Stage 1/2 run regardless of caps.cooperative_vector_fp32_config: the whole
    // point is to find out whether the missing FP32 configuration is fatal, and
    // "the driver accepts it anyway" vs "it produces garbage" is a measurement,
    // not an assumption.
    // ------------------------------------------------------------------
    stage(1, "vector store/load layout contract");
    const auto layout = stage1_vector_layout(device, stream);

    stage(2, "cooperative_mat_mul vs CPU reference (FP32, transpose=false)");
    MatMulReport matmul;
    const bool allow_unsafe_matmul =
        argc > 2 && std::string_view{argv[2]} == "--allow-unsafe-matmul";
    if (layout.outcome != Outcome::passed) {
        std::printf("SKIP: stage 2: stage 1 did not establish the store layout, so a "
                    "matrix cannot be prepared for the matmul (exact stage: 1).\n");
    } else if (!caps.cooperative_vector_fp32_config && !allow_unsafe_matmul) {
        // MEASURED failure mode (RTX 4080 SUPER, driver 13050): with no FP32 /
        // transpose=false configuration the driver accepts the SPIR-V and then the
        // process SEGFAULTS when the dispatch executes, which no in-process check
        // can catch. The unsafe path therefore has to be requested explicitly,
        // with the log above as the evidence.
        std::printf("SKIP: stage 2: the device enumerates no FP32 transpose=false "
                    "cooperative-vector configuration (fp32_config=0, %u configs), and "
                    "executing that matmul anyway crashed the process on the "
                    "development box. Re-run with --allow-unsafe-matmul to reproduce.\n",
                    caps.cooperative_vector_config_count);
    } else {
        matmul = stage2_matmul(device, stream);
    }

    // ------------------------------------------------------------------
    // Stage 3 is a DECISION, not a benchmark, until stage 2 is correct: timing a
    // kernel whose output does not match the reference would be a fake pass.
    // ------------------------------------------------------------------
    stage(3, "decision gate (>= 1.5x the existing DSL GEMV with equal results)");
    if (matmul.outcome != Outcome::passed) {
        std::printf("REJECTED: stage 3 not attempted. cooperative_mat_mul does not "
                    "reproduce a CPU reference on this device, so there is nothing "
                    "correct to time. Expected cause: the driver enumerates %u "
                    "cooperative-vector configurations and none of them is the "
                    "FLOAT32/FLOAT32/FLOAT32/FLOAT32 with transpose=false that "
                    "spirv_codegen/instruction.cpp emits (fp32_config=%d).\n",
                    caps.cooperative_vector_config_count,
                    caps.cooperative_vector_fp32_config ? 1 : 0);
        std::printf("VERDICT: NOT ADOPTED. No engine integration was performed and "
                    "none is planned on this evidence.\n");
        return summarize("qwen coopvec spike", argv[1]);
    }
    std::printf("stage3: stage 2 matched a CPU reference (layout_a=%d layout_b=%d); "
                "the representative-shape benchmark is the next step and is NOT "
                "implemented in this revision because the gate above did not open on "
                "the development box.\n",
                matmul.layout_a ? 1 : 0, matmul.layout_b ? 1 : 0);
    std::printf("VERDICT: layout and matmul verified; adoption still requires the "
                ">= 1.5x cold-cache benchmark vs the DSL GEMV (opt_plan/04 section 3 "
                "step 2), which this build does not claim.\n");
    return summarize("qwen coopvec spike", argv[1]);
}
