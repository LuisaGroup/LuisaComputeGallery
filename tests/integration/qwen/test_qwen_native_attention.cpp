// Qwen CUDA tensor-core PREFILL ATTENTION device test
// (tests/integration/qwen).
//
// Usage: test_qwen_native_attention <backend> e.g. test_qwen_native_attention cuda
//
// The backend is EXPLICIT argv[1] and must be supplied. This suite covers ONE
// accelerator: the FlashAttention-2 prefill attention kernel that
// `NativeShaderExt`/`NativeShaderLanguage::CUDA_NVRTC` JIT-compiles (opt_plan/09
// N1b). No other backend has that route (dx/vk reject CUDA C++ source), so on any
// other backend - and on a cuda device without the extension or without NVRTC - the
// suite prints a SKIP line and returns 0 instead of failing the DX/VK runs it is
// compiled into. The device-free contract checks run on EVERY backend, exactly like
// test_qwen_native_gemm.
//
// What is checked, all against independent oracles:
//  * the device-free contract: tiling/config validation, the static shared-memory
//    ceiling, the fail-closed input gate (the DSL kernels' causal/capacity
//    contract), the dispatch plan, and the generated CUDA source (tiling baked in,
//    one `__global__`, both `mma.sync` and both `ldmatrix` forms, the sm_80 guard);
//  * THREE-WAY agreement at S = 1, 2, 3, 4, 31, 32, 33, 63, 64, 65, 127, 128, 129,
//    1024 and pos0 = 0 (plus pos0 = 63/64/512 = non-zero history, chunk
//    boundaries): native vs `ref_qwen_attn_prefill` (the double-precision CPU
//    oracle) vs `make_qwen35_attn_prefill_tiled_kernel` (the DSL tiled model path).
//    The metric is normalized by the reference max/RMS, because the kernel rounds
//    Q/K/V and P to BF16 while the references stay FP32 - see the tolerance block
//    below, which prints every measurement instead of hiding it;
//  * ragged tiles: every S above is either not a multiple of the 16-token query tile
//    or of the 32-key tile (only 32/64/128 are, which pins the aligned path too);
//  * cache bounds: kv_len == capacity (full cache), kv_len < capacity (the cache is
//    allocated for a longer context but not filled), and pos0 > 0;
//  * the sigmoid gate epilogue specifically: the same softmax weighted average is
//    scaled elementwise by sigmoid(gate), which is checked as a ratio between two
//    runs that differ only in `gate`;
//  * stream order and lifetime: several dispatches from ONE compiled instance
//    (including a deferred commit), an overwrite of a previously written result, and
//    an explicit `reset()` that destroys the shader before the device.
#include "qwen_device_test_utils.h"

#include "qwen/dsl_ops.h"
#include "qwen/native_attn.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <algorithm>
#include <cmath>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::qwen::dsl;
using namespace qwen_test;

namespace {

constexpr uint32_t kQHeads = 16u;
constexpr uint32_t kKvHeads = 4u;
constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kQStride = kQHeads * kHeadDim;
constexpr uint32_t kKvStride = kKvHeads * kHeadDim;

// Tolerances. The references are FP32 (the CPU oracle accumulates in double); this
// kernel
//  * rounds Q and K to BF16 once (the mma operands),
//  * rounds the softmax weights P = exp(S - m) to BF16 for the P.V mma,
//  * rounds V to BF16 when it stages the shared tile,
//  * keeps the running max/sum, the accumulators and the epilogue in FP32,
//  * evaluates exp() as exp2(x * log2 e) with the hardware EX2 unit.
// Against the BF16-operand oracle that leaves only the FP32 accumulation order and
// the EX2/rounding-flip noise (measured max|d|/max|ref| ~1.3e-3, relative RMS
// ~5e-4); against the FP32 references the four roundings above are added (measured
// ~3e-3 peak, relative RMS ~2e-3 on random-sign data). The gates below are those
// measurements with a ~3x margin, printed on every check so a drift is visible.
constexpr double kRelMaxTol = 1.0e-2;
constexpr double kRelRmsTol = 3.0e-3;

// Cancellation-robust agreement check that always prints its metric.
bool check_agreement(luisa::span<const float> got, luisa::span<const float> want,
                     double rel_max_tol, double rel_rms_tol, const char *name) {
    if (got.size() != want.size()) {
        check(false, name, luisa::string{"size mismatch"});
        return false;
    }
    double max_ref = 0.0, max_diff = 0.0, sum_ref = 0.0, sum_diff = 0.0;
    size_t worst = 0u;
    for (size_t i = 0; i < got.size(); i++) {
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
            check(false, name, luisa::format("index {} is not finite (got {}, want {})",
                                             i, got[i], want[i]));
            return false;
        }
        const double d = double(got[i]) - double(want[i]);
        if (std::fabs(d) > max_diff) {
            max_diff = std::fabs(d);
            worst = i;
        }
        max_ref = std::max(max_ref, std::fabs(double(want[i])));
        sum_diff += d * d;
        sum_ref += double(want[i]) * double(want[i]);
    }
    const double rel_max = max_ref > 0.0 ? max_diff / max_ref : max_diff;
    const double rel_rms = std::sqrt(sum_diff / (sum_ref + 1e-30));
    const bool ok = rel_max <= rel_max_tol && rel_rms <= rel_rms_tol;
    check(ok, name,
          luisa::format("max|d|/max|ref| {:.3e} (tol {:.0e}), rms ratio {:.3e} (tol {:.0e}), "
                        "worst index {} device {:.9g} reference {:.9g}",
                        rel_max, rel_max_tol, rel_rms, rel_rms_tol, worst,
                        got[worst], want[worst]));
    return ok;
}

// ---------------------------------------------------------------------------
// 1. device-free contract (every backend)
// ---------------------------------------------------------------------------
void test_contract() {
    QwenNativeAttnConfig cfg{};
    check(validate_qwen_native_attn_config(cfg).ok(), "native attention default tiling accepted");
    check(cfg.q_tile == 16u && cfg.kv_tile == 32u, "native attention default is a 16-token / 32-key tile");
    check(cfg.gqa() == 4u && cfg.warps() == 8u && cfg.block() == 256u,
          "native attention packs 4 q heads per kv head into 8 warps / 256 threads");
    check(qwen_native_attn_shared_bytes(cfg) == 2u * 32u * 264u * 2u + 4u * 16u * 33u * 4u,
          "native attention shared footprint is the K/V tiles plus the score exchange");
    check(qwen_native_attn_shared_bytes(cfg) <= 48u * 1024u,
          "native attention fits the 48 KiB per-block STATIC shared-memory limit");

    {
        auto bad = cfg;
        bad.head_dim = 128u;
        check(!validate_qwen_native_attn_config(bad).ok(),
              "native attention refuses a head_dim other than 256");
    }
    {
        auto bad = cfg;
        bad.kv_heads = 8u;// 16 / 8 = 2 q heads per kv head
        check(!validate_qwen_native_attn_config(bad).ok(),
              "native attention refuses a GQA ratio other than 4:1");
    }
    {
        auto bad = cfg;
        bad.q_tile = 32u;
        check(!validate_qwen_native_attn_config(bad).ok(),
              "native attention refuses a query tile that is not one m16 fragment");
    }
    {
        auto bad = cfg;
        bad.kv_tile = 64u;
        check(!validate_qwen_native_attn_config(bad).ok(),
              "native attention refuses a 64-key tile (it would need 66 KiB of shared memory)");
    }

    // Input gate: the DSL kernels' causal/capacity contract, fail closed.
    check(validate_qwen_native_attn_inputs(1u, 0u, 1u, 1u, cfg).ok(),
          "native attention accepts a one-token prompt in a one-slot cache");
    check(validate_qwen_native_attn_inputs(1024u, 0u, 1024u, 1024u, cfg).ok(),
          "native attention accepts a full 1024-token cache");
    check(validate_qwen_native_attn_inputs(16u, 512u, 528u, 528u, cfg).ok(),
          "native attention accepts non-zero history (pos0 = 512)");
    check(validate_qwen_native_attn_inputs(40u, 0u, 40u, 512u, cfg).ok(),
          "native attention accepts a kv_len below the cache capacity");
    check(validate_qwen_native_attn_inputs(16u, 0u, 16u, 16u, cfg).ok(),
          "native attention accepts kv_len == capacity");
    check(!validate_qwen_native_attn_inputs(17u, 0u, 16u, 16u, cfg).ok(),
          "native attention refuses a causal window that exceeds kv_len (append first)");
    check(!validate_qwen_native_attn_inputs(8u, 4u, 8u, 8u, cfg).ok(),
          "native attention refuses tokens > kv_len - pos0");
    check(!validate_qwen_native_attn_inputs(16u, 0u, 0u, 0u, cfg).ok(),
          "native attention refuses an empty cache");
    check(!validate_qwen_native_attn_inputs(16u, 0u, 32u, 16u, cfg).ok(),
          "native attention refuses kv_len > capacity");

    {
        auto plan = plan_qwen_native_attn_dispatch(1024u, cfg);
        check(plan.ok(), "native attention plan S=1024");
        check(plan.ok() && plan.value->blocks_x == 64u && plan.value->blocks_y == 4u &&
                  plan.value->threads_x == 64u * 256u && plan.value->blocks == 256u,
              "native attention plan tiles S by 16 tokens and dispatches one block per kv head");
    }
    {
        auto plan = plan_qwen_native_attn_dispatch(0u, cfg);
        check(plan.ok() && plan.value->empty(), "native attention plan with S = 0 is empty");
    }
    {
        auto plan = plan_qwen_native_attn_dispatch(17u, cfg);
        check(plan.ok() && plan.value->blocks_x == 2u && plan.value->blocks == 8u,
              "native attention plan covers a ragged 17-token prompt with two token tiles");
    }
    // The measured selection rule: below 32 rows the DSL tiled kernel is faster.
    check(qwen_native_attn_min_rows == 32u, "native attention row floor is the measured 32");
    check(!qwen_native_attn_prefers_rows(16u) && !qwen_native_attn_prefers_rows(31u),
          "native attention leaves prompts below 32 rows to the DSL tiled kernel (measured)");
    check(qwen_native_attn_prefers_rows(32u) && qwen_native_attn_prefers_rows(4096u),
          "native attention takes prefill rows >= 32");

    // Generated source: the tiling is baked in and both gemms are present.
    {
        auto source = qwen_native_attn_cuda_source(cfg);
        const auto count = [&](luisa::string_view needle) {
            size_t n = 0u;
            for (auto pos = source.find(needle); pos != luisa::string::npos;
                 pos = source.find(needle, pos + needle.size())) {
                n++;
            }
            return n;
        };
        check(count("__global__") == 1u, "generated source declares exactly one __global__ kernel");
        check(count("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ") == 1u,
              "generated source issues m16n8k16 bf16 mma (one helper for both gemms)");
        check(count("ldmatrix.sync.aligned.m8n8.x2.shared.b16") == 1u &&
                  count("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16") == 1u,
              "generated source reads K and V with the row / transposed ldmatrix forms");
        check(count("ex2.approx.f32") == 1u && count("qn_sigmoid") >= 2u,
              "generated source evaluates exp2 and the sigmoid gate on the hardware EX2 unit");
        check(source.find("#define QN_QT 16\n") != luisa::string::npos &&
                  source.find("#define QN_KT 32\n") != luisa::string::npos &&
                  source.find("#define QN_HD 256\n") != luisa::string::npos &&
                  source.find("#define QN_BLOCK 256\n") != luisa::string::npos &&
                  source.find("#define QN_GQA 4\n") != luisa::string::npos,
              "generated source carries the configured tiling and model geometry");
        check(source.find("@") == luisa::string::npos, "generated source has no unsubstituted token");
        check(source.find("__CUDA_ARCH__ < 800") != luisa::string::npos,
              "generated source fails closed below sm_80");
        // The online softmax must be FP32 and the score sentinel finite.
        check(source.find("-1.0e30f") != luisa::string::npos,
              "generated source masks with a finite sentinel (no inf*0 in the online softmax)");
        check(source.find("const float scale") != luisa::string::npos &&
                  source.find("capacity") != luisa::string::npos,
              "generated source keeps tokens/scale/pos0/kv_len/capacity runtime scalars");
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
struct Case {
    uint32_t tokens;
    uint32_t pos0;
    uint32_t capacity;
    uint32_t kv_len;
    const char *label;
};

struct Buffers {
    Buffer<float> q, gate, k, v, out;
    size_t qn{0u};
};

Buffers make_buffers(Device &device, Stream &stream, const Case &c, Rng &rng,
                     luisa::vector<float> &q_host, luisa::vector<float> &gate_host,
                     luisa::vector<float> &k_host, luisa::vector<float> &v_host) {
    Buffers b;
    b.qn = size_t(c.tokens) * kQStride;
    const size_t kn = size_t(c.capacity) * kKvStride;
    q_host.resize(b.qn);
    gate_host.resize(b.qn);
    k_host.resize(kn);
    v_host.resize(kn);
    rng.fill(q_host, -1.f, 1.f);
    rng.fill(gate_host, -2.f, 2.f);
    rng.fill(k_host, -1.f, 1.f);
    rng.fill(v_host, -1.f, 1.f);
    b.q = upload(device, stream, q_host);
    b.gate = upload(device, stream, gate_host);
    b.k = upload(device, stream, k_host);
    b.v = upload(device, stream, v_host);
    b.out = device.create_buffer<float>(b.qn);
    return b;
}

// One native dispatch; returns false (with a check) when the command could not be
// built, which is the only failure mode left after create() validated the geometry.
bool dispatch_native(Stream &stream, const QwenNativeAttn &attn, const Buffers &b,
                     const Case &c, const char *name) {
    const auto scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    auto command = attn.build_dispatch(b.q.view(), b.gate.view(), b.k.view(), b.v.view(),
                                       b.out.view(), c.tokens, scale, c.pos0, c.kv_len,
                                       c.capacity);
    if (command == nullptr) {
        check(false, name, "build_dispatch returned no command");
        return false;
    }
    stream << std::move(command) << synchronize();
    return true;
}

// The DSL tiled model-path kernel on the same buffers (the fallback path).
luisa::vector<float> run_dsl_tiled(Device &device, Stream &stream, const Buffers &b,
                                   const Case &c, const luisa::vector<float> &q_host,
                                   const luisa::vector<float> &gate_host,
                                   const luisa::vector<float> &k_host,
                                   const luisa::vector<float> &v_host) {
    QwenAttnPrefillTiledConfig cfg{};
    if (auto s = select_qwen35_attn_prefill_tiled_config(32u, 0u, cfg); !s.ok()) { return {}; }
    auto shader = device.compile(make_qwen35_attn_prefill_tiled_kernel(cfg));
    if (!shader.compile_ok()) { return {}; }
    auto plan = plan_qwen35_attn_prefill_tiled_dispatch(c.tokens, cfg);
    if (!plan.ok()) { return {}; }
    auto bq = upload(device, stream, q_host);
    auto bg = upload(device, stream, gate_host);
    auto bk = upload(device, stream, k_host);
    auto bv = upload(device, stream, v_host);
    auto bo = device.create_buffer<float>(size_t(c.tokens) * kQStride);
    // fp8 KV takes two per-(head,64-block) scale buffers; this FP32 geometry test
    // binds a 1-element dummy and the kernel never reads it.
    auto bscale_dummy = device.create_buffer<float>(1);
    const auto scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    stream << shader(bq.view(), bg.view(), bk.view(), bv.view(), bscale_dummy.view(),
                     bscale_dummy.view(), bo.view(), c.tokens, scale,
                     c.pos0, c.kv_len, c.capacity)
                  .dispatch(qwen_dispatch_threads(*plan.value))
           << synchronize();
    return download(stream, bo, size_t(c.tokens) * kQStride);
}

// ---------------------------------------------------------------------------
// 2. three-way differential: native vs the CPU oracle vs the DSL tiled kernel
// ---------------------------------------------------------------------------
void test_differential(QwenNativeAttn &attn, Device &device, Stream &stream, Rng &rng) {
    const auto scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    // Prompt lengths of opt_plan/02 section 5 item 1 (ragged vs the 16-token query
    // tile and the 32-key tile both) with pos0 = 0 (full cache, kv_len == capacity).
    const uint32_t lengths[] = {1u, 2u, 3u, 4u, 31u, 32u, 33u, 63u, 64u, 65u,
                                127u, 128u, 129u, 1024u};
    for (auto tokens : lengths) {
        Case c{tokens, 0u, tokens, tokens, "pos0=0"};
        luisa::vector<float> q, gate, k, v;
        auto bufs = make_buffers(device, stream, c, rng, q, gate, k, v);
        luisa::string name = luisa::format("native attention S={} {}", c.tokens, c.label);
        if (!dispatch_native(stream, attn, bufs, c, name.c_str())) { continue; }
        const auto got = download(stream, bufs.out, bufs.qn);
        // Oracle 1: the double-precision CPU reference of the model math.
        luisa::vector<float> cpu_ref;
        if (auto s = ref_qwen_attn_prefill(q, gate, k, v, c.tokens, kQHeads, kKvHeads, kHeadDim,
                                           scale, c.pos0, c.kv_len, cpu_ref);
            !s.ok()) {
            check(false, (name + " cpu oracle").c_str(), luisa::format("{}: {}", s.message, s.detail));
            continue;
        }
        check_agreement(got, cpu_ref, kRelMaxTol, kRelRmsTol, (name + " vs cpu oracle").c_str());
        // Oracle 2: the DSL tiled kernel the engine falls back to.
        const auto dsl = run_dsl_tiled(device, stream, bufs, c, q, gate, k, v);
        if (dsl.empty()) {
            check(false, (name + " dsl tiled").c_str(), "the DSL tiled kernel did not run");
        } else {
            check_agreement(got, dsl, kRelMaxTol, kRelRmsTol, (name + " vs dsl tiled").c_str());
        }
    }
    // Non-zero history and a cache that is longer than the valid prefix: pos0 at and
    // around the KV-tile boundary (32) and far beyond it.
    const Case history[] = {
        {1u, 63u, 64u, 64u, "pos0=63 full cache"},
        {8u, 63u, 128u, 71u, "pos0=63 ragged"},
        {16u, 64u, 96u, 80u, "pos0=64 ragged kv_len<capacity"},
        {17u, 32u, 64u, 49u, "pos0=32 ragged tokens"},
        {33u, 64u, 256u, 97u, "pos0=64 kv_len<capacity"},
        {16u, 512u, 528u, 528u, "pos0=512 full 528-slot cache"},
        {40u, 0u, 512u, 40u, "kv_len 40 in a 512-slot cache"},
    };
    for (const auto &c : history) {
        luisa::vector<float> q, gate, k, v;
        auto bufs = make_buffers(device, stream, c, rng, q, gate, k, v);
        luisa::string name = luisa::format("native attention S={} {}", c.tokens, c.label);
        if (!dispatch_native(stream, attn, bufs, c, name.c_str())) { continue; }
        const auto got = download(stream, bufs.out, bufs.qn);
        luisa::vector<float> cpu_ref;
        if (auto s = ref_qwen_attn_prefill(q, gate, k, v, c.tokens, kQHeads, kKvHeads, kHeadDim,
                                           scale, c.pos0, c.kv_len, cpu_ref);
            !s.ok()) {
            check(false, (name + " cpu oracle").c_str(), luisa::format("{}: {}", s.message, s.detail));
            continue;
        }
        check_agreement(got, cpu_ref, kRelMaxTol, kRelRmsTol, (name + " vs cpu oracle").c_str());
        const auto dsl = run_dsl_tiled(device, stream, bufs, c, q, gate, k, v);
        if (!dsl.empty()) {
            check_agreement(got, dsl, kRelMaxTol, kRelRmsTol, (name + " vs dsl tiled").c_str());
        }
    }
}

// ---------------------------------------------------------------------------
// 3. the sigmoid gate epilogue, in isolation
// ---------------------------------------------------------------------------
void test_gate(QwenNativeAttn &attn, Device &device, Stream &stream, Rng &rng) {
    // out = (softmax(q.K^T scale) . V) * sigmoid(gate). Two runs that differ ONLY in
    // `gate` must therefore have the elementwise ratio sigmoid(g1)/sigmoid(g0).
    Case c{40u, 0u, 64u, 40u, "gate"};
    luisa::vector<float> q, gate, k, v;
    auto bufs = make_buffers(device, stream, c, rng, q, gate, k, v);
    const auto scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    luisa::vector<float> gate_a(bufs.qn, 0.0f);// sigmoid(0) = 0.5 exactly
    luisa::vector<float> gate_b(bufs.qn, 1.5f);// sigmoid(1.5)
    auto run_with = [&](const luisa::vector<float> &g) {
        auto bg = upload(device, stream, g);
        auto bo = device.create_buffer<float>(bufs.qn);
        auto command = attn.build_dispatch(bufs.q.view(), bg.view(), bufs.k.view(), bufs.v.view(),
                                          bo.view(), c.tokens, scale, c.pos0, c.kv_len, c.capacity);
        if (command == nullptr) {
            check(false, "native attention gate dispatch", "build_dispatch returned no command");
            return luisa::vector<float>{};
        }
        stream << std::move(command) << synchronize();
        return download(stream, bo, bufs.qn);
    };
    const auto out_a = run_with(gate_a);
    const auto out_b = run_with(gate_b);
    if (out_a.empty() || out_b.empty()) { return; }
    const double ratio = (1.0 / (1.0 + std::exp(-1.5))) / 0.5;
    double worst = 0.0;
    bool finite = true;
    size_t compared = 0u;
    for (size_t i = 0; i < out_a.size(); i++) {
        if (!std::isfinite(out_a[i]) || !std::isfinite(out_b[i])) {
            finite = false;
            break;
        }
        // The ratio is only meaningful where the gated average is not ~0.
        if (std::fabs(out_a[i]) < 1e-4f) { continue; }
        compared++;
        worst = std::max(worst, std::fabs(double(out_b[i]) / double(out_a[i]) - ratio));
    }
    const auto gate_name = luisa::format(
        "native attention applies the sigmoid output gate elementwise ({} of {} elements compared)",
        compared, out_a.size());
    check(finite && compared > out_a.size() / 2u && worst <= 1e-3, gate_name.c_str(),
          luisa::format("worst |ratio - sigmoid(1.5)/sigmoid(0)| = {:.3e}", worst));

}

// ---------------------------------------------------------------------------
// 4. sequencing, overwrite, deferred commit and lifetime
// ---------------------------------------------------------------------------
void test_sequence_and_lifetime(QwenNativeAttn &attn, Device &device, Stream &stream, Rng &rng) {
    const auto scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    Case c1{33u, 0u, 64u, 33u, "seq1"};
    Case c2{17u, 32u, 96u, 49u, "seq2"};
    luisa::vector<float> q1, g1, k1, v1, q2, g2, k2, v2;
    auto b1 = make_buffers(device, stream, c1, rng, q1, g1, k1, v1);
    auto b2 = make_buffers(device, stream, c2, rng, q2, g2, k2, v2);
    {
        // Deferred commit: both commands are built and held, then submitted in one
        // list, so the launcher's packed argument buffers must outlive the build call.
        auto cmd1 = attn.build_dispatch(b1.q.view(), b1.gate.view(), b1.k.view(), b1.v.view(),
                                       b1.out.view(), c1.tokens, scale, c1.pos0, c1.kv_len,
                                       c1.capacity);
        auto cmd2 = attn.build_dispatch(b2.q.view(), b2.gate.view(), b2.k.view(), b2.v.view(),
                                       b2.out.view(), c2.tokens, scale, c2.pos0, c2.kv_len,
                                       c2.capacity);
        check(cmd1 != nullptr && cmd2 != nullptr,
              "native attention builds two commands up front");
        auto list = CommandList::create(2u);
        if (cmd1 != nullptr) { list << std::move(cmd1); }
        if (cmd2 != nullptr) { list << std::move(cmd2); }
        stream << list.commit() << synchronize();
    }
    const auto got1 = download(stream, b1.out, b1.qn);
    const auto got2 = download(stream, b2.out, b2.qn);
    luisa::vector<float> ref1, ref2;
    if (ref_qwen_attn_prefill(q1, g1, k1, v1, c1.tokens, kQHeads, kKvHeads, kHeadDim, scale,
                             c1.pos0, c1.kv_len, ref1).ok()) {
        check_agreement(got1, ref1, kRelMaxTol, kRelRmsTol, "native attention sequence arm 1");
    }
    if (ref_qwen_attn_prefill(q2, g2, k2, v2, c2.tokens, kQHeads, kKvHeads, kHeadDim, scale,
                             c2.pos0, c2.kv_len, ref2).ok()) {
        check_agreement(got2, ref2, kRelMaxTol, kRelRmsTol, "native attention sequence arm 2");
    }

    // Overwrite: a second dispatch into the same `out` must replace, not accumulate.
    {
        if (dispatch_native(stream, attn, b1, c1, "native attention overwrite")) {
            const auto second = download(stream, b1.out, b1.qn);
            bool differs = false;
            for (size_t i = 0; i < second.size(); i++) { differs = differs || second[i] != got1[i]; }
            check(!differs, "native attention overwrite reproduces the same result (no accumulate)");
        }
    }

    // Back-to-back dispatches from ONE instance (8x) plus a mid-stream reset of the
    // *other* buffers: stream order must hold and no dispatch may leak into another.
    {
        bool all_ok = true;
        for (uint32_t i = 0; i < 8u; i++) {
            all_ok = all_ok && dispatch_native(stream, attn, (i % 2u) == 0u ? b1 : b2,
                                               (i % 2u) == 0u ? c1 : c2,
                                               "native attention repeat dispatch");
        }
        check(all_ok, "native attention survives eight sequential dispatches from one instance");
        const auto last = download(stream, b2.out, b2.qn);
        check_agreement(last, ref2, kRelMaxTol, kRelRmsTol,
                        "native attention eighth dispatch result");
    }
}

}// namespace

int main(int argc, char **argv) {
    unbuffered_output();
    if (argc < 2 || argv[1][0] == '\0') {
        std::fprintf(stderr,
                     "usage: %s <backend>\n"
                     "  backend is explicit (e.g. cuda, dx, vk); the CUDA tensor-core PREFILL\n"
                     "  ATTENTION needs NativeShaderExt/CUDA_NVRTC, so every other backend SKIPs.\n",
                     argv[0]);
        return 2;
    }
    // The device-free contract is checked on every backend.
    test_contract();

    Context ctx{argv[0]};
    auto device = ctx.create_device(argv[1]);
    if (!device) {
        std::fprintf(stderr, "failed to create device for backend '%s' (is the backend built/enabled?)\n",
                     argv[1]);
        return 2;
    }
    auto stream = device.create_stream();
    const auto backend = luisa::string{device.backend_name()};
    std::printf("backend: %s\n", backend.c_str());

    if (backend != "cuda" || device.extension<NativeShaderExt>() == nullptr) {
        std::printf("SKIP: native tensor attention requires the cuda backend with NativeShaderExt "
                    "(CUDA_NVRTC); backend '%s' %s. The DSL tiled prefill attention covers this "
                    "backend and is tested by test_qwen_ops_dsl.\n",
                    backend.c_str(),
                    backend == "cuda" ? "has no NativeShaderExt" : "has no native CUDA route");
        return summarize("qwen native tensor attention", argv[1]);
    }

    QwenNativeAttn attn;
    QwenNativeAttnConfig cfg{};
    if (auto status = attn.create(device, cfg); !status.ok()) {
        std::printf("SKIP: the CUDA tensor-core prefill attention could not be compiled at run time "
                    "(%s%s%s) - no NVRTC or sm_80+ hardware. The engine keeps the DSL tiled kernel, "
                    "which is what this SKIP documents.\n",
                    status.message.c_str(), status.detail.empty() ? "" : ": ",
                    status.detail.c_str());
        return summarize("qwen native tensor attention", argv[1]);
    }
    check(attn.valid(), "native attention compiled through NativeShaderExt/CUDA_NVRTC");
    check(attn.jit_ms() > 0.0, "native attention reports its cold NVRTC cost",
          luisa::format("{:.1f} ms", attn.jit_ms()));
    check(attn.bindings().size() == 5u, "native attention reflects five buffer bindings");
    check(attn.bindings().size() == 5u &&
              attn.bindings()[0].usage == Usage::READ &&
              attn.bindings()[1].usage == Usage::READ &&
              attn.bindings()[2].usage == Usage::READ &&
              attn.bindings()[3].usage == Usage::READ &&
              attn.bindings()[4].usage == Usage::READ_WRITE,
          "native attention binding usages are q/gate/k/v READ and out READ_WRITE");
    check(attn.config().q_tile == 16u && attn.config().kv_tile == 32u &&
              attn.config().warps() == 8u,
          "native attention compiled the 16-token / 32-key / 8-warp tile");
    std::printf("native attention: %ux%u tile, %u warps, %u B static shared, cold JIT %.1f ms\n",
                attn.config().q_tile, attn.config().kv_tile, attn.config().warps(),
                attn.shared_bytes(), attn.jit_ms());

    Rng rng;
    test_differential(attn, device, stream, rng);
    test_gate(attn, device, stream, rng);
    test_sequence_and_lifetime(attn, device, stream, rng);

    // Destroy the native shader explicitly (before the device) and prove the object
    // reports itself invalid afterwards.
    stream << synchronize();
    attn.reset();
    check(!attn.valid(), "native attention reset() destroys the shader instance");
    return summarize("qwen native tensor attention", argv[1]);
}
