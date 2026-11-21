// Qwen Tile operator device regression test (tests/integration/qwen).
//
// Usage: test_qwen_ops_tile <backend>      e.g. test_qwen_ops_tile dx
// The backend is EXPLICIT argv[1]; there is no default and no silent skip.
// Requires no checkpoint: every captured Tile kernel is compiled on the real
// backend and run on deterministic synthetic data with ragged shapes, in
// BOTH weight formats (BF16 and FP8 E4M3), against the independent CPU
// oracles in qwen/tile_ops.h. Six external-contract fixtures compare device
// results against INDEPENDENT hand-computed expectations. Exits nonzero on
// any mismatch and prints located, actionable errors (including capture /
// compile diagnostics).
//
// Consolidated from the verified scratch device harness. Backend notes:
//  * Tile FP8 GEMM requires K % 4 == 0 and FP8 embedding hidden % 64 == 0
//    (word-aligned packed rows); the shapes below respect that fail-closed
//    contract while staying ragged w.r.t. tiles and scale groups.
//  * The GDN conv step uses SEPARATE ping-pong ring buffers with whole-buffer
//    descriptors (Vulkan legacy HLSL ABI alignment), exactly like the
//    executor; ring_in is never written in place.
#include "qwen_device_test_utils.h"

#include "qwen/tile_ops.h"

#include <luisa/tile/runtime.h>

#include <algorithm>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::qwen::tile_ops;
using namespace qwen_test;
namespace t = luisa::compute::tile;

namespace {

// Compile + dispatch a captured kernel; report capture/compile diagnostics
// as actionable failures (never silently skip).
template<typename... Bufs>
bool run(Device &device, Stream &stream, TileCapture &cap, const char *name, Bufs &&...bufs) {
    if (!cap.ok()) {
        check(false, name, cap.status.message);
        for (auto &d : cap.diagnostics) { std::printf("  capture diag: %s\n", d.c_str()); }
        return false;
    }
    auto shader = t::compile(device, *cap.kernel);
    if (!shader) {
        check(false, name, luisa::string{"tile::compile failed: "} + shader.metadata().error.c_str());
        return false;
    }
    stream << shader(std::forward<Bufs>(bufs)...).dispatch() << synchronize();
    return true;
}

void test_embedding(Device &dev, Stream &stream) {
    for (auto format : {QwenTileWeightFormat::bf16, QwenTileWeightFormat::fp8_e4m3}) {
        // bf16 hidden ragged; fp8 hidden must be a multiple of 64 (contract).
        const uint32_t H = format == QwenTileWeightFormat::bf16 ? 130 : 128, V = 7;
        luisa::vector<float> w(size_t(V) * H);
        for (size_t i = 0; i < w.size(); i++) { w[i] = std::sin(float(i) * 0.13f) * 0.25f; }
        luisa::vector<uint32_t> words;
        luisa::vector<float> scales;
        check(tile_host_pack_matrix(format, w, H, V, words, scales).ok(), "embedding host pack");
        luisa::vector<int32_t> ids{0, 6, 3, 99, 2};// 99 >= V: guarded zero row
        luisa::vector<float> want;
        check(ref_tile_embedding(format, ids, words, scales, H, V, want).ok(), "embedding oracle");
        auto cap = capture_tile_embedding(format, uint32_t(ids.size()), H, V);
        auto bi = upload(dev, stream, ids);
        auto bw = upload(dev, stream, words);
        auto bs = upload(dev, stream, scales.empty() ? luisa::vector<float>{0.f} : scales);
        auto bo = dev.create_buffer<float>(want.size());
        stream << synchronize();
        if (run(dev, stream, cap, "embedding", bi, bw, bs, bo)) {
            auto got = download(stream, bo, want.size());
            stream << synchronize();
            check_close(got, want, 1e-5f, 1e-5f,
                        format == QwenTileWeightFormat::bf16 ? "tile embedding bf16" : "tile embedding fp8");
        }
    }
}

void test_rmsnorm(Device &dev, Stream &stream) {
    for (auto epi : {TileNormEpilogue::plain, TileNormEpilogue::residual}) {
        const uint32_t R = 5, D = 33;
        luisa::vector<float> x(R * D), w(D), res(R * D);
        for (size_t i = 0; i < x.size(); i++) { x[i] = std::sin(float(i) * 0.7f) * 2.f; }
        for (size_t i = 0; i < w.size(); i++) { w[i] = 1.0f + std::cos(float(i) * 0.3f) * 0.1f; }// pre-added
        for (size_t i = 0; i < res.size(); i++) { res[i] = std::sin(float(i) * 0.11f) * 0.5f; }
        const bool residual = epi == TileNormEpilogue::residual;
        luisa::vector<float> wy, ws;
        check(ref_tile_rmsnorm(x, w, res, residual, R, D, 1e-6f, wy, ws).ok(), "rmsnorm oracle");
        auto cap = capture_tile_rmsnorm(epi, R, D, 1e-6f);
        auto bx = upload(dev, stream, x);
        auto bw = upload(dev, stream, w);
        auto br = upload(dev, stream, residual ? res : luisa::vector<float>{0.f});
        auto by = dev.create_buffer<float>(wy.size());
        auto bs = dev.create_buffer<float>(ws.empty() ? 1 : ws.size());
        stream << synchronize();
        if (run(dev, stream, cap, "rmsnorm", bx, bw, br, by, bs)) {
            auto gy = download(stream, by, wy.size());
            luisa::vector<float> gs;
            if (!ws.empty()) { gs = download(stream, bs, ws.size()); }
            stream << synchronize();
            check_close(gy, wy, 2e-5f, 2e-5f, residual ? "tile rmsnorm residual y" : "tile rmsnorm plain y");
            if (!ws.empty()) { check_close(gs, ws, 1e-6f, 1e-6f, "tile rmsnorm residual sum"); }
        }
    }
}

void test_gemm(Device &dev, Stream &stream) {
    for (auto format : {QwenTileWeightFormat::bf16, QwenTileWeightFormat::fp8_e4m3}) {
        // Ragged M/N; K ragged (fp8 must keep K % 4 == 0 per the packed ABI).
        const uint32_t M = 3, N = 70, K = format == QwenTileWeightFormat::bf16 ? 130 : 132;
        luisa::vector<float> w(size_t(N) * K), x(M * K), res(M * N);
        for (size_t i = 0; i < w.size(); i++) { w[i] = std::cos(float(i) * 0.07f) * 0.25f; }
        for (size_t i = 0; i < x.size(); i++) { x[i] = std::sin(float(i) * 0.13f) * 0.5f; }
        for (size_t i = 0; i < res.size(); i++) { res[i] = float(i) * 0.01f; }
        luisa::vector<uint32_t> words;
        luisa::vector<float> scales;
        check(tile_host_pack_matrix(format, w, K, N, words, scales).ok(), "gemm host pack");
        luisa::vector<float> want;
        check(ref_tile_matmul(format, x, words, scales, res, true, M, N, K, want).ok(), "gemm oracle");
        TileGemmShape gs;
        gs.m = M;
        gs.n = N;
        gs.k = K;
        gs.tile_m = 16;
        gs.tile_n = 16;
        auto cap = capture_tile_gemm(format, TileMatmulEpilogue::residual, gs);
        auto bx = upload(dev, stream, x);
        auto bw = upload(dev, stream, words);
        auto bs = upload(dev, stream, scales.empty() ? luisa::vector<float>{0.f} : scales);
        auto br = upload(dev, stream, res);
        auto by = dev.create_buffer<float>(want.size());
        stream << synchronize();
        if (run(dev, stream, cap, "gemm", bx, bw, bs, br, by)) {
            auto got = download(stream, by, want.size());
            stream << synchronize();
            check_close(got, want, 2e-4f, 2e-4f,
                        format == QwenTileWeightFormat::bf16 ? "tile gemm bf16 residual" : "tile gemm fp8 residual");
        }
        // Decode-shaped GEMV M=1, ragged N=65.
        TileGemmShape gv;
        gv.m = 1;
        gv.n = 65;
        gv.k = 64;
        gv.tile_m = 1;
        gv.tile_n = 16;
        gv.gemv_mma = dev.backend_name() == luisa::string_view{"dx"};
        luisa::vector<float> w2(65 * 64), x2(64);
        for (size_t i = 0; i < w2.size(); i++) { w2[i] = std::sin(float(i) * 0.11f) * 0.3f; }
        for (size_t i = 0; i < x2.size(); i++) { x2[i] = std::cos(float(i) * 0.17f); }
        luisa::vector<uint32_t> words2;
        luisa::vector<float> scales2;
        check(tile_host_pack_matrix(format, w2, 64, 65, words2, scales2).ok(), "gemv host pack");
        luisa::vector<float> want2;
        check(ref_tile_matmul(format, x2, words2, scales2, {}, false, 1, 65, 64, want2).ok(), "gemv oracle");
        auto cap2 = capture_tile_gemm(format, TileMatmulEpilogue::none, gv);
        auto bx2 = upload(dev, stream, x2);
        auto bw2 = upload(dev, stream, words2);
        auto bs2 = upload(dev, stream, scales2.empty() ? luisa::vector<float>{0.f} : scales2);
        auto br2 = dev.create_buffer<float>(1);
        auto by2 = dev.create_buffer<float>(want2.size());
        stream << synchronize();
        if (run(dev, stream, cap2, "gemv", bx2, bw2, bs2, br2, by2)) {
            auto got2 = download(stream, by2, want2.size());
            stream << synchronize();
            check_close(got2, want2, 2e-4f, 2e-4f,
                        format == QwenTileWeightFormat::bf16 ? "tile gemv bf16" : "tile gemv fp8");
        }
        // M=1 K-segment ABI/numerics regression: the GEMV specialization keeps
        // the general five-argument form (dummy residual), unlike M>1 kseg.
        TileGemmShape gseg = gv;
        gseg.k = 128;
        luisa::vector<float> w3(65 * 128), x3(128);
        for (size_t i = 0; i < w3.size(); i++) { w3[i] = std::sin(float(i) * 0.07f) * 0.2f; }
        for (size_t i = 0; i < x3.size(); i++) { x3[i] = std::cos(float(i) * 0.13f); }
        luisa::vector<uint32_t> words3;
        luisa::vector<float> scales3;
        check(tile_host_pack_matrix(format, w3, 128, 65, words3, scales3).ok(), "gemv kseg host pack");
        auto bx3 = upload(dev, stream, x3);
        auto bw3 = upload(dev, stream, words3);
        auto bs3 = upload(dev, stream, scales3.empty() ? luisa::vector<float>{0.f} : scales3);
        auto br3 = dev.create_buffer<float>(1);
        auto by3 = dev.create_buffer<float>(65);
        luisa::vector<float> accumulated(65, 0.0f);
        stream << synchronize();
        for (uint32_t ko : {0u, 64u}) {
            auto cap3 = capture_tile_gemm_kseg(format, gseg, ko, 64u);
            if (run(dev, stream, cap3, "gemv kseg", bx3, bw3, bs3, br3, by3)) {
                auto got3 = download(stream, by3, 65);
                stream << synchronize();
                luisa::vector<float> partial(65, 0.0f);
                for (uint32_t n = 0; n < 65; n++) {
                    double want = 0.0;
                    for (uint32_t kk = 0; kk < 64; kk++) {
                        want += double(x3[ko + kk]) * tile_host_decode_packed(
                                                       format, words3, scales3, 128, size_t(n) * 128 + ko + kk);
                    }
                    partial[n] = float(want);
                    accumulated[n] += float(want);
                }
                check_close(got3, partial, 2e-4f, 2e-4f, "tile gemv kseg partial");
            }
        }
        luisa::vector<float> total;
        check(ref_tile_matmul(format, x3, words3, scales3, {}, false, 1, 65, 128, total).ok(),
              "gemv kseg total oracle");
        check_close(accumulated, total, 3e-4f, 3e-4f, "tile gemv kseg accumulated");
    }
}

void test_swiglu(Device &dev, Stream &stream) {
    luisa::vector<float> g(77), u(77), want;
    for (size_t i = 0; i < g.size(); i++) {
        g[i] = std::sin(float(i) * 0.2f) * 3.f;
        u[i] = std::cos(float(i) * 0.4f);
    }
    check(ref_tile_swiglu(g, u, want).ok(), "swiglu split oracle");
    auto cap = capture_tile_swiglu_split(77);
    auto bg = upload(dev, stream, g);
    auto bu = upload(dev, stream, u);
    auto bo = dev.create_buffer<float>(want.size());
    stream << synchronize();
    if (run(dev, stream, cap, "swiglu split", bg, bu, bo)) {
        auto got = download(stream, bo, want.size());
        stream << synchronize();
        check_close(got, want, 2e-5f, 2e-5f, "tile swiglu split");
    }
    luisa::vector<float> fused(3 * 38), fwant;
    for (size_t i = 0; i < fused.size(); i++) { fused[i] = std::sin(float(i) * 0.17f) * 2.f; }
    check(ref_tile_swiglu_fused(fused, 3, 19, fwant).ok(), "swiglu fused oracle");
    auto capf = capture_tile_swiglu_fused(3, 19);
    auto bf = upload(dev, stream, fused);
    auto bfo = dev.create_buffer<float>(fwant.size());
    stream << synchronize();
    if (run(dev, stream, capf, "swiglu fused", bf, bfo)) {
        auto fgot = download(stream, bfo, fwant.size());
        stream << synchronize();
        check_close(fgot, fwant, 2e-5f, 2e-5f, "tile swiglu fused");
    }
}

void test_utilities(Device &dev, Stream &stream) {
    const uint32_t n = 1003u;// ragged tail
    luisa::vector<float> a(n), b(n);
    for (uint32_t i = 0; i < n; i++) {
        a[i] = std::sin(float(i) * 0.01f);
        b[i] = std::cos(float(i) * 0.013f);
    }
    auto ba = upload(dev, stream, a);
    auto bb = upload(dev, stream, b);
    auto bo = dev.create_buffer<float>(n);
    stream << synchronize();
    auto cap = capture_tile_residual_add(n);
    if (run(dev, stream, cap, "residual add", ba, bb, bo)) {
        auto got = download(stream, bo, n);
        stream << synchronize();
        luisa::vector<float> want(n);
        for (uint32_t i = 0; i < n; i++) { want[i] = a[i] + b[i]; }
        check_close(got, want, 1e-6f, 1e-6f, "tile residual add ragged");
    }
    auto capz = capture_tile_zero_fill(n);
    if (run(dev, stream, capz, "zero fill", bo)) {
        auto got = download(stream, bo, n);
        stream << synchronize();
        luisa::vector<float> want(n, 0.f);
        check_exact(got, want, "tile zero fill ragged");
    }
}

// ---------------------------------------------------------------------------
// Attention LENGTH BUCKETS (phase 9d). The device-visible-length shader streams
// a STATIC key range, so the executor compiles it per ceil(kv_len/bucket)*bucket
// bucket instead of once for the full capacity. This proves, on the device:
//   * tile_attention_bucket_capacity() (the shared policy) at the bucket
//     boundaries 0/1/511/512/513/..., including the capacity clamp;
//   * every bucket specialization matches the independent CPU oracle at its
//     kv_len, so the shorter scan is not a shorter answer;
//   * every bucket specialization is BIT-IDENTICAL to the full-capacity
//     specialization at the same length: the keys the bucket does not visit
//     contribute exactly acc*1 + v*0 in the full-capacity arm, so dropping them
//     cannot move a bit. This is what makes the bucket a pure cost reduction.
//   * the same for a prefill bucket arm (rows > 1, causal diagonal).
// ---------------------------------------------------------------------------
void test_attention_buckets(Device &dev, Stream &stream) {
    // -- policy (device-free) ------------------------------------------------
    {
        const uint32_t CAP = 4096;
        check(tile_attention_bucket_capacity(0, 512, CAP) == CAP, "bucket policy: kv_len 0 -> full capacity");
        check(tile_attention_bucket_capacity(1, 512, CAP) == 512, "bucket policy: kv_len 1 -> 512");
        check(tile_attention_bucket_capacity(511, 512, CAP) == 512, "bucket policy: kv_len 511 -> 512");
        check(tile_attention_bucket_capacity(512, 512, CAP) == 512, "bucket policy: kv_len 512 -> 512");
        check(tile_attention_bucket_capacity(513, 512, CAP) == 1024, "bucket policy: kv_len 513 -> 1024");
        check(tile_attention_bucket_capacity(1024, 512, CAP) == 1024, "bucket policy: kv_len 1024 -> 1024");
        check(tile_attention_bucket_capacity(1025, 512, CAP) == 1536, "bucket policy: kv_len 1025 -> 1536");
        check(tile_attention_bucket_capacity(4095, 512, CAP) == CAP, "bucket policy: kv_len 4095 -> capacity");
        check(tile_attention_bucket_capacity(4096, 512, CAP) == CAP, "bucket policy: kv_len 4096 -> capacity");
        check(tile_attention_bucket_capacity(4000, 0, CAP) == CAP, "bucket policy: bucket 0 disables bucketing");
        // The clamp is the last, partially filled bucket (max_context not a
        // multiple of the bucket) and never goes below kv_len.
        check(tile_attention_bucket_capacity(600, 512, 1000) == 1000, "bucket policy: clamps to capacity");
        check(tile_attention_bucket_capacity(999, 512, 1000) == 1000, "bucket policy: clamp keeps kv_len covered");
        check(tile_attention_bucket_capacity(300, 128, 1000) == 384, "bucket policy: bucket 128");
    }
    QwenTileConfig cfg;
    cfg.q_heads = 2;
    cfg.kv_heads = 1;
    cfg.attn_head_dim = 4;
    cfg.rope_dim = 2;
    const uint32_t QH = cfg.q_heads, KH = cfg.kv_heads, HD = cfg.attn_head_dim;
    const uint32_t CAP = 1536, BUCKET = 512;
    const auto scale = cfg.attn_scale();
    luisa::vector<float> kc(size_t(CAP) * KH * HD), vc(size_t(CAP) * KH * HD);
    for (size_t i = 0; i < kc.size(); i++) {
        kc[i] = std::sin(float(i) * 0.021f) * 0.8f;
        vc[i] = std::cos(float(i) * 0.017f + 0.4f) * 0.6f;
    }
    luisa::vector<float> q1(QH * HD), g1(QH * HD);
    for (size_t i = 0; i < q1.size(); i++) {
        q1[i] = std::sin(float(i) * 0.31f + 1.1f);
        g1[i] = std::cos(float(i) * 0.23f) * 2.0f;
    }
    auto bkc = upload(dev, stream, kc);
    auto bvc = upload(dev, stream, vc);
    auto bq = upload(dev, stream, q1);
    auto bg = upload(dev, stream, g1);
    stream << synchronize();
    // kv lengths at and around every bucket boundary this capacity has.
    const uint32_t lens[] = {1, 511, 512, 513, 1023, 1024, 1025, 1535, 1536};
    auto run_decode = [&](const t::Shader &sh, uint32_t kv_len) {
        auto blen = upload(dev, stream, luisa::vector<int32_t>{int32_t(kv_len)});
        auto bout = dev.create_buffer<float>(QH * HD);
        stream << synchronize();
        stream << sh(bq, bg, bkc, bvc, blen, bout).dispatch() << synchronize();
        return download(stream, bout, QH * HD);
    };
    // Full-capacity arm: the reference every bucket must reproduce bit-exactly.
    auto cap_full = capture_tile_attn_decode_dynamic(cfg, CAP);
    if (!cap_full.ok()) {
        check(false, "attn bucket: full-capacity capture", cap_full.status.message);
    } else {
        auto sh_full = t::compile(dev, *cap_full.kernel);
        if (!sh_full) {
            check(false, "attn bucket: full-capacity compile",
                  luisa::string{sh_full.metadata().error.c_str()});
        } else {
            for (auto kv_len : lens) {
                const auto cap = tile_attention_bucket_capacity(kv_len, BUCKET, CAP);
                const auto tag = luisa::string{"kv_len "} + std::to_string(kv_len).c_str() +
                                 " bucket " + std::to_string(cap).c_str();
                luisa::vector<float> want;
                if (!ref_tile_attn_decode(q1, g1, kc, vc, QH, KH, HD, scale, kv_len, want).ok()) {
                    check(false, "attn bucket: decode oracle", tag);
                    continue;
                }
                auto full = run_decode(sh_full, kv_len);
                check_close(full, want, 3e-5f, 3e-5f,
                            (luisa::string{"attn decode full-capacity vs oracle "} + tag).c_str());
                auto cap_b = capture_tile_attn_decode_dynamic(cfg, cap);
                auto sh_b = t::compile(dev, *cap_b.kernel);
                if (!sh_b) {
                    check(false, "attn bucket: compile", tag);
                    continue;
                }
                auto got = run_decode(sh_b, kv_len);
                check_close(got, want, 3e-5f, 3e-5f,
                            (luisa::string{"attn decode bucket vs oracle "} + tag).c_str());
                check_close(got, full, 0.0f, 0.0f,
                            (luisa::string{"attn decode bucket == full capacity "} + tag).c_str());
            }
        }
    }
    // Prefill bucket arm (rows > 1, causal diagonal inside the bucket).
    {
        const uint32_t T = 3;
        for (auto kv_len : {512u, 513u, 1024u, 1536u}) {
            const uint32_t pos0 = kv_len - T;
            const auto cap = tile_attention_bucket_capacity(kv_len, BUCKET, CAP);
            const auto tag = luisa::string{"prefill kv_len "} + std::to_string(kv_len).c_str() +
                             " bucket " + std::to_string(cap).c_str();
            luisa::vector<float> pq(size_t(T) * QH * HD), pgate(size_t(T) * QH * HD);
            for (size_t i = 0; i < pq.size(); i++) {
                pq[i] = std::sin(float(i) * 0.19f + 0.3f);
                pgate[i] = std::cos(float(i) * 0.11f) * 1.5f;
            }
            luisa::vector<float> want;
            if (!ref_tile_attn_prefill(pq, pgate, kc, vc, T, QH, KH, HD, scale, pos0, kv_len, want).ok()) {
                check(false, "attn prefill bucket oracle", tag);
                continue;
            }
            auto bpq = upload(dev, stream, pq);
            auto bpg = upload(dev, stream, pgate);
            auto bpl = upload(dev, stream, luisa::vector<int32_t>{int32_t(pos0), int32_t(kv_len)});
            stream << synchronize();
            for (auto use_cap : {cap, CAP}) {
                auto pc = capture_tile_attn_prefill_dynamic(cfg, T, use_cap);
                auto sh = t::compile(dev, *pc.kernel);
                if (!sh) {
                    check(false, "attn prefill bucket compile", tag);
                    continue;
                }
                auto bout = dev.create_buffer<float>(want.size());
                stream << sh(bpq, bpg, bkc, bvc, bpl, bout).dispatch() << synchronize();
                auto got = download(stream, bout, want.size());
                stream << synchronize();
                check_close(got, want, 3e-5f, 3e-5f,
                            (luisa::string{use_cap == cap ? "attn prefill bucket vs oracle "
                                                          : "attn prefill full-capacity vs oracle "} +
                             tag)
                                .c_str());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Attention chain: prepare q/k/v -> rope (polynomial + host-table variants)
// -> kv append -> prefill -> decode (static + dynamic-length variants).
// ---------------------------------------------------------------------------
void test_attention(Device &dev, Stream &stream) {
    QwenTileConfig cfg;
    cfg.q_heads = 4;
    cfg.kv_heads = 2;
    cfg.attn_head_dim = 8;
    cfg.rope_dim = 4;
    const uint32_t T = 3, P0 = 2, CAP = 9, KV = 5;
    const auto QN = cfg.attn_q_proj_n(), KVN = cfg.attn_kv_n(), HD = cfg.attn_head_dim;
    luisa::vector<float> qp(T * QN), kp(T * KVN), vp(T * KVN), qw(HD), kw(HD);
    for (size_t i = 0; i < qp.size(); i++) { qp[i] = std::sin(float(i) * 0.19f); }
    for (size_t i = 0; i < kp.size(); i++) { kp[i] = std::cos(float(i) * 0.23f); }
    for (size_t i = 0; i < vp.size(); i++) { vp[i] = std::sin(float(i) * 0.31f + 1.f); }
    for (size_t i = 0; i < qw.size(); i++) {
        qw[i] = 1.0f + 0.05f * std::sin(float(i) * 0.7f);// pre-added GGUF weights
        kw[i] = 1.0f + 0.05f * std::cos(float(i) * 0.9f);
    }
    luisa::vector<float> q, gate, k, v;
    check(ref_tile_attn_prepare(qp, kp, vp, qw, kw, T, cfg, q, gate, k, v).ok(), "attn prepare oracle");
    auto capq = capture_tile_attn_prepare_q(cfg, T);
    auto capk = capture_tile_attn_prepare_k(cfg, T);
    auto capv = capture_tile_attn_prepare_v(cfg, T);
    auto bqp = upload(dev, stream, qp);
    auto bkp = upload(dev, stream, kp);
    auto bvp = upload(dev, stream, vp);
    auto bqw = upload(dev, stream, qw);
    auto bkw = upload(dev, stream, kw);
    auto bq = dev.create_buffer<float>(q.size());
    auto bg = dev.create_buffer<float>(gate.size());
    auto bk = dev.create_buffer<float>(k.size());
    auto bv = dev.create_buffer<float>(v.size());
    stream << synchronize();
    if (run(dev, stream, capq, "attn prepare q", bqp, bqw, bq, bg) &&
        run(dev, stream, capk, "attn prepare k", bkp, bkw, bk) &&
        run(dev, stream, capv, "attn prepare v", bvp, bv)) {
        auto gq = download(stream, bq, q.size());
        auto gg = download(stream, bg, gate.size());
        auto gk = download(stream, bk, k.size());
        auto gv = download(stream, bv, v.size());
        stream << synchronize();
        check_close(gq, q, 2e-5f, 2e-5f, "tile attn prepare q");
        check_close(gg, gate, 1e-6f, 1e-6f, "tile attn prepare gate");
        check_close(gk, k, 2e-5f, 2e-5f, "tile attn prepare k");
        check_close(gv, v, 1e-6f, 1e-6f, "tile attn prepare v");
    }
    // RoPE q and k (device-polynomial variant).
    luisa::vector<int32_t> pos{2, 3, 4};
    luisa::vector<float> inv_freq;
    check(tile_host_rope_inv_freq(cfg.rope_dim, cfg.rope_theta, inv_freq).ok(), "rope inv_freq host");
    luisa::vector<float> qr, kr;
    check(ref_tile_rope_partial(q, pos, T, cfg.q_heads, HD, cfg.rope_dim, cfg.rope_theta, qr).ok(), "rope q oracle");
    check(ref_tile_rope_partial(k, pos, T, cfg.kv_heads, HD, cfg.rope_dim, cfg.rope_theta, kr).ok(), "rope k oracle");
    auto bpos = upload(dev, stream, pos);
    auto bif = upload(dev, stream, inv_freq);
    stream << synchronize();
    auto capr = capture_tile_rope_partial(T, cfg.q_heads, HD, cfg.rope_dim);
    auto bqr = dev.create_buffer<float>(qr.size());
    if (run(dev, stream, capr, "rope q", bq, bpos, bif, bqr)) {
        auto gqr = download(stream, bqr, qr.size());
        stream << synchronize();
        check_close(gqr, qr, 2e-5f, 2e-5f, "tile rope q polynomial");
    }
    auto caprk = capture_tile_rope_partial(T, cfg.kv_heads, HD, cfg.rope_dim);
    auto bkr = dev.create_buffer<float>(kr.size());
    if (run(dev, stream, caprk, "rope k", bk, bpos, bif, bkr)) {
        auto gkr = download(stream, bkr, kr.size());
        stream << synchronize();
        check_close(gkr, kr, 2e-5f, 2e-5f, "tile rope k polynomial");
    }
    // Host-table RoPE (the variant model inference uses).
    {
        const uint32_t MC = 9;
        luisa::vector<float> cos_tab, sin_tab;
        check(tile_host_rope_tables(cfg.rope_dim, cfg.rope_theta, MC, cos_tab, sin_tab).ok(), "rope host tables");
        auto bcos = upload(dev, stream, cos_tab);
        auto bsin = upload(dev, stream, sin_tab);
        stream << synchronize();
        auto captq = capture_tile_rope_partial_table(T, cfg.q_heads, HD, cfg.rope_dim, MC);
        auto bqrt = dev.create_buffer<float>(qr.size());
        if (run(dev, stream, captq, "rope q table", bq, bpos, bcos, bsin, bqrt)) {
            auto gqrt = download(stream, bqrt, qr.size());
            stream << synchronize();
            check_close(gqrt, qr, 2e-5f, 2e-5f, "tile rope q host-table");
        }
        auto captk = capture_tile_rope_partial_table(T, cfg.kv_heads, HD, cfg.rope_dim, MC);
        auto bkrt = dev.create_buffer<float>(kr.size());
        if (run(dev, stream, captk, "rope k table", bk, bpos, bcos, bsin, bkrt)) {
            auto gkrt = download(stream, bkrt, kr.size());
            stream << synchronize();
            check_close(gkrt, kr, 2e-5f, 2e-5f, "tile rope k host-table");
        }
    }
    // KV append into zeroed caches.
    luisa::vector<float> kc(CAP * KVN, 0.f), vc(CAP * KVN, 0.f);
    check(ref_tile_kv_append(kr, v, T, KVN, P0, kc, vc).ok(), "kv append oracle");
    auto capa = capture_tile_kv_append(T, KVN, P0, CAP);
    auto bkc = upload(dev, stream, luisa::vector<float>(kc.size(), 0.f));
    auto bvc = upload(dev, stream, luisa::vector<float>(vc.size(), 0.f));
    stream << synchronize();
    if (run(dev, stream, capa, "kv append", bkr, bv, bkc, bvc)) {
        auto gkc = download(stream, bkc, kc.size());
        auto gvc = download(stream, bvc, vc.size());
        stream << synchronize();
        check_close(gkc, kc, 1e-6f, 1e-6f, "tile kv append k cache");
        check_close(gvc, vc, 1e-6f, 1e-6f, "tile kv append v cache");
    }
    // Prefill.
    luisa::vector<float> pwant;
    check(ref_tile_attn_prefill(qr, gate, kc, vc, T, cfg.q_heads, cfg.kv_heads, HD, cfg.attn_scale(), P0, KV, pwant).ok(),
          "attn prefill oracle");
    auto capp = capture_tile_attn_prefill(cfg, T, P0, KV, CAP);
    auto bpo = dev.create_buffer<float>(pwant.size());
    if (run(dev, stream, capp, "attn prefill", bqr, bg, bkc, bvc, bpo)) {
        auto pgot = download(stream, bpo, pwant.size());
        stream << synchronize();
        check_close(pgot, pwant, 3e-5f, 3e-5f, "tile attn prefill");
    }
    // Decode at the last position (static kv_len).
    luisa::vector<float> q1(qr.end() - cfg.q_heads * HD, qr.end());
    luisa::vector<float> g1(gate.end() - cfg.q_heads * HD, gate.end());
    luisa::vector<float> dwant;
    check(ref_tile_attn_decode(q1, g1, kc, vc, cfg.q_heads, cfg.kv_heads, HD, cfg.attn_scale(), KV, dwant).ok(),
          "attn decode oracle");
    auto capd = capture_tile_attn_decode(cfg, KV, CAP);
    auto bq1 = upload(dev, stream, q1);
    auto bg1 = upload(dev, stream, g1);
    auto bdo = dev.create_buffer<float>(dwant.size());
    stream << synchronize();
    if (run(dev, stream, capd, "attn decode", bq1, bg1, bkc, bvc, bdo)) {
        auto dgot = download(stream, bdo, dwant.size());
        stream << synchronize();
        check_close(dgot, dwant, 3e-5f, 3e-5f, "tile attn decode");
    }
    // Dynamic-length variants: one compiled shader, device-visible lengths.
    {
        auto cappd = capture_tile_attn_prefill_dynamic(cfg, T, CAP);
        luisa::vector<int32_t> pos_len{int32_t(P0), int32_t(KV)};
        auto bpl = upload(dev, stream, pos_len);
        auto bpod = dev.create_buffer<float>(pwant.size());
        stream << synchronize();
        if (run(dev, stream, cappd, "attn prefill dynamic", bqr, bg, bkc, bvc, bpl, bpod)) {
            auto got = download(stream, bpod, pwant.size());
            stream << synchronize();
            check_close(got, pwant, 3e-5f, 3e-5f, "tile attn prefill dynamic");
        }
        auto capdd = capture_tile_attn_decode_dynamic(cfg, CAP);
        auto bkv = upload(dev, stream, luisa::vector<int32_t>{int32_t(KV)});
        auto bdod = dev.create_buffer<float>(dwant.size());
        stream << synchronize();
        if (run(dev, stream, capdd, "attn decode dynamic", bq1, bg1, bkc, bvc, bkv, bdod)) {
            auto got = download(stream, bdod, dwant.size());
            stream << synchronize();
            check_close(got, dwant, 3e-5f, 3e-5f, "tile attn decode dynamic");
        }
    }
    test_attention_buckets(dev, stream);
}

// ---------------------------------------------------------------------------
// GDN chain: ping-pong conv steps -> gates -> q/k L2 norm -> recurrent
// (SSA-carry) + columns variant -> gated norm.
// ---------------------------------------------------------------------------
void test_gdn(Device &dev, Stream &stream) {
    const uint32_t C = 12, KS = 4, T = 3, P0 = 2;
    const uint32_t QK = 2, VH = 4, HD = 4;
    luisa::vector<float> src(T * C), cw(KS * C), ring(C * (KS - 1));
    for (size_t i = 0; i < src.size(); i++) { src[i] = std::sin(float(i) * 0.41f); }
    for (size_t i = 0; i < cw.size(); i++) { cw[i] = std::cos(float(i) * 0.13f) * 0.3f; }
    for (size_t i = 0; i < ring.size(); i++) { ring[i] = std::sin(float(i) * 0.7f + 3.f) * 0.2f; }
    luisa::vector<float> cwant, ring_ref = ring;
    check(ref_tile_gdn_conv(src, cw, ring_ref, T, C, KS, P0, cwant).ok(), "gdn conv oracle");
    // Ping-pong one-token conv steps with separate ring_in/ring_out buffers.
    // Whole per-token buffers (offset 0) keep every storage descriptor aligned
    // on Vulkan's legacy HLSL ABI.
    auto bcw = upload(dev, stream, cw);
    auto bring_a = upload(dev, stream, ring);
    auto bring_b = dev.create_buffer<float>(ring.size());
    auto bpos1 = dev.create_buffer<int32_t>(1);
    luisa::vector<Buffer<float>> tsrc, tout;
    for (uint32_t tt = 0; tt < T; tt++) {
        tsrc.push_back(upload(dev, stream, luisa::vector<float>(src.begin() + tt * C, src.begin() + (tt + 1) * C)));
        tout.push_back(dev.create_buffer<float>(C));
    }
    stream << synchronize();
    luisa::vector<float> cgot(cwant.size()), gring(ring.size());
    bool conv_ok = true;
    for (uint32_t tt = 0; tt < T; tt++) {
        const int32_t pos = int32_t(P0 + tt);
        auto capc = capture_tile_gdn_conv_silu_step(C, KS, uint32_t(pos) % (KS - 1));
        stream << bpos1.copy_from(luisa::span{&pos, 1}) << synchronize();
        auto &rin = (tt % 2 == 0) ? bring_a : bring_b;
        auto &rout = (tt % 2 == 0) ? bring_b : bring_a;
        if (!run(dev, stream, capc, "gdn conv step", tsrc[tt], bcw, rin, bpos1, rout, tout[tt])) {
            conv_ok = false;
            break;
        }
    }
    if (conv_ok) {
        auto &final_ring = (T % 2 == 0) ? bring_a : bring_b;
        for (uint32_t tt = 0; tt < T; tt++) {
            stream << tout[tt].view().copy_to(
                luisa::span<float>{cgot.data() + tt * C, C});
        }
        stream << final_ring.view().copy_to(luisa::span{gring});
        stream << synchronize();
        check_close(cgot, cwant, 2e-5f, 2e-5f, "tile gdn conv ping-pong");
        check_close(gring, ring_ref, 1e-6f, 1e-6f, "tile gdn conv ring update");
    }
    // -----------------------------------------------------------------------
    // Multi-token conv chunk (phase 9d): ONE kernel per chunk carries the
    // causal window in registers and produces the ring image. Differential
    // against (a) the CPU oracle and (b) the per-token step kernel above, at
    // chunk boundaries: tokens < R, tokens == R, tokens > R, pos0 % R != 0, and
    // a non-zero ring history (so a wrong slot rotation cannot pass).
    // -----------------------------------------------------------------------
    {
        const uint32_t R = KS - 1;
        luisa::vector<float> span_src(28 * C);
        for (size_t i = 0; i < span_src.size(); i++) { span_src[i] = std::sin(float(i) * 0.29f + 0.7f); }
        struct ConvCase { uint32_t pos0, rows; };
        const ConvCase conv_cases[] = {
            {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 1}, {1, 2}, {1, 3},
            {2, 3}, {3, 5}, {5, 7}, {6, 2}, {7, 1}, {9, 4}};
        for (const auto &cc : conv_cases) {
            luisa::vector<float> ring0(size_t(C) * R);
            for (size_t i = 0; i < ring0.size(); i++) { ring0[i] = std::sin(float(i) * 0.53f + 1.3f) * 0.15f; }
            auto ring_ref = ring0;
            luisa::vector<float> want;
            if (!ref_tile_gdn_conv(luisa::span{span_src}.subspan(cc.pos0 * C, cc.rows * C), cw,
                                   ring_ref, cc.rows, C, KS, cc.pos0, want)
                     .ok()) {
                check(false, "gdn conv chunk oracle");
                continue;
            }
            luisa::vector<float> rows_src(span_src.begin() + cc.pos0 * C,
                                          span_src.begin() + (cc.pos0 + cc.rows) * C);
            auto bsrc = upload(dev, stream, rows_src);
            auto brin = upload(dev, stream, ring0);
            auto brout = dev.create_buffer<float>(ring0.size());
            auto bposc = upload(dev, stream, luisa::vector<int32_t>{int32_t(cc.pos0)});
            auto bout = dev.create_buffer<float>(want.size());
            stream << synchronize();
            auto cap = capture_tile_gdn_conv_silu_chunk(C, KS, cc.rows);
            const auto tag = luisa::string{"chunk pos0="} + std::to_string(cc.pos0).c_str() +
                             " rows=" + std::to_string(cc.rows).c_str();
            if (run(dev, stream, cap, "gdn conv chunk", bsrc, bcw, brin, bposc, brout, bout)) {
                auto got = download(stream, bout, want.size());
                auto gring = download(stream, brout, ring0.size());
                stream << synchronize();
                check_close(got, want, 2e-5f, 2e-5f, (luisa::string{"tile gdn conv chunk out "} + tag).c_str());
                check_close(gring, ring_ref, 1e-6f, 1e-6f, (luisa::string{"tile gdn conv chunk ring "} + tag).c_str());
            }
        }
        // Kernel size 3 (R = 2) and a ragged channel count: the window is two
        // slots wide, so the initial rotation and the shift have a different
        // arity than the model's Conv4. Same oracle.
        {
            const uint32_t C3 = 7, KS3 = 3, R3 = KS3 - 1;
            luisa::vector<float> w3(C3 * KS3), ring3(size_t(C3) * R3), src3(5 * C3);
            for (size_t i = 0; i < w3.size(); i++) { w3[i] = std::sin(float(i) * 0.17f) * 0.4f; }
            for (size_t i = 0; i < ring3.size(); i++) { ring3[i] = std::cos(float(i) * 0.23f) * 0.3f; }
            for (size_t i = 0; i < src3.size(); i++) { src3[i] = std::sin(float(i) * 0.31f + 2.f); }
            const uint32_t pos0 = 3, rows = 4;
            auto ring_ref3 = ring3;
            luisa::vector<float> want3;
            if (ref_tile_gdn_conv(luisa::span{src3}.subspan(pos0 * C3, rows * C3), w3, ring_ref3,
                                  rows, C3, KS3, pos0, want3)
                    .ok()) {
                auto bw3 = upload(dev, stream, w3);
                auto brin3 = upload(dev, stream, ring3);
                auto brout3 = dev.create_buffer<float>(ring3.size());
                auto bpos3 = upload(dev, stream, luisa::vector<int32_t>{int32_t(pos0)});
                auto bout3 = dev.create_buffer<float>(want3.size());
                luisa::vector<float> rows3(src3.begin() + pos0 * C3, src3.begin() + (pos0 + rows) * C3);
                auto bsrc3 = upload(dev, stream, rows3);
                stream << synchronize();
                auto cap3 = capture_tile_gdn_conv_silu_chunk(C3, KS3, rows);
                if (run(dev, stream, cap3, "gdn conv chunk ks3", bsrc3, bw3, brin3, bpos3, brout3, bout3)) {
                    auto got3 = download(stream, bout3, want3.size());
                    auto gring3 = download(stream, brout3, ring3.size());
                    stream << synchronize();
                    check_close(got3, want3, 2e-5f, 2e-5f, "tile gdn conv chunk out ks3 rows4 pos3");
                    check_close(gring3, ring_ref3, 1e-6f, 1e-6f, "tile gdn conv chunk ring ks3 rows4 pos3");
                }
            } else {
                check(false, "gdn conv chunk ks3 oracle");
            }
        }
        // Kernel-vs-kernel: the same chunk through the per-token step path (its
        // ping-pong rings) and through the chunk kernel must agree to FP32
        // rounding, not bit-exactly: both sum the taps in the same order but the
        // step form materializes (history+source)*weight per tap while the chunk
        // form contracts the register window, so mul/add contraction differs by
        // ~1 ulp. rtol 1e-6 is ~8 ulp of a unit-magnitude accumulator.
        {
            const uint32_t pos0 = 2, rows = 4;
            luisa::vector<float> ring0(size_t(C) * R);
            for (size_t i = 0; i < ring0.size(); i++) { ring0[i] = std::cos(float(i) * 0.37f + 0.2f) * 0.2f; }
            luisa::vector<float> rows_src(span_src.begin() + pos0 * C, span_src.begin() + (pos0 + rows) * C);
            // Chunk arm.
            auto bsrc_c = upload(dev, stream, rows_src);
            auto brin_c = upload(dev, stream, ring0);
            auto brout_c = dev.create_buffer<float>(ring0.size());
            auto bpos_c = upload(dev, stream, luisa::vector<int32_t>{int32_t(pos0)});
            auto bout_c = dev.create_buffer<float>(rows_src.size());
            // Step arm: whole per-token buffers and its own ping-pong pair.
            auto brin_s = upload(dev, stream, ring0);
            auto brout_s = dev.create_buffer<float>(ring0.size());
            luisa::vector<Buffer<float>> st_src, st_out;
            for (uint32_t t = 0; t < rows; t++) {
                st_src.push_back(upload(dev, stream, luisa::vector<float>(
                                                       rows_src.begin() + t * C, rows_src.begin() + (t + 1) * C)));
                st_out.push_back(dev.create_buffer<float>(C));
            }
            auto bpos_s = dev.create_buffer<int32_t>(1);
            stream << synchronize();
            auto cap_chunk = capture_tile_gdn_conv_silu_chunk(C, KS, rows);
            bool ok = run(dev, stream, cap_chunk, "gdn conv chunk (diff arm)",
                          bsrc_c, bcw, brin_c, bpos_c, brout_c, bout_c);
            for (uint32_t t = 0; t < rows && ok; t++) {
                const int32_t pos = int32_t(pos0 + t);
                stream << bpos_s.copy_from(luisa::span{&pos, 1}) << synchronize();
                auto &rin = (t % 2u == 0u) ? brin_s : brout_s;
                auto &rout = (t % 2u == 0u) ? brout_s : brin_s;
                auto cap_step = capture_tile_gdn_conv_silu_step(C, KS, (pos0 + t) % R);
                ok = run(dev, stream, cap_step, "gdn conv step (diff arm)",
                         st_src[t], bcw, rin, bpos_s, rout, st_out[t]);
            }
            if (ok) {
                luisa::vector<float> chunk_got(rows * C), step_got(rows * C);
                luisa::vector<float> chunk_ring(ring0.size()), step_ring(ring0.size());
                stream << bout_c.view().copy_to(luisa::span{chunk_got});
                for (uint32_t t = 0; t < rows; t++) {
                    stream << st_out[t].view().copy_to(luisa::span<float>{step_got.data() + t * C, C});
                }
                stream << brout_c.view().copy_to(luisa::span{chunk_ring});
                // The step arm's newest ring is the buffer the last token wrote.
                auto &step_last = (rows % 2u == 0u) ? brin_s : brout_s;
                stream << step_last.view().copy_to(luisa::span{step_ring});
                stream << synchronize();
                check_close(chunk_got, step_got, 0.0f, 1e-6f, "tile gdn conv chunk == per-token step (out)");
                check_close(chunk_ring, step_ring, 0.0f, 0.0f, "tile gdn conv chunk == per-token step (ring)");
            }
        }
    }
    // Gates.
    luisa::vector<float> a(T * VH), b(T * VH), ssm_a(VH), dt(VH), bwant, gwant;
    for (size_t i = 0; i < a.size(); i++) {
        a[i] = std::sin(float(i) * 0.5f) * 4.f;
        b[i] = std::cos(float(i) * 0.3f) * 2.f;
    }
    for (uint32_t h = 0; h < VH; h++) {
        ssm_a[h] = -0.5f * float(h);// pre-folded GGUF -exp(A_log)
        dt[h] = 0.1f * float(h);
    }
    check(ref_tile_gdn_gates(a, b, ssm_a, dt, T, VH, 20.f, bwant, gwant).ok(), "gdn gates oracle");
    auto capg = capture_tile_gdn_gates(T, VH, 20.f);
    auto ba = upload(dev, stream, a);
    auto bb = upload(dev, stream, b);
    auto bsa = upload(dev, stream, ssm_a);
    auto bdt = upload(dev, stream, dt);
    auto bbo = dev.create_buffer<float>(bwant.size());
    auto bgo = dev.create_buffer<float>(gwant.size());
    stream << synchronize();
    if (run(dev, stream, capg, "gdn gates", ba, bb, bsa, bdt, bbo, bgo)) {
        auto bgot = download(stream, bbo, bwant.size());
        auto ggot = download(stream, bgo, gwant.size());
        stream << synchronize();
        check_close(bgot, bwant, 2e-5f, 2e-5f, "tile gdn gates beta");
        check_close(ggot, gwant, 2e-5f, 2e-5f, "tile gdn gates g");
    }
    // Q/K L2 norm over a packed conv output.
    const uint32_t VC = VH * HD, stride = 2 * QK * HD + VC;
    luisa::vector<float> conv(T * stride), qwant, kwant;
    for (size_t i = 0; i < conv.size(); i++) { conv[i] = std::sin(float(i) * 0.17f); }
    const float qscale = 1.f / std::sqrt(float(HD));
    check(ref_tile_gdn_qk_norm(conv, T, QK, HD, VC, 1e-6f, qscale, qwant, kwant).ok(), "gdn qk oracle");
    auto capqn = capture_tile_gdn_q_l2norm(T, QK, HD, VC, 1e-6f, qscale);
    auto capkn = capture_tile_gdn_k_l2norm(T, QK, HD, VC, 1e-6f);
    auto bconv = upload(dev, stream, conv);
    auto bqo = dev.create_buffer<float>(qwant.size());
    auto bko = dev.create_buffer<float>(kwant.size());
    stream << synchronize();
    if (run(dev, stream, capqn, "gdn q norm", bconv, bqo) &&
        run(dev, stream, capkn, "gdn k norm", bconv, bko)) {
        auto qgot = download(stream, bqo, qwant.size());
        auto kgot = download(stream, bko, kwant.size());
        stream << synchronize();
        check_close(qgot, qwant, 2e-5f, 2e-5f, "tile gdn q l2 norm");
        check_close(kgot, kwant, 2e-5f, 2e-5f, "tile gdn k l2 norm");
    }
    // Recurrent over T tokens (SSA state carry).
    luisa::vector<float> state(VH * HD * HD), rwant, state_ref;
    for (size_t i = 0; i < state.size(); i++) { state[i] = std::sin(float(i) * 0.11f) * 0.05f; }
    state_ref = state;
    check(ref_tile_gdn_recurrent(qwant, kwant, conv, bwant, gwant, state_ref, T, VH, HD, QK, rwant).ok(),
          "gdn recurrent oracle");
    auto capr = capture_tile_gdn_recurrent(T, VH, QK, HD);
    auto bst = upload(dev, stream, state);
    auto bro = dev.create_buffer<float>(rwant.size());
    stream << synchronize();
    if (run(dev, stream, capr, "gdn recurrent", bqo, bko, bconv, bbo, bgo, bst, bro)) {
        auto rgot = download(stream, bro, rwant.size());
        auto gstate = download(stream, bst, state.size());
        stream << synchronize();
        check_close(rgot, rwant, 3e-5f, 3e-5f, "tile gdn recurrent out");
        check_close(gstate, state_ref, 3e-5f, 3e-5f, "tile gdn recurrent state");
    }
    // Column-parallel variant (the executor's DX/VK route): same math, state
    // read+write in place per column.
    {
        auto caprc = capture_tile_gdn_recurrent_columns(T, VH, QK, HD);
        auto bstc = upload(dev, stream, state);
        auto broc = dev.create_buffer<float>(rwant.size());
        stream << synchronize();
        if (run(dev, stream, caprc, "gdn recurrent columns", bqo, bko, bconv, bbo, bgo, bstc, broc)) {
            auto rgot = download(stream, broc, rwant.size());
            auto gstate = download(stream, bstc, state.size());
            stream << synchronize();
            check_close(rgot, rwant, 3e-5f, 3e-5f, "tile gdn recurrent columns out");
            check_close(gstate, state_ref, 3e-5f, 3e-5f, "tile gdn recurrent columns state");
        }
    }
    // Gated norm (direct weight, norm before gate).
    luisa::vector<float> z(T * VH * HD), nw(HD), nwant;
    for (size_t i = 0; i < z.size(); i++) { z[i] = std::cos(float(i) * 0.2f) * 3.f; }
    for (size_t i = 0; i < nw.size(); i++) { nw[i] = 0.5f + 0.1f * float(i); }
    check(ref_tile_gdn_gated_norm(rwant, z, nw, T, VH, HD, 1e-6f, nwant).ok(), "gdn gated norm oracle");
    auto capn = capture_tile_gdn_gated_norm(T, VH, HD, 1e-6f);
    auto bz = upload(dev, stream, z);
    auto bnw = upload(dev, stream, nw);
    auto bno = dev.create_buffer<float>(nwant.size());
    stream << synchronize();
    if (run(dev, stream, capn, "gdn gated norm", bro, bz, bnw, bno)) {
        auto ngot = download(stream, bno, nwant.size());
        stream << synchronize();
        check_close(ngot, nwant, 3e-5f, 3e-5f, "tile gdn gated norm");
    }
}

// ---------------------------------------------------------------------------
// EXTERNAL-CONTRACT DEVICE FIXTURES: hand-computed expectations, independent
// of the shared oracles.
// ---------------------------------------------------------------------------
void test_fixtures(Device &dev, Stream &stream) {
    // (a) pre-added direct rmsnorm: x=[3,4,0,-1], w=[2,0.5,1,1.5].
    {
        const luisa::vector<float> x{3.f, 4.f, 0.f, -1.f}, w{2.f, 0.5f, 1.f, 1.5f};
        auto cap = capture_tile_rmsnorm(TileNormEpilogue::plain, 1, 4, 1e-6f);
        auto bx = upload(dev, stream, x);
        auto bw = upload(dev, stream, w);
        auto bd = dev.create_buffer<float>(1);
        auto by = dev.create_buffer<float>(4);
        stream << synchronize();
        if (run(dev, stream, cap, "fixture rmsnorm", bx, bw, bd, by, bd)) {
            auto y = download(stream, by, 4);
            stream << synchronize();
            const double inv = 1.0 / std::sqrt(26.0 / 4.0 + 1e-6);
            const luisa::vector<float> expect{float(3 * inv * 2), float(4 * inv * 0.5), 0.f, float(-1 * inv * 1.5)};
            check_close(y, expect, 1e-5f, 1e-6f, "fixture device rmsnorm pre-added direct");
        }
    }
    // (b) negative ssm_a gates: a=[1,30], b=[0,2], ssm_a=[-2,-0.5], dt=[0.25,-1].
    {
        const luisa::vector<float> a{1.f, 30.f}, b{0.f, 2.f}, sa{-2.f, -0.5f}, dt{0.25f, -1.f};
        auto cap = capture_tile_gdn_gates(1, 2, 20.0f);
        auto ba = upload(dev, stream, a);
        auto bb = upload(dev, stream, b);
        auto bsa = upload(dev, stream, sa);
        auto bdt = upload(dev, stream, dt);
        auto bbo = dev.create_buffer<float>(2);
        auto bgo = dev.create_buffer<float>(2);
        stream << synchronize();
        if (run(dev, stream, cap, "fixture gates", ba, bb, bsa, bdt, bbo, bgo)) {
            auto beta = download(stream, bbo, 2);
            auto g = download(stream, bgo, 2);
            stream << synchronize();
            const luisa::vector<float> gw{float(-2.0 * std::log1p(std::exp(1.25))), float(-0.5 * 29.0)};
            const luisa::vector<float> bw{0.5f, float(1.0 / (1.0 + std::exp(-2.0)))};
            check_close(g, gw, 1e-5f, 1e-6f, "fixture device gates negative ssm_a g");
            check_close(beta, bw, 1e-6f, 1e-6f, "fixture device gates negative ssm_a beta");
        }
    }
    // (c) channel-major impulse conv through the ping-pong step kernel:
    // C=2, K=4, w[c*K+j]=c*10+j+1, impulse at t=0 channel 0.
    {
        const uint32_t C = 2, K = 4, T = 4;
        luisa::vector<float> w(C * K), src(T * C, 0.f), ring0(C * (K - 1), 0.f);
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t j = 0; j < K; j++) { w[c * K + j] = float(c * 10 + j + 1); }
        }
        src[0] = 1.f;
        auto bw = upload(dev, stream, w);
        auto ra = upload(dev, stream, ring0);
        auto rb = upload(dev, stream, ring0);
        auto bpos = dev.create_buffer<int32_t>(1);
        luisa::vector<Buffer<float>> tsrc, tout;
        for (uint32_t tt = 0; tt < T; tt++) {
            tsrc.push_back(upload(dev, stream, luisa::vector<float>(src.begin() + tt * C, src.begin() + (tt + 1) * C)));
            tout.push_back(dev.create_buffer<float>(C));
        }
        stream << synchronize();
        bool ok = true;
        for (uint32_t tt = 0; tt < T && ok; tt++) {
            auto cap = capture_tile_gdn_conv_silu_step(C, K, tt % (K - 1));
            const int32_t pos = int32_t(tt);
            stream << bpos.copy_from(luisa::span{&pos, 1}) << synchronize();
            auto &rin = (tt % 2 == 0) ? ra : rb;
            auto &rout = (tt % 2 == 0) ? rb : ra;
            ok = run(dev, stream, cap, "fixture conv step", tsrc[tt], bw, rin, bpos, rout, tout[tt]);
        }
        if (ok) {
            luisa::vector<float> out(T * C), expect(T * C);
            for (uint32_t tt = 0; tt < T; tt++) {
                stream << tout[tt].view().copy_to(
                    luisa::span<float>{out.data() + tt * C, C});
                for (uint32_t c = 0; c < C; c++) {
                    const double acc = (c == 0 && tt < K) ? double(w[K - 1 - tt]) : 0.0;
                    expect[tt * C + c] = float(acc / (1.0 + std::exp(-acc)));
                }
            }
            stream << synchronize();
            check_close(out, expect, 1e-5f, 1e-6f, "fixture device conv channel-major impulse");
        }
    }
    // (d) tiled V-head pairing (columns variant): QK=2, VH=4, HD=32.
    {
        const uint32_t QK = 2, VH = 4, HD = 32;
        luisa::vector<float> q(QK * HD, 0.f), k(QK * HD, 0.f);
        q[0] = 1.f;
        q[HD + 1] = 1.f;
        k[0] = 1.f;
        k[HD + 1] = 1.f;
        const uint32_t C = 2 * QK * HD + VH * HD;
        luisa::vector<float> conv(C, 0.f);
        for (uint32_t vh = 0; vh < VH; vh++) { conv[2 * QK * HD + vh * HD] = float(vh + 1); }
        luisa::vector<float> beta(VH, 1.f), g(VH, 0.f), state(VH * HD * HD, 0.f);
        auto cap = capture_tile_gdn_recurrent_columns(1, VH, QK, HD);
        auto bq = upload(dev, stream, q);
        auto bk = upload(dev, stream, k);
        auto bc = upload(dev, stream, conv);
        auto bb = upload(dev, stream, beta);
        auto bg = upload(dev, stream, g);
        auto bs = upload(dev, stream, state);
        auto bo = dev.create_buffer<float>(VH * HD);
        stream << synchronize();
        if (run(dev, stream, cap, "fixture tiled recurrent", bq, bk, bc, bb, bg, bs, bo)) {
            auto out = download(stream, bo, VH * HD);
            stream << synchronize();
            luisa::vector<float> expect(VH * HD, 0.f);
            for (uint32_t vh = 0; vh < VH; vh++) { expect[vh * HD] = float(vh + 1); }
            check_close(out, expect, 1e-5f, 1e-6f, "fixture device gdn tiled v-head pairing");
        }
    }
    // (e) NeoX rope via the host-table kernel: hd=8, rd=4, theta=1e4, pos=1.
    {
        const luisa::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
        luisa::vector<float> ct, st;
        check(tile_host_rope_tables(4, 10000.0f, 2, ct, st).ok(), "fixture rope tables host");
        const luisa::vector<int32_t> pos{1};
        auto cap = capture_tile_rope_partial_table(1, 1, 8, 4, 2);
        auto bx = upload(dev, stream, x);
        auto bp = upload(dev, stream, pos);
        auto bc = upload(dev, stream, ct);
        auto bs = upload(dev, stream, st);
        auto bo = dev.create_buffer<float>(8);
        stream << synchronize();
        if (run(dev, stream, cap, "fixture rope table", bx, bp, bc, bs, bo)) {
            auto out = download(stream, bo, 8);
            stream << synchronize();
            const double f1 = std::pow(10000.0, -0.5);
            const double c0 = std::cos(1.0), s0 = std::sin(1.0);
            const double c1 = std::cos(f1), s1 = std::sin(f1);
            const luisa::vector<float> expect{float(c0 - 3 * s0), float(2 * c1 - 4 * s1),
                                              float(s0 + 3 * c0), float(2 * s1 + 4 * c1), 5, 6, 7, 8};
            check_close(out, expect, 1e-5f, 1e-6f, "fixture device rope NeoX rotate-half");
        }
    }
    // (f) near-zero L2 rows: q=[1e-4,0...], k=[0...].
    {
        const uint32_t QK = 1, HD = 32, VN = 32, C = 2 * QK * HD + VN;
        luisa::vector<float> conv(C, 0.f);
        conv[0] = 1e-4f;
        const float qs = 1.0f / std::sqrt(32.0f);
        auto capq = capture_tile_gdn_q_l2norm(1, QK, HD, VN, 1e-6f, qs);
        auto capk = capture_tile_gdn_k_l2norm(1, QK, HD, VN, 1e-6f);
        auto bc = upload(dev, stream, conv);
        auto bq = dev.create_buffer<float>(HD);
        auto bk = dev.create_buffer<float>(HD);
        stream << synchronize();
        if (run(dev, stream, capq, "fixture l2 q", bc, bq) && run(dev, stream, capk, "fixture l2 k", bc, bk)) {
            auto q = download(stream, bq, HD);
            auto k = download(stream, bk, HD);
            stream << synchronize();
            luisa::vector<float> qw(HD, 0.f), kw(HD, 0.f);
            qw[0] = float(1e-4 / std::sqrt(1e-8 + 1e-6) * qs);
            check_close(q, qw, 1e-6f, 1e-6f, "fixture device l2norm near-zero q");
            check_exact(k, kw, "fixture device l2norm near-zero k");
        }
    }
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2 || argv[1][0] == '\0') {
        std::fprintf(stderr,
                     "usage: %s <backend>\n"
                     "  backend is explicit (e.g. dx, vk); no default, no weights needed.\n",
                     argv[0]);
        return 2;
    }
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    if (!device) {
        std::fprintf(stderr, "failed to create device for backend '%s' (is the backend built/enabled?)\n", argv[1]);
        return 2;
    }
    auto stream = device.create_stream(StreamTag::COMPUTE);
    std::printf("backend: %s\n", luisa::string(device.backend_name()).c_str());

    test_embedding(device, stream);
    test_rmsnorm(device, stream);
    test_gemm(device, stream);
    test_swiglu(device, stream);
    test_utilities(device, stream);
    test_attention(device, stream);
    test_gdn(device, stream);
    test_fixtures(device, stream);

    return summarize("qwen tile device ops", argv[1]);
}
