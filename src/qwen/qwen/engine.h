#pragma once
// Public text-only, single-sequence Qwen3.5-4B inference engine interface.
// A prefill consumes every prompt token and returns logits for the next token.
// decode consumes one previously selected token and returns the following
// logits. Implementations own request state; weights are borrowed and immutable.
// Engines and their staging/session state are single-threaded objects: calls on
// one engine and its associated GGUF upload/stream must be externally serialized.
// Public execution/reset calls must use the same Stream supplied to create();
// sparse KV residency and ordered teardown intentionally rely on that identity.

#include <cstdint>
#include <memory>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "qwen/benchmark.h"
#include "qwen/caps.h"
#include "qwen/kv_cache_layout.h"// KV dtype/layout contract (allocator + kernels)
#include "qwen/w4_importance.h"
#include "qwen/weights.h"

namespace luisa::qwen {

enum class QwenKernelKind : uint32_t {
    dsl = 0,
    tile = 1,
};
[[nodiscard]] luisa::string_view qwen_kernel_kind_name(QwenKernelKind kind) noexcept;

struct QwenEngineOptions {
    QwenKernelKind kernel{QwenKernelKind::dsl};
    // Logical full-attention KV token capacity, reserved virtually and backed
    // by sparse physical tiles on demand. Must be >0 and <= the model's native
    // context length. GDN state is fixed-size and allocated separately.
    uint32_t max_context{4096};
    // KV sparse ALLOCATION unit in bytes (0 = the allocator's 2 MiB default).
    // One heap is allocated per page, so this trades growth granularity for
    // allocation count: DX/VK cost one D3D12MA/VkDeviceMemory allocation per
    // page and Vulkan's maxMemoryAllocationCount is commonly 4096, so raise this
    // for very long contexts on Vulkan (the allocator warns with the exact value
    // it would need). Must be a multiple of the probed sparse tile size; a
    // non-conforming value fails closed with invalid_argument.
    uint64_t kv_page_bytes{0};
    // Full-attention K/V STORED DTYPE (opt_plan/05-quantization.md section 4 item
    // Q3). `f32` is the legacy dense FP32 cache addressed [position][kv_head][channel]
    // and stays the default until the parent's measurement decides otherwise; `bf16`
    // stores two round-to-nearest-even BF16 codes per 32-bit word (2 B per value,
    // HALF the traffic and half the capacity) in the plan's 64-position blocked
    // layout ((head * position_blocks + position/64) * 64 + position%64) * head_dim +
    // channel. Activations, accumulators and the softmax stay FP32 on both paths, and
    // the page geometry follows the dtype (qwen/kv_cache_layout.h, opt_plan 01
    // section 2.7). There is NO silent fallback: a backend or geometry that cannot
    // host the requested dtype fails create() with StatusCode::unsupported, and
    // QwenEngineInfo reports the dtype/layout that was actually built.
    QwenKvDType kv_dtype{QwenKvDType::f32};
    // Number of prompt rows per engine prefill call. DSL subdivides these rows
    // into serial 64-token WY/UT chunks; this is an activation/launch tradeoff.
    uint32_t prefill_chunk{8};
    // Adaptive outer prefill chunking (opt_plan/02 section 4). When true the
    // engine picks the chunk for each prefill as
    //   chunk = clamp(next_pow2(prompt_rows / 4), 64, prefill_chunk_max)
    // minus any per-backend conservative limit, splits the prompt into that many
    // chunks (the last one ragged) and reports the selection in QwenEngineInfo.
    // `prefill_chunk` is then only the workspace floor/legacy value and is NOT
    // used to size the prompt split. False keeps the exact fixed-chunk behavior
    // every existing caller and the --prefill-chunk override rely on.
    bool prefill_chunk_adaptive{false};
    // Upper bound of the adaptive chunk (configurable per run). 1024 is the
    // plan default: the activation workspace grows linearly with it and 1024
    // rows of [chunk, max_proj] scratch is ~200 MiB after workspace aliasing.
    uint32_t prefill_chunk_max{1024};
    // DSL prefill uses the FP32 WY/UT chunk-64 GDN algorithm by default.
    // Set false for the exact per-token recurrence as a differential/debug
    // path. Decode always uses the recurrence. Tile currently has its own
    // recurrent GDN schedule and does not consult this DSL-specific switch.
    bool dsl_gdn_wy_ut{true};
    // Calibration-only executor mode (opt_plan/05 section 3 item 2). TRUE keeps the
    // REFERENCE (non-fused) M=1 decode route: the fused FFN decode and the fused GDN
    // conv/gate kernels compute their intermediate activations in-kernel, so an
    // activation accumulator cannot see the ffn_down input. With this flag the engine
    // uses the same projections as prefill (each through project()), which is exactly
    // what begin_activation_calibration() needs. It changes NO numerics contract: the
    // reference route is the differential-debug path the tree already ships, and it is
    // slower. A normal run must never set it.
    bool calibration_only{false};
// Fused decode GDN conv4+SiLU+ring update+QK L2 norm (opt_plan/09 N3.3): one
// launch per GDN layer per decode token instead of two, and the Q/K half of the
// conv output never round-trips through global memory. Requires the model
// head geometry (head_dim 128) and dsl_decode_specialized; otherwise the pair of
// reference kernels is used and info().gdn_decode_fused stays false.
bool dsl_gdn_decode_fused{true};
// Vectorized 128-bit decode realizations (opt_plan/09 N2): uint4 weight loads,
// float4 activations, one FP8 scale per uint4 and a branchless LUT-free E4M3
// decode for the M=1 projection GEMV, the fused FFN decode and the fused
// alpha/beta gate GEMV. True REQUESTS the family; create() verifies every
// projection shape and a successful compile, and info().decode_packed records
// which realization actually runs. False keeps the 32-bit kernels.
// MEASURED DEFAULT false: implemented, differentially tested, and REJECTED by the
// end-to-end A/B (BF16 13.91 -> 14.03 ms/token, +0.9 %, 3 order-reversed pairs) and
// by the isolated FP8 arm (gdn_z 529 -> 425 GB/s). Set true to reproduce.
bool dsl_decode_packed{false};
// FP8 escape hatch: force the 128-bit realization for FP8 weights too. It is
// OFF by default because it MEASURED slower than the 32-bit shared-LUT decode
// on every model shape (see docs/examples/example_qwen.md); it exists so the
// rejection stays reproducible end to end (--decode-packed-fp8 1).
bool dsl_decode_packed_fp8{false};
// Model-specialized decode attention prepare (opt_plan/09 N3.2): head_dim 256,
// rope_dim 64, 16 q / 4 kv heads as compile-time constants with a per-lane
// register cache and one RoPE pair per lane. Request only; the geometry and a
// successful compile decide (info().attn_prepare_model).
bool dsl_attn_prepare_model{true};
// Per-prompt GDN prefill selection (opt_plan/09 V4 follow-up). When true the
// engine runs the exact per-token recurrence for prompts <= dsl_gdn_auto_threshold
// and the WY/UT chunk-64 kernel above it, using the measured crossover. An
// explicit --gdn-wy-ut (dsl_gdn_wy_ut) wins and turns this off.
bool dsl_gdn_auto{true};
// Measured crossover: the exact per-token recurrence won at EVERY measured prompt
// (128..8192, 15-17 % faster TTFT, both orders), so the auto rule selects it up to
// the largest MEASURED prompt and keeps the previous WY/UT default above it rather
// than extrapolating.
uint32_t dsl_gdn_auto_threshold{8192u};
// Compile-time GDN head specialization (opt_plan/09 V4): the WY/UT chunk-64
// prefill kernel and the per-token recurrent fallback get head_dim = 128 as a
// compile-time constant, which removes every runtime index of the lane-owned
// state column (local memory -> registers). false selects the generic
// runtime-extent kernels for A/B and differential debugging.
bool dsl_gdn_state_specialized{true};
  // FUSED one-CTA-per-head GDN recurrence + gated output RMSNorm (opt_plan/03
  // section 5, the "GDN recurrent + gated norm" row). The default decode schedule
  // runs the 4-CTA-per-head recurrence (`make_qwen35_gdn_decode4_kernel`) and then a
  // SECOND launch for the gated RMSNorm, because that norm reduces over the head's
  // whole 128 channels and the four blocks of a head are independent -- so the head
  // row round-trips through global `_gdn_out`. With this flag the ONE-CTA-per-head
  // realization owns the whole head in one block, does the same state update and the
  // same recurrent output, and reduces the gated RMSNorm in the SAME block: one
  // launch instead of two and no `_gdn_out` round trip.
  // It is a REQUEST: dsl_decode_specialized (MTP verify/replay, calibration-only and
  // the reference decode keep one reduction order), the model geometry (head_dim
  // 128, T=1) and a successful compile decide, and QwenEngineInfo records the
  // outcome. MEASURED DEFAULT false -- see docs/examples/example_qwen.md and
  // .kimix_cache/qwen_benchmark/phase9e/REPORT.md: the fusion saves one launch and
  // the round trip per GDN layer, but the one-CTA recurrence loses more than that to
  // its 4x smaller grid, so the default schedule stays decode4 + separate norm.
  // Prefill (rows > 1) and every MTP path are untouched either way,
  // (--gdn-fused-recurrent 1 reproduces the measurement).
  bool dsl_gdn_recurrent_norm{false};

// Register-cached one-pass RMSNorm specialization (opt_plan/09 N3.1): with
// hidden = 2560 each thread caches its 10 elements plus the norm weight in
// registers and touches global memory once. false selects the generic wide
// kernel (A/B and differential debugging).
bool dsl_rmsnorm_reg{true};
    // Single-row decode specializations change reduction order. MTP forces this
    // false so verify, accepted-prefix replay, and plain reference decode share
    // the original numerical path. Ordinary optimized decode keeps it true.
    bool dsl_decode_specialized{true};
    // DSL prefill attention: the model path uses the tiled shared-memory
    // FlashAttention-style kernel (compile-time head_dim/q_heads/kv_heads, real
    // K/V staging, no runtime-indexed local accumulator). Set false to run the
    // reference kernel (one thread per (token, q_head), runtime head_dim local
    // accumulator) as the differential/debug path, exactly like dsl_gdn_wy_ut.
    // Only affects the DSL engine and rows > 1 (decode is unchanged).
    bool dsl_tiled_prefill_attn{true};
    // DSL prefill projections (rows > 1) use the weight-reusing warp-per-column
    // GEMM (opt_plan/02 section 3) instead of the one-thread-per-(m,n)
    // reference GEMM. Decode (rows == 1) keeps its GEMV specializations and is
    // never routed here. Set false for the reference GEMM as an A/B/debug path,
    // exactly like dsl_tiled_prefill_attn. `dsl_prefill_gemm_mt_rows` is the
    // compile-time m tile (MT) of that kernel; 16 is the measured default.
    bool dsl_prefill_gemm_mt{true};
    uint32_t dsl_prefill_gemm_mt_rows{16};
    // CUDA TENSOR-CORE prefill GEMM through the native-shader route
    // (`NativeShaderExt` / `NativeShaderLanguage::CUDA_NVRTC`, opt_plan/09 N1a).
    // Independent of dsl_prefill_gemm_mt: this one is a real
    // `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` kernel, hand-written
    // in CUDA C++, JIT-compiled by NVRTC and dispatched as a native command - not
    // a DSL kernel, and not tile::mma (the separate opt-in CUDA 13.5 Tile FP8
    // experiment is below). Eligible only where it is exact and faster:
    //  * backend cuda with a NativeShaderExt instance,
    //  * BF16 or FP8 E4M3 packed weights (BF16 copies the codes as-is, FP8 widens
    //    codes*scale to BF16 on the way into the shared tile; no host conversion
    //    and no repacked weight buffer),
    //  * prefill rows >= qwen_native_gemm_min_rows (16). Decode/MTP rows keep the
    //    GEMV family, and the static CUDA decode graph never contains a native
    //    command (nor does the sparse-map ordering change).
    // Any NVRTC / reflection / geometry failure disables the WHOLE path at
    // create() (fail closed) and the engine keeps the DSL MT/legacy GEMM, so
    // `false` is the A/B arm and the fallback, never a half-applied mode.
    // MEASURED DEFAULT (BF16 and FP8 both promoted): see
    // docs/examples/example_qwen.md. Correctness is a ~1.6e-3 relative-RMS
    // deviation from the FP32-activation reference (the BF16 activation rounding at
    // the shared-tile boundary) and the gate was the prefill-GEMM arm of
    // benchmark_qwen_decode against the DSL MT kernel.
    bool native_tensor_gemm{true};
    // EXPERIMENTAL, STRICTLY OPT-IN: prefill GEMM on the CUDA native Tile IR route
    // with TYPED FP8 weight storage (src/qwen/qwen/cuda_tile_gemm.h: a Tile DSL
    // kernel captured for one static (m, n, k), compiled by NVRTC's Tile IR backend
    // and assembled by `tileiras`, i.e. CUDA 13.5 toolkit on sm_89 here). It is
    // NEVER selected automatically: with this flag false the engine neither probes
    // nor JIT-compiles anything and every prefill projection follows the existing
    // native NVRTC / DSL MT / legacy GEMM chain exactly as before (that measured
    // default stays NVRTC, see docs/examples/example_qwen.md).
    // Why it is not the default: it is CORRECT (differential-tested against the
    // FP32-activation reference and the native kernel, test_qwen_native_gemm) but
    // SLOWER on the measured RTX 4080 SUPER sm_89 part - 64x64 tile ~1.31x the
    // native NVRTC time (geomean over the prefill shapes), 128x64 ~2.6x - and every
    // static (m, n, k, epilogue) shape costs one lazy Tile IR + tileiras compile
    // (~1.0 s each, 12 shapes ~12.5 s of start-up/TTFT), so it exists as a
    // research/capability path, not as a performance path.
    // Scope and gates (all fail closed, none of them a silent no-op):
    //  * prefill only, rows >= 16 (qwen_native_gemm_min_rows) and N >= 256 (the
    //    existing native-eligible column rule); decode, MTP and the static CUDA
    //    decode graph are untouched and keep their GEMV family,
    //  * the PREFILL weight copy must be FP8 E4M3 with the 64-element scale group:
    //    an ordinary dsl-fp8 session and the FP8 second copy of a dual-copy W4
    //    session (plan option (c)) qualify; BF16/Q8_0/W4-backed prefill does not,
    //  * backend cuda, `LUISA_CUDA_TILE_IR=1` exported by the caller, a CUDA 13.4+
    //    toolkit with `tileiras`, and a device/toolkit that compiles typed FP8 Tile
    //    storage (the helper probes it at create instead of guessing an SM number),
    //  * requesting it makes create() FAIL with the helper's diagnostic when any of
    //    that does not hold (never a quiet fallback), and a per-shape dispatch
    //    failure is an engine error for the same reason.
    // N (all model-eligible widths are multiples of 64) is never ragged here; the
    // toolkit rejects PARTIAL typed-FP8 tile loads below sm_90, and the helper
    // reports that capability (cuda_tile_prefill_gemm_ragged_n) instead of assuming
    // it. No activation is quantized: activations stay FP32, the weight codes are
    // widened to BF16 exactly, the BF16 partial dot accumulates in FP32 and the
    // FP32 group scale multiplies the partial AFTER the contraction.
    bool cuda_tile_fp8_prefill{false};
    // Output tile of that kernel, per program. Both must be powers of two in
    // [16, 128] with a tile volume within the native Tile budget (validated);
    // tile_k and the scale group stay fixed at 64 (the FP8 quantization group).
    // 64/64 is the measured default; 128/64 was measured as the slower arm.
    uint32_t cuda_tile_m{64u};
    uint32_t cuda_tile_n{64u};
    // CUDA TENSOR-CORE prefill ATTENTION through the native-shader route
    // (`NativeShaderExt` / `NativeShaderLanguage::CUDA_NVRTC`, opt_plan/09 N1b).
    // A real FlashAttention-2 kernel (`mma.sync` QK^T and P.V, FP32 online softmax,
    // FP32 accumulators, sigmoid gate epilogue), hand-written in CUDA C++, JIT-
    // compiled by NVRTC and dispatched as a native command - not a DSL kernel. It
    // consumes exactly the buffers the DSL tiled kernel consumes (q/gate
    // [chunk,q_heads,hd] FP32, the paged FP32 KV cache, out [chunk,q_heads,hd]), so
    // it is directly differential-testable against
    // `make_qwen35_attn_prefill_tiled_kernel` and ref_qwen_attn_prefill.
    // Eligible only where it is exact and faster:
    //  * backend cuda with a NativeShaderExt instance,
    //  * the model geometry the kernel was generated for (head_dim 256,
    //    16 q heads / 4 kv heads, i.e. the 4:1 GQA pack),
    //  * prefill chunks with rows >= qwen_native_attn_min_rows.
    // Any NVRTC / reflection / geometry failure disables the WHOLE path at create()
    // (fail closed) and prefill attention keeps the DSL tiled kernel, so `false` is
    // the A/B arm and the fallback, never a half-applied mode. Decode attention and
    // the static CUDA decode graph are untouched (a native command is never captured).
    // MEASURED DEFAULT true: correct against the DSL tiled kernel at every prompt
    // length of the differential suite (documented BF16 tolerance, NOT bit equality)
    // and 7.7x-10.6x faster at S = 1024/4096 (see docs/examples/example_qwen.md).
    bool native_tensor_attention{true};
    // CUDA bounded device-greedy decode captures the device-only step ONCE as a
    // STATIC graph and launches it per token with zero graph updates
    // (opt_plan/03 section 2.3): every per-token scalar (absolute position, KV
    // length, greedy history slot and the conv-ring position of all 24 GDN
    // layers) is read from a device state buffer, so one capture serves the
    // whole session. Format-agnostic: BF16 and FP8 both use it. Disabled for
    // benchmark mode, for MTP verify/replay (forced false there) and for the
    // host-logits decode() path; non-CUDA devices keep ordinary command lists.
    // False selects the per-token command-queue A/B path. Capture/instantiate
    // failures are recoverable and never execute a partially consumed list.
    bool dsl_cuda_graph{true};
    // CONDITIONAL CUDA GRAPH WHILE CHUNK (opt_plan/09 N4.2, opt_plan/03 section
    // 2.5 follow-on). When requested, ONE graph launch executes 1..N decode tokens:
    // the argument-invariant device-state step becomes the body of a WHILE node
    // whose condition the body's finish kernel re-arms on the device, so the loop
    // stops on EOS, on an invalid selection, at the context limit or when the token
    // budget `dsl_cuda_graph_while_tokens` is spent -- all without a host round
    // trip. It is OPT-IN and gated twice: CudaGraphExt must report conditional
    // nodes AND the dedicated instantiate/execute/zero-trip probe
    // (qwen/cuda_graph_while.h) must pass on the running driver, because the
    // capability flag is a toolkit gate that says nothing about the device. A
    // refusal (or dsl_cuda_graph == false) keeps the per-token static graph as the
    // safe default, and `decode_device_greedy_chunk` then runs the same chunk as a
    // sequence of per-token steps, reporting `fallback`. The EOS id is MANDATORY:
    // without it a chunk could only stop on its budget, so the request is refused
    // instead of generating past end-of-sequence.
    bool dsl_cuda_graph_while{false};
    // Tokens one WHILE graph launch may execute (>= 1; validated against the
    // remaining context of the launch). Only used when dsl_cuda_graph_while.
    uint32_t dsl_cuda_graph_while_tokens{4u};
    // Model EOS token id, baked into the WHILE body's termination logic. Required
    // by dsl_cuda_graph_while (UINT32_MAX = unset = the request fails closed).
    uint32_t dsl_eos_token_id{UINT32_MAX};
    // PINNED DOWNLOAD STAGING (opt_plan/03 section 6, opt_plan/09 N4.3). True lets
    // the executor allocate a persistent page-locked token ring + logits staging
    // buffer through PinnedMemoryExt and download into them (stream-ordered copy ->
    // synchronize -> memcpy into the caller's span). It changes no result, only
    // where the DMA lands. False is the A/B arm and keeps every download on the
    // pageable/download-pool path. When the extension is absent the engine selects
    // the pageable path regardless and reports why (QwenEngineInfo::pinned_*).
    bool dsl_pinned_transfers{true};
    // PDL reporting request (opt_plan/03 section 2.5). Enabling PDL is a DEVICE
    // CREATION property (CUDADeviceConfigExt::enable_programmatic_stream_serialization,
    // set by the CLI before the device exists), so this flag only tells the engine
    // that the caller asked: it then probes CUDAPdlStreamExt, reports the
    // supported/auto-prologue/launch-stat state in QwenEngineInfo::pdl, and logs why
    // the path is inert when it is. It never changes kernel selection: no kernel of
    // this engine emits grid_dependency_synchronize(), and PDL is a no-op below
    // compute capability 9.0.
    bool dsl_pdl{false};
    // OFFLINE merged input projections (opt_plan/03 section 3 + D5). One decode GEMV
    // per attention/GDN input-projection group instead of one per projection:
    // attention Q/gate+K+V (N = 10240) and GDN QKV+Z+alpha+beta (N = 12352) each
    // become a single launch whose rows are the offline-merged matrix's rows, with
    // segment-aware destinations (no intermediate copy) and, for the GDN gate rows,
    // the SAME alpha/beta epilogue as the fused gate kernel. Per-output FP32
    // accumulation order is unchanged, so decode logits are bit-identical to the
    // unmerged route (differential-tested, see test_qwen_ops_dsl).
    // Only ever engages when the uploaded weights came from a matching offline pack
    // (--projection-pack): without one, every projection is uploaded and launched as
    // before, so this flag alone changes nothing. It is disabled (fail closed, and
    // reported in QwenEngineInfo) for W4 weights (no 4-bit merged realization), for
    // the 128-bit packed decode realization, for MTP/reference decode
    // (dsl_decode_specialized == false, which must keep one reduction order) and for
    // calibration_only. Prefill (rows > 1) keeps its ordinary projections and the
    // native/DSL MT paths untouched. False is the A/B arm on a pack-only weight set:
    // the per-segment views then run the ordinary per-projection GEMVs.
    bool dsl_merged_decode{true};
    // DSL decode attention (opt_plan/03 section 4.2): split-KV with GQA dedup.
    // True routes decode attention at kv_len >= dsl_split_kv_threshold through the
    // one-block-per-(kv_head, key_split) family, which stages K/V once for all 4
    // q_heads of a KV head (1x KV read instead of the per-q_head kernel's 4x) and
    // launches kv_heads * max_splits blocks instead of q_heads. Shorter contexts
    // keep the per-q_head kernel, and a static CUDA graph holds BOTH realizations
    // as statically guarded nodes, so crossing the threshold needs no graph update.
    // False is the reference path (and the differential A/B: --split-kv 0).
    // Forced false when dsl_decode_specialized is false (MTP verify/replay), which
    // must keep one reduction order across verify, replay and plain decode.
    bool dsl_split_kv{true};
    // kv_len at which the split family takes over. Below it the per-q_head kernel
    // runs (16 blocks, 4x KV reads, but no combine launch).
    uint32_t dsl_split_kv_threshold{512};
    // Compile-time grid bound of the split family (blocks = kv_heads * this).
    // 0 = derive it from the device: `2 * sm_count / kv_heads` (the measured 2
    // blocks/SM knee) clamped to [1, qwen_attn_decode_split_cap = 40], or -- when the
    // backend exposes no SM count (DX/VK) -- the documented measured default
    // qwen_attn_decode_split_default_bound = 40. The CUDA backend DOES answer the SM
    // count (patches/luisa-cuda-multiprocessor-query.patch), and on an 80-SM part the
    // derived bound is 40 = the cap, i.e. the same grid as the default.
    // A nonzero value overrides that bound and is gated fail-closed: above the cap (or
    // a config the kernels could not index) the engine keeps the per-q_head kernel
    // instead of compiling an out-of-bounds grid, and the K/V tile stays the one the
    // reported shared-memory limit selected. Raising it adds blocks and shrinks the
    // keys each split scans; the combine then reads more partials.
    uint32_t dsl_split_kv_max_splits{0};
          // Opt-in command-level profiling. False keeps ordinary batched command
      // lists. True uses the supplied stream to commit + synchronize after every
      // kernel/transfer and exposes raw host-wall samples via benchmark_report().
      bool benchmark{false};
      // W4A16 prefill policy (opt_plan/05 section 3). Plan option (c) -- the
      // DEFAULT when the uploaded dual-copy pack provides the FP8 copies -- runs
      // W4 decode and FP8 prefill from a SECOND resident 8-bit weight copy,
      // including an FP8 LM head; the CUDA tensor-core prefill GEMM consumes that
      // FP8 copy through its existing kernel. True here only ENABLES that policy;
      // the copies are what decide it (a single-copy W4 upload is still refused
      // unless dsl_w4_prefill_in_kernel is set).
      bool dsl_dual_prefill{true};
      // Explicit opt-in for the measured A/B arm (plan option (b)): with W4 decode
      // and NO FP8 copy, prefill (and the LM head) run through the W4 tensors with
      // in-kernel dequantization. Never the silent default: without this flag such
      // an upload fails closed at create, and info() records which arm ran.
      bool dsl_w4_prefill_in_kernel{false};
      // Keep the LM head 8-bit in a W4 session (quality: it is the most sensitive
      // matrix). True uses the FP8 partner of token_embd; requires the dual copy.
      bool dsl_w4_lm_head_8bit{true};
  };

// PDL (programmatic dependent launch) STATE, reported so a log states what the
// device/driver actually offers instead of assuming a tier
// (opt_plan/03 section 2.5, opt_plan/04 section 7). `requested` is the engine
// option; `available` is CUDAPdlStreamExt presence; `supported` is
// pdl_supported() (compute capability >= 9.0 AND driver >= 12.0); the three
// launch counters are read from the extension at info() time. `reason` is
// "not available on this device" plus the probe that said so, or the measured
// refusal. No engine kernel emits grid_dependency_synchronize(), so the relaxed /
// strict counters are the backend's own and no speedup is claimed from this path.
struct QwenPdlStatus {
    bool requested{false};
    bool available{false};
    bool supported{false};
    bool auto_prologue{false};
    bool stream_enabled{false};
    uint64_t relaxed_launches{0};
    uint64_t strict_launches{0};
    uint64_t driver_fallbacks{0};
    luisa::string reason;
    [[nodiscard]] luisa::string summary() const noexcept;
};

struct QwenEngineInfo {
          QwenKernelKind kernel{QwenKernelKind::dsl};
      WeightStorage weight_storage{WeightStorage::bf16_packed};
      // Format of the PREFILL/emit copy (plan option (c)): equal to
      // weight_storage in every single-copy session; fp8_packed when a W4 decode
      // session runs the second 8-bit copy. `w4_decode` states whether the decode
      // weights are W4A16 at all, and `w4_prefill_in_kernel` whether the explicit
      // option-(b) A/B arm (W4 in-kernel dequantization for prefill) is active.
      WeightStorage prefill_storage{WeightStorage::bf16_packed};
      bool w4_decode{false};
      bool w4_prefill_in_kernel{false};
    uint32_t position{0};
    uint32_t max_context{0};
    // Configured (fixed) prefill chunk, i.e. the workspace row capacity.
    uint32_t prefill_chunk{0};
    // Adaptive chunking state: the configured flag/max, the per-backend limit
    // actually enforced, the chunk the LAST prefill selected, and the number of
    // chunks it split the prompt into. selected/prefill_chunks are session data
    // (0 until a prefill ran) so a benchmark log can attribute the chunking that
    // was really used instead of the one that was requested.
    bool prefill_chunk_adaptive{false};
    uint32_t prefill_chunk_max{0};
    uint32_t prefill_chunk_backend_limit{0};
    uint32_t prefill_chunk_selected{0};
    uint32_t prefill_chunks{0};
    // Configured logical KV payload + fixed GDN state bytes. Sparse alignment
    // padding and time-varying physical residency are allocator properties.
    uint64_t cache_bytes{0};
      // KV storage actually built (never assumed): the dtype/layout pair, the codes
      // per 32-bit word, the addressable size of ONE K or V region in words, the
      // payload bytes WITHOUT the 64-position block padding (the number to quote
      // against a tier capacity model), and the page/residency geometry the
      // allocator derived from that dtype. 0 before create().
      QwenKvDType kv_dtype{QwenKvDType::f32};
      QwenKvLayout kv_layout{QwenKvLayout::position_major};
      uint32_t kv_elements_per_word{0};
      uint64_t kv_region_words{0};
      uint64_t kv_payload_bytes{0};
      uint32_t kv_page_bytes{0};
      uint32_t kv_pages_per_region{0};
      uint32_t kv_heap_slots{0};
      uint32_t kv_min_commit_pages{0};
      // True when the K/V append runs as a separate packed-word kernel (the BF16
      // path: two RNE codes per word cannot be written one code at a time) instead
      // of inline in the attention prepare kernel. It costs one extra launch per
      // prepare call and is reported so a timeline can attribute it.
      bool kv_packed_append{false};
      // fp8_e4m3 only: TRUE when the append runs the ONLINE-SCALE kernel (one CTA
      // per (64-position block, kv_head), which owns the block maximum, the block
      // scale and any rescale of earlier codes). One launch per prepare call, plus
      // the rewrite of a block whose maximum grew; FALSE on the f32/bf16 arms.
      bool kv_scaled_append{false};
      // Dense FP32 scales a scaled cache keeps (2*layer_count*kv_heads*
      // ceil(max_context/64) floats) and that count per K or V region. They live
      // outside the sparse mapping, so a scale read is never unmapped. Both 0 for
      // f32/bf16.
      uint32_t kv_scales_per_region{0};
      uint64_t kv_scale_bytes{0};
    uint64_t workspace_bytes{0};
    // Chunk-scaled part of workspace_bytes (the aliased activation pool plus the
    // three permanent [chunk, hidden] streams) and that pool's per-row float
    // count. Together with workspace_bytes they show how much of the footprint
    // the chunk rule is responsible for (opt_plan/09 V6 + N3.5).
    uint64_t workspace_chunk_bytes{0};
    uint64_t workspace_pool_floats_per_row{0};
          uint64_t compiled_shader_count{0};
      // Tile engine only (a DSL session leaves every Tile field at its default).
      // `tile_chunk_conv` states which Conv4+SiLU schedule the engine records
      // (one multi-token kernel per chunk/layer vs one kernel plus three
      // commands per token); `tile_attention_bucket_keys` is the configured
      // length bucket (0 = the dynamic attention streams the full capacity),
      // `tile_attention_buckets` the derived, bounded bucket count,
      // `tile_attention_precompiled` the bucket shaders compiled at create and
      // `tile_attention_lazy_compiles` the ones compiled on the REQUEST path
      // (ragged tails), which is the number to look at when a TTFT number looks
      // like it contains a JIT stall.
      bool tile_chunk_conv{false};
      uint32_t tile_attention_bucket_keys{0};
      uint32_t tile_attention_buckets{0};
      uint32_t tile_attention_precompiled{0};
      uint32_t tile_attention_lazy_compiles{0};
    // Static CUDA decode graph counters. `cuda_graph_steps` counts graph decode
    // LAUNCHES (one capture included) and `cuda_graph_updates` counts whole-graph
    // updates, which the static schedule never performs: an argument-invariant
    // body means the expected reading is steps == decode launches and
    // updates == 0.
      uint64_t cuda_graph_steps{0};
      uint64_t cuda_graph_updates{0};
      // CONDITIONAL WHILE chunk (opt_plan/09 N4.2). `cuda_graph_while` is the
      // SELECTED state of this session (option AND conditional-node capability AND a
      // passed instantiate/execute/zero-trip probe); `cuda_graph_while_probe` records
      // that the probe itself ran and succeeded, so "not requested" stays
      // distinguishable from "requested but refused". `cuda_graph_while_chunk` is the
      // token budget of one launch, `*_launches` the WHILE graph launches of the
      // session, `*_tokens` the tokens those launches ACTUALLY executed (summed from
      // the device readback, so an early EOS exit is visible), and `*_early_exits` the
      // chunks that stopped before their budget. `*_fallbacks` counts chunk calls
      // served by the per-token path because the WHILE was unavailable: those are
      // launches of `cuda_graph_steps`, never of `cuda_graph_while_launches`.
      bool cuda_graph_while{false};
      bool cuda_graph_while_probe{false};
      uint32_t cuda_graph_while_chunk{0};
      uint64_t cuda_graph_while_launches{0};
      uint64_t cuda_graph_while_tokens{0};
      uint64_t cuda_graph_while_early_exits{0};
      uint64_t cuda_graph_while_fallbacks{0};
      luisa::string cuda_graph_while_reason;
      // PINNED download staging (opt_plan/03 section 6, opt_plan/09 N4.3).
      // `pinned_transfers` is the selected state, `pinned_available` the extension
      // lookup (so "no PinnedMemoryExt here" stays distinguishable from "opted out"),
      // `pinned_reason` the exact selection/refusal text, and the counters the number
      // of pinned downloads, the bytes they moved and the host memcpys into the
      // caller's span (which the pinned path does NOT eliminate).
      bool pinned_transfers{false};
      bool pinned_available{false};
      uint32_t pinned_ring_tokens{0};
      uint64_t pinned_downloads{0};
      uint64_t pinned_bytes{0};
      uint64_t pinned_host_copies{0};
      luisa::string pinned_reason;
      // PDL state of the device this engine runs on (see QwenPdlStatus).
      QwenPdlStatus pdl{};

    bool gdn_wy_ut{false};
    // True when the GDN state-column kernels (WY/UT chunk-64 prefill and the
    // per-token recurrent fallback) were compiled with the COMPILE-TIME 128-channel
    // head specialization (opt_plan/09 V4) instead of the generic runtime-extent
    // realization. geometry.gdn_head_dim decides it once, at create.
    // True when the zero-centered RMSNorm runs the register-cached one-pass
 // specialization (compile-time hidden = 2560, block 256; opt_plan/09 N3.1).
 bool rmsnorm_reg{false};
 bool gdn_state_specialized{false};
// True when the decode GDN conv+SiLU and QK L2 norm run as ONE fused launch
// (opt_plan/09 N3.3). False keeps the reference two-launch pair (--gdn-decode-fused 0).
bool gdn_decode_fused{false};
  // True when the decode GDN recurrence AND its gated output RMSNorm run as ONE
  // fused one-CTA-per-head launch (opt_plan/03 section 5), so the head row never
  // round-trips through `_gdn_out` and no second launch is recorded per GDN layer.
  // False keeps the default schedule: the 4-CTA-per-head recurrence plus the
  // separate gated norm kernel. `gdn_recurrent_norm_reason` is the selection or
  // refusal text of the session, so "not requested" stays distinguishable from
  // "requested but refused".
  bool gdn_recurrent_norm{false};
  luisa::string gdn_recurrent_norm_reason;
    // True when the M=1 decode GEMV, the fused FFN decode and the fused alpha/beta
    // gate GEMV run the VECTORIZED 128-bit realization (uint4 weight loads, float4
    // activations, one FP8 scale per uint4, branchless LUT-free E4M3 decode;
    // opt_plan/09 N2). False keeps the 32-bit realizations (--decode-packed 0).
      bool decode_packed{false};
      bool decode_packed_fp8{false};
      // ---- OFFLINE merged input projections (opt_plan/03 section 3 + D5) ------
      // `merged_decode` is the SELECTION actually in force for rows == 1 (see
      // QwenEngineOptions::dsl_merged_decode); `merged_available` states whether the
      // uploaded weight set carried an offline pack at all, so "no pack" stays
      // distinguishable from "pack present but the merged route was refused".
      // `merged_matrices`/`merged_segments` are the descriptor counts, `*_n` the
      // merged row widths (12352 GDN / 10240 attention at the frozen geometry) and
      // `merged_launches_saved` the per-decode-token launch reduction the selection
      // buys on THIS weight set: 2 per GDN layer (QKV+z+gate 3 -> 1) and 2 per
      // attention layer (q+k+v 3 -> 1), i.e. 64 at the frozen 24 GDN + 8 attention
      // geometry. Prefill is untouched, so this is a decode-only number.
      bool merged_decode{false};
      bool merged_available{false};
      uint32_t merged_gdn_n{0};
      uint32_t merged_attn_n{0};
      uint32_t merged_launches_saved{0};
      uint64_t merged_matrices{0};
      uint64_t merged_segments{0};
      // Provenance of the pack that was consumed (empty when none was): the source
      // checkpoint name and the pack's format. Fail-closed checks (source size +
      // rolling source digest, per-payload size/hash, geometry, format) are all
      // performed before this record is filled.
      luisa::string projection_pack_source;
      luisa::string projection_pack_format;
    // True when the decode attention prepare (QK norm + partial RoPE + KV append)
    // runs the model-specialized compile-time kernel (head_dim 256, rope_dim 64,
    // 16 q / 4 kv heads; opt_plan/09 N3.2). False keeps the generic kernel
    // (--attn-prepare-model 0).
    bool attn_prepare_model{false};
    // True when the GDN prefill schedule is chosen per prompt from the measured
    // crossover (opt_plan/09 V4 follow-up): the exact per-token recurrence for
    // short prompts, WY/UT chunk-64 above gdn_auto_threshold. False means the
    // fixed dsl_gdn_wy_ut selection is in force (--gdn-auto 0).
    bool gdn_auto{false};
    uint32_t gdn_auto_threshold{0};
    // Which prefill attention realization the DSL engine compiled and will
    // dispatch: true = tiled kernel, false = reference kernel. Always false on
    // the Tile engine (it keeps its own schedule) and whenever the device could
    // not host the tiled kernel (QwenBackendCaps.warp_size != 32, shared budget).
    bool prefill_attn_tiled{false};
    // Tiling actually compiled (0 when the reference kernel is used).
    uint32_t prefill_attn_q_tile{0};
    uint32_t prefill_attn_k_tile{0};
    uint32_t prefill_attn_shared_bytes{0};
    // True when the weight-reusing prefill GEMM (opt_plan/02 section 3) is
    // compiled and used for rows > 1, and its compile-time m tile.
    bool prefill_gemm_mt{false};
    uint32_t prefill_gemm_mt_rows{0};
    // True when the CUDA tensor-core prefill GEMM (NativeShaderExt CUDA_NVRTC:
    // opt_plan/09 N1a) is compiled and takes every projection with
    // rows >= native_prefill_gemm_min_rows. `jit_ms` is the measured COLD NVRTC
    // cost of compiling its two variants (plain + residual) at create time - the
    // honest start-up price of this path, which no disk cache amortizes (native
    // shaders bypass ShaderSerializer); `launches` counts the dispatches of the
    // session so a benchmark can prove the realization engaged instead of
    // assuming it.
    // True when the CUDA tensor-core prefill ATTENTION (NativeShaderExt CUDA_NVRTC
    // FlashAttention-2, opt_plan/09 N1b) is compiled and takes every prefill chunk
    // with rows >= native_prefill_attn_min_rows. `jit_ms` is its measured COLD NVRTC
    // cost at create and `launches` counts the dispatches of the session, so a
    // benchmark can prove the realization engaged instead of assuming it.
    bool native_prefill_attn{false};
    uint32_t native_prefill_attn_q_tile{0};
    uint32_t native_prefill_attn_kv_tile{0};
    uint32_t native_prefill_attn_warps{0};
    uint32_t native_prefill_attn_shared_bytes{0};
    uint32_t native_prefill_attn_min_rows{0};
    uint64_t native_prefill_attn_launches{0};
    double native_prefill_attn_jit_ms{0.0};
    bool native_prefill_gemm{false};
    uint32_t native_prefill_gemm_min_rows{0};
    uint32_t native_prefill_gemm_block_m{0};
    uint32_t native_prefill_gemm_block_n{0};
    uint32_t native_prefill_gemm_block_k{0};
    uint64_t native_prefill_gemm_launches{0};
    double native_prefill_gemm_jit_ms{0.0};
    // EXPERIMENTAL CUDA native-Tile typed-FP8 prefill GEMM (QwenEngineOptions::
    // cuda_tile_fp8_prefill, src/qwen/qwen/cuda_tile_gemm.h). `selected` is true
    // ONLY when the caller opted in AND the helper's create-time gate (backend cuda,
    // LUISA_CUDA_TILE_IR=1, typed-FP8 Tile compile probe) passed: the path is never
    // selected automatically, so with default options every field here stays false/0
    // and a report can never claim the Tile realization engaged. `tile_m/tile_n` are
    // the per-program tile (`tile_k` is the fixed 64-element FP8 group), `min_rows`
    // the prefill row floor (16), `launches` the dispatches of the session,
    // `compiles`/`cache_hits`/`shaders` the LAZY per-static-shape compiles and their
    // cache behaviour, `jit_ms` the sum of those compiles (a real per-shape
    // start-up/TTFT cost, NOT amortized by any disk cache) and `probe_ms` the
    // create-time probe. `ragged_n` is the MEASURED capability to compile a PARTIAL
    // (masked) typed-FP8 weight tile; it is false on sm_89, where the toolkit refuses
    // that construct below sm_90, so a ragged N is refused there instead of assumed.
    bool cuda_tile_prefill_gemm{false};
    uint32_t cuda_tile_prefill_gemm_m{0};
    uint32_t cuda_tile_prefill_gemm_n{0};
    uint32_t cuda_tile_prefill_gemm_k{0};
    uint32_t cuda_tile_prefill_gemm_min_rows{0};
    uint64_t cuda_tile_prefill_gemm_launches{0};
    uint32_t cuda_tile_prefill_gemm_compiles{0};
    uint32_t cuda_tile_prefill_gemm_cache_hits{0};
    uint32_t cuda_tile_prefill_gemm_shaders{0};
    double cuda_tile_prefill_gemm_jit_ms{0.0};
    double cuda_tile_prefill_gemm_probe_ms{0.0};
    bool cuda_tile_prefill_gemm_ragged_n{false};
    // DSL decode attention realization (opt_plan/03 section 4.2). `split` is the
    // configuration: the split-KV + GQA-dedup family is compiled and enabled, so
    // decode attention at kv_len >= threshold reads the KV cache ONCE. `dedup`
    // states whether the step at `kv_len` actually took it -- the distinction the
    // roof model needs (a dedup-capable engine that is still below its threshold,
    // or one whose longest observed context never reached it, must be scored with
    // the legacy 4x term).
    bool decode_attn_split{false};
    bool decode_attn_dedup{false};
    uint32_t decode_attn_threshold{0};
    uint32_t decode_attn_kv_len{0};   // key count of the last/next decode step
    uint32_t decode_attn_splits{0};   // splits that step uses (0 = per-q_head kernel)
    uint32_t decode_attn_blocks{0};   // blocks that step's decode attention launches
    uint32_t decode_attn_max_splits{0};
    uint32_t decode_attn_k_tile{0};
    uint32_t decode_attn_shared_bytes{0};
    uint32_t decode_attn_partial_bytes{0};
    // blocks / caps.sm_count, i.e. the occupancy evidence gate 11 asks for, derived
    // with qwen_decode_attn_blocks_per_sm() (caps.h). 0 when the backend exposes no
    // SM count (DX/VK, see caps.sm_count): the harness must then report "occupancy
    // evidence unavailable" instead of computing a ratio, and a 0 SM count can never
    // become inf/NaN here.
    float decode_attn_blocks_per_sm{0.0f};
    // Device capability record captured at create (backend, warp size, sparse
    // tile, effective KV page geometry, graph/PDL/VK extension negotiation).
    // Carried here so a benchmark log is self-describing: "the accelerator did
    // not help" stays distinguishable from "the accelerator never engaged".
    QwenBackendCaps caps{};
};

// Result of one chunked device-greedy decode call (see
// QwenEngine::decode_device_greedy_chunk). `executed` is measured, never assumed:
// the conditional WHILE path reads the device's iteration budget back, so a chunk
// that stopped on EOS reports the tokens it really produced.
struct QwenChunkResult {
    uint32_t requested{0};  // tokens the caller asked for
    uint32_t executed{0};   // tokens the device actually produced (1..requested)
    bool eos{false};        // the last produced token was the EOS token
    bool context_limit{false};// the loop stopped because position reached max_context
    bool fallback{false};   // served by per-token launches (WHILE unavailable)
    [[nodiscard]] luisa::string summary() const noexcept;
};

class QwenEngine;

// All engine factories allocate through Luisa's allocator. The deleter must
// use the same allocator; EASTL's default_delete otherwise mixes CRT and
// mimalloc heaps in the gallery's static-library configuration.
struct QwenEngineDeleter {
    void operator()(QwenEngine *p) const noexcept;
};
using QwenEnginePtr = luisa::unique_ptr<QwenEngine, QwenEngineDeleter>;

class QwenEngine {
public:
    virtual ~QwenEngine() noexcept = default;
    QwenEngine(const QwenEngine &) = delete;
    QwenEngine &operator=(const QwenEngine &) = delete;

    // Create exactly the requested implementation. No DSL<->Tile fallback and
    // no BF16<->FP8 substitution. `weights` must outlive the returned engine.
    [[nodiscard]] static Result<QwenEnginePtr> create(
        compute::Device &device, compute::Stream &stream,
        const Qwen35Model &model, const QwenDeviceWeights &weights,
        const QwenEngineOptions &options);

    // Clear KV/GDN/conv state and position. The stream is synchronized before
    // returning so subsequent host actions observe a fully reset session.
    [[nodiscard]] virtual Status reset(compute::Stream &stream) = 0;

    // Consume a nonempty prompt and download FP32 logits for the next token.
    // The implementation synchronizes before returning. Input ids are checked
    // against the vocabulary and total context capacity.
    [[nodiscard]] virtual Status prefill(compute::Stream &stream,
                                         luisa::span<const uint32_t> prompt_tokens,
                                         luisa::vector<float> &logits_out) = 0;

    // Consume one selected token after a nonempty prefill/decode sequence and
    // download FP32 logits for the next token. Synchronizes before returning.
    [[nodiscard]] virtual Status decode(compute::Stream &stream, uint32_t input_token,
                                        luisa::vector<float> &logits_out) = 0;

    // Optional greedy-only in-flight decode. The first call passes a host input
    // token; later calls pass nullptr and consume the previous device argmax.
    // Each call records one forward without a host sync and writes the selected
    // token to history_slot. download_device_greedy performs the bounded sync
    // and validates the returned range. Unsupported engines keep the ordinary
    // synchronous logits API; temperature sampling never uses this path.
    [[nodiscard]] virtual bool supports_device_greedy_decode() const noexcept { return false; }
    [[nodiscard]] virtual Status decode_device_greedy(compute::Stream &, const uint32_t *,
                                                      uint32_t) {
        return Status::error(StatusCode::unsupported, "device greedy decode is not supported");
    }
      [[nodiscard]] virtual Status download_device_greedy(compute::Stream &, uint32_t,
                                                          luisa::span<uint32_t>) {
          return Status::error(StatusCode::unsupported, "device greedy download is not supported");
      }
      // OPTIONAL chunked in-flight decode (opt_plan/09 N4.2). Consumes `count`
      // tokens under the same contract as decode_device_greedy (the first input is
      // either a host token or nullptr for "reuse the device argmax") and writes each
      // token to history_slot + i. `result` reports what ACTUALLY ran: an engine that
      // puts the chunk in one conditional WHILE graph launch may stop early on EOS /
      // the context limit, and must then report the real `executed` count (1..count)
      // instead of the request, so a caller can never emit a token that was never
      // generated. An engine without the conditional path serves the same chunk as
      // per-token steps and reports executed == count with `fallback` set, so the
      // tokens are identical either way (only the launch count differs).
      [[nodiscard]] virtual Status decode_device_greedy_chunk(compute::Stream &, const uint32_t *,
                                                              uint32_t, uint32_t,
                                                              QwenChunkResult &) {
          return Status::error(StatusCode::unsupported,
                               "chunked device greedy decode is not supported");
      }


    // ---- calibration-only mode (opt_plan/05 section 3 item 2) ----------------
    // While enabled, the executor records ONE squared-activation accumulation per
    // projection weight into device accumulators it owns, in the SAME command batch
    // as the projection itself (so the accumulation reads exactly the activation
    // the projection consumes, with no ordering assumptions) and counts the
    // activation rows it saw per weight. finish() downloads the sums, divides by
    // the per-tensor token count (m[j] = S[j] / tokens, the input-channel second
    // moment written by qwen/w4_importance.h) and disables the mode.
    // The statistics are NEVER synthesized from the weights: a weight that never
    // saw an activation row reports tokens == 0 and the writer refuses it.
    // The DSL executor implements this; the Tile executor refuses with
    // `unsupported` instead of approximating. Needs
    // QwenEngineOptions::calibration_only at create() time (it keeps the reference,
    // non-fused decode route whose projections the engine can observe).
    [[nodiscard]] virtual Status begin_activation_calibration(compute::Stream &) {
        return Status::error(StatusCode::unsupported,
                             "activation calibration is not implemented by this executor");
    }
    [[nodiscard]] virtual Result<QwenCalibrationReport> finish_activation_calibration(compute::Stream &) {
        return Result<QwenCalibrationReport>::failure(
            StatusCode::unsupported, "activation calibration is not implemented by this executor");
    }

    [[nodiscard]] virtual QwenEngineInfo info() const noexcept = 0;
    // Null when options.benchmark was false. Otherwise the latest
    // prefill/decode report, owned by the engine and valid until the next call.
    [[nodiscard]] virtual const QwenBenchmarkReport *benchmark_report() const noexcept {
        return nullptr;
    }

protected:
    QwenEngine() noexcept = default;
};

// Host greedy selection with deterministic lowest-index tie breaking. Kept
// outside the operators so all four variants share exactly the same sampler.
[[nodiscard]] Result<uint32_t> qwen_argmax(luisa::span<const float> logits) noexcept;

// Reproducible categorical sampling over FP32 logits. temperature == 0 is the
// exact qwen_argmax path; positive finite temperatures apply softmax(logits/T).
// `draw_index` is the absolute generated-token index, not a decode-call count:
// counter-based SplitMix64 randomness therefore stays reproducible across
// prefill chunking and speculative replay. No mutable RNG state is required.
struct QwenSampleOptions {
    float temperature{0.0f};
    uint64_t seed{0x51f15eED1234abcdull};
};
[[nodiscard]] Result<uint32_t> qwen_sample(
    luisa::span<const float> logits, const QwenSampleOptions &options,
    uint64_t draw_index) noexcept;

}// namespace luisa::qwen
