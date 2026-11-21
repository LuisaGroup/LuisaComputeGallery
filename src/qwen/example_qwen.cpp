// example_qwen -- command-line front end for the text-only Qwen3.5-4B engine.
//
// Five explicitly selectable variants (no silent fallback):
//   dsl-bf16  dsl-fp8  dsl-q8  tile-bf16  tile-fp8
// BF16 weights are the raw GGUF BF16 payload packed 2-per-uint32; FP8 means
// true E4M3 quantized from the BF16 source AT LOAD TIME (per-row groups of 64
// K elements, FP32 scales) and the conversion is reported. dsl-q8 loads a Q8_0
// checkpoint NATIVELY: the GGUF int8 block codes are copied verbatim (4 per
// uint32) and each block's FP16 scale is widened exactly to FP32 (one FP32
// scale per 32 K elements per row), so nothing is re-quantized and Q8_0 is
// never routed through FP8. Every other variant rejects Q8_0 GGUF files
// explicitly (INT8 block quantization is NOT FP8).
//
// Offline entry points (no device, no model file):
//   example_qwen --list-versions   support matrix + precision semantics
//   example_qwen --self-test       DSL + Tile operator self-tests and the
//                                  tokenizer self-test (device-free)
//   example_qwen --help
//
// Generation (needs a backend plugin and data/Qwen3.5-4B-BF16.gguf):
//   example_qwen --backend cuda --version dsl-bf16 --prompt "Hello" --max-new-tokens 32
// example_qwen cuda --headless --version tile-bf16 --prompt "Hello" --chat
// Greedy (temperature 0) or reproducible temperature-based categorical
// sampling; stops on the GGUF EOS (<|im_end|>). No vision tower. MTP and
// WY/UT prefill support are reported explicitly by the CLI and engine.
//
// Exit codes: 0 success/help/list/self-test; 1 bad arguments; 2 engine or
// model refusal (reported, never fabricated output).
//
// Encoding: Windows narrow argv is the ANSI code page, NOT UTF-8. On Windows
// wmain converts every argument with WideCharToMultiByte(CP_UTF8) so prompts
// and paths reach the tokenizer as UTF-8 (same convention as example_ocr).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <string>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/backends/ext/cuda/cuda_config_ext.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "qwen/dsl_ops.h"
#include "qwen/engine.h"
#include "qwen/gguf.h"
#include "qwen/model.h"
#include "qwen/mtp.h"
#include "qwen/mtp_executor.h"
#include "qwen/prompt_ids.h"
#include "qwen/tile_engine.h"
#include "qwen/tile_ops.h"
#include "qwen/tokenizer.h"
#include "qwen/w4_pack.h"
#include "qwen/weights.h"

#ifdef _WIN32
#include <windows.h>
#endif

using namespace luisa;
using namespace luisa::qwen;

namespace {

// PDL (programmatic dependent launch) creation hook, opt_plan/03 section 2.5.
// CUDADeviceConfigExt is a DeviceConfigExt: the request has to be installed into
// DeviceConfig BEFORE the device exists, because
// enable_programmatic_stream_serialization() makes the backend compile an automatic
// wait-at-entry prologue into every DSL kernel (and into its shader cache key).
// Below compute capability 9.0 the prologue compiles to nothing and the launch
// policy is inert, so this hook is a no-op there -- the engine reports that with the
// CUDAPdlStreamExt probe instead of assuming a tier.
//
// use_cudadevrt() is deliberately NOT overridden (opt_plan/09 N4.4): the conditional
// WHILE body's set_graph_conditional() is a cudadevrt entry point, so an opt-out
// would break the WHILE and device-launch paths. The backend default (true) stays,
// and docs/examples/example_qwen.md documents the 200-400 ms cuLink probe it costs.
class QwenPdlDeviceConfig final : public compute::CUDADeviceConfigExt {
public:
    [[nodiscard]] bool enable_programmatic_stream_serialization() const noexcept override {
        return true;
    }
};

// ---------------------------------------------------------------------------
// CLI surface
// ---------------------------------------------------------------------------

struct VersionSpec {
    luisa::string_view name;
    QwenKernelKind kernel;
    bool fp8;
          // Native Q8_0 checkpoint storage (int8 block codes copied verbatim from the
      // GGUF, no re-quantization). Requires a Q8_0 GGUF; every other variant
      // rejects one explicitly.
      bool q8_0;
      uint32_t default_prefill_chunk;
      luisa::string_view status;
      // W4A16 decode weights read from an offline pack (--w4-pack). Requires the
      // pack AND the BF16 checkpoint the pack was derived from (tokenizer + model
      // contract); never substituted for another format.
      bool w4{false};
  };

// The support matrix. Status strings reflect the verified state recorded by
// the kernel/executor agents (see docs/examples/example_qwen.md); they are
// claims about prior verification runs, not live probes.
constexpr VersionSpec kVersions[] = {
    {"dsl-bf16", QwenKernelKind::dsl, false, false, 64,
     "real-GGUF inference verified on cuda/dx/vk; 1305-check device suite passes on all three"},
    {"dsl-fp8", QwenKernelKind::dsl, true, false, 64,
     "real-GGUF coherent chat verified on cuda; 1305-check device suite passes on cuda/dx/vk"},
    {"dsl-q8", QwenKernelKind::dsl, false, true, 64,
     "native Q8_0 (data/Qwen3.5-4B-Q8_0.gguf): int8 block codes copied verbatim, one exact "
     "FP32 scale per 32 weights, zero re-quantization; real-model inference verified on "
     "cuda/dx/vk; 83-check q8 device oracle suite passes on all three"},
          {"dsl-w4", QwenKernelKind::dsl, false, false, 64,
       "W4A16 (opt_plan/05 Q2): signed int4 codes from an OFFLINE pack (--w4-pack), "
       "one FP32 scale per 128-element group from a deterministic clipping search; "
       "W4 decode + FP8 prefill/LM-head second copy (plan option (c)). EXPERIMENTAL: "
       "the plan's 32-prompt quality gate was RUN on T16 and FAILED (mean prefix "
       "agreement 0.32 vs 0.90 required, first-32 identical 10/32 vs 28/32, KL 0.29 "
       "vs FP8's 0.006), so the 8-bit variants remain the shipping formats; see "
       "docs/examples/example_qwen.md", true},
      {"tile-bf16", QwenKernelKind::tile, false, false, 1,
     "verified on dx; vk 164-check suite and short real-model inference pass; cuda Tile path unavailable"},
    {"tile-fp8", QwenKernelKind::tile, true, false, 1,
     "verified on dx; vk 164-check suite and short real-model inference pass; cuda Tile path unavailable"},
};

struct CliArgs {
    bool help{false};
    bool list_versions{false};
    bool self_test{false};
    bool headless{false};
    bool chat{false};
    bool gdn_recurrent{false};
    bool legacy_prefill_attn{false};
    bool benchmark{false};
    bool cuda_graph{true};
    // Split-KV decode attention with GQA dedup (opt_plan/03 section 4.2).
    // --split-kv 0 selects the reference per-q_head kernel for every context.
    bool split_kv{true};
    uint32_t split_kv_threshold{512};
    uint32_t split_kv_max_splits{0};// 0 = derive from the device/SM bound
    // Fused decode GDN conv+SiLU+QK L2 norm (opt_plan/09 N3.3); 0 selects the
    // reference two-launch pair for A/B and differential debugging.
    bool gdn_decode_fused{true};
    // FUSED one-CTA-per-head GDN recurrence + gated output RMSNorm (opt_plan/03
    // section 5). OPT-IN and measured: 1 reproduces the phase-9e A/B
    // (.kimix_cache/qwen_benchmark/phase9e/REPORT.md). 0 keeps the default
    // decode schedule (4-CTA recurrence + the separate gated norm kernel).
    bool gdn_fused_recurrent{false};
    // Compile-time GDN head specialization (opt_plan/09 V4); 0 = generic kernels.
    bool gdn_state_specialized{true};
    // Register-cached RMSNorm specialization (opt_plan/09 N3.1); 0 = generic.
    bool rmsnorm_reg{true};
    // Vectorized 128-bit decode realizations (opt_plan/09 N2); 0 = 32-bit kernels.
    // MEASURED DEFAULT 0: the 128-bit decode family is implemented and tested but lost
    // the end-to-end A/B; 1 reproduces the measurement.
    bool decode_packed{false};
    bool decode_packed_fp8{false};
    // Model-specialized decode attention prepare (opt_plan/09 N3.2); 0 = generic.
    bool attn_prepare_model{true};
    // Per-prompt GDN prefill selection (opt_plan/09 V4 follow-up). --gdn-wy-ut
    // pins the schedule explicitly and turns auto off; --gdn-auto 0 goes back to
    // the fixed --gdn-wy-ut/--gdn-recurrent choice.
    bool gdn_auto{true};
    uint32_t gdn_auto_threshold{8192};// measured crossover (engine default)
      bool have_gdn_wy_ut{false};
      bool gdn_wy_ut{true};
      // Tile engine (phase 9d) schedule selection. Both default to the engine
      // defaults so an unflagged run and the engine default cannot diverge, and
      // both exist so the two schedules can be A/B'd and differentially tested
      // from the same binary.
      bool tile_chunk_conv{true};
      uint32_t tile_attn_bucket{512};
      bool have_tile_attn_bucket{false};
    bool mtp_test{false};
    bool mtp{false};
    uint32_t draft_depth{2};
    bool have_benchmark_csv{false};
    luisa::string benchmark_csv;
    luisa::string backend;
    luisa::string version{"dsl-bf16"};
    luisa::string model{"data/Qwen3.5-4B-BF16.gguf"};
    luisa::string prompt;
    bool have_prompt{false};
    // Text prompt read from a FILE (UTF-8). An 8 K-token document is ~30 KB of
    // text, which does not fit a Windows command line; the long-context quality
    // gate (scripts/qwen_quality_gate.py) writes the needle document and passes
    // this path, so the real-text set is reproducible from a file alone.
    luisa::string prompt_file;
    bool have_prompt_file{false};
    luisa::vector<uint32_t> tokens;
    bool have_tokens{false};
    // Prompt sources that do not travel on the command line (Windows caps a
    // whole command line at 32767 characters, so an 8192-token --tokens list
    // cannot be launched there; opt_plan/09 N4.1).
    luisa::string tokens_file;
    bool have_tokens_file{false};
    uint32_t synthetic_prompt_len{0};
    bool have_synthetic_prompt{false};
    uint32_t max_new_tokens{32};
    uint32_t decode_flight{4};// greedy CUDA: forwards per bounded host sync; 0 disables
    // CONDITIONAL CUDA GRAPH WHILE CHUNK (opt_plan/09 N4.2). --cuda-graph-while <n>
    // puts up to n tokens in ONE graph launch whose WHILE body decides on the device
    // where to stop (EOS / context limit / budget); 0 = off, i.e. the per-token static
    // graph stays the safe default. --graph-flight is the same thing with the
    // --decode-flight chunk size ("one launch per in-flight chunk").
    uint32_t cuda_graph_while{0};
    bool graph_flight{false};
    // PINNED download staging (opt_plan/03 section 6): --pinned-transfers 0 keeps every
    // download on the pageable/download-pool path (the A/B arm).
    bool pinned_transfers{true};
    // PDL request (opt_plan/03 section 2.5): installed into the DEVICE CONFIG before
    // creation (CUDADeviceConfigExt::enable_programmatic_stream_serialization), because
    // the wait-at-entry prologue is compiled into every kernel. Inert below compute
    // capability 9.0; the engine reports the probe result and the launch counters.
    bool pdl{false};
    float temperature{0.0f};
    uint64_t seed{0x51f15eED1234abcdull};
      uint32_t max_context{0};// 0 = per-kernel-family safe default
      uint32_t prefill_chunk{0};// 0 = per-version default (0 also enables adaptive)
 // Adaptive outer chunking (opt_plan/02 section 4): enabled when --prefill-chunk
 // is absent, capped by this value and the per-backend conservative limit.
 uint32_t prefill_chunk_max{1024};
 bool legacy_prefill_gemm{false};// A/B: reference one-thread-per-(m,n) GEMM
 // CUDA tensor-core prefill GEMM (NativeShaderExt / CUDA_NVRTC, opt_plan/09
 // N1a). --native-tensor-gemm 0|1 selects it explicitly; absent, the
 // engine default applies.
 bool native_tensor_gemm{true};
 bool have_native_tensor_gemm{false};
 // EXPERIMENTAL CUDA native-Tile typed-FP8 prefill GEMM (src/qwen/qwen/
 // cuda_tile_gemm.h). --cuda-tile-fp8 is the opt-in switch; --cuda-tile-m /
 // --cuda-tile-n override the 64x64 per-program tile and are a USAGE ERROR without
 // the switch (an override that is silently ignored is how a run ends up measuring
 // a tiling nobody asked for). Off by default: the measured default stays the
 // native NVRTC GEMM, this path is slower on sm_89 and JITs one shader per static
 // shape.
 bool cuda_tile_fp8{false};
 uint32_t cuda_tile_m{64};
 uint32_t cuda_tile_n{64};
 bool have_cuda_tile_m{false};
 bool have_cuda_tile_n{false};
 // CUDA tensor-core prefill ATTENTION (NativeShaderExt / CUDA_NVRTC, opt_plan/09
 // N1b: FlashAttention-2). --native-tensor-attention 0|1 selects it explicitly;
 // absent, the engine default applies.
 bool native_tensor_attention{true};
 bool have_native_tensor_attention{false};
              // KV sparse allocation unit in bytes (0 = the 2 MiB default). Raising it
        // cuts the physical allocation count for very long contexts on Vulkan,
        // at the cost of a coarser growth granularity.
        uint64_t kv_page_bytes{0};
    // KV stored dtype (opt_plan/05 Q3): false = FP32 legacy cache (default until
    // measured), true = BF16 codes in the 64-position blocked layout. Requested
    // explicitly so a run log always states which cache it used.
    QwenKvDType kv_dtype{QwenKvDType::f32};
    bool have_kv_dtype{false};
      // W4A16 (opt_plan/05 Q2). `--w4-pack <dir>` is the offline pack the dsl-w4
      // variant reads its weights from; `--export-w4-pack <dir>` runs the offline
      // packer (no device) and exits.
 luisa::string w4_pack;
 bool have_w4_pack{false};
 // OFFLINE merged input projections (opt_plan/03 section 3 + D5).
 // `--projection-pack <dir>` consumes an offline merged projection pack for the
 // bf16 / fp8 / q8_0 variants (the W4 variant refuses one: there is no 4-bit merged
 // realization). `--export-projection-pack <dir>` runs the offline packer (no
 // device, no request) and exits; `--export-projection-format bf16|fp8|q8_0` picks
 // the pack format, and `--merged-decode 0` is the A/B arm that keeps the ordinary
 // per-projection GEMVs on the pack's segment views.
 luisa::string projection_pack;
 bool have_projection_pack{false};
 luisa::string export_projection_pack;
 bool have_export_projection_pack{false};
 luisa::string export_projection_format{"bf16"};
 bool merged_decode{true};
 bool have_merged_decode{false};
      luisa::string export_w4_pack;
      bool have_export_w4_pack{false};
 // --w4-importance <dir>: importance-aware clipping for --export-w4-pack.
 luisa::string w4_importance;
 bool have_w4_importance{false};
 // --w4-pack-only <substring>: TOOLING/SMOKE partial pack (never runnable).
 luisa::string w4_pack_only;
 // --w4-calibrate <dir>: CALIBRATION ONLY (activation second moments for the
 // importance-aware W4 packer); no inference output.
 luisa::string w4_calibrate;
 bool have_w4_calibrate{false};
 uint32_t w4_calib_prompts{4};
 uint32_t w4_calib_prompt_len{128};
 uint32_t w4_calib_decode{32};
      uint32_t w4_group{128};
      float w4_clip_min{0.85f};
      float w4_clip_max{1.0f};
      uint32_t w4_clip_steps{31};
      bool w4_export_fp8{true};
      // Explicit plan option (b) A/B arm: run prefill/LM head through the W4
      // tensors with in-kernel dequantization when no FP8 copy exists.
      bool w4_prefill_in_kernel{false};
      bool w4_lm_head_8bit{true};
      // Quality-gate dump: greedy ids + the prefill and final last-position
      // FP32 logits in a bounded binary file (scripts/qwen_quality_gate.py).
      luisa::string dump_logits;
      bool have_dump_logits{false};
      luisa::string tier;
      luisa::string device_tag;
  };

void print_usage() {
    std::fputs(
        "example_qwen -- text-only Qwen3.5-4B inference (LuisaCompute)\n"
        "\n"
        "usage:\n"
        "  example_qwen --help | --list-versions | --self-test     (offline, no device)\n"
        "  example_qwen [<backend>] [--headless] --backend <name> [options]\n"
        "\n"
        "options:\n"
        "  --backend <name>        backend plugin: cuda | dx | vk | ...\n"
                  "  --version <name> dsl-bf16 (default) | dsl-fp8 | dsl-q8 | dsl-w4 | tile-bf16 | tile-fp8\n"
        "  --model <file.gguf>     default data/Qwen3.5-4B-BF16.gguf; dsl-q8 needs a Q8_0 GGUF\n"
        "                          (any other variant rejects Q8_0: INT8 block is NOT FP8)\n"
        "  --prompt <text>         UTF-8 prompt text (mutually exclusive with --tokens,\n"
        "                          --tokens-file and --synthetic-prompt-len)\n"
"  --prompt-file <path> read the prompt TEXT from a UTF-8 file (mutually exclusive\n"
"                              with --prompt; the long-context needle gate uses\n"
"                              it because an 8 K-token document does not fit a\n"
"                              command line)\n"
        "  --tokens <id,id,...>    explicit prompt token ids (alternative to --prompt)\n"
        "  --tokens-file <path>    read prompt token ids from a file: decimal ids separated\n"
        "                          by commas or whitespace; the only way to pass a very long\n"
        "                          prompt (Windows caps a command line at 32767 characters)\n"
        "  --synthetic-prompt-len <n>  deterministic id list of n tokens, identical to the\n"
        "                          harness generator in scripts/benchmark_qwen.py (LCG seeded\n"
        "                          with n); no command-line length limit\n"
        "  --chat                  wrap the prompt in the Qwen im_start user/assistant template\n"
        "                          (non-thinking default; --prompt only)\n"
        "  --max-new-tokens <n> generated-token budget (default 32)\n"
        "  --decode-flight <n> greedy CUDA DSL forwards per sync (default 4; 0 disables)\n"
        "  --cuda-graph            CUDA: record the greedy decode step ONCE as a static graph and\n"
        "                          launch it per token with zero updates (default; BF16 and FP8)\n"
        "  --no-cuda-graph         CUDA: whole-step command lists per token (A/B and debugging)\n"
        "  --cuda-graph-while <n> CUDA: run up to n decode tokens in ONE conditional\n"
        "                            WHILE graph launch instead of n launches. The loop\n"
        "                            stops ON THE DEVICE (EOS, context limit or the n\n"
        "                            budget) and the engine reads the real trip count\n"
        "                            back, so an early exit never emits a token that was\n"
        "                            not generated. Opt-in and gated twice: the\n"
        "                            CudaGraphExt conditional-node capability AND a\n"
        "                            dedicated instantiate/execute/zero-trip probe; a\n"
        "                            refusal keeps the per-token graph and says why.\n"
        "                            0 = off (default); needs dsl_cuda_graph and the\n"
        "                            tokenizer EOS id (always supplied here)\n"
        "  --graph-flight CUDA: the same WHILE chunk, sized by --decode-flight\n"
        "  --pinned-transfers <0|1> download logits and the bounded greedy token history\n"
        "                            into persistent page-locked staging buffers when\n"
        "                            PinnedMemoryExt is available (copy, synchronize,\n"
        "                            memcpy into the caller vector). Default 1; 0 is the\n"
        "                            pageable/download-pool A/B arm. Both realizations\n"
        "                            move the bytes twice, so no copy is claimed to\n"
        "                            disappear; the engine prints the selection\n"
        "  --pdl <0|1> request programmatic dependent launch (CUDA) BEFORE device\n"
        "                            creation: the backend then compiles an automatic\n"
        "                            wait-at-entry prologue into every kernel and may\n"
        "                            relax launches. Inert below compute capability 9.0;\n"
        "                            the engine reports the CUDAPdlStreamExt probe and the\n"
        "                            relaxed/strict/driver-fallback counters. Default off;\n"
        "                            no engine kernel is PDL-eligible today, so this is\n"
        "                            measurement-only\n"
        "  --split-kv <0|1> DSL decode attention with GQA dedup (default 1): one block per\n"
        " (kv_head,key_split) staging K/V once for all 4 q heads of that\n"
        " KV head, plus a combine kernel that applies the sigmoid gate.\n"
        " 0 keeps the reference per-q_head kernel for every context (A/B).\n"
        "  --split-kv-threshold <n> kv_len at which the split family takes over (default 512);\n"
        " shorter contexts keep the per-q_head kernel. It is a\n"
        " COMPILE-time guard on both kernels, so a static CUDA graph\n"
        " crosses the threshold with zero updates.\n"
                    "  --split-kv-max-splits <n> compile-time grid bound (blocks = kv_heads * n); 0 = derive\n"
                    " from 2 blocks/SM x the reported SM count, else the documented\n"
                    " default 40; n > 40 is refused (legacy kernel stays in charge)\n"
                      "  --gdn-decode-fused <0|1> fuse the decode GDN conv+SiLU+ring update with the QK L2\n"
                      " norm (one launch instead of two; default 1)\n"
                      "  --gdn-state-specialized <0|1> compile the GDN state-column kernels with a\n"
 "  --gdn-fused-recurrent <0|1> fuse the decode GDN recurrence with its gated output\n"
 " RMSNorm (one-CTA-per-head, one launch instead of two; default 0 = the measured\n"
 " default, see .kimix_cache/qwen_benchmark/phase9e/REPORT.md)\n"
                      " compile-time head_dim 128 instead of a runtime extent (default 1)\n"
                      "  --rmsnorm-reg <0|1> register-cached one-pass RMSNorm (hidden 2560; default 1)\n"
        "  --temperature <t> 0 = greedy argmax (default); >0 = categorical sampling\n"
        "  --seed <n>              sampling seed; reproducible by generated-token index\n"
        "  --max-context <n>       KV capacity (default 4096 dsl / 64 tile, <= 262144)\n"
        "  --kv-page-bytes <n>     KV sparse allocation unit; 0 = 2 MiB default, must be\n"
        "                          a multiple of the backend sparse tile (raise it for very\n"
        "                          long contexts on Vulkan, see the caps log)\n"
    "  --kv-dtype <f32|bf16|fp8> K/V cache stored dtype (default f32). bf16 stores two\n"
    "                              RNE BF16 codes per word, fp8 four E4M3FN codes\n"
    "                              per word plus one FP32 scale per (kv_head,\n"
    "                              64-position block); both use the 64-position\n"
    "                              blocked layout, half / a quarter of the FP32 KV\n"
    "                              bytes and read traffic, FP32 activations and\n"
    "                              softmax. fp8 appends through the online-scale\n"
    "                              kernel: a block whose maximum grows is\n"
    "                              rescaled (never a stale code with a new scale).\n"
    "                              The Tile engine fails closed (use --kernel dsl)\n"
    "  --prefill-chunk <n> FIXED prompt rows per engine prefill chunk (default 64 dsl /\n"
    "                              1 tile); passing it disables the adaptive rule below\n"
        "  --prefill-chunk-max <n> adaptive chunk ceiling (default 1024); the effective value is\n"
        " min(this, per-backend limit) and is reported in the\n"
        " 'prefill chunking:' line after the prefill\n"
        "  --legacy-prefill-gemm DSL: use the reference one-thread-per-(m,n) prefill GEMM\n"
        " instead of the weight-reusing warp-per-column GEMM\n"
        "  --native-tensor-gemm 0|1 CUDA tensor-core prefill GEMM through\n"
        " NativeShaderExt/CUDA_NVRTC (mma.sync.aligned.m16n8k16 bf16,\n"
        " FP32 accumulate; x is rounded to BF16 at the shared-tile\n"
                    " boundary). Takes every prefill projection with rows >= 16 and\n"
            " N >= 256 on cuda, for BF16 and FP8 E4M3 weights alike (FP8 is\n"
            " widened and scaled into the BF16 shared tile). Fail-closed\n"
            " otherwise (dx and vk keep the DSL MT path). Decode and the\n"
            " static decode graph are never routed here. Default 1\n"
            " (measured; 0 is the A/B arm)\n"
 "  --native-tensor-attention 0|1 CUDA tensor-core prefill ATTENTION through\n"
 " NativeShaderExt/CUDA_NVRTC (FlashAttention-2: mma.sync\n"
 " bf16 QK^T and P.V with FP32 accumulators and an FP32\n"
 " online softmax; Q/K/V are rounded FP32->BF16 on the\n"
 " shared load, one 16-token query tile x 32-key tile, 4 q\n"
 " heads packed per kv head). Every prefill chunk on cuda with\n"
 " the model geometry; fail-closed otherwise (dx and vk keep the\n"
 " DSL tiled kernel, so does a failed native creation). Decode\n"
 " and the static decode graph are never routed here. Default 1\n"
 " (measured; 0 is the A/B arm)\n"
 "  --cuda-tile-fp8       EXPERIMENTAL, OPT-IN, PREFILL-ONLY, dsl-* engines: run\n"
 "                  the prefill projection GEMM on the CUDA native Tile IR route with TYPED\n"
 "                  FP8 weight storage (tile::Kernel captured per static shape,\n"
 "                  NVRTC Tile IR + tileiras). Requires the cuda backend, the\n"
 "                  CUDA 13.5 toolkit on PATH (CUDA_PATH/CUDA_HOME so tileiras\n"
 "                  finds nvJitLink) and LUISA_CUDA_TILE_IR=1 exported; requires\n"
 "                  FP8 E4M3 prefill weights (dsl-fp8, or the FP8 second copy of\n"
 "                  a dual-copy dsl-w4 session). Takes prefill rows >= 16 and\n"
 "                  N >= 256, i.e. exactly the native/DSL-eligible shapes; the\n"
 "                  narrow N = 32 gate/alpha/beta pair, decode, MTP and the static\n"
 "                  decode graph are untouched. NOT a performance path: measured\n"
 "                  SLOWER than the native NVRTC GEMM on RTX 4080 SUPER sm_89\n"
 "                  (~1.3x at the 64x64 tile, ~2.6x at 128x64) and it JITs one\n"
 "                  shader per static shape (~1 s each), so it is off by default\n"
 "                  and never selected automatically. An explicit request that\n"
 "                  cannot be honoured FAILS engine creation with the reason\n"
 "                  (missing LUISA_CUDA_TILE_IR, non-cuda backend, non-FP8\n"
 "                  weights, tileiras/NVRTC refusal) instead of falling back\n"
 "  --cuda-tile-m <n>    output tile rows of --cuda-tile-fp8 (power of two in\n"
 "                  [16, 128]; default 64). A usage error without --cuda-tile-fp8\n"
 "  --cuda-tile-n <n>    output tile columns of --cuda-tile-fp8 (power of two in\n"
 "                  [16, 128]; default 64). A usage error without --cuda-tile-fp8;\n"
 "                  every model-eligible N is a multiple of 64, and a ragged N\n"
 "                  needs PARTIAL FP8 tile loads, which only compile from sm_90\n"
        "  --gdn-recurrent        DSL: use per-token recurrence instead of WY/UT chunk-64\n"
        "  --gdn-wy-ut 0|1 DSL: pin the GDN prefill schedule (1 = WY/UT chunk-64, 0 ="
        "                  " "exact per-token recurrence) and disable --gdn-auto\n"
        "  --gdn-auto 0|1 DSL: per-prompt GDN prefill selection (default 1): the exact\n"
        "                  per-token recurrence for prompts <= --gdn-auto-threshold and\n"
        "                  WY/UT chunk-64 above it (measured crossover)\n"
        "  --gdn-auto-threshold <n> prompt length where --gdn-auto switches to WY/UT\n"
        "  --decode-packed 0|1 DSL: 128-bit (uint4/float4) decode GEMV, fused FFN and\n"
        "                  gate GEMV with a branchless LUT-free FP8 decode. Default 0:\n"
        "                  implemented, tested and REJECTED by measurement\n"
        "  --decode-packed-fp8 0|1 force the 128-bit realization for FP8 weights too\n"
        "                  implemented and tested but REJECTED by measurement)\n"
        "  --decode-packed-fp8 0|1 force the 128-bit realization for FP8 weights\n"
        "                  gate GEMV with a branchless LUT-free FP8 decode (default 1)\n"
        "  --attn-prepare-model 0|1 DSL: model-specialized decode attention prepare\n"
        "                  (compile-time head_dim 256 / rope_dim 64; default 1)\n"
  "  --legacy-prefill-attn DSL: use the untiled reference prefill attention (one\n"
  " thread per (token,q_head), runtime head_dim local array) for A/B\n"
  "  --tile-chunk-conv 0|1 Tile: one multi-token GDN Conv4+SiLU kernel for the\n"
  " whole chunk + its ring image instead of one kernel + three commands per\n"
  " token (default 1; 0 keeps the per-token differential arm)\n"
  "  --tile-attn-bucket <n> Tile: attention length bucket in keys, so the\n"
  " device-visible-length shader scans ceil(kv_len/n)*n instead of the whole\n"
  " capacity (default 512; 0 streams the full capacity)\n"
        "  --benchmark synchronize after every kernel/transfer and print raw wall timings\n"
        "  --benchmark-csv <file> benchmark mode; append every raw sample to CSV\n"
        "  --mtp-test probe GGUF NextN/MTP weights and measured-default policy; no GPU needed\n"
        "  --mtp                   enable real DSL NextN draft+verify (requires blk.32.* GGUF)\n"
        "  --draft-depth <n>       drafted tokens per MTP round (default 2; prefill chunk >= n+1)\n"
        "  --self-test device-free DSL + Tile operator and tokenizer self-tests\n"
        "  --list-versions         offline support matrix + precision semantics\n"
        "  --headless              accepted for gallery convention; this tool never opens a window\n"
        "\n"
        "precision semantics:\n"
        "  bf16: raw GGUF BF16 payload, packed two codes per uint32 (no conversion)\n"
        "  fp8 : E4M3 quantized from BF16 AT LOAD TIME, FP32 scale per 64-element K group;\n"
        "        the conversion is reported per matrix. Q8_0 GGUF is INT8, never FP8.\n"
        "  q8  : (dsl-q8) the Q8_0 checkpoint's own int8 block codes copied verbatim plus one\n"
        "        exact FP16->FP32 block scale per 32 K elements per row (no re-quantization).\n"
        "  all accumulations/norms/GDN state/softmax are FP32; host sampling is FP64.\n",
        stdout);
}

void print_versions() {
    std::fputs("example_qwen support matrix (text-only Qwen3.5-4B, single sequence)\n\n", stdout);
    for (const auto &v : kVersions) {
        std::printf("  %-9s kernel=%-4s weights=%-5s default-prefill-chunk=%u\n",
                      luisa::string{v.name}.c_str(),
                      luisa::string{qwen_kernel_kind_name(v.kernel)}.c_str(),
                                                v.w4 ? "w4" : (v.q8_0 ? "q8_0" : (v.fp8 ? "fp8" : "bf16")),
                          v.default_prefill_chunk);
        std::printf("            status: %s\n", luisa::string{v.status}.c_str());
    }
    std::fputs(
        "\n"
        "precision semantics:\n"
        "  bf16 = raw GGUF BF16 bytes packed 2-per-uint32 (little-endian, low element first)\n"
        "  fp8  = true E4M3 (max 448) quantized from the BF16 source AT LOAD TIME,\n"
        "         FP32 scale per contiguous group of 64 K elements per output row;\n"
        "         every converted matrix is reported (fp8_quantized diagnostic).\n"
                  "  w4   = W4A16 (dsl-w4 only): 8 signed int4 codes [-8,7] per uint32 word from the\n"
          "         OFFLINE pack (--w4-pack), one FP32 scale per 128-element group per row\n"
          "         produced by a deterministic clipping search (0.85..1.0, 31 steps,\n"
          "         MSE-minimizing -- NOT plain RTN). K%8==0 and K%group==0 are enforced.\n"
          "         The pack carries BOTH payloads: W4 for decode and FP8 E4M3 for\n"
          "         prefill and the LM head (plan option (c)); the LM head is never 4-bit.\n"
          "         --w4-prefill in-kernel runs the measured option (b) A/B arm instead\n"
          "  --w4-importance DIR switches --export-w4-pack from the weights-only\n"
          "  clipping search to the measured ACTIVATION-SECOND-MOMENT weighted\n"
          "  objective (sqrt(importance)-weighted group MSE, normalized per group).\n"
          "  The statistics are bound to the checkpoint by size AND rolling payload\n"
          "  digest and must cover every quantized matrix name+K, else the packer\n"
          "  fails closed. This is NOT AWQ: only the weight of the clipping error\n"
          "  comes from activations.\n"
          "  --w4-calibrate DIR [--w4-calib-prompts N --w4-calib-prompt-len N\n"
          "  --w4-calib-decode N] is the CALIBRATION-ONLY mode that produces that\n"
          "  directory: it runs the BF16 model over a deterministic synthetic prompt\n"
          "  set with per-projection device accumulators and exports m[j]=E[x[j]^2]\n"
          "  plus its manifest; no inference output.\n"
          "  --w4-pack-only SUBSTR is a TOOLING knob: packs only the matching\n"
          "  inventory tensors (a partial pack, never a runnable dsl-w4 checkpoint).\n"
          "         (W4 prefill with in-kernel dequantization); it is never the default.\n"
          "  q8_0 = native Q8_0 (dsl-q8 only): the checkpoint's signed int8 block codes\n"
        "         copied byte-for-byte into the device format, plus one exact\n"
        "         FP16->FP32 scale per 32-element block per output row. ZERO host\n"
        "         re-quantization, and Q8_0 is never routed through FP8. Every other\n"
        "         variant rejects a Q8_0 GGUF explicitly (INT8 block is NOT FP8).\n"
        "  FP32 everywhere else: accumulations, norms, softmax, GDN recurrent state.\n"
        "K/V cache dtype (--kv-dtype, independent of the weight format):\n"
        "  f32  = legacy dense FP32 cache [position][kv_head][channel] (default, the A/B arm)\n"
        "  bf16 = two RNE BF16 codes per word in the 64-position blocked layout\n"
        "  fp8  = four E4M3FN codes per word in the same layout PLUS one FP32 scale per\n"
        "         (kv_head, 64-position block). The append kernel derives the block\n"
        "         maximum, raises the scale when it grows and REWRITES the block's\n"
        "         earlier codes against the new scale: no FP32 shadow cache and never\n"
        "         a stale code paired with a new scale. The scales are dense per\n"
        "         region, so a scale read is always a mapped read.\n"
        "  Activations, accumulators and the softmax stay FP32 for all three.\n"
        "  validated: dsl kernels on cuda/dx/vk (codec, online append, prefill/decode/\n"
        "  split attention, device-state graph) and a short real-GGUF greedy run with\n"
        "  IDENTICAL ids on all three backends; the long-context gate is\n"
        "  scripts/qwen_quality_gate.py --gate q3 (16 prompts at 8 K + a needle set).\n"
        "  The Tile engine and the CUDA native prefill attention REFUSE a non-f32 KV\n"
        "  dtype (fail-closed: both keep the dense FP32 cache contract).\n"
        "\n"
          "scope: text-only backbone (24 GDN + 8 full-attention layers), tied LM head,\n"
          "  greedy or temperature sampling with GGUF EOS stop; DSL WY/UT chunk-64 prefill;\n"
          "  opt-in DSL NextN/MTP draft+verify with complete blk.32.* weights.\n"
          "offline MERGED input projections (opt_plan/03 section 3 + D5):\n"
          "  --export-projection-pack DIR [--export-projection-format bf16|fp8|q8_0]\n"
          "  writes one row-merged matrix per layer group (GDN attn_qkv|attn_gate|\n"
          "  ssm_alpha|ssm_beta -> N=12352, attention attn_q|attn_k|attn_v -> N=10240)\n"
          "  plus a manifest with the source names/roles/layers/shapes, the segment\n"
          "  rows/order, the format and every payload/source hash. No device needed.\n"
          "  --projection-pack DIR replaces those source tensors at upload (no duplicate\n"
          "  residency) after verifying the pack's source identity (checkpoint size +\n"
          "  rolling digest), every payload hash, the geometry and the format; a\n"
          "  mismatch is refused. Decode then runs ONE GEMV per group instead of one per\n"
          "  projection (2 launches saved per layer, 64 per token at this geometry) with\n"
          "  the SAME per-output FP32 accumulation order (bit-identical greedy ids) and\n"
          "  segment-aware destinations -- no intermediate buffer. Prefill is untouched\n"
          "  (the ordinary per-projection GEMMs read the same rows). --merged-decode 0 is\n"
          "  the A/B arm on the same pack; a W4 session refuses a pack.\n"
          "  NOT implemented: vision tower, top-k/top-p, Tile WY/UT or Tile MTP.\n",
          stdout);
}

[[nodiscard]] bool parse_u32(luisa::string_view s, uint32_t &out) noexcept {
    if (s.empty()) { return false; }
    uint64_t v = 0;
    for (auto c : s) {
        if (c < '0' || c > '9') { return false; }
        v = v * 10u + static_cast<uint64_t>(c - '0');
        if (v > 0xffffffffull) { return false; }
    }
    out = static_cast<uint32_t>(v);
    return true;
}

[[nodiscard]] bool parse_u64(luisa::string_view s, uint64_t &out) noexcept {
    if (s.empty()) { return false; }
    uint64_t v = 0;
    for (auto c : s) {
        if (c < '0' || c > '9') { return false; }
        const auto digit = static_cast<uint64_t>(c - '0');
        if (v > (UINT64_MAX - digit) / 10u) { return false; }
        v = v * 10u + digit;
    }
    out = v;
    return true;
}

[[nodiscard]] bool parse_temperature(luisa::string_view s, float &out) noexcept {
    if (s.empty()) { return false; }
    luisa::string owned{s};
    char *end = nullptr;
    const auto value = std::strtof(owned.c_str(), &end);
    if (end != owned.c_str() + owned.size() || !std::isfinite(value) || value < 0.0f) { return false; }
    out = value;
    return true;
}

[[nodiscard]] bool parse_token_list(luisa::string_view s, luisa::vector<uint32_t> &out) {
    out.clear();
    size_t pos = 0;
    while (pos <= s.size()) {
        const auto comma = s.find(',', pos);
        const auto end = comma == luisa::string_view::npos ? s.size() : comma;
        const auto item = s.substr(pos, end - pos);
        uint32_t id = 0;
        if (!parse_u32(item, id)) { return false; }
        out.emplace_back(id);
        if (comma == luisa::string_view::npos) { break; }
        pos = comma + 1;
        if (pos == s.size()) { return false; }// trailing comma
    }
    return !out.empty();
}

// --- --tokens-file ---------------------------------------------------------
// The format (decimal ids separated by commas or whitespace) and its
// diagnostics live in qwen/prompt_ids.h so the host test suite can cover them
// without a device or a model; only the file reading stays here.

// Read the whole file (binary), then parse. Returns false with `error` set.
[[nodiscard]] bool read_tokens_file(luisa::string_view utf8_path,
                                    luisa::vector<uint32_t> &out,
                                    luisa::string &error) {
    luisa::filesystem::path path;
    if (!luisa::path_from_utf8(luisa::string{utf8_path}, path)) {
        error = "path is not valid UTF-8";
        return false;
    }
    std::FILE *file = nullptr;
#ifdef _WIN32
    file = _wfopen(path.c_str(), L"rb");
#else
    file = std::fopen(path.string().c_str(), "rb");
#endif
    if (file == nullptr) {
        error = luisa::format("cannot open '{}'", luisa::string{utf8_path});
        return false;
    }
    luisa::string text;
    char buffer[4096];
    while (true) {
        const auto n = std::fread(buffer, 1u, sizeof(buffer), file);
        if (n > 0u) { text.append(buffer, n); }
        if (n != sizeof(buffer)) { break; }
    }
    const auto read_failed = std::ferror(file) != 0;
    std::fclose(file);
    if (read_failed) {
        error = luisa::format("read error on '{}'", luisa::string{utf8_path});
        return false;
    }
    return qwen_parse_prompt_ids(text, out, error);
}

// Returns false on a bad argument (message already printed).
[[nodiscard]] bool parse_args(int argc, const char *const *argv, CliArgs &args) {
    bool positional_backend = false;
    for (int i = 1; i < argc; i++) {
        const luisa::string_view a{argv[i]};
        const auto next = [&](luisa::string_view &out) -> bool {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "example_qwen: missing value for %s\n", luisa::string{a}.c_str());
                return false;
            }
            out = luisa::string_view{argv[++i]};
            return true;
        };
        if (a == "--help" || a == "-h") {
            args.help = true;
        } else if (a == "--list-versions") {
            args.list_versions = true;
        } else if (a == "--self-test") {
            args.self_test = true;
        } else if (a == "--headless" || a == "--offline") {
            args.headless = true;
        } else if (a == "--chat") {
            args.chat = true;
        } else if (a == "--gdn-recurrent") {
            args.gdn_recurrent = true;
        } else if (a == "--benchmark") {
            args.benchmark = true;
        } else if (a == "--cuda-graph") {
            args.cuda_graph = true;
        } else if (a == "--no-cuda-graph") {
            args.cuda_graph = false;
            } else if (a == "--split-kv") {
                luisa::string_view v;
                if (!next(v) || (v != "0" && v != "1")) {
                    std::fprintf(stderr, "example_qwen: --split-kv expects 0 or 1\n");
                    return false;
                }
                args.split_kv = v == "1";
            } else if (a == "--split-kv-threshold") {
                luisa::string_view v;
                if (!next(v) || !parse_u32(v, args.split_kv_threshold) || args.split_kv_threshold == 0u) {
                    std::fprintf(stderr, "example_qwen: --split-kv-threshold expects an integer >= 1\n");
                    return false;
                }
            } else if (a == "--split-kv-max-splits") {
                luisa::string_view v;
                if (!next(v) || !parse_u32(v, args.split_kv_max_splits)) {
                    std::fprintf(stderr, "example_qwen: --split-kv-max-splits expects an integer\n");
                    return false;
                }
          } else if (a == "--gdn-decode-fused") {
              luisa::string_view v;
              if (!next(v) || (v != "0" && v != "1")) {
                  std::fprintf(stderr, "example_qwen: --gdn-decode-fused expects 0 or 1\n");
                  return false;
              }
              args.gdn_decode_fused = v == "1";
          } else if (a == "--gdn-state-specialized") {
              luisa::string_view v;
              if (!next(v) || (v != "0" && v != "1")) {
                  std::fprintf(stderr, "example_qwen: --gdn-state-specialized expects 0 or 1\n");
                  return false;
              }
              args.gdn_state_specialized = v == "1";
          } else if (a == "--gdn-fused-recurrent") {
              luisa::string_view v;
              if (!next(v) || (v != "0" && v != "1")) {
                  std::fprintf(stderr, "example_qwen: --gdn-fused-recurrent expects 0 or 1\n");
                  return false;
              }
              args.gdn_fused_recurrent = v == "1";
          } else if (a == "--rmsnorm-reg") {
              luisa::string_view v;
              if (!next(v) || (v != "0" && v != "1")) {
                  std::fprintf(stderr, "example_qwen: --rmsnorm-reg expects 0 or 1\n");
                  return false;
              }
              args.rmsnorm_reg = v == "1";

              } else if (a == "--decode-packed") {
                  luisa::string_view v;
                  if (!next(v) || (v != "0" && v != "1")) {
                      std::fprintf(stderr, "example_qwen: --decode-packed expects 0 or 1\n");
                      return false;
                  }
                  args.decode_packed = v == "1";
                  } else if (a == "--decode-packed-fp8") {
                      luisa::string_view v;
                      if (!next(v) || (v != "0" && v != "1")) {
                          std::fprintf(stderr, "example_qwen: --decode-packed-fp8 expects 0 or 1\n");
                          return false;
                      }
                      args.decode_packed_fp8 = v == "1";
              } else if (a == "--attn-prepare-model") {
                  luisa::string_view v;
                  if (!next(v) || (v != "0" && v != "1")) {
                      std::fprintf(stderr, "example_qwen: --attn-prepare-model expects 0 or 1\n");
                      return false;
                  }
                  args.attn_prepare_model = v == "1";
              } else if (a == "--gdn-auto") {
                  luisa::string_view v;
                  if (!next(v) || (v != "0" && v != "1")) {
                      std::fprintf(stderr, "example_qwen: --gdn-auto expects 0 or 1\n");
                      return false;
                  }
                  args.gdn_auto = v == "1";
              } else if (a == "--gdn-auto-threshold") {
                  luisa::string_view v;
                  uint32_t n = 0;
                  if (!next(v) || !parse_u32(v, n)) {
                      std::fprintf(stderr, "example_qwen: --gdn-auto-threshold expects an integer >= 0\n");
                      return false;
                  }
                  args.gdn_auto_threshold = n;
              } else if (a == "--gdn-wy-ut") {
                  luisa::string_view v;
                  if (!next(v) || (v != "0" && v != "1")) {
                      std::fprintf(stderr, "example_qwen: --gdn-wy-ut expects 0 or 1\n");
                      return false;
                  }
                  args.gdn_wy_ut = v == "1";
                  args.have_gdn_wy_ut = true;        } else if (a == "--benchmark-csv") {
            luisa::string_view v;
            if (!next(v)) { return false; }
            args.benchmark_csv = luisa::string{v};
            args.have_benchmark_csv = true;
        } else if (a == "--mtp-test") {
            args.mtp_test = true;
        } else if (a == "--mtp") {
            args.mtp = true;
        } else if (a == "--draft-depth") {
            luisa::string_view v;
            if (!next(v) || !parse_u32(v, args.draft_depth) || args.draft_depth == 0u) {
                std::fprintf(stderr, "example_qwen: --draft-depth expects an integer >= 1\n");
                return false;
            }
          } else if (a == "--legacy-prefill-attn") {
              args.legacy_prefill_attn = true;
          } else if (a == "--tile-chunk-conv") {
              luisa::string_view v;
              if (!next(v) || (v != "0" && v != "1")) {
                  std::fprintf(stderr, "example_qwen: --tile-chunk-conv expects 0 or 1\n");
                  return false;
              }
              args.tile_chunk_conv = v == "1";
          } else if (a == "--tile-attn-bucket") {
              luisa::string_view v;
              if (!next(v) || !parse_u32(v, args.tile_attn_bucket)) {
                  std::fprintf(stderr, "example_qwen: --tile-attn-bucket expects an integer (0 = full capacity)\n");
                  return false;
              }
              args.have_tile_attn_bucket = true;
} else if (a == "--w4-pack") {
 luisa::string_view v;
 if (!next(v)) { return false; }
 args.w4_pack = luisa::string{v};
 args.have_w4_pack = true;
 } else if (a == "--projection-pack") {
 luisa::string_view v;
 if (!next(v)) { return false; }
 args.projection_pack = luisa::string{v};
 args.have_projection_pack = true;
 } else if (a == "--export-projection-pack") {
 luisa::string_view v;
 if (!next(v)) { return false; }
 args.export_projection_pack = luisa::string{v};
 args.have_export_projection_pack = true;
 } else if (a == "--export-projection-format") {
 luisa::string_view v;
 if (!next(v)) { return false; }
 args.export_projection_format = luisa::string{v};
 } else if (a == "--merged-decode") {
 luisa::string_view v;
 if (!next(v)) { return false; }
 uint32_t flag = 0u;
 if (!parse_u32(v, flag) || flag > 1u) {
     std::fprintf(stderr, "example_qwen: --merged-decode expects 0 or 1\n");
     return false;
 }
 args.merged_decode = flag != 0u;
 args.have_merged_decode = true;
          } else if (a == "--export-w4-pack") {
              luisa::string_view v;
              if (!next(v)) { return false; }
              args.export_w4_pack = luisa::string{v};
              args.have_export_w4_pack = true;
            } else if (a == "--w4-importance") {
                luisa::string_view v;
                if (!next(v)) { return false; }
                args.w4_importance = luisa::string{v};
                args.have_w4_importance = true;
            } else if (a == "--w4-pack-only") {
                luisa::string_view v;
                if (!next(v) || v.empty()) {
                    std::fprintf(stderr, "example_qwen: --w4-pack-only expects a tensor-name substring\n");
                    return false;
                }
                args.w4_pack_only = luisa::string{v};
            } else if (a == "--w4-calibrate") {
                luisa::string_view v;
                if (!next(v)) { return false; }
                args.w4_calibrate = luisa::string{v};
                args.have_w4_calibrate = true;
            } else if (a == "--w4-calib-prompts") {
                luisa::string_view v;
                if (!next(v) || !parse_u32(v, args.w4_calib_prompts) || args.w4_calib_prompts == 0u) {
                    std::fprintf(stderr, "example_qwen: --w4-calib-prompts expects an integer >= 1\n");
                    return false;
                }
            } else if (a == "--w4-calib-prompt-len") {
                luisa::string_view v;
                if (!next(v) || !parse_u32(v, args.w4_calib_prompt_len) || args.w4_calib_prompt_len == 0u) {
                    std::fprintf(stderr, "example_qwen: --w4-calib-prompt-len expects an integer >= 1\n");
                    return false;
                }
            } else if (a == "--w4-calib-decode") {
                luisa::string_view v;
                if (!next(v) || !parse_u32(v, args.w4_calib_decode)) {
                    std::fprintf(stderr, "example_qwen: --w4-calib-decode expects an integer >= 0\n");
                    return false;
                }
            } else if (a == "--w4-group") {
                luisa::string_view v;
                if (!next(v) || !parse_u32(v, args.w4_group) ||
                    (args.w4_group != 32u && args.w4_group != 64u && args.w4_group != 128u)) {
                    std::fprintf(stderr, "example_qwen: --w4-group expects 32, 64 or 128\n");
                    return false;
                }
          } else if (a == "--w4-clip") {
              // --w4-clip <min>,<max>,<steps>
              luisa::string_view v;
              if (!next(v)) { return false; }
              const auto comma1 = v.find(',');
              const auto comma2 = comma1 == luisa::string_view::npos ? comma1 : v.find(',', comma1 + 1u);
              if (comma1 == luisa::string_view::npos || comma2 == luisa::string_view::npos ||
                  !parse_temperature(v.substr(0u, comma1), args.w4_clip_min) ||
                  !parse_temperature(v.substr(comma1 + 1u, comma2 - comma1 - 1u), args.w4_clip_max) ||
                  !parse_u32(v.substr(comma2 + 1u), args.w4_clip_steps) ||
                  args.w4_clip_steps < 2u || args.w4_clip_min > args.w4_clip_max) {
                  std::fprintf(stderr, "example_qwen: --w4-clip expects <min>,<max>,<steps> with 0<min<=max\n");
                  return false;
              }
          } else if (a == "--w4-export-fp8") {
              luisa::string_view v;
              if (!next(v) || (v != "0" && v != "1")) {
                  std::fprintf(stderr, "example_qwen: --w4-export-fp8 expects 0 or 1\n");
                  return false;
              }
              args.w4_export_fp8 = v == "1";
            } else if (a == "--w4-export-fp8") {
                luisa::string_view v;
                if (!next(v) || (v != "0" && v != "1")) {
                    std::fprintf(stderr, "example_qwen: --w4-export-fp8 expects 0 or 1\n");
                    return false;
                }
                args.w4_export_fp8 = v == "1";
          } else if (a == "--w4-prefill") {
              luisa::string_view v;
              if (!next(v) || (v != "in-kernel" && v != "decode" && v != "fp8")) {
                  std::fprintf(stderr, "example_qwen: --w4-prefill expects in-kernel|decode|fp8 "
                                       "(fp8 is the plan option (c) default, in-kernel is the "
                                       "option (b) A/B arm)\n");
                  return false;
              }
              args.w4_prefill_in_kernel = v == "in-kernel";
          } else if (a == "--w4-lm-head") {
              luisa::string_view v;
              if (!next(v) || (v != "8bit" && v != "4bit")) {
                  std::fprintf(stderr, "example_qwen: --w4-lm-head expects 8bit|4bit\n");
                  return false;
              }
              args.w4_lm_head_8bit = v == "8bit";
          } else if (a == "--dump-logits") {
              luisa::string_view v;
              if (!next(v)) { return false; }
              args.dump_logits = luisa::string{v};
              args.have_dump_logits = true;
          } else if (a == "--tier") {
              luisa::string_view v;
              if (!next(v) || (v != "t16" && v != "t32")) {
                  std::fprintf(stderr, "example_qwen: --tier expects t16 or t32 (the 16/32 GiB "
                                       "target tiers; it is recorded in quality dumps)\n");
                  return false;
              }
              args.tier = luisa::string{v};
          } else if (a == "--device-tag") {
              luisa::string_view v;
              if (!next(v)) { return false; }
              args.device_tag = luisa::string{v};
          } else if (a == "--backend") {
            luisa::string_view v;
            if (!next(v)) { return false; }
            args.backend = luisa::string{v};
        } else if (a == "--version") {
            luisa::string_view v;
            if (!next(v)) { return false; }
            args.version = luisa::string{v};
        } else if (a == "--model") {
            luisa::string_view v;
            if (!next(v)) { return false; }
            args.model = luisa::string{v};
        } else if (a == "--prompt") {
            luisa::string_view v;
            if (!next(v)) { return false; }
            args.prompt = luisa::string{v};
            args.have_prompt = true;
        } else if (a == "--prompt-file") {
            luisa::string_view v;
            if (!next(v)) { return false; }
            args.prompt_file = luisa::string{v};
            args.have_prompt_file = true;
        } else if (a == "--tokens") {
            luisa::string_view v;
            if (!next(v)) { return false; }
            if (!parse_token_list(v, args.tokens)) {
                std::fprintf(stderr, "example_qwen: --tokens expects a comma-separated id list\n");
                return false;
            }
            args.have_tokens = true;
        } else if (a == "--tokens-file") {
            luisa::string_view v;
            if (!next(v) || v.empty()) {
                std::fprintf(stderr, "example_qwen: --tokens-file expects a path\n");
                return false;
            }
            args.tokens_file = luisa::string{v};
            args.have_tokens_file = true;
        } else if (a == "--synthetic-prompt-len") {
            luisa::string_view v;
            if (!next(v) || !parse_u32(v, args.synthetic_prompt_len) ||
                args.synthetic_prompt_len == 0u) {
                std::fprintf(stderr,
                             "example_qwen: --synthetic-prompt-len expects an integer >= 1\n");
                return false;
            }
            args.have_synthetic_prompt = true;
        } else if (a == "--max-new-tokens") {
            luisa::string_view v;
            if (!next(v) || !parse_u32(v, args.max_new_tokens)) { return false; }
        } else if (a == "--decode-flight") {
            luisa::string_view v;
            if (!next(v) || !parse_u32(v, args.decode_flight)) { return false; }
        } else if (a == "--cuda-graph-while") {
            // Opt-in conditional WHILE chunk: up to n tokens per single graph launch.
            luisa::string_view v;
            if (!next(v) || !parse_u32(v, args.cuda_graph_while)) { return false; }
        } else if (a == "--graph-flight") {
            // The same chunked launch, sized by --decode-flight.
            args.graph_flight = true;
        } else if (a == "--pinned-transfers") {
            luisa::string_view v;
            if (!next(v) || (v != "0" && v != "1")) {
                std::fprintf(stderr, "example_qwen: --pinned-transfers expects 0 or 1\n");
                return false;
            }
            args.pinned_transfers = v == "1";
        } else if (a == "--pdl") {
            luisa::string_view v;
            if (!next(v) || (v != "0" && v != "1")) {
                std::fprintf(stderr, "example_qwen: --pdl expects 0 or 1\n");
                return false;
            }
            args.pdl = v == "1";
        } else if (a == "--temperature") {
            luisa::string_view v;
            if (!next(v) || !parse_temperature(v, args.temperature)) {
                std::fprintf(stderr, "example_qwen: --temperature expects a finite value >= 0\n");
                return false;
            }
        } else if (a == "--seed") {
            luisa::string_view v;
            if (!next(v) || !parse_u64(v, args.seed)) {
                std::fprintf(stderr, "example_qwen: --seed expects an unsigned 64-bit integer\n");
                return false;
            }
        } else if (a == "--max-context") {
            luisa::string_view v;
            if (!next(v) || !parse_u32(v, args.max_context)) { return false; }
        } else if (a == "--prefill-chunk") {
            luisa::string_view v;
            if (!next(v) || !parse_u32(v, args.prefill_chunk)) { return false; }
 } else if (a == "--prefill-chunk-max") {
 luisa::string_view v;
 if (!next(v) || !parse_u32(v, args.prefill_chunk_max)) { return false; }
 } else if (a == "--legacy-prefill-gemm") {
 args.legacy_prefill_gemm = true;
 } else if (a == "--native-tensor-gemm") {
 luisa::string_view v;
 if (!next(v) || (v != "0" && v != "1")) {
     std::fprintf(stderr, "example_qwen: --native-tensor-gemm expects 0 or 1\n");
     return false;
 }
 args.native_tensor_gemm = v == "1";
 args.have_native_tensor_gemm = true;
 } else if (a == "--native-tensor-attention") {
 luisa::string_view v;
 if (!next(v) || (v != "0" && v != "1")) {
 std::fprintf(stderr, "example_qwen: --native-tensor-attention expects 0 or 1\n");
 return false;
 }
 args.native_tensor_attention = v == "1";
 args.have_native_tensor_attention = true;
 } else if (a == "--cuda-tile-fp8") {
 // Same shape as the benchmark flag (bare switch), because this is the same
 // experimental arm: `benchmark_qwen_decode --prefill-gemm --format fp8
 // --cuda-tile-fp8` and this CLI must name it identically.
 args.cuda_tile_fp8 = true;
 } else if (a == "--cuda-tile-m") {
 luisa::string_view v;
 if (!next(v) || !parse_u32(v, args.cuda_tile_m)) {
     std::fprintf(stderr, "example_qwen: --cuda-tile-m expects a power of two in [16, 128] "
                          "(the --cuda-tile-fp8 output tile rows)\n");
     return false;
 }
 args.have_cuda_tile_m = true;
 } else if (a == "--cuda-tile-n") {
 luisa::string_view v;
 if (!next(v) || !parse_u32(v, args.cuda_tile_n)) {
     std::fprintf(stderr, "example_qwen: --cuda-tile-n expects a power of two in [16, 128] "
                          "(the --cuda-tile-fp8 output tile columns)\n");
     return false;
 }
 args.have_cuda_tile_n = true;
        } else if (a == "--kv-page-bytes") {
            luisa::string_view v;
            if (!next(v) || !parse_u64(v, args.kv_page_bytes)) {
                std::fprintf(stderr, "example_qwen: --kv-page-bytes expects an unsigned 64-bit byte count\n");
                return false;
            }
    } else if (a == "--kv-dtype") {
        luisa::string_view v;
        if (!next(v) || !qwen_kv_dtype_from_name(v, args.kv_dtype)) {
            std::fprintf(stderr, "example_qwen: --kv-dtype expects f32, bf16 or fp8\n");
            return false;
        }
        args.have_kv_dtype = true;
        } else if (!a.empty() && a[0] != '-' && !positional_backend && args.backend.empty()) {
            args.backend = luisa::string{a};// gallery convention: positional backend
            positional_backend = true;
        } else {
            std::fprintf(stderr, "example_qwen: unknown argument '%s' (see --help)\n",
                         luisa::string{a}.c_str());
            return false;
        }
    }
    // A tile-shape override without the switch would be ignored, which is exactly the
    // silent-no-op this path must never have: refuse it as a usage error instead.
    if (!args.cuda_tile_fp8 && (args.have_cuda_tile_m || args.have_cuda_tile_n)) {
        std::fprintf(stderr,
                     "example_qwen: --cuda-tile-m/--cuda-tile-n configure the EXPERIMENTAL "
                     "CUDA native Tile FP8 prefill GEMM and have no effect without it; "
                     "pass --cuda-tile-fp8 as well\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Self-test (device-free)
// ---------------------------------------------------------------------------

[[nodiscard]] int run_self_test() {
    int failures = 0;
    const auto run = [&](const char *name, Status s, const luisa::vector<luisa::string> &report) {
        for (const auto &line : report) { std::printf("  [%s] %s\n", name, line.c_str()); }
        if (s.ok()) {
            std::printf("[%s] PASS (%zu report lines)\n", name, report.size());
        } else {
            std::printf("[%s] FAIL: %s (%s)\n", name, s.message.c_str(),
                        luisa::string{status_code_name(s.code)}.c_str());
            failures += 1;
        }
    };
    {
        luisa::vector<luisa::string> report;
        const auto s = dsl::dsl_ops_self_test(&report);
        run("dsl-ops", s, report);
    }
    {
        luisa::vector<luisa::string> report;
        const auto r = tile_ops::tile_ops_self_test(&report);
        Status s = r.ok() ? Status::success()
                          : Status::error(StatusCode::internal_error, r.message);
        run("tile-ops", s, report);
    }
    {
        luisa::vector<luisa::string> report;
        const auto s = QwenTokenizer::self_test(&report);
        run("tokenizer", s, report);
    }
    {
        luisa::vector<luisa::string> report;
        const auto s = mtp_self_test(&report);
        run("mtp-policy", s, report);
    }
    // Shared sampler contract (host, deterministic).
    {
        const float tied[] = {1.0f, 3.0f, 3.0f, 2.0f};
        auto a = qwen_argmax(tied);
        const float nan[] = {1.0f, NAN};
        auto b = qwen_argmax(nan);
        auto c = qwen_argmax(luisa::span<const float>{});
        if (a.ok() && *a.value == 1u && !b.ok() && !c.ok()) {
            std::printf("[argmax] PASS (lowest-index ties, NaN rejection, empty rejection)\n");
        } else {
            std::printf("[argmax] FAIL\n");
            failures += 1;
        }
    }
    std::printf("self-test: %s\n", failures == 0 ? "ALL PASS" : "FAILURES");
    return failures == 0 ? 0 : 2;
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

// argv[0] for the Context (set by run_cli before run_generation).
const char *&argv0_holder() {
    static const char *value = "example_qwen";
    return value;
}

[[nodiscard]] double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

[[nodiscard]] int run_mtp_test(const CliArgs &args) {
    luisa::filesystem::path path;
    if (!luisa::path_from_utf8(args.model, path)) {
        std::fprintf(stderr, "example_qwen: --model path is not valid UTF-8\n");
        return 1;
    }
    auto reader = GgufReader::open(path);
    if (!reader.ok()) {
        std::fprintf(stderr, "example_qwen: cannot open model: %s (%s)\n",
                     reader.status.message.c_str(), reader.status.detail.c_str());
        return 2;
    }
    const auto report = probe_mtp_eligibility(**reader.value);
    for (const auto &line : report.report_lines()) { std::printf("%s\n", line.c_str()); }
    std::printf("mtp policy: available=%s benchmarked=%s accelerates=%s default=%s\n",
                report.available ? "true" : "false",
                report.benchmarked ? "true" : "false",
                report.accelerates ? "true" : "false",
                report.default_enabled ? "enabled" : "disabled");
    return 0;
}

[[nodiscard]] Status emit_benchmark(const CliArgs &args, QwenEngine &engine,
                                    uint64_t iteration, bool &csv_started) {
    const auto *report = engine.benchmark_report();
    if (report == nullptr) {
        return Status::error(StatusCode::internal_error,
                             "benchmark mode requested but engine returned no report");
    }
    std::printf("benchmark: commands=%llu submits=%llu syncs=%llu record=%.3f ms "
                "submit+sync=%.3f ms sampled-total=%.3f ms "
                "(synchronized HOST WALL, not pure GPU time)\n",
                static_cast<unsigned long long>(report->command_count),
                static_cast<unsigned long long>(report->submit_count),
                static_cast<unsigned long long>(report->synchronize_count),
                report->record_ms, report->submit_sync_ms, report->total_ms);
    if (args.benchmark) {
        for (const auto &s : report->samples) {
            std::printf("  [%04llu] %s pos=%u rows=%u layer=", static_cast<unsigned long long>(s.ordinal),
                        luisa::string{qwen_benchmark_phase_name(s.phase)}.c_str(), s.position, s.rows);
            if (s.layer == UINT32_MAX) { std::printf("global"); } else { std::printf("%u", s.layer); }
            std::printf(" %-52s record=%8.4f ms submit+sync=%9.4f ms total=%9.4f ms\n",
                        s.name.c_str(), s.record_ms, s.submit_sync_ms, s.total_ms);
        }
    }
    if (args.have_benchmark_csv) {
        luisa::filesystem::path path;
        if (!luisa::path_from_utf8(args.benchmark_csv, path)) {
            return Status::error(StatusCode::invalid_argument, "benchmark CSV path is not valid UTF-8");
        }
        auto status = qwen_benchmark_append_csv(path, *report, csv_started,
                                                args.version, args.backend, iteration);
        if (!status.ok()) { return status; }
        csv_started = true;
    }
    return Status::success();
}

  // ---------------------------------------------------------------------------
  // Quality-gate dump (scripts/qwen_quality_gate.py).
  //
  // Bounded binary format, little-endian, no text formatting loss: a header with
  // the identity of the run (format, backend, tier, device tag, prompt/generated
  // counts, vocab, temperature, seed), then the prompt ids, the generated greedy
  // ids and the two FP32 last-position distributions (the prefill's, i.e. the one
  // that picked token 0, and the final step's). The script computes prefix
  // agreement, first-32 equality and KL against a BF16 reference dump; anything
  // missing or mismatched is a hard failure there, never a guessed metric.
  // ---------------------------------------------------------------------------
  constexpr char kDumpMagic[8] = {'Q', 'W', 'L', 'O', 'G', 'I', 'T', 'S'};
  constexpr uint32_t kDumpVersion = 1u;

  struct QualityDumpHeader {
      char magic[8];
      uint32_t version;
      uint32_t vocab;
      uint32_t prompt_len;
      uint32_t generated;
      float temperature;
      uint64_t seed;
      char format[32];
      char backend[16];
      char tier[8];
      char device_tag[64];
  };

  [[nodiscard]] bool write_quality_dump(const luisa::string &path, const CliArgs &args,
                                        luisa::string_view format, luisa::string_view backend,
                                        uint32_t vocab, luisa::span<const uint32_t> prompt,
                                        luisa::span<const uint32_t> generated,
                                        luisa::span<const float> prefill_logits,
                                        luisa::span<const float> final_logits) {
      luisa::filesystem::path p;
      if (!luisa::path_from_utf8(path, p)) {
          std::fprintf(stderr, "example_qwen: --dump-logits path is not valid UTF-8\n");
          return false;
      }
      std::FILE *f = nullptr;
  #ifdef _WIN32
      f = _wfopen(p.c_str(), L"wb");
  #else
      f = std::fopen(p.string().c_str(), "wb");
  #endif
      if (f == nullptr) {
          std::fprintf(stderr, "example_qwen: cannot open --dump-logits '%s'\n", path.c_str());
          return false;
      }
      QualityDumpHeader header{};
      std::memcpy(header.magic, kDumpMagic, sizeof(header.magic));
      header.version = kDumpVersion;
      header.vocab = vocab;
      header.prompt_len = static_cast<uint32_t>(prompt.size());
      header.generated = static_cast<uint32_t>(generated.size());
      header.temperature = args.temperature;
      header.seed = args.seed;
      const auto copy_field = [](char *dst, size_t cap, luisa::string_view src) {
          const size_t n = std::min(cap - 1u, src.size());
          std::memcpy(dst, src.data(), n);
          dst[n] = '\0';
      };
      copy_field(header.format, sizeof(header.format), format);
      copy_field(header.backend, sizeof(header.backend), backend);
      copy_field(header.tier, sizeof(header.tier), args.tier);
      copy_field(header.device_tag, sizeof(header.device_tag), args.device_tag);
      bool ok = std::fwrite(&header, sizeof(header), 1u, f) == 1u;
      ok = ok && (prompt.empty() || std::fwrite(prompt.data(), sizeof(uint32_t), prompt.size(), f) == prompt.size());
      ok = ok && (generated.empty() || std::fwrite(generated.data(), sizeof(uint32_t), generated.size(), f) == generated.size());
      ok = ok && (prefill_logits.empty() || std::fwrite(prefill_logits.data(), sizeof(float), prefill_logits.size(), f) == prefill_logits.size());
      ok = ok && (final_logits.empty() || std::fwrite(final_logits.data(), sizeof(float), final_logits.size(), f) == final_logits.size());
      ok = (std::fclose(f) == 0) && ok;
      if (!ok) {
          std::fprintf(stderr, "example_qwen: writing --dump-logits '%s' failed\n", path.c_str());
          return false;
      }
      std::printf("quality dump: %s (vocab=%u prompt=%zu generated=%zu, prefill+final logits)\n",
                  path.c_str(), vocab, prompt.size(), generated.size());
      return true;
  }

  // ---------------------------------------------------------------------------
  // Offline W4 packer (--export-w4-pack): no device, no request.
  //
  // --w4-importance <dir> switches the per-group clipping objective from the
  // weights-only search to the measured activation-second-moment-weighted search
  // (qwen/w4_importance.h). The statistics must describe the SAME checkpoint
  // (size + rolling payload digest) and must cover every matrix this pack
  // quantizes (name + K), otherwise the packer fails closed.
  // ---------------------------------------------------------------------------
  [[nodiscard]] int run_export_w4_pack(const CliArgs &args) {
      luisa::filesystem::path model_path;
      if (!luisa::path_from_utf8(args.model, model_path)) {
          std::fprintf(stderr, "example_qwen: --model path is not valid UTF-8\n");
          return 1;
      }
      luisa::filesystem::path dir;
      if (!luisa::path_from_utf8(args.export_w4_pack, dir)) {
          std::fprintf(stderr, "example_qwen: --export-w4-pack path is not valid UTF-8\n");
          return 1;
      }
      auto reader = GgufReader::open(model_path);
      if (!reader.ok()) {
          std::fprintf(stderr, "example_qwen: cannot open model: %s (%s)\n",
                       reader.status.message.c_str(), reader.status.detail.c_str());
          return 2;
      }
      auto model = Qwen35Model::load(**reader.value);
      if (!model.ok()) {
          std::fprintf(stderr, "example_qwen: model validation failed: %s (%s)\n",
                       model.status.message.c_str(), model.status.detail.c_str());
          return 2;
      }
      W4PackOptions options;
      options.clip.group = args.w4_group;
      options.clip.min_ratio = args.w4_clip_min;
      options.clip.max_ratio = args.w4_clip_max;
      options.clip.steps = args.w4_clip_steps;
      options.emit_fp8_payloads = args.w4_export_fp8;
      // Importance-aware arm: load, verify and pin the calibration statistics.
      luisa::optional<W4Importance> importance;
      if (args.have_w4_importance) {
          luisa::filesystem::path importance_dir;
          if (!luisa::path_from_utf8(args.w4_importance, importance_dir)) {
              std::fprintf(stderr, "example_qwen: --w4-importance path is not valid UTF-8\n");
              return 1;
          }
          auto stats = read_w4_importance(importance_dir);
          if (!stats.ok()) {
              std::fprintf(stderr, "example_qwen: --w4-importance is not usable: %s (%s)\n",
                           stats.status.message.c_str(), stats.status.detail.c_str());
              return 2;
          }
          const auto &imp = *stats.value;
          const uint64_t file_size = (*reader.value)->file_size();
          if (imp.source.size != file_size) {
              std::fprintf(stderr,
                           "example_qwen: importance statistics were collected on a different "
                           "checkpoint (source size %llu vs %llu bytes): refusing stale data\n",
                           static_cast<unsigned long long>(imp.source.size),
                           static_cast<unsigned long long>(file_size));
              return 2;
          }
          auto digest = compute_w4_source_hash(**reader.value, *model.value);
          if (!digest.ok()) {
              std::fprintf(stderr, "example_qwen: cannot recompute the checkpoint digest: %s\n",
                           digest.status.message.c_str());
              return 2;
          }
          if (*digest.value != imp.source.hash) {
              std::fprintf(stderr,
                           "example_qwen: importance statistics were collected on a checkpoint "
                           "with different contents but the SAME size (digest %s vs %s): "
                           "refusing stale data\n",
                           qwen_pack_hash_string(imp.source.hash).c_str(),
                           qwen_pack_hash_string(*digest.value).c_str());
              return 2;
          }
          std::printf("w4 importance: %llu matrices, source=%s tokens=%llu prompts=%u x %u "
                      "(+%u decode), digest=%s\n",
                      static_cast<unsigned long long>(imp.tensors.size()),
                      imp.source.name.c_str(),
                      static_cast<unsigned long long>(imp.source.tokens), imp.source.prompts,
                      imp.source.prompt_len, imp.source.decode_tokens,
                      qwen_pack_hash_string(imp.source.hash).c_str());
          importance.emplace(std::move(*stats.value));
          options.importance = &*importance;
      }
      // A filtered inventory is a TOOLING/SMOKE pack (it is not a runnable dsl-w4
      // checkpoint): it exercises the packer without spending the full multi-minute
      // model pass. It is loudly reported and its manifest digest identifies the
      // SUBSET, never the whole checkpoint.
      luisa::vector<QwenTensorRef> subset;
      luisa::span<const QwenTensorRef> inventory = (*model.value).tensors();
      if (!args.w4_pack_only.empty()) {
          const auto needle = luisa::string_view{args.w4_pack_only};
          for (const auto &ref : (*model.value).tensors()) {
              if (ref.info == nullptr) { continue; }
              if (luisa::string_view{ref.info->name}.find(needle) != luisa::string_view::npos) {
                  subset.push_back(ref);
              }
          }
          if (subset.empty()) {
              std::fprintf(stderr, "example_qwen: --w4-pack-only '%s' matched no tensor\n",
                           args.w4_pack_only.c_str());
              return 2;
          }
          inventory = luisa::span<const QwenTensorRef>{subset.data(), subset.size()};
          std::printf("w4 packer: --w4-pack-only '%s' selected %llu of %llu tensors -- this is a "
                      "partial TOOLING pack, not a runnable dsl-w4 checkpoint\n",
                      args.w4_pack_only.c_str(),
                      static_cast<unsigned long long>(subset.size()),
                      static_cast<unsigned long long>((*model.value).tensors().size()));
      }
      luisa::string base_name = args.model;
      if (const auto slash = base_name.rfind('/'); slash != luisa::string::npos) {
          base_name = base_name.substr(slash + 1u);
      }
      if (const auto bslash = base_name.rfind('\\'); bslash != luisa::string::npos) {
          base_name = base_name.substr(bslash + 1u);
      }
      std::printf("w4 packer: model=%s size=%llu tensors=%zu clip=(group=%u min=%g max=%g steps=%u) "
                  "payloads=%s importance=%s -> %s\n",
                  args.model.c_str(), static_cast<unsigned long long>((*reader.value)->file_size()),
                  inventory.size(), options.clip.group,
                  static_cast<double>(options.clip.min_ratio),
                  static_cast<double>(options.clip.max_ratio), options.clip.steps,
                  options.emit_fp8_payloads ? "w4+fp8 (plan option (c))" : "w4 only (option (b) arm)",
                  options.importance != nullptr ? "measured activation second moments"
                                                : "none (clipping only)",
                  args.export_w4_pack.c_str());
      std::fflush(stdout);
      W4PackReport report;
      auto st = subset.empty()
                    ? pack_w4_from_gguf(**reader.value, *model.value, base_name, dir, options, &report)
                    : pack_w4_from_gguf(**reader.value, inventory, base_name, dir, options, &report);
      if (!st.ok()) {
          std::fprintf(stderr, "example_qwen: w4 packer failed: %s (%s)\n",
                       st.message.c_str(), st.detail.c_str());
          return 2;
      }
      std::printf("w4 packer: %llu tensors (%llu w4 matrices, %llu dense) in %.1f s; "
                  "w4=%.1f MiB scales=%.1f MiB fp8=%.1f MiB fp8-scales=%.1f MiB f32=%.1f MiB\n",
                  static_cast<unsigned long long>(report.tensors),
                  static_cast<unsigned long long>(report.w4_matrices),
                  static_cast<unsigned long long>(report.dense_tensors), report.seconds,
                  static_cast<double>(report.w4_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(report.w4s_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(report.fp8_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(report.fp8s_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(report.f32_bytes) / (1024.0 * 1024.0));
      // Aggregate statistics are MODEL-WIDE and normalized once (element-weighted
      // MSE, group-weighted mean ratio), never a per-tensor leftover.
      std::printf("w4 clip search: elements=%llu groups=%llu clipped=%llu mean_ratio=%.4f "
                  "mse=%.6e rtn_mse=%.6e (improvement %.2f%%)\n",
                  static_cast<unsigned long long>(report.w4_elements),
                  static_cast<unsigned long long>(report.groups),
                  static_cast<unsigned long long>(report.clipped_groups),
                  static_cast<double>(report.mean_ratio), report.mse, report.rtn_mse,
                  report.mse_improvement());
      if (report.importance_matrices != 0u) {
          std::printf("w4 importance-aware objective: weighted_mse=%.6e rtn_weighted_mse=%.6e "
                      "(improvement %.2f%%); matrices weighted=%llu unweighted=%llu\n",
                      report.weighted_mse, report.rtn_weighted_mse,
                      report.weighted_improvement(),
                      static_cast<unsigned long long>(report.importance_matrices),
                      static_cast<unsigned long long>(report.unweighted_matrices));
      }
        return 0;
    }

    // ---------------------------------------------------------------------------
    // Offline MERGED input-projection packer (--export-projection-pack DIR).
    // opt_plan/03 section 3 + risk D5, opt_plan/05 section 5.
    //
    // Writes one row-merged matrix per decoder layer -- GDN attn_qkv|attn_gate|
    // ssm_alpha|ssm_beta (N = 12352) and attention attn_q|attn_k|attn_v
    // (N = 10240) -- in the DEVICE ABI of one format, plus a manifest recording the
    // source names/roles/layers/shapes, the segment rows/order, the format and every
    // payload + source hash. No device, no request, no re-quantization of the source
    // (an fp8 pack quantizes exactly like the uploader's load-time conversion, from
    // the same host reference).
    //
    // Format: --export-projection-format bf16 (BF16 checkpoint), fp8 (BF16/F32
    // checkpoint, E4M3 with per-row groups of 64) or q8_0 (Q8_0 checkpoint, block
    // codes copied verbatim). A format that does not match the checkpoint is
    // refused instead of converting silently.
    // ---------------------------------------------------------------------------
    [[nodiscard]] int run_export_projection_pack(const CliArgs &args) {
        luisa::filesystem::path model_path;
        if (!luisa::path_from_utf8(args.model, model_path)) {
            std::fprintf(stderr, "example_qwen: --model path is not valid UTF-8\n");
            return 1;
        }
        luisa::filesystem::path dir;
        if (!luisa::path_from_utf8(args.export_projection_pack, dir)) {
            std::fprintf(stderr, "example_qwen: --export-projection-pack path is not valid UTF-8\n");
            return 1;
        }
        ProjectionPackOptions options;
        const auto &fmt = args.export_projection_format;
        if (fmt == "bf16") {
            options.format = WeightStorage::bf16_packed;
            options.group = 0u;
        } else if (fmt == "fp8" || fmt == "fp8_e4m3") {
            options.format = WeightStorage::fp8_packed;
            options.group = kFp8GroupSize;
        } else if (fmt == "q8" || fmt == "q8_0") {
            options.format = WeightStorage::q8_0_packed;
            options.group = kQ8_0BlockSize;
        } else {
            std::fprintf(stderr,
                         "example_qwen: --export-projection-format '%s' is not one of "
                         "bf16|fp8|q8_0 (W4 has no merged realization)\n",
                         fmt.c_str());
            return 2;
        }
        auto reader = GgufReader::open(model_path);
        if (!reader.ok()) {
            std::fprintf(stderr, "example_qwen: cannot open model: %s (%s)\n",
                         reader.status.message.c_str(), reader.status.detail.c_str());
            return 2;
        }
        auto model = Qwen35Model::load(**reader.value);
        if (!model.ok()) {
            std::fprintf(stderr, "example_qwen: model validation failed: %s (%s)\n",
                         model.status.message.c_str(), model.status.detail.c_str());
            return 2;
        }
        luisa::string base_name = args.model;
        if (const auto slash = base_name.rfind('/'); slash != luisa::string::npos) {
            base_name = base_name.substr(slash + 1u);
        }
        if (const auto bslash = base_name.rfind('\\'); bslash != luisa::string::npos) {
            base_name = base_name.substr(bslash + 1u);
        }
        std::printf("projection packer: model=%s size=%llu format=%s group=%u -> %s\n",
                    args.model.c_str(),
                    static_cast<unsigned long long>((*reader.value)->file_size()),
                    weight_storage_name(options.format).data(), options.group,
                    args.export_projection_pack.c_str());
        std::fflush(stdout);
        ProjectionPackReport report;
        if (auto st = pack_projection_pack_from_gguf(**reader.value, *model.value, base_name,
                                                     dir, options, &report);
            !st.ok()) {
            std::fprintf(stderr, "example_qwen: projection packer failed: %s (%s)\n",
                         st.message.c_str(), st.detail.c_str());
            return 2;
        }
        std::printf("projection packer: %llu matrices / %llu segments in %.1f s; "
                    "words=%.1f MiB scales=%.1f MiB source=%.1f MiB read, source digest=%s\n",
                    static_cast<unsigned long long>(report.matrices),
                    static_cast<unsigned long long>(report.segments), report.seconds,
                    static_cast<double>(report.word_bytes) / (1024.0 * 1024.0),
                    static_cast<double>(report.scale_bytes) / (1024.0 * 1024.0),
                    static_cast<double>(report.source_bytes_read) / (1024.0 * 1024.0),
                    qwen_pack_hash_string(report.source_hash).c_str());
          return 0;
      }

    // ---------------------------------------------------------------------------
    // W4 importance calibration (--w4-calibrate DIR): CALIBRATION ONLY, no request.
    //
    // Runs the BF16 model over a small DETERMINISTIC prompt set and, at every
  // projection site, accumulates the squared input activation per input channel on
  // the device (QwenEngine::begin/finish_activation_calibration). The exported
  // statistic is the measured second moment m[j] = E[x[j]^2] (diag(H) under
  // independent channels), which --w4-importance feeds into the offline packer's
  // weighted group objective. Nothing here is derived from the weights, and no
  // quality claim is attached: the gate stays the only evidence.
  //
  // The prompt set is qwen_synthetic_prompt_ids(prompt_len + i) for i in
  // [0, prompts) -- the same deterministic generator the CLI/harness use -- plus
  // `decode` greedy tokens per prompt. It is written into the manifest, so a
  // calibration directory states exactly what produced it and can be reproduced.
  // ---------------------------------------------------------------------------
  [[nodiscard]] int run_w4_calibration(const CliArgs &args) {
 if (args.backend.empty()) {
 std::fprintf(stderr, "example_qwen: --w4-calibrate needs --backend (cuda|dx|vk)\n");
 return 1;
 }
 // The calibration engine runs BF16/F32 weights through the reference M=1 route, so the
 // opt-in CUDA Tile FP8 prefill GEMM can never apply to it: refusing beats silently
 // accepting a flag that would do nothing in this mode.
 if (args.cuda_tile_fp8) {
 std::fprintf(stderr, "example_qwen: --cuda-tile-fp8 does not apply to --w4-calibrate "
                      "(the calibration runs BF16/F32 weights, not an FP8 prefill copy)\n");
 return 1;
 }
      luisa::filesystem::path model_path;
      if (!luisa::path_from_utf8(args.model, model_path)) {
          std::fprintf(stderr, "example_qwen: --model path is not valid UTF-8\n");
          return 1;
      }
      luisa::filesystem::path out_dir;
      if (!luisa::path_from_utf8(args.w4_calibrate, out_dir)) {
          std::fprintf(stderr, "example_qwen: --w4-calibrate path is not valid UTF-8\n");
          return 1;
      }
      if (args.w4_calib_prompts == 0u || args.w4_calib_prompt_len == 0u) {
          std::fprintf(stderr, "example_qwen: --w4-calib-prompts/--w4-calib-prompt-len must be > 0\n");
          return 1;
      }
      auto reader = GgufReader::open(model_path);
      if (!reader.ok()) {
          std::fprintf(stderr, "example_qwen: cannot open model: %s (%s)\n",
                       reader.status.message.c_str(), reader.status.detail.c_str());
          return 2;
      }
      // The statistics must describe the checkpoint the pack will be built FROM, and
      // a W4 pack is only ever produced from BF16/F32: refuse Q8_0 here.
      for (const auto &t : (*reader.value)->tensors()) {
          if (t.type == static_cast<uint32_t>(GgufTensorType::q8_0)) {
              std::fprintf(stderr,
                           "example_qwen: --w4-calibrate needs the BF16/F32 checkpoint the W4 "
                           "pack is derived from; '%s' is Q8_0\n", t.name.c_str());
              return 2;
          }
      }
      auto model = Qwen35Model::load(**reader.value);
      if (!model.ok()) {
          std::fprintf(stderr, "example_qwen: model validation failed: %s (%s)\n",
                       model.status.message.c_str(), model.status.detail.c_str());
          return 2;
      }
      const uint32_t vocab = (*model.value).config().geometry.vocab;
      // Deterministic prompt set.
      luisa::vector<luisa::vector<uint32_t>> prompts;
      uint64_t total_rows = 0u;
      for (uint32_t p = 0u; p < args.w4_calib_prompts; ++p) {
          auto ids = qwen_synthetic_prompt_ids(args.w4_calib_prompt_len + p);
          if (ids.empty()) {
              std::fprintf(stderr, "example_qwen: empty synthetic prompt %u\n", p);
              return 1;
          }
          for (auto id : ids) {
              if (id >= vocab) {
                  std::fprintf(stderr, "example_qwen: synthetic id %u outside the vocabulary\n", id);
                  return 1;
              }
          }
          total_rows += ids.size() + args.w4_calib_decode;
          prompts.emplace_back(std::move(ids));
      }
      const uint32_t context_tokens =
          args.w4_calib_prompt_len + args.w4_calib_prompts + args.w4_calib_decode + 8u;

      compute::Context context{argv0_holder()};
      auto device = context.create_device(args.backend.c_str());
      if (!device) {
          std::fprintf(stderr, "example_qwen: backend '%s' not available\n", args.backend.c_str());
          return 2;
      }
      auto stream = device.create_stream();
      if (!stream) {
          std::fprintf(stderr, "example_qwen: cannot create stream\n");
          return 2;
      }
      std::printf("w4 calibration: model=%s backend=%s prompts=%u x %u..%u (+%u greedy) "
                  "rows=%llu context=%u\n",
                  args.model.c_str(), args.backend.c_str(), args.w4_calib_prompts,
                  args.w4_calib_prompt_len, args.w4_calib_prompt_len + args.w4_calib_prompts - 1u,
                  args.w4_calib_decode, static_cast<unsigned long long>(total_rows), context_tokens);
      std::fflush(stdout);
      // BF16 (or F32) weights: the calibration runs the SAME numerics the pack is
      // derived from, never a quantized copy.
      QwenWeightUploadOptions upload_options;
      QwenWeightUploadReport upload_report;
      auto weights = QwenDeviceWeights::upload(device, stream, **reader.value, *model.value,
                                               upload_options, &upload_report);
      if (!weights.ok()) {
          std::fprintf(stderr, "example_qwen: weight upload failed: %s (%s)\n",
                       weights.status.message.c_str(), weights.status.detail.c_str());
          return 2;
      }
      stream.synchronize();
      std::printf("w4 calibration: weights=%llu tensors (f32=%llu bf16=%llu) %.1f MiB device\n",
                  static_cast<unsigned long long>(upload_report.tensor_count),
                  static_cast<unsigned long long>(upload_report.f32_count),
                  static_cast<unsigned long long>(upload_report.bf16_count),
                  static_cast<double>(upload_report.device_bytes) / (1024.0 * 1024.0));
      QwenEngineOptions options;
      options.kernel = QwenKernelKind::dsl;
      options.max_context = context_tokens;
      options.prefill_chunk = 64u;
      options.calibration_only = true;
      options.dsl_cuda_graph = false;// the calibration is a host-logits run
      auto created = QwenEngine::create(device, stream, *model.value, *weights.value, options);
      if (!created.ok()) {
          std::fprintf(stderr, "example_qwen: engine creation failed: %s (%s)\n",
                       created.status.message.c_str(), created.status.detail.c_str());
          return 2;
      }
      auto engine = std::move(*created.value);
      if (auto s = engine->begin_activation_calibration(stream); !s.ok()) {
          std::fprintf(stderr, "example_qwen: cannot start the calibration: %s (%s)\n",
                       s.message.c_str(), s.detail.c_str());
          return 2;
      }
      const auto t_run = std::chrono::steady_clock::now();
      for (uint32_t p = 0u; p < prompts.size(); ++p) {
          if (auto s = engine->reset(stream); !s.ok()) {
              std::fprintf(stderr, "example_qwen: reset failed: %s\n", s.message.c_str());
              return 2;
          }
          luisa::vector<float> logits;
          if (auto s = engine->prefill(stream, luisa::span<const uint32_t>{prompts[p].data(),
                                                                           prompts[p].size()},
                                       logits);
              !s.ok()) {
              std::fprintf(stderr, "example_qwen: prefill failed: %s (%s)\n",
                           s.message.c_str(), s.detail.c_str());
              return 2;
          }
          for (uint32_t d = 0u; d < args.w4_calib_decode; ++d) {
              auto next = qwen_argmax(luisa::span<const float>{logits.data(), logits.size()});
              if (!next.ok()) {
                  std::fprintf(stderr, "example_qwen: argmax failed: %s\n", next.status.message.c_str());
                  return 2;
              }
              if (auto s = engine->decode(stream, *next.value, logits); !s.ok()) {
                  std::fprintf(stderr, "example_qwen: decode failed: %s (%s)\n",
                               s.message.c_str(), s.detail.c_str());
                  return 2;
              }
          }
          std::printf("w4 calibration: prompt %u/%zu done (%llu tokens)\n", p + 1u,
                      prompts.size(),
                      static_cast<unsigned long long>(prompts[p].size() + args.w4_calib_decode));
          std::fflush(stdout);
      }
      auto report = engine->finish_activation_calibration(stream);
      if (!report.ok()) {
          std::fprintf(stderr, "example_qwen: calibration failed: %s (%s)\n",
                       report.status.message.c_str(), report.status.detail.c_str());
          return 2;
      }
      const double run_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run).count();
      // The checkpoint identity the statistics belong to: the same rolling payload
      // digest a W4 pack records, so the packer can prove the data is fresh.
      auto digest = compute_w4_source_hash(**reader.value, *model.value);
      if (!digest.ok()) {
          std::fprintf(stderr, "example_qwen: cannot compute the checkpoint digest: %s (%s)\n",
                       digest.status.message.c_str(), digest.status.detail.c_str());
          return 2;
      }
      W4ImportanceSource source;
      luisa::string base_name = args.model;
      if (const auto slash = base_name.rfind('/'); slash != luisa::string::npos) {
          base_name = base_name.substr(slash + 1u);
      }
      if (const auto bslash = base_name.rfind('\\'); bslash != luisa::string::npos) {
          base_name = base_name.substr(bslash + 1u);
      }
      source.name = base_name;
      source.size = (*reader.value)->file_size();
      source.hash = *digest.value;
      source.prompts = args.w4_calib_prompts;
      source.prompt_len = args.w4_calib_prompt_len;
      source.decode_tokens = args.w4_calib_decode;
      source.tokens = report.value->tokens;
      if (auto s = write_w4_importance(out_dir, source, *report.value); !s.ok()) {
          std::fprintf(stderr, "example_qwen: cannot write the importance statistics: %s (%s)\n",
                       s.message.c_str(), s.detail.c_str());
          return 2;
      }
      uint64_t channels = 0u;
      double mean_all = 0.0;
      for (const auto &t : report.value->observed) {
          double sum = 0.0;
          for (auto v : t.second_moment) { sum += static_cast<double>(v); }
          channels += t.k;
          mean_all += sum;
      }
      std::printf("w4 calibration: %llu matrices, %llu channels, %llu tokens in %.1f s; "
                  "mean second moment %.6g; source=%s size=%llu digest=%s -> %s\n",
                  static_cast<unsigned long long>(report.value->tensors),
                  static_cast<unsigned long long>(channels),
                  static_cast<unsigned long long>(report.value->tokens), run_seconds,
                  channels != 0u ? mean_all / static_cast<double>(channels) : 0.0,
                  source.name.c_str(), static_cast<unsigned long long>(source.size),
                  qwen_pack_hash_string(source.hash).c_str(), args.w4_calibrate.c_str());
      return 0;
  }

  [[nodiscard]] int run_generation(const CliArgs &args) {
    // --- resolve the variant -------------------------------------------------
    const VersionSpec *spec = nullptr;
    for (const auto &v : kVersions) {
        if (v.name == args.version) { spec = &v; }
    }
    if (spec == nullptr) {
        std::fprintf(stderr, "example_qwen: unknown --version '%s' (see --list-versions)\n",
                     args.version.c_str());
        return 1;
    }
    if (args.backend.empty()) {
        std::fprintf(stderr, "example_qwen: --backend is required for generation (cuda|dx|vk)\n");
        return 1;
    }
    // Exactly one prompt source. --tokens-file and --synthetic-prompt-len exist
    // because an 8192-token --tokens list exceeds the Windows command-line cap.
    {
        const auto sources = static_cast<int>(args.have_prompt) +
        static_cast<int>(args.have_prompt_file) +
                             static_cast<int>(args.have_tokens) +
                             static_cast<int>(args.have_tokens_file) +
                             static_cast<int>(args.have_synthetic_prompt);
        if (sources != 1) {
            std::fprintf(stderr,
                         "example_qwen: exactly one of --prompt / --prompt-file / --tokens / --tokens-file / "
                         "--synthetic-prompt-len is required (got %d)\n",
                         sources);
            return 1;
        }
    }
    if (args.chat && !(args.have_prompt || args.have_prompt_file)) {
        std::fprintf(stderr, "example_qwen: --chat requires --prompt or --prompt-file\n");
        return 1;
    }
    if (args.max_new_tokens == 0) {
        std::fprintf(stderr, "example_qwen: --max-new-tokens must be >= 1\n");
        return 1;
    }
    if (args.mtp && spec->kernel != QwenKernelKind::dsl) {
        std::fprintf(stderr, "example_qwen: --mtp currently requires a dsl-* version (no Tile draft fallback)\n");
        return 1;
    }
    // The experimental CUDA native-Tile FP8 prefill GEMM is a QwenDslEngine option; the
    // Tile engine has its own frozen schedule and would silently ignore the request, so
    // refuse the combination instead of letting a run believe it measured that path.
    if (args.cuda_tile_fp8 && spec->kernel != QwenKernelKind::dsl) {
        std::fprintf(stderr, "example_qwen: --cuda-tile-fp8 is a dsl-* engine option "
                             "(QwenDslEngine); the Tile engine keeps its own prefill schedule\n");
        return 1;
    }
    uint32_t prefill_chunk = args.prefill_chunk != 0 ? args.prefill_chunk
                                                     : spec->default_prefill_chunk;
    if (args.mtp) { prefill_chunk = std::max(prefill_chunk, args.draft_depth + 1u); }
    const uint32_t max_context = args.max_context != 0 ? args.max_context
                                                       : (spec->kernel == QwenKernelKind::tile ? 64u : 4096u);

    std::printf("example_qwen: version=%s backend=%s model=%s\n", args.version.c_str(),
                args.backend.c_str(), args.model.c_str());
    std::printf("precision: %s\n",
                spec->w4
                ? "W4A16 (dsl-w4) -- 8 signed int4 codes [-8,7] per uint32 word from the "
                  "OFFLINE pack, one FP32 scale per 128-element K group per row from a "
                  "deterministic clipping search (MSE-minimizing, not RTN). Decode reads the "
                  "W4 copy; prefill and the LM head read the pack's FP8 E4M3 second copy "
                  "(plan option (c)), so the LM head is never 4-bit."
                : (spec->q8_0
                    ? "Q8_0 NATIVE (dsl-q8) -- the checkpoint's signed int8 block codes are "
                      "copied verbatim into the device format and every block's FP16 scale is "
                      "widened exactly to FP32 (one scale per 32 K elements per row): ZERO host "
                      "re-quantization, never routed through FP8. Activations/accumulation FP32."
                    : (spec->fp8 ? "FP8 E4M3 -- quantized from the BF16 GGUF source AT LOAD TIME "
                                   "(FP32 scale per 64-element K group per row; conversion reported below). "
                                   "Q8_0 is INT8, NOT FP8, and is rejected here "
                                   "(use --version dsl-q8 for Q8_0 checkpoints)."
                                 : "BF16 -- raw GGUF BF16 payload packed 2-per-uint32 (no conversion). "
                                   "Q8_0 is INT8, NOT FP8, and is rejected here "
                                   "(use --version dsl-q8 for Q8_0 checkpoints).")));
 // Fixed vs adaptive outer chunking (opt_plan/02 section 4). The chunk of THIS
 // prompt is reported in the 'prefill chunking:' line after the prefill, so a
 // benchmark log distinguishes configured from actually used.
 const bool adaptive_chunk = spec->kernel == QwenKernelKind::dsl && args.prefill_chunk == 0u &&
 !args.mtp;
 const bool gdn_pinned = spec->kernel == QwenKernelKind::dsl &&
                         (args.have_gdn_wy_ut || args.gdn_recurrent || !args.gdn_auto);
 const char *gdn_schedule =
     spec->kernel != QwenKernelKind::dsl
         ? "exact per-token GDN recurrence"
         : (gdn_pinned ? (args.gdn_wy_ut && !args.gdn_recurrent
                              ? "FP32 WY/UT chunk-64 GDN"
                              : "exact per-token GDN recurrence")
                        : "per-prompt GDN schedule (measured crossover)");
 std::printf("prefill: %s in %s chunks%s\n", gdn_schedule,
             adaptive_chunk ? "adaptive" : "fixed",
             args.mtp ? "; MTP draft+verify enabled" : "");

    // --- GGUF + model + tokenizer (host) --------------------------------------
    const auto t_open = std::chrono::steady_clock::now();
    luisa::filesystem::path model_path;
    if (!luisa::path_from_utf8(args.model, model_path)) {
        std::fprintf(stderr, "example_qwen: --model path is not valid UTF-8\n");
        return 1;
    }
    auto reader = GgufReader::open(model_path);
    if (!reader.ok()) {
        std::fprintf(stderr, "example_qwen: cannot open model: %s (%s)\n",
                     reader.status.message.c_str(), reader.status.detail.c_str());
        return 2;
    }
    // A dsl-w4 run reads its weights from an offline pack; the pack's source identity
// is verified against the checkpoint below (size immediately, the rolling payload
// digest once the model inventory exists).
luisa::filesystem::path w4_pack_dir;
luisa::optional<W4PackManifest> w4_pack_manifest;
// Q8_0 gate. dsl-q8 loads a Q8_0 checkpoint NATIVELY; every other variant
    // refuses it with a clear message before any device work (the uploader also
    // fails closed). A dsl-q8 run on a non-Q8_0 checkpoint is refused too: the
    // variant never silently substitutes one weight format for another.
    {
        uint64_t q8_tensors = 0;
        luisa::string first_q8;
        for (const auto &t : (*reader.value)->tensors()) {
            if (t.type == static_cast<uint32_t>(GgufTensorType::q8_0)) {
                if (q8_tensors == 0u) { first_q8 = t.name; }
                ++q8_tensors;
            }
        }
        if (!spec->q8_0 && q8_tensors != 0u) {
            std::fprintf(stderr,
                         "example_qwen: '%s' is Q8_0 (INT8 block quantization). Q8_0 is NOT FP8; "
                         "use --version dsl-q8 to load it natively, or the BF16 GGUF for the "
                         "bf16/fp8 variants (fp8 quantizes BF16 at load time).\n",
                         first_q8.c_str());
            return 2;
        }
                  if (spec->w4 && args.have_w4_pack) {
 // Read the pack now (cheap, fails before any device work) and check the recorded
 // source SIZE. The rolling payload digest is verified right after the model
 // inventory exists, because it is defined over exactly that inventory: a
 // same-size checkpoint with different weights is refused there (review A1).
 if (!luisa::path_from_utf8(args.w4_pack, w4_pack_dir)) {
  std::fprintf(stderr, "example_qwen: --w4-pack path is not valid UTF-8\n");
  return 2;
 }
 auto manifest = read_w4_pack_manifest(w4_pack_dir);
 if (!manifest.ok()) {
  std::fprintf(stderr, "example_qwen: --w4-pack is not a usable pack: %s (%s)\n",
   manifest.status.message.c_str(), manifest.status.detail.c_str());
  return 2;
 }
 if (manifest.value->source_size != (*reader.value)->file_size()) {
  std::fprintf(stderr,
   "example_qwen: W4 pack was built from a different checkpoint "
   "(source size %llu vs %llu bytes): refusing to mix weights\n",
   static_cast<unsigned long long>(manifest.value->source_size),
   static_cast<unsigned long long>((*reader.value)->file_size()));
  return 2;
 }
 w4_pack_manifest.emplace(std::move(*manifest.value));
}
if (spec->w4 && !args.have_w4_pack) {
              std::fprintf(stderr,
                           "example_qwen: --version dsl-w4 requires --w4-pack <dir>: W4 weights are "
                           "ONLY read from an offline pack (the clipping search takes minutes and "
                           "never runs inside a request). Build one with "
                           "--export-w4-pack <dir> --model <bf16.gguf>.\n");
              return 2;
          }
          if (!spec->w4 && args.have_w4_pack) {
              std::fprintf(stderr,
                           "example_qwen: --w4-pack is only valid with --version dsl-w4 (the pack is "
                           "W4A16 storage; no variant silently substitutes it)\n");
              return 2;
          }
          if (spec->w4 && q8_tensors != 0u) {
              std::fprintf(stderr,
                           "example_qwen: --version dsl-w4 needs the BF16 checkpoint the pack was "
                           "derived from (the pack loader compares the source size); "
                           "'%s' is Q8_0\n",
                           args.model.c_str());
              return 2;
          }
          if (spec->q8_0 && q8_tensors == 0u) {
            std::fprintf(stderr,
                         "example_qwen: --version dsl-q8 requires a Q8_0 GGUF checkpoint "
                         "(no Q8_0 tensor found in '%s'; the BF16 checkpoint is for "
                         "dsl-bf16/dsl-fp8).\n",
                         args.model.c_str());
            return 2;
        }
    }
    if (args.mtp) {
        const auto eligibility = probe_mtp_eligibility(**reader.value);
        if (!eligibility.benchmark_feasible) {
            std::fprintf(stderr, "example_qwen: --mtp refused: %s\n", eligibility.reason.c_str());
            return 2;
        }
    }
    auto model = args.mtp ? Qwen35Model::load_with_nextn(**reader.value)
                          : Qwen35Model::load(**reader.value);
    if (!model.ok()) {
        std::fprintf(stderr, "example_qwen: model validation failed: %s (%s)\n",
                     model.status.message.c_str(), model.status.detail.c_str());
        return 2;
    }
    if (w4_pack_manifest.has_value()) {
  // Same-size/different-payload checkpoint => the rolling digest differs and this
  // refuses BEFORE a single weight is uploaded. A size comparison alone cannot
  // catch that (phase 8b review A1).
  if (auto s = verify_w4_pack_source(**reader.value, *model.value, *w4_pack_manifest); !s.ok()) {
   std::fprintf(stderr, "example_qwen: %s: %s\n", s.message.c_str(), s.detail.c_str());
   return 2;
  }
  const auto &m = *w4_pack_manifest;
  std::printf("w4 pack: %s source=%s size=%llu hash=%s (verified against this checkpoint) "
   "formats=%s clip=(group=%u min=%g max=%g steps=%u) importance=%s\n",
   args.w4_pack.c_str(), m.source_name.c_str(),
   static_cast<unsigned long long>(m.source_size),
   qwen_pack_hash_string(m.source_hash).c_str(), m.format_order.c_str(),
   m.clip.group, static_cast<double>(m.clip.min_ratio),
   static_cast<double>(m.clip.max_ratio), m.clip.steps,
   m.has_importance
    ? luisa::format("tokens={} source={} payload={}", m.importance_tokens,
       qwen_pack_hash_string(m.importance_source_hash),
       qwen_pack_hash_string(m.importance_hash)).c_str()
      : "none (clipping only)");
   }
   // ---------------------------------------------------------------------------
   // OFFLINE merged input projections (opt_plan/03 section 3 + D5).
   // The pack is read and verified HERE, before any device work:
   //  * the manifest must describe the canonical merged inventory of THIS model
   //    (segment names/roles/rows/order, K, N, format, group) -- validate_...;
   //  * its SOURCE identity must match this checkpoint: byte size AND the rolling
   //    digest of exactly the merged source tensors (a same-size / different-weights
   //    file is refused), which is the same rule the W4 pack uses;
   //  * every payload's byte size and FNV-1a 64 hash is re-verified while uploading
   //    (QwenDeviceWeights::upload -> append_merged_projections), so a truncated or
   //    edited payload file never reaches the device.
   // A W4 session refuses a pack: there is no 4-bit merged realization, and the W4
   // pack already carries its own dual payloads.
   // ---------------------------------------------------------------------------
   luisa::filesystem::path projection_pack_dir;
   luisa::optional<ProjectionPackManifest> projection_pack_manifest;
   if (args.have_projection_pack) {
       if (spec->w4 || args.have_w4_pack) {
           std::fprintf(stderr,
                        "example_qwen: --projection-pack is not available for the W4 variant "
                        "(no 4-bit merged projection realization; the W4 pack keeps its own "
                        "dual payloads)\n");
           return 2;
       }
       if (spec->kernel != QwenKernelKind::dsl) {
           std::fprintf(stderr,
                        "example_qwen: --projection-pack applies to the DSL executor only\n");
           return 2;
       }
       if (!luisa::path_from_utf8(args.projection_pack, projection_pack_dir)) {
           std::fprintf(stderr, "example_qwen: --projection-pack path is not valid UTF-8\n");
           return 2;
       }
       auto manifest = read_projection_pack_manifest(projection_pack_dir);
       if (!manifest.ok()) {
           std::fprintf(stderr, "example_qwen: --projection-pack is not a usable pack: %s (%s)\n",
                        manifest.status.message.c_str(), manifest.status.detail.c_str());
           return 2;
       }
       auto inventory = qwen35_merged_projection_inventory(*model.value);
       if (!inventory.ok()) {
           std::fprintf(stderr, "example_qwen: merged projection inventory: %s (%s)\n",
                        inventory.status.message.c_str(), inventory.status.detail.c_str());
           return 2;
       }
       if (auto s = validate_projection_pack_manifest(*manifest.value, *inventory.value); !s.ok()) {
           std::fprintf(stderr, "example_qwen: projection pack geometry: %s (%s)\n",
                        s.message.c_str(), s.detail.c_str());
           return 2;
       }
       const auto want = spec->q8_0 ? WeightStorage::q8_0_packed
                         : spec->fp8 ? WeightStorage::fp8_packed
                                     : WeightStorage::bf16_packed;
       if (manifest.value->storage != want) {
           std::fprintf(stderr,
                        "example_qwen: the projection pack was built in format %s but this variant "
                        "uploads %s (build a matching pack with --export-projection-format; a "
                        "merged matrix is never re-quantized)\n",
                        weight_storage_name(manifest.value->storage).data(),
                        weight_storage_name(want).data());
           return 2;
       }
       const auto t_pack = std::chrono::steady_clock::now();
       if (auto s = verify_projection_pack_source(**reader.value, *model.value,
                                                  *manifest.value);
           !s.ok()) {
           std::fprintf(stderr, "example_qwen: %s: %s\n", s.message.c_str(), s.detail.c_str());
           return 2;
       }
       std::printf("projection pack: %s source=%s size=%llu hash=%s format=%s matrices=%zu "
                   "(source digest verified in %.1f ms)\n",
                   args.projection_pack.c_str(), manifest.value->source_name.c_str(),
                   static_cast<unsigned long long>(manifest.value->source_size),
                   qwen_pack_hash_string(manifest.value->source_hash).c_str(),
                   manifest.value->format_order.c_str(), manifest.value->matrices.size(),
                   ms_since(t_pack));
       projection_pack_manifest.emplace(std::move(*manifest.value));
   }
 auto tokenizer = QwenTokenizer::load(**reader.value, (*model.value).config().geometry.vocab);
    if (!tokenizer.ok()) {
        std::fprintf(stderr, "example_qwen: tokenizer load failed: %s (%s)\n",
                     tokenizer.status.message.c_str(), tokenizer.status.detail.c_str());
        return 2;
    }
    const auto &tok = **tokenizer.value;
    std::printf("tokenizer: vocab=%llu merges=%llu specials=%llu eos=%u (%s) pre=%s [exact qwen35 BPE]\n",
                static_cast<unsigned long long>(tok.vocab_size()),
                static_cast<unsigned long long>(tok.info().merge_count),
                static_cast<unsigned long long>(tok.info().special_token_count),
                tok.eos_id(), tok.id_to_token(tok.eos_id()).value_or("?").c_str(),
                tok.info().pre.c_str());

    // --- prompt ids -------------------------------------------------------------
    const auto vocab_size = tok.vocab_size();
    luisa::vector<uint32_t> prompt_ids;
    luisa::string prompt_origin;
    if (args.have_tokens) {
        prompt_ids = args.tokens;
        prompt_origin = "--tokens";
    } else if (args.have_tokens_file) {
        luisa::string error;
        if (!read_tokens_file(args.tokens_file, prompt_ids, error)) {
            std::fprintf(stderr, "example_qwen: --tokens-file '%s': %s\n",
                         args.tokens_file.c_str(), error.c_str());
            return 1;
        }
        prompt_origin = luisa::format("--tokens-file '{}'", args.tokens_file);
    } else if (args.have_synthetic_prompt) {
        prompt_ids = qwen_synthetic_prompt_ids(args.synthetic_prompt_len);
        if (prompt_ids.size() != args.synthetic_prompt_len) {
            std::fprintf(stderr,
                         "example_qwen: --synthetic-prompt-len %u produced %zu ids\n",
                         args.synthetic_prompt_len, prompt_ids.size());
            return 1;
        }
        prompt_origin = luisa::format("--synthetic-prompt-len {}", args.synthetic_prompt_len);
    } else if (args.have_prompt_file) {
        // Binary mode: no newline translation, so the byte count the gate recorded is
        // the byte count that was tokenized.
        std::FILE *pf = std::fopen(args.prompt_file.c_str(), "rb");
        if (pf == nullptr) {
            std::fprintf(stderr, "example_qwen: --prompt-file '%s' cannot be opened\n",
                         args.prompt_file.c_str());
            return 1;
        }
        luisa::string text;
        char buffer[65536];
        size_t got = 0u;
        while ((got = std::fread(buffer, 1u, sizeof(buffer), pf)) > 0u) {
            text.append(buffer, got);
            if (text.size() > (64u << 20)) { break; }// bounded: 64 MiB of text
        }
        std::fclose(pf);
        if (text.empty()) {
            std::fprintf(stderr, "example_qwen: --prompt-file '%s' is empty\n",
                         args.prompt_file.c_str());
            return 1;
        }
        if (args.chat) {
            auto ids = tok.chat_prompt_ids(text);
            if (!ids.ok()) {
                std::fprintf(stderr, "example_qwen: chat prompt rejected: %s\n",
                             ids.status.message.c_str());
                return 1;
            }
            prompt_ids = std::move(*ids.value);
            prompt_origin = luisa::format("--prompt-file '{}' [chat wrapper]", args.prompt_file);
        } else {
            auto ids = tok.encode(text, false);
            if (!ids.ok()) {
                std::fprintf(stderr, "example_qwen: prompt encode failed: %s\n",
                             ids.status.message.c_str());
                return 1;
            }
            prompt_ids = std::move(*ids.value);
            prompt_origin = luisa::format("--prompt-file '{}' ({} bytes)", args.prompt_file, text.size());
        }
    } else if (args.chat) {
        auto ids = tok.chat_prompt_ids(args.prompt);
        if (!ids.ok()) {
            std::fprintf(stderr, "example_qwen: chat prompt rejected: %s\n", ids.status.message.c_str());
            return 1;
        }
        prompt_ids = std::move(*ids.value);
        prompt_origin = "--prompt [chat wrapper]";
    } else {
        auto ids = tok.encode(args.prompt, false);
        if (!ids.ok()) {
            std::fprintf(stderr, "example_qwen: prompt encode failed: %s\n", ids.status.message.c_str());
            return 1;
        }
        prompt_ids = std::move(*ids.value);
        prompt_origin = "--prompt";
    }
    if (prompt_ids.empty()) {
        std::fprintf(stderr, "example_qwen: empty prompt (source %s)\n",
                     prompt_origin.c_str());
        return 1;
    }
    // Every id is validated against the loaded vocabulary; the index is printed
    // because a bad id in an 8192-token file is otherwise impossible to find.
    for (size_t i = 0u; i < prompt_ids.size(); ++i) {
        if (prompt_ids[i] >= vocab_size) {
            std::fprintf(stderr,
                         "example_qwen: prompt token %zu has id %u, outside the loaded "
                         "vocabulary (%llu ids); source %s\n",
                         i, prompt_ids[i],
                         static_cast<unsigned long long>(vocab_size),
                         prompt_origin.c_str());
            return 1;
        }
    }
    if (prompt_ids.size() + args.max_new_tokens > max_context) {
        std::fprintf(stderr,
                     "example_qwen: prompt (%zu ids from %s) + max-new-tokens (%u) exceeds "
                     "max-context (%u); raise --max-context (<= 262144) or shorten the prompt\n",
                     prompt_ids.size(), prompt_origin.c_str(),
                     args.max_new_tokens, max_context);
        return 1;
    }
    {
        std::printf("prompt: %zu tokens%s (source: %s)\n", prompt_ids.size(),
                    args.chat ? " [chat wrapper]" : "", prompt_origin.c_str());
        // A long id list would otherwise flood the log (and the decoded text is
        // meaningless for synthetic ids), so only a short prompt is printed in
        // full. The count above is what the harness parses.
        constexpr size_t kMaxPrintedIds = 64u;
        if (prompt_ids.size() <= kMaxPrintedIds) {
            auto prompt_text = tok.decode(prompt_ids, false, true);
            if (prompt_text.ok()) { std::printf("prompt text: %s\n", (*prompt_text.value).c_str()); }
            std::printf("prompt ids:");
            for (auto id : prompt_ids) { std::printf(" %u", id); }
            std::printf("\n");
        } else {
            std::printf("prompt ids (first 8):");
            for (size_t i = 0u; i < 8u; ++i) { std::printf(" %u", prompt_ids[i]); }
            std::printf(" ... (last 2):");
            for (size_t i = prompt_ids.size() - 2u; i < prompt_ids.size(); ++i) {
                std::printf(" %u", prompt_ids[i]);
            }
            std::printf(" [truncated: %zu ids]\n", prompt_ids.size());
        }
    }
    const double host_load_ms = ms_since(t_open);

    // --- device / stream / weights / engine --------------------------------------
    // Destruction order is explicit below: engine -> weights -> stream -> device
    // -> context. Every engine/weight buffer must be released while its device
    // is still alive (backend teardown crashes otherwise).
    compute::Context context{argv0_holder()};
      // PDL must be requested BEFORE the device exists (see QwenPdlDeviceConfig); any
      // other backend has no such extension, so the request is reported and ignored
      // instead of being silently dropped.
      compute::DeviceConfig device_config;
      if (!args.pdl) {
          // default: no config extension, exactly the previous creation path
      } else if (args.backend == "cuda") {
          device_config.extension = luisa::make_unique<QwenPdlDeviceConfig>();
      } else {
          std::printf("pdl: requested=%d backend=%s (PDL is a CUDA launch property; ignored on this backend)\n",
                      1, args.backend.c_str());
      }
      auto device = context.create_device(args.backend.c_str(),
                                          device_config.extension ? &device_config : nullptr);
    if (!device) {
        std::fprintf(stderr, "example_qwen: backend '%s' not available (installed:", args.backend.c_str());
        for (const auto &b : context.installed_backends()) { std::fprintf(stderr, " %s", b.c_str()); }
        std::fprintf(stderr, ")\n");
        return 2;
    }
    auto stream = device.create_stream();
    if (!stream) {
        std::fprintf(stderr, "example_qwen: cannot create stream\n");
        return 2;
    }

    int exit_code = 0;
    {
        const auto t_upload = std::chrono::steady_clock::now();
 QwenWeightUploadOptions up;
 up.fp8 = spec->fp8;
 up.q8_0 = spec->q8_0;
 // OFFLINE merged input projections (opt_plan/03 section 3 + D5): hand the verified
 // pack to the uploader, which then SKIPS the source tensors of every merged segment
 // and uploads the merged matrix (plus one sub-view alias per segment) from the
 // pack payload. Nothing is uploaded twice.
 QwenProjectionPackSource projection_pack_source;
 if (projection_pack_manifest.has_value()) {
     projection_pack_source.dir = projection_pack_dir;
     projection_pack_source.manifest = &*projection_pack_manifest;
     up.projection_pack = &projection_pack_source;
 }
 QwenWeightUploadReport report;
          Result<QwenDeviceWeights> weights =
              Result<QwenDeviceWeights>::failure(StatusCode::internal_error, "unset");
          if (spec->w4) {
              // Plan option (c) default: the pack is the ONLY weight source; its
              // dual payloads give W4 decode and FP8 prefill/LM head. The GGUF is
              // the tokenizer + model contract (already validated above).
              luisa::filesystem::path pack_dir;
              if (!luisa::path_from_utf8(args.w4_pack, pack_dir)) {
                  std::fprintf(stderr, "example_qwen: --w4-pack path is not valid UTF-8\n");
                  return 2;
              }
              weights = upload_w4_pack(device, stream, pack_dir, 16ull * 1024ull * 1024ull, &report);
          } else {
              weights = QwenDeviceWeights::upload(device, stream, **reader.value, *model.value, up, &report);
          }
        if (!weights.ok()) {
            std::fprintf(stderr, "example_qwen: weight upload failed: %s (%s)\n",
                         weights.status.message.c_str(), weights.status.detail.c_str());
            return 2;
        }
        luisa::optional<QwenDeviceWeights> draft_weights;
        QwenWeightUploadReport draft_report;
        if (args.mtp) {
            auto uploaded = QwenDeviceWeights::upload_nextn(device, stream, **reader.value, up, &draft_report);
            if (!uploaded.ok()) {
                std::fprintf(stderr, "example_qwen: NextN weight upload failed: %s (%s)\n",
                             uploaded.status.message.c_str(), uploaded.status.detail.c_str());
                return 2;
            }
            draft_weights.emplace(std::move(*uploaded.value));
        }
        stream.synchronize();
        const double upload_ms = ms_since(t_upload);
                  std::printf("weights: %llu tensors (f32=%llu bf16=%llu fp8=%llu q8_0=%llu w4=%llu), "
                      "%.2f MiB device, upload %.1f ms\n",
                      static_cast<unsigned long long>(report.tensor_count),
                      static_cast<unsigned long long>(report.f32_count),
                      static_cast<unsigned long long>(report.bf16_count),
                      static_cast<unsigned long long>(report.fp8_count),
                      static_cast<unsigned long long>(report.q8_0_count),
                      static_cast<unsigned long long>(report.w4_count),
                      static_cast<double>(report.device_bytes) / (1024.0 * 1024.0), upload_ms);
          if (spec->w4) {
              std::printf("w4 storage: %llu matrices are 8-code int4 words + one FP32 scale per "
                          "group (deterministic clipping search); %llu FP8 second copies feed "
                          "prefill and the LM head (plan option %s)\n",
                          static_cast<unsigned long long>(report.w4_count),
                          static_cast<unsigned long long>(report.fp8_count),
                          args.w4_prefill_in_kernel ? "(b) A/B arm" : "(c) W4 decode + FP8 prefill");
          }
        if (spec->q8_0) {
            std::printf("q8_0 storage: %llu matrices are the checkpoint's own int8 block codes "
                        "(byte copy) with exact FP16->FP32 block scales (one per 32 K elements "
                        "per row); no conversion, no re-quantization\n",
                        static_cast<unsigned long long>(report.q8_0_count));
        }
        if (spec->fp8) {
            std::printf("fp8 conversion: %zu matrices quantized BF16->E4M3 at load time "
                        "(per-row groups of 64, FP32 scales)\n",
                        report.diagnostics.size());
        }

        QwenEngineOptions options;
        options.kernel = spec->kernel;
        options.max_context = max_context;
        options.prefill_chunk = prefill_chunk;
 // An explicit --prefill-chunk keeps the historical FIXED chunk (A/B and the
 // existing scripts); without it the engine applies the adaptive rule of
 // opt_plan/02 section 4, capped by --prefill-chunk-max.
 options.prefill_chunk_adaptive = args.prefill_chunk == 0u;
 options.prefill_chunk_max = args.prefill_chunk_max;
    options.dsl_gdn_wy_ut = !args.gdn_recurrent;
    options.dsl_tiled_prefill_attn = !args.legacy_prefill_attn;
 options.dsl_prefill_gemm_mt = !args.legacy_prefill_gemm;
 if (args.have_native_tensor_gemm) { options.native_tensor_gemm = args.native_tensor_gemm; }
 if (args.have_native_tensor_attention) { options.native_tensor_attention = args.native_tensor_attention; }
 // EXPERIMENTAL CUDA native-Tile typed-FP8 prefill GEMM: an explicit REQUEST that
 // the engine fails closed on (missing LUISA_CUDA_TILE_IR, non-cuda device, non-FP8
 // prefill weights, invalid tiling, tileiras/NVRTC refusal). Never inferred: without
 // --cuda-tile-fp8 the option stays false and nothing is probed or compiled.
 if (args.cuda_tile_fp8) {
     options.cuda_tile_fp8_prefill = true;
     options.cuda_tile_m = args.cuda_tile_m;
     options.cuda_tile_n = args.cuda_tile_n;
 }
    options.kv_page_bytes = args.kv_page_bytes;
    options.kv_dtype = args.kv_dtype;
    options.dsl_cuda_graph = args.cuda_graph;
    // CONDITIONAL WHILE chunk (opt_plan/09 N4.2). A REQUEST only: the engine gates it
    // on the CudaGraphExt conditional-node capability AND its own instantiate/execute
    // probe, and falls back to the per-token static graph when either refuses.
    // --graph-flight is the same thing with the --decode-flight chunk size.
    options.dsl_cuda_graph_while = args.cuda_graph_while != 0u || args.graph_flight;
    options.dsl_cuda_graph_while_tokens =
        args.cuda_graph_while != 0u ? args.cuda_graph_while : args.decode_flight;
    // The EOS id is MANDATORY for the chunked path (the body's termination needs it);
    // the tokenizer is the only authority for it.
    options.dsl_eos_token_id = static_cast<uint32_t>(tok.eos_id());
    options.dsl_pinned_transfers = args.pinned_transfers;
    options.dsl_pdl = args.pdl;
    options.dsl_split_kv = args.split_kv;
    options.dsl_split_kv_threshold = args.split_kv_threshold;
    options.dsl_split_kv_max_splits = args.split_kv_max_splits;
    options.dsl_gdn_decode_fused = args.gdn_decode_fused;
    options.dsl_gdn_state_specialized = args.gdn_state_specialized;
    // FUSED one-CTA recurrence + gated RMSNorm (opt_plan/03 section 5): a request
    // only -- dsl_decode_specialized, the model geometry and a successful compile
    // decide, and info() reports the outcome and the reason.
    options.dsl_gdn_recurrent_norm = args.gdn_fused_recurrent;
    options.dsl_rmsnorm_reg = args.rmsnorm_reg;

    options.dsl_decode_packed = args.decode_packed;
 // OFFLINE merged input projections (opt_plan/03 section 3): a request only. Without
 // a pack it changes nothing; with a pack, create() validates the descriptors and
 // enables the merged decode GEMVs unless the format/realization rules refuse them.
 options.dsl_merged_decode = args.merged_decode;
    options.dsl_decode_packed_fp8 = args.decode_packed_fp8;
    options.dsl_attn_prepare_model = args.attn_prepare_model;
    // An explicit --gdn-wy-ut pin wins over the per-prompt rule; otherwise auto
    // selects the measured GDN prefill schedule per prompt.
    if (args.have_gdn_wy_ut || args.gdn_recurrent) {
        options.dsl_gdn_wy_ut = args.gdn_wy_ut;
        options.dsl_gdn_auto = false;
    } else {
        options.dsl_gdn_auto = args.gdn_auto;
    }
    options.dsl_gdn_auto_threshold = args.gdn_auto_threshold;
              options.benchmark = args.benchmark || args.have_benchmark_csv;
          options.dsl_w4_prefill_in_kernel = args.w4_prefill_in_kernel;
          options.dsl_w4_lm_head_8bit = args.w4_lm_head_8bit;
        const auto t_compile = std::chrono::steady_clock::now();
        QwenEnginePtr engine;
        QwenMtpExecutor::Ptr mtp;
        QwenEngine *engine_ptr = nullptr;
        if (args.mtp) {
            auto created = QwenMtpExecutor::create(device, stream, *model.value, *weights.value,
                                                   *draft_weights, options, args.draft_depth);
            if (!created.ok()) {
                std::fprintf(stderr, "example_qwen: MTP engine creation failed: %s (%s)\n",
                             created.status.message.c_str(), created.status.detail.c_str());
                return 2;
            }
            mtp = std::move(*created.value);
            engine_ptr = mtp->target();
            std::printf("mtp: draft_depth=%u history=%.2f MiB snapshots=%.2f MiB verify-logits=%.2f MiB\n",
                        args.draft_depth,
                        static_cast<double>(mtp->info().history_bytes) / (1024.0 * 1024.0),
                        static_cast<double>(mtp->info().snapshot_bytes) / (1024.0 * 1024.0),
                        static_cast<double>(mtp->info().verify_logits_bytes) / (1024.0 * 1024.0));
          } else {
              // The Tile engine's schedule knobs live in QwenTileEngineOptions,
              // so a tile-* version is created directly (never through a silent
              // conversion of the DSL options).
              QwenEnginePtr created_engine;
              Status created_status;
              if (spec->kernel == QwenKernelKind::tile) {
                  QwenTileEngineOptions topt;
                  topt.base = options;
                  topt.chunk_conv = args.tile_chunk_conv;
                  if (args.have_tile_attn_bucket) { topt.attention_bucket_keys = args.tile_attn_bucket; }
                  auto created = QwenTileEngine::create(device, stream, *model.value, *weights.value, topt);
                  if (!created.ok()) {
                      created_status = created.status;
                  } else {
                      created_engine = std::move(*created.value);
                  }
              } else {
                  auto created = QwenEngine::create(device, stream, *model.value, *weights.value, options);
                  if (!created.ok()) {
                      created_status = created.status;
                  } else {
                      created_engine = std::move(*created.value);
                  }
              }
              if (!created_status.ok()) {
                  std::fprintf(stderr, "example_qwen: engine creation failed: %s (%s)\n",
                               created_status.message.c_str(), created_status.detail.c_str());
                  return 2;
              }
              engine = std::move(created_engine);
              engine_ptr = engine.get();
          }
        const double compile_ms = ms_since(t_compile);
        {
            const auto info = engine_ptr->info();
            std::printf("engine: kernel=%s storage=%s max_context=%u prefill_chunk=%u gdn=%s prefill_attn=%s "
                        "cache=%.2f MiB workspace=%.2f MiB shaders=%llu (compile %.1f ms)\n",
                        luisa::string{qwen_kernel_kind_name(info.kernel)}.c_str(),
                        luisa::string{weight_storage_name(info.weight_storage)}.c_str(),
                        info.max_context, info.prefill_chunk,
                        info.kernel == QwenKernelKind::dsl ? (info.gdn_wy_ut ? "wy-ut-64" : "recurrent") : "recurrent-tile",
                        info.kernel != QwenKernelKind::dsl ? "tile-schedule" :
                            (info.native_prefill_attn ?
                                   luisa::format("tensor-core(q{}kv{},{})",
                                                 info.native_prefill_attn_q_tile,
                                                 info.native_prefill_attn_kv_tile,
                                                 info.native_prefill_attn_shared_bytes)
                                       .c_str() :
                                   (info.prefill_attn_tiled ?
                                 luisa::format("tiled(q{}k{},{}B)", info.prefill_attn_q_tile,
                                               info.prefill_attn_k_tile, info.prefill_attn_shared_bytes)
                                     .c_str() :
                                 "legacy")),
                        static_cast<double>(info.cache_bytes) / (1024.0 * 1024.0),
                        static_cast<double>(info.workspace_bytes) / (1024.0 * 1024.0),
                        static_cast<unsigned long long>(info.compiled_shader_count), compile_ms);
            // Prefill projection/attention realization actually compiled, plus the
 // workspace split (chunk-scaled pool vs fixed buffers). One line each so a
 // benchmark log is self-describing.
 std::printf("prefill gemm: %s native=%s\n",
 info.prefill_gemm_mt
 ? luisa::format("mt(mt={})", info.prefill_gemm_mt_rows).c_str()
 : "legacy",
 info.native_prefill_gemm
     ? luisa::format("tensor_core({}x{}x{}w{}, rows>={}, cold_jit={:.1f}ms, launches={})",
                     info.native_prefill_gemm_block_m,
                     info.native_prefill_gemm_block_n,
                     info.native_prefill_gemm_block_k,
                     info.native_prefill_gemm_block_m / 32u,
                     info.native_prefill_gemm_min_rows,
                     info.native_prefill_gemm_jit_ms,
                     static_cast<unsigned long long>(info.native_prefill_gemm_launches))
           .c_str()
     : "off");
 // EXPERIMENTAL CUDA native-Tile typed-FP8 prefill GEMM (--cuda-tile-fp8). A line of
 // its own because the `prefill gemm:` line above is parsed by
 // scripts/benchmark_qwen.py with an anchored regex; this one is only emitted by an
 // engine that was CREATED with the option and passed its whole create-time gate, so
 // a default run prints `off` and can never be read as the Tile realization engaging.
 // `lazy_jit` is the honest price of the path: one Tile IR + tileiras compile per
 // static (m, n, k, epilogue) shape, counted and timed by the helper.
 std::printf("prefill cuda tile: %s\n",
             info.cuda_tile_prefill_gemm
                 ? luisa::format("native_tile_fp8({}x{}x{}, rows>={}, launches={}, "
                                 "lazy_compiles={}, cache_hits={}, shaders={}, lazy_jit={:.1f}ms, "
                                 "probe={:.1f}ms, ragged_n={})",
                                 info.cuda_tile_prefill_gemm_m,
                                 info.cuda_tile_prefill_gemm_n,
                                 info.cuda_tile_prefill_gemm_k,
                                 info.cuda_tile_prefill_gemm_min_rows,
                                 static_cast<unsigned long long>(info.cuda_tile_prefill_gemm_launches),
                                 info.cuda_tile_prefill_gemm_compiles,
                                 info.cuda_tile_prefill_gemm_cache_hits,
                                 info.cuda_tile_prefill_gemm_shaders,
                                 info.cuda_tile_prefill_gemm_jit_ms,
                                 info.cuda_tile_prefill_gemm_probe_ms,
                                 info.cuda_tile_prefill_gemm_ragged_n ? "supported" : "refused")
                       .c_str()
                 : "off (experimental, opt-in: --cuda-tile-fp8)");
 std::printf("prefill attn: %s native=%s\n",
               info.prefill_attn_tiled
                   ? luisa::format("tiled(q={},k={},shared={}B)", info.prefill_attn_q_tile,
                                   info.prefill_attn_k_tile, info.prefill_attn_shared_bytes)
                         .c_str()
                   : "reference",
               info.native_prefill_attn
                   ? luisa::format("tensor_core(q={},kv={},w{},static_smem={}B,rows>={},"
                                   "cold_jit={:.1f}ms,launches={})",
                                   info.native_prefill_attn_q_tile,
                                   info.native_prefill_attn_kv_tile,
                                   info.native_prefill_attn_warps,
                                   info.native_prefill_attn_shared_bytes,
                                   info.native_prefill_attn_min_rows,
                                   info.native_prefill_attn_jit_ms,
                                   static_cast<unsigned long long>(info.native_prefill_attn_launches))
                         .c_str()
                   : "off");
 std::printf("workspace: total=%.2f MiB chunk=%.2f MiB pool_per_row=%llu floats\n",
 static_cast<double>(info.workspace_bytes) / (1024.0 * 1024.0),
 static_cast<double>(info.workspace_chunk_bytes) / (1024.0 * 1024.0),
 static_cast<unsigned long long>(info.workspace_pool_floats_per_row));
              // Model-path specializations actually compiled (opt_plan/09 V4 and
              // N3.1/N3.3). One line so a benchmark log states which realization ran.

              std::printf("model specializations: rmsnorm=%s gdn_state=%s "
                          "gdn_decode_fused=%s gdn_recurrent_norm=%s "
                          "decode_packed=%s attn_prepare=%s gdn_prefill=%s\n",
                          info.rmsnorm_reg ? "reg" : "generic",
                          info.gdn_state_specialized ? "compile_time_hd128" : "generic",
                          info.gdn_decode_fused ? "fused" : "pair",
                          info.gdn_recurrent_norm ? "fused" : "pair",
                          info.decode_packed ? "128bit" : "32bit",
                          info.attn_prepare_model ? "model_hd256" : "generic",
                          info.gdn_auto
                              ? luisa::format("auto<={}_recurrent", info.gdn_auto_threshold).c_str()
                              : (info.gdn_wy_ut ? "wy_ut" : "recurrent"));
                // Fused recurrent+norm selection/refusal text: a log must state WHY the
                // one-CTA fusion did not run when it was requested.
                if (!info.gdn_recurrent_norm_reason.empty()) {
                    std::printf("gdn recurrent+norm: %s (%s)\n",
                                info.gdn_recurrent_norm ? "fused_one_cta" : "separate_pair",
                                info.gdn_recurrent_norm_reason.c_str());
                }
    // OFFLINE merged input projections (opt_plan/03 section 3 + D5): the SELECTION
   // this session runs, the segment descriptors that back it and the launch
   // reduction it buys at rows == 1. Printed even when no pack was used, so a log
   // always states which route ran instead of leaving it implicit.
   if (info.merged_available) {
       std::printf("merged projections: %s matrices=%llu segments=%llu gdn_n=%u attn_n=%u "
                   "launches_saved_per_token=%u format=%s source=%s\n",
                   info.merged_decode ? "decode_gemv" : "unmerged(A/B or refused)",
                   static_cast<unsigned long long>(info.merged_matrices),
                   static_cast<unsigned long long>(info.merged_segments),
                   info.merged_gdn_n, info.merged_attn_n, info.merged_launches_saved,
                   info.projection_pack_format.c_str(), info.projection_pack_source.c_str());
   } else {
       std::printf("merged projections: off (no --projection-pack: every projection is "
                   "uploaded and launched per matrix)\n");
   }
   // KV page geometry and the one-line capability summary: a captured
            // benchmark log then states both the allocation unit actually used
            // and which accelerator fast paths this device could take.
            std::printf("kv cache: page=%u B tiles/page=%u pages/region=%u heap-slots=%u "
                        "(sparse tile %u B, sparse %s)\n",
                        info.caps.kv_page_bytes, info.caps.kv_tiles_per_page,
                        info.caps.kv_pages_per_region, info.caps.kv_heap_slots,
                        info.caps.sparse_tile_bytes, info.caps.sparse_buffer ? "yes" : "no");
    // KV dtype/layout contract (opt_plan/05 Q3): what was BUILT, not what was
    // requested, plus the byte model a capacity comparison needs. bf16 stores two
    // RNE codes per word in the 64-position blocked layout, fp8 stores four E4M3
    // codes per word plus one dense FP32 scale per (kv_head, 64-position block),
    // so the payload is half / a quarter of the FP32 number per layer.
    std::printf("kv dtype: %s layout=%s codes/word=%u payload=%.2f MiB region-words=%llu pages/region=%u first-token-pages=%u packed-append=%s scaled-append=%s\n",
                luisa::string{qwen_kv_dtype_name(info.kv_dtype)}.c_str(),
                luisa::string{qwen_kv_layout_name(info.kv_layout)}.c_str(),
                info.kv_elements_per_word,
                static_cast<double>(info.kv_payload_bytes) / 1048576.0,
                static_cast<unsigned long long>(info.kv_region_words),
                info.kv_pages_per_region, info.kv_min_commit_pages,
                info.kv_packed_append ? "yes" : "no",
                info.kv_scaled_append ? "yes" : "no");
    if (info.kv_scales_per_region != 0u) {
        // fp8 capacity line: the dense scales the payload does NOT include, and
        // what the tier model should charge. The scale array is dense (always
        // mapped), so it is a real allocation, not an estimate.
        std::printf("kv fp8 scales: %u per region ([kv_head][64-position block]), %.1f KiB dense total (payload above excludes them; the FP32 cache of the same geometry is 4x the payload)\n",
                    info.kv_scales_per_region,
                    static_cast<double>(info.kv_scale_bytes) / 1024.0);
    }
            qwen_log_backend_caps(info.caps);
            std::printf("variant status: %s\n", luisa::string{spec->status}.c_str());
        }

        // --- generation ------------------------------------------------------------
        QwenSampleOptions sample_options;
        sample_options.temperature = args.temperature;
        sample_options.seed = args.seed;
        std::printf("sampling: temperature=%g seed=%llu\n", static_cast<double>(args.temperature),
                    static_cast<unsigned long long>(args.seed));
        auto detok = tok.make_detokenizer(true);
        luisa::vector<float> logits;
        luisa::vector<uint32_t> generated;
        const auto t_prefill = std::chrono::steady_clock::now();
                  auto st = args.mtp ? mtp->prefill(stream, prompt_ids, logits)
                             : engine_ptr->prefill(stream, prompt_ids, logits);
          // The quality gate needs the distribution that picked token 0 too, not
          // just the last step's.
          luisa::vector<float> prefill_logits = logits;
        const double ttft_ms = ms_since(t_prefill);
 // ONE harness-parseable line: configured vs ACTUALLY selected chunking of this
 // prompt (opt_plan/02 section 4). The engine reports the selection, so a log can
 // attribute TTFT to the chunk rule instead of to the requested value.
 {
 const auto chunk_info = engine_ptr->info();
 std::printf("prefill chunking: mode=%s configured_chunk=%u configured_max=%u "
 "backend_limit=%u selected=%u chunks=%u\n",
 chunk_info.prefill_chunk_adaptive ? "adaptive" : "fixed",
 chunk_info.prefill_chunk, chunk_info.prefill_chunk_max,
 chunk_info.prefill_chunk_backend_limit, chunk_info.prefill_chunk_selected,
 chunk_info.prefill_chunks);
 }
        if (!st.ok()) {
            std::fprintf(stderr, "example_qwen: prefill failed: %s (%s)\n",
                         st.message.c_str(), st.detail.c_str());
            exit_code = 2;
        }
        double decode_total_ms = 0.0;
        double argmax_total_ms = 0.0;
        double detok_total_ms = 0.0;
        uint32_t decode_steps = 0;
        bool eos_hit = false;
        bool csv_started = false;
        uint64_t benchmark_iteration = 0;
        if (st.ok() && options.benchmark) {
            st = emit_benchmark(args, *engine_ptr, benchmark_iteration++, csv_started);
            if (!st.ok()) {
                std::fprintf(stderr, "example_qwen: benchmark reporting failed: %s (%s)\n",
                             st.message.c_str(), st.detail.c_str());
                exit_code = 2;
            }
        }
        // Emit one id with the historical EOS/budget ordering: EOS counts but
        // is never decoded; the budget is checked before another forward.
        const auto emit_one = [&](uint32_t id) -> bool {
            generated.emplace_back(id);
            if (id == tok.eos_id()) {
                eos_hit = true;
                return false;
            }
            const auto t_detok = std::chrono::steady_clock::now();
            auto piece = detok.push(id);
            detok_total_ms += ms_since(t_detok);
            if (!piece.ok()) {
                std::fprintf(stderr, "example_qwen: detokenize failed: %s\n", piece.status.message.c_str());
                exit_code = 2;
                return false;
            }
            if (!(*piece.value).empty()) {
                std::fputs((*piece.value).c_str(), stdout);
                std::fflush(stdout);
            }
            return generated.size() < args.max_new_tokens;
        };
        uint32_t pending = 0u;
        bool have_pending = false;
        const bool use_flight = args.decode_flight != 0u && !args.mtp &&
                                    !options.benchmark && args.temperature == 0.0f &&
                                    args.backend == "cuda" &&
                                    engine_ptr->supports_device_greedy_decode();
        // Conditional WHILE chunk (opt_plan/09 N4.2): the caller REQUESTS it, the engine
        // decides (capability + instantiate/execute probe). When it is not active the same
        // chunk is served by per-token launches, so this only changes the launch count.
        const bool want_chunk = (args.cuda_graph_while != 0u || args.graph_flight) && use_flight;
        const uint32_t flight_chunk = args.cuda_graph_while != 0u ? args.cuda_graph_while
                                                                : args.decode_flight;
        // Room left before the engine's context limit, so a chunk is never requested that
        // the engine would have to refuse (the per-token path errors at the limit, which
        // would end the run instead of the generation).
        const auto prefill_info = engine_ptr->info();
        uint32_t context_room = prefill_info.max_context > prefill_info.position
                                    ? prefill_info.max_context - prefill_info.position
                                    : 0u;
        if (use_flight) {
            if (want_chunk) {
                std::printf("decode policy: device-greedy in-flight chunk=%u while-chunk=%u\n",
                            args.decode_flight, flight_chunk);
            } else {
                std::printf("decode policy: device-greedy in-flight chunk=%u (per-token launches)\n",
                            args.decode_flight);
            }
        }
        // Recorded selection for A/B and benchmark logs: the request, whether the
        // engine can host the static graph at all (CudaGraphExt on CUDA), and the
        // format arm. The counters in "engine final:" prove what actually ran.
        {
            const auto graph_info = engine_ptr->info();
            const bool graph_requested = args.cuda_graph && !args.mtp && !options.benchmark;
            const bool graph_active = use_flight && graph_requested && graph_info.caps.cuda_graph;
            std::printf("decode graph: requested=%s active=%s cuda-graph-ext=%d storage=%s\n",
                        graph_requested ? "static" : "off",
                        graph_active ? "static" : (graph_requested ? "unavailable" : "off"),
                        graph_info.caps.cuda_graph ? 1 : 0,
                        luisa::string{weight_storage_name(graph_info.weight_storage)}.c_str());
        }
        // Conditional WHILE chunk / pinned downloads / PDL: what was requested, what the
        // engine selected and WHY (opt_plan/09 N4.2-N4.4). The counters below are zero
        // here by construction; "engine final:" reports what actually ran.
        {
            const auto winfo = engine_ptr->info();
            std::printf("decode while: requested=%d active=%d probe=%d chunk=%u storage=%s reason='%s'\n",
                        (args.cuda_graph_while != 0u || args.graph_flight) ? 1 : 0,
                        winfo.cuda_graph_while ? 1 : 0, winfo.cuda_graph_while_probe ? 1 : 0,
                        winfo.cuda_graph_while_chunk,
                        luisa::string{weight_storage_name(winfo.weight_storage)}.c_str(),
                        winfo.cuda_graph_while_reason.empty()
                            ? "not requested"
                            : winfo.cuda_graph_while_reason.c_str());
            std::printf("pinned downloads: requested=%d extension=%d active=%d ring-tokens=%u reason='%s'\n",
                        args.pinned_transfers ? 1 : 0, winfo.pinned_available ? 1 : 0,
                        winfo.pinned_transfers ? 1 : 0, winfo.pinned_ring_tokens,
                        winfo.pinned_reason.empty() ? "unknown" : winfo.pinned_reason.c_str());
            std::printf("pdl: requested=%d extension=%d supported=%d auto-prologue=%d "
                        "stream-enabled=%d reason='%s'\n",
                        winfo.pdl.requested ? 1 : 0, winfo.pdl.available ? 1 : 0,
                        winfo.pdl.supported ? 1 : 0, winfo.pdl.auto_prologue ? 1 : 0,
                        winfo.pdl.stream_enabled ? 1 : 0,
                        winfo.pdl.reason.empty() ? "unknown" : winfo.pdl.reason.c_str());
        }
        // Split-KV decode attention provenance (opt_plan/03 section 4.2). `requested` is
          // the option, `active` what the engine actually compiled, and the threshold /
          // grid bound / shared-memory footprint are the compile-time constants the
          // benchmark's roof model needs to decide between the 1x deduped KV term and the
          // legacy 4x term. sm is the probed SM count: cuda answers it from the driver
          // (patches/luisa-cuda-multiprocessor-query.patch), dx/vk answer nothing and
          // print sm=0 with source 'unqueryable', which means blocks/SM must be reported
          // as unavailable rather than guessed.
        {
            const auto attn_info = engine_ptr->info();
            // The split family is a DSL engine kernel: the Tile engine keeps its own
    // decode schedule, so requesting it there is not "unavailable", it simply
    // does not apply (the harness then keeps the legacy 4x KV term).
    const bool attn_requested = args.split_kv && !args.mtp &&
                                args.temperature == 0.0f &&
                                attn_info.kernel == QwenKernelKind::dsl;
            // A string_view is not NUL-terminated, so %s needs a real string here.
            const luisa::string sm_source = attn_info.caps.sm_count_source.empty()
                        ? luisa::string{"unqueryable"}
                        : attn_info.caps.sm_count_source;
            std::printf("decode attn: requested=%s active=%s threshold=%u max_splits=%u k_tile=%u "
                        "shared=%u partial_bytes=%u blocks=%u splits=%u kv_len=%u dedup=%d "
                        "sm=%u sm_source=%s\n",
                        attn_requested ? "split" : "legacy",
                        attn_info.decode_attn_split ? "split" : (attn_requested ? "unavailable" : "legacy"),
                        attn_info.decode_attn_threshold, attn_info.decode_attn_max_splits,
                        attn_info.decode_attn_k_tile, attn_info.decode_attn_shared_bytes,
                        attn_info.decode_attn_partial_bytes, attn_info.decode_attn_blocks,
                        attn_info.decode_attn_splits, attn_info.decode_attn_kv_len,
                        attn_info.decode_attn_dedup ? 1 : 0, attn_info.caps.sm_count,
                        sm_source.c_str());
        }
        if (use_flight && exit_code == 0) {
            // The prefill still uses the reference host sampler for token zero.
            // Subsequent selections stay on-device until each bounded chunk.
            const auto t_select = std::chrono::steady_clock::now();
            auto next = qwen_sample(logits, sample_options, 0u);
            argmax_total_ms += ms_since(t_select);
            if (!next.ok()) {
                std::fprintf(stderr, "example_qwen: token selection failed: %s\n",
                             next.status.message.c_str());
                exit_code = 2;
            } else {
                pending = *next.value;
            }
            bool need_host_input = true;
            if (exit_code == 0 && emit_one(pending)) {
                luisa::vector<uint32_t> flight_ids;
                while (exit_code == 0 && !eos_hit && generated.size() < args.max_new_tokens &&
                       context_room > 0u) {
                    const auto remaining = static_cast<uint32_t>(args.max_new_tokens - generated.size());
                    const uint32_t count = std::min(std::min(flight_chunk, remaining), context_room);
                    const uint32_t first_slot = static_cast<uint32_t>(generated.size());
                    const auto t_decode = std::chrono::steady_clock::now();
                    // `executed` is what the device ACTUALLY produced. The per-token path
                    // always makes all `count` tokens; the conditional WHILE may stop early
                    // (EOS / context limit) and then reports fewer, and only those are
                    // downloaded and emitted -- an early exit can never look like unexecuted
                    // tokens ran.
                    uint32_t executed = count;
                    if (want_chunk) {
                        QwenChunkResult chunk_result;
                        st = engine_ptr->decode_device_greedy_chunk(
                            stream, need_host_input ? &pending : nullptr,
                            first_slot, count, chunk_result);
                        if (st.ok()) {
                            executed = chunk_result.executed;
                            need_host_input = false;
                            if (chunk_result.executed < chunk_result.requested) {
                                std::printf("decode while chunk: %s\n", chunk_result.summary().c_str());
                            }
                        }
                    } else {
                        for (uint32_t j = 0; j < count && exit_code == 0; j++) {
                            st = engine_ptr->decode_device_greedy(
                                stream, need_host_input ? &pending : nullptr, first_slot + j);
                            if (!st.ok()) { break; }
                            need_host_input = false;
                        }
                    }
                    if (!st.ok() || exit_code != 0) {
                        std::fprintf(stderr, "\nexample_qwen: in-flight decode failed: %s (%s)\n",
                                     st.message.c_str(), st.detail.c_str());
                        if (st.ok()) { exit_code = 2; }
                        break;
                    }
                    if (executed == 0u) {
                        std::fprintf(stderr, "\nexample_qwen: in-flight chunk executed no token\n");
                        exit_code = 2;
                        break;
                    }
                    flight_ids.resize(executed);
                    st = engine_ptr->download_device_greedy(stream, first_slot, flight_ids);
                    decode_total_ms += ms_since(t_decode);
                    decode_steps += executed;
                    context_room = executed < context_room ? context_room - executed : 0u;
                    if (!st.ok()) {
                        std::fprintf(stderr, "\nexample_qwen: in-flight download failed: %s (%s)\n",
                                     st.message.c_str(), st.detail.c_str());
                        exit_code = 2;
                        break;
                    }
                    for (const auto id : flight_ids) {
                        if (!emit_one(id)) { break; }
                    }
                }
            }
        }
        while (exit_code == 0 && !use_flight) {
            if (!have_pending) {
                const auto t_select = std::chrono::steady_clock::now();
                auto next = qwen_sample(logits, sample_options,
                                        static_cast<uint64_t>(generated.size()));
                argmax_total_ms += ms_since(t_select);
                if (!next.ok()) {
                    std::fprintf(stderr, "example_qwen: token selection failed: %s\n", next.status.message.c_str());
                    exit_code = 2;
                    break;
                }
                pending = *next.value;
                have_pending = true;
            }
            if (args.mtp && mtp->can_speculate()) {
                const auto t_decode = std::chrono::steady_clock::now();
                QwenMtpRoundResult round;
                st = mtp->round(stream, pending, sample_options, round);
                const double decode_ms = ms_since(t_decode);
                decode_total_ms += decode_ms;
                decode_steps += 1;
                if (st.ok() && options.benchmark) {
                    std::printf("mtp round=%llu input-token=%u wall=%.3f ms accepted=%u\n",
                                static_cast<unsigned long long>(mtp->rounds()), pending, decode_ms, round.accepted);
                    st = emit_benchmark(args, *engine_ptr, benchmark_iteration++, csv_started);
                }
                if (!st.ok()) {
                    std::fprintf(stderr, "\nexample_qwen: MTP round failed: %s (%s)\n",
                                 st.message.c_str(), st.detail.c_str());
                    exit_code = 2;
                    break;
                }
                bool continue_generation = true;
                for (const auto id : round.emitted) {
                    if (!emit_one(id)) { continue_generation = false; break; }
                }
                if (!continue_generation) { break; }
                // The bonus is pending, not emitted or consumed. The next round
                // verifies it first; near context end the plain path emits it.
                logits = std::move(round.next_logits);
                pending = round.next_pending;
                have_pending = true;
                continue;
            }

            // Plain target-only step (also the MTP context-end fallback).
            const uint32_t id = pending;
            if (!emit_one(id)) { break; }
            const auto t_decode = std::chrono::steady_clock::now();
            st = engine_ptr->decode(stream, id, logits);
            const double decode_ms = ms_since(t_decode);
            decode_total_ms += decode_ms;
            decode_steps += 1;
            if (st.ok() && options.benchmark) {
                std::printf("decode step=%u input-token=%u wall=%.3f ms\n", decode_steps, id, decode_ms);
                st = emit_benchmark(args, *engine_ptr, benchmark_iteration++, csv_started);
            }
            if (!st.ok()) {
                std::fprintf(stderr, "\nexample_qwen: decode failed: %s (%s)\n",
                             st.message.c_str(), st.detail.c_str());
                exit_code = 2;
                break;
            }
            have_pending = false;
        }
        if (exit_code == 0) {
            auto tail = detok.finish();
            if (tail.ok() && !(*tail.value).empty()) { std::fputs((*tail.value).c_str(), stdout); }
            std::printf("\n");
        }
        // --- report ------------------------------------------------------------------
                  std::printf("generated ids (%zu):", generated.size());
          for (auto id : generated) { std::printf(" %u", id); }
          std::printf("\n");
          if (args.have_dump_logits) {
              // The dump format names BOTH axes the gate compares: the weight storage
              // (w4/q8_0/fp8/bf16) and the K/V cache dtype (`+kv_f32|bf16|fp8_e4m3`). The
              // Q3 gate must prove the two arms share one weight format and differ ONLY in
              // the KV dtype, so the KV axis travels with the dump instead of being assumed
              // from a command line.
              const auto weight_format = spec->w4 ? luisa::string_view{"w4"}
                                                  : spec->q8_0 ? luisa::string_view{"q8_0"}
                                                               : spec->fp8 ? luisa::string_view{"fp8"}
                                                                           : luisa::string_view{"bf16"};
              const auto dump_format_storage =
                  luisa::format("{}+kv_{}", weight_format, qwen_kv_dtype_name(args.kv_dtype));
              const auto dump_format = luisa::string_view{dump_format_storage};
              // The device-greedy in-flight path never brings the last step's
              // logits back to the host, so `logits` is still the prefill vector
              // there. Write the final distribution only when it is real; the
              // reader accepts both layouts and reports the missing one as n/a.
              const auto final_span = use_flight
                                          ? luisa::span<const float>{}
                                          : luisa::span<const float>{logits.data(), logits.size()};
              if (use_flight) {
                  std::printf("quality dump: device-greedy in-flight decode -> only the "
                              "prefill last-position distribution is dumped (use "
                              "--decode-flight 0 for the final-step one)\n");
              }
              if (!write_quality_dump(args.dump_logits, args, dump_format, args.backend,
                                      static_cast<uint32_t>(logits.size()), prompt_ids,
                                      generated, prefill_logits, final_span)) {
                  exit_code = 2;
              }
          }
        auto full_text = tok.decode(generated, true, true);
        if (full_text.ok()) { std::printf("generated text: %s\n", (*full_text.value).c_str()); }
        std::printf("stop: %s\n", eos_hit ? "EOS (<|im_end|>)" : "max-new-tokens budget");
        std::printf("timing: host-load %.1f ms | weight-upload %.1f ms | engine-compile %.1f ms | "
                    "TTFT(prefill %zu tokens) %.1f ms",
                    host_load_ms, upload_ms, compile_ms, prompt_ids.size(), ttft_ms);
        if (decode_steps > 0) {
            std::printf(args.mtp ? " | decode %.2f ms/round (%u target/draft rounds)"
                                 : " | decode %.2f ms/token (%u steps)",
                        decode_total_ms / decode_steps, decode_steps);
        }
        std::printf("\n");
        if (args.mtp) {
            std::printf("mtp final: rounds=%llu accepted=%llu/%llu accept-rate=%.3f\n",
                        static_cast<unsigned long long>(mtp->rounds()),
                        static_cast<unsigned long long>(mtp->accepted_total()),
                        static_cast<unsigned long long>(mtp->drafted_total()),
                        mtp->accept_rate());
        }
        std::printf("host: token-selection %.3f ms total | detokenize %.3f ms total%s\n",
                    argmax_total_ms, detok_total_ms,
                    options.benchmark ? " | per-kernel samples include sync overhead" : "");
        {
            const auto info = engine_ptr->info();
        std::printf("engine final: position=%u shaders=%llu cuda-graph-steps=%llu updates=%llu native-prefill-launches=%llu\n",
                    info.position, static_cast<unsigned long long>(info.compiled_shader_count),
                    static_cast<unsigned long long>(info.cuda_graph_steps),
                    static_cast<unsigned long long>(info.cuda_graph_updates),
                    static_cast<unsigned long long>(info.native_prefill_gemm_launches));
              // Tile schedule (phase 9d): which conv+SiLU schedule this session
              // recorded and the attention length-bucket bookkeeping. A nonzero
              // lazy count means a request-path compile happened (ragged tails),
              // which is exactly what a TTFT number must be read against.
              if (info.kernel == QwenKernelKind::tile) {
                  std::printf("engine final: tile-chunk-conv=%d tile-attn-bucket=%u buckets=%u "
                              "bucket-precompiled=%u bucket-lazy=%u\n",
                              info.tile_chunk_conv ? 1 : 0, info.tile_attention_bucket_keys,
                              info.tile_attention_buckets, info.tile_attention_precompiled,
                              info.tile_attention_lazy_compiles);
              }
        // Conditional WHILE chunk (opt_plan/09 N4.2): launches vs tokens is the honest
        // ratio (an early EOS exit makes them differ), and fallbacks prove that a
        // refused/unavailable WHILE never silently claimed a launch it did not make.
        std::printf("engine final: while-launches=%llu while-tokens=%llu while-early-exits=%llu "
                    "while-fallbacks=%llu cached=%d\n",
                    static_cast<unsigned long long>(info.cuda_graph_while_launches),
                    static_cast<unsigned long long>(info.cuda_graph_while_tokens),
                    static_cast<unsigned long long>(info.cuda_graph_while_early_exits),
                    static_cast<unsigned long long>(info.cuda_graph_while_fallbacks),
                    info.cuda_graph_while ? 1 : 0);
        // Pinned download staging (opt_plan/09 N4.3): bytes and the host memcpys the
        // pinned path does NOT eliminate, so the log cannot overclaim.
        std::printf("engine final: pinned-downloads=%llu pinned-bytes=%llu pinned-host-copies=%llu "
                    "pinned-active=%d\n",
                    static_cast<unsigned long long>(info.pinned_downloads),
                    static_cast<unsigned long long>(info.pinned_bytes),
                    static_cast<unsigned long long>(info.pinned_host_copies),
                    info.pinned_transfers ? 1 : 0);
        // PDL (opt_plan/03 section 2.5): the counters come from CUDAPdlStreamExt, so a
        // "not available" device reports zeros with the reason above instead of a claim.
        std::printf("engine final: pdl-relaxed=%llu pdl-strict=%llu pdl-driver-fallbacks=%llu\n",
                    static_cast<unsigned long long>(info.pdl.relaxed_launches),
                    static_cast<unsigned long long>(info.pdl.strict_launches),
                    static_cast<unsigned long long>(info.pdl.driver_fallbacks));
 std::printf("engine final: native-attn-launches=%llu\n",
 static_cast<unsigned long long>(info.native_prefill_attn_launches));
 // EXPERIMENTAL CUDA native-Tile typed-FP8 prefill GEMM: the dispatch counter is what
 // proves the requested realization actually ran (0 means every projection stayed on
 // the native/DSL chain, which the `prefill cuda tile:` line explains at create).
 std::printf("engine final: cuda-tile-prefill=%d cuda-tile-prefill-launches=%llu "
             "cuda-tile-prefill-lazy-compiles=%u cuda-tile-prefill-lazy-jit-ms=%.1f\n",
             info.cuda_tile_prefill_gemm ? 1 : 0,
             static_cast<unsigned long long>(info.cuda_tile_prefill_gemm_launches),
             info.cuda_tile_prefill_gemm_compiles,
             info.cuda_tile_prefill_gemm_jit_ms);
 }
        // Teardown: complete GPU work, then let the allocator-matched engine
        // handle leave scope before weights, stream and device are released.
        stream.synchronize();
    }
    stream.synchronize();
    stream = compute::Stream{};
    device = compute::Device{};
    return exit_code;
}

[[nodiscard]] int run_cli(int argc, const char *const *argv) {
    argv0_holder() = argv[0];
    CliArgs args;
    if (!parse_args(argc, argv, args)) { return 1; }
    if (args.help) {
        print_usage();
        return 0;
    }
    if (args.list_versions) {
        print_versions();
        return 0;
    }
    if (args.self_test) {
        return run_self_test();
    }
          if (args.have_w4_calibrate) {
              return run_w4_calibration(args);
          }
            if (args.have_export_w4_pack) {
            return run_export_w4_pack(args);
        }
        if (args.have_export_projection_pack) {
            return run_export_projection_pack(args);
        }
      if (args.mtp_test) {
          return run_mtp_test(args);
    }
    return run_generation(args);
}

}// namespace

#ifdef _WIN32
int wmain(int argc, wchar_t **wargv) {
    luisa::vector<luisa::string> utf8;
    luisa::vector<const char *> argv;
    utf8.reserve(static_cast<size_t>(argc));
    argv.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; i++) {
        luisa::string arg;
        const int length = static_cast<int>(::wcslen(wargv[i]));
        if (length > 0) {
            const int needed = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wargv[i],
                                                     length, nullptr, 0, nullptr, nullptr);
            if (needed <= 0) {
                std::fprintf(stderr, "example_qwen: argument %d is not convertible to UTF-8\n", i);
                return 1;
            }
            arg.resize(static_cast<size_t>(needed));
            ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wargv[i], length,
                                  arg.data(), needed, nullptr, nullptr);
        }
        utf8.emplace_back(std::move(arg));
    }
    for (const auto &a : utf8) { argv.emplace_back(a.c_str()); }
    return run_cli(argc, argv.data());
}
#else
int main(int argc, char **argv) {
    return run_cli(argc, argv);
}
#endif
