example_qwen

Text-only, single-sequence Qwen3.5-4B inference on LuisaCompute: four explicitly selectable engine variants (ordinary DSL or Tile kernels × BF16 or true FP8 E4M3 weights), an exact GGUF-loaded tokenizer, greedy or temperature sampling, DSL WY/UT chunk-64 GDN prefill, and opt-in DSL NextN/MTP speculative decoding.

| | |
|---|---|
| Group | qwen |
| Sources | src/qwen/example_qwen.cpp, src/qwen/qwen/{engine,tokenizer,dsl_engine,tile_engine,dsl_ops,tile_ops,kv_cache_allocator,benchmark,mtp,nextn_draft,speculative,mtp_executor}.cpp, src/qwen/qwen/{common,gguf,model,weights}.cpp (reuses src/ocr/ocr/{tokenizer,detokenizer,status}.cpp) |
| Backends | dsl: cuda, dx, vk; tile: dx, vk (real-model inference and device suite); Tile CUDA is unavailable in this build |
| Build | xmake build example_qwen |
| Run | example_qwen --list-versions (offline) / example_qwen --backend cuda --prompt "Hello" --temperature 0.8 --seed 42 |

Purpose

example_qwen runs the frozen Qwen3.5-4B text backbone (hidden 2560, 32 layers arranged as 8 × [GDN, GDN, GDN, full attention], FFN 9216, vocab 248320, tied embedding/LM head) from a GGUF checkpoint. It is deliberately narrow: one sequence, text only, no vision tower. Greedy decoding is the default; positive-temperature categorical sampling and NextN draft+verify are explicit options. Every refusal is a recoverable, reported status — the tool never fabricates output, draft weights, or a kernel-family/weight-format fallback.

Variants and precision semantics

| Version | Kernels | Weight storage | Default --prefill-chunk |
|---|---|---|---|
| dsl-bf16 (default) | ordinary DSL | raw GGUF BF16, packed 2 codes per little-endian uint32 | 64 |
| dsl-fp8 | ordinary DSL | E4M3, packed 4 codes per uint32 + FP32 scales | 64 |
| tile-bf16 | Tile | raw GGUF BF16, same packing | 1 |
| tile-fp8 | Tile | E4M3, same packing + scales | 1 |

- BF16 is a raw copy of the GGUF BF16 payload bytes. FP8 is true E4M3 quantized from BF16 at model load, per output row and 64-K-element group, with an FP32 scale. Every converted matrix is reported.
- Q8_0 is INT8 block quantization, not FP8, and is rejected explicitly.
- Accumulations, norms, softmax, GDN state, WY/UT intermediates, and acceptance math remain FP32 on device; host probability/acceptance calculations use FP64. BF16/FP8 describes weight storage only.

Backend status (verified state, not a live support probe)

| Variant | cuda | dx | vk |
|---|---|---|---|
| dsl-bf16 | real-GGUF inference; 473-check device suite | 473-check device suite | 473-check device suite |
| dsl-fp8 | coherent real-GGUF chat; 473-check device suite | 473-check device suite | 473-check device suite |
| tile-bf16 / tile-fp8 | Tile route unavailable in this build | real-GGUF DSL-matching inference; 86-check device suite | 86-check device suite; repeated BF16 inference and a six-token-prefill + decode run completed cleanly |

The original Tile-VK device-lost investigation isolated two different problems: an M=1 K-segment argument-count mismatch introduced during optimization, and a backend-sensitive M=1 MMA realization that could still lose the Vulkan device. The final executor uses one-row MMA on DX and a reduction realization on VK. Repeated short VK inference and a six-token-prefill + decode run are verified; long-run stability and performance remain separate gates. The installed older Vulkan validation layer also rejects newer NVIDIA device-feature structures, so `LUISA_VULKAN_VALIDATION=1` is not a usable validation configuration on this machine; Luisa's own validation layer works.

Sampling

`--temperature 0` selects deterministic lowest-index argmax, preserving the original greedy behavior. A positive finite temperature applies categorical softmax sampling. `--seed` selects the reproducible random sequence; SplitMix64 counter-based draws are keyed by generated-token index rather than mutable call history, so prefill chunking and replay do not desynchronize them.

NaN logits fail closed. Negative infinity masks probability mass, positive-infinity entries share the limiting mass, and an all-negative-infinity row is rejected. EOS remains the GGUF `<|im_end|>` id: it counts toward `--max-new-tokens` and is never decoded. The budget is checked before another target forward.

NextN/MTP speculative decoding

The implementation has three separate layers:

1. `nextn_draft` executes the real one-layer full-attention NextN block: `eh_proj(concat(enorm(embedding), hnorm(hidden)))`, attention with its own FP32 KV cache, FFN, `shared_head.norm`, and tied or dedicated LM head. Target hidden positions, shifted token ids, and autoregressive draft positions follow vLLM's `qwen3_5_mtp.py` semantics.
2. `mtp_executor` retains target hidden history, drafts tokens, verifies `[pending,d1..dk]` with per-row target logits, snapshots/restores every GDN recurrent state and convolution ring, and replays only the pending token plus accepted drafts after rejection. The bonus stays pending and unconsumed. Rejected attention KV slots are harmless because logical length bounds every later read. The first executor deliberately forces the recurrent target GDN path so a one-row replay cannot switch numerical algorithms after a multi-row WY/UT verify.
3. `speculative` implements greedy acceptance and temperature-based speculative sampling. Sampling accepts with `min(1,p_target(d)/p_draft(d))`; a rejection bonus comes from normalized `max(p_target-p_draft,0)`, and an all-accept bonus comes from the extra target row. Counter-based accept, bonus, and draft draws have separate namespaces.

For correctness, the current draft recomputes its prefill from retained target hidden history every round. This avoids draft-cache rollback errors but can cost more than it saves. MTP also forces the single-row reference numerics mode (`dsl_decode_specialized=false`) so multi-row verify, accepted-prefix replay, and plain-reference decode do not diverge through different reduction orders; the ordinary optimized decode path is unaffected. MTP is therefore opt-in and no acceleration/default-enable claim is made.

The supplied `data/Qwen3.5-4B-BF16.gguf` still has only 426 trunk tensors: no `qwen35.nextn_predict_layers` metadata and no `blk.32.*` draft weights. `--mtp` refuses this file with that exact reason. A complete combined GGUF is required for real-draft quality and performance measurements.

Validation without a combined checkpoint uses explicit synthetic fixtures, never reported as model-quality evidence:

- `test_qwen_speculative`: 30 host tests / 367 assertions for acceptance, residual sampling, malformed inputs, and reproducibility.
- `test_qwen_mtp <backend>`: real trunk + zero NextN draft; three actual draft/verify/reject/rollback rounds matched plain target prefixes, bonuses, and positions (36 checks on each of CUDA, DX, and VK).
- `test_qwen_mtp <backend> zero`: 64 checks over a 17-token multi-chunk history, full acceptance, exact partial acceptance (`1/2`, including two-row restore+replay), and reproducible temperature-based verify rounds (verified on CUDA and DX).

WY/UT chunk-64 GDN prefill

DSL prefill now implements the FP32 WY representation + UT transform over serial 64-token chunks. Per V-head it computes chunk-local `G=cumsum(g)`, decay-scaled strict-lower KKT, an in-place unit-lower inverse by forward substitution, W and unscaled new values, output from the chunk-start state, and the gated low-rank state update. State remains K-major FP32 and V-head h uses QK-head `h % qk_heads`, matching the GGUF tiled-head convention.

The implementation avoids `exp(-global_Gamma)`: decay appears only in bounded within-chunk `G_i-G_j` factors and `exp(G_last)` state updates. Ragged tails, zero decay, strong-decay underflow, nonzero initial state, and cross-chunk carry are covered at lengths 1, 63, 64, 65, 128, and 129 against the independent FP64 recurrent oracle on CUDA, DX, and VK. A six-token real-CUDA WY/UT versus recurrent run selected the same next token. `--gdn-recurrent` retains the exact per-token prefill path for differential debugging; decode always uses it.

Tile still uses exact per-token recurrent GDN prefill. Tile WY/UT is not claimed.

Tokenizer

The tokenizer is loaded and validated from GGUF metadata. The loader constructs the exact HF-style tokenizer.json and reuses the OCR byte-level BPE/detokenizer. The checkpoint declares `tokenizer.ggml.pre == "qwen35"`; QwenTokenizer implements that Unicode regex exactly and rejects other pre identities. EOS is 248046 (`<|im_end|>`), no BOS is inserted, and `--chat` uses the fixed single-user-turn non-thinking wrapper rather than a Jinja interpreter.

Usage

Offline entry points need neither a device nor a model file:

```bat
example_qwen --list-versions
example_qwen --self-test
example_qwen --help
example_qwen --mtp-test --model data/Qwen3.5-4B-BF16.gguf
```

Real inference:

```bat
rem Greedy DSL with WY/UT prefill
example_qwen --backend cuda --version dsl-bf16 --prompt "Hello" --max-new-tokens 32

rem Reproducible temperature sampling
example_qwen --backend cuda --prompt "Hello" --temperature 0.8 --seed 42

rem Differential recurrent prefill path
example_qwen --backend cuda --prompt "Hello" --gdn-recurrent

rem Explicit NextN mode; requires a complete combined trunk+draft GGUF
example_qwen --backend cuda --version dsl-bf16 --model combined.gguf --prompt "Hello" --mtp --draft-depth 2
```

| Option | Meaning |
|---|---|
| --backend <name> | backend plugin (cuda, dx, vk, ...); positional backend also accepted |
| --version <name> | dsl-bf16 (default), dsl-fp8, tile-bf16, tile-fp8 |
| --model <file.gguf> | default data/Qwen3.5-4B-BF16.gguf; Q8_0 rejected |
| --prompt / --tokens | UTF-8 prompt or explicit comma-separated token ids |
| --chat | fixed Qwen user/assistant wrapper, --prompt only |
| --max-new-tokens <n> | generated-token budget (default 32) |
| --decode-flight <n> | greedy CUDA DSL forwards per bounded host sync (default 4; 0 restores synchronous host-logits decode) |
| --no-cuda-graph | disable FP8 CUDA whole-step graph replay for A/B or debugging |
| --temperature <t> | 0 = greedy; positive finite value = categorical sampling |
| --seed <n> | unsigned 64-bit sampling seed |
| --max-context <n> | KV capacity (default 4096 DSL / 64 Tile, ≤ 262144) |
| --prefill-chunk <n> | prompt rows per engine chunk (default 64 DSL / 1 Tile) |
| --gdn-recurrent | use recurrent DSL GDN prefill instead of WY/UT |
| --mtp | opt into real DSL NextN draft+verify; fails without complete draft weights |
| --draft-depth <n> | drafts per round (default 2); verify needs n+1 workspace rows |
| --benchmark | commit + synchronize after every kernel/transfer and print raw wall intervals |
| --benchmark-csv <file> | benchmark mode and append raw samples to UTF-8 CSV |
| --mtp-test | host-only GGUF NextN inventory/policy probe |
| --self-test | device-free operator, tokenizer, and policy tests |
| --list-versions | offline support matrix |
| --headless | accepted for gallery convention; no window is opened |

Benchmarking and command batching

Normal DSL generation batches a whole decode and all prefill chunks into stream-ordered command lists. Greedy CUDA defaults to four in-flight decode steps per host synchronization: a DSL argmax writes the next device token, and only a small token-history range is downloaded. FP8 CUDA captures the device-only decode body as one CUDA graph and updates scalar arguments on subsequent steps. Chunk-boundary host token uploads remain ordinary ordered pre-graph commands; position ramps are derived in-kernel. The final engine line reports graph launch/update counts, and capture/update failures are returned without executing a consumed list. `--no-cuda-graph` selects the ordinary-list A/B path. EOS is checked at each bounded chunk; any later work in that chunk is discarded. `--decode-flight 0` restores per-token full-logits download and host selection. Temperature sampling, MTP, and `--benchmark` always retain their reference synchronization behavior. Tile DX uses bounded commit-only lists; Tile VK retains conservative stage synchronization. `--benchmark` deliberately synchronizes every command and reports host-wall intervals, not hardware GPU timestamps. MTP adds draft proposal, hidden-history copies, snapshots, verify, possible restore/replay, and acceptance downloads, so its milliseconds per round must not be compared as though they were plain milliseconds per token.

For paired repeated ordinary-engine measurements use `scripts/benchmark_qwen.py`; raw logs and summaries are written under `.kimix_cache/qwen_benchmark/` by default. No MTP speedup or default-policy change is claimed without a real combined-checkpoint A/B that includes draft, verify, snapshot, restore, and replay costs.

DSL CUDA optimization

The ordinary-DSL decode now uses model-shape-specialized kernels while retaining the generic operators as test/reference paths: 256-thread wide RMS norms, parallel attention epilogue merging, a four-way split-key GDN recurrence with shared Q/K staging and XOR warp reductions, fused FFN gate/up/SwiGLU, and fused alpha/beta/gate projection. FP8 packed GEMV uses an exact 256-entry shared E4M3 lookup table instead of per-weight exponent-normalization loops. A four-stream BF16 GEMV experiment was rejected after paired runs measured a 0.175% regression; large BF16 projections were already near the RTX 4080 SUPER memory roof.

Representative Nsight Compute `full` results (RTX 4080 SUPER, base-clock/cache-controlled replay; not request-wall samples):

| Kernel | Before | After | Final achieved occupancy | Final evidence |
|---|---:|---:|---:|---|
| hidden RMS norm | 45.952 µs | **8.320 µs** | 15.14% (one 256-thread block) | 25 registers, zero spills |
| decode attention | 61.920 µs | **9.632 µs** | 7.63% | 34 registers, zero spills |
| GDN recurrence | 26.528 µs | **8.160 µs** | 12.11% | 128 blocks, 250 registers, zero spills |
| alpha/beta/gates | ~14.7 µs total | **6.112 µs** | 7.70% | one launch instead of three |
| greedy argmax | 85.536 µs | **4.704 + 2.880 µs** | 40.37% / 12.91% | 256-block partial + finish |
| packed QKV GEMV | 63.136 µs | 63.264 µs | 92.21% | 663 GB/s DRAM; unchanged bandwidth-roof path |
| tied LM head | — | 1.896 ms | 88.48% | 697 GB/s DRAM, zero spills |

The raw reports are under `.kimix_cache/qwen_dsl_opt/nsight/final_keys/`. Fused FFN is retained for its measured request-level launch/intermediate-traffic win, not a claim that its isolated kernel always beats two separate GEMVs.

### FP8 bandwidth, fusion, and graph pass

The current FP8 pass enables the same fused gate/up/SwiGLU route already used by BF16, removes DSL position upload/copy commands, and replays FP8 CUDA device-greedy decode through `CudaGraphExt`. The final paired cache-resistant FFN pipeline benchmark measured **0.08042 ms for two GEMVs plus SwiGLU versus 0.07540 ms fused**. On the real four-token prompt/eight-decode-step diagnostic configuration, FP8 improved from **8.55 to 8.03 ms/token** with identical IDs and **8 graph launches / 7 updates**; BF16 measured **13.98 ms/token** on its non-graph path. The final FP8/BF16 request-wall ratio is **1.74×**. These are short A/B diagnostics, not a broad latency distribution.

Matched Nsight Compute `full` captures with cache flushing explain why halving E4M3 payload bytes does not produce an automatic 2× end-to-end result:

| N=9216, K=2560 GEMV | DRAM read | Duration | DRAM bandwidth | DRAM utilization | Achieved occupancy |
|---|---:|---:|---:|---:|---:|
| BF16 generic | 47.205 MB | 69.376 µs | 680.43 GB/s | 94.65% | 91.65% |
| FP8 specialized | 26.239 MB | 40.032 µs | 655.44 GB/s | 91.20% | 87.15% |

The FP8 payload alone is 23.593 MB; the remaining measured traffic is FP32 scales and activation reads. Thus the actual byte-reduction ratio is about **1.80×**, and both kernels are already near the DRAM roof. Occupancy-only tuning cannot create a 2× whole-decode result under this ABI. A baseline Nsys timeline measured approximately **8.11 ms of kernels plus 0.62 ms of internal launch gaps per token** (~438 launches), which motivated graph replay.

`benchmark_qwen_decode` provides checkpoint-free, cache-resistant shape sweeps, paired FFN measurements, and attention timings; the saved raw artifacts also include the block-size sweep. Vectorized loads, adjacent-row pairing, small dual-projection fusion, native FP8 conversion, block-size changes, and a second attention merge barrier were also tested; none produced a broad enough win to retain. Raw logs and Nsight reports are under `.kimix_cache/qwen_fp8_opt/`.

Measured optimization snapshot

Short one-token-prompt/two-token-output smokes on the RTX 4080 SUPER are correctness plus diagnostic measurements, not a statistically rigorous benchmark:

| Variant | Backend | Result |
|---|---|---|
| dsl-bf16 | cuda | original-vs-final, three interleaved warm processes per arm: 17.150 → **13.970 ms/token** (−18.54%), 58.31 → **71.58 tokens/s** (+22.76%); TTFT 38.000 → **14.500 ms**; identical IDs. Raw artifacts: `.kimix_cache/qwen_dsl_opt/final_bf16/` |
| dsl-fp8 | cuda | one paired real-model process per arm: 13.680 → **9.820 ms/token** (−28.22%), identical IDs; two unchanged-binary follow-ups measured **8.700/8.820 ms/token**. These isolate decode, not the ~116–119 s FP8 conversion or first-use JIT/TTFT costs. Raw artifacts: `.kimix_cache/qwen_dsl_opt/fp8_final/`, `fp8_warm/`, `fp8_warm2/` |
| tile-bf16 | dx | baseline 9.157 s decode; final one-row MMA 1.361/1.367 s in two repeated runs, identical ids; about 6.7× lower than the single baseline run |
| tile-bf16 | vk | reduction realization completed repeatedly at 14.80/14.85 s/token with identical ids; a separate six-token prefill + decode also completed cleanly. Correctness improved, but performance remains poor |
| dsl-bf16 | cuda | six-token prompt: WY/UT 129.4 ms TTFT vs recurrent 136.1 ms in one unpaired smoke, same next id; no speedup claim |

The main Tile changes are contiguous embedding tile loads (removing quadratic select-scans), a dedicated M=1 projection realization, and backend-specific DX MMA versus VK reduction selection. M=1 K-segment dispatch and shader-cache keys include the realization mode. Larger repeated benchmarks are still required before replacing the older long-form performance tables.

Sparse KV-cache allocation

Commit 0375b17 originally allocated one dense FP32 K buffer and one dense FP32 V buffer for each of the eight full-attention layers, sized to the entire configured `max_context`, and zeroed all sixteen buffers on reset. The current implementation centralizes this state in `QwenKvCacheAllocator`:

- One `SparseBuffer<float>` reserves the complete maximum-context virtual capacity. It contains sixteen tile-aligned regions in `[K0,V0,K1,V1,...]` order. Each region still exposes exactly `[max_context, kv_heads * head_dim]` through a `BufferView<float>`.
- Existing kernels and numerical addressing are unchanged: token `p`, channel `c` remains element `p * 1024 + c` within a layer's K or V view. There is no block table, prefix sharing, multi-request scheduler, or non-contiguous logical token order in this single-sequence example.
- One `vector<SparseBufferHeap>` owns the physical tiles. `ensure_capacity(stream, pos0 + rows)` allocates and maps only newly required tiles in every K/V region. Stable virtual views and existing tile mappings preserve prior contents across growth; normal growth adds no host synchronization.
- Reset does not clear maximum-context storage. After synchronizing the last possible use, it unmaps committed ranges and moves the retired heaps into a `CommandList` completion callback. The sparse virtual buffer remains available for remap. Engine teardown also drains the retained creation stream before any child resource is released.
- All calls on an engine or NextN draft must use its creation stream. Sparse-mediated execution/reset paths reject a different stream rather than risking an unmapped in-flight cache. Backends without sparse-buffer support fail closed; no dense fallback is silently selected.

The CUDA backend fix in `patches/luisa-cuda-sparse-buffer-tile-offset.patch` is required for more than one tile per buffer: `cuMemMap` now receives the tile-adjusted destination address and a zero offset into the one-tile heap. The previous call mistakenly applied the tile offset to the heap and left the destination at the buffer base.

Memory and lifetime

Approximate device memory for one sequence:

| Item | Size |
|---|---|
| BF16 weights | ≈ 7.9 GiB (426 tensors; tied embedding counted once) |
| FP8 E4M3 weights + scales | ≈ 4.1 GiB |
| GDN recurrent state | 48 MiB fixed |
| GDN conv history | 2.25 MiB DSL; 4.5 MiB Tile ping-pong rings |
| Full-attention KV cache | 64 KiB logical payload per token across eight layers; physical residency is allocated on demand and rounded up to backend sparse tiles |
| DSL WY/UT scratch | two `[prefill-chunk,32,128]` FP32 buffers (2 MiB at chunk 64) |
| MTP hidden history | `max_context × 2560 × 4` bytes (40 MiB at 4096) |
| MTP GDN snapshots | ≈ 50.25 MiB |
| MTP verify logits | `(draft_depth+1) × vocab × 4` bytes |

Both engines and the MTP objects retain their creating Device and Stream until after child buffers/shaders are destroyed. `QwenEngineInfo::cache_bytes` remains the configured logical KV payload plus fixed GDN state, not current sparse physical residency; allocator accessors distinguish logical, virtual padded, and committed bytes. Reset synchronizes before state or staging reuse. Host upload spans are persistent or synchronized before destruction; in-flight token-history downloads synchronize before their host vectors are resized or reused. Device greedy selection preserves lowest-index ties and rejects NaN with a checked sentinel. WY scratch is engine-owned for the whole session. MTP snapshots/history use device-to-device copies, and replay never consumes the bonus. Weight-upload staging relies on immediate stream encoding; it must not be moved to deferred batches without changing ownership.

Tests

```bat
xmake build test_qwen_engine test_qwen_speculative test_qwen_gguf test_qwen_weights
xmake build test_qwen_kv_cache test_qwen_ops_dsl test_qwen_ops_tile test_qwen_mtp
build\windows\x64\release\test_qwen_engine.exe
build\windows\x64\release\test_qwen_speculative.exe
build\windows\x64\release\test_qwen_kv_cache.exe cuda
build\windows\x64\release\test_qwen_kv_cache.exe dx
build\windows\x64\release\test_qwen_kv_cache.exe vk
build\windows\x64\release\test_qwen_ops_dsl.exe cuda
build\windows\x64\release\test_qwen_ops_tile.exe vk
build\windows\x64\release\test_qwen_mtp.exe cuda
build\windows\x64\release\test_qwen_mtp.exe cuda zero
build\windows\x64\release\test_qwen_mtp.exe cuda fp8-zero
example_qwen --self-test
```

Sparse-cache validation: **49 allocator checks pass on each of CUDA, DX, and VK**, including multi-tile writes, growth with preserved data, reset/remap, rejection of a foreign stream, and reset/destruction after dispatched work. Real BF16 inference passes through sparse K/V on DSL CUDA (including four in-flight greedy forwards) and Tile DX with matching token IDs; the CUDA synthetic MTP suite passes **64 checks**. The `fp8-zero` synthetic engine probe adds **12 checks** for FP8 engine creation, prefill, four graph launches, three updates, and deterministic history without running the host quantizer.

Final optimized validation: **473 DSL operator checks pass on each of CUDA, DX, and VK**, and **36 MTP checks pass on each backend**. Host suites pass with 439 engine assertions, 367 speculative assertions, 223 GGUF assertions, and 3164 weight assertions; the offline self-test also passes. A real 33-token chat run produces identical IDs with `--decode-flight 0` and `4`. The final relinked BF16 executable reproduces 13.970 ms/token and 14.5–14.6 ms TTFT in the benchmark/smoke configuration.

Known limitations

- Text only: no vision tower. Sampling currently has temperature only, not top-k/top-p.
- Qwen now requires a sparse-buffer-capable backend. Small or heavily padded configurations can commit more physical bytes than their logical KV payload because residency is tile-granular; short sessions otherwise avoid committing the full maximum context.
- The supplied GGUF has no NextN weights. Real-draft output quality and end-to-end acceleration are unmeasured; MTP is opt-in and the current draft re-prefill policy may be slower than plain decoding.
- Tile GDN remains recurrent rather than WY/UT. Tile MTP is not implemented and never falls back silently to DSL.
- Tile VK short inference is fixed and verified, but it is slow and long-run stress coverage is still needed. Tile CUDA is unavailable in this build.
- MTP currently forces recurrent target GDN and the single-row reference numerics mode for verify/replay/plain-reference parity. Combining WY/UT target verify with one-row replay requires per-row chunk-state restoration.
- FP8 remains lossy load-time E4M3 quantization of BF16, not a separately trained/validated FP8 checkpoint; token trajectories may diverge from BF16. FP32 scales, FP32 activations, and fixed non-projection kernels mean a 2× end-to-end speedup over BF16 is not implied by the nominal weight-size ratio.
- Exit codes: 0 success/help/list/self-test/mtp-test, 1 bad arguments, 2 engine/model refusal or generation failure.
