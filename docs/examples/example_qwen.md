example_qwen

Text-only, single-sequence Qwen3.5-4B inference on LuisaCompute: six explicitly selectable engine variants (ordinary DSL or Tile kernels × BF16, true FP8 E4M3, native Q8_0 int8-block or W4A16 group-wise weights), an exact GGUF-loaded tokenizer, greedy or temperature sampling, DSL WY/UT chunk-64 GDN prefill, and opt-in DSL NextN/MTP speculative decoding.

| | |
|---|---|
| Group | qwen |
| Sources | src/qwen/example_qwen.cpp, src/qwen/qwen/{engine,tokenizer,dsl_engine,tile_engine,dsl_ops,tile_ops,kv_cache_allocator,benchmark,mtp,nextn_draft,speculative,mtp_executor}.cpp, src/qwen/qwen/{common,gguf,model,weights,native_gemm,cuda_tile_gemm}.cpp (+ native_gemm_kernel.h, the CUDA C++ text NVRTC compiles, and cuda_tile_gemm.h, the experimental opt-in CUDA Tile IR FP8 prefill GEMM) (reuses src/ocr/ocr/{tokenizer,detokenizer,status}.cpp) |
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
| dsl-q8 | ordinary DSL | native Q8_0: the checkpoint's signed int8 block codes copied verbatim (4 per uint32) + one exact FP16→FP32 scale per 32 K elements | 64 |
| dsl-w4 | ordinary DSL | W4A16 from an **offline pack** (`--w4-pack`): 8 signed int4 codes [-8, 7] per uint32 + one FP32 scale per 128 K elements from a deterministic per-group clipping search. The same pack carries an FP8 E4M3 copy for prefill and the LM head (plan option (c)) | 64 |
| tile-bf16 | Tile | raw GGUF BF16, same packing | 1 |
| tile-fp8 | Tile | E4M3, same packing + scales | 1 |

- BF16 is a raw copy of the GGUF BF16 payload bytes. FP8 is true E4M3 quantized from BF16 at model load, per output row and 64-K-element group, with an FP32 scale. Every converted matrix is reported.
- dsl-q8 is **not** a conversion: the `data/Qwen3.5-4B-Q8_0.gguf` Q8_0 block payload is uploaded as-is. Every 32-weight block's 32 signed int8 codes are copied byte for byte (four per little-endian uint32, low element first) and the block's FP16 scale is widened **exactly** to FP32 (one scale per 32 K elements per output row, K/32 per row). Zero host re-quantization, and Q8_0 is never routed through FP8. See "Native Q8_0 weights" below for the exact ABI.
- A Q8_0 checkpoint is rejected by every other variant, and `--version dsl-q8` rejects a non-Q8_0 checkpoint: the CLI never substitutes one weight format for another.
- Accumulations, norms, softmax, GDN state, WY/UT intermediates, and acceptance math remain FP32 on device; host probability/acceptance calculations use FP64. BF16/FP8/Q8_0 describes weight storage only.

Backend status (verified state, not a live support probe)

| Variant | cuda | dx | vk |
|---|---|---|---|
| dsl-bf16 | real-GGUF inference; 1305-check device suite | 1305-check device suite | 1305-check device suite |
| dsl-fp8 | coherent real-GGUF chat; 1305-check device suite | 1305-check device suite | 1305-check device suite |
| dsl-q8 | real-GGUF inference on `data/Qwen3.5-4B-Q8_0.gguf` (coherent 8-token answer, greedy ids) + 1829-check device suite (83 of them Q8_0) | real-GGUF inference on the Q8_0 checkpoint + 1829-check device suite | real-GGUF inference on the Q8_0 checkpoint + 1829-check device suite |
| dsl-w4 | real-model inference from an offline pack (static CUDA graph, W4 decode + FP8 prefill); clipping-only and importance-aware 32-prompt + 1 K quality gates both FAILED, so this is EXPERIMENTAL, not a shipping format. Device suite includes W4 oracle paths on cuda/dx/vk; see "Phase 8b" below | W4 oracle paths verified; no long W4 inference run recorded | W4 oracle paths verified; no long W4 inference run recorded |
| tile-bf16 / tile-fp8 | Tile route unavailable in this build (`test_qwen_ops_tile cuda` reports the exact probe: native tileiras needs CUDA 13.4+, this toolkit is 13.0, and the TIRx bridge is not built) | real-GGUF DSL-matching inference; 164-check device suite | 164-check device suite; repeated BF16 inference and a six-token-prefill + decode run completed cleanly |

The original Tile-VK device-lost investigation isolated two different problems: an M=1 K-segment argument-count mismatch introduced during optimization, and a backend-sensitive M=1 MMA realization that could still lose the Vulkan device. The final executor uses one-row MMA on DX and a reduction realization on VK. Repeated short VK inference and a six-token-prefill + decode run are verified; long-run stability and performance remain separate gates. The installed older Vulkan validation layer also rejects newer NVIDIA device-feature structures, so `LUISA_VULKAN_VALIDATION=1` is not a usable validation configuration on this machine; Luisa's own validation layer works.

Native Q8_0 weights (dsl-q8, opt_plan/05 Q1 — phase 8a)

`--version dsl-q8` makes `data/Qwen3.5-4B-Q8_0.gguf` a first-class engine input instead of a reference-only file. It is the first *storage* format that is uploaded exactly as the checkpoint stores it:

| | |
|---|---|
| Device layout | `Buffer<uint32_t>` of `K*N/4` words, four **signed int8** codes per little-endian word, LOW element first — a byte-for-byte copy of the GGUF Q8_0 block's 32 code bytes |
| Scales | `Buffer<float>` of `N * (K/32)`, one value per 32-element K group per output row; each is the **exact** FP16→FP32 widening of that block's 2-byte GGUF scale (FP16→FP32 is lossless, so the value is identical to the checkpoint) |
| Geometry | `K % 32 == 0` for every matrix (whole 32-weight blocks, i.e. 8 packed words per group); `K=2560/4096/9216` and the tied LM head/embedding `K=2560, N=248320` all qualify. A non-multiple-of-32 K, a non-matrix Q8_0 tensor, or a non-32 group is refused fail-closed by the loader, the classifiers and every planner |
| Zero re-quantization | No host quantization path exists for Q8_0: `host_pack_matrix(q8_0, …)` is refused by design, and the loader only splits scales from codes. `test_qwen_gguf` walks real GGUF blocks and asserts byte equality plus the exact widening; `test_qwen_weights` (device) compares device words/scales against the real checkpoint payload for `token_embd`, `ffn_gate`, `ffn_down`, `attn_qkv` and `attn_q` |
| Sensitive tensors | Norms, `ssm_a`, `ssm_dt.bias` and the depthwise `ssm_conv1d` stay dense F32 exactly as in bf16/fp8 (in the real Q8_0 checkpoint those 177 tensors are F32 already, so nothing is converted) |
| Not FP8 | Q8_0 is never routed through FP8, and FP8 is never substituted for Q8_0: `QwenDeviceWeights` refuses a Q8_0 checkpoint unless `q8_0` is set, `--version dsl-q8` refuses a non-Q8_0 checkpoint, and every other variant keeps the historical explicit rejection |

**Scale representation decision (documented, measured).** Raw FP16 scales would halve the scale traffic; this arm stores FP32 and widens at load instead, because (a) it keeps ONE scale ABI across fp8 and q8 (`Buffer<float>` scales, identical kernel signatures and planners) so no kernel family is duplicated, (b) the widening is exact, and (c) the measured cost is small: the Q8_0 device form is 1.1875 B/weight (1.0625 codes + 0.125 scale) versus FP8's 1.0625 B/weight, and the decode GEMV measured **4–5 % slower than FP8, 1.35–1.73× faster than BF16** (table below). A raw-FP16-scale variant would recover roughly 2–3 % of that at the cost of a second scale ABI; it is not implemented.

Kernel coverage (all of it real inference needs; each is the same kernel family as fp8 with the int8 decode substituted in-kernel):

| Path | Realization |
|---|---|
| Embedding gather | `make_qwen_embedding_kernel(q8_0)`: signed-int8 decode × per-32 block scale |
| Decode projection GEMV | `make_qwen35_decode_gemv_kernel(q8_0)`: the four-accumulator model specialization, one FP32 scale per 8 packed words, no LUT and no barrier (int8→fp32 is `(code ^ 0x80) - 128`) |
| Decode GEMV, 128-bit | `make_qwen35_decode_gemv_packed_kernel(q8_0)`: `uint4` weights / `float4` activations, one scale per two `uint4`s (`group/16 = 2`); opt-in via `--decode-packed 1` |
| Fused FFN decode | `make_qwen_ffn_decode_kernel(q8_0)` (and the `_packed` twin): gate/up + SiLU in one launch |
| Fused GDN gates | `make_qwen_gdn_decode_gates_kernel(q8_0)` (and `_packed`): alpha/beta GEMV + softplus/sigmoid |
| Prefill GEMM | `make_qwen35_gemm_mt_kernel(q8_0)`: the same weight-reusing MT tile (one warp per output column, MT-row stripe, one scale per packed word); the reference `make_qwen_gemm_kernel(q8_0)` stays available via `--legacy-prefill-gemm` |
| LM head | the tied `token_embd` GEMV through the same decode specialization (`N=248320`) |
| CUDA tensor-core prefill GEMM | **not** implemented for Q8_0: `qwen_native_gemm_supported_format` refuses it and the engine keeps the DSL MT GEMM for every projection (`prefill gemm: mt(mt=16) native=off`). An `mma.sync` kernel needs BF16 operands and widening int8 codes into the BF16 shared tile would be its own measured realization; refusing is fail-closed, and Q8_0 prefill is correct through the DSL path |
| Static CUDA decode graph | format-agnostic and unchanged: the captured body only contains the kernels above (device state, no host scalars), so `decode graph: requested=static active=static … storage=q8_0_packed` with `cuda-graph-steps=7 updates=0` in the run below |
| Tile variants | unchanged: `tile-bf16`/`tile-fp8` refuse the native Q8_0 storage explicitly |

Measurements (RTX 4080 SUPER 16 GiB, CUDA 13.0, one box, medians of 20 launches; raw logs in `.kimix_cache/qwen_benchmark/phase8a/`)

Real checkpoint, `benchmark_qwen_decode cuda --format all`, decode GEMV (M = 1, medians of 20 launches):

| shape | N | K | bf16 ms | fp8 ms | q8_0 ms | q8_0 GB/s | fp8 GB/s | q8 vs bf16 | q8 vs fp8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gdn_gate | 32 | 2560 | 0.00628 | 0.00781 | 0.00669 | 15.1 | 12.2 | 0.94× | 1.17× |
| attn_kv | 1024 | 2560 | 0.01014 | 0.00747 | 0.00747 | 375 | 361 | 1.36× | 1.00× |
| gdn_out_attn_out | 2560 | 4096 | 0.03309 | 0.01908 | 0.01971 | 598 | 583 | 1.68× | 0.97× |
| gdn_z | 4096 | 2560 | 0.03410 | 0.02093 | 0.02170 | 539 | 532 | 1.57× | 0.96× |
| attn_q_gdn_qkv | 8192 | 2560 | 0.06420 | 0.03640 | 0.03818 | 619 | 612 | 1.68× | 0.95× |
| ffn_gate_up | 9216 | 2560 | 0.07104 | 0.03954 | 0.04120 | 643 | 634 | 1.72× | 0.96× |
| ffn_down | 2560 | 9216 | 0.07180 | 0.03979 | 0.04174 | 633 | 630 | 1.72× | 0.95× |
| lm_head (`--lm-head`) | 248320 | 2560 | 1.833 | 0.987 | 1.048 | 683 | 686 | 1.75× | 0.94× |

The GB/s columns are computed from each format's own **device** byte count, so Q8_0's includes its FP32 block scales (1.1875 B/weight) and FP8's its 64-element scales (1.0625 B/weight): the two formats achieve the *same* effective bandwidth (538–643 vs 532–634 GB/s), and Q8_0 is 4–5 % slower in wall time purely because it moves 11.8 % more bytes. `n=32`/`K=2560` is launch-bound (16 blocks) for all three formats, which is why its GB/s is 12–27 and why it is not the row to reason from. The fused FFN decode (`--ffn`) and the joined alpha/beta gate GEMV (`--gdn-gates`) behave the same way: `ffn_gate_up` fused 0.0794 ms (q8) vs 0.0757 (fp8) vs 0.1395 (bf16); `gdn_gates` median 0.00798 / `_packed` 0.00736 ms (q8) vs fp8 0.00818 / 0.00820 and bf16 0.00737 / 0.00772 — both realizations validated against each other (`GDN_GATES summary: validated=ok`), and the 128-bit one wins for Q8_0 like it does for BF16.

Prefill (`--prefill-gemm --format all`, `ffn_gate_up` N=9216 K=2560, every row `validated=ok` against the reference): MT GEMM q8_0 **0.991 / 3.851 / 16.394 ms** at M = 64 / 256 / 1024 versus fp8 1.017 / 3.981 / 16.280 ms, bf16 0.628 / 2.392 / 10.038 ms and the reference (legacy) GEMM 5.62 / 22.38 / 89.56 ms for Q8_0. Q8_0 prefill is therefore at parity with FP8 (it is the DSL MT kernel either way), and the native tensor-core row is reported `PREFILL-GEMM native: unavailable for q8_0 (native tensor GEMM has no realization for this weight format: q8_0)`.

End-to-end (one prompt, greedy, `--tokens 3710,369,279,6511,314,9338,30`, `--max-context 4096`, 32 new tokens, `real_dsl-*_cuda.log`)

| version | checkpoint | weight upload | decode | generated ids |
|---|---|---:|---:|---|
| dsl-bf16 | Qwen3.5-4B-BF16.gguf | 3278 ms | 16.92 ms/token | 271 760 6511 314 9338 369 11751 13 … |
| dsl-fp8 | Qwen3.5-4B-BF16.gguf (E4M3 at load) | 118354 ms | 7.59 ms/token | same 32 ids |
| dsl-q8 | Qwen3.5-4B-Q8_0.gguf (native copy) | 1178 ms | 7.95 ms/token | same 32 ids |

So on this box a real dsl-q8 token is **2.13× faster than dsl-bf16** and 4.5 % slower than dsl-fp8 (the 1.1875 vs 1.0625 B/weight scale overhead), and the upload is 100× cheaper than dsl-fp8's load-time quantization because nothing is converted. **This is not a quality gate**: 32 identical greedy ids on ONE prompt is an anecdote, and Q8_0 is not bit-identical to BF16 or FP8 by construction. The plan's gate (≥ 32 fixed prompts × ≥ 128 tokens, ≥ 90 % prefix agreement, KL between logit distributions, 1 K-token NaN/repeat check) has not been run; treat dsl-q8 as a validated *format and operator* path with a measured speed/efficiency profile, not as a validated quality tier.

The same checkpoint also runs on dx and vk (`real_dsl-q8_dx.log`, `real_dsl-q8_vk.log`): identical 8 greedy ids, `decode graph … active=unavailable storage=q8_0_packed` (the static graph is CUDA-only), DX 142.76 ms/token and VK 9.45 ms/token — DX's decode is dominated by its own command-dispatch cost, exactly as for bf16/fp8, so no Q8_0-specific claim is made from it.

Benchmark harness: `scripts/benchmark_qwen.py --versions dsl-q8` now maps the arm to `data/Qwen3.5-4B-Q8_0.gguf` automatically (`resolve_arm_model`; an explicit `--model` still wins), records `model_per_version`, and the roof byte model knows `storage=q8_0_packed` (measured 4514.92 MiB = 4.73424e9 B, including the FP32-widened scales; the source-GGUF 1.0625 B/weight figure would undercount this device format) — without that key it reports n/a instead of guessing. `benchmark_qwen_decode --format q8|all` adds the Q8_0 rows to every decode/prefill/FFN/gate arm; `--format both` keeps its historical bf16+fp8 meaning so existing baselines are unchanged.

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
rem Long prompt without a long command line (Windows caps it at 32767 chars)
example_qwen --backend cuda --synthetic-prompt-len 8192 --max-new-tokens 1 --max-context 8200
rem Same ids from a file (decimal ids, one per line or comma/space separated)
example_qwen --backend cuda --tokens-file prompt_ids.txt --max-new-tokens 8 --max-context 1024
```

| Option | Meaning |
|---|---|
| --backend <name> | backend plugin (cuda, dx, vk, ...); positional backend also accepted |
| --version <name> | dsl-bf16 (default), dsl-fp8, dsl-q8, tile-bf16, tile-fp8 |
| --model <file.gguf> | default data/Qwen3.5-4B-BF16.gguf. dsl-q8 needs a Q8_0 checkpoint (data/Qwen3.5-4B-Q8_0.gguf); every other variant rejects one explicitly |
| --prompt / --tokens | UTF-8 prompt or explicit comma-separated token ids |
| --tokens-file <path> | prompt token ids from a file (decimal ids separated by commas or whitespace); the way to pass a prompt that does not fit a Windows command line |
| --synthetic-prompt-len <n> | deterministic id list of exactly n tokens, identical to `scripts/benchmark_qwen.py`'s generator (64-bit LCG seeded with n); no command-line length limit |
| --chat | fixed Qwen user/assistant wrapper, --prompt only |
| --max-new-tokens <n> | generated-token budget (default 32) |
| --decode-flight <n> | greedy CUDA DSL forwards per bounded host sync (default 4; 0 restores synchronous host-logits decode) |
| --cuda-graph | CUDA: record the greedy decode step **once** as a static graph and launch it per token with zero updates (default; BF16 and FP8) |
| --no-cuda-graph | CUDA: ordinary per-token command lists instead of the static graph (A/B or debugging) |
| --cuda-graph-while <n> | CUDA, opt-in: run up to **n** decode tokens in **one** conditional WHILE graph launch whose loop stops *on the device* (EOS, context limit, or the n budget). Gated twice — the `CudaGraphExt` conditional-node capability **and** a dedicated instantiate/execute/replay/zero-trip probe — and the trip count is read back from the device, so an early exit never emits a token that was not generated. 0 = off (default); a refusal keeps the per-token graph and prints the reason. Needs `--cuda-graph` and an EOS id (always supplied by the tokenizer) |
| --graph-flight | the same WHILE chunk sized by `--decode-flight` ("one launch per in-flight chunk") |
| --pinned-transfers <0\|1> | download the FP32 logits and the bounded greedy token history into persistent page-locked staging buffers when `PinnedMemoryExt` exists (stream-ordered copy → synchronize → memcpy into the caller's vector). Default 1; **0 is the pageable/download-pool A/B arm**. Both realizations move the bytes twice (one DMA + one host memcpy) — no copy is claimed to disappear — and the engine prints which one it selected and why |
| --pdl <0\|1> | request programmatic dependent launch **before device creation** (`CUDADeviceConfigExt::enable_programmatic_stream_serialization`), so the backend compiles an automatic wait-at-entry prologue into every kernel. Inert below compute capability 9.0 (this box: CC 8.9 → `pdl_supported() == false`, all launches stay strict); the engine reports the probe, the reason and the relaxed/strict/driver-fallback counters. Default off; no engine kernel is PDL-eligible today, so this is measurement-only |
| --split-kv <0\|1> | DSL decode attention with GQA dedup (default 1): one block per `(kv_head, key_split)` stages K/V **once** for all four q_heads of that KV head and a combine kernel applies the sigmoid gate. `0` keeps the reference per-q_head kernel for every context (A/B) |
| --split-kv-threshold <n> | `kv_len` at which the split family takes over (default 512). Shorter contexts keep the per-q_head kernel. The threshold is a **compile-time guard on both kernels**, so one captured CUDA graph crosses it with zero updates |
| --split-kv-max-splits <n> | compile-time grid bound (`blocks = kv_heads * n`); 0 derives it from `2 blocks/SM * sm_count / kv_heads` (clamped to `[1, 40]`), else the documented measured default (40); `n > 40` is refused fail-closed and the per-q_head kernel stays in charge |
| --gdn-decode-fused <0\|1> | fuse the decode GDN depthwise conv + SiLU + ring update with the QK L2 norm into **one** launch (default 1). The Q/K half of the conv output never round-trips through global memory and 24 launches/token disappear. Bit-identical to the pair on CUDA (last-bit envelope on DX/VK); `0` keeps the reference two-launch pair for A/B |
| --gdn-fused-recurrent <0\|1> | fuse the decode GDN recurrence with its **gated output RMSNorm** into ONE one-CTA-per-head launch (opt_plan/03 §5), so a head row never round-trips through global `_gdn_out` and the second launch disappears. One 128-thread block owns a whole V head, reduces the norm block-wide (per-lane `o²` → 32-lane warp trees → a fixed 4-term fold, one barrier), then applies the **direct** norm weight and the SiLU(z) gate in registers. Model geometry only (T=1, head_dim 128); refused fail-closed otherwise, never selected for prefill (`rows > 1`), MTP/verify or calibration. **Default 0**: implemented, differentially tested on cuda/dx/vk and measured as a small end-to-end LOSS — see the phase-9e section. `1` reproduces the measurement; `info().gdn_recurrent_norm` + `gdn_recurrent_norm_reason` report what ran and why |
| --gdn-state-specialized <0\|1> | compile the GDN state-column kernels (WY/UT chunk-64 prefill and the per-token recurrent fallback) with a **compile-time** `head_dim = 128` instead of a runtime extent (default 1). Removes every runtime index of the lane-owned 128-float state column, so it lives in registers instead of local memory. `0` selects the generic kernels (A/B, differential tests) |
| --gdn-auto <0\|1> | per-prompt GDN prefill selection (default 1). The exact per-token recurrence runs for prompts `<= --gdn-auto-threshold` and WY/UT chunk-64 above it, using the **measured** crossover; `0` falls back to the fixed schedule |
| --gdn-auto-threshold <n> | prompt length where `--gdn-auto` switches to WY/UT (default 8192 = the largest prompt measured; nothing above it is extrapolated) |
| --gdn-wy-ut <0\|1> | pin the GDN prefill schedule explicitly (1 = WY/UT chunk-64, 0 = exact per-token recurrence) and disable `--gdn-auto`. `--gdn-recurrent` is the same as `--gdn-wy-ut 0` |
| --decode-packed <0\|1> | 128-bit (`uint4` weight / `float4` activation) decode GEMV, fused FFN and gate GEMV with one FP8 scale per `uint4` and a branchless LUT-free E4M3 decode. **Default 0**: implemented, differentially tested and REJECTED by measurement (see below) |
| --decode-packed-fp8 <0\|1> | force the 128-bit realization for FP8 weights too, so the FP8 rejection stays reproducible end to end (default 0) |
| --attn-prepare-model <0\|1> | model-specialized decode attention prepare (compile-time `head_dim` 256 / `rope_dim` 64 / 16 q / 4 kv heads, one register cache per lane, one RoPE pair per lane; default 1). `0` selects the generic kernel |
| --rmsnorm-reg <0\|1> | register-cached one-pass RMSNorm for `hidden = 2560` (default 1): each thread caches its 10 elements plus the norm weight, so the normalization does not re-read `x`/`sum_out` from global memory. `0` selects the generic wide kernel |
| --temperature <t> | 0 = greedy; positive finite value = categorical sampling |
| --seed <n> | unsigned 64-bit sampling seed |
| --max-context <n> | KV capacity (default 4096 DSL / 64 Tile, ≤ 262144) |
| --kv-page-bytes <n> | KV sparse allocation unit in bytes; 0 = 2 MiB default; must be a whole multiple of the backend sparse tile (see the sparse-cache section) |
| --kv-dtype <f32\|bf16\|fp8> | stored K/V dtype (**f32** default). `bf16` stores two round-to-nearest-even BF16 codes per 32-bit word, `fp8` four E4M3FN codes per word plus one FP32 scale per (kv_head, 64-position block) -- both in the plan's 64-position blocked layout (`((head * position_blocks + position/64) * 64 + position%64) * head_dim + channel`): half the KV bytes and half the KV read traffic per token, FP32 activations/accumulators/softmax unchanged, and the page geometry follows the dtype (2 MiB covers 512 positions/token of FP32 payload but 1024 at BF16). FAILS CLOSED, never silently: a geometry the contract rejects (odd `head_dim` at BF16, `head_dim % 4 != 0` at FP8, FP8 outside the blocked layout), a backend that cannot host it, or the Tile engine (which has no blocked-layout realization) is an explicit `unsupported`/`invalid_argument`; the `kv dtype:` line and `QwenEngineInfo` report what was actually built. `--kv-dtype` is also a `scripts/benchmark_qwen.py` flag, where it halves the KV term of the documented byte/capacity model. |
| --prefill-chunk <n> | FIXED prompt rows per engine chunk (default 64 DSL / 1 Tile). Passing it disables the adaptive rule |
| --prefill-chunk-max <n> | ceiling of the adaptive chunk (default 1024; effective value is min(this, per-backend limit), reported after the prefill) |
| --legacy-prefill-gemm | DSL: run the reference one-thread-per-(m,n) prefill GEMM instead of the default weight-reusing warp-per-column GEMM; differential/A-B path only |
| --native-tensor-gemm <0\|1> | CUDA tensor-core prefill GEMM through `NativeShaderExt`/`CUDA_NVRTC` (`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, FP32 accumulate, double-buffered shared A/B tiles, `cp.async` for BF16 weights and register staging for FP8). Takes every prefill projection with `rows >= 16` and `N >= 256` on cuda, for **both** weight formats. **Default 1** (measured: 6.7-7.0x TTFT on BF16 and 5.9x on FP8 at P1024/P4096, greedy ids identical); `0` is the A/B arm and the fallback. Fail-closed at `create()` when the extension, NVRTC or a projection geometry is unsupported |
| --native-tensor-attention <0\|1> | CUDA tensor-core prefill **attention** through `NativeShaderExt`/`CUDA_NVRTC`: a real FlashAttention-2 kernel (`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` for QK^T **and** P.V, FP32 online softmax, FP32 accumulators, sigmoid gate epilogue; Q/K/V rounded FP32->BF16 on the shared load, 16-token query tile, 32-key key tile, the 4 q heads of a kv head packed into one 8-warp block). Takes every prefill chunk on cuda with the model geometry (head_dim 256, 16 q / 4 kv heads) and `rows >= 32` (the measured crossover; a shorter chunk keeps the DSL tiled kernel). **Default 1** (measured: 7.9x faster than the DSL tiled kernel at S=1024 and 10.1x at S=4096; TTFT 227.6 -> 204.4 ms at P=1024 and 920.4 -> 673.1 ms at P=4096, greedy ids identical); `0` is the A/B arm. Fail-closed at `create()` (extension, NVRTC, geometry) and the DSL tiled kernel keeps prefill attention; decode and the static graph are never routed here |
| --gdn-recurrent | use recurrent DSL GDN prefill instead of WY/UT |
| --legacy-prefill-attn | DSL: run the untiled reference prefill attention (one thread per (token, q_head), runtime `head_dim` local accumulator) instead of the default tiled FlashAttention-style kernel; differential/A-B path only |
| --projection-pack <dir> | consume an **offline merged input-projection pack** (opt_plan/03 §3 + D5; bf16/fp8/q8_0 variants only — `dsl-w4` refuses one). The pack's 32 merged matrices REPLACE their 120 source tensors at upload, so the device holds no duplicate; the source identity (checkpoint size + rolling digest of exactly the merged source tensors), every payload size/hash, the segment geometry and the format are verified fail-closed before the first upload. Decode then runs ONE GEMV per projection group instead of one per projection. See "Offline merged input projections (phase 9a)" |
| --merged-decode <0\|1> | A/B arm on a pack-only weight set: `0` keeps the ordinary per-projection decode GEMVs on the pack's segment views (default 1; without a pack the flag changes nothing) |
| --export-projection-pack <dir> | run the OFFLINE packer (no device, no request) and exit |
| --export-projection-format <bf16\|fp8\|q8_0> | pack format for `--export-projection-pack` (default bf16). The format must match the checkpoint (`bf16` needs BF16 matrices, `q8_0` needs a native Q8_0 GGUF); W4 is refused (no 4-bit merged realization) |
| --mtp | opt into real DSL NextN draft+verify; fails without complete draft weights |
| --draft-depth <n> | drafts per round (default 2); verify needs n+1 workspace rows |
| --benchmark | commit + synchronize after every kernel/transfer and print raw wall intervals |
| --benchmark-csv <file> | benchmark mode and append raw samples to UTF-8 CSV |
| --mtp-test | host-only GGUF NextN inventory/policy probe |
| --self-test | device-free operator, tokenizer, and policy tests |
| --list-versions | offline support matrix |
| --headless | accepted for gallery convention; no window is opened |

Prefill GEMM: weight reuse across prompt rows
--------------------------------------------
The DSL prefill (`rows > 1`) uses the weight-reusing GEMM
`make_qwen35_gemm_mt_kernel` by default; `--legacy-prefill-gemm` selects the
reference `make_qwen_gemm_kernel` for A/B. The reference kernel runs one thread per
`(m, n)` output with a serial K loop, so adjacent lanes own adjacent output columns and
read weight rows `K*2` bytes apart (one packed word per 32 B sector), and the whole
matrix is re-streamed once per 16-row block group. The default kernel instead gives one
32-lane warp one output column (K split across the lanes, so consecutive lanes read
consecutive packed words) and one block `warps` columns and an `MT = 16`-row stripe of
m. Each weight word is decoded once and reused by all MT rows in registers, so weight
traffic is `weight_bytes * ceil(m / MT)` instead of `ceil(m / 16)` with 8x sector
amplification. FP32 accumulation, the same `[M, N]` layouts, the same `none`/`residual`
epilogues and the same ten kernel arguments mean the executor only switches the
factory. Decode (`rows == 1`) keeps its GEMV/decode-GEMV specializations untouched, and
MTP verify/replay use this family for both paths so their numerics stay aligned.
The family's shape contract is validated by the host planner and fails closed to the
reference GEMM at create time: `K` must be a multiple of the packed width (2 for BF16,
4 for FP8) and the FP8 group a multiple of 4, i.e. every weight row starts on a packed
word and one word never straddles a scale group. The model's K values
(2560/4096/9216) all satisfy it. Ragged M and N are supported (`m` tail clamped, last
column block guarded). One DX-specific note is recorded in the kernel: all MT warp
reductions are issued before any lane diverges and use the explicit butterfly sum
(`warp_read_lane`), because on HLSL a wave intrinsic issued after a divergent region
can run with the other lanes masked off (that produced wrong rows for `m >= 2` on dx
during development).
Measured isolated (`benchmark_qwen_decode <backend> --prefill-gemm`, N = 9216,
K = 2560, BF16, median): legacy 4.18 / 15.89 / 62.64 ms vs new 0.59 / 2.20 / 9.17 ms at
M = 64 / 256 / 1024 on CUDA (7.1x / 7.2x / 6.8x), and the same shape on DX and VK gives
6.7-7.3x. Effective weight-streaming bandwidth rises from ~48 GB/s to ~325-390 GB/s
against the 634 GB/s measured DRAM roof, i.e. the kernel is no longer bandwidth bound
at these shapes: it is bound by the per-row activation loads (each word's elements are
re-read MT times), which is why `opt_plan/09` N2 (vectorized `uint4`/`float4` loads)
and shared-memory x staging are the documented next step rather than a bigger MT.

Prefill geometry: ABI/contract
------------------------------
* K/N shapes: `K` must divide the packed width and (FP8) the group must divide it;
  everything else stays on the reference GEMM.
* `M` is arbitrary; the last `MT` stripe clamps its rows and the last column block
  guards `col < n`.
* Reduction order differs from the reference kernel (K split across lanes, one warp
  reduction per row), so results agree to FP32 reassociation (measured max |diff| ~2e-4
  against a value scale of ~80 at N = 9216, K = 2560).

Adaptive outer chunking
-----------------------
Without `--prefill-chunk` the DSL engine picks the outer chunk per prompt:
`chunk = clamp(next_pow2(prompt_rows / 4), 64, min(prefill_chunk_max, backend_limit))`,
then splits the prompt into `ceil(rows / chunk)` chunks (the last one may be ragged).
`prefill_chunk_max` defaults to 1024 and the per-backend ceiling is 1024 on cuda/vk and
512 on dx (D3D12 tolerates fewer nodes per submit); an unknown backend gets the smallest
limit, not the largest. GDN WY/UT arithmetic is unchanged (it always processes 64-token
sub-chunks internally) and attention is the dynamic-length tiled kernel, so a larger
chunk costs no extra shader compiles. The workspace is sized for the largest chunk the
rule can select, and the engine reports both the configured and the actually selected
chunking once per run:
```
prefill chunking: mode=adaptive configured_chunk=1024 configured_max=1024 backend_limit=1024 selected=256 chunks=4
prefill chunking: mode=fixed configured_chunk=64 configured_max=1024 backend_limit=1024 selected=64 chunks=16
```
`QwenEngineInfo` carries `prefill_chunk_adaptive`, `prefill_chunk_max`,
`prefill_chunk_backend_limit`, `prefill_chunk_selected` and `prefill_chunks`;
`scripts/benchmark_qwen.py` parses the line (validating `selected <= min(max, limit)` and
`mode=fixed => selected == configured_chunk`) and prints it in the per-arm summary.
Measured at P = 1024 (CUDA, dsl-bf16, greedy ids identical): chunk 64 2228 ms, chunk 256
2082 ms, chunk 1024 2115 ms — the rule's 256 is within 1.6 % of the best fixed chunk, so
the adaptive chunk is a second-order win next to the GEMM change (see the phase 3
record).

Activation workspace: aliased arenas
-----------------------------------
The executor used to hold ~27 `[chunk, width]` activation buffers simultaneously
(~116 320 floats per chunk row, ~454 MiB at chunk 1024). Most are never live at the same
time, so `qwen_workspace_geometry` / `qwen_workspace_members` (`qwen/dsl_ops.h`) now
publish an explicit aliasing plan: ten arenas, each one device buffer, and every logical
buffer is a handle copy of its arena. Aliasing is only allowed between members that
cannot overlap in the recorded command order — with one documented exception pair: a
GDN-only and a full-attention-only buffer never coexist because a layer is exactly one of
the two. `QwenWorkspaceMember` carries each member's arena and step interval so the host
test re-verifies disjointness instead of trusting a comment; `_gdn_beta` and `_gdn_g`
deliberately keep their own tiny arenas because the decode path fuses the gates kernel
*before* the conv, which widens their live range past every other GDN buffer. Aliasing
by handle copy means the backend still sees one resource, so its automatic barriers stay
correct, and every buffer handle the executor holds is stable (sparse KV is untouched).
The pool is 44 096 floats per chunk row plus the three permanent `[chunk, hidden]`
streams (hidden/normed/residual are never aliased: the residual stream is live across a
whole layer). Total workspace at chunk 1024 drops from ~454 MiB to ~203 MiB (2.2x), and
the engine reports the split:
```
workspace: total=203.23 MiB chunk=202.25 MiB pool_per_row=44096 floats
```
Greedy ids are identical at fixed chunk 64 / 256 / 1024 and at the adaptive chunk, i.e.
no aliased stage is overwritten while live.

Prefill attention: tiled vs reference
------------------------------------
The DSL engine's model path uses a tiled, causal, GQA FlashAttention-style kernel
(`make_qwen35_attn_prefill_tiled_kernel`) by default and keeps the original
`make_qwen_attn_prefill_kernel` as the differential/A-B reference
(`--legacy-prefill-attn`).

What changes is structural, not numerical: the reference kernel runs one thread per
`(token, q_head)` and keeps its online-softmax accumulator in a `Local<float>{1024}`
indexed by a *runtime* `head_dim`, i.e. 4 KiB of dynamic local memory per thread that
no compiler can register-allocate. The tiled kernel makes `head_dim = 256`,
`q_heads = 16` and `kv_heads = 4` compile-time constants and every array extent and
index a compile-time constant, so Q, the running max/sum and the output accumulator
live in registers. One 128-thread block serves one (query tile, KV head): its four
warps are the four q heads of that KV head, so all of them reuse one shared-memory
K/V tile (GQA dedup) and the score reduction is a warp-level butterfly sum
(`warp_read_lane`). The online softmax keeps the reference kernel's running
max/sum/output contract, the `[position, kv_head, head_dim]` cache layout and the
`out = softmax(qK^T·scale)·V · sigmoid(gate)` epilogue, so results differ only by
FP32 reassociation. `tokens`, `pos0`, `kv_len` and `capacity` stay runtime scalar
arguments: one compiled shader serves every prompt length, position, ragged tail,
cache capacity and defensive `kv_len` clamp, so there is no per-position JIT. Kernel
K/V tile extents are the measured ones (q_tile 2, k_tile 2, 4 KiB of shared memory);
larger tiles measured slower because the unrolled body already reaches the 255-register
ceiling and loses occupancy.

The choice is fail-closed through `QwenBackendCaps`: a device that does not report a
32-lane warp (or cannot host the tile) keeps the reference kernel, and the engine line
states which one is live:

```
engine: kernel=dsl ... gdn=wy-ut-64 prefill_attn=tiled(q2k2,4096B) cache=... shaders=25
engine: kernel=dsl ... gdn=wy-ut-64 prefill_attn=legacy            cache=... shaders=24
```

`QwenEngineInfo` carries the same fields (`prefill_attn_tiled`,
`prefill_attn_q_tile`, `prefill_attn_k_tile`, `prefill_attn_shared_bytes`). MTP forces
the reference kernel for verify/replay parity, exactly like `dsl_gdn_wy_ut`. Decode attention kept the per-q_head kernel in that phase; the split-KV +
GQA-dedup family below took over for long contexts in Phase 5.

Measured on the target box (RTX 4080 SUPER, CUDA, S = 1024, full causal window):
legacy 28.2 ms vs tiled 2.31 ms = 12.2x; isolated arm in
`benchmark_qwen_decode --prefill-attention`. The tiled kernel reaches ~1.9 TFLOP/s of
FP32 (≈4 % of the SKU's FP32 peak) because it is occupancy/latency bound, not
bandwidth or shared-memory bound — that gap is what the tensor-core item in
`opt_plan/09` N1b targets.

Long prompts and prompt-id transport
-----------------------------------

A prompt is exactly one of `--prompt`, `--tokens`, `--tokens-file` or `--synthetic-prompt-len`; passing two is a bad-argument error naming the count found, never a silent preference. Every id is validated against the loaded vocabulary with the offending index (`prompt token 4096 has id 300000, outside the loaded vocabulary`), and the prompt plus `--max-new-tokens` must fit `--max-context`.

The two new forms exist because a Windows command line is capped at 32767 characters and an id list costs about six characters per token, so a prefill-length sweep with P >= 8192 cannot be expressed as `--tokens`. `--synthetic-prompt-len <n>` regenerates exactly the list `scripts/benchmark_qwen.py`'s `synthetic_prompt_ids(n)` produces (the constants and rule are duplicated in src/qwen/qwen/prompt_ids.h, documented in the option help, and cross-checked by both the C++ host test and the Python unit test); `--tokens-file` reads a file of decimal ids. For a prompt longer than 64 ids the CLI prints the count and only the first eight and last two ids, so a long sweep does not flood the log with a 50 KiB line.

Bandwidth calibration
---------------------

`benchmark_qwen_decode <backend> --calibrate` is the achievable-DRAM-bandwidth measurement every "percentage of roof" needs (opt_plan/06 section 3 step 3, 08 section 7 step 7). It runs a real 1 GiB device-to-device copy (512 MiB moved per iteration, so 1 GiB of read+write traffic) and a real FP32 triad `a = b + s*c` over three 256 MiB arrays (768 MiB of traffic), 30 iterations by default, then prints min/median/p95 milliseconds and GB/s per kernel. Host initialization happens once outside the timed region, and both results are read back and compared against the host pattern afterwards, so a compiler that elided the work fails loudly (exit 3) instead of reporting an impossible bandwidth.

The measured numbers on the current box (RTX 4080 SUPER, 16 GiB, driver 621.01, CUDA capability 8.9):

| kernel | traffic per iteration | median | min | p95 |
|---|---:|---:|---:|---:|
| device-to-device copy | 1 GiB | 624.6 GB/s | 629.1 GB/s | 508.7 GB/s |
| FP32 triad | 768 MiB | 660.8 GB/s | 666.3 GB/s | 567.9 GB/s |

The `CALIBRATION summary` line prints `achievable_GBps`, the better of the two medians (660.8 GB/s here); pass that value to `scripts/benchmark_qwen.py --bandwidth`, which is the only denominator the harness will use for a roof percentage (without it the column is `n/a`, never a vendor figure). Raw output: `.kimix_cache/qwen_benchmark/phase1/`.

Harness tier/device stamping
----------------------------

`scripts/benchmark_qwen.py` stamps `--tier t16|t32` and `--device-tag <sku>` into every sample, arm record and summary, refuses a prompt or context the tier cannot hold and refuses `--report-md` (or a `--report-from` summary) that does not carry both, so a T16 and a T32 number can never share a table. It also samples `nvidia-smi` for utilization, SM clock, power and temperature before/during/after a run, refuses to launch when another tenant is on the GPU (`utilization.gpu > --gpu-idle-util-pct`, `--no-idle-check` downgrades that to a recorded warning) and records the before/after throttle probe. Prompts that do not fit the command line are transported as `--synthetic-prompt-len` (sweeps) or a generated `--tokens-file` (an explicit long `--tokens` list), so a P = 8192..32768 sweep is now launchable on Windows.

Capacity is checked per (tier, weight format), because opt_plan/08 section 5 gives one verdict per configuration, not one per tier: solving its budget (weights + FP32 KV at 64 KiB/token + 0.15 GiB fixed, against 15.0 / 31.0 GiB usable) means T16 holds 115015 tokens in BF16, 174981 in FP8/Q8_0 and 209715 in W4, while T32 holds the model's native 262144 in every format. Both the prompt plus `--max-new-tokens` and the `--max-context` actually passed are validated, so a 128-token prompt with `--max-context 262144` is refused on T16 instead of allocating 16 GiB of KV, and a version whose weight format the harness does not know fails closed to 32768 tokens on either tier rather than running ungated (`dsl-q8` is implemented since phase 8a and `dsl-w4` remains the seam for 05 Q2). The dry run prints the ceiling the gate applied to each requested version.

The decode byte model behind the report's `roof %` counts the FP32 KV cache with the **read amplification of the decode attention that actually ran** (opt_plan/03 sections 4.1/4.2): 4x for the legacy per-q_head kernel (one block per q_head, so the four q_heads of a KV head each stream the same K/V rows) and 1x for the split-KV + GQA-dedup family. The harness only claims the 1x term when the engine's own `decode attn: active=split` line says the split family was compiled **and** the measured step's context reached that engine's reported `threshold`; a legacy engine, `--split-kv 0`, a short context or a log without the line keeps the 4x term (the `DECODE_KV_READ_AMPLIFICATION` constant, unchanged, so reports stored before Phase 5 keep their meaning). The per-arm `roof` block in the summary JSON records `kv_dedup`, `kv_read_amplification`, the threshold and the whole parsed `decode attn:` line, so the number can never be attributed to the wrong kernel.

Backend capability record
-------------------------

The engine prints `QwenBackendCaps` once at Info level (warp size, max shared memory, memory granularity, total device memory, sparse tile and KV page geometry, CUDA-graph capabilities, PDL, async copy, split barrier, cooperative vector/matrix) and the harness pastes it into the run log. Four fields requested by opt_plan/04 section 7 are present but **unqueryable in the pinned LuisaCompute**: the CUDA architecture and toolkit version, the L2 cache size, cluster/cluster launch control, and Vulkan's `maxMemoryAllocationCount`. The CUDA backend's `device.query()` implements only `device_name`, `shader_bfloat16`, `total_memory` and `free_memory`; DX and VK implement a similarly short fixed list, no extension exposes the other four values, and an unknown key logs a warning. They are therefore reported as `unknown`/0 with an explicit note, and `total_memory` (the one real key that distinguishes a 16 GiB from a 32 GiB tier) is recorded. A 0 or `false` there means "not queryable", never "the hardware lacks it"; the harness records the compute capability and toolkit from `nvidia-smi`/nvcc and the tier explicitly.
Two corrections landed in phase 9c: the four unqueryable fields are unchanged, but
`cooperative_vector` / `cooperative_matrix` no longer read `false` on a device that negotiated them -- they
come from `VulkanExtensionStatusExt::cooperative_status()` (a `patches/luisa-vk-cooperative-extension-status.patch`
addition, see the Phase 9c section), which also reports whether the only configuration the codegen can emit
(`FLOAT32`, `transpose = false`) exists (`cooperative_vector_fp32_config` / `cooperative_vector_usable`) and how
many configurations the driver enumerated. `split_barrier` is now two fields as well: `split_barrier_enabled`
(the extension is on the logical device) and `split_barrier` (a compiled shader has already declared
`SPV_EXT_split_barrier`), so "enabled but not used yet" cannot be mistaken for "unavailable".

Benchmarking and command batching

Normal DSL generation batches a whole decode and all prefill chunks into stream-ordered command lists. Greedy CUDA defaults to four in-flight decode steps per host synchronization: a DSL argmax writes the next device token, and only a small token-history range is downloaded. Every CUDA variant (BF16 and FP8) captures the device-only decode body as a **static** CUDA graph: the four per-token scalars of a step (absolute position, KV length, greedy history slot, and the conv-ring position of all 24 GDN layers) live in a 16-byte device state buffer, so the recorded body is argument-invariant and is instantiated **once per session**. Subsequent tokens are a bare `cuGraphLaunch` with no whole-graph update, no per-node update and no DAG rebuild; `QwenEngineInfo::cuda_graph_updates` therefore stays 0 and `cuda_graph_steps` equals the number of graph decode launches. The in-graph argmax finish advances the device state (position + 1, `kv_len = position + 1`, slot + 1) and the host mirrors that advance for `info().position`. Chunk-boundary host token uploads, device-state re-uploads (session start, prefill, a host-logits `decode()` or a caller slot change) and sparse KV page maps remain ordinary ordered pre-graph stream commands; no map command is ever captured. KV growth is page-granular and, because the allocator never moves a region base, needs no graph update either. Capture/instantiate failures are returned as a recoverable `unsupported` status and never execute a consumed list; `reset()` keeps the captured graph and forces a state re-upload for the new session. `--no-cuda-graph` selects the ordinary-list A/B path; MTP verify/replay, `--benchmark` and the temperature/host-logits decode path never use the graph. The `decode graph: requested=... active=... cuda-graph-ext=... storage=...` line records the selection before generation and `engine final:` reports the launch/update counters. EOS is checked at each bounded chunk; any later work in that chunk is discarded. `--decode-flight 0` restores per-token full-logits download and host selection. Tile DX uses bounded commit-only lists; Tile VK retains conservative stage synchronization. `--benchmark` deliberately synchronizes every command and reports host-wall intervals, not hardware GPU timestamps. MTP adds draft proposal, hidden-history copies, snapshots, verify, possible restore/replay, and acceptance downloads, so its milliseconds per round must not be compared as though they were plain milliseconds per token.

For paired repeated ordinary-engine measurements use `scripts/benchmark_qwen.py`; raw logs and summaries are written under `.kimix_cache/qwen_benchmark/` by default. No MTP speedup or default-policy change is claimed without a real combined-checkpoint A/B that includes draft, verify, snapshot, restore, and replay costs.

Split-KV decode attention with GQA dedup
----------------------------------------
The legacy decode attention is one 128-thread block **per q_head**: each of the four
q_heads of a KV head re-streams the same K/V rows (4x read amplification) and the whole
kernel is 16 blocks, i.e. 0.2 blocks/SM on an 80-SM part. `--split-kv 1` (default)
routes `kv_len >= --split-kv-threshold` (default 512) through one block per
`(kv_head, key_split)` with 128 threads = 4 warps, one warp per q head of that KV head.
All four warps consume the same staged K/V tile, every accumulator is a compile-time
sized `Local` (`head_dim/32` floats per lane, never a runtime `Local<256>`), each warp
publishes `(m, l, acc[head_dim])` and a combine kernel reduces the `splits <=
max_splits` partials with the rescaling identity before applying the sigmoid gate.

`kv_len < threshold` keeps the legacy kernel, `--split-kv 0` keeps it for every context,
and the threshold is a **compile-time guard on both kernels** (`kv_len` itself stays a
device value), so one captured static CUDA graph holds the legacy, split and combine
nodes and crosses the threshold with **zero** graph updates - `test_qwen_mtp cuda
split-ab` asserts 64 greedy ids across the threshold and a KV page with
`cuda-graph-steps == 64, updates == 0`, and again after `reset()` + re-prefill. The grid
is `kv_heads * max_splits` blocks for every token, splits beyond
`clamp(ceil(kv_len/split_chunk), 1, max_splits)` are empty (they publish `l == 0`, which
the combine skips, so stale partials are never read), and the two partial buffers are
fixed at `max_splits` (`40 * 16 * 256 + 40 * 16 * 2` floats = 645 KiB).

The selection is measured, not guessed: the K/V tile is the **16 KiB** one (`k_tile 8`)
because occupancy, not staging, is the binding resource in a 48 KiB static-shared budget
(16 KiB lets three blocks/SM co-reside; the isolated arm measured 0.0362 ms at kv_len 128
and 0.663 ms at 32768 versus 0.0393 / 0.703 for the 32 KiB tile, and the 64 KiB tile does
not fit at all), and the split bound targets **2 blocks/SM** - the measured knee of the
isolated arm (64 blocks 0.8/SM: 408 GB/s at kv_len 32768; 160 blocks 2/SM: 639 GB/s,
against a 634 GB/s calibrated DRAM roof). The measured tile is a CEILING, not a guess: when
a device reports a *smaller* static shared-memory limit the selection halves `k_tile` until
K+V staging fits (4 KiB report -> 2 keys, 8/12 KiB -> 4 keys), and it refuses a budget below
the one-key 2 KiB floor, so a nonzero report is always honoured instead of compiling an image
the driver would reject. A 0-byte report (DX exposes no limit) and the target's 48 KiB keep
the measured 16 KiB tile. The SM count that bound needs is a **real query answer** where the
backend has one: CUDA answers `DeviceInterface::query("multi_processor_count")` with
`cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)` through
`patches/luisa-cuda-multiprocessor-query.patch` (decimal text, empty answer and no warning when
the driver call fails), so this box reports `sm_count=80` with
`sm_count_source=device_query:multi_processor_count` and `2 * 80 / 4 = 40 = the cap`, i.e. the
same grid the documented default produces (`176` split blocks = `2.2` blocks/SM, `16` legacy
blocks = `0.2`). DX and VK implement no such key, so there `QwenBackendCaps::sm_count` stays 0
with `sm_count_source=unqueryable` and the bound falls back to the documented default (40)
instead of a fabricated probe;
`--split-kv-max-splits` and `QwenEngineOptions::dsl_split_kv_max_splits` override it, and an
override above the 40-split plan bound is refused fail-closed (the per-q_head kernel stays in
charge) so the fixed partial buffers can never be indexed out of bounds.
MTP verify/replay and `--split-kv 0` force the legacy kernel (`dsl_decode_specialized`
false implies no split reduction), because those paths need one reduction order.

Measured on the target box (RTX 4080 SUPER, CUDA, BF16 GGUF, static graph, 32 new tokens,
`.kimix_cache/qwen_benchmark/phase5/`): identical greedy ids in every pair, and

| prompt | legacy ms/token | split ms/token | delta |
|---:|---:|---:|---|
| 8 | 13.38 | 13.40 | +0.1 % (two guarded no-op nodes per attention layer) |
| 128 | 14.05 | 14.08 | +0.2 % |
| 384 | 15.55 | 15.58 | +0.2 % |
| 1024 | 19.33 | **14.19** | -26.6 % |
| 2048 | 25.32 | **14.34** | -43.4 % |
| 4096 | 37.30 | **14.49** | -61.2 % |

The isolated single-query arm (`benchmark_qwen_decode --decode-attention`, one layer,
legacy vs split on the same buffers, median of 20) measures 0.0614/0.2302/0.9110/1.825/
4.457/23.80 ms for legacy and 0.0403/0.1385/0.1397/0.1410/0.1550/0.4159 ms for split at
kv_len 128/512/2048/4096/8192/32768 (57x at 32768), and Nsight Compute confirms the
mechanism with hardware counters: at kv_len 4096 the split kernel reads 33.6 MB of DRAM
and 33.9 MB through L2 for 33.6 MB of unique K/V (1.01x), while the legacy kernel reads
680 MB through L2 (20x) at 0.02 waves/SM and 8.3 % occupancy versus 0.67 waves/SM and
16.5 % occupancy. The 4x amplification in the roof model is therefore a *lower* bound on
the legacy kernel's real L2 traffic.

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

The current FP8 pass enables the same fused gate/up/SwiGLU route already used by BF16, removes DSL position upload/copy commands, and replays FP8 CUDA device-greedy decode through `CudaGraphExt`. The final paired cache-resistant FFN pipeline benchmark measured **0.08042 ms for two GEMVs plus SwiGLU versus 0.07540 ms fused**. On the real four-token prompt/eight-decode-step diagnostic configuration, FP8 improved from **8.55 to 8.03 ms/token** with identical IDs and **8 graph launches / 7 updates**; that run predates the static graph of opt_plan/03 section 2.3 (milestone M-c), which replaced the per-token `update()` with a single capture per session (`updates == 0`) and extended the same schedule to BF16. BF16 measured **13.98 ms/token** on its then non-graph path. The final FP8/BF16 request-wall ratio is **1.74x**. These are short A/B diagnostics, not a broad latency distribution; the M-c numbers are in `.kimix_cache/qwen_benchmark/phase4/`.

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

- One `SparseBuffer<float>` reserves the complete maximum-context virtual capacity. It contains sixteen page-aligned regions in `[K0,V0,K1,V1,...]` order. Each region still exposes exactly `[max_context, kv_heads * head_dim]` through a `BufferView<float>`.
- Existing kernels and numerical addressing are unchanged: token `p`, channel `c` remains element `p * 1024 + c` within a layer's K or V view. There is no block table, prefix sharing, multi-request scheduler, or non-contiguous logical token order in this single-sequence example.
- One `vector<SparseBufferHeap>` holds ONE heap per PAGE, and each page is mapped by exactly one `map_tile` call covering `tiles_per_page` backend tiles. `page_size_bytes = ceil_div(max(2 MiB, tile size, kv_page_bytes), tile size) * tile size`, so a page is always a whole number of backend sparse tiles and at least 2 MiB; `pages_per_region = ceil(logical bytes / page size)` and the region stride is padded to whole pages. Mapping several tile ranges out of one heap is not portable (DX derives RangeStart from the heap base and the mappings would alias), which is exactly why one heap is one page.
- `ensure_capacity(stream, pos0 + rows)` allocates and page-maps only newly required pages in every K/V region, all-or-nothing: every new heap of every region is allocated before the single map batch is submitted. Stable virtual views and existing page mappings preserve prior contents across growth; normal growth adds no host synchronization.
- Measured geometry at the default configuration (max_context 4096, 1024 values per token, eight full-attention layers = 16 regions): CUDA reports a 2 MiB sparse tile, so the page is 2 MiB with 1 tile per page, 8 pages per region and 128 heap slots; DX and VK report a 64 KiB tile, so the 2 MiB page floor applies and a page is 32 tiles, still 8 pages per region and 128 heap slots. A fully resident 4096-token context therefore allocates 128 heaps on all three backends instead of 4096 one-tile heaps (Vulkan `maxMemoryAllocationCount` is commonly 4096, and DX would issue 4096 D3D12MA allocations). Measured committed bytes: 128 tokens -> 16 heaps / 32 MiB, 512 tokens -> 16 heaps / 32 MiB, 4096 tokens -> 128 heaps / 256 MiB, identical on CUDA, DX and VK.
- Trade-off, stated plainly: growth granularity becomes one page = 512 tokens of context per region on DX/VK, so a 100-token session commits 1 page x 16 regions = 32 MiB instead of the previous 7 tiles x 16 = 7 MiB. It never exceeds the old worst case at any context of 512 tokens or more, and `--kv-page-bytes` (any whole multiple of the backend tile; the 2 MiB floor still applies) is the knob for a variant that needs many short sessions. Partial page mapping is not implemented because it cannot be expressed portably.
- **Stored KV dtype and layout (`--kv-dtype`, plan 05 Q3).** One contract
  (`src/qwen/qwen/kv_cache_layout.h`) decides where a stored K/V value lives, and the
  allocator, the engine, every attention kernel and the tests all read it. Two
  independent axes:
  * dtype: `f32` (legacy) keeps one FP32 code per 32-bit word; `bf16` packs **two
    RNE BF16 codes per word, low code first** (the same word ABI as the BF16 weight
    tensors) and stores exactly half the bytes. Activations, accumulators and the
    softmax stay FP32, so the only numerics change is the K/V storage rounding.
  * layout: `position_major` is the legacy `[position][kv_head][channel]`; `blocked64`
    is `[kv_head][position_block 64][position%64][channel]`, i.e. a 64-position block of
    one head is one contiguous run (the plan's block-major layout, 03 §4.3). The native
    pair is `f32 + position_major` and `bf16 + blocked64`; both cross combinations are
    legal so the dtype and the layout can be measured apart, and the engine reports
    which pair it built (`QwenEngineInfo::kv_dtype/kv_layout`, `info.caps.
    kv_elements_per_word`, and a `kv dtype:` stdout line).
  * A BF16 cache is appended by a **separate packed-word kernel** (one thread per word,
    both codes in one thread) because two adjacent channels of one position share a word
    and the prepare kernels give adjacent channels to different lanes -- a per-code store
    would be a read-modify-write race. Cost: one extra launch per prepare call
    (`QwenEngineInfo::kv_packed_append`); the FP32 arm's launch list is unchanged.
  * The CUDA tensor-core prefill attention binds the K/V cache as DENSE FP32, so a BF16
    (or FP8) KV session turns it off explicitly (logged,
    `info().native_prefill_attn == false`) and keeps the DSL tiled/reference kernel. It
    never reinterprets packed codes as floats. The Tile engine has no blocked-layout
    realization at all and refuses any non-f32 `--kv-dtype` with `unsupported` (use
    `--kernel dsl`).
  * `fp8_e4m3` is a **scaled** dtype: four E4M3FN codes per 32-bit word (the same codec
    as the FP8 weight path: RNE, saturation at 448, NaN code `0x7f`) plus **one dense
    FP32 scale per (kv_head, 64-position block)**. The scales live in a per-region
    `Buffer<float>` owned by the allocator *outside* the sparse mapping, so a scale read
    is never an unmapped read; `QwenEngineInfo::kv_scales_per_region/kv_scale_bytes`
    (and a `kv fp8 scales:` stdout line) report them and `logical_bytes()` includes
    them.
  * The FP8 append is **online and self-consistent**: one CTA per (touched
    64-position block, kv_head), K then V. It reduces max|v| over the new rows,
    computes `scale = max(old_scale, max_abs/448)` -- the scale itself carries the
    block's running maximum, so no FP32 shadow cache exists -- and when the scale grows
    it **rewrites the block's whole valid prefix**, re-encoding every earlier code from
    the scale it was written with (`value = code*old_scale`,
    `code' = round(value/new_scale)`). A non-growing append writes only its own rows
    and leaves every earlier byte intact. Each thread owns whole 32-bit words, so no
    read-modify-write crosses lanes. `make_qwen_kv_append_fp8_kernel` is the only writer
    of an FP8 cache and its host twin `qwen_kv_append_reference` is compared with it
    word-for-word by the device suite (growth, non-growth, block boundary, ragged tail,
    device-state arm).
  * The device codec is **bit-identical** to `qwen::f32_to_fp8_e4m3`: the reference
    rounds twice (first to the 2^-9 subnormal grid, then to the 3-bit significand) and
    so does the kernel, because rounding the raw FP32 value once differs exactly on the
    values whose discarded bits tip a mantissa tie. The scale is one rounded reciprocal
    multiply (`max_abs * (1/448)`) on both sides, so a fast-math build cannot move its
    last bit.
  * Cost, measured and reported: one extra launch per prepare call
    (`QwenEngineInfo::kv_scaled_append`) plus the rewrite of a grown block (up to 64
    rows = 16 KiB per head and per K/V). The A/B below quantifies whether it wins at
    long context.
- **Page geometry follows the dtype** (`01` §2.7). At the model geometry
  (1024 values per token, 16 regions) a BF16 region is 8 MiB instead of 16 MiB, so
  `pages_per_region` halves (4 instead of 8 at `--max-context 4096`) and the
  1024-heap long-context warning below moves accordingly. The *token* coverage of one
  2 MiB page must be read per layout, and the two numbers are different:
  * `f32 + position_major`: 2 MiB / (1024 values/token x 4 B) = **512 positions** per
    page (the granularity the trade-off bullet below quotes).
  * `bf16 + blocked64`: the payload is 2 KiB/token, so 2 MiB covers **1024 positions**
    of *payload*, but residency is per HEAD: each head owns a separate block array, so
    the first token already maps the page containing every head's first block
    (`kv_min_commit_pages`, 4 at the model geometry, i.e. 8 MiB per region instead of
    the FP32 2 MiB). The allocator maps exactly the pages intersecting each head's
    active blocks (a page UNION, never a linear prefix), reports the first-token page
    count in its `INFO` line, and `--kv-page-bytes` remains the knob for a variant that
    needs a finer granularity.
- **Capacity model per tier (08 §5).** The KV term halves with `--kv-dtype bf16` and
  quarters with `fp8`, so a 32 K context costs 1.0 GiB (BF16) or 0.5 GiB (FP8) after the
  GQA dedup instead of 2.0 GiB, and a 128 K context 4.0/2.0 GiB instead of 8.0 GiB: on
  T16 (15.0 GiB usable) BF16 KV is what makes 128 K fit next to 8-bit weights and FP8
  keeps it comfortable; on T32 either makes 262144 tokens fit.
  `scripts/benchmark_qwen.py --kv-dtype bf16|fp8` applies exactly that model
  (`KV_BYTES_PER_TOKEN_BY_DTYPE`, FP8 = codes/4 + 4 B/token of dense scales) to the roof
  and capacity columns.
- **Measured (phase 8d, RTX 4080 SUPER, CUDA, dsl-bf16 weights, T16, one
  sequence).** Quality: `scripts/qwen_quality_gate.py --gate q3` at 16 prompts of
  8192 tokens each (effective 8 K) reports **mean token-id agreement 1.0000, worst
  prompt 1.0000** against the `--kv-dtype f32` reference -- FP32 and FP8 KV produce
  *identical* ids on every prompt -- with KL(prefill) mean/max 0.00015, a 64-token
  repeat/NaN check on a real needle document (identical ids, no NaN, non-degenerate)
  and a needle-style long-context set of four ~8223-token text documents at depths
  0.10/0.19/0.40/0.69 (the candidate retrieves at least as often as the reference; the
  reference itself answers 2/4 and the candidate matches it exactly). Device evidence:
  the codec/append/attention suites (2220 checks, including non-finite-only blocks and long-generation verdict integration) pass on cuda, dx and vk, and a short
  real-GGUF greedy run gives identical ids for f32/bf16/fp8 KV on all three backends.
  Performance (`scripts/benchmark_qwen_kv.py`, medians, interleaved A/B): decode at
  8 K context f32 14.46 / bf16 14.47 / **fp8 16.34 ms/token**; at 32 K f32 16.46 /
  bf16 15.77 / **fp8 18.79 ms/token**; prefill (TTFT) at 8 K 1395 / 2747 / 3017 ms and
  at 32 K 17081 / 44779 / 43914 ms. So on this GPU **FP8 KV is a capacity win, not a
  latency win**: the KV read is not the bottleneck for one sequence at these contexts
  (weights are), the online-scale append costs one launch per prepare call plus its
  block rescales, and every non-f32 KV dtype loses the CUDA tensor-core prefill
  attention (BF16 pays that too). Hoisting the per-key scale loads out of the K/V
  staging loops cut the FP8 decode regression against the f32 baseline from
  +27 % to +13 % at 8 K and from +35 % to +14 % at 32 K (against bf16 it is
  +13 % and +19 %); the residual is the append/scale work. FP8 KV therefore
  stays **opt-in** (BF16 remains the fallback) while the payload shrinks 4x
  (518 -> 130 MiB at 8 K, 2053 -> 513 MiB at 32 K).
- Reset does not clear maximum-context storage. After synchronizing the last possible use, it unmaps every resident page run of each region (one coalesced `unmap_tile` per contiguous run; blocked64 can have one run per head) and moves the retired heaps into a `CommandList` completion callback. The sparse virtual buffer remains available for remap. Engine teardown also drains the retained creation stream before any child resource is released.
- All calls on an engine or NextN draft must use its creation stream. Sparse-mediated execution/reset paths reject a different stream rather than risking an unmapped in-flight cache. Backends without sparse-buffer support fail closed; no dense fallback is silently selected.
- Long contexts on Vulkan: 4096 heaps is a soft ceiling rather than a device query. When `region_count * pages_per_region` exceeds the commonly reported `maxMemoryAllocationCount` of 4096, `create()` warns and names `kv_page_bytes` plus the value that would fit. Measured: `--max-context 262144` at the default page needs 512 pages per region x 16 regions = 8192 slots (16 GiB virtual, nothing committed) and the warning asks for `kv_page_bytes` of 4194304 B, which gives 256 pages per region and exactly 4096 slots. The allocator never clamps and never fails: the page model already reduces the default requirement 32x.

The CUDA backend fix in `patches/luisa-cuda-sparse-buffer-tile-offset.patch` is required for more than one tile per buffer: `cuMemMap` now receives the tile-adjusted destination address and a zero offset into the mapped heap. The previous call mistakenly applied the tile offset to the heap and left the destination at the buffer base. CUDA reports a 2 MiB tile, which is exactly the page floor, so one heap covers one page with a single-tile mapping there; the page model only changes the heap COUNT, not the mapping shape.

Memory and lifetime

Approximate device memory for one sequence:

| Item | Size |
|---|---|
| BF16 weights | ≈ 7.9 GiB (426 tensors; tied embedding counted once) |
| FP8 E4M3 weights + scales | ≈ 4.1 GiB |
| GDN recurrent state | 48 MiB fixed |
| GDN conv history | 2.25 MiB DSL; 4.5 MiB Tile ping-pong rings |
| Full-attention KV cache | FP32: 64 KiB/token across eight layers (256 MiB at 4096). BF16: 32 KiB/token (128 MiB). FP8 E4M3: 16 KiB/token (64 MiB) plus dense per-(head,64-block) scales. BF16/FP8 add bounded 64-position tail padding; physical code residency is committed on demand in 2 MiB pages, one sparse heap per page |
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
build\windows\x64\release\test_qwen_mtp.exe cuda bf16-zero
build\windows\x64\release\test_qwen_mtp.exe cuda graph-ab
example_qwen --self-test
```

Sparse-cache validation: **78 allocator checks pass on each of CUDA, DX, and VK**, including multi-tile writes, growth with preserved data, reset/remap, rejection of a foreign stream, reset/destruction after dispatched work, the page-geometry invariants, the no-alias probe over every (region, page), a page-straddling write, page-crossing growth, reset/remap page counts, the envelope guards, and byte-identical results for the one-tile-page and default paged configurations. The same suite measures the engine geometry above: 32 MiB committed at 128 and at 512 tokens, 256 MiB at 4096 tokens, 128 heap slots, and it checks the long-context geometry of the 4096-slot warning. Real BF16 inference passes through sparse K/V on DSL CUDA (including four in-flight greedy forwards) and reproduces identical IDs with `--kv-page-bytes 4194304` (2 tiles per page, 4 pages per region, 64 heap slots), which is the end-to-end proof that the page geometry does not change results; Tile DX matched IDs on a 16 GB card but fails `DXGI_ERROR_DEVICE_REMOVED` at pipeline creation on this machine's 8 GB GPU, identically with the legacy tile-granular allocator, so that is an environment limit and not the page model; the CUDA synthetic MTP suite passes **64 checks**. The **static-graph** probes need no host quantizer and no real weight payload: `fp8-zero` and `bf16-zero` upload an all-zero trunk and add **13 checks each** (engine creation, prefill, four graph launches, `cuda_graph_updates == 0`, host position accounting, deterministic history), which is what proves the schedule is format-agnostic. `graph-ab` runs the REAL BF16 trunk twice (graph on / graph off) for 64 greedy tokens from a 504-token prompt -- i.e. across the 512-token KV page boundary, where the graph-external `ensure_capacity` path engages -- then decodes two host-logits tokens and four more device-greedy tokens, then resets, re-prefills and decodes eight more: **218 checks**, all ID lists identical between the arms, `cuda_graph_steps == 64` and `cuda_graph_updates == 0` on the graph arm. `test_qwen_ops_dsl` adds **40** device-state differential checks (GDN conv ring over six successive positions vs. the CPU recurrence, attention prepare/decode at three positions, and the argmax finish state advance incl. the NaN sentinel and the out-of-range slot guard) plus **88** split-KV decode-attention checks (three-way CPU oracle / legacy / split at kv_len 1, 511, 512, 513 and capacity with a 16-way reduction, the host-scalar and device-state realizations byte-identical, the guarded pairing leaving the split partials untouched below the threshold, empty splits publishing `l == 0` and being skipped, an all-empty reduction staying a finite zero, the gate ratio, the reported shared-memory budget sweep 2/4/8/12/16/48 KiB and just below the 2 KiB one-key floor with `shared_bytes <= limit` and a valid planner in every case, the same for an overridden `max_splits` incl. the cap and the refusal of an over-budget tile, and a `k_tile 2` realization actually compiled and compared against the oracle) - **1745 checks on each of cuda, dx and vk** (was 1569; **+176**, all new: the vectorized 128-bit decode realizations and the model-specialized attention prepare). `test_qwen_mtp cuda split-ab` runs the REAL BF16 trunk twice (`dsl_split_kv` on/off) for 64 greedy tokens from a 504-token prompt, i.e. across the split threshold and a KV page, then resets, re-prefills and decodes 8 more: **204 checks**, identical ids between the arms, `cuda-graph-steps == 64, updates == 0`, the reported threshold/realization/dedup agreeing with the kernels that ran (203 checks on dx and vk, where the steps stay 0 because those backends use command lists). The KV-dtype migration (05 Q3) adds, on top of that list, a device-free
`kv contract:` stage (BF16 RNE ties/subnormals/NaN/inf, the blocked region arithmetic,
address injectivity and bounds over a ragged 19250-position context, and the page/token
coverage numbers) plus a `bf16 geometry:` device stage in `test_qwen_kv_cache`
(dtype/layout accessors, the per-head page-union residency at 1/64/65/5000/17000/19250
positions, per-head no-alias markers, data preservation and reset/remap, and the FP32 A/B),
and a three-way `kv f32 / kv bf16 / kv f32-vs-bf16` differential in `test_qwen_ops_dsl`
covering prefill (legacy + tiled) at pos0 0/3/61/64/640, decode at kv_len
1/63/64/65/511/512/513/640/650 (legacy + split-KV) and a word-exact check of the packed
append (including the device-state arm used by the static graph). Measured result: **110 allocator checks and 2042 DSL checks pass on each of cuda, dx and vk**. Real BF16-weight / BF16-KV inference at P128 produced the same 64 greedy ids as FP32 KV on cuda and the same first 16 on dx/vk; CUDA static graph steps=63, updates=0. At max_context 8192 the KV payload is 256 MiB instead of 512 MiB. Performance is deliberately not hidden: P4096 TTFT was 671.0 ms FP32+native-attention vs 1005.6 ms BF16+DSL-attention, and decode 14.43 vs 14.70 ms/token; P128 was 292.9/13.97 vs 275.9/13.81 in this one-process pair. Native attention currently supports only the FP32 layout and the packed append adds eight launches per token, so **FP32 KV remains the default**; BF16 is the explicit capacity/long-context arm, not a claimed universal latency win. Raw logs: .kimix_cache/qwen_benchmark/phase8c/parent/.

Final optimized validation: **473 DSL operator checks pass on each of CUDA, DX, and VK**, and **36 MTP checks pass on each backend**. Host suites pass with 470 engine assertions, 367 speculative assertions, 223 GGUF assertions, and 3164 weight assertions; the offline self-test also passes. A real 33-token chat run produces identical IDs with `--decode-flight 0` and `4`. The final relinked BF16 executable reproduces 13.970 ms/token and 14.5–14.6 ms TTFT in the benchmark/smoke configuration.

Known limitations

- Text only: no vision tower. Sampling currently has temperature only, not top-k/top-p.
- Qwen now requires a sparse-buffer-capable backend. Residency is page-granular: a session that uses 100 tokens still commits one 2 MiB page in each of the 16 trunk regions (32 MiB), where the old one-heap-per-tile model committed 7 MiB. A full 4096-token context allocates 128 heaps, and a context beyond roughly 260k tokens needs `--kv-page-bytes` of 4 MiB or more to stay under Vulkan's commonly reported 4096-allocation limit (the allocator warns with the exact value).
- The supplied GGUF has no NextN weights. Real-draft output quality and end-to-end acceleration are unmeasured; MTP is opt-in and the current draft re-prefill policy may be slower than plain decoding.
- Tile GDN remains recurrent rather than WY/UT. Tile MTP is not implemented and never falls back silently to DSL.
- Tile VK short inference is fixed and verified, but it is slow and long-run stress coverage is still needed. Tile CUDA is unavailable in this build.
- MTP currently forces recurrent target GDN and the single-row reference numerics mode for verify/replay/plain-reference parity. Combining WY/UT target verify with one-row replay requires per-row chunk-state restoration.
- FP8 remains lossy load-time E4M3 quantization of BF16, not a separately trained/validated FP8 checkpoint; token trajectories may diverge from BF16. FP32 scales, FP32 activations, and fixed non-projection kernels mean a 2× end-to-end speedup over BF16 is not implied by the nominal weight-size ratio.
- Exit codes: 0 success/help/list/self-test/mtp-test, 1 bad arguments, 2 engine/model refusal or generation failure.

### Phase 6 (M-c): model-path decode/prefill specializations

Three model-geometry specializations were added next to their generic reference kernels. Each was measured A/B on the RTX 4080 SUPER (fresh `--calibrate` roof: 626 GB/s copy, 660 GB/s triad) with the real BF16 trunk and the static CUDA graph, and each keeps the generic factory as the differential path.

**GDN state-column compile-time `head_dim` (prefill).** `make_qwen35_gdn_chunk_kernel(128)` and `make_qwen35_gdn_recurrent_kernel(128)` make every index of the lane-owned `col[]` state column a constant. The generic kernels index that `Local<float>` with a runtime `head_dim`, which cannot be proven constant and therefore lives in local memory across the whole chunk loop. Isolated chunk-64 arm (32 blocks, one launch): T=64 1.2746 -> 0.8482 ms, T=128 2.3793 -> 1.5840 ms, T=512 9.4675 -> 6.3124 ms (-33 %), T=1024 19.258 -> 13.364 ms (-31 %). In-engine, the same 96 launches of a 512-token prefill (24 GDN layers x 4 chunks) drop from 241.6 to 169.2 ms (-30 %) and TTFT from 1007.6 to 931.0 ms (-7.6 %); at 1024 tokens 1966.2 -> 1821.8 ms (-7.3 %). Output and end state agree with the generic kernel bit for bit on CUDA (last-bit envelope on DX/VK) and with the independent FP64 per-token oracle at T = 1, 63, 64, 65, 127, 128, 129.

**Register-cached one-pass RMSNorm (decode).** Each thread caches its `hidden/block = 10` elements and the norm weight, so the epilogue pass reads neither `x` nor `sum_out` again. Isolated batched arm (64 launches per sample, so the per-launch cost is measured without one stream sync each): plain 0.2281 -> 0.2063 us (-9.6 %), snapshot 0.2438 -> 0.2266 us (-7.1 %). End-to-end, 65 norm launches per token are worth 14.09 -> 13.93 ms/token on CUDA (-1.2 %) and 14.61 -> 14.47 ms/token on DX (-1.0 %). The kernel is bit-identical to the wide kernel on CUDA and inside the FP32 last-bit envelope on DX/VK.

**Fused decode GDN conv+SiLU+ring+QK-norm.** One 128-thread block per 512 conv channels, one warp per 128-channel head row: warps below `qk_heads` emit normalized+scaled Q, the next `qk_heads` emit normalized K, the rest write the V channels the recurrence consumes, and every warp advances the conv ring. This removes the `_gdn_conv` Q/K round trip and 24 launches per token. It is bit-identical to the pair on CUDA (last-bit envelope on DX/VK) at positions 0..5 and in the device-state realization. End-to-end it is **neutral on CUDA** (13.98 vs 13.97 ms/token) and a small win on DX (14.47 vs 14.51 ms/token), because a decode step already runs at 92 % of the measured DRAM roof and the two-launch pair's extra launch is hidden by the queue; it stays enabled for the launch-count property and is reachable through `--gdn-decode-fused 0` for A/B.

Two further items from `opt_plan/09` N2/N3.2 were investigated and **not** promoted; the evidence is in `.kimix_cache/qwen_benchmark/phase6/README.md` (decode is at 92-97 % of the achievable DRAM roof, so 128-bit weight loads and branchless LUT-free FP8 decode have no bytes to win on the BF16 trunk, and the FP8 decode GEMV's remaining 14 % gap needs an end-to-end FP8 run to be judged).

### Phase 6 completion: 128-bit decode realizations, model attention prepare, measured GDN schedule

Three items closed in this phase. All numbers are from the RTX 4080 SUPER (80 SMs, CC 8.9) with a
fresh `--calibrate` roof of **627.8 GB/s copy / 635.9 GB/s triad**, `benchmark_qwen_decode` for the
isolated arms and the CLI `timing:` line for the end-to-end arms.

**Vectorized 128-bit decode (N2): implemented, measured, REJECTED.** Three new realizations ship
next to the 32-bit ones: `make_qwen35_decode_gemv_packed_kernel` (M=1 projection GEMV),
`make_qwen_ffn_decode_packed_kernel` (fused gate/up/SwiGLU) and
`make_qwen_gdn_decode_gates_packed_kernel` (T=1 alpha/beta gate GEMV). They bind the packed weights
as `Buffer<uint4>` and the activation as `Buffer<float4>` (4x fewer weight-load and activation-load
instructions), read ONE FP8 scale per `uint4` (a `uint4` provably lies inside one 64-element group,
so 4x fewer scale loads), and decode E4M3 with a branchless, LUT-free bit construction that is
bit-identical to `qwen_fp8_e4m3_to_f32` for all 256 codes (verified against an independent
double-precision oracle, subnormals, signed zero and both NaN codes included). The block barrier and
the 256-entry `Shared<float>` LUT disappear. The 32-bit kernels and the generic/ragged fallbacks stay
compiled, and every geometry the packed contract cannot serve is a planning FAILURE, never a silent
fallback (`K % 512` for FP8, `K % 256` for BF16, `group % 16` and `group | K` for FP8).
Measured (median of 21 iterations, GB/s, roof 636):

| shape (N x K) | BF16 generic | BF16 32-bit | BF16 128-bit | FP8 32-bit (LUT) | FP8 128-bit |
|---|---:|---:|---:|---:|---:|
| attn_kv 1024x2560 | 511.2 | 510.3 | 503.8 | **370.2** | 344.2 |
| gdn_z 4096x2560 | 613.2 | 613.5 | **625.6** | **528.6** | 424.5 |
| gdn_out 2560x4096 | 635.7 | 632.8 | 635.0 | **583.2** | 556.5 |
| attn_q / gdn_qkv 8192x2560 | 650.8 | 651.5 | **656.2** | **613.6** | 583.9 |
| ffn_gate_up 9216x2560 | 662.1 | 661.9 | 661.7 | **636.7** | 597.1 |
| ffn_down 2560x9216 | 656.6 | 656.9 | **658.6** | **630.7** | 612.6 |
| fused FFN 9216x2560 | - | 678.5 | 678.1 | **636.1** | 597.1 |
| lm_head 248320x2560 | 693.9 | 685.7 | 693.6 | **686.8** | 686.7 |
| GDN gates 32x2560 (us) | - | 5.861 | **5.441** | **6.047** | 6.254 |

The isolated picture is a small BF16 win and a clear FP8 loss: the 32-bit shared-LUT decode beats the
branchless conversion on every FP8 model shape (-3 % on lm_head/fused FFN up to **-20 % on gdn_z**),
because a random-index shared-memory read is cheaper than a data-dependent select chain here. The
end-to-end arms confirm it and also reject BF16: real BF16 trunk, 128-token prompt, 32 new tokens,
three order-reversed pairs -> **13.90/13.91/13.93 ms/token at 32-bit vs 14.02/14.02/14.01 ms/token at
128-bit (+0.9 %)**; real BF16-source `dsl-fp8` trunk (load-time E4M3 quantization), 16 new tokens,
both orders -> **8.33/8.32 vs 8.67/8.65 ms/token (+4.0 %)**. Greedy ids are identical in every pair. The DX and VK
decode timings of this session were too unstable to decide anything (DX: 14.46-18.25 ms/token for
the SAME 32-bit configuration across three repeats, with the packed arm inside that same band), so
the CUDA numbers are the basis of the verdict and no DX/VK speed claim is made.
Both flags therefore default to **0**; `--decode-packed 1 [--decode-packed-fp8 1]` and
`benchmark_qwen_decode --impl all` reproduce the whole table.

**Model-specialized attention prepare (N3.2): PROMOTED.** `make_qwen35_attn_prepare_model_kernel` and
its device-state twin fix `head_dim = 256`, `rope_dim = 64`, 16 q heads and 4 kv heads at compile
time. Each lane owns exactly eight constant-indexed elements and the rotated prefix is exactly the
pair `(lane, lane+32)`, so the norm pass caches the eight values AND the eight norm weights in
registers, ONE `qwen_rope_pair` covers the whole rotated prefix instead of two, and the epilogue
re-reads neither `q_proj`/`k_proj` nor the norm weights -- the generic kernel reads them three times.
Layout, element order, the four-partial/warp reduction order, the `rsqrt(mean(x^2)+eps)` form, the
gate copy and the K/V append slot are identical, and the differential suite pins it against the
generic kernel at positions 0/1/5/37 for 1 and 3 tokens in both realizations (every output buffer and
the whole KV cache). End-to-end: BF16 14.02/14.03 vs 14.03/14.04 ms/token and real `dsl-fp8`
8.31 vs 8.34 ms/token, i.e. equal-or-better and never a regression; it is enabled by default
(`--attn-prepare-model 0` for A/B).

**Measured per-prompt GDN prefill schedule (C).** The exact per-token recurrence was compared against
WY/UT chunk-64 at the same prompts in both orders (real BF16 trunk, TTFT, 4 new tokens):

| prompt | WY/UT chunk-64 | exact recurrence | delta |
|---:|---:|---:|---:|
| 1024 | 1817.6 / 1820.9 ms | 1522.7 / 1519.8 ms | -16.2 % |
| 2048 | 3688.8 / 3690.8 ms | 3090.9 / 3091.5 ms | -16.2 % |
| 4096 | 7629.5 / 7627.4 ms | 6414.6 / 6430.0 ms | -15.8 % |
| 8192 | 15827.7 / 15832.9 ms | 13414.7 / 13411.6 ms | -15.2 % |

No crossover exists in 128..8192 tokens (the earlier phase measured 128..2048 at -11..-17 %), and the
greedy ids are identical in every pair. The default is therefore a **measured per-prompt rule**:
recurrent for prompts `<= 8192` (the largest prompt actually measured), WY/UT above it rather than
extrapolating. The decision is taken ONCE per prompt -- every chunk of a prompt stays on one
numerical path -- and is reported both in the `GDN prefill for N tokens -> ...` log line and as
`gdn_prefill=auto<=8192_recurrent` in the `model specializations:` line. `--gdn-wy-ut 0|1`,
`--gdn-recurrent` and `--gdn-auto 0|1` give explicit control, MTP keeps forcing the recurrence (with
`dsl_gdn_auto` disabled so the rule cannot re-select WY/UT), and the WY/UT kernel stays compiled as
the reference that the chunk differential tests validate against.
### Phase 7a: CUDA tensor-core prefill GEMM via `NativeShaderExt` (opt_plan/09 N1a)

The prefill projections no longer run on CUDA cores on the CUDA backend. A hand-written CUDA C++
kernel issued through `NativeShaderExt` (`NativeShaderLanguage::CUDA_NVRTC`, the CUDA route of
`luisa/backends/ext/native_shader_ext.h`) now does them with
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. This is the only tensor-core route that exists
in this build: `tile::mma` needs CUDA 13.4+ `tileiras`/Tile headers, which the gallery's xmake CUDA
build does not have (opt_plan/09 V1). Scope is the GEMM only — FlashAttention-2 prefill is the next
phase — and decode is untouched: the M=1 GEMV family and the static CUDA decode graph keep running the
DSL kernels, and no native command is ever recorded into that graph.

**Implementation.** `src/qwen/qwen/native_gemm_kernel.h` holds the device code as one raw CUDA C++
string (the single source of truth, also compiled by a standalone NVRTC harness during development);
`src/qwen/qwen/native_gemm.{h,cpp}` substitutes the tiling, compiles it through `NativeShaderExt`, and
binds the five buffers by reflection index. The kernel is a 128x64x32 block tile (8 warps, 256
threads, 32x32 warp tile = 2 `m16` x 4 `n8` fragments, 32 FP32 accumulators per thread) with
double-buffered shared A/B tiles: the weight tile travels with `cp.async` (16-byte `cg` when `K % 8 ==
0`, 4-byte `ca` for any other even `K`) while the activation is read from global into registers and
converted FP32 -> BF16 **on the way into the shared tile**, so the conversion latency hides behind the
previous stage's `mma`. `ldmatrix.x4`/`x2` feed the fragments, shared rows carry 4 words (16 B) of
padding to keep the `ldmatrix` alignment and spread the banks, and out-of-range rows/columns are
zero-filled so ragged M/N/K need no epilogue masking beyond the store guard.
The ABI is exactly the DSL GEMM's (`x [M,K]` FP32, `W [N,K]` packed BF16 row-per-output, `residual
[M,N]` FP32 or a dummy, `y [M,N]` FP32), so the same `QwenDeviceTensor` feeds both paths, no weight is
re-packed, and `y` may alias the residual in place.

**Selection.** `QwenEngineOptions::native_tensor_gemm` / `--native-tensor-gemm 0|1`, reported as
`native=tensor_core(128x64x32w8, rows>=16, cold_jit=..., launches=...)` in the `prefill gemm:` line,
with `QwenEngineInfo::native_prefill_gemm{,_block_m,_block_n,_block_k,_min_rows,_launches,_jit_ms}`.
Eligibility is CUDA + a `NativeShaderExt` instance + a supported packed format (BF16 or FP8 E4M3) +
`rows >= 16` + `N >= 256`, and dx/vk never enter the branch. Everything is gated once, at `create()`: a
projection whose geometry the tiling cannot index, a missing extension or an NVRTC/reflection failure
leaves the object invalid and the DSL GEMM runs for the whole session — never a per-shape fallback in
the middle of a prompt. The `N >= 256` threshold is measured, not assumed: at `N = 32` (the GDN
gate/alpha/beta projections) the DSL MT kernel wins at every M because a 64-wide block tile is half
empty; from `N = 1024` up the tensor-core kernel wins by 1.8x (M=64) to 16x (M=1024).

**FP8 E4M3.** The same kernel has an FP8 realization (`QN_FP8`): the packed `[N,K]` codes and their
64-element FP32 scales are read into registers one stage ahead, and each code is decoded with the
branchless magic-constant E4M3 identity (bit-identical to `qwen_fp8_e4m3_to_f32_branchless`, i.e. to
the DSL decoder for all 256 codes), multiplied by its group scale and rounded to BF16 on the way into
the shared tile. The weight buffers and their scale groups are unchanged, so BF16 and FP8 share the
candidate selection and the differential tests. That realization cannot use `cp.async` for the weights
(the tile is BF16 while the buffer holds E4M3 bytes plus a per-group scale), which is why its isolated
bandwidth is lower than the BF16 arm's even though the buffers are half the size.

**Numerics.** One BF16 rounding of the activation at the shared-tile boundary (plus, for FP8, one BF16
rounding of `fp8(code) * scale`), weights consumed exactly as stored, FP32 accumulation. Against a reference run on activations that were *already* rounded to
BF16 the kernel agrees to `max|d|/max|ref| = 1.4e-6` (FP32 accumulation order only) — that is the
correctness gate of `test_qwen_native_gemm`. Against the FP32-activation reference (`ref_qwen_matmul`
and the DSL MT/legacy kernels) the deviation is that single rounding: `1.5e-3 .. 1.6e-3` relative RMS
on random-sign data, up to `4.7e-2` relative max on the benchmark's deliberately cancellation-dominated
synthetic activation, and **greedy-id identical** prefill/decode on the real BF16 trunk at
P = 128/512/1024/4096. The documented tolerance for this path is therefore `1e-2` relative max against
the FP32 reference on well-conditioned data, with cancellation-dominated inputs allowed to reach
`5e-2`; `test_qwen_native_gemm` uses `1e-1` because its synthetic weights are ~3e-5 against O(1)
activations.

**Measured (RTX 4080 SUPER, CC 8.9, CUDA 13.0, real `data/Qwen3.5-4B-BF16.gguf`, greedy, 16 new
tokens, `--native-tensor-gemm 0` vs `1`).**

| prompt | DSL MT (arm 0) TTFT | tensor core (arm 1) TTFT | speedup | greedy ids | decode |
|---:|---:|---:|---:|---|---|
| 128 | 216.1 ms | **72.5 ms** | 2.98x | identical (8/8) | 14.10 vs 14.18 ms/token |
| 512 | 778.6 ms | **167.1 ms** | 4.66x | identical (8/8) | 14.14 vs 14.15 ms/token |
| 1024 | 1517.7 ms | **227.6 ms** | 6.67x | identical (16/16) | 14.08 vs 14.11 ms/token |
| 4096 | 6421.4 ms | **923.5 ms** | 6.95x | identical (16/16) | 14.43 vs 14.43 ms/token |

Cold JIT of the two variants (plain + residual) is **127-131 ms** — reported, not hidden: native
shaders bypass `ShaderSerializer`, so there is no disk cache for them, while the DSL kernels are
cached. Isolated GEMM shapes (`benchmark_qwen_decode cuda --prefill-gemm`, median of 30, effective
weight GB/s = modelled traffic over median wall time):

| shape | M | DSL MT | tensor core | speedup |
|---|---:|---:|---:|---:|
| N=9216 K=2560 | 64 | 0.631 ms | **0.061 ms** | 10.3x |
| N=9216 K=2560 | 1024 | 10.096 ms | **0.627 ms** | 16.1x |
| N=2560 K=9216 | 1024 | 9.147 ms | **0.592 ms** | 15.5x |
| N=8192 K=2560 | 1024 | 8.308 ms | **0.548 ms** | 15.2x |
| N=1024 K=2560 | 1024 | 1.023 ms | **0.096 ms** | 10.6x |
| N=32 K=2560 (FP8) | 1024 | 0.0725 ms | **0.0692 ms** | 1.05x but below the N>=256 rule |
| N=32 K=2560 | 1024 | **0.0516 ms** | 0.0618 ms | 0.83x (stays on DSL) |
| N=9216 K=2560 (FP8) | 1024 | 16.373 ms | **0.7257 ms** | 22.6x |
| N=2560 K=9216 (FP8) | 1024 | 15.591 ms | **0.6968 ms** | 22.4x |

The tensor-core kernel reaches ~74 TFLOPS of the 103 TFLOPS BF16/FP32-accumulate peak on this SKU at
M=1024, N=9216, K=2560 (the 64x64/4-warp tile reaches 52 TFLOPS; `QwenNativeGemmConfig` is a `create()`
parameter, and the benchmark prints the tile it ran). The remaining gap is the FP32 activation
re-read from L2 (BN=64 keeps `2/BN + 1/BM` bytes per FLOP), not the mma issue rate.

**Tests.** `test_qwen_native_gemm <backend>` (new, `tests/integration/qwen`) has 134 checks on cuda:
the device-free contract (tiling validation, fail-closed shape gate, plan, generated source), ragged
M/N/K in both epilogues against `ref_qwen_matmul` and the DSL kernels, the BF16-oracle exactness gate,
two model-sized shapes against the DSL MT and reference GEMMs, deferred commit, overwrite semantics,
`y == residual` aliasing, eight sequential dispatches from one instance and explicit `reset()`. On dx
and vk it prints one SKIP line and passes its 28 device-free checks, so the DX/VK runs of the suite
are unaffected. The pre-existing suites are unchanged: `test_qwen_ops_dsl` 1745/1745 on cuda, dx and
vk, `test_qwen_kv_cache` 78/78 on all three, `test_qwen_ops_tile` 86/86 on dx, `test_qwen_mtp` 36/36 on
cuda and the four host suites at 4298 asserts.

**Limitations.** GEMM only (the tensor-core *attention* is the next section). No persistent NVRTC
cache, so every process pays the ~130 ms JIT (BF16) or ~130-460 ms (FP8, first compile of a process). The activation is
re-read per N tile from L2 rather than staged in a wider tile, so an N=128 block tile or a 3-stage
pipeline would likely add a few percent and was not attempted. The `N < 256` and `rows < 16` shapes
are not accelerated by design.

CUDA tensor-core prefill attention (FlashAttention-2)
-----------------------------------------------------
Prefill attention no longer runs on CUDA cores on the CUDA backend either. `QwenNativeAttn`
(`src/qwen/qwen/native_attn.{h,cpp}`, device code in `native_attn_kernel.h`) is a real
FlashAttention-2 kernel - `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` for **both** QK^T and
P.V, an FP32 online softmax (running max, running sum, ONE accumulator rescale per tile), FP32
accumulators and the existing sigmoid gate epilogue - hand-written in CUDA C++, compiled once by
NVRTC through `NativeShaderExt` (`NativeShaderLanguage::CUDA_NVRTC`) and dispatched as a native
command from `QwenDslEngine`'s prefill path. It is neither a DSL kernel nor a wrapper around one: no
AST, no XIR, no `S x S` score materialization, and no per-position or per-length JIT - `tokens`,
`pos0`, `kv_len` and `capacity` stay runtime scalars, so one compiled kernel serves every prompt
length, position, ragged tail and cache capacity, exactly like the DSL tiled kernel.

**Shape, and why it is that shape.** `head_dim = 256` is the binding constraint: one `m16n8k16`
fragment needs 16 query rows and a `[16 x 256]` FP32 accumulator is 128 registers per lane, so a warp
can hold either its Q fragments or a full-width output accumulator, never both. The kernel therefore
 * splits the head dimension across a WARP PAIR: warp `w` owns q head `kv_head*4 + (w >> 1)` and
   head-dim half `(w & 1) * 128`, i.e. 64 accumulator registers per lane;
 * packs the 4 q heads of one kv head into one 256-thread block, so one K/V staging serves
   4 x 16 query rows (the GQA dedup, the same one the DSL tiled kernel gets from its 4 warps);
 * uses a 16-token query tile and a 32-key key tile. 32 keys is the largest tile that fits the
   48 KiB per-block **static** shared-memory limit of the native launch path (the native dispatch
   passes `sharedMemBytes = 0`, so it cannot opt into the 100 KiB dynamic budget): 2 x 32 x 264 x 2 B
   of BF16 K/V rows plus 4 x 16 x 33 x 4 B of score exchange = **42 240 B**. A 64-key tile would need
   66 KiB for the K/V tiles alone and is rejected by `validate_qwen_native_attn_config`;
 * exchanges the two head-dim halves of the QK^T partial through one shared `[16 x 32]` FP32 buffer
   per q head (the second half publishes its partial, the first adds its own and applies the causal
   mask, then BOTH run the identical FP32 softmax and build identical BF16 P fragments, so the result
   is deterministic), and issues the NEXT K/V tile's global loads into registers before the P.V mma so
   their latency is covered by tensor-core work.

**Numerics, and what the tolerance is.** One BF16 rounding of Q, K and V when the shared tiles are
staged, one BF16 rounding of the softmax weights `P = exp2((S - max) * log2 e)` for the P.V mma, `exp`
as the hardware EX2 unit (`<= 2^-22` relative - the same exp2 substitution the DSL tiled kernel
makes), and FP32 everywhere else (scores, running max/sum, denominator, accumulators, epilogue).
Against a reference run on the *same* BF16-rounded operands only the FP32 accumulation order remains:
`max|d|/max|ref| ~ 1.3e-3`. Against the FP32 references (`ref_qwen_attn_prefill`'s double-precision
oracle and the DSL tiled kernel) the deviation is `max|d|/max|ref| = 2.1e-3 .. 3.2e-3` with relative RMS
`1.4e-3 .. 2.4e-3` across every prompt length (1..4096) and every pos0/kv_len combination of the
suite (21 differential cases; the envelope does not grow with the prompt length). The documented envelope of this path is therefore **1e-2 relative max**, and
`test_qwen_native_attention` gates at 1e-2 / 3e-3 while printing both metrics per shape. This is NOT
bit-comparable with the FP32 kernels; the end-to-end effect is that prefill attention stops being a
TTFT term of its own (below), and the greedy ids are unchanged.

**Measured, isolated (RTX 4080 SUPER, CC 8.9, CUDA 13.0).**
`benchmark_qwen_decode cuda --prefill-attention --prefill-attn-lengths 128,512,1024,4096
--prefill-attn-iters 20` - one layer's attention over all 16 q heads, causal, full cache
(`kv_len == capacity == S`), median wall clock; GFLOP/s is the causal
`2*S*(S+1)*head_dim*q_heads` count:
| S | untiled reference | DSL tiled | native BF16 | native vs tiled | native vs reference | native GFLOP/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0493 ms | **0.0116 ms** | 0.0167 ms | 0.69x | 2.95x | 0.5 |
| 8 | 0.2222 ms | **0.0152 ms** | 0.0173 ms | 0.88x | 12.8x | 17.0 |
| 16 | 0.4067 ms | **0.0185 ms** | 0.0216 ms | 0.86x | 18.8x | 51.6 |
| 32 | 0.7682 ms | 0.0256 ms | **0.0219 ms** | 1.17x | 35.1x | 197.5 |
| 64 | 1.4357 ms | 0.0385 ms | **0.0258 ms** | 1.49x | 55.6x | 660.4 |
| 128 | 2.916 ms | 0.0844 ms | **0.0339 ms** | 2.49x | 86.1x | 1 998 |
| 512 | 11.436 ms | 0.6546 ms | **0.1166 ms** | 5.62x | 98.1x | 9 231 |
| 1024 | 28.079 ms | 2.3059 ms | **0.2932 ms** | 7.87x | 95.8x | 14 665 |
| 4096 | 706.427 ms | 33.8157 ms | **3.3420 ms** | 10.12x | 211.4x | 20 568 |
On the same buffers the arm prints `PREFILL_ATTN validate native: S=1024 vs_ref max_abs=3.529e-04
rel=3.209e-03 | vs_tiled ...`, i.e. it validates the kernel before reporting a speedup.
The first three rows are why the path has a **measured row floor of 32** (`qwen_native_attn_min_rows`):
below it the launch is one 16-token query tile per block with a single 32-key tile (15/16 of each
query tile masked off), so it is all fixed cost and the DSL tiled kernel's taller grid wins. From 32
rows on the native kernel wins, 2.5x at 128 and 10.1x at 4096.
The 20.6 TFLOP/s at S=4096 is 20 % of the 103 TFLOP/s BF16/FP32-accumulate peak of this SKU, and
Nsight Compute on the same launch (1024 blocks, 3.53 ms) corroborates it: `sm__pipe_tensor_cycles_active`
**18.4 % of peak**, `sm__warps_active` **16.7 %** (8 of 48 warps: 223 registers per thread allow one
block per SM), L1TEX 31.5 %, L2 28.0 %, DRAM 7.6 %, no local-memory spills. The kernel is therefore
**latency bound at one block per SM, not bandwidth bound**; the measured `__launch_bounds__(256, 2)`
variant fits 128 registers with 176 B of spills and was not faster at any length, and double-buffering
the K/V tiles - the obvious next step - is exactly what the 48 KiB static limit blocks.

**Measured, end to end (real `data/Qwen3.5-4B-BF16.gguf`, greedy, 8 new tokens, two runs per arm,
identical prompts via `--synthetic-prompt-len`, `--native-tensor-attention 0` vs `1`):**
| prompt | arm 0 TTFT | arm 1 TTFT | speedup | greedy ids | decode |
|---:|---:|---:|---:|---|---|
| 512 | 167.1 ms | **160.2 ms** | 1.04x | identical (8/8) | 14.09 vs 14.14 ms/token |
| 1024 | 227.6 ms | **204.4 ms** | 1.11x | identical (8/8) | 14.37 vs 14.35 ms/token |
| 4096 | 920.4 ms | **673.1 ms** | 1.37x | identical (8/8) | 14.68 vs 14.65 ms/token |
(`arm 0` = the DSL tiled kernel; at P=4096 the two runs per arm were 921.9/918.8 ms and 674.2/672.0
ms, so the 247 ms difference is well outside the run-to-run spread and matches the isolated prediction
- 8 layers x (33.8 - 3.3) ms = 244 ms.) A real 54-token chat prompt generates 32/32 identical ids on
both arms. Cold JIT of the kernel is 63-68 ms and is reported, not hidden
(`QwenEngineInfo::native_prefill_attn_jit_ms`, and the `prefill attn:` log line); native shaders bypass
`ShaderSerializer`, so no disk cache amortizes it.

**Selection.** `QwenEngineOptions::native_tensor_attention` / `--native-tensor-attention 0|1`, with
`QwenEngineInfo::native_prefill_attn` plus q tile, kv tile, warps, static shared bytes, min rows, cold
JIT and the session's dispatch count (`engine final: native-attn-launches=N`). `create()` gates it once
and **fails closed**: cuda backend, `NativeShaderExt` present, the model geometry the source was
generated for (head_dim 256, 16 q / 4 kv heads, i.e. the 4:1 GQA pack), a successful NVRTC compile and
reflection (5 buffer bindings, the configured block size). Any failure leaves the object invalid and
the DSL tiled kernel keeps prefill attention for the whole session; a `build_dispatch` failure at run
time is a hard error, never a silent fallback. The row threshold is evaluated **per prefill chunk**
(one launch per chunk and layer), so a ragged final chunk below 32 rows runs the DSL tiled kernel while
the padded chunks before it run the native one; each chunk's attention depends only on its own queries
and the KV prefix, so that changes the numerical realization of that chunk (documented) and never the
result of another chunk, and it is not a failure fallback. Decode attention, the split-KV family and the static
CUDA decode graph are untouched - a native command is never recorded into the graph - and dx/vk report
`prefill attn: ... native=off` and dispatch the DSL tiled kernel. The MTP executor forces the option
off for its target engine, exactly like `dsl_tiled_prefill_attn`: MTP verify and the accepted-prefix
replay must execute ONE numerical path, and this kernel is a different (BF16) realization of the same
math.

**Tests.** `test_qwen_native_attention <backend>` (new, `tests/integration/qwen`, registered in both
`tests/xmake.lua` and `tests/CMakeLists.txt`) has **89 checks on cuda**: the device-free contract
(tiling/config validation, the 48 KiB static-shared gate, the fail-closed input gate, the plan, and the
generated source - one `__global__`, both mma forms, the row and transposed `ldmatrix`, the EX2 unit,
the finite softmax sentinel, the sm_80 guard), then a THREE-WAY differential (native vs
`ref_qwen_attn_prefill` vs the DSL tiled kernel) at S = 1, 2, 3, 4, 31, 32, 33, 63, 64, 65, 127, 128,
129 and 1024 (ragged against both the 16-token query tile and the 32-key tile), at pos0 =
63/64/512 (non-zero history, chunk boundaries), with `kv_len == capacity` (full cache) and
`kv_len < capacity`, the sigmoid gate epilogue as a ratio between two runs that differ only in `gate`,
a deferred two-command commit, an overwrite, eight sequential dispatches from one instance and an
explicit `reset()`. On dx and vk it prints one SKIP line after its 34 device-free checks, so those runs
are unaffected. The pre-existing suites are unchanged on cuda, dx and vk:
`test_qwen_ops_dsl` 1745/1745, `test_qwen_native_gemm` 134/134, `test_qwen_kv_cache` 78/78,
`test_qwen_ops_tile` 86/86 (dx), `test_qwen_mtp` 36/36 and the four host suites at 4298 asserts.


### Phase 8b: W4A16 group-wise weights (`dsl-w4`) — opt_plan/05 Q2

The sixth weight format is 4-bit, group-wise, and **offline**: nothing is quantized inside a request.

**Storage ABI.** `Buffer<uint32_t>` with **eight signed 4-bit two's-complement codes per word**, low nibble
first (element `k` lives in nibble `k % 8` of word `k / 8`), plus `Buffer<float>` with **one FP32 scale per
contiguous group of 128 K elements per output row**. The code range is `[-8, 7]` — symmetric, **no zero
point** — so

```
dequantized weight = int4(code) * scale[row * (K/group) + k/group],  int4(code) = ((code ^ 8) - 8)
```

The sign extension is two integer ops on every backend (the Q8_0 rule, one nibble instead of one byte).
Geometry is **fail-closed**: `K % 8 == 0` (whole words) and `K % group == 0` (whole groups), with
`group ∈ {128, 64, 32}`; `K = 2560 / 4096 / 9216` and every model matrix satisfy both at `group = 128`
(`2560 = 20·128`, `4096 = 32·128`, `9216 = 72·128`, `1024 = 8·128`), and a ragged K is refused rather than
padded. Norms, `ssm_a`, `ssm_dt.bias` and `ssm_conv1d.weight` stay **dense F32**, exactly like every other
variant.

**Not plain round-to-nearest.** `pack_w4_matrix` runs a **deterministic clipping search per group**: the
max-abs scale `max_abs / 7` is multiplied by a grid of 31 ratios in `[0.85, 1.0]`, and the ratio minimizing
the group's dequantization MSE (computed in `double` against the **stored FP32 scale**, so the search scores
what the device will actually decode) wins; ties keep the smaller ratio. There is no RNG, no FP
contraction and no time dependence, so the same input bytes and config produce **byte-identical** words and
scales (checked by a host test and by a device cross-check against the independent common.h packer). On the
real checkpoint the search picked a mean ratio of **0.8664** and improved the mean squared dequantization
error by **22.5 %** over plain max-abs RTN (`6.269e3` vs `8.092e3`) — measured, not claimed.

**Offline pack + manifest.** `example_qwen --export-w4-pack <dir> --model <bf16.gguf>` walks the validated
GGUF once, packs every matrix, and writes

```
<dir>/manifest.txt              one line-oriented, order-stable text file
<dir>/tensors/<i>.w4.bin        8 int4 codes per uint32
<dir>/tensors/<i>.w4s.bin       one FP32 scale per 128 K
<dir>/tensors/<i>.fp8.bin       the SAME matrix as FP8 E4M3 quads   (plan option (c))
<dir>/tensors/<i>.fp8s.bin      its FP32 scales (one per 64 K)
<dir>/tensors/<i>.f32.bin       dense F32 payload (norms/ssm_a/dt/conv)
```

The manifest records the source checkpoint (name, size, and a rolling FNV-1a 64 digest of every tensor's
payload hash), the clip-search configuration, the segment/format order (`w4,fp8`), and per tensor: name,
role, layer, shape, group, and **every payload's byte count and 64-bit FNV-1a hash**. The manifest is
written **last**, so a pack without one is an incomplete pack. The loader (`upload_w4_pack`) verifies the
version, every size, every hash and every geometry, and fails closed on any mismatch — a truncated, stale
or foreign pack never loads partially. `--version dsl-w4` additionally compares the pack's recorded source
size with the checkpoint named by `--model` and refuses to mix two checkpoints. The runtime performs **zero
clipping search**: a request loads the pack (9.5 s for 6.4 GiB) instead of re-deriving it (the offline run
took 490.6 s).

**Prefill policy: plan option (c), not (b).** The default is **W4 decode + an FP8 prefill/LM-head second
copy** from the same pack:

* decode (`rows == 1`) reads the W4 tensors through the 4-bit GEMV family;
* prefill (`rows > 1`, embedding gather excluded) reads the **FP8** copies through a second compiled GEMM
  set (`_gemm_pf`, `_gemm_mt_pf`), and the **CUDA tensor-core prefill GEMM consumes that FP8 copy** through
  its existing kernel — no W4 tensor-core kernel exists and none is pretended;
* the **LM head stays 8-bit**: it runs the FP8 copy of `token_embd` through its own GEMV pair, never the
  4-bit table. The engine refuses a W4 session without that copy unless the explicit A/B arm is requested.
* `--w4-prefill in-kernel` selects **plan option (b)** — W4 tensors for prefill with in-kernel
  dequantization — and prints `(b) A/B arm` in its weights line. It is never the silent default, and a pack
  built with `--w4-export-fp8 0` is the only pack it can run.

The device footprint is therefore the dual copy: **6 394.61 MiB** measured for the full 426-tensor pack
(2 005.0 MiB W4 + 125.3 MiB W4 scales + 4 010.0 MiB FP8 + 250.6 MiB FP8 scales + 3.7 MiB dense), i.e.
**6.24 GiB** weights, which is what the harness' capacity table now budgets (`w4: 6.24`, with
`W4_IN_KERNEL_PREFILL_WEIGHT_GIB = 2.05` reported only for the option-(b) arm).

**Kernels (really implemented, not documented-only).** Every matrix path the model needs has a W4
realization:

| path | realization | notes |
|---|---|---|
| embedding gather | per-row nibble decode + group scale | exact decode parity with the host ABI |
| generic packed GEMV (`M = 1..4`, MTP) | one word = 8 codes per lane iteration | used for `M > 1` and as the reference for `M = 1` |
| **M = 1 decode GEMV** | **one lane reads one uint32 = 8 weights → 32 lanes cover 256 K per iteration; 8 products per word split over FOUR independent accumulators; one hoisted group scale per word (8 products per scale load)**; `$for` guards the K tail | `K = 2560` → 320 words, `K = 4096` → 512, `K = 9216` → 1152; N = 32 (the alpha/beta gates) is exercised |
| fused FFN decode | gate + up words + SwiGLU in one launch | 8 products per stream per word |
| fused GDN alpha/beta gates | same decode dot | the T = 1 gate fusion |
| reference GEMM, MT prefill GEMM | `qwen_read_weight` / word loop extended to nibbles | option (b) prefill and the differential reference |

There is **no 128-bit (uint4) W4 realization** and the planner **refuses** it
(`plan_qwen35_decode_gemv_packed_dispatch` returns `invalid_argument`): one `uint4` would be 32 codes, a
1024-element lane stride, and no model shape (`K = 2560` is not a multiple of 1024) — so the arm is
refused instead of timing a kernel that cannot serve the model. `--decode-packed` is therefore rejected for
W4 with a message, and `benchmark_qwen_decode --format w4 --impl packed` exits 2 for the same reason.

**Static CUDA graph stays format-agnostic.** The captured decode body is the same 43-shader body; only the
weight buffers and the compiled `_w4` kernels differ, and no per-token scalar moved. Measured on the real
pack: `decode graph: requested=static active=static ... storage=w4_packed`, `cuda-graph-steps=3`,
`cuda-graph-updates=0`.

**Measurements (this box: RTX 4080 SUPER, 16 GiB, CUDA 13.0; T16 tier).**

End-to-end, one 8-token prompt, greedy, static graph: **6.72 ms/token (dsl-w4)** vs 16.11 ms/token
(dsl-bf16) on the same prompt — 2.4× — with 9.4 s pack load and 4.5 s extra shader compile (43 shaders).

Isolated decode GEMV (`benchmark_qwen_decode cuda --format w4 --impl decode --iters 10`, M = 1, one warp per
output row, byte model = int4 codes + 4 B per 128 K + one activation read + one output write):

| shape | N | K | median ms | effective GB/s |
|---|---:|---:|---:|---:|
| `gdn_gate` | 32 | 2560 | 0.00585 | 9.2 |
| `attn_kv` | 1024 | 2560 | 0.00723 | 194.5 |
| `gdn_out_attn_out` | 2560 | 4096 | 0.01742 | 321.2 |
| `gdn_z` | 4096 | 2560 | 0.01951 | 286.8 |
| `attn_q_gdn_qkv` | 8192 | 2560 | 0.03058 | 365.8 |
| `ffn_gate_up` | 9216 | 2560 | 0.03373 | 373.0 |
| `ffn_down` | 2560 | 9216 | 0.03518 | 357.6 |
| `lm_head` (**FP8 copy**) | 248320 | 2560 | 0.8062 | 420.1 |
| fused FFN decode | 9216 | 2560 | 0.04013 | 627.6 |
| separate gate/up + SwiGLU (same data) | 9216 | 2560 | 0.06938 | 363.1 |

**Honest reading of those numbers (this is the problem with 4-bit on this box).** W4 halves the bytes, but
the DSL decode GEMV does **not** reach the FP8 arm's effective bandwidth: FP8 measured 538–643 GB/s on the
wide shapes (phase 8a), W4 measures 357–374 GB/s there. Per byte the 4-bit kernel issues twice the ALU
(sign-extend + scale per 8 products over half as many bytes) and its per-lane loop is shorter, so it becomes
**instruction/latency-bound, not DRAM-bound** on an Ada part — exactly the risk opt_plan/05 §8 lists. The
consequence is a **~1.2× wall-time gain on the wide shapes** (0.0337 vs 0.0398 ms for `ffn_gate_up`), not
2×; the end-to-end 2.4× over BF16 comes mostly from BF16 being 2× the bytes *and* the graph, not from W4
being at its roof. Reporting W4 as "2× decode" would be wrong on this hardware. The next lever is more
words per lane per iteration (16 products) to raise memory-level parallelism, or a 128-bit load variant
with a group-aligned layout; neither is implemented here because neither has been measured.

**Tests.** `test_qwen_ops_dsl <backend>` is **1901 checks on cuda, dx and vk** (up from 1829), 70 of them
W4, and gained a W4 block (registered after the Q8_0 block): host ABI
facts (geometry fail-closed, clipping ≤ RTN MSE, dsl/common packer byte equality, exact host decode
parity), generic GEMV, reference GEMM, MT prefill GEMM at `M = 16/20`, the decode GEMV at the model K
values with `N = 64/40/32`, the fused FFN decode (including a ragged `N = 32` tail), the fused GDN
alpha/beta gates and the embedding gather — each against `ref_qwen_matmul(w4, ...)`, which decodes the same
words and scales. `test_qwen_weights` gained five tests (host suite now 7534 asserts in 12 tests): W4
geometry/classification, pack/decode round trip with reproducibility, NaN/overflow refusal, dsl↔common
packer and decoder parity, a manifest write/read/tamper round trip, and — with a backend argument — a
**device pack round trip**: a tiny two-tensor pack is written, loaded through `upload_w4_pack`, compared
word-for-word and scale-for-scale with the host ABI (plus the `<name>.fp8` partner and the dense tensor),
then a single flipped payload byte and a truncated scale file must both be refused (hash/size fail-closed).

**Limitations (do not over-claim).**
* The plan's **quality gate** (≥ 32 fixed prompts, ≥ 128 greedy tokens, ≥ 90 % mean prefix agreement, first
  32 ids identical on ≥ 28/32, KL vs BF16, 1 K-generation repeat/NaN check) is implemented as
  `scripts/qwen_quality_gate.py` with the CLI's `--dump-logits` binary dump (greedy ids + the prefill and
  final last-position FP32 logits). Its **measured result on this T16 box is recorded below** and it says
  nothing about a T32 machine.
* 4-bit decode is ALU/latency-bound on this GPU (see above): the byte saving is real, the wall-time saving
  is ~1.2× per wide projection.
* The MT prefill GEMM is compiled for the FP8 copy only in a dual-copy session (the decode-format MT set is
  deliberately not compiled, so `_prefill_gemm_mt` is reported as true through the prefixed flag).
* No W4 support in the Tile engine, the MTP draft (`blk.32.*` comes from the GGUF, and `--mtp` is refused
  with a W4 pack because the draft tensors are not in the pack), or the nextn executor.
* Only `group = 128` has been run end-to-end; 64/32 are implemented, validated by geometry/decode tests and
  selectable from the packer (`--w4-group`), but not measured end to end.
* The pack is ~6.3 GB and lives under `.kimix_cache/qwen_benchmark/phase8b/w4pack`; it is **not** in git
  and must be regenerated with the packer.

**Quality gate result (T16, RTX 4080 SUPER, 32 prompts × 128 greedy tokens).**

**Quality gate result (T16, RTX 4080 SUPER, 32 prompts × 128 greedy tokens).** Implemented and **RUN**,
not assumed: `scripts/qwen_quality_gate.py run --prompts 32 --max-new-tokens 128 --generation-check 1024
--tier t16 --device-tag "RTX 4080 SUPER"`; artifacts under
`.kimix_cache/qwen_benchmark/phase8b/quality/` (`summary.json`, 32 paired dumps, every raw log).

| metric | measured | plan threshold | verdict |
|---|---|---|---|
| mean greedy prefix agreement | **0.3179** | ≥ 0.90 | **FAIL** |
| prompts with identical first 32 ids | **10 / 32** | ≥ 28/32 | **FAIL** |
| first token identical | 32 / 32 | – | – |
| KL (prefill last position), mean / max | **0.2867 / 1.7945** | reported, no threshold | – |
| 1 K-generation repeat + NaN check | **pass** (1024 ids identical across two runs, no NaN/inf, longest single-id run 1) | pass | **PASS** |

**dsl-w4 at group 128 with the clipping search alone does NOT pass the plan's quality gate on this box, so
it is not reported as a shippable W4 format.** Per opt_plan/05 §3 the fallback applies: the 8-bit floor
(dsl-fp8 / dsl-q8) stays the shipping decode format, and 4-bit stays an explicit, experimental
`--version dsl-w4` that requires a pack and says so in `--list-versions`.

Two controls make the number interpretable rather than a bare failure:

* **The prompt family is brutal for greedy agreement.** The 32 prompts are the harness' deterministic
  *synthetic* token ids (never free text), on which the model degenerates into a repeating cycle; a shift by
  one id then breaks the whole prefix. Running the **same protocol with `dsl-fp8` as the candidate**
  (8 prompts of 128/256 tokens, `.kimix_cache/qwen_benchmark/phase8b/quality_fp8_control/`) measures mean
  prefix **0.5156**, first-32 identical **4/8**, KL mean **0.00618** — i.e. even the 8-bit format the plan
  treats as shippable misses "≥ 90 % prefix, ≥ 28 of 32" on these prompts. The gate as literally specified
  is therefore not a usable discriminator on synthetic id prompts; it needs natural-language prompts (or a
  relaxed prefix rule) before it can decide a format. That is a finding about the gate, not an excuse.
* **W4 is genuinely ~46× further from BF16 than FP8 in distribution distance** on the same protocol (KL
  mean 0.2867 vs 0.00618, max 1.79 vs 0.0089). That is real 4-bit quality loss, **not** a kernel defect:
  every W4 device path matches the host oracle (1901/1901 device checks on cuda, dx and vk; 70 of them W4),
  and free-text prompts stay coherent and identical to BF16 (`The capital of France is` → ` Paris.` for both
  dsl-bf16 and dsl-w4). The next lever the plan names — activation-aware (importance) scaling in the clip search — is **implemented in the phase 8b review pass as an explicit EXPERIMENT** (`--w4-calibrate` + `--w4-importance`, see the review section below); it has **no quality-gate result yet**, so it does not change this verdict: without a passing gate, 4-bit g128 is not shippable on this model.

The gate run used the device-greedy in-flight decode, whose host-side `logits` stay at the prefill step, so
that run wrote the same vector twice (hence `KL final` == `KL prefill` in `summary.json`). The dump now
writes **only** the prefill distribution in that case and the script reports `kl_final: null` plus an `n/a`
line, so a future run cannot present a duplicated vector as a second measurement.

### Phase 8b review: source identity, honest aggregates, and the importance-aware W4 experiment

The review of the phase 8b W4 work found three concrete defects and one missing experiment. They are fixed in
the tree, each with a regression test, and the experiment now has a measured (failed) T16 quality gate.
Nothing below upgrades W4: the measured clipping-only gate FAILED, W4 stays **EXPERIMENTAL and rejected as a
shipping format**, and the 8-bit floor (`dsl-fp8` / `dsl-q8`) stays the shipping decode format.

**1. The pack's source identity is now a rolling payload digest, not a file size.** The manifest's
`source hash=` is FNV-1a 64 over the *per-tensor raw payload hashes* of the checkpoint inventory, folded in
inventory order (`w4_roll_source_hash` in `src/qwen/qwen/w4_pack.h`, the single place the rule lives — the
packer and every verifier call it). `w4_hash_source_tensor` / `w4_source_digest` /
`compute_w4_source_hash` recompute it from the current `GgufReader` + model in bounded chunks, and
`verify_w4_pack_source` refuses a pack whose recorded size **or** digest differs. `--version dsl-w4` now calls
that verifier before uploading anything, so a **same-size / different-payload** checkpoint is rejected with
“SAME file size (rolling source payload digest mismatch)”. The regression test builds two checkpoints of
identical byte size with different weights, flips one payload byte in a third copy, and asserts all three
cases (plus a tampered recorded hash) fail closed while the true checkpoint passes.

**2. `W4PackReport` aggregates are model-wide and normalized once.** `mse` / `rtn_mse` are element-weighted
means (`sum of squared error / w4_elements`), `mean_ratio` is group-weighted across the whole pack,
`groups` / `w4_elements` / `clipped_groups` are totals, and the packer resets a caller-supplied report first
(so `--export-w4-pack` can no longer print a total squared-error sum as an “MSE”, or the last tensor's
clipping ratio as the model's). `mse_improvement()` / `weighted_improvement()` derive the gain from those
aggregates. The regression test packs a multi-tensor fixture and compares every field against an independent
per-tensor recomputation, then re-packs into a deliberately dirtied report.

**3. `upload_w4_pack`'s report is internally consistent.** `tensor_count` counts **device tensors**
(`f32 + bf16 + fp8 + q8_0 + w4`) and equals their sum; the number of manifest records it walked is the new
`manifest_records`. For the phase 8b pack that is `tensor_count=675` (249 W4 + 249 FP8 + 177 F32) with
`manifest_records=426` — the old `426` was the record count printed as a tensor count. The device test asserts
both numbers and their consistency.

**4. The importance-aware option (opt_plan/05 §3 item 2) — implemented and measured; quality gate FAILED.** Calibration collects the
measured **activation second moment per projection input channel** and the offline packer weights its group
error with it:

```
m[j]      = (1 / tokens) * sum over activation tokens x of x[j]^2        (diag(H), independent channels)
E_g(s)    = sum_{j in g} m[j] * (v[j] - q_j(s) * s)^2 / sum_{j in g} m[j]   (per group g, per row)
objective = min over the same 31-step ratio grid of E_g(s)
```

`E_g` is the squared residual weighted by the channel's measured activation energy, normalized by the group's
total weight, i.e. the residual scaled by `sqrt(m[j])` before squaring — hence “sqrt(importance)-weighted
group MSE”. Only the **scale** choice changes (codes stay nearest-int4 for the chosen scale), a constant weight
vector reproduces the clipping-only search byte for byte, `mse`/`rtn_mse` keep their plain per-element meaning
(comparable across packs) and `weighted_mse`/`rtn_weighted_mse` report the objective that was actually
minimized. This is **not AWQ**: no per-channel scaling is folded into the weights, there is no GPTQ-style
rounding compensation, and the measured gate below is the only quality evidence.

* `--w4-calibrate <dir>` is a **calibration-only** mode (no request, no output token): it runs the BF16 model
  with `QwenEngineOptions::calibration_only` (the reference, non-fused M=1 route, because the fused FFN/GDN
  kernels compute the `ffn_down` input in-kernel) over a deterministic set
  `qwen_synthetic_prompt_ids(prompt_len + i)`, `i < prompts`, plus `--w4-calib-decode` greedy tokens each,
  accumulates `x[j]^2` per channel per projection **on the device** (one thread per channel, no atomics, row
  order, so a rerun is byte-identical) and exports
  `<dir>/importance.txt` + `<dir>/importance.f32.bin`. The manifest pins the checkpoint name/size/**digest**,
  the prompt set, the token count and, per tensor, `name`, `K`, `tokens`, `mean|min|max` and the payload hash.
* **Coverage: ALL quantized matrices** (the 249 W4 tensors: 248 block projections + the tied LM head), not a
  sensitive subset — the accumulator set is exactly the engine's resolved projection tensors, and a tensor that
  never saw an activation row makes the writer fail closed instead of inventing statistics.
* `--export-w4-pack ... --w4-importance <dir>` binds the statistics **by tensor name + K**, so a missing
  tensor (`not_found`), a different K (`unvalidated`), a different checkpoint size or digest (`unvalidated`),
  or non-finite/negative/all-zero data is refused before a payload is written. The pack manifest records the
  calibration (`importance name=… payload=… source=… tokens=…`), and `upload_w4_pack` reports
  `w4_pack_importance_aware` with those hashes.
* `--w4-pack-only <substr>` is a **tooling** knob (a partial pack for smokes; it is loudly reported and is not
  a runnable dsl-w4 checkpoint). It exists so an importance-aware pack can be exercised in seconds instead of
  repeating the 8-minute full-model pass.

**Measured smoke (real checkpoint, subset `blk.0`, cuda, T16).** Logs:
`.kimix_cache/qwen_benchmark/phase8b/review/`.
Calibration (`--w4-calib-prompts 2 --w4-calib-prompt-len 8 --w4-calib-decode 2`): **249 matrices, 899 584
channels, 21 activation tokens** (projections 21 rows, LM head 6 last-position rows), mean second moment
0.6902, e.g. `blk.0.ffn_gate.weight` mean 0.218 / max 35.65, `blk.0.ffn_down.weight` mean 6.97e-4 /
max 0.135 — i.e. strongly skewed, which is the point.

| pack of the same 8 W4 matrices (`blk.0`) | plain MSE | vs RTN | weighted objective | vs RTN (weighted) | mean ratio | clipped groups |
|---|---|---|---|---|---|---|
| clipping only (measured, shipped phase 8b arm) | 1.315268e-06 | −22.40 % | not computed | – | 0.8701 | 880 085 / 881 920 |
| importance-aware (new) | 1.340694e-06 | −20.90 % | **1.279527e-06** | −23.46 % | 0.8758 | 867 867 / 881 920 |

The trade is the expected one and is reported as a trade: the importance-aware pack is **1.9 % worse in plain
per-element MSE** (it no longer optimizes that objective) and **better on its own weighted objective**, with
fewer aggressively clipped groups. Payload bytes differ from the clipping-only pack for every matrix. **A full
quality gate has now been run on an importance-aware pack (8 calibration prompts × 128..135 tokens + 32 decode rows, all 249 matrices): mean prefix 0.3254, first-32 identical 10/32, mean/max KL 0.3604/2.2681 — FAILED, with slightly worse distribution distance than clipping-only W4. The two 1024-token repeat runs were deterministic and finite but the top two ids occupied 84.3 % of positions, so the strengthened repeat check also FAILED. Raw artifacts: .kimix_cache/qwen_benchmark/phase8b/importance-full/.

The full calibration → 6.39-GiB dual-copy pack → 32-prompt/1 K protocol is complete and FAILED on T16. Remaining work is not another claim of support: a natural-language prompt set would make the prefix rule more discriminating (the FP8 synthetic control also fails), and a T32 gate would be required for any cross-tier claim. Neither can retroactively make these two measured W4 packs shippable; dsl-w4 stays EXPERIMENTAL and the shipping decode formats remain dsl-fp8 / dsl-q8.


**Same-session 4-format comparison** (`benchmark_qwen_decode cuda --format all --impl decode --iters 10`,
M = 1, effective GB/s from the byte model of each format; median of 10):

| shape | N | K | bf16 | fp8 | q8_0 | **w4** |
|---|---:|---:|---:|---:|---:|---:|
| `attn_kv` | 1024 | 2560 | 509 | 373 | 397 | 190 |
| `gdn_out_attn_out` | 2560 | 4096 | 629 | 579 | 594 | 323 |
| `gdn_z` | 4096 | 2560 | 607 | 525 | 535 | 287 |
| `attn_q_gdn_qkv` | 8192 | 2560 | 649 | 608 | 616 | 367 |
| `ffn_gate_up` | 9216 | 2560 | 659 | 631 | 642 | 373 |
| `ffn_down` | 2560 | 9216 | 653 | 629 | 630 | 209 (358 in the isolated run) |

Wall time per projection: `ffn_gate_up` 0.0337 ms (w4) vs 0.0398 ms (fp8) vs 0.0717 ms (bf16) — **1.18×**
over FP8 on the bytes-halved format, because 4-bit is instruction/latency-bound, not DRAM-bound, on this
Ada part (opt_plan/05 §8 risk). `ffn_down` (N = 2560, K = 9216) is the worst case: only 20 blocks of 128
threads for a 12.5 MiB read, so it runs at 209–358 GB/s and varies run to run; a wider per-lane tile is the
documented next step and is not implemented.

**The FP8 copy is really what prefill reads.** A 512-token dsl-w4 run reports
`prefill gemm: mt(mt=16)` and `engine final: ... native-prefill-launches=800` (32 layers × 25
projections), i.e. the CUDA tensor-core prefill GEMM dispatched against the dual copy's FP8 payloads,
with `TTFT(512) = 351.9 ms` and `decode 7.52 ms/token`.

**Measured option-(b) A/B (why the default stays (c)).** Running the same 1024-token prompt twice from the
same pack and binary, `--w4-prefill in-kernel` (option (b): W4 prefill with in-kernel dequantization) against
the default (option (c): FP8 prefill copy):

| arm | TTFT (prefill 1024) | decode | greedy ids (8) | weights on device |
|---|---:|---:|---|---:|
| (c) W4 decode + FP8 prefill (default) | 511.5 ms | 7.78 ms/token | `220 98429 220 98429 …` | 6 394.61 MiB |
| (b) W4 decode + W4 in-kernel prefill | 523.3 ms | 7.61 ms/token | identical | same pack, only the FP8 prefill tensors are not read |

Both arms produce the **same greedy ids**, so the 4-bit prefill GEMM is numerically consistent with the FP8
one at this length; the TTFT difference (2.3 %, one run each) is inside this box's noise. The honest reading
is that option (c)'s 4.2 GiB of extra resident weights bought **no measurable TTFT** at P = 1024 here — the
prefill projections are not
bandwidth-bound on T16 at that length — while single-copy option (b) would make 128 K context fit with a
2.05 GiB weight footprint. The plan's default (c) is kept (the 8-bit prefill is the accuracy-preserving
choice, and the LM head stays 8-bit in both arms), but `--w4-prefill in-kernel` is a real, measured arm,
not a label.

### Phase 9a: offline merged input projections (`--projection-pack`) — opt_plan/03 §3 + D5

**What it is.** The two decode-time input-projection groups are concatenated **offline**, along N,
into one row-per-output matrix per decoder layer:

| group | source tensors (in this order) | merged N |
|---|---|---:|
| `gdn_qkv_z_ab` (24 GDN layers) | `attn_qkv.weight` (8192) \| `attn_gate.weight` (4096) \| `ssm_alpha.weight` (32) \| `ssm_beta.weight` (32) | 12352 |
| `attn_qkv` (8 full-attention layers) | `attn_q.weight` (8192) \| `attn_k.weight` (1024) \| `attn_v.weight` (1024) | 10240 |

The merge is a pure **row concatenation** of the `[K, N]` GGUF matrices, in that order, so a merged
row *is* the source row: no re-quantization, no element reordering, and a per-row FP8/Q8_0 scale stays
next to its own row and its own segment boundary. `src/qwen/qwen/projection_pack.{h,cpp}` is the pack
(writer + reader + verifier) and `qwen35_merged_projection_inventory()` in `src/qwen/qwen/model.{h,cpp}`
is the single source of truth for the segment order that the packer, the manifest and the engine all
validate against.

**Why a pack.** At 4 B parameters the repack is ~2 GiB of I/O and must never happen inside a request
(plan §3). The pack also **replaces** its source tensors at upload: `QwenWeightUploadOptions::projection_pack`
makes the uploader *skip* every `(role, layer)` a merged segment covers and upload the merged payload
instead, then create one **first-class per-segment alias** per source tensor. The alias's
`packed`/`scales` are *sub-views* of the merged buffers, so its role, name, dims and storage stay the
source tensor's (`find(role, layer)`, the prefill GEMMs, the FFN path and the calibration route all keep
working) while **no byte is duplicated** — `device_bytes` after a packed upload is bit-for-bit the same
number the unmerged upload reports (8023.67 MiB bf16 / 4264.29 MiB fp8 on this box).

**Pack format** (self-describing, manifest written last):

```
<dir>/manifest.txt                 line-oriented, order-stable
<dir>/matrices/<index>.bin         merged packed words [N][K/pack] uint32 LE (the device ABI itself)
<dir>/matrices/<index>.scales.bin  merged FP32 scales [N][scales_per_row] (fp8/q8_0 only)
```

The manifest records the version, the format (`bf16` / `fp8_e4m3` / `q8_0`), the scale group, the code
units per word, the scales per row, the segment order, the **source checkpoint identity** (file name,
byte size and a rolling FNV-1a 64 digest over the raw payloads of exactly the merged source tensors in
inventory order — the same rolling rule as the W4 pack) and, per matrix and per segment, the SOURCE
name/role/layer, the row count and row offset, the payload byte sizes and FNV-1a 64 hashes.

**Fail-closed gates** (all before the first byte is uploaded):

* the manifest must describe the canonical inventory of *this* model (kind, segment roles/names/rows/
  offsets, `K = hidden`, `N = Σ rows`, format, group, per-matrix and total payload sizes);
* the checkpoint byte size **and** the recomputed rolling source digest must match (a same-size /
  different-weights file is refused);
* every payload's byte size and hash are re-verified *while streaming* it to the device;
* the pack's format must equal the format the variant uploads (a bf16 session refuses an fp8 pack and
  vice versa: a merged matrix is never re-quantized);
* `dsl-w4` refuses a pack outright (there is no 4-bit merged realization).

**Decode kernels** (`make_qwen35_merged_{gdn,attn}_gemv_kernel`, `src/qwen/qwen/dsl_ops.{h,cpp}`):
one warp per merged output row, exactly like the realizations they replace, and the *same dot body per
row group* — projection rows use the decode projection realization of the format (BF16: the generic
packed pair loop; FP8/Q8_0: the four-stream specialization with the shared E4M3 LUT, ragged-K fallback
included) and the 64 GDN gate rows use the fused gate kernel's own dot and epilogue
(`g = ssm_a·softplus(a + dt_bias)` → `_gdn_g`, `beta = sigmoid(b)` → `_gdn_beta`). Consequences:

* the per-output FP32 **accumulation order is unchanged**, so the decode logits and greedy ids are
  **bit-identical** to the unmerged session (differential-tested, not tolerated: see below);
* each warp stores its scalar **straight into the row's real destination** (`_proj_a`, `_proj_b`,
  `_gdn_beta`, `_gdn_g` / `_attn_proj_q`, `_attn_proj_k`, `_attn_proj_v`) — no 12352-wide intermediate,
  no extra pass;
* the GDN launch **replaces** the two projection GEMVs *and* the fused alpha/beta gate launch, so the
  saving is 2 launches per GDN layer and 2 per attention layer = **64 launches per decode token** at
  this geometry (measured: the decode phase of `--benchmark-csv` drops from 732 to 604 commands per
  two decode steps). The plan's original "7 → 2" figure predates the alpha/beta fusion that is already
  in this tree, so the honest number is 3 → 1 per group.
* prefill (rows > 1) is **untouched**: it reads the same rows through the per-segment aliases with the
  ordinary (native tensor-core / DSL MT / reference) GEMMs. The FFN decode kernel and the LM head are
  not merged. `rows == 1` only.

**Refusals (reported in `QwenEngineInfo`, never silent):** W4 weights, the 128-bit packed decode family
(it re-partitions K, i.e. it would break the "same accumulation order" contract), MTP / reference decode
(`dsl_decode_specialized == false` must keep one reduction order) and `calibration_only`. Without a pack
`--merged-decode` changes nothing, so the **unmerged path remains the default and the A/B**.

**Measured (RTX 4080 SUPER, CUDA 13.0, 8 tokens after a 5-token prompt).**

Isolated per-group launch time (`benchmark_qwen_decode --impl decode`, cache-resistant replica batches;
the merged rows include the gate epilogue, the unmerged rows are the launches it replaces):

| arm | bf16 | fp8 |
|---|---:|---:|
| `attn_q_gdn_qkv` (N=8192) | 0.0640 ms @ 656 GB/s | 0.0367 ms @ 608 GB/s |
| `gdn_z` (N=4096) | 0.0346 ms @ 608 GB/s | 0.0213 ms @ 523 GB/s |
| `gdn_gate` (N=32) ×2 | 0.00645 ms @ 27 GB/s | 0.00788 ms @ 12 GB/s |
| **`gdn_qkv_z_ab_merged` (N=12352)** | **0.0945 ms @ 670 GB/s** | **0.0530 ms @ 634 GB/s** |
| `attn_kv` (N=1024) ×2 + `attn_q` | 0.0848 ms | 0.0523 ms |
| **`attn_qkv_merged` (N=10240)** | **0.0792 ms @ 663 GB/s** | **0.0437 ms @ 638 GB/s** |

The merged launch is the *fastest* of the family per byte (no per-launch ramp four times in a row):
−15 % (bf16) / −28 % (fp8) for the GDN group and −7 % / −16 % for the attention group versus the
segment launches alone, before counting the gate launch the GDN merge also absorbs.

End-to-end TPOT, two order-reversed pairs each (identical greedy ids in every arm):
`11751 13 198 32 13 2912 198 33`):

| variant | unmerged | merged (`--projection-pack`) |
|---|---:|---:|
| dsl-bf16 | 13.43 / 13.45 / 13.46 / 13.53 ms/token | 13.46 / 13.46 / 13.51 / 13.53 ms/token |
| dsl-fp8 | 8.23 (cold) / 7.87 / 7.82 ms/token | 7.87 / 7.88 ms/token |
| dsl-q8 (real Q8_0 pack) | 8.19 / 8.17 ms/token | 8.22 / 8.22 ms/token |

The run-to-run spread on this box is ±0.07 ms/token, i.e. the two arms are indistinguishable; a packed
upload is also slower (bf16 4.85 s vs 2.83 s, q8_0 2.37 s vs 1.15 s) because the pack is streamed and
hash-verified from a second file instead of read once from the checkpoint — except `dsl-fp8`, where the
pack removes the 120-matrix host conversion (119.7 s → 93.4 s).

**Honest reading:** the launch reduction is real and measured, and each group kernel is 7–28 % faster in
isolation, but this box is DRAM-bound at 8.0 GB (bf16) / 4.3 GB (fp8) of weights per token, so removing
64 launches per token (≈0.1 ms of a 13.5 ms token at graph-replay launch cost) is not measurable. The
default therefore stays **unmerged** (a pack must be requested explicitly); the merged route is the A/B
and is what the plan expects to pay off on the higher-bandwidth tiers.

Side effects worth knowing: the pack removes the **host FP8 conversion** of the merged matrices — a
`dsl-fp8` run's weight upload drops from 119.7 s to 93.4 s here (the 120 merged matrices stream from the
pack; the other 129 are still converted) — and the pack's source verification costs 1.75 s at load.

**Tests.** `tests/unit/qwen/test_qwen_weights.cpp` gains a host contract test
(`qwen_projection_pack_manifest`: writer/reader round trip of every recorded field, the payload size+hash
gate with a flipped byte and a truncated file, and eleven fail-closed manifest mutations) plus a
device-arm block (real packer on the synthetic full-inventory checkpoint, payload + source-identity
verification, the sort/skip/alias counts, `device_bytes` equality with the unmerged upload, a sampled
**byte-exact** row comparison of every segment against its source payload, and both format-mismatch
refusals). `tests/integration/qwen/test_qwen_ops_dsl.cpp` gains `test_merged_decode_gemv`, which runs the
merged kernel against the *unmerged* per-segment GEMVs and the fused gate kernel on the same packed bytes
for bf16, fp8_e4m3 and q8_0 at **ragged** widths (GDN N=12346, attention N=10236, so blocks mix segments)
and requires **exact** equality of every destination, plus the three fail-closed plans. Suite state on
this box: `test_qwen_ops_dsl` 2277/2277 on **cuda** and **dx** (`vk` run recorded under
`.kimix_cache/qwen_benchmark/phase9a/logs/`), `test_qwen_weights` 18 host tests / 7837 asserts plus the
cuda device arm, `test_qwen_engine` 544, `test_qwen_gguf` 32661, `test_qwen_speculative` 367 asserts, and
the harness's own `python scripts/test_benchmark_qwen.py` 89 tests — all passing.

**Limitations (explicit).** W4 is not mergeable; the 128-bit packed decode family and MTP/reference
decode do not use the merged route; prefill, the fused FFN decode and the LM head are unchanged
(so no "megakernel"); the pack is one format (build one per variant) and costs as much disk as the
matrices it replaces; the merged widths are compile-time-validated against the frozen geometry rather
than generalized to other models; and no tier above this T16-class box was measured.

### Phase 9b: conditional WHILE decode chunk, pinned downloads, PDL probe (opt_plan/09 N4.2–N4.4)

**Conditional WHILE chunk (`--cuda-graph-while <n>` / `--graph-flight`).** Phase 9a made the greedy
step argument-invariant (position, KV length, history slot and the conv-ring position live in
`_dev_state`), which is what makes a *conditional* loop expressible. With the flag, one
`cuGraphLaunch` executes **1..n** tokens:

* the parent graph owns one node (a 1-thread zero-fill of the unused dummy buffer — CUDA refuses a
  node-less graph) followed by a `CU_GRAPH_COND_TYPE_WHILE` container whose condition is created with
  `CU_GRAPH_COND_ASSIGN_DEFAULT` and a launch default of **1** (do-while: a chunk always makes
  progress, and the default is re-applied on every launch, so a replay is reproducible);
* the body is the *same* argument-invariant step the per-token graph captures, with the argmax finish
  replaced by `make_qwen_greedy_argmax_finish_while_kernel`, which writes the loop condition from the
  device: `continue = remaining != 0 && !eos && next_position < max_context`, where `remaining` is the
  token budget the engine uploads before each launch and `eos` flags "the selection ended generation"
  (EOS id, `>= vocab`, or the NaN sentinel). The conditional handle is a session constant baked into the
  body (the body is only ever populated through `build_body_graph`, after the handle exists), so the
  body stays argument-invariant;
* `QwenEngine::decode_device_greedy_chunk` reports `executed` from a **device readback** of
  `remaining`/`position`/`eos` (pinned staging when selected), so an early EOS or context-limit exit
  cannot look like unexecuted tokens ran: `QwenChunkResult { requested, executed, eos, context_limit,
  fallback }`. The host `_position` is taken from the device, never from the request.

Gating is deliberately redundant (`opt_plan/03` §2.5): the `CudaGraphExt::capabilities()`
`conditional_nodes` flag is a **toolkit** gate (CUDA ≥ 12.3) that says nothing about the device, so
`qwen_probe_cuda_graph_while()` builds two throw-away WHILE graphs and *measures*: a do-while body must
run exactly 3 times, the replayed launch must run 3 times again, and a default-0 launch must run the body
0 times. Only then is the chunked path selected. A refusal, a missing EOS id, a failed capture or a
non-CUDA device all fall back to the per-token static graph (`result.fallback` + `while-fallbacks`), so
the ids are identical either way and only the launch count differs. On this box the probe passes
(CC 8.9, CUDA 13.0), so the WHILE is **real, not simulated** — and it is also gated per launch by
`--cuda-graph` and by an EOS id, and `--benchmark`/MTP keep per-token lists.

**Pinned download staging (`--pinned-transfers`).** `qwen/pinned_ring.{h,cpp}` owns a page-locked token
ring (1024 tokens by default) and a page-locked `[vocab]` logits buffer, allocated once through
`PinnedMemoryExt::allocate_pinned_memory` (`write_combined = false`: the host *reads* these). A download
is a stream-ordered device→device copy **into** the pinned buffer, a `synchronize()`, then a host memcpy
into the caller's span; an arbitrary caller range larger than the ring is served in ring-sized slices, so
`download_device_greedy(stream, first_slot, span)` keeps its contract for every range. Honest copy
accounting: the pre-existing pageable path is a DtoH DMA into the stream's 32 MiB download pool plus a
stream-ordered host→host memcpy into the caller's span (`cuda_command_encoder.cpp`,
`BufferDownloadCommand`); the pinned path is one DMA into host-mapped memory plus one host memcpy. **Both
move the bytes twice**, and `QwenEngineInfo::pinned_host_copies` reports the second copy instead of
pretending it disappeared. Selection is fail-closed and reported (`pinned_transfers`, `pinned_available`,
`pinned_ring_tokens`, `pinned_reason`); `--benchmark` keeps the pageable path because that mode
instruments per-`CommandList` commands and a pinned download is not one.

**PDL (`--pdl`).** `enable_programmatic_stream_serialization()` is a *device-creation* property (the
wait-at-entry prologue is compiled into every kernel and changes the shader cache key), so the CLI
installs a `CUDADeviceConfigExt` into `DeviceConfig::extension` before `create_device()`. The engine then
reports the `CUDAPdlStreamExt` probe (`supported`, `auto_prologue`, `stream_enabled`, the
relaxed/strict/driver-fallback counters) and the reason. On this CC 8.9 box `pdl_supported()` is false,
every launch stays on the strict path (`pdl-strict` counts them) and the log says
"not available on this device"; **no speedup is claimed**. No engine kernel emits
`grid_dependency_synchronize()`: the intended placement ("first weight-word chunk issued, then a
dependency wait before the activation read") does not exist in these GEMV kernels, whose activation read
*precedes* every weight stream, so there is no eligible realization to add and the probe is all that
ships (`opt_plan/03` §2.5's "measure, then keep or drop").

**cudadevrt (`N4.4`).** `CUDADeviceConfigExt::use_cudadevrt()` stays at the backend default (**true**)
and no opt-out is offered: `set_graph_conditional()`, i.e. the entire conditional-WHILE mechanism, is a
`cudadevrt` entry point (`cuda_device_graph.h`), and the same library backs device-side graph launch. A
flag that skips the ~200–400 ms `cuLink` probe on `create_device` would therefore silently break the
WHILE and device-launch paths, which is exactly what `N4.4` forbids. This is documented behaviour, not a
new switch.

**Measured on this box (RTX 4080 SUPER, CC 8.9, CUDA 13.0, `data/Qwen3.5-4B-BF16.gguf`).** 15 decode
steps, 3 order-reversed repetitions, parsing only what the binary printed:

| arm | ms/token (median, min-max) | total decode | launches | ids |
|---|---:|---:|---:|---|
| per-token command lists (`--no-cuda-graph`) | 13.89 (13.85-13.93) | 208.4 ms | 15 | identical |
| **per-token static graph (default)** | **13.33 (13.32-13.34)** | **199.9 ms** | 15, updates 0 | identical |
| `--cuda-graph-while 2` | 13.81 (13.80-13.82) | 207.2 ms | **8** | identical |
| `--cuda-graph-while 4` | 13.82 (13.78-13.84) | 207.3 ms | **4** | identical |
| `--cuda-graph-while 8` | 13.77 (13.76-13.82) | 206.5 ms | **2** | identical |
| `--pinned-transfers 1` / `0` | 13.34 / 13.33 | 200.1 / 199.9 ms | 15 | identical |
| `--pdl 1` / `0` | 13.34 / 13.32 | 200.1 / 199.8 ms | 15 | identical |

*Launch amortization is worth nothing here:* 8 launches and 2 launches differ by 0.05 ms/token on a
13.3 ms/token DRAM-bound stream, and the WHILE is **+0.45 ms/token (+3.4 %) slower** than the per-token
static graph -- a cost that is **per body iteration, not per launch** (identical at 2/4/8 launches), i.e.
the conditional container plus its host-visible handle write. Its honest win is device-side termination:

| arm (chat prompt, EOS at token 6) | decode steps actually run | launches | total decode | ids |
|---|---:|---:|---:|---|
| per-token static graph (`--decode-flight 4`) | 8 (3 wasted past EOS) | 8 | 108.6 ms | identical |
| `--graph-flight` (chunk 4) | 5 | 2 | 75.7 ms | identical |
| `--cuda-graph-while 8` | 5 | 1 | 76.0 ms | identical |

i.e. **-30 % total decode time** because the loop stops at EOS instead of finishing the in-flight chunk
past it. Pinned vs pageable downloads are indistinguishable at this size, as the copy accounting above
predicts (both move the bytes twice).

**Tests (all passing).** `test_qwen_ops_dsl cuda` gains `test_greedy_while_conditional` (probe, then the
production finish kernel in a real WHILE: budget 4 → 4 tokens, EOS → 1 token, context limit → 1 token
with the eos flag clear, budget 1, NaN sentinel → 1 token, replay/reuse, and exactly-the-executed-slots
history checks) and `test_pinned_download_ring` (143-slice full range through a 7-element ring, arbitrary
interior/tail ranges, empty range, out-of-range refusal, reuse, float logits, copy accounting).
`test_qwen_engine` adds the device-free `qwen_validate_while_request` negative paths (no EOS id, zero
chunk, zero vocabulary, zero capacity, position at `max_context`, chunk past the remaining context, and
the "not requested" no-op). Both device tests report "not available on this backend" and skip on dx/vk.

### Phase 9c: honest Vulkan cooperative-extension reporting, cooperative-vector spike, split-barrier experiment (opt_plan/04 §3/§7/§8, 09 V7 / M-f)

Three deliverables, one of them upstream: (1) the capability record finally reports the two cooperative
extensions the Vulkan backend negotiates (**fixed with a LuisaCompute patch**, because they are negotiated
*outside* the optional-extension registry and every query used to answer `false`); (2) the standalone
`VK_NV_cooperative_vector` spike the plan gates on — the store/layout contract is **proven**, the matmul is
**rejected on this hardware with the exact stage recorded**, no engine integration; (3) a standalone,
differentially verified split-barrier experiment whose result is **DROP** (the barrier split is measurably
*slower* than `sync_block()` on the same K-loop). Nothing in this phase changes an engine default, and **no
Qwen speedup is claimed anywhere**: the two device experiments are measurements, not integrations.

#### What the record actually says on this box (RTX 4080 SUPER, driver 13050, Vulkan)

```
caps-report: backend=vk cooperative_vector=1 fp32_config=0 usable=0 configs=16
             matrix_enabled=1 matrix_extension='VK_KHR_cooperative_matrix' matrix_workgroup_scope=1
             split_barrier_enabled=1 vector_physical_support=1 vector_memory_model=1
             vector_replicated_composites=1
  accelerators: pdl false, native async copy false, split barrier used false (enabled true),
                cooperative vector true (fp32 config false, usable false, 16 configs),
                cooperative matrix true (extension 'VK_KHR_cooperative_matrix', enabled never emitted)
```

* `cooperative_vector = true` — the extension **is** enabled (physical support + `vulkanMemoryModel`
  + `vulkanMemoryModelDeviceScope` + `shaderReplicatedComposites` all hold). Before phase 9c this field was
  `false` on this machine, which is exactly the misreading `opt_plan/09` V7 predicted.
* `cooperative_vector_fp32_config = false`, `usable = false` — the driver enumerates **16** cooperative-vector
  configurations and **none** is `FLOAT32/FLOAT32/FLOAT32/FLOAT32` with `transpose = false`, which is the only
  one the SPIR-V codegen can emit (`spirv_codegen/instruction.cpp` hardcodes the FP32 component types and
  `transpose = false`). "Enabled" and "usable" are therefore separate fields, not one optimistic bool.
* `cooperative_matrix = true`, `matrix_extension = 'VK_KHR_cooperative_matrix'`,
  `matrix_workgroup_scope = true` (`VK_NV_cooperative_matrix2` workgroup scope is enabled on top) — and
  `is_active()` stays `false` for **every** cooperative-matrix name: "negotiated" must never read as
  "emitted", and no kernel in this tree lowers to a cooperative-matrix instruction.
* `split_barrier_enabled = true` but `split_barrier used = false`: the extension is on the logical device
  (a split-barrier kernel compiles and runs) while no compiled shader has declared `SPV_EXT_split_barrier`
  yet. Two fields, because "enabled" and "already used" are different answers.

`dx`/`cuda` report the whole group `false`/`0` (fail-closed), and the per-name queries
(`is_supported`/`is_enabled`/`is_active`) now answer for both cooperative families even though neither is an
optional-registry entry. Coverage: `tests/integration/qwen/test_qwen_caps.cpp` (runs on every backend; 173
checks on vk, 18 on dx/cuda) plus the new value-initialized/conjunction assertions in
`tests/unit/qwen/test_qwen_engine.cpp`.

#### Upstream patch: `patches/luisa-vk-cooperative-extension-status.patch`

Scope: three files, ~240 lines, **read-only diagnostic surface** — no codegen, no negotiation, no behaviour
change.

* `include/luisa/backends/ext/vk_extension_ext.h` gains `VulkanCooperativeStatus` (matrix extension name /
  enabled / NV2 workgroup scope; vector physical support / memory model / replicated composites / enabled /
  FP32 config / enumerated config count) and
  `VulkanExtensionStatusExt::cooperative_status()`. The docs on the struct state explicitly that
  `matrix_enabled` never means "emitted" and that `enabled && !fp32_config` means "negotiated, not usable".
  The cooperative names stay out of the `extensions()` list on purpose: they are not registry entries, so a
  flat entry would have to invent `requested`/`active`.
* `src/backends/vk/device.{h,cpp}` record the negotiation that already happened in `_init_device`
  (physical support, the two SPIR-V prerequisites, the enumerated configuration count, which
  cooperative-matrix twin was enabled) and implement `cooperative_status()` plus the per-name answers. The
  imported-logical-device path clears them, exactly like every other unattestable capability.
* The patch is applied in the shared checkout and was verified two ways:
  `git -C LuisaCompute apply --check --reverse ../patches/luisa-vk-cooperative-extension-status.patch`
  (it is applied) and `patch -p1 --dry-run` against a pristine `git archive HEAD` export of the three files
  (it applies cleanly). The other four patches in `patches/` are untouched.

#### Cooperative-vector spike (`tests/integration/qwen/test_qwen_coopvec_spike.cpp`) — REJECTED, stage 2

Standalone, throwaway, no engine integration; on `dx`/`cuda` it prints the reason and exits 0 having
attempted nothing. On `vk`:

| stage | what it does | measured result |
|---|---|---|
| 0 | capability gate | `enabled=1 usable=0 fp32_config=0 configs=16` → proceeds (the point is to measure, not assume) |
| 1 | **store/layout contract**: `cooperative_vector_store` of `[1..16]` at byte offset 0 and `[101..116]` at byte offset 64, then `cooperative_vector_load` back, then a raw host read of the 128 bytes | **PASS, exactly as hoped**: the stores are plain contiguous FP32 (`max\|diff\| = 0`), the byte offset is honored (the second vector lands at +64 B, the first 64 B untouched), and the load round-trip is bit-exact. A cooperative vector is laid out as the N contiguous FP32 words a host upload would produce, so a weight matrix *can* be prepared offline. |
| 2 | `cooperative_mat_mul<float,float>` (K=8, M=16) against a CPU reference, both candidate matrix layouts, one-invocation and warp variants | **FAILS AT `device.compile`, not at run time**: the SPIR-V module is produced, and then the Vulkan driver kills the process (SIGSEGV, exit 139) inside pipeline creation. Reproduce with `--allow-unsafe-matmul`; the log ends at `stage2[1 invocation]: compile`. This is the honest "exact stage" the plan asks for: with no matching configuration the driver does not return an error, it crashes. |
| 3 | the ≥ 1.5× decision gate | **not attempted**: timing a kernel whose output cannot be computed correctly would be a fake pass |

Decision: **NOT ADOPTED, no engine integration, no follow-up work in the engine.** The recorded reasons:
this driver's only float configuration is `FLOAT16` with `transpose = TRUE`; the codegen hardcodes
`FLOAT32` and `transpose = false`; making the FP16/transposed configuration reachable is an **upstream
codegen** task (an AST-level transpose knob plus FP16 cooperative-vector support, then re-running this
spike) — and even then it would be an FP16 weight path, i.e. a different numerical contract that needs its
own accuracy and byte-economics gate, not a free win. The store/layout half is good news and is recorded so
a future upstream change starts from evidence rather than from the log line.

#### Split-barrier experiment (`tests/integration/qwen/test_qwen_split_barrier.cpp`) — measured, DROP

A tiled prefill-GEMM K-loop with a 3-stage shared-memory ring (16×16 output tile, 256 threads, `BK = 32`,
one FP32 barrier per K iteration). The two arms differ **only** in the barrier:

* `sync` — `sync_block()` at the top of each iteration;
* `split` — `split_barrier_arrive()` (publish the current stage) → stage the **next** K chunk of both tiles
  inside the barrier window → `split_barrier_wait()` (acquire the current stage).

Three stages are required, not two: an arrive/wait pair lets an invocation run one iteration ahead, so the
buffer being written must not be the buffer another invocation is still reading. Both arms issue exactly one
barrier per iteration, so the comparison is like-for-like.

Correctness: the sync arm matches an independent CPU reference, the split arm matches it too, and the two
device buffers are **bit-identical** (also at the timed shapes). A `K`-chunk sweep (1/2/3/4 chunks,
`worst|diff| ≤ 3e-7`) pins the ring, and an occupancy probe (1024/1024 slots, `max tid = 255`) pins the
`dispatch()`/block mapping.

| shape | arm | cold-ish ms (median of 30) | GFLOP/s | hot ms | vs sync |
|---|---|---:|---:|---:|---:|
| 1024×1024×2560 | `sync_block()` | 1.820 / 1.718 / 1.713 | 2950 / 3125 / 3134 | 1.872 / 1.768 / 1.768 | — |
| 1024×1024×2560 | split arrive/wait | 1.800 / 1.801 / 1.799 | 2983 / 2981 / 2985 | 1.852 / 1.856 / 1.854 | **−1.1 % / +4.8 % / +5.0 %** |
| 512×512×5120 | `sync_block()` | 0.873 / 0.875 / 0.867 | 3076 / 3069 / 3098 | 0.910 / 0.909 / 0.909 | — |
| 512×512×5120 | split arrive/wait | 0.916 / 0.906 / 0.923 | 2931 / 2964 / 2908 | 0.969 / 0.972 / 0.969 | **+5.0 % / +3.5 % / +6.5 %** |

(six independent runs on a quiet machine; cold-ish = a 96 MiB flush kernel, larger than the 64 MiB L2, timed
separately and subtracted; 4 launches per rep, 30 reps, median.) Decision rule, fixed before the run: keep
only if the split arm is both correct and ≥ 1 % faster. It is slower at the second shape in **every** run and
at the first shape in five of six (the exception is the very first measurement of the first run, where the
sync arm carries the clock warm-up), and slower in the hot/L2-resident case every time. That is what the
mechanism predicts: the arrive costs an extra control-barrier instruction while the split window here contains
only a handful of shared-memory stores, so there is no latency to hide; both arms run at ~3.1 TFLOP/s FP32,
i.e. the loop is not barrier-bound.

Gating, all of it exercised: the split arm is selected only for `backend == "vk" && extension enabled`
(`may_select_split_barrier`), and the device is created with an **explicit**
`VulkanDeviceConfigExt::enable_shader_split_barrier()` opt-in. A second run with
`--opt-out-probe` (a separate process: the Vulkan backend allows one live Device per process) proves that
`enable_shader_split_barrier() = false` really leaves the extension disabled and therefore unsaleable by the
policy. `dx`/`cuda` print "split arm NOT SELECTED", run the `sync_block()` arm only, and the policy checks
assert they can never select the split path even if a capability bit were wrong. There is **no silent
fallback** anywhere: every skip prints its reason.

Limitations, stated plainly: this is a **standalone measurement of one isolated pipeline shape**, not an
engine integration, and its result is negative. No Qwen kernel emits split barriers, the tiled prefill GEMM
in `qwen/dsl_ops.cpp` keeps `sync_block()`, and the recorded outcome is "do not adopt". A future attempt
would need a pipeline where the barrier window actually contains the next tile's *global* loads (i.e. a
producer/consumer split across warps, with `cp.async` or an explicit copy warp) rather than a few shared
stores; that is a bigger restructuring than this phase scoped.

### Phase 9d: Tile cleanup — multi-token GDN conv chunk and attention length buckets (opt_plan/04 §6 items 1–2)

Two bounded items from `opt_plan/04-backend-accelerators.md` §6, nothing else. The Tile engine stays the
differential/cross-backend reference: no broad rewrite, no WY/UT port (item 3 is still missing, see
`tile_ops_chunk_prefill_status()`), and no DSL change (`test_qwen_ops_dsl` is 2295/2295 on dx, unchanged).
All work is uncommitted; phases 0–9c are preserved.

#### 1. Multi-token GDN Conv4+SiLU (`capture_tile_gdn_conv_silu_chunk`, `--tile-chunk-conv`)

The old schedule ran `capture_tile_gdn_conv_silu_step` **once per token per GDN layer** and paid, per token,
a position upload, a source-row staging copy, the conv launch and an output gather copy — 4 commands, 32 per
layer for an 8-token chunk. The new capture does the whole chunk in **one** kernel:

* the causal window (`kernel_size-1` rows) is carried in SSA registers and shifted one row per token, so the
  in-chunk/ring distinction needs no per-row bound test: the window starts as absolute positions
  `[pos0-R, pos0)` read from the ring and each row shifts in the row it just consumed;
* the only position-dependent parts are the initial window's ring-slot rotation `pos0 % R` and the
  negative-position mask, both from a **device-visible `pos0`** (ONE int32 upload per chunk, shared by every
  GDN layer, instead of one per token per layer) — no per-position or per-token compile;
* the ring image is produced by the same kernel (the last `min(tokens, R)` rows own the newest slots), and
  the ring pair is ping-ponged **per invocation** instead of per position parity, so `ring_in`/`ring_out` are
  distinct buffers for any row count (the Tile runtime rejects one buffer bound to two arguments when either
  is writable, and an even-length chunk would otherwise alias them);
* the packed-QKV projection now writes a dense `[chunk, qkv_n]` buffer that the kernel reads in place, so the
  per-token staging and gather copies disappear entirely.

A one-row chunk (decode, or `--prefill-chunk 1`) keeps the per-token step kernel: it is one launch either way,
and with the generalized kernel forced onto one-row chunks the step arm measured 1384.14/1384.99 ms/token
against 1412.37/1413.25 ms/token (+2.0 %, two runs per arm, identical ids) because the window build and the ring
image are pure overhead for a single row. So `--tile-chunk-conv 1` means "chunks with more than one row". `--tile-chunk-conv 0` restores the historical
path for every row count as the differential/AB arm. Verified wiring: at `--prefill-chunk 1` the two arms are
identical (54 shaders, TTFT 10983.7 vs 10991.0 ms, TPOT 1366.25 vs 1366.96 ms, same ids), because no
row-count>1 chunk can occur and no chunk-conv shader is compiled. Exactly `prefill_chunk` captures exist (keyed by
`(channels, kernel_size, rows)`), no per-token JIT.

Device evidence (`tests/integration/qwen/test_qwen_ops_tile.cpp`): the chunk kernel is compared against the
independent CPU oracle over 13 `(pos0, rows)` cases including `rows < R`, `rows == R`, `rows > R`,
`pos0 % R ≠ 0` and a non-zero ring history (outputs and ring image), and against the per-token step kernel on
the same inputs (out and ring agree to FP32 rounding, `rtol 1e-6`; the tap sum order is identical but the step
form materializes `(history+source)*weight` per tap while the chunk form contracts the register window, so
mul/add contraction differs by ~1 ulp).

Launch counts, dx, 8-token prefill chunk, 24 GDN layers (`--benchmark-csv`, per-command samples):

| schedule | commands per chunk | conv+SiLU ms per chunk |
|---|---:|---:|
| per-token step (phase 9c) | 768 (`upload_pos` 192 + `stage_src` 192 + `conv_silu_step` 192 + `gather` 192) | 111.97 (78.74 in the 192 step kernels, 33.23 in the 576 copies) |
| multi-token chunk (9d) | **25** (`upload_chunk_pos` 1 + `conv_silu_chunk` 24) | 111.61 (0.07 + 111.54) |

So the item removes **96.7 % of the conv commands** and all per-token staging traffic, but it does **not**
move the wall clock: the chunk kernel costs 4.65 ms per launch at `rows=8` against 0.41 ms per step launch
(8 steps = 3.28 ms), i.e. it is ~1.4× slower than the eight launches it replaces, which cancels the 33 ms of
copies it deletes. The whole conv block is only 1.9 % of a measured 8-token chunk, so even a perfect kernel
would move TTFT by <1 %. The per-launch cost is recorded as an open question, not explained: the serial nest
carries three register tiles, and the DX/VK XIR route materializes per-program snapshots, which is the most
likely mechanism, but it was not profiled and no claim is made.

#### 2. Attention length buckets (`--tile-attn-bucket`, default 512)

The device-visible-length attention shaders stream a **static** key range, so the phase-9c engine compiled one
specialization per `(rows, capacity)` with `capacity = max_context` and masked everything above `kv_len`: a
20-token session scanned 4096 keys per query per layer. `tile_attention_bucket_capacity()` (one policy, shared
by the executor, the unit test and the device test) now selects `ceil(kv_len/bucket)*bucket` clamped to
`max_context`, so the scan is `ceil(kv_len/512)*512` and the length **stays device-visible** (no per-token
compile, no static shape).

Bounded by construction: at most `ceil(max_context/bucket)+1` buckets (64 is the hard limit the engine
validates) and at most `ceil(max_context/bucket)+1` decode + the same number of full-chunk prefill + `rows-1`
small-bucket prefill specializations precompiled at create. Anything else — in practice a ragged prefill tail
above the smallest bucket — goes through a **bounded, logged** lazy cache
(`max_attention_bucket_lazy`, default 32): every request-path compile prints
`tile attention: compiling bucket shader on the REQUEST path (rows=…, key capacity=…, lazy n/32)` and is
counted in `info().tile_attention_lazy_compiles`; exceeding the bound is a recoverable failure, never a silent
fallback to the full capacity. `--tile-attn-bucket 0` restores the phase-9c behaviour exactly (one
full-capacity specialization, same shader count).

Device evidence (`test_qwen_ops_tile`, 164 checks): the bucket policy at 0/1/511/512/513/1024/1025/1535/1536 and
the capacity clamp; every bucket specialization vs the CPU oracle at `kv_len ∈ {1, 511, 512, 513, 1023, 1024,
1025, 1535, 1536}` (3e-5) **and bit-identical to the full-capacity specialization at the same length** — the
keys a bucket does not visit contribute exactly `acc*1 + v*0` in the full-capacity arm, so dropping them cannot
move a bit; plus a prefill bucket arm (rows = 3, causal diagonal) at 512/513/1024/1536. Measured per-launch
attention time on dx (`--benchmark-csv`): `attn.prefill_dyn` 1237.6 ms → 168.1 ms (−86.4 %) and
`attn.decode_dyn` 50.9 ms → 8.3 ms per launch (−83.7 %) at a 17-token context, which is the mechanism stated
as a number: cost is the bucket, not the capacity.

#### End-to-end A/B (real GGUF, identical greedy ids in every arm)

dx, `data/Qwen3.5-4B-BF16.gguf`, 17-token prompt, `--prefill-chunk 8 --max-new-tokens 4`, two runs per arm,
each arm its own process (all 8 runs produced `271 248068 271 248069`):

| arm | TTFT (ms) | decode (ms/token) | Δ TTFT | Δ decode |
|---|---:|---:|---:|---:|
| phase-9c binary | 17363.7 / 17375.5 | 1789.48 / 1791.38 | — | — |
| chunk conv only | 17391.6 / 17347.2 | 1788.07 / 1789.35 | ±0.2 % | ±0.1 % |
| buckets only | 14735.0 / 14855.8 | 1441.84 / 1442.50 | **−15.2 % / −14.5 %** | **−19.4 % / −19.5 %** |
| both (default) | 14777.5 / 14900.6 | 1442.12 / 1442.76 | **−14.9 % / −14.2 %** | **−19.4 % / −19.5 %** |

vk (same checkpoint, 8-token prompt, `--prefill-chunk 8 --max-new-tokens 2`, one decode step measured; the
phase-9c binary is the reference):

| arm | TTFT (ms) | decode (ms/token) |
|---|---:|---:|
| phase-9c binary | 234753.1 / 233254.0 | 13637.1 / 13046.8 |
| chunk conv only | 233350.0 | 13068.9 |
| buckets only | 229037.9 | 10759.7 |
| both (default) | 228947.5 / 228819.9 | 10709.5 / 10587.9 |

On vk the attention bucket is −18.5 % decode (−1.9 % TTFT, because a `rows=8` prefill is dominated by the
scalar-emulated `tile::mma` GEMVs); the conv item is inside the ±4 % run-to-run spread of that backend.
`cuda`: **Tile is unavailable in this build** and no CUDA number is claimed — `test_qwen_ops_tile cuda` says
`CUDA native Tile IR is experimental; set LUISA_CUDA_TILE_IR=1 or use Lowering::TIRX`, and with
`LUISA_CUDA_TILE_IR=1` it says `CUDA Tile IR is unavailable: configure CMake with CUDA 13.4+ Tile headers and
tileiras` (this box has CUDA 13.0). The TIRx route is not an alternative here either: the build has
`lc_tile_tirx_bridge = false` and `lc_tvm_stack = false`, so `Lowering::TIRX` is compiled out
(`CUDA backend was built without the optional TIRx bridge`).

Shader counts (dx, `info().compiled_shader_count`, `--max-new-tokens 4`, cold vs warm is the backend's disk
shader cache): 233 (phase 9c) → **255** with both items at `max_context 4096`, `bucket 512`, `chunk 8`: +8 chunk
conv captures (rows 1–8) and +14 attention specializations (8 decode buckets + 8 prefill buckets + 7 small-bucket
row variants, minus the 9 that phase 9c already compiled). With `--tile-attn-bucket 0 --tile-chunk-conv 0` the
count is exactly the phase-9c 233. A 32-bucket configuration (`--max-context 512 --tile-attn-bucket 16`) reports
`buckets=32 bucket-precompiled=71 bucket-lazy=1`, and the lazy compile is visible in the log.

Create-time cost of the extra specializations: +22 shaders cost ~12.5 s on dx when **cold** (0.57 s each) and
~0.4 s when the backend's shader cache is warm; on the 32-bucket configuration a cold create compiled 71 bucket
shaders in 65.9 s. That is a one-time cost, bounded by the bucket policy, and `--tile-attn-bucket 0` removes it.

Not done, and not claimed: Tile WY/UT chunk-64 GDN prefill (item 3) is still missing and still reported by
`tile_ops_chunk_prefill_status()`; `tile_n = 16` is still forced by the 256 KiB/worker snapshot budget (item 4);
no DSL kernel, no `tile::mma` path and no state layout changed. The Tile engine remains a correctness reference,
not a performance path — with buckets it is ~18 % faster at decode, i.e. 1.44 s/token against the DSL's 14 ms.

### Phase 9e: fused one-CTA GDN recurrence + gated output RMSNorm (opt_plan/03 §5)

The last remaining row of `opt_plan/03-decode-latency.md` §5. **Implemented, differentially clean on cuda/dx/vk,
and measured as a small end-to-end LOSS on all three backends, so it is NOT the default** (opt-in:
`--gdn-fused-recurrent 1`). Nothing else changed: prefill (`rows > 1`), every MTP path, the static CUDA decode
graph contract and all six LuisaCompute patches are untouched, and no commit was made.

**Why the default cannot host this fusion.** The default T=1 schedule launches
`make_qwen35_gdn_decode4_kernel` — **four blocks per V head**, each owning a 32-row slice of the key axis — and
then the separate gated output RMSNorm `make_qwen_gdn_gated_norm_kernel`, because that norm reduces over the
head's whole 128 channels and the four blocks of a head are independent. The head row therefore round-trips
through global `_gdn_out` and costs a second launch per GDN layer (24 launches/token). The plan's alternatives
were (a) keep two launches, (b) clusters/DSM (unavailable: `sm_90+` datacenter only, and consumer Blackwell
exposes a `1×1×1` cluster) and (c) the one-CTA-per-head kernel where a launch dominates. This phase implements
(c) **as a real fused kernel** and measures it.

**The kernel** (`make_qwen35_gdn_recurrent_norm_kernel`, model geometry only, 128 threads = one state column
per lane):
* the state update, the `err`/output dots and their FP32 accumulation order are exactly
  `make_qwen35_gdn_recurrent_kernel(head_dim = 128)`'s (decay the column, per-lane serial dot over the key axis
  with shared k/q broadcast, rank-1 update, output dot);
* the state column is written back **before** the reduction, so its 128 registers die before the barrier: a
  column alive across the barrier + shuffle tree spilled on DX (measured, see below);
* the gated RMSNorm is a reduction over **distinct registers** (one lane owns one value column): per-lane `o²`
  → `warp_active_sum` (the partial of channels `[32w, 32w+32)`) → `sred[w] = partial` by lane 0 →
  **one `sync_block()`** → a fixed 4-term fold of the four warp partials that every lane performs identically,
  so `inv = rsqrt(total/128 + eps)` is bit-identical across the four warps and no broadcast is needed. That is
  the block-wide reduction the 4-CTA kernel cannot do;
* the **direct** norm weight (`ssm_norm`, not `1+w`) and the SiLU(z) gate (`z = attention_gate` projection) are
  applied there, so neither the recurrent output nor the normalized output ever touches global memory:
  **no `_gdn_out` round trip, no second launch**. Shared memory is 2·128 floats + 4 floats.
Fail-closed: `head_dim != 128` or `tokens != 1` is a guarded no-op in the kernel **and** refused by
`plan_qwen35_gdn_recurrent_norm_dispatch` / `validate_qwen35_gdn_recurrent_norm_inputs` (both are asserted by the
device suite), and the engine refuses to select it for any other geometry.
Selection is a host-side engine-build decision (`QwenEngineOptions::dsl_gdn_recurrent_norm`, requiring
`dsl_decode_specialized`), so the recorded decode body stays **argument-invariant**: with the fused kernel in the
body CUDA still reports `cuda-graph-steps == 23 == decode steps` and `updates == 0` (measured, 24-token run).

**One unresolved inconsistency.** The kernel was first written with the head row *staged in shared memory*
(which made it bit-identical to the two-launch pair on cuda and vk) and later changed to the register reduction
above. Isolated, the two are indistinguishable (cuda 8.58 vs 8.46 µs, dx 33.37 vs 33.27 µs); end to end they are
not (staging: 13.35/13.32 ms/token = +0.9 %; register: 13.67/13.67 = +3.3 %). That gap is far larger than the
isolated difference can explain and was **not** resolved — both are losses, which is what the decision rests on.
If the opt-in is ever revived, the staging form is the better starting point.

**Numerics.** The shipped kernel runs the one-CTA recurrence unchanged and re-associates only the head
reduction, so its deviation from the one-CTA pair is <= 2.9e-06 (cuda) / 3.8e-06 (dx) / 1.9e-06 (vk)
absolute over six **carried** decode steps (the superseded staging form was bit-identical on cuda and
vk). The independent FP64 oracle
(`ref_qwen_gdn_recurrent` → `ref_qwen_gdn_gated_norm`) gates both realizations at 2e-4/2e-3. The device suite
also checks a decayed-to-zero state, an all-zero gate/conv row and extreme `silu(z)` arguments for finiteness
(no NaN/inf anywhere), a reset that reproduces step 0 bit-exactly, and decode4-vs-fused (≤ 1.1e-05). See
`test_gdn_recurrent_norm` in `tests/integration/qwen/test_qwen_ops_dsl.cpp` (cuda 2401 checks, dx 2353, vk 2353).

**Isolated measurement** (`benchmark_qwen_decode --gdn-recurrent-norm`, per GDN layer, one reported sample = one
submit of 24 chains divided by 24, min of 50 samples; `pair_one_cta` and `pair_decode4` are the two-launch
schedules, `fused` is the new kernel):

| backend | recurrent_only (1 launch) | gated_norm_only (1 launch) | pair_one_cta (2) | fused (1) | pair_decode4 (2, default) |
|---|---:|---:|---:|---:|---:|
| cuda | 6.19 µs | 4.53 µs | 10.63 µs | **8.46 µs** | 10.50 µs |
| dx | 7.98 µs | 5.66 µs | 11.60 µs | **33.27 µs** | 13.60 µs |
| vk | 8.22 µs | 5.77 µs | 12.02 µs | **10.05 µs** | 17.43 µs |

Per layer the fused kernel removes one launch and 2 head rows (**32 KiB**) of global traffic on top of the
4.29 MB chain both recurrent schedules move (2 MiB state read + 2 MiB state write + q/k + the V row). So the
isolated number says, against the **default** `pair_decode4` chain: **−19.4 % on cuda, −42.3 % on vk, and
+145 % on dx** (against `pair_one_cta`: −20.4 %, −16.4 %, +187 %; a repeat run of the same arms moved these by
≤ 6 % on cuda/vk and ≤ 1 % on dx). The cuda win is the launch
plus the round trip; the dx loss is unexplained and recorded as such (the same 128-iteration unrolled body is
fast in the reference kernel, so the most plausible mechanism is that the HLSL compiler spills the 128-float
column in the larger kernel — D3D12 spills go to a scratch UAV — but it was not profiled and no claim is made).

**End-to-end** (real `data/Qwen3.5-4B-BF16.gguf`, 17-token prompt, `--max-context 512 --prefill-chunk 8
--max-new-tokens 24`, each run its own process; every arm on every backend produced the **same 24 greedy ids**,
and the FP8 arm on cuda produced the same 4 ids as its pair arm):

| backend | pair (default) ms/token | fused ms/token | Δ | runs |
|---|---:|---:|---:|---|
| cuda | 13.23 / 13.24 | 13.67 / 13.67 | **+3.3 %** | 4 + 4, incl. an order-reversed control |
| cuda, `--max-context 4096` | 13.28 / 13.29 | 13.83 / 13.83 | **+4.1 %** | 2 + 2 |
| dx | median 18.44 (14.22…20.11) | median 20.52 (18.31…21.08) | **+11 %** | 6 + 6 |
| vk | median 17.88 (16.85…19.28) | median 18.49 (16.08…19.74) | +3.4 % (±9 % spread) | 6 + 6 |

TTFT is unchanged in every pair (32.1–32.3 ms on cuda, 52–68 ms on dx, 77–100 ms on vk), which is what the
`rows == 1` gate promises.

**Decision: DROP as the default.** The isolated win does not survive a real decode step: on cuda the fused
chain is 2.0 µs cheaper in isolation, yet a 23-step token moves from 13.23 to 13.67 ms (+3.3 %, four consistent
runs at two context sizes, including an order-reversed control), i.e. the fusion is ~18 µs/layer **slower** in
situ. The most plausible reading (an
inference, not a profile) is that the isolated arm keeps each layer's 2 MiB state L2-resident, whereas a real
token streams 8 GiB of weights and 48 MiB of GDN state through the same 64 MiB L2: the one-CTA recurrence has
4× longer per-lane dependent chains and only 32 blocks (of 80 SMs), so it loses more to latency than the launch
and the 32 KiB round trip are worth. On dx the codegen penalty above makes it 11 % worse outright. The default
therefore stays **decode4 + the separate gated norm**, the fused kernel stays opt-in and reported
(`gdn recurrent+norm: fused_one_cta|separate_pair (reason)`), and the row is no longer "not fused, measure, do
not guess" but "fused, measured, rejected" — with the one-CTA grid, not the fusion, as the reason.

### Phase 10: final combined-tree acceptance on T16 (`opt_plan/06` §7 gates)

The closing regression + benchmark matrix, run once against the final uncommitted tree (phases 0–9e). Full
evidence: **`.kimix_cache/qwen_benchmark/phase10/REPORT.md`** (with `summary.json`/`summary.md`, per-bench
`bench/*/report.md`, and `logs/`). Tier **T16** only (`RTX 4080 SUPER`, CC 8.9, 16 GiB): every T32 cell is
`NOT RUN`, never inferred.

* Suites: `test_qwen_ops_dsl` 2401/2353/2353 and `test_qwen_kv_cache` 174/174/174 on cuda/dx/vk, `test_qwen_caps`
  18/18/173, Tile 164 (dx) / 164 (vk), MTP 36, native GEMM 134, native attention 89, weights (device) 18 tests;
  host 4700/32661/7837/367 asserts, 139 Python tests, `--self-test` ALL PASS.
* **One real failure found and fixed**: the `test_qwen_kv_cache` non-finite FP8-append arm still asserted the
  superseded "excluded from the maximum" rule (both the device kernel and the host reference map a non-finite
  value to one full-scale unit, `qwen_kv_counts_for_max`). Test expectation corrected, one extra finite-row check
  added, no production code changed.
* Environment limits, named not skipped: **CUDA Tile IR** is refused at engine creation ("CUDA native Tile IR is
  experimental; set `LUISA_CUDA_TILE_IR=1` or use `Lowering::TIRX`"), PDL is a no-op below CC 9.0, Vulkan
  exposes `VK_NV_cooperative_vector` with no FP32 config. (The SM count used to be in this list; it is no
  longer — see the utilization bullet below.)
* Correctness cross product (real GGUF, greedy ids): bf16/cuda with KV f32/bf16/fp8, fp8 + native Q8_0 with f32
  KV, dx and vk with all three KV dtypes, graph on/off, pinned on/off, the opt-in WHILE chunk (including a
  device-side EOS early exit: `requested=15 executed=11 eos=1`), merged projection pack on/off for bf16/fp8/q8,
  and the fused-GDN opt-in all reproduce the **same ids**; the fail-closed combinations (Tile with a non-f32 KV,
  W4 with a projection pack, native attention with a bf16 KV) refuse with a named reason. W4 stays experimental
  (its 32-prompt T16 quality gate failed in phase 8b) and no W4 performance is promoted; the Q3 FP8-KV gate that
  passed in phase 8d is cited, and its short final ids are re-validated here.
* Performance headline (median of 5 cold processes, roof = the **measured** 603 GB/s of this machine): cuda bf16
  **13.79–14.32 ms/token** at ctx 32768 = **100–107 % of the roof**, FP8 **8.15–8.55** (1.66×), native Q8
  **8.53–8.64** (1.67×); cold TTFT at P=128/512/2048/4096 is 72.5/160.4/339.2/673.4 ms (bf16) and
  342/344/628 ms (fp8 at 128/512/2048); dx 14.68/14.91 and vk 15.13/15.89 ms/token at ctx 4096. FP8 KV is a
  **capacity** feature on this part (cache 563 MiB vs 2100 MiB) but costs ~4 ms/token at 32 K versus bf16 KV.
* Utilization (gate 11) is **measured**, not unavailable: `test_qwen_caps` reports the CUDA driver's
  **80 SMs** (`sm_count_source=device_query:multi_processor_count`, added by
  `patches/luisa-cuda-multiprocessor-query.patch`), so blocks/SM is **2.2** for the long-context
  split+dedup grid (176 blocks) and **0.2** for the legacy per-q_head grid (16 blocks), and
  `ncu --graph-profiling graph` measures the WHOLE decode step as one range:
  `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` = **89.11 %** active (10.89 % idle) at
  P128/C512 legacy, **86.07 %** (13.93 %) at P2048 and **93.05 %** (6.95 %) at P32000/C32768 split.
  dx/vk still answer no SM count (`unqueryable`, never a guess). Details in the report §17.

### Experimental: CUDA 13.5 native-Tile typed-FP8 prefill GEMM (`--cuda-tile-fp8`)

Status: **implemented, correct, measured SLOWER on this machine, default OFF and never
auto-selected.** The measured default is unchanged everywhere: the native NVRTC tensor-core
GEMM (Phase 7a) keeps every eligible prefill projection on cuda, and the DSL MT/legacy GEMM
keeps dx, vk and every non-FP8 weight format. This path is a research/capability arm for the
CUDA 13.x Tile IR, not a faster GEMM.

### CUDA 13.5 feature triage

The toolkit review was filtered by what actually exists on this RTX 4080 SUPER (Ada `sm_89`) and by the hot paths in this engine; it was not a checklist of vendor headlines.

| CUDA 13.5 feature | Decision |
|---|---|
| Tile IR `f8E4M3FN` / `f8E5M2` on Ada | **Implemented experimentally** as the typed-FP8 prefill GEMM below. It uses FP8 weight storage, not FP8 activations or block-scaled MMA. |
| `num_worker_warps_per_cta` = powers of two 1…32 | **Implemented and measured** through `LUISA_CUDA_TILE_WORKER_WARPS`; rejected as a performance fix (§worker-warp evidence below). |
| CUB reductions with device-resident problem sizes in graphs | Not adopted: decode argmax has a fixed vocabulary extent and a specialized device kernel; there is no dynamic CUB reduction on the hot path to replace. |
| `cub::DeviceBatchedTopK` | Not adopted: the engine performs greedy argmax, not batched top-k sampling; the new API's unsorted, nondeterministic initial contract is not a drop-in replacement. |
| `-nvvm-latest` for pre-Blackwell | Not adopted: the engine compiles through NVRTC/driver loading and has no nvJitLink/libNVVM device-link stage. NVIDIA documents this option for nvJitLink/libNVVM/nvlink LTO, not as a general NVRTC flag. |
| `__sin2pif` / `__cos2pif` / `__sincos2pif` | Not adopted: RoPE does not already carry cycle-based phase, and changing its validated approximation for one instruction has no established end-to-end benefit. |
| cuBLAS low-alignment BF16 improvements | Not applicable on Ada: the release note targets Hopper and newer; the engine also owns its BF16 prefill kernel rather than calling cuBLAS. |
| `ct::scaled_mma()` / `mmaf_scaled` | Not applicable on Ada: the underlying scaled operation is Blackwell-only. No unsupported substitute is emitted. |
| PDL, TMA multicast/locality, CC 10+ atomics | Not applicable on `sm_89`. The existing PDL probe remains truthful and reports zero benefit on this machine. |

What it is: `src/qwen/qwen/cuda_tile_gemm.h/.cpp` captures ONE Tile DSL kernel per static
(m, n, k, epilogue) shape, lowers it with `Lowering::NATIVE` (CUDA Tile C++ -> NVRTC Tile IR ->
`tileiras`) and dispatches it over the **existing** packed FP8 ABI: the same FP32 activation
view, the same `uint32` packed E4M3 weight buffer (reinterpreted in place as
`tile::float8_e4m3fn`: no repack, no widened copy, no host staging), the same FP32 64-element
group scales, and the same residual/output views as the native and DSL GEMMs.

Numerics (deliberate, not a bit-exact path): no activation is quantized. Per 64-element K
block the FP32 activation tile and the typed FP8 weight tile are widened to BF16 (the E4M3 ->
BF16 weight widening is exact), ONE BF16 tensor-core partial dot accumulates in FP32, and the
FP32 group scale multiplies the **partial dot AFTER the contraction** - never the individual
weights, so no narrow arithmetic and no scale-rounding of the weight operand appears anywhere.
The optional residual is added once at the end. Against the FP32-activation reference the
deviation is the single FP32 -> BF16 activation rounding plus FP32 reassociation
(~1.6e-3 relative RMS, the figure the native kernel reports); against an oracle that rounds
the activation the same way it agrees to FP32 accumulation order. `test_qwen_native_gemm`
measures both.

Engine wiring (`src/qwen/qwen/{engine.h,dsl_engine.cpp}`), exactly:

- `QwenEngineOptions::cuda_tile_fp8_prefill` (default **false**) plus `cuda_tile_m` /
  `cuda_tile_n` (defaults 64/64; powers of two in [16, 128], tile volume inside the native
  Tile budget, `tile_k` == scale group == 64). With the flag false the engine does **not**
  probe, does **not** JIT and does **not** change a single dispatch: the native NVRTC /
  DSL MT / legacy chain is byte for byte the previous behaviour.
- With it true, `create()` is **fail closed** and returns an error with the reason for: an
  invalid tiling, a prefill weight copy that is not FP8 E4M3 with the 64-element scale group
  (an ordinary `dsl-fp8` session and the FP8 second copy of a dual-copy `dsl-w4` session
  qualify; BF16/Q8_0/W4 prefill do not), a non-cuda backend, a missing `LUISA_CUDA_TILE_IR=1`,
  a device/toolkit whose typed-FP8 Tile compile probe fails, or a statically known shape the
  kernel cannot serve (K not a multiple of 64, extent outside the static bounds, ragged N
  without PARTIAL typed-FP8 support). There is no quiet no-op.
- In `project()` for prefill `rows > 1` the Tile arm is tried **before** the native NVRTC
  kernel, and only for the shapes the measured native rule already serves: `rows >= 16` and
  `N >= 256`. The narrow `N = 32` GDN gate/alpha/beta pair, decode (`rows == 1`), MTP and the
  static CUDA decode graph are untouched and never contain this command. A `build_dispatch`
  failure is an **engine error**, not a fallback: the caller asked for this realization.
- Reporting: `prefill cuda tile: native_tile_fp8(64x64x64, rows>=16, launches=..., lazy_compiles=...,
  cache_hits=..., shaders=..., lazy_jit=...ms, probe=...ms, ragged_n=refused)` and
  `engine final: cuda-tile-prefill=... cuda-tile-prefill-launches=...`. A default run prints
  `prefill cuda tile: off (experimental, opt-in: --cuda-tile-fp8)` and
  `cuda-tile-prefill=0`, so a report can never claim the Tile realization engaged. The
  `prefill gemm:` line itself is unchanged (scripts/benchmark_qwen.py parses it with an
  anchored regex).

Limits, measured or read out of the toolkit rather than assumed:

- Toolkit: the run needs `LUISA_CUDA_TILE_IR=1` exported (the CUDA backend requires an
  explicit opt-in for its experimental native Tile IR route) and a CUDA 13.4+ toolkit whose `cuda.h`/`nvrtc.h`/`tileiras` match
  (see `patches/luisa-cuda-tile-ir-xmake.patch`). The numbers below come from **CUDA 13.5
  (V13.5.37)** with the scratch config `.kimix_cache/xmake-tile135` (builddir
  build-tile135). On Windows, `patches/luisa-cuda-tile-toolkit-env.patch` derives the toolkit
  root from the configured `tileiras` and pins `CUDA_PATH`/`CUDA_HOME` in its child process, so
  a machine-wide setting for an older CUDA cannot mix a mismatched `nvJitLink` into the build.
  The caller/system environment is not modified.
- **DX/VK are unchanged.** The Tile -> XIR bridge rejects FP8 storage explicitly, so both
  backends keep the DSL MT/legacy GEMM; their correctness and performance numbers in the other
  phases of this page stand as recorded.
- Typed-FP8 PARTIAL (masked) tile loads only compile from **sm_90** up: below it, an FP8 weight
  tile whose N is ragged against `tile_n` cannot be compiled ("the `__nv_fp8_e5m2` and
  `__nv_fp8_e4m3` types are not supported in tile code for architectures below sm_90"). The
  helper probes that at create time and reports it as `ragged_n`; it is refused on the measured
  sm_89 machine. **Every model-eligible N (1024 / 2560 / 8192 / 9216, with K 2560 / 4096 /
  9216) is a multiple of 64**, so no eligible projection is ragged at the default tile. A
  ragged M is always fine, because its tiles are FP32.
- JIT per static shape: the Tile IR needs the extents, so the FIRST dispatch of every
  (m, n, k, epilogue) compiles. Measured on this machine: 12 prefill shapes = **10.5 s** of
  lazy JIT at 64x64 and **14.9 s** at 128x64 (default worker-warp hint), i.e. ~1 s per shape of
  start-up/TTFT cost that no disk cache amortizes. `lazy_compiles` / `lazy_jit` report it
  instead of hiding it.
- Performance (honest): the 64x64 tile is **~1.3x** (1.31-1.32x across the arms) the
  same-run native NVRTC time (geomean over the 12 measurable prefill rows) and 128x64 is
  **~2.6x** (2.56-2.66x). Widening CUDA 13.5's
  `num_worker_warps_per_cta` (`LUISA_CUDA_TILE_WORKER_WARPS`, see
  `patches/luisa-cuda135-tile-worker-warps.patch`) does not fix it: 8 worker warps is the best
  arm at 2.56x for 128x64 and still 1.92x for 64x64, and 1/2 are pathological. Therefore the
  default stays the native NVRTC GEMM and this path has no performance claim.
- End-to-end wiring check (real `dsl-fp8` GGUF, 128 prompt tokens, 8 generated tokens): the
  default and Tile arms produced the exact same ids `220 17 15 16 22 220 17 15`. The Tile arm
  reported 200 prefill launches, six static-shape compiles and 5344.3 ms of lazy JIT; native
  prefill launches fell from 200 to zero while decode graph steps and updates stayed 7 and 0.
  Its 5386.8 ms cold TTFT therefore includes that JIT and is NOT a warm-throughput comparison
  with the default's 41.8 ms. Decode was 8.59 vs 8.53 ms/token on these short runs, consistent
  with the untouched decode path, not a claimed improvement. Raw logs:
  `.kimix_cache/cuda135/model_fp8_default.log` and `model_fp8_cuda_tile.log`.

Commands (the normal `build/windows/x64/release` configuration is CUDA 13.5; a complete
from-scratch configure command is in `.kimix_cache/cuda135/REPORT.md`):

```
xmake build -P . example_qwen test_qwen_engine test_qwen_native_gemm

cd build/windows/x64/release
# the experimental arm (prefill only, dsl-fp8 weights):
LUISA_CUDA_TILE_IR=1 ./example_qwen.exe --backend cuda --version dsl-fp8 --cuda-tile-fp8 \
    --prompt "Hello" --max-new-tokens 32
# the same experiment at another tile (128x64 is the measured-slower large tile):
LUISA_CUDA_TILE_IR=1 ./example_qwen.exe --backend cuda --version dsl-fp8 --cuda-tile-fp8 \
    --cuda-tile-m 128 --cuda-tile-n 64 --prompt "Hello" --max-new-tokens 32
# the isolated prefill-GEMM arm (per-shape rows, skip rows and the lazy-compile totals):
LUISA_CUDA_TILE_IR=1 ./benchmark_qwen_decode.exe cuda --prefill-gemm --format fp8 --iters 9 \
    --cuda-tile-fp8 --cuda-tile-m 64 --cuda-tile-n 64
```

Fail-closed arms (each one is an explicit error, never a silent fallback):

```
--version dsl-bf16 --cuda-tile-fp8        -> refused: prefill weights are not FP8 E4M3
LUISA_CUDA_TILE_IR unset --cuda-tile-fp8  -> refused: requires LUISA_CUDA_TILE_IR=1
--backend dx|vk --cuda-tile-fp8           -> refused: requires the cuda backend
--version tile-bf16|tile-fp8 --cuda-tile-fp8 -> refused: a dsl-* engine option
--w4-calibrate --cuda-tile-fp8         -> refused: the calibration runs BF16/F32
--cuda-tile-m 128 (without --cuda-tile-fp8)-> usage error (the override would be inert)
--cuda-tile-m 100                          -> refused: tile_m/tile_n must be a power of two
```

Evidence (scratch artifacts under `.kimix_cache/`, git-ignored - **logs of one machine, not
tracked performance guarantees**):

- `.kimix_cache/cuda135/tile_fp8_benchmark_evidence.md` - 64x64 vs the same-run native NVRTC rows.
- `.kimix_cache/cuda135/tile_worker_warps_evidence.md` - the worker-warp sweep, the per-row table
  and the lazy-JIT totals quoted above, with the raw `ww_*.log` / `final3_cuda_tile.log` runs.
- `.kimix_cache/cuda135/test_native_gemm_tile_unset.log`, `test_native_gemm_tile_ww16.log` - the
  differential suite on this route (CUDA 215 checks; dx/vk 67 keep the unchanged DSL path).
- Upstream patches that make the route exist:
  `patches/luisa-cuda-tile-ir-xmake.patch`, `patches/luisa-cuda-tile-fp8-storage.patch`,
  `patches/luisa-cuda135-tile-worker-warps.patch`, `patches/luisa-cuda-tile-toolkit-env.patch`.
- NVIDIA CUDA toolkit release notes (authority for the toolkit-side Tile IR / FP8 statements):
  <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>
- CUDA Tile IR 13.5 operations and architecture restrictions: <https://docs.nvidia.com/cuda/tile-ir/13.5/sections/operations.html>
- CUB device-wide reduction API (deferred-count constraints): <https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceReduce.html>
- nvJitLink options (`-nvvm-latest` scope): <https://docs.nvidia.com/cuda/nvjitlink/>
