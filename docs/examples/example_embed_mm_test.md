# example_embed_mm_test

Self-check and PyTorch-parity suite for the EmbeddingGemma-2 multimodal engine: tokenizer, weight package, feature bundles, every GPU operator, and layer-by-layer drift against the official tutorial outputs.

| | |
|---|---|
| Group | `embedding_mm` |
| Sources | `src/embedding_mm/example_embed_mm_test.cpp`, `src/embedding_mm/embed/package.h`, `src/embedding_mm/embed/feature_loader.h`, `src/embedding_mm/embed/session.h`, `src/embedding_mm/embed/ops/`, `src/embedding_mm/embed/models/` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_embed_mm_test` |
| Run | `example_embed_mm_test dx --headless` |

## Purpose

One binary that tells you whether the multimodal engine still computes what PyTorch computes. It walks the stack from the host-side loaders (BPE tokenizer, `.mmegm` weight package, `.mmfeat` feature bundles) through every atomic GPU operator to the full text / vision / audio / video / interleaved encode, comparing intermediate tensors against golden JSON at a cosine similarity above 0.9999. Run it after changing an operator, a package layout or a preprocessor.

## Architecture

- Two-layer engine. `embed/ops/` holds the atomic LuisaCompute kernels (`ops_gemm`, `ops_norm`, `ops_rope`, `ops_attention`, `ops_mlp`, `ops_conv`, `ops_pool`, `ops_embed`); `embed/models/` composes them into `vision_tower`, `audio_tower`, `language_model` and the `embedding_gemma2` orchestrator that `Session` drives.
- Weight package. `Package::load` memory-maps a `.mmegm` (`EGM2` v1) file produced by `tools/pack_model.py` from HuggingFace safetensors; linear weights are stored pre-transposed to `[K, N]` so the GEMM kernels read them coalesced.
- Feature bundles. Media never goes through a C++ decoder: `tools/preprocess.py` runs the official HuggingFace `EmbeddingGemma2Processor` and writes `.mmfeat` (`MMFT` v1) bundles — a 64-byte header (`modality`, `num_tokens`, `num_frames`, `max_patches`, patch grid, audio frames/bins) followed by `input_ids`, `attention_mask`, soft-token scatter indices, `pixel_values`, position ids and audio features. `FeatureLoader` reads them straight into device buffers.
- Suites. `test_tokenizer` (BOS/EOS/PAD ids, truncation, batching), `test_package` (header, tensor lookups, vision and audio tensors), `test_feature_loader` (image 280/1120, audio, video, interleaved), `test_kernels` (RMSNorm, L2Normalize, GEMM, ScalePixels, AxialRoPE, spatial pooler, scatter soft tokens, Conformer kernels — each checked against a host reference), `test_selective_encoder_loading` (text-only → full encoder variants) and `test_official_golden_suite` (MRL dims 768/512/256/128, vision and audio layer states, video at 1 fps and 16 timestamped frames, three interleaved queries).
- Skipping instead of failing. A suite whose input file is absent prints `[SKIP]` and the run continues, so the GPU operator checks are meaningful on a checkout without any Gemma weights; the run ends non-zero only when a check that did execute fails.

## Usage

The gallery data package (`python scripts/unpack_data.py`) restores `data/` with the bge-m3 weights and the NTC sample; this suite additionally wants an EmbeddingGemma-2 `.mmegm` package and the `.mmfeat` bundles, which you produce with `python src/embedding_mm/tools/pack_model.py <hf-model-dir> model.mmegm` and `python src/embedding_mm/tools/preprocess.py --model-dir <hf-model-dir> --export-golden`.

```bash
example_embed_mm_test dx --headless
example_embed_mm_test vk model.mmegm tokenizer.json data --headless
example_embed_mm_test cuda model.mmegm tokenizer.json data --headless
```

| Argument | Meaning |
|---|---|
| `[backend]` | `dx` (default), `vk` or `cuda` |
| `[model.mmegm]` | packed weights (default `model.mmegm`, searched in the current and parent directories) |
| `[tokenizer.json]` | Gemma tokenizer (default `tokenizer.json`) |
| `[data_dir]` | directory holding the `.mmfeat` bundles and `golden_official.json` (default `examples/embedding_mm/data`) |
| `--headless`, `--cli` | never opens a window (this tool never does); accepted for the uniform gallery CLI |

## Result

A banner, one block per suite, and a verdict. Example run on `dx` without a model package:

```text
================================================================
 LuisaCompute EmbeddingGemma-2 Automated Test & Parity Suite
 Backend: dx
================================================================
[TEST] Running Tokenizer tests...
  [SKIP] Tokenizer file 'tokenizer.json' not found.
[TEST] Running Kernel GPU unit tests...
  [PASS] RMSNorm kernel verified.
  [PASS] GEMM kernel verified.
 ALL EMBEDDINGGEMMA-2 UNIT & GOLDEN PARITY TESTS PASSED!
```

With a package and the bundles present the skipped suites run too and print `cos_sim` per layer, e.g. `Audio Layer [Subsample Conv Projection] cos_sim = 1.000000 [PASS]`.

## See also

example_embed_text and example_embed_retrieval (the text-only bge-m3 engine), example_embed_mm_vision / _audio / _video / _bench (per-modality demos on the same session).
