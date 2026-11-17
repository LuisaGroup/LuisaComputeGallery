# example_embed_text

Encodes sentences with the bge-m3 embedding engine on the GPU and prints the dense vector plus, on request, the sparse lexical and ColBERT heads.

| | |
|---|---|
| Group | `embedding` |
| Sources | `src/embedding/example_embed_text.cpp`, `src/embedding/embed/session.h`, `src/embedding/embed/scheduler.h`, `src/embedding/embed/tokenizer.h`, `src/embedding/embed/package.h` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_embed_text` |
| Run | `example_embed_text dx data/bge-m3/model.bgm data/bge-m3/tokenizer.json "hello world" --headless` |

## Purpose

The smallest complete path through the bge-m3 engine: load a packed weight package and a tokenizer, create one `Session`, and turn text into a 1024-wide L2-normalised embedding. It is the example to read before `example_embed_retrieval`, and the place to check that a freshly packed `model.bgm` produces unit-norm vectors on your backend.

## Architecture

- Host side. `parse_args` reads `<backend>`, the package and the tokenizer as three positionals, then treats every further positional as a sentence; `--sparse` and `--colbert` switch the retrieval heads on. With no sentence arguments the tool reads stdin one line per sentence.
- Session. `embed::Session::create(device, model.bgm, tokenizer.json)` composes the four engine pieces: `Tokenizer::load` (a pure-C++ Unigram/XLM-R re-implementation of the HF fast tokenizer, metaspace pre-tokenizer and `<s>`/`</s>` template), `Package` (the `BGM3` v1/v2/v3 blob table: fp32, bf16/fp16 or per-tensor int8), `WeightStore` (uploads the weights once behind the `weights_ready` fence) and an executor pool behind `Scheduler` (two continuous-batching workers by default).
- Encoding. `session->encode(text, EncodeOptions{max_seq_len = 512})` tokenizes, dispatches the 24 encoder layers (`EmbedLookupKernel`, `GemmKernel`, `AttentionScoresKernel`/`SoftmaxRowKernel`/`AttentionPvKernel` or the fused `FlashAttentionKernel`, `ResidualLayerNormKernel`) and assembles an `Embedding`: `dense` (CLS, L2-normalised), `sparse` (`(token_id, weight)` pairs of the lexical head) and `colbert` (one row per token, per-row normalised).
- What the tool prints. It tokenizes a second time with the engine `Tokenizer` so the reported token count is exactly the number of input rows the executor saw, and it recomputes `|v|` from the returned dense vector: a norm outside 1 ± 1e-3 counts as a failure and makes the process exit 1.

## Usage

The weights and the tokenizer come from `data/`, restored with `python scripts/unpack_data.py`
(`data/bge-m3/model.bgm`, `data/bge-m3/tokenizer.json`).

```bash
example_embed_text dx data/bge-m3/model.bgm data/bge-m3/tokenizer.json "hello world" --headless
example_embed_text vk data/bge-m3/model.bgm data/bge-m3/tokenizer.json \
    "a photograph of a red apple on a wooden table" "un chat sur un clavier" --headless
example_embed_text cuda data/bge-m3/model.bgm data/bge-m3/tokenizer.json --sparse --colbert \
    "how do I mount a texture?" --headless
printf 'first sentence\nsecond sentence\n' | example_embed_text vk data/bge-m3/model.bgm data/bge-m3/tokenizer.json --headless
```

| Option | Meaning |
|---|---|
| `<backend>` | `dx`, `vk`, `cuda`, or another installed backend plugin name |
| `<model.bgm>` | packed bge-m3 weights (`tools/pack_model.cpp`, target `example_embed_pack_model`) |
| `<tokenizer.json>` | HF fast-tokenizer JSON of bge-m3 (the Unigram model) |
| `sentence...` | sentences to encode; with none, one sentence per line is read from stdin until EOF |
| `--sparse` | also compute and list the sparse lexical terms with weight > 0 |
| `--colbert` | also compute the ColBERT projection and print the row count |
| `--headless`, `--cli` | never opens a window (this tool never does); accepted for the uniform gallery CLI |

## Result

One block per sentence. Example output of `example_embed_text vk data/bge-m3/model.bgm data/bge-m3/tokenizer.json "hello world" --headless`:

```text
[info] embedding Session ready: 24 layers, H=1024, 2 weights-queue workers, tensor-path='legacy', gemm='rb_64x64_b128_r84 (default)'.
[info] sentence 0: "hello world"
[info] tokens : 5 (with <s>/</s>, truncated to 512)
[info] dense  : first 8 of 1024 = [-0.0403959, 0.0370346, -0.0289747, 0.0161167, -0.0356918, -0.040738, -0.0550539, -0.0408672], |v| = 1.000000
[info] example_embed_text: success, 1 sentence(s) encoded on 'vk' (headless=true).
```

Exit code 0 means every sentence produced a unit-norm dense vector; 1 means bad arguments, a missing file, a backend that refused the device, or an empty/non-normalised vector.

## See also

example_embed_retrieval (indexes whole directories with the same session, hybrid dense + sparse scoring), example_embed_dump_golden (dumps the per-layer activations of one sentence).
