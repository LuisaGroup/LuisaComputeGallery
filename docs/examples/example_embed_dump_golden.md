# example_embed_dump_golden

Dumps the per-layer activations of one encoded sentence to raw `fp32` files, optionally comparing them against pinned golden layer states.

| | |
|---|---|
| Group | `embedding` |
| Sources | `src/embedding/tools/dump_golden.cpp` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_embed_dump_golden` |
| Run | `example_embed_dump_golden dx data/bge-m3/model.bgm data/bge-m3/tokenizer.json "hello" --layers 0,1 --headless` |

## Purpose

The bisect tool for the encoder: when a sentence embeds "about right" but not exactly right, this answers *which* of the 24 layers diverged and by how much. It exercises the Session layer-trace hook, host-thread readback of the traced buffers and a small JSON manifest, and it is the only example that writes the engine's intermediate tensors to disk.

## Architecture

- Trace wiring. The tool registers a `TraceFn` callback on the session (`set_layer_trace`) through the first passthrough the `Session` API exposes; each callback receives the layer index, an immutable `BufferView<float>` and `(rows, cols)`. Reads go through a tool-owned read stream (`view.copy_to(host_span) << synchronize()`), never through a worker's stream.
- Layer indexing. `0` is the embedding output, `1..24` are the post-LN2 states of encoder layers 0..23 — exactly the indexing of the pinned `golden_layers.bin`.
- Sentence resolution. A pure-digit argument is a line index into `corpus.txt`, which is read next to the tokenizer JSON (lines unescape `\\`, `\n`, `\r`); anything else is used as the literal UTF-8 sentence.
- Output. Into the current directory: `dump_layer_NN.bin` per selected layer (`rows * cols` little-endian floats), `dump_dense.bin` (the L2-normalised dense vector) and `dump_meta.json`, which records the backend, package, sentence, token rows and one entry per layer with `rows`, `cols`, `floats` and file name.
- Comparison. With `--compare <golden-dir>` the tool reads `golden_meta.json` and `golden_layers.bin` from that directory, matches states by layer number, and prints per-layer `max_abs_diff` with the offending element index plus one overall figure; a layer without a golden state, or with a different column count, is reported and skipped instead of silently passing. Only sentence index `0` or `3` is accepted here, because those are the states a pinned baseline holds.

## Usage

The weights and tokenizer come from `data/`, restored with `python scripts/unpack_data.py`
(`data/bge-m3/model.bgm`, `data/bge-m3/tokenizer.json`).

```bash
example_embed_dump_golden dx data/bge-m3/model.bgm data/bge-m3/tokenizer.json "hello" --layers 0,1 --headless
example_embed_dump_golden vk data/bge-m3/model.bgm data/bge-m3/tokenizer.json "hello" --headless
example_embed_dump_golden cuda data/bge-m3/model.bgm data/bge-m3/tokenizer.json "a cat" --layers=0,12,24 --headless
# --compare resolves its sentence through corpus.txt next to the tokenizer
example_embed_dump_golden dx data/bge-m3/model.bgm data/bge-m3/tokenizer.json 0 --compare golden/ --headless
```

| Option | Meaning |
|---|---|
| `<backend>` | `dx`, `vk`, `cuda`, or another installed backend plugin name |
| `<model.bgm>`, `<tokenizer.json>` | packed bge-m3 weights and the matching tokenizer |
| `<sentence_index\|text>` | digits index `corpus.txt` next to the tokenizer; anything else is the literal sentence |
| `--layers` | dump all 25 traced states (no value) or only the listed layer numbers |
| `--layers=a,b,...` | same list in `=` form |
| `--compare <golden-dir>` | compare against `golden_meta.json` + `golden_layers.bin` in that directory; sentence index must be 0 or 3 |
| `--headless`, `--cli` | never opens a window (this tool never does); accepted for the uniform gallery CLI |

## Result

Two traced layers, a dense vector and a manifest, then the file name is echoed back:

```text
[info] dump_golden: wrote 2 layer dump(s) + dump_dense.bin + dump_meta.json for "hello"
[info] dump_golden: success (2 layer dump(s), headless=true).
```

`dump_meta.json` is what the comparison and any external checker read (abridged):

```json
{
  "tool": "dump_golden", "backend": "vk", "sentence_index": null,
  "text": "hello", "token_rows": 4, "dense_floats": 1024,
  "layers": [
    {"layer": 0, "rows": 4, "cols": 1024, "floats": 4096, "file": "dump_layer_00.bin"}
  ]
}
```

With `--compare` each compared state adds one line of the form
`layer  1: rows=4 cols=1024 compared=4096 max_abs_diff=... (at k, got ... vs golden ...)`, followed by
`dump_golden: N layer(s) compared, overall max_abs_diff = ...` and a final `compare PASSED` (`FAILED` only when the golden files cannot be read).

## See also

example_embed_text (the same encode without the trace hook), example_embed_retrieval (hybrid search built on the same session).
