# example_embed_retrieval

Whole-file hybrid retrieval over a directory: every document and every query is encoded by the bge-m3 session, and the index fuses the dense cosine with the sparse lexical cosine (0.7 / 0.3).

| | |
|---|---|
| Group | `embedding` |
| Sources | `src/embedding/example_embed_retrieval.cpp`, `src/embedding/embed/retrieval_index.h`, `src/embedding/embed/retrieval_service.h`, `src/embedding/embed/retrieval_rpc.cpp`, `src/embedding/embed/retrieval_disk_cache.h`, `src/embedding/api/example_embed_retrieval_lib.h` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_embed_retrieval` |
| Run | `example_embed_retrieval --model data/bge-m3/model.bgm --tokenizer data/bge-m3/tokenizer.json --backend dx` |

## Purpose

Shows what a Session is for beyond one sentence: a directory is walked, each regular file is encoded once, and queries are ranked in memory. The same steps are reachable three ways — an interactive console, a newline-delimited JSON-RPC server, and a C ABI shared library — which makes the example a complete tour of the retrieval engine (`embed::RetrievalService`, `RetrievalIndex`, `DiskEmbedCache`).

## Architecture

- Startup. `Session::create` loads the package and tokenizer, then one dense warm-up encode drains the execution path so the first real query does not pay for shader compilation. `--backend` selects the plugin (default `vk`); a failed device creation prints the installed backend names.
- Encoding. The service is built from a hybrid encode callback: `EncodeOptions{max_seq_len = 512, want_sparse = true}`, so each text yields the dense CLS vector *and* the sparse lexical weights from the same forward pass. Sparse pairs are canonicalised (duplicate ids summed, zeros dropped, sorted by id, L2-normalised) before scoring.
- Index. `RetrievalIndex` keeps one document and one hybrid embedding per file and scores `0.7 * dense_cosine + 0.3 * sparse_cosine`, sorting by descending score and then by canonical path. `load` recursively admits files matching a comma-separated extension rule (default `.md,.txt`), rejects duplicate or nested roots, and skips non-matching files; `update` re-encodes only changed content and is transactional per file; `unload` removes exactly one root.
- Cache. `DiskEmbedCache` writes `.lc_embed_cache.json` plus `.lc_embed_cache/<md5>.bin` (dense) and `<md5>.sparse.bin` (sparse pairs) next to an indexed directory, so a second `load` reports `cached=<n>` instead of re-encoding. The index itself only lives in process memory.
- Front ends. `run_repl` loops on a small quoted-argument grammar; `--server` instead starts `RetrievalRpcServer` with `load`, `unload`, `update` and `query` methods, one JSON object per line, loopback-only with no authentication. The C ABI in `src/embedding/api/example_embed_retrieval_lib.h` exports the same steps (`eer_context_create`, `eer_load_dir`, `eer_query`, ...) for any host language.

## Usage

The weights and tokenizer come from `data/`, restored with `python scripts/unpack_data.py`
(`data/bge-m3/model.bgm`, `data/bge-m3/tokenizer.json`).

```bash
example_embed_retrieval --model data/bge-m3/model.bgm --tokenizer data/bge-m3/tokenizer.json --backend dx
example_embed_retrieval --model data/bge-m3/model.bgm --tokenizer data/bge-m3/tokenizer.json --backend vk --server
echo "" | example_embed_retrieval --model data/bge-m3/model.bgm --tokenizer data/bge-m3/tokenizer.json --backend cuda --headless
printf 'load docs\nquery --top 3 how do I run headless\nquit\n' | example_embed_retrieval --model data/bge-m3/model.bgm --tokenizer data/bge-m3/tokenizer.json --backend dx --headless
```

| Option | Meaning |
|---|---|
| `--model <model.bgm>` | packed bge-m3 weights, required |
| `--tokenizer <tokenizer.json>` | tokenizer JSON, required |
| `--backend <name>` | backend plugin, default `vk` |
| `--server` | serve JSON-RPC over TCP instead of the console (the two loops never run together) |
| `--listen <address>`, `--port <port>` | listener, default `127.0.0.1:8765`; port `0` asks for an ephemeral port and logs it |
| `--headless`, `--cli` | never opens a window (this tool never does); with stdin at EOF the console indexes nothing and exits 0 |
| `--help` | process command line |

Console commands: `load <dir> [.ext,.ext]`, `unload <loaded-root>`, `update <file-or-subdir>`, `query [--top N] <text>` (default 5, 1–100), `help`, `quit`. Quote arguments that contain spaces; a backslash escapes a quote only.

## Result

A console banner followed by a prompt. Example session with two text files, abridged:

```text
Embedding retrieval console; type help for commands.
embed> loaded docs extensions=.md,.txt added=2 cached=0 skipped=0 failed=0 total_documents=2
embed> 1 0.590441 docs/raytracing.txt
2 0.349284 docs/ntc.txt
```

Scores are at most 1.0. A query on an empty corpus is a recoverable error, and `--server` answers with JSON-RPC errors (`-32601` unknown method, `-32602` invalid params, `-32000` operational failure) instead of closing the connection.

## See also

example_embed_text (single sentences, no index), example_embed_dump_golden (layer-level parity of the same encoder).
