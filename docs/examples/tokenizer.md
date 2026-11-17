# tokenizer

N-gram speculative drafting on the device: a corpus is tokenized on the host, uploaded as a flat token-ID library, and a DSL kernel finds the earliest match of the query's trailing n-gram and proposes the k tokens that follow it.

| | |
|---|---|
| Group | `compute` |
| Sources | `src/compute/tokenize/main.cpp`, `src/compute/tokenize/ngram_library.cpp`, `src/compute/tokenize/ngram_retriever.cpp`, `src/compute/tokenize/ngram_kernels.cpp`, `src/compute/tokenize/ngram_trainer.cpp`, `src/compute/tokenize/ngram_tokenizer.cpp` |
| Backends | dx, vk, cuda |
| Build | `xmake build tokenizer` |
| Run | `tokenizer dx --headless` |

## Purpose

A port of vLLM's n-gram "prompt lookup" proposer for speculative decoding: given a corpus
and a query, find where the query's last 2–4 tokens already occurred and copy the tokens
that followed. Tokenization, vocabulary and index building are host work; all matching,
ranking and drafting is one kernel, kept honest against a CPU oracle.

## Architecture

- **Tokenizer (`ngram_tokenizer.cpp`).** `NgramTokenizer` normalizes and splits UTF-8 text:
  one token per ASCII alphanumeric run, per CJK codepoint, per other multibyte codepoint;
  whitespace and punctuation separate; `decode_utf8()` maps bad bytes to U+FFFD.
- **Library (`ngram_library.cpp`).** `NgramLibrary::add_document()` tokenizes a document and
  appends dense IDs (first-seen order) to one flat `tokens` vector; `finalize()` seals
  `doc_offsets` / `doc_lengths`. `tokens_buf`, `offsets_buf` and `lengths_buf` are the device
  copies. `detail::fnv1a64_tokens()` is the single hash used by the host index builders *and*
  by the kernels, so keys are comparable across the boundary.
- **Kernels (`ngram_kernels.h/.cpp`).** One shared signature (corpus, queries and
  `query_stride`, query lengths, drafts and draft lengths, `min_n`/`max_n`/`k`, request
  offsets) and four variants: K1 `make_retrieve_naive_kernel()` — one thread per query scans
  the corpus for every length, slow on purpose and kept as the device ground truth;
  K2 `make_retrieve_parallel_kernel(block_size)` — one block per query, suffix staged in
  shared memory (`ngram_max_suffix_tokens = 64`), strided scan reduced with `fetch_min`;
  K3 `make_retrieve_hash_kernel(cap_log2)` — one thread per query, hash and open-addressing
  lookup in the host-built index (`build_ngram_hash_index`) with on-device content
  verification, so `O(max_n)` instead of a scan; K6 `parallel_mle` — K2 plus the trained
  count table, proposing the most frequent continuation.
- **Retriever (`ngram_retriever.cpp`).** `NgramRetriever` owns the buffers and the compiled
  variant: upload queries, dispatch (one row per query, or `rows * block_size`), download
  drafts. `retrieve()` splits into `upload_queries()` / `dispatch_queries()` /
  `download_results()` for benchmarking; `dispatch_requests_multi()` launches several
  requests' grids from **one** command, each resolving rows as `req_off_buf[kernel_id()]`.
- **Training and modes (`ngram_trainer.cpp`, `ngram_train_kernels.cpp`, `main.cpp`).**
  "Training = counting": a unigram histogram, n-gram counts over the retrievable window set,
  then Add-k or backoff scoring and perplexity in LM mode, with `reference.h` /
  `reference_model.h` as the host oracles. `main.cpp` dispatches `demo` (default), `--train`,
  `--test` and `--benchmark`.

## Usage

```bash
tokenizer dx --headless
tokenizer dx --train --headless
tokenizer dx --benchmark --kernel hash --lib-size 4194304 --queries 1024
```

| Option | Meaning |
|---|---|
| `<backend>` | `dx`, `vk` or `cuda` |
| `--train`, `--test`, `--benchmark [opts]` | training demo; the Boost.UT suite; the benchmark of one strategy |
| `--kernel naive\|parallel\|hash\|parallel_mle` | variant to benchmark (default `parallel`) |
| `--lib-size N`, `--queries N`, `--reps N` | corpus tokens (4194304), query batch (1024) and timed repetitions (20) |
| `--min-n N`, `--max-n N`, `--k N` | retrieval parameters (defaults 2, 4, 5) |
| `--block-size N`, `--vocab N`, `--seed N` | threads per query; corpus generator knobs |
| `--bench-train`, `--requests N`, `--req-queries N`, `--multi-mode seq\|pipeline\|fiber\|multi\|all` | time the training stages; multi-request concurrency |
| `--headless`, `--cli` | accepted before or after the mode; the example never opens a window |

## Result

Example console output on `dx`:

```text
[info] library: 31 tokens, 3 docs, vocab 17
[info] [naive] query_len=6 draft_len=4 draft=[0 1 2 3 ] ok (expected [0 1 2 3 ])
[info] hash index: 128 slots (2^7), built in 0.00 ms
[info] [hash ] query_len=6 draft_len=4 draft=[0 1 2 3 ] ok (expected [0 1 2 3 ])
[info] demo: all variants match the reference oracle
[info] tokenize: demo mode, PASSED (headless=true).
```

Draft IDs are vocabulary indices in first-seen order, so `[0 1 2 3]` is the corpus' first
four distinct words. `--train` ends with oracle checks instead, e.g. `P(爱|我) [Add-k
k=0.1] = 0.700000 (oracle 0.700000) ok`.

## See also

`example_multi_head_attention` (attention kernels plus a CPU reference),
`example_gdeflate` (another `Buffer` tool with host-side oracles).
