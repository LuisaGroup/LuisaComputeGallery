# example_multi_head_attention

Four attention paths — fused MHA, MLA, cooperative-vector MLA and paged MHA — on one fixed problem, each verified against a CPU reference.

| | |
|---|---|
| Group | `ml` |
| Sources | `src/ml/multi_head_attention.cpp`, `src/ml/attention_kernels.cpp`, `src/ml/attention_runner.cpp`, `src/ml/attention_host_data.cpp`, `src/ml/attention_cpu_reference.cpp`, `src/ml/paged_attention.cpp`, `src/ml/attention_config.h` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_multi_head_attention` |
| Run | `example_multi_head_attention dx 1 0 0 --headless` |

## Purpose

A correctness harness for the attention kernels of an inference stack. The problem
is fixed (batch 16, 8 heads, 256 tokens, head dim 64), the host data is
deterministic and every path is compared element-wise with a multi-threaded CPU
reference at `1e-4`. Use it to check a backend or a kernel variant, and to read how
online softmax, MLA and a paged KV cache are written in the DSL.

## Architecture

- **Configuration (`attention_config.h`).** batch 16, 8 heads, `seq_len` 256,
  `head_dim` 64; MLA adds hidden 512, latent 64, decoupled RoPE 16, content 48.
  Derived sizes, the `1 / sqrt(head_dim)` scale, the per-token projection block sizes
  (256 and 128 threads) and the cooperative-vector chunk 16 are constant expressions
  a DSL lambda can use; RoPE frequencies are precomputed once on the host.
- **Host data (`attention_host_data.cpp`).** Deterministic Q/K/V, hidden states and
  the five weight matrices (`W_Q`, `W_DKV`, `W_UK`, `W_UV`, `W_KR`), uploaded once;
  byte-buffer aliases are derived device-side for the cooperative-vector loads.
- **MHA path.** `create_mha_online_attention_kernel()` — one fused online-softmax
  (flash-attention style) kernel with shared-memory K/V tiles, dispatched once with
  `batch * num_heads * seq_len` threads.
- **MLA path.** `create_project_q_kernel<coop>()`,
  `create_project_kv_kernel<coop>()` (KV latent and RoPE key from one hidden read)
  and `create_online_attention_kernel<coop>()`. Matrix absorption means the content
  key is never materialised: the score uses `(W_UK^T q)^T c_t^KV`, and only `c_t^KV`
  plus `k_t^R` are cached per token. `cooperative_vector` picks the instantiation
  whose inner loop uses cooperative-vector loads through the `ByteBuf` arguments; the
  scalar fallback ignores them. The summary prints the saving: 4096 B of KV cache per
  token for MHA, 768 B for MLA (18.75 %).
- **Paged path (`use_paged`, `paged_attention.cpp`).** `PagedAttention` keeps the KV
  cache in sparse-buffer tiles, carves attention pages out of a tile
  (`pages_per_tile = tile_elems / page_elems`), maps logical pages to scattered
  physical pages through a block table and grows the pool by copying the live pages
  into a larger segment. `create_reshape_kv_to_paged_kernel()` scatters K/V into the
  pool, `create_paged_attention_kernel(tokens_per_page)` runs online softmax per
  `(b, h, i)` row through the page index; without sparse buffers the backend falls
  back to dense MHA and logs it.
- **Verification.** The output is downloaded, `run_cpu_reference` recomputes it,
  every element is checked for finiteness and tolerance, and the exit code follows
  `Verification: PASSED` or `FAILED`; a failure prints up to five mismatches.

## Usage

After the backend, three positional switches are read with `atoi`, in this order:

```bash
example_multi_head_attention vk 1 1 0 --headless     # MLA with cooperative vector
example_multi_head_attention cuda 0 0 1 --headless   # paged MHA
```

| Argument | Meaning |
|---|---|
| `use_mla` (2nd) | `1` MLA (default), `0` plain MHA |
| `cooperative_vector` (3rd) | `1` enables the cooperative-vector inner loop; it only affects the MLA path (default `0`) |
| `use_paged` (4th) | `1` runs vLLM-style paged attention on the MHA path and overrides the other two, with a warning (default `0`) |
| `--headless`, `--cli`, `--frames N`, `--output-dir DIR` | the shared convention flags; nothing opens a window, and `--frames`/`--output-dir` are unused in this one-shot validation |

## Result

```text
Mode: MLA
Matrix config: batch=16, heads=8, seq_len=256, head_dim=64
MLA config: hidden_dim=512, latent_dim=64, rope_dim=16, content_dim=48
KV-cache per token: MHA=4096 B, MLA=768 B, ratio=18.75%
Verification: PASSED
Max absolute error: 1.5483238e-08 at index 560310 (GPU=0.0013446449, CPU=0.0013446294)
multi_head_attention: PASSED (headless=true).
```

With `use_paged=1` the mode becomes `Paged-MHA` and the run first reports the
geometry it discovered, e.g. `Paged KV: tile=65536 B (16384 floats), tokens/page=32
(vLLM block_size), pages/tile=1, pages/seq=8, layout [page][h][t][d]`, plus the
logical-to-physical page mapping; the error stays below the tolerance either way.

## See also

`example_tensor` (tile kernels and ML training demos), `example_helloworld` (the
minimal device and dispatch skeleton).
