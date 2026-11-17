# example_ocr_bench

Correctness-gated benchmark harness for the OCR engine: runs a suite of synthetic workloads on the device and reports latency statistics, counters and correctness flags as JSON.

| | |
|---|---|
| Group | `ocr` |
| Sources | `src/ocr/example_ocr_bench.cpp`, `src/ocr/ocr/benchmark.cpp`, `src/ocr/ocr/benchmark.h`, `src/ocr/bench/` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_ocr_bench` |
| Run | `example_ocr_bench dx ops --headless` |

## Purpose

Measures the engine instead of judging it by eye: every row is a declared engine graph with a fixed seed, an input fingerprint, a counter set and a `correct` flag, so a change is compared against a stored baseline rather than against last week's impression. The synthetic suites need no weights at all, which makes the tool usable as a regression gate on any machine with a GPU.

## Architecture

- Suites. `<suite>` selects the workload family: `ops` (atomic operators such as the ragged `ops.gemm.ragged_17x19x23` GEMM), `decode` (autoregressive decode steps), `cnn` (detector/recogniser convolutions), `layout` (document layout and region scheduling), `gdn` (hybrid Gated DeltaNet cache path), `moe` (router plus expert loop), `model` (composed slices) and `all`. Rows can be narrowed with `--target`, `--case`, `--variant` or one `--workload` id.
- Measurement. `bench::run_benchmark` creates the device, records environment metadata (backend, OS, compiler, warp size, device-creation time), then times each workload after `--warmup` iterations and `--iterations` measured ones, with `--dispatch-batch` dispatches per iteration. Optional toggles (`--graph`, `--fusion`, `--cooperative-norm`, `--cache`, `--instrument`) select the code path being measured.
- Report. `serialize_report` writes one deterministic JSON object: `schema_version` (`luisa.ocr.benchmark.v1`), `environment`, the full `options` block (which doubles as provenance, including any `--source-revision`/`--gpu-name`/`--provenance k=v` overrides), and `rows`. Each row carries `workload` (id, suite, target, case, variant, seed, fingerprint, shape, counts), `status`, `reason`, `correct`, `fallback_used`, `truncated`, the raw `samples`, `statistics` (min/median/mean/p95/max/stddev/MAD/CI95) and derived `counters` such as `gflops_per_second`.
- Artifact classes. Rows are `synthetic` by default. `--artifact-class real` additionally needs `--package` and `--manifest`, and is refused as `unvalidated` until a recipe leaves the planned state — so a real-artifact run reports a single `refused` row instead of numbers, and exits 2.

## Usage

There is nothing to unpack for this tool — no `python scripts/unpack_data.py` step — because the synthetic suites generate their own inputs; the real-artifact path needs a pinned `.lcocr` package, which the repository does not ship.

```bash
example_ocr_bench dx ops --headless
example_ocr_bench vk all --iterations 20 --warmup 5 --headless
example_ocr_bench cuda decode --target dense_synthetic --json report_decode.json --headless
example_ocr_bench dx ops --iterations 20 --csv series.csv --no-pretty --headless
```

| Option | Meaning |
|---|---|
| `<backend>` | backend plugin: `dx`, `vk`, `cuda`, `fallback`, ... |
| `<suite>` | `ops`, `decode`, `cnn`, `layout`, `gdn`, `moe`, `model` or `all` |
| `--help` | usage text, no backend or suite needed, exit 0 |
| `--target <name>`, `--case <name>`, `--variant <name>`, `--workload <id>` | selectors; empty means no filter, duplicates are rejected |
| `--warmup <n>`, `--iterations <n>`, `--dispatch-batch <n>` | measurement shape (defaults 3, 20 and 1) |
| `--graph on\|off`, `--fusion on\|off`, `--cooperative-norm on\|off`, `--instrument on\|off` | code-path toggles (defaults off, off, on, off) |
| `--cache miss\|hit\|disabled` | cache mode to measure (default `miss`) |
| `--artifact-class synthetic\|real` | input class; `real` needs `--package` and `--manifest` |
| `--json <path>`, `--no-pretty` | report file (default stdout) and minified output |
| `--csv <path>`, `--baseline <path>` | append a CSV projection; compare medians against a previous report |
| `--source-revision`, `--source-tree-state`, `--gpu-name`, `--driver-version`, `--provenance k=v` | provenance overrides captured into `environment` |
| `--headless` | same work, no window; this benchmark never opens one |

Note the argument style: long options take their value as the next argument (`--iterations 20`), not with `=`.

## Result

The report on stdout (or `--json`), prefixed by the usual device log. Example for `dx ops`, abridged:

```json
{
  "schema_version": "luisa.ocr.benchmark.v1",
  "environment": {"backend": "dx", "os": "windows", "warp_size": 32, ...},
  "options": {"suite": "ops", "warmup": 3, "iterations": 20, "dispatch_batch": 1, ...},
  "rows": [
    {"workload": {"id": "ops.gemm.ragged_17x19x23", "suite": "ops", "target": "gemm",
                  "case": "ragged_17x19x23", "variant": "reference", "seed": 24301},
     "status": "ok", "correct": true, "fallback_used": false, "truncated": false,
     "statistics": {"record_ms": {"n": 20, "median": 0.0004, ...},
                    "submit_sync_ms": {"n": 20, "median": 0.0791, ...}},
     "counters": {"gflops_per_second": "0.2279", "effective_gb_per_second": "0.9313"}}
  ]
}
```

Exit code 0 means every row finished `ok` and correct; 2 means a row failed, was invalid, or was refused; 3 means the report could not be written.

## See also

example_ocr (the recognition front end and its support report), example_software_lbvh_bench (another correctness-gated benchmark harness in the gallery).
