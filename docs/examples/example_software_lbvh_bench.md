# example_software_lbvh_bench

Performance benchmark for the software LBVH: adversarial scenes through the build and the traversal, with a per-stage breakdown and traversal counters.

| | |
|---|---|
| Group | `compute` |
| Sources | `src/compute/lbvh/bench/benchmark_lbvh.cpp`, `src/compute/lbvh/bench/bench_harness.cpp`, `src/compute/lbvh/bench/bench_scenes.cpp`, `src/compute/lbvh/bench/bench_stats.cpp`, `src/compute/lbvh/lbvh_storage.cpp`, `src/compute/lbvh/lbvh_sort.cpp` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_software_lbvh_bench` |
| Run | `example_software_lbvh_bench dx --headless --scene worst --max-seconds 5 --iters 1 --warmup 0` |

## Purpose

The measurement half of the `example_software_lbvh` family. Instead of one small demo
scene it drives a catalogue of scenes that are worst cases of the LBVH algorithm
itself — duplicate Morton codes, a "Morton caterpillar", sliver soup, a teapot in a
stadium — and reports where the time goes. Use it to compare backends and sizes, not
to check correctness. Only release-mode numbers are meaningful: a debug build enables
the runtime's bounds checks and measures those instead.

## Architecture

- **Scenes (`bench_scenes.cpp`).** Each catalogue entry pairs a geometry with the ray
  distribution that makes it expensive, plus a default triangle/instance/ray count:
  `uniform` (the baseline), `coincident` (all N Morton codes identical),
  `grid-duplicates` (1024 cells of exact duplicates),
  `exponential` (spacing decays like 2^-i: the most unbalanced 30-bit radix tree),
  `line`, `sliver-soup`, `bimodal`, `instance-chain` (256 meshes ⇒ 256 trees);
  `--mesh <file.obj>` replaces the catalogue with one real asset.
- **Shared build.** It builds through the same `LbvhStorage`, `BlasBuilder` and
  `TlasBuilder` as the demo, so a measurement is the real code path; `--validate` adds
  the structural self-check and the RTX hit comparison.
- **Harness (`bench_harness.cpp`).** Two hard rules keep the device alive. Memory:
  the exact byte count of one measurement is computed before the first allocation and
  compared with `--budget-gib`, and a scene that does not fit is skipped. Dispatch
  time: no single submission may be predicted to exceed `--dispatch-budget-ms`, so the
  traversal is planned by growing a *strided* slice until the measured time fills the
  budget, and an oversized build is refused with the predicted-safe size.
- **Instrumentation.** `LbvhBuildTimings` splits the build into primitive AABBs,
  Morton codes, the radix sort and the node stage (a second rebuild with fences, so
  the stage sum slightly exceeds the headline total); an instrumented walk reports
  nodes, slab tests, culling ratio, triangle tests and the deepest stack.
- **Reporting (`bench_stats.cpp`).** Per measurement it prints min/median/mean,
  Mprim/s or Mray/s, ns/ray, hits and misses, tree shape (`max_depth`,
  `depth / log2(n)`, descent failures), the memory estimate against the allocation,
  the worst submission made — and a machine-readable `bench_scene` / `bench_build` /
  `bench_tree` / `bench_trace` / `bench_dispatch` / `bench_memory` record.

## Usage

```bash
example_software_lbvh_bench dx --list
example_software_lbvh_bench dx --scene worst --max-seconds 5 --iters 1 --warmup 0
```

| Option | Meaning |
|---|---|
| `<backend>` | `dx`, `vk` or `cuda` |
| `--scene <name\|all\|worst>`, `--list` | which catalogue entries to run (`all` by default), or just print the catalogue |
| `--triangles`, `--instances`, `--rays`, `--seed` | override the scene sizes and the ray seed, so the same scene can be compared across backends |
| `--iters N`, `--warmup N` | timed and untimed iterations (the headline is the min) |
| `--budget-gib <f>`, `--dispatch-budget-ms <f>` | memory and per-submission limits (defaults 5 GiB and 1000 ms) |
| `--max-seconds <f>`, `--max-triangles <n>` | bounds for the sweeps and for a long measurement |
| `--stress-build`, `--stress-traversal` | also sweep the size in log2 steps, fitting a scaling exponent |
| `--validate`, `--repeat-check` | structural and RTX check; bit-identical results across iterations |
| `--compact[=as-built]`, `--release-scratch`, `--headroom <f>` | also measure the compacted copy (optionally retiring the scratch) |
| `--mesh <file.obj>`, `--headless`, `--frames N` | one real asset instead of the catalogue; headless is implied, `--frames` = `--iters` |

## Result

Example console output from the `--scene worst --iters 1 --warmup 0` run on `dx`; the
catalogue has seven scenes, the first one is shown:

```text
[info] build : min 1.898 / median 1.898 / mean 1.898 ms over 1 iteration(s), 138.11 Mprim/s, 524288 nodes
[info] stages: prim 0.235 ms (16.3%), morton 0.136 ms (9.5%), sort 0.191 ms (13.2%), node 0.877 ms (61.0%)
[info] trace : min 234.733 ms, 0.28 Mray/s, 3581.7 ns/ray, 10299 hits / 55237 misses
[info] dispatch: worst build submission 1.898 ms, worst traversal submission 320.375 ms, budget 1000.0 ms
bench_scene scene=coincident step=- prims=262145 triangles=262144 instances=1 blas=1 rays=65536 seed=0
```

`coincident` is pathological on purpose — nearly every ray walks the whole tree — so
0.28 Mray/s against 135 Mray/s on `grid-duplicates` is the expected spread; a script
should parse the `bench_*` lines.

## See also

`example_software_lbvh` (the demo it benchmarks), `example_software_lbvh_sort_bench`
(the sort stage on its own), `example_software_lbvh_test` (correctness).
