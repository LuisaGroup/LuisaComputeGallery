# example_software_lbvh_test

The correctness suite of the software LBVH: boundary sizes, degenerate geometry, determinism and compaction, each compared with a host reference and with the RTX structure.

| | |
|---|---|
| Group | `compute` |
| Sources | `src/compute/lbvh/test/lbvh_test.cpp`, `src/compute/lbvh/software_lbvh.cpp`, `src/compute/lbvh/blas.cpp`, `src/compute/lbvh/tlas.cpp`, `src/compute/lbvh/lbvh_storage.cpp`, `src/compute/lbvh/lbvh_sort.cpp` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_software_lbvh_test` |
| Run | `example_software_lbvh_test dx --headless` |

## Purpose

`example_software_lbvh` proves one small scene; this target proves the implementation.
It builds and traces a catalogue of hand-picked awkward cases — one triangle, powers of
two and their neighbours, coincident centroids, zero-area triangles, huge extents,
multi-tree scenes, strided slices — and checks each run from three sides at once: the
tree must be structurally valid, the device hits must match a host reference, and (where
the backend has ray tracing) they must match the Luisa `Accel`. It exits non-zero on
the first failure, so it can gate a change to the builders.

## Architecture

- **The catalogue.** Each check runs one scene and prints a `[ok]`/`[fail]` line with its
  counters. Families: `boundary` (`n = 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 255,
  256, 257, 4097` triangles — the sizes where a radix tree or a warp-wide reduction
  changes shape), `degenerate` (identical triangles, one shared centroid, zero area,
  slivers, a far outlier, axis-aligned quads, huge extents), `determinism` (rebuild and
  re-trace), and multi-tree and strided-slice variants of the same scenes.
- **Per-check invariants.** The line prints `problems`, `mismatches`, `rtx`, `ties`,
  `grazing`, `boundary`, `conditioned` and `guard` counts. A `problems` value above zero
  means `validate_tree`, the node/primitive range bookkeeping, the bindless heap check,
  the miss contract or the build/trace repeat check failed; `mismatches` counts hits
  that disagree with the host reference. Grazing, boundary and ill-conditioned cases are
  counted separately and excluded from the strict comparison because the two traversals
  may legitimately disagree there — the summary prints how many were skipped and why.
- **Reuses the production code.** Every check builds through `LbvhStorage`,
  `BlasBuilder`, `TlasBuilder` and the compiled `make_trace_kernel()`, so it tests the
  code the demo and the benchmarks run, not a copy.
- **Compaction checks.** Several checks build, compact, and then compare the compacted
  structure with the loose one: node count and bytes reclaimed, the hits before and
  after, and the storage after the scratch is retired.
- **Reporting.** Per scene `[ok] … <ms>`, then three summary lines: number of checks,
  problems per class, and software-vs-reference / RTX-vs-reference mismatch totals.

## Usage

```bash
example_software_lbvh_test dx --headless
example_software_lbvh_test vk --quick --headless
example_software_lbvh_test cuda --verbose --seed 7 --headless
```

| Option | Meaning |
|---|---|
| `<backend>` | `dx`, `vk` or `cuda`; the RTX cross-checks need hardware ray tracing |
| `--quick` | run a reduced subset (50 checks) instead of the full catalogue |
| `--verbose` | print the details of every check, not only the verdict |
| `--seed <n>` | seed of the generated scenes and rays, so a failure can be replayed |
| `--headless`, `--cli` | no window; accepted for the uniform gallery CLI |

## Result

Example console output on `dx` with `--quick`:

```text
lbvh_test: backend=dx seed=0 quick=1
[ok] boundary n=1 problems=0 mismatches=0 rtx=0 ties=0 grazing=0 boundary=0 conditioned=0 guard=0 84ms
[ok] degenerate identical problems=0 mismatches=0 rtx=0 ties=0 grazing=0 boundary=0 conditioned=0 guard=0
summary: 50 check(s), 50 passed, 0 failed in 0.9 s (backend dx, seed 0, quick)
summary: problems: tree 0, ranges 0, heap 0, hit contract 0, slices 0, repeats 0, compaction 0, empty reference 0
summary: software-vs-reference mismatches 0, RTX-vs-reference mismatches 0 (42 RTX cross-checks)
```

The `--verbose` run prints, for a mismatch, the full ray, the expected and the
actual hit, and the triangle's determinant diagnostics — enough to replay the case.

## See also

`example_software_lbvh` (the demo these checks protect),
`example_software_lbvh_bench` and `example_software_lbvh_sort_bench` (the performance
side of the same code).
