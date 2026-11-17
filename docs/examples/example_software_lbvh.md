# example_software_lbvh

A two-level BVH built entirely by hand in Luisa kernels — no hardware RTX API — validated ray by ray against Luisa's own RTX traversal.

| | |
|---|---|
| Group | `compute` |
| Sources | `src/compute/software_lbvh.cpp`, `src/compute/lbvh/software_lbvh.cpp`, `src/compute/lbvh/blas.cpp`, `src/compute/lbvh/tlas.cpp`, `src/compute/lbvh/lbvh_storage.cpp`, `src/compute/lbvh/lbvh_sort.cpp`, `src/compute/lbvh/scene.h` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_software_lbvh` |
| Run | `example_software_lbvh dx --headless` |

## Purpose

This is the gallery's "bring your own acceleration structure" example: the BVH is
built by Luisa kernels in plain `Buffer`s and staged exactly like the hardware
backends stage theirs, so it can later serve devices without ray tracing. The run
also proves it right: the same rays go through a Luisa `Accel`, hit by hit.

## Architecture

The data flow is the backend order `estimate → create → pre_build → build → trace`.

- **Scene (host, `scene.h`).** `make_uv_sphere`, `make_box` and `make_torus` produce
  `HostMesh` soups, flattened into one `Buffer<float3>`, one `Buffer<Triangle>` and a
  `MeshRange` per mesh — so a BLAS is just a triangle range.
- **Two levels.** A `Blas` (`blas.cpp`) is an LBVH over one mesh's triangles; a
  `Tlas` (`tlas.cpp`) is an LBVH over instances, each naming a `Blas` plus a
  transform. `SoftwareLbvh` (`lbvh/software_lbvh.cpp`) owns the storage, both
  builders and the compiled traversal kernel.
- **Shared stages (`lbvh_storage.cpp`, `lbvh_sort.cpp`).** `LbvhStorage` keeps one
  primitive and one node buffer for the whole scene, each tree owning a contiguous
  `TreeRange`. `build_tree()` runs primitive AABBs, a 30-bit Morton code per
  primitive, a stable 4 × 8-bit LSD radix sort of the `(code, slot)` pairs, then the
  Karras radix tree (`delta` / `determineRange` / `findSplit`). `lbvh_sort.h` picks
  `Method::single_block` below 8192 keys and `Method::multi_block` above.
- **The bindless heap.** A `Tlas` owns a `BindlessArray`: slot 1 is its own node
  region, slot `2 + i` the region of BLAS `i`. `pre_build_accel()` registers one
  `BufferView<LbvhNode>` per tree and a handle resolves as
  `heap[slot][handle - node_offset]`, so one node buffer serves a two-level walk.
- **Traversal (`software_lbvh.cpp`).** `make_trace_kernel()` is a `Kernel1D` with
  `set_block_size(64)`, one thread per ray; thread `i` handles ray
  `ray_offset + i * ray_stride`, so the demo can submit strided slices instead of a
  contiguous prefix. `tlas_traversal()` walks the top level through slot 1, descends
  into the instance's BLAS and writes an `LbvhHit`.
- **Checks.** `make_heap_check_kernel()` re-reads every BLAS root through its slot
  and through the node buffer; the host adds a structural self-check, `compact()`, a
  second self-check and an RTX comparison over 262144 rays.

Four drivers share that implementation:
`software_lbvh.cpp` is the demo above; `lbvh/bench/benchmark_lbvh.cpp` measures
throughput under adversarial scenes; `lbvh/bench/sort_bench.cpp` times the radix
sort alone; `lbvh/test/lbvh_test.cpp` is the correctness suite. The benchmarks share
`bench_harness.cpp` (budgets, timing, `bench_*` records) and `bench_stats.cpp`, and
`bench_scenes.cpp` generates the catalogue.

## Usage

```bash
example_software_lbvh dx                      # window (needs the gui module)
example_software_lbvh dx --headless --no-compact --headroom 4
```

| Option | Meaning |
|---|---|
| `<backend>` | `dx`, `vk` or `cuda`; the RTX comparison needs a backend with hardware ray tracing |
| `--headless`, `--cli` | no window: build, trace, check, print, exit |
| `--compact` (default), `--no-compact` | compact the loose structure after the build (and re-check the hits), or keep it for an A/B |
| `--headroom <f>` | size the storage as `scene * f` (default `2`); `1` is exact, so compaction reclaims nothing |
| `--release-scratch` (default), `--keep-scratch` | retire the build scratch with the compacted copy, or keep it |

## Result

Example console output on `dx`:

```text
[info] software LBVH built in 20.041 ms: 3 BLAS (2267 nodes), TLAS with 9 instances (17 nodes)
[info] structural self-check: 3 BLAS + 1 TLAS, bindless heap (5 slot(s)), 0 problem(s)
[info] compaction: 4 tree(s), 4576 -> 2284 node(s), 73344 B reclaimed, 50.1%
[info] mismatch: hit/miss 0, instance 0, primitive 0, distance 0 (tol 0.001), barycentric 0 (tol 0.005)
[info] software LBVH traversal matches the Luisa RTX reference (headless=true).
```

The zero-mismatch line is the point: all 262144 rays hit the same instance and
primitive as the hardware structure.

## See also

`example_software_lbvh_bench` (throughput), `example_software_lbvh_sort_bench` (the
sort alone) and `example_software_lbvh_test` (correctness) drive the same BVH.
