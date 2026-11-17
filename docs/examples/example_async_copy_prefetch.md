# example_async_copy_prefetch

Two-stage `cp.async` prefetch into shared memory, using the pipeline primitives instead of a barrier per tile.

| | |
|---|---|
| Group | `compute` |
| Sources | `src/compute/async_copy_prefetch.cpp`, `src/common/headless.h` |
| Backends | cuda |
| Build | `xmake build example_async_copy_prefetch` |
| Run | `example_async_copy_prefetch cuda --headless` |

## Purpose

The example is about overlap: while the second tile is still travelling from global
memory to shared memory, the block already computes on the first one. It is the
canonical CUDA LDGSTS pattern (`cp.async` + `commit` + `wait_prior`) written with
Luisa's DSL primitives, and it is the smallest program in the gallery that needs a
raw device address.

## Architecture

- **Data.** `N = 2048` floats, `host_src[i] = i`, uploaded once into
  `Buffer<float> src`; `Buffer<float> dst` receives the result. The kernel doubles
  every element, so a wrong number of in-flight stages shows up as an equality
  failure rather than as a timing difference.
- **Block shape.** `set_block_size(256)` — `block_size = 256`, two tiles of 256
  floats in shared memory: `$shared<float> tile_a{block_size}` and `tile_b`. One
  block therefore covers `2 * block_size = 512` elements.
- **Addressing.** The async copy takes a raw address, not a buffer handle:
  `auto src_base = src_buf.device_address()` and the byte offset
  `cast<luisa::ulong>(base * 4u)`. This is the `buffer_address()` feature, and it
  is the reason the example is CUDA-only: the DirectX and Vulkan backends do not
  implement it.
- **The two stages.** Stage 1 issues
  `async_copy(1u, tile_a[tid], src_base + …, 4u, 1u, 4u, 0u)` (one 4-byte element
  per lane) and `pipeline_commit()`; stage 2 does the same for `tile_b` and commits
  again. `pipeline_wait_prior(1u)` then waits only until *one* stage is still in
  flight — stage 1 is done, stage 2 is not — `sync_block()` publishes
  `tile_a`, and each thread writes `tile_a[tid] * 2.0f` to `dst`.
  `pipeline_wait_prior(0u)` drains stage 2 and the same block processes `tile_b`.
- **Dispatch.** `num_blocks = ceil(N / (2 * block_size)) = 4` and the launch is
  `shader(src, dst).dispatch(num_blocks * block_size)` = 1024 threads, one element
  per tile per thread. The bounds `$if` guards make the last block safe for sizes
  that are not a multiple of the tile.
- **Timing and check.** A `Clock` wraps the dispatch and `synchronize()`, then
  `dst.copy_to(...)` plus a host loop compares each element with `i * 2.0f` within
  `1e-4`. The first mismatch is reported with its index and the run returns 1.

## Usage

```bash
example_async_copy_prefetch cuda
example_async_copy_prefetch cuda --headless
```

| Option | Meaning |
|---|---|
| `<backend>` | must be a backend that implements `buffer_address()` and `cp.async`: CUDA with compute capability 8.0 or newer |
| `--headless`, `--cli` | no window; accepted for the uniform gallery CLI — the example never opens one |
| `--frames N`, `--output-dir DIR` | consumed by the shared parser and unused |

On a device below CC 8.0 the kernel has no pipeline to lower to; run the example on
a recent NVIDIA GPU.

## Result

Example console output on `cuda` (the example raises the log level itself, so the
compilation step is visible too):

```text
[debug] Generated CUDA source in 0.3973 ms.
[info] Async copy two-stage prefetch: PASSED (0.067 ms, headless=true)
```

The millisecond figure is the measured dispatch, so it changes from run to run; the
verdict and the exit code are what a script should check. `LUISA_DUMP_SOURCE=1` writes
the generated CUDA source out, the quickest way to see the `cp.async` instructions the
two stages lowered to.

## See also

`example_software_lbvh` (a build made of many kernels, without raw pointers),
`example_native_shader_minimal` (dispatch a kernel whose source you wrote by hand).
