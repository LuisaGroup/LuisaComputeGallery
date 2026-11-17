# example_cuda_vk_kernel_launch

Compiles DSL kernels with the CUDA backend and dispatches them inside a Vulkan command buffer through `VK_NV_cuda_kernel_launch`.

| | |
|---|---|
| Group | `interop` |
| Sources | `src/interop/cuda_vk_kernel_launch.cpp`, `src/common/headless.h` |
| Backends | vk, cuda |
| Build | `xmake build example_cuda_vk_kernel_launch` |
| Run | `example_cuda_vk_kernel_launch vk --headless` |

## Purpose

`VK_NV_cuda_kernel_launch` is the tightest of the interop paths: instead of moving data
between two APIs it runs a CUDA kernel *inside* a Vulkan command buffer, so one queue
and one set of barriers cover both. It is the reference for that flow — compile with
CUDA, import the shader into Vulkan, dispatch it like any other command — and a
barrier test, since imported launches and Vulkan kernels interleave on one stream.

## Architecture

- **Pairing, and when to give up.** `device.extension<VkCudaInterop>()` on the
  Vulkan device, `cuda_device_index()` for the CUDA device behind the same adapter,
  then `context.create_device("cuda", &config)` with that `device_index` — the module
  image targets that GPU's compute capability. A missing extension, a false
  `cuda_kernel_launch_supported()`, no matching CUDA device or no `"cuda"` in
  `context.installed_backends()`: each logs a warning naming the missing piece and
  exits 0.
- **Import and launch ABI.** `ext->create_cuda_kernel(cuda_device.compile(kernel))`
  turns a CUDA `Shader` into a RAII `vk_cuda_interop::CudaShader` that remembers the
  compiled block size. The backend then packs the arguments the way the DSL does: one
  by-value `Params` blob with 16-byte-aligned slots (`LCBuffer { ptr, size_bytes }`
  per buffer, raw values for uniforms) plus an `ls_kid` uint4 trailer with the exact
  thread count — hence `set_block_size(256u)`, and `.dispatch(n)` rather than a grid.
- **Buffers and usages.** Everything comes from `ext->create_buffer<float>(n)`, i.e.
  Vulkan-owned memory CUDA can already address. Each argument's usage is baked into
  the typed instance (`shader.kernel<...CudaArg<Buffer<float>, Usage::READ>...>()`),
  so the dispatch site passes bare views and the barrier recorder knows the accesses.
- **Six scenarios, one report.** `run_saxpy` (1024 elements, `y = 2x + y`),
  `run_block_reduce` (4096 elements through a `Shared<float>` tree over 256 threads,
  16 partial sums added up on the host), `run_chained_command_list` (`saxpy` then
  `scale` in one `CommandList`), `run_dsl_interop` (Vulkan kernels around an imported
  launch, expected `2i + 1 + 10`), `run_histogram` (1024 values into 16 bins with
  `bins->atomic(bin).fetch_add(1u)`) and `run_reimport` (one shader imported twice,
  dispatched after the source `Shader` is destroyed). `check_span_near`, `check_near`
  and `check_span_eq` print one `PASS [...]`/`FAIL [...]` line each, and
  `ok = run_x(...) && ok` keeps every later scenario running.

## Usage

```bash
example_cuda_vk_kernel_launch vk # no window; the report is the console output
example_cuda_vk_kernel_launch vk --headless
```

| Option | Meaning |
|---|---|
| `<backend>` | device that owns the stream; defaults to `vk`, the only one providing `VkCudaInterop` |
| `--headless`, `--cli` | no window; accepted for the uniform gallery CLI — the example never opens one |
| `--frames N`, `--output-dir DIR` | consumed by the shared parser and unused: the work is six fixed-size scenarios |

On any other backend name the device loads, the extension lookup fails, and the run
ends with `VkCudaInterop extension is not available; skipping example.` and exit code
0. Only a `vk` run proves anything, and it needs the CUDA plugin and a device with
`VK_NV_cuda_kernel_launch`.

## Result

Example console output on `vk`:

```text
[info] PASS [saxpy]: 1024 elements verified (eps=1e-05), sample [0]=1 [last]=2047
[info] PASS [block_reduce]: got 48945.75 want 48945.75 (eps=4.895575)
[info] PASS [chained_command_list]: 1024 elements verified (eps=1e-05), sample [0]=3 [last]=6141
[info] PASS [dsl_interop]: 1024 elements verified (eps=1e-05), sample [0]=11 [last]=2057
[info] PASS [histogram]: 16 bins verified exactly
[info] PASS [reimport_a]: 512 elements verified (eps=0), sample [0]=42.5 [last]=42.5
[info] PASS [reimport_b]: 512 elements verified (eps=0), sample [0]=43.5 [last]=43.5
[info] cuda_vk_kernel_launch: PASSED (headless=true).
```

`[last]=2047` is `2 * 1023 + 1` and `[last]=6141` is `3 * (2 * 1023 + 1)`; the two
`reimport` lines differ by exactly the `+1` the second import was given. A failure
shows up as a `FAIL [...]` line and exit code 1.

## See also

`example_cuda_vk_interop` (synchronization between the two APIs rather than kernel
launch), `example_cuda_dx_interop` (the DX flavour of the handshake).
