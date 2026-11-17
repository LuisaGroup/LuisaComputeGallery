# example_cuda_dx_interop

Shares one buffer between a DirectX and a CUDA device, orders the two transfers with a shared timeline semaphore, and reads the value back.

| | |
|---|---|
| Group | `interop` |
| Sources | `src/interop/cuda_dx_interop.cpp`, `src/common/headless.h` |
| Backends | dx, cuda |
| Build | `xmake build example_cuda_dx_interop` |
| Run | `example_cuda_dx_interop dx --headless` |

## Purpose

The smallest correct cross-API transfer: a `uint` written by a `dx` stream is read
by a `cuda` stream with no host round trip in between. What it teaches is the shape
of an interop handshake — export a DX allocation to CUDA, wrap it as a Luisa
`Buffer` on the CUDA device, and order the two streams with one timeline fence
instead of a `synchronize()` per step. Any larger interop case — a denoiser consuming
a `dx` render, a training loop reading D3D12 input — is this same pattern with more
payload.

## Architecture

- **Two devices, one GPU.** `context.create_device("dx")` first, then
  `dx_device.extension<DxCudaInterop>()`. `interop_ext->cuda_device_index()` names
  the CUDA device behind the same physical adapter, and the CUDA device is created
  with `context.create_device("cuda")`; each side gets its own `Stream`.
- **The event.** `interop_ext->create_timeline_event()` returns a
  `DxCudaTimelineEvent`, one fence seen through two APIs: `dx_event` is the D3D12
  timeline fence and the constructor has already opened the matching CUDA external
  semaphore from it (`cuda_event()`). That single object is what makes
  `dx_signal(v)` / `cuda_wait(v)` cheap fence-value operations rather than a
  flush-and-block.
- **The buffer.** `interop_ext->create_buffer<uint>(1)` allocates on the DX side.
  `cuda_buffer(interop_buffer.handle(), &cuda_ptr, &cuda_handle)` exports it twice:
  `cuda_ptr` is the CUDA device pointer, `cuda_handle` the `CUexternalMemory` that
  must be released again. `cuda_device.import_external_buffer<uint>(
  reinterpret_cast<void *>(cuda_ptr), 1)` then wraps the pointer as an ordinary
  Luisa `Buffer<uint>`, so kernels address it like any other buffer.
- **The handshake.** `input = 114514` is copied up and fence value `1` signalled on
  the DX side (`dx_stream << interop_buffer.copy_from(...) <<
  interop_event.dx_signal(1);`), then the CUDA stream waits for that same value
  before reading (`cuda_stream << interop_event.cuda_wait(1) <<
  cuda_buffer.copy_to(...) << synchronize();`). The literal fence value is the only
  thing tying the two orderings together.
- **Teardown.** `unmap(cuda_ptr, cuda_handle)` releases the export, then the event
  is synchronized *itself* (`interop_event.dx_event.synchronize(1)`). A wait on a
  fence value from the consuming stream is not the same as draining the producing
  side, so both happen — that is the order the API expects.
- **Check.** The host compares `output` with `input`, warns
  `dx -> cuda interop mismatch: expected ..., got ...` on a difference, and returns
  1 in that case.

## Usage

```bash
example_cuda_dx_interop dx # no window; the report is the console output
example_cuda_dx_interop dx --headless
```

| Option | Meaning |
|---|---|
| `<backend>` | accepted and ignored: the pair is always a `dx` device plus a `cuda` device |
| `--headless`, `--cli` | no window; accepted for the uniform gallery CLI — the example never opens one |
| `--frames N`, `--output-dir DIR` | consumed by the shared parser and unused: the work is one transfer |

The run needs a machine where both the `dx` and the `cuda` backend load plus the
driver's CUDA interop support; there is no software fallback for the handoff.

## Result

Example console output:

```text
[info] Result: 114514
[info] cuda_dx_interop: PASSED (result=114514, headless=true).
```

`Result` is what the CUDA stream copied out of the buffer the DX stream filled, so
`114514` on both sides is the proof; exit code 0 means the value survived the
export, the import and the fence. A mismatch prints the warning above and exits 1.

## See also

`example_cuda_vk_interop` (the same handshake on Vulkan, with an external
`VkDevice` handed to the CUDA backend), `example_cuda_vk_kernel_launch` (kernels
compiled by CUDA and dispatched inside a Vulkan stream).
