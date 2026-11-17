# example_cuda_vk_interop

Shares one buffer between a Vulkan and a CUDA device, with the CUDA backend created on the very `VkDevice` the Vulkan device owns.

| | |
|---|---|
| Group | `interop` |
| Sources | `src/interop/cuda_vk_interop.cpp`, `src/common/headless.h` |
| Backends | vk, cuda |
| Build | `xmake build example_cuda_vk_interop` |
| Run | `example_cuda_vk_interop vk --headless` |

## Purpose

The Vulkan counterpart of the DX interop example, and the one that shows the extra
step Vulkan needs: the CUDA backend must be told about the existing `VkDevice`, not
merely about a matching GPU. With that in place a `uint` written through a `vk`
stream is read by a `cuda` stream, ordered by a single event, and the value is
checked on the host.

## Architecture

- **Vulkan device first.** `context.create_device("vk")`, then
  `vk_device.extension<VkCudaInterop>()`. Unlike the DX case, the driver-side objects
  have to travel to the other backend: the CUDA context must be created on the same
  Vulkan device that owns the allocation.
- **The config extension.** `CUDADeviceConfigExtImpl` is a small subclass of
  `CUDADeviceConfigExt` that carries a `CUDADeviceConfigExt::ExternalVkDevice`
  (`VkPhysicalDevice` + `VkDevice`) and returns it from
  `get_external_vk_device()`. The example fills it with
  `interop_ext->get_external_vk_device()` — the handles the Vulkan backend already
  created — and passes it down in `DeviceConfig`:
  `DeviceConfig{.extension = std::move(ext_device), .device_index =
  interop_ext->cuda_device_index()}`. Both fields matter: the extension supplies
  the shared device, `device_index` the matching physical GPU.
- **The event.** This example uses an ordinary `cuda_device.create_event()`
  rather than a DX-style timeline event, and orders the two APIs through the
  extension: `interop_ext->vk_signal(interop_event)` on the Vulkan stream,
  `interop_event.wait()` on the CUDA stream. `vk_signal`/`vk_wait` translate a
  Luisa `Event` into the handle and fence value the other side understands.
- **The buffer.** `interop_ext->create_buffer<uint>(1)` allocates on the Vulkan
  side; `interop_ext->cuda_buffer(interop_buffer.handle(), &cuda_ptr,
  &cuda_handle)` returns the CUDA device pointer and the external-memory handle;
  `cuda_device.import_external_buffer<uint>(reinterpret_cast<void *>(cuda_ptr), 1)`
  wraps it as a `Buffer<uint>` usable from kernels.
- **The handshake.** `vk_stream << interop_buffer.copy_from(luisa::span{&input, 1})
  << interop_ext->vk_signal(interop_event);` followed by
  `cuda_stream << interop_event.wait() << cuda_buffer.copy_to(luisa::span{&output, 1})
  << synchronize();` — one signal, one wait, no host barrier in the middle.
- **Teardown and check.** `unmap(cuda_ptr, cuda_handle)` releases the export, then
  the event is synchronized on its own (`interop_event.synchronize()`); the host
  compares `output` with `input` (114514), warns
  `vk -> cuda interop mismatch: expected ..., got ...`, and returns 1 on mismatch.

## Usage

```bash
example_cuda_vk_interop vk # no window; the report is the console output
example_cuda_vk_interop vk --headless
```

| Option | Meaning |
|---|---|
| `<backend>` | accepted and ignored: the pair is always a `vk` device plus a `cuda` device |
| `--headless`, `--cli` | no window; accepted for the uniform gallery CLI — the example never opens one |
| `--frames N`, `--output-dir DIR` | consumed by the shared parser and unused: the work is one transfer |

Both backends must load and the driver must be able to export the Vulkan
allocation as a shareable handle; there is no fallback if that fails.

## Result

Example console output:

```text
[info] Result: 114514
[info] cuda_vk_interop: PASSED (result=114514, headless=true).
```

`Result` is the value the CUDA stream read out of the buffer the Vulkan stream
filled, so seeing `114514` on both sides of the report is the whole proof: the
external `VkDevice` was accepted, the memory was imported, and the event ordered
the two queues. Exit code 0 on success, 1 on a mismatch.

## See also

`example_cuda_dx_interop` (the DX flavour of the same handshake),
`example_cuda_vk_kernel_launch` (the same extension, used to run CUDA-compiled
kernels inside a Vulkan command buffer).
