# example_bindless_mip

Reads and samples every mip level of one image through a bindless heap, and shows that both routes resolve the same texel.

| | |
|---|---|
| Group | `compute` |
| Sources | `src/compute/bindless_mip.cpp`, `src/common/headless.h` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_bindless_mip` |
| Run | `example_bindless_mip dx --headless` |

## Purpose

Two ways of reaching a texture level are compared side by side: an explicit
`read` of a level's texel and a `sample` with an explicit level of detail. Both go
through a `BindlessArray` slot rather than a per-draw descriptor, which is the
modern bindless model — one heap update, any number of loads. The example is a
good template for "I have an image with a mip chain and a kernel that must pick a
level at run time".

## Architecture

- **Texture.** `device.create_image<float>(PixelStorage::BYTE4, make_uint2(8u, 8u), level_count)`
  with `level_count = 4`: one 8×8 image with a full mip chain (8×8, 4×4, 2×2, 1×1).
- **Per-level upload.** `make_level_pixels()` fills a `luisa::vector<std::byte>` with
  a constant ramp — `value`, `value + 1`, `value + 2`, `255` — and `value` is
  `32 + level * 48`, so each level is a different flat colour. The loop takes
  `texture.view(level)` (a 2D view sized to that level) and enqueues
  `view.copy_from(luisa::span{pixels})` on the stream, so all four uploads are one
  batch.
- **Bindless heap.** `device.create_bindless_array(1u)` plus
  `heap.emplace_on_update(0u, texture, Sampler::point_edge()).update()` puts the
  whole image, sampler included, into slot 0. The sampler is point/edge, so a
  sample at a texel centre returns that texel exactly — that is what makes the
  comparison meaningful.
- **Kernel.** `Kernel1D read_and_sample = [](BindlessVar bindless, BufferVar<float4> output)`
  takes `level = dispatch_id().x`, fetches `bindless.tex2d(0u)`, then
  `tex.read(make_uint2(0u), level)` (texel 0,0 of that level) and
  `tex.sample(make_float2(0.5f), cast<float>(level))` (explicit LOD = level).
  Both `float4` values land in `output` at `level * 2` and `level * 2 + 1`, so the
  result buffer is `2 * level_count` elements.
- **Dispatch and readback.** `shader(heap, out).dispatch(level_count)` launches four
  threads — one per level. `out.copy_to(luisa::span{host})` followed by
  `synchronize()` brings all eight values back in one transfer.
- **Report.** The host prints one line per level with both colours and then a
  summary; there is no image, the console is the result.

## Usage

```bash
example_bindless_mip dx
example_bindless_mip vk --headless
```

| Option | Meaning |
|---|---|
| `<backend>` | `dx`, `vk`, `cuda`, …; the image format and the bindless heap come from it |
| `--headless`, `--cli` | no window; accepted for the uniform gallery CLI — the example never writes an image |
| `--frames N`, `--output-dir DIR` | consumed by the shared parser and unused |

Running without arguments prints `Usage: … <backend> [--headless]` and returns 0,
because nothing was attempted.

## Result

Example console output on `dx` — the `read` and `sample` columns agree, and the
colour rises with the level because of the `32 + level * 48` ramp:

```text
[info] level 0: read=(0.1254902, 0.12941177, 0.13333334, 1), sample=(0.1254902, 0.12941177, 0.13333334, 1)
[info] level 1: read=(0.3137255, 0.31764707, 0.32156864, 1), sample=(0.3137255, 0.31764707, 0.32156864, 1)
[info] level 2: read=(0.5019608, 0.5058824, 0.50980395, 1), sample=(0.5019608, 0.5058824, 0.50980395, 1)
[info] level 3: read=(0.6901961, 0.69411767, 0.69803923, 1), sample=(0.6901961, 0.69411767, 0.69803923, 1)
[info] bindless_mip: success (4 mip levels read and sampled, headless=true).
```

Byte values are reported normalized to `[0, 1]`, which is why the ramp that was
written as `32, 80, 128, 176` shows up as `0.125`, `0.314`, `0.502`, `0.690`.

## See also

`example_helloworld` (the minimal buffer round trip),
`example_image_processing` (multi-pass `Kernel2D` work on images instead of one
level probe).
