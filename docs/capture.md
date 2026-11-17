# Capture mode: save the result to a picture

Every gallery example that produces a picture can write it to a file you name,
from the command line, without a window:

```bash
example_path_tracing dx --capture cornell.jpg --spp 64
example_blackhole vk --capture blackhole.png --frames 1
example_mpm88 cuda --capture mpm88.jpeg --frames 8
```

`--capture <path>` (alias `--screenshot <path>`) means:

- the example runs its window-less path - asking for a capture implies
  `--headless`/`--offline`, so there is no window, no swapchain and no
  interaction;
- the picture goes to exactly `<path>`, and **the extension picks the
  container**;
- the run still exits `0` on success and non-zero on failure, so it composes
  with `&&` in scripts and CI;
- if the example writes several intermediate images (a texture generator, an
  image-processing filter chain), the first - the one the user asked for - goes
  to `<path>`, and the rest keep going to `--output-dir`.

## Supported containers

| Extension | Writer | Notes |
|---|---|---|
| `.png` | `stbi_write_png` | lossless, RGBA8, the default when the extension is unknown |
| `.jpg`, `.jpeg` | `stbi_write_jpg` | quality 95, RGB |
| `.bmp` | `stbi_write_bmp` | uncompressed |
| `.tga` | `stbi_write_tga` | uncompressed |

All four come from [stb_image_write](https://github.com/nothings/stb) through
`src/common/image_io.h`, which is the single place that maps an extension to a
writer:

```cpp
luisa::gallery::image_io::write("out.jpg", pixels, width, height, 4);
```

Missing parent directories are created, so `--capture docs/images/x.jpg` works
even when `docs/images` does not exist yet.

## Where the option comes from

| Examples | Parser | Field |
|---|---|---|
| `rendering/`, `simulation/`, `coro/` renderers | `luisa::ref::ExampleOptions` (`src/common/reference_compare.h`) | `capture_path`, `image_path(default_name)` |
| `gui/`, `compute/`, `coro/`, `ntc/` tools | `luisa::gallery::headless::parse` (`src/common/headless.h`) | `capture_path`, `path(default_name)` |

Both parsers accept `--capture <file>` and `--screenshot <file>`, and both keep
the historical behaviour when the option is absent: the picture is written to
`<output-dir>/<default name>.png`.

## Capturing every example at once

`scripts/capture_gallery.py` runs the whole catalogue headless, one `--capture`
per target and backend, and records what happened:

```bash
python scripts/capture_gallery.py --backend dx --backend vk --backend cuda
python scripts/capture_gallery.py --backend cuda --filter rendering
python scripts/make_doc_images.py            # downscale into docs/images/
```

The images in this documentation were produced exactly that way; see
[headless.md](headless.md) for the headless convention itself and
[README.md](README.md) for the per-example pages.
