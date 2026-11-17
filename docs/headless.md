# Headless mode (CLI / CI / agents)

Every example in this repository runs without a window, so a script, a CI job or
an agent can execute it and check the result without touching a desktop.

## Convention

```
<example> <backend> --headless [--frames N] [--output-dir DIR] [example options]
```

* `--headless` — no window, no swapchain, bounded work, exit code 0 on success.
  `--offline` is the historical spelling of the same thing and stays supported;
  `--cli` is a synonym as well.
* `--frames N` — how many frames/iterations to run (`--iterations N` for the
  examples that predate this switch). `0` (or omitted) means "one frame, the
  smallest work the example can do meaningfully".
  * `--output-dir DIR` — where images/reports are written; defaults to the current
    directory. Examples that produce an image write `<name>.png` there.
  * `--capture FILE` (alias `--screenshot FILE`) — write the result picture to
    exactly `FILE`; the extension chooses the container (`.png`, `.jpg`/`.jpeg`,
    `.bmp`, `.tga`) and asking for a capture implies `--headless`. See
    [capture.md](capture.md).

* The examples also take their own options after these (`--spp 1`,
  `--max-spp-per-dispatch N`, `--compare <reference.png>`, ...); run
  `<example> <backend> --help`-style usage is printed when the arguments are
  wrong, and every example logs its usage line through `LUISA_INFO`.

Backends: `cuda`, `dx` (Windows), `vk`, `metal` (macOS), `hip`, `fallback`, or
whatever the LuisaCompute build enabled. The executable name `test_*` forces the
headless path in the examples that have a test mirror.

## Running everything

```bash
# xmake build, CUDA backend, every runnable example
python scripts/headless_smoke.py
# the same catalogue on DirectX or Vulkan
python scripts/headless_smoke.py --backend dx -k 4
python scripts/headless_smoke.py --backend vk -k 4
# CMake build, only the rendering group, two at a time
python scripts/headless_smoke.py --exe-dir build-cmake/bin --filter rendering -k 2
# what would run (and what is skipped, and why)
python scripts/headless_smoke.py --list
# probe a backend: also run the examples the manifest marks as skipped
python scripts/headless_smoke.py --backend dx --include-skipped -k 4
```


The driver writes one log per example to `.kimix_cache/headless/<name>.log`,
prints a summary, and exits non-zero if anything failed or timed out. Examples
that need model weights or a foreign toolchain are listed with a `skip` reason
instead of being silently ignored.

A `headless_skip` entry in `scripts/gallery_targets.json` is either one reason
that applies to every backend, or an object keyed by backend
(`{"cuda": "..."}`), so a CUDA-only limitation does not keep the DirectX and
Vulkan runs out of the catalogue. `--include-skipped` ignores the reasons and
runs everything, which is how a newly enabled backend is probed.

With CMake, `ctest` has the same coverage:

```bash
cd build-cmake
cmake -DGALLERY_TEST_BACKENDS="cuda;dx;vk" . # one entry per backend (default: cuda)
ctest -L headless --output-on-failure        # runs <exe> <backend> --headless
ctest -R example_path_tracing                # one example
```

With more than one backend the entries are named
`headless_<target>_<backend>`; `SKIP_BACKENDS` from the manifest filters the
combinations that cannot work (e.g. the spectrum example is registered for `dx`
and `vk` only).


## Adding headless mode to an example

```cpp
#include "common/headless.h"   // src/common/headless.h

int main(int argc, char *argv[]) {
    auto headless = luisa::gallery::headless::parse(argc, argv);  // consumes the generic flags
    ...
    if (headless.enabled()) {
        // device-only path: no Window, no Swapchain, write PNGs into headless.output_dir
    }
}
```

Examples that parse `luisa::ref::ExampleOptions` (all of `rendering/` and
`simulation/`) get `--headless`, `--frames` and `--output-dir` for free: the
parser treats them exactly like `--offline`/`--iterations`.

## Per-group status

| Group | Headless entry point |
|---|---|
| `rendering/` | `--headless` == `--offline`; `--spp 1 --frames 1` is the fast path. PNG output honoured |
| `simulation/` | same; `--frames` bounds the simulation loop |
| `compute/` | device-only binaries; `--headless` is accepted and ignored (they never open a window) |
| `coro/` | `--headless` runs the scheduler without the debug viewer; `--frames N` = scheduler dispatches, images honoured |
| `gui/` | `--headless` runs the offscreen path: no window, no swapchain; `gui_imgui.png`, `gui_swapchain.png`, `gui_swapchain_static.png`, `gui_win_hdr.png`, `gui_image_process.png` and the procedural-texture PNGs are written to `--output-dir`. `example_mnist` is a shared library driven by `test_mnist.py` (CUDA/DX interop + MNIST weights), and `example_swapchain_wx` / `example_swapchain_qt` are CMake-only and need the wxWidgets/Qt toolchain — none of the three has a headless mode |
| `ml/`, `tensor/` | already CLI driven |
| `extension/`, `interop/`, `dx/` | device-only; several need a second backend or a vendor SDK (see `scripts/headless_smoke.py --list`) |
| `ocr/`, `embedding*/`, `ntc/` | CLI tools, but they need model data (`model.bgm`, OCR package, NTC textures) that is not in the repository |
| `ios/` | moved verbatim, needs the Apple toolchain; not wired into the gallery build |
