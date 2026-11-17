# example_wave_equation

A finite-difference solver for the 2D wave equation on a 512x512 grid, shaded as rippling water with caustics.

| | |
|---|---|
| Group | `simulation` |
| Sources | `src/simulation/wave_equation.cpp` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_wave_equation` |
| Run | `example_wave_equation dx --headless --capture wave.jpg` |

![example_wave_equation](../images/example_wave_equation.jpg)

## Purpose

The classic explicit solver for `u_tt = c^2 * laplacian(u)`, written out as a
five-point stencil over three images. It is the smallest example in the simulation
group that is a real PDE integrator, and it shows two patterns worth copying: a
three-buffer rotation (`prev`, `curr`, `next`) for second-order time stepping, and a
presentation pass that is completely separate from the physics.

## Architecture

- **Host side.** Three `Image<float>` of 512x512 with `PixelStorage::FLOAT1` hold the
  height field at three time levels: `height_prev`, `height_curr`, `height_next`.
  Parameters are fixed and chosen for stability: wave speed `c = 0.2`, `dt = 1.0`,
  damping `0.995`, so `c^2 dt^2 = 0.04` stays well below the 0.5 CFL limit. `display`
  is a 1024x1024 `PixelStorage::BYTE4` image (2x upscale of the simulation grid).
- **`clear`** (2D, 16x16 block) zeroes an image; it is dispatched three times to start
  from a flat surface and again whenever the user presses `SPACE` in the window.
- **`wave_step`** (2D, 16x16 block over 512x512) reads the four neighbours with
  clamped coordinates, builds the five-point Laplacian
  `left + right + up + down - 4 * curr`, and computes
  `next = 2 * curr - prev + c^2 dt^2 * laplacian`. The result is damped by 0.995 and
  clamped to `[-2, 2]` so a bad input cannot blow up the field.
- **`drop_droplet`** (2D) adds a Gaussian bump of radius 15 at a given centre with a
  given strength; it is how both user input and rain enter the simulation.
- **`render`** (2D, 16x16 block over 512x512) turns height into water:
  slopes are estimated from samples two pixels away, `smoothstep` picks highlights and
  foam, and a sine-based shimmer adds surface detail.
- **`upscale`** (2D, 16x16 block over 1024x1024) copies `uv / 2` from the shaded
  512x512 image to the display image.
- **`simulate_frame`** builds one `CommandList`: droplet dispatches for the current
  mouse stroke (a stroke is interpolated in up to ten steps so fast drags leave a
  continuous wake), an occasional random rain drop (`rng() % 120`), four wave
  sub-steps with the three images rotated after each one, then `render` and `upscale`.
- **Offline path.** 500 frames with `std::mt19937` seeded with 42, so the random rain
  is reproducible; feature checks on bright and wave pixel counts and on occupied
  32x32 tiles guard the output before the PNG is written.

## Usage

```bash
# interactive (needs a window and the gui module)
example_wave_equation dx

# offline: 500 frames, one image out
example_wave_equation dx --headless --capture wave.jpg
example_wave_equation dx --offline --output-dir out
```

| Option | Meaning |
|---|---|
| `--headless`, `--offline` | no window, no swapchain; run the bounded loop and write the image |
| `--capture <file>` | write the result to exactly this file (`.png`, `.jpg`/`.jpeg`, `.bmp`, `.tga` select the container, anything else falls back to PNG); implies the offline path |
| `--output-dir <dir>` | where the default `test_wave_equation.png` is written (default: the working directory) |
| `--frames N` | parsed by the shared option parser; this example's offline loop is fixed at 500 frames |
| `-c`, `--compare <ref.png>` | compare the result against a reference image and exit non-zero when it does not match |

In the window, click and drag to stir the surface, `SPACE` resets the field and `ESC`
quits.

## Result

Interfering ripples over deep-blue water, with bright foam where two wavefronts meet:

```text
Headless wave equation finished: 500 frame(s), wrote 'wave.jpg'.
```

## See also

`example_game_of_life` (a two-buffer stencil), `example_mpm88` and `example_mpm3d`
(grid solvers coupled to particles).
