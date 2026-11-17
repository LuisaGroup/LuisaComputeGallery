# example_native_shader_minimal

The smallest complete native-shader example: one file that compiles a hand-written kernel (or a vertex/pixel pair), dispatches it and verifies the result on the host.

| | |
|---|---|
| Group | `compute` |
| Sources | `src/compute/native_shader_examples/minimal.cpp`, `src/common/headless.h` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_native_shader_minimal` |
| Run | `example_native_shader_minimal dx --headless` |

## Purpose

Start here before `example_native_shader`: no dispatch document, no JSON codec, no GUI —
only the five steps the native route always takes, in order. It answers "what must my
application call, and what does a native shader look like on each backend". Two modes:
`compute` (`dst[i] = src[i] * k + c` over 1024 floats) and `raster` (two quads).

## Architecture

The whole example is five calls, in the order the file documents them:

1. `auto ext = device.extension<NativeShaderExt>()` — the extension, or a `[skip]` line.
2. `ext->compile(info)` with a `NativeShaderCompileInfo` holding `language`, `source`,
   `entry_point`, `push_constant_size` and (for CUDA) `block_size`. The result carries the
   binary, the reflected block size and the reflected binding table, which the example
   prints.
3. `ext->load(compiled)` returns the metadata, and `NativeShader shader{*ext, std::move(metadata)}`
   is the RAII owner — it must outlive the dispatch.
4. `auto launcher = shader.launcher(); launcher.add_buffer_by_index(0u, src.view(), Usage::READ)
   .add_buffer_by_index(1u, dst.view(), Usage::WRITE).add_uniform(k).add_uniform(c);`
   then `std::move(launcher).build(make_uint3(element_count, 1u, 1u))`. `build()` takes the
   **total thread count**, not a grid: the block size comes from the shader's reflected
   `[numthreads]` / `local_size_*`.
5. `stream << std::move(command) << synchronize()`, then `dst.copy_to(...)` and a host
   comparison with `host_src[i] * k + c` (`k = 2.5`, `c = -1.5`).

The kernel sources are three literals next to the code, one per language: `hlsl_dx`
(`StructuredBuffer<float> : register(t0)`, `RWStructuredBuffer<float> : register(u0)`,
`cbuffer : register(b0)` with `float k; float c;`, `[numthreads(64,1,1)]`), `glsl_vk`
(`layout(local_size_x = 64)`, `binding = 0/1`, `layout(push_constant)`), and
`cuda_source` (`extern "C" __global__ void scale(const float *, float *, float, float)`).
`run_compute()` picks the one the backend speaks; an unknown backend prints `[skip]`
instead of failing.

`raster` mode uses the same extension for graphics: `ext->compile_raster(...)` compiles
the vertex and pixel stages, `ext->load_raster(...)` creates the pipeline, and the draw is
described with a `MeshFormat`, a `RasterState`, two render targets, a `DepthBuffer` and a
viewport. Two quads are drawn and three samples per target are checked — the tinted colour,
its complement, and the clear colour in the gap — so a wrong second target or a wrong
viewport is caught. A backend without the native raster route reports `[skip]`.

## Usage

```bash
example_native_shader_minimal dx --headless
example_native_shader_minimal dx compute --headless
example_native_shader_minimal vk raster glsl --headless
example_native_shader_minimal vk raster hlsl --headless
```

| Option | Meaning |
|---|---|
| `<backend>` | `dx`, `vk` or `cuda` for `compute`; `raster` needs a backend with the native raster route (`dx`, `vk`) |
| `compute` \| `raster` | which mode to run (default `compute`) |
| `hlsl` \| `glsl` | language of the `raster` stages (default: `hlsl` on `dx`, `glsl` elsewhere); `compute` picks by backend |
| `--headless`, `--cli` | no window; accepted for the uniform gallery CLI — the example never opens one |

`dx` speaks HLSL, `vk` speaks HLSL and GLSL, `cuda` speaks CUDA C++ for `compute` only.

## Result

Example console output for `compute` on `dx` — the reflection first, then the verdict:

```text
compiled a native dx shader: 3596 byte(s), workgroup (64 1 1), 3 reflected binding(s)
  binding set 0 register 0: kind 0, usage 0x1
1024 element(s) verified, max |error| = 0
PASSED compute on 'dx'
[info] native_shader_minimal: PASSED (headless=true).
```

`raster` prints one line per checked pixel and ends with `PASSED raster (glsl) on 'vk'`.
The process exits non-zero if any check fails.

## See also

`example_native_shader` (the same extension driven by a JSON dispatch document);
`src/compute/native_shader_examples/README.md` holds the full language matrix and schema.
