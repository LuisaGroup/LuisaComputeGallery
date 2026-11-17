# example_native_shader

Compiles hand-written HLSL, GLSL or CUDA C++ through `NativeShaderExt` and replays a JSON dispatch document that declares the shaders, the resources and the workflow.

| | |
|---|---|
| Group | `compute` |
| Sources | `src/compute/native_shader_examples/native_shader.cpp`, `src/compute/native_shader_examples/native_shader_dispatch.cpp`, `src/compute/native_shader_examples/native_shader_runtime.cpp`, `src/compute/native_shader_examples/simple_add.json` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_native_shader` |
| Run | `example_native_shader dx src/compute/native_shader_examples/simple_add.json --headless` |

## Purpose

Every other example writes shaders in the DSL, which Luisa translates per backend; here
the source is yours, already written in the target's own language, and Luisa only
compiles, reflects, loads and dispatches it. The workload — which shaders to build, which
buffers and textures to create, what to submit, read back and verify — is a **dispatch
document**, a plain JSON file. This is the entry point for running a shader you have.

## Architecture

- **Command line and modes (`native_shader.cpp`).** Parses the document and the
  command-line shaders, picks the offline or interactive path, runs the workflow, prints
  the report and exits non-zero on the first failure; it also owns the `--self-test`
  corpora and the ImGui pass that displays `mode.display_image`.
- **Document model and codec (`native_shader_dispatch.{h,cpp}`).** `ParseResult` and
  `Diagnostics`: parsing is total, every problem becomes `<json path>: <message>`, and the
  semantic pass checks references, sizes, key sets and language compatibility. The writer
  is canonical and byte-stable (that is what `--dump-dispatch` emits), and the unit has no
  device and no filesystem, so it is host-testable.
- **Execution (`native_shader_runtime.{h,cpp}`).** `ResourceRegistry` creates what the
  document declares, `ShaderRegistry` compiles each variant through
  `NativeShaderExt::compile()` / `load()` and registers the DSL kernels,
  `WorkflowExecutor` replays the commands into one command list on a `Stream`, and
  `PathResolver` resolves relative paths against the document directory (or `--workdir`).
- **Document shape.** `version`; `mode` (`offline` | `interactive`, `frames`,
  `display_image`); `config` (`default_language`, `output_dir`, `limits`); `shaders[]`
  (`name`, `language`, `path` or inline `source`, `entry_point`, `push_constant_size`,
  `block_size`, optional manual `bindings`); `resources[]` (`buffer`, `texture`,
  `volume`, `bindless_array`, `accel`, `mesh`); `workflow[]` (`buffer_upload`,
  `native_dispatch`, `buffer_download` with an optional `verify`, `accel_build`,
  `shader_dispatch`, `custom_command`, `synchronize`).
- **One variant per language.** A shader `name` may appear once per `language`; the backend
  picks what it speaks (dx → HLSL, vk → GLSL then HLSL, cuda → CUDA C++) and logs the
  skipped variants, which is how one document runs everywhere.
  `native_dispatch.bindings` address the reflection table by `index`, by `register`/`space`
  or positionally, each naming a resource and a `usage` (`read`, `write`, `read_write`);
  `uniforms` are scalars/vectors of a self-describing type, or raw `hex`. CUDA is CUDA C++
  through NVRTC only, and only on a driver that accepts the toolkit's PTX.
- **Sample documents.** `simple_add.json` (inline sources, three languages),
  `simple_add_manual.json` (hand-declared bindings), `scale_offline.json` /
  `scale_interactive.json` with `shaders/scale.{hlsl,glsl,cuda}`, `all_commands_offline.json`
  (every command once); `native_shader_embedded.h` is the default workflow.

## Usage

```bash
example_native_shader dx --self-test
example_native_shader dx src/compute/native_shader_examples/simple_add.json --headless
example_native_shader vk src/compute/native_shader_examples/scale_offline.json --headless --stats
```

| Option | Meaning |
|---|---|
| `<backend>`, `[dispatch.json]`, `[shader...]` | the device, the document (omitted: the embedded default workflow), and native sources appended to its `shaders` |
| `--headless`, `--offline`, `--interactive` | no window (the default), or open the ImGui window on `mode.display_image` (needs the gui module) |
| `--frames N`, `--exit-after-frames N` | how often the workflow runs, or when the window closes |
| `--output-dir DIR`, `--workdir DIR` | prefix for relative output sinks; base directory for the document's paths |
| `--strict`, `--stats`, `--sync-uploads` | warnings become errors; per-stage timings; synchronise after each upload |
| `--self-test`, `--dump-dispatch FILE`, `--print-schema` | codec/negative/execution corpora; write the effective document and exit; print the schema |
| `--entry NAME`, `--push-constant-size N`, `--language LANG` | entry point, uniform bytes, `hlsl`/`glsl`/`cuda_nvrtc` for command-line shaders |

## Result

Example console output for `simple_add.json` on `dx` — one dispatch of
`dst[i] = src[i] + 10` over 32 floats, verified against `src * k + c`:

```text
[info] compiling native hlsl shader 'add' with DXC (DXIL): entry 'CSMain', shader model 65, block size (32, 1, 1), 4 uniform byte(s)
[info] compiled a native hlsl shader 'add': 3544 bytes, workgroup size (32 1 1), 3 reflected binding(s)
[info] dispatch document executed on 'dx': 1 frame(s), 3 device command(s)
[info] native_shader: PASSED (backend='dx', 1 frame(s)).
```

`--self-test` prints a check count instead, e.g. `self test on 'dx': 218 check(s), 0 failure(s).`

## See also

`example_native_shader_minimal` (the same extension in one file, no document); the full document schema is in `src/compute/native_shader_examples/README.md`.
