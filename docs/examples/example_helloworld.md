# example_helloworld

The smallest complete LuisaCompute program: one kernel writes a struct into a buffer and the host reads it back.

| | |
|---|---|
| Group | `compute` |
| Sources | `src/compute/helloworld.cpp`, `src/common/headless.h` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_helloworld` |
| Run | `example_helloworld dx --headless` |

## Purpose

Everything an application has to do before and after a kernel is here exactly
once: create a context and a device, allocate a buffer, compile a kernel, submit it
on a stream, copy the result back and check it. It also shows how a value type
reaches the device — the record holds a `half` and a `uint16_t`, so it exercises
the 16-bit paths of the backends as well as the host/device struct ABI.

## Architecture

- **Record.** `struct alignas(4) Test1 { half a; uint16_t b; };` is registered with
  `LUISA_STRUCT(Test1, a, b)`, which is how the DSL learns the layout. The
  `alignas(4)` and a `static_assert` pin `sizeof == 4` / `alignof == 4`: the two
  fields on their own only need two bytes each, but the classic DXC/cbuffer ABI
  on DirectX pads to four, and the host must agree with the shader compiler.
- **Host setup.** `Context ctx{executable}`, `Device device = ctx.create_device(argv[1])`,
  then `Buffer<Test1> bf = device.create_buffer<Test1>(1u)` — a one-element device
  buffer; `Test1 t{}` is its host mirror.
- **Kernel.** `device.compile<1>([&] { ... })` builds a `Kernel1D` that takes no
  arguments and closes over `bf`. It declares `Var<Test1> tt`, assigns
  `tt.a = 1.5f`, `tt.b = 66`, doubles the value *on the device*
  (`tt.b = tt.b + tt.b`) and stores it with `bf->write(0u, tt)`.
- **Dispatch and readback.** One `Stream` carries everything:
  `stream << s().dispatch(1u) << bf.copy_to(luisa::span{&t, 1u}) << synchronize();`
  The single thread is enough because the kernel writes one element.
- **Report.** `LUISA_INFO("{}, {}", ...)` prints the two fields, then compares them
  with `expected_a`/`expected_b` (`132 == 66 + 66`). A mismatch logs
  `LUISA_WARNING` and returns 1, which is what makes the mirrored `test_helloworld`
  target a conformance check instead of a smoke test.
- **Command line.** `luisa::gallery::headless::parse()` consumes the shared gallery
  flags, so `argv[1]` is still the backend after parsing.

## Usage

```bash
example_helloworld dx
example_helloworld cuda --headless
```

| Option | Meaning |
|---|---|
| `<backend>` | `dx`, `vk`, `cuda`, … — the only required argument; the device is created from it |
| `--headless`, `--cli` | no window; accepted for the uniform gallery CLI — the example never opens one |
| `--frames N`, `--output-dir DIR` | consumed by the shared parser and unused: the work is one dispatch |

Without a backend the program logs its usage line through `LUISA_INFO` and exits
with 1.

## Result

Example console output on `dx`:

```text
[info] 1.5, 132
[info] helloworld: success (backend='dx', headless=true).
```

Exit code 0 means the device wrote `a = 1.5` and `b = 132`; the line before the verdict
is what the host read back. Called without a backend it prints its usage line and exits
with 1:

```text
[info] Usage: example_helloworld <backend> [--headless]. This example never opens a window; --headless only makes that explicit for scripts and CI.
```

## See also

`example_bindless_mip` (images, mip levels and a bindless slot),
`example_gdeflate` (two compiled kernels over `ByteBuffer` plus a timing loop).
