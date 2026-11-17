# example_dx_custom_sdk

Creates the DirectX device through a `DirectXDeviceConfigExt` subclass that asks for its own Agility SDK version and DLL directory.

| | |
|---|---|
| Group | `dx` |
| Sources | `src/dx/custom_sdk/custom_sdk.cpp`, `src/common/headless.h` |
| Backends | dx |
| Build | `xmake build example_dx_custom_sdk` |
| Run | `example_dx_custom_sdk dx --headless` |

## Purpose

The DX backend ships with a bundled Agility SDK, which pins the D3D12 interfaces and
shader models every other `dx` example compiles against. This example overrides that
choice per device: a subclass of `DirectXDeviceConfigExt` names the SDK version and
the directory holding `D3D12Core.dll`, and the backend applies both while the device
is created — the switch to reach for when an application needs a newer (or pinned
older) Agility SDK than the backend was built with, without rebuilding anything.

## Architecture

- **The config extension.** `class CustomSDKConfig final : public DirectXDeviceConfigExt`
  overrides exactly two virtuals: `GetSDKVersion()` returns `619u` and `GetSDKPath()`
  returns `"./D3D12_619/"`. The base class defaults — version `0`, empty path — mean
  "keep what the backend was built with", so an override is opt-in per value.
- **Device creation.** `DeviceConfig config{}; config.extension = std::move(ext);
  auto device = ctx.create_device("dx", &config);`. The extension is queried and
  consumed during creation; the rest of the program sees an ordinary `Device`. The
  example never reads a backend argument — the device is always the `dx` backend and
  the positional word only labels the run.
- **How it reaches the runtime.** During `Device` construction the backend updates its
  `D3D12SDKVersion` / `D3D12SDKPath` exports from those two virtuals, before
  `CreateDXGIFactory2` / `D3D12CreateDevice`, so the D3D12 loader resolves
  `D3D12Core.dll` from the directory you named instead of the built-in one.
- **What you must provide.** A directory `D3D12_619` must exist relative to the
  working directory of the executable and hold the redistributable `D3D12Core.dll` of
  the matching Microsoft Agility SDK release (`d3d12SDKLayers.dll` next to it is worth
  having while developing). The version returned by `GetSDKVersion()` and the DLL you
  drop there must agree; point both at whatever release you downloaded.
- **Without that directory.** The built-in SDK serves the run and everything still
  works, which is why the example passes on a plain checkout. It also means the
  example is not a proof of *which* `D3D12Core.dll` was loaded: it prints the two log
  lines it expects the backend to emit when the override applies, and then verifies
  the device the ordinary way.
- **Smoke kernel.** `device.create_stream(StreamTag::COMPUTE)`, a 16-element
  `Buffer<uint>`, a `device.compile<1>` kernel writing `tid * 2u` dispatched with
  `.dispatch(16u)`, read back with `buffer.copy_to(luisa::span{result})` and compared
  entry by entry — a device built this way is proven *usable*, not only constructible.

## Usage

```bash
example_dx_custom_sdk dx # no window; the report is the console output
example_dx_custom_sdk dx --headless
```

| Option | Meaning |
|---|---|
| `<backend>` | accepted and ignored: the device is always the `dx` backend |
| `--headless`, `--cli` | no window; accepted for the uniform gallery CLI — the example never opens one |
| `--frames N`, `--output-dir DIR` | consumed by the shared parser and unused: the work is one dispatch |

To run against your own SDK copy, change the two overrides to your version and
directory, and place that directory where the process will find it.

## Result

Example console output on `dx`:

```text
Device created: dx
Custom Agility SDK configured.
Check the log output for:
  "Using custom Agility SDK version: 619"
  "Using custom Agility SDK path: ./D3D12_619/"
Kernel execution verified successfully.
[info] dx_custom_sdk: PASSED (backend='dx', headless=true).
```

`Kernel execution verified successfully.` plus exit code 0 is the real test: the device
accepted a compiled kernel and returned the expected 16 values. The two quoted lines
are what the example tells you to look for in the log; whether they appear depends on
the override having been applied, so check the `D3D12_619` directory you provided.

## See also

`example_helloworld` (the same device/kernel/readback skeleton without the config
extension), `example_dml` (another `dx`-only example).
