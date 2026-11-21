# Agent notes for LuisaComputeGallery

The gallery keeps the LuisaCompute examples (moved out of
`LuisaCompute/examples`), plus their tests, in one buildable project with both
xmake and CMake. This file is the short version of what an automated agent needs
to work here; the user-facing documentation lives in `docs/`.

## Layout

```
LuisaCompute        symlink to the machine's LuisaCompute checkout (LC_DIR)
thirdparty/         the submodule path, used only if LC_DIR points there
src/<group>/        the examples, one generated xmake.lua + CMakeLists.txt per group
src/common/         shared headers: headless.h, reference_compare.h, image_io.h
tests/              tests moved alongside the examples
gallery/ app/       GalleryRT runtime library and the gallery.exe viewer
scripts/            build generator, runners, data pack/unpack, doc tooling
docs/               build.md, headless.md, capture.md, examples/*.md, images/
data/               example weights/textures (git-ignored, see below)
```

The `LuisaCompute` entry at the repository root is a **symlink to the shared
LuisaCompute checkout of the machine** (`C:\dev\LuisaCompute` on this one, which
is itself a junction to the checkout), and `options.lua` sets
`LC_DIR = "LuisaCompute"` so every build uses it. Never point the gallery at a
second copy: the build cache and the `patches/` below assume that one tree.
`thirdparty/LuisaCompute` stays an uninitialized submodule for the CI case where
no shared checkout exists (`git submodule update --init thirdparty/LuisaCompute`
and `LC_DIR = "thirdparty/LuisaCompute"`).

`scripts/gallery_targets.json` is the single source of truth for the target
list. After editing it run `python scripts/gen_build.py` (it also has a
`--check` mode that fails when the generated files drift).

## Build and run

```bash
xmake f                                  # reads options.lua (LC_DIR, lc_options)
xmake build -j 16                        # everything
xmake build example_path_tracing         # one target
```

Binaries land in `build/windows/x64/release/`. Every example runs without a
window:

```bash
<example> <backend> --headless [--frames N] [--output-dir DIR] [options]
<example> <backend> --spp 64 --capture result.jpg     # save the picture
```

`--capture <file>` writes the result to exactly that path; the extension picks
the container (`.png`, `.jpg`/`.jpeg`, `.bmp`, `.tga`). See `docs/capture.md`.

Runner scripts:

```bash
python scripts/headless_smoke.py --backend dx --jobs 1          # pass/fail only
python scripts/capture_gallery.py --backend dx --backend vk --backend cuda
python scripts/make_doc_images.py                               # -> docs/images/
python scripts/gen_docs_index.py                                # -> docs/README.md
```

## Example data: `pack_data.py` and `unpack_data.py`

Some examples need weights or textures that are far too large for git. They live
in `data/` (git-ignored) and travel as a single 7-Zip archive attached to the
`Data` release of `https://github.com/LuisaGroup/LuisaComputeGallery`.

### `scripts/unpack_data.py` — restore `data/`

Downloads the **newest** `<date-time>.7z` asset of the `Data` release with `gh`
and extracts it into `data/`, overwriting anything already there (`7z x -aoa`):

```bash
python scripts/unpack_data.py                  # newest package -> data/
python scripts/unpack_data.py --list           # what is published (name, size, date)
python scripts/unpack_data.py --keep-archive    # keep the downloaded .7z in dist/
python scripts/unpack_data.py --asset 20260101-120000.7z
python scripts/unpack_data.py --dry-run        # print the commands, touch nothing
```

Options: `--data-dir DIR`, `--cache-dir DIR` (default `dist/`), `--repo
owner/name`, `--tag TAG`, `--7z PATH`, `--gh PATH`.

Requirements: `gh` authenticated with a token that can read the release, and
7-Zip (`7z` on `PATH`, `$LUISA_7Z`, or the default Windows install path).

### `scripts/pack_data.py` — publish a new package

Compresses `data/` into `dist/<YYYYmmdd-HHMMSS>.7z` (7-Zip, `-mx9`, multithreaded,
archive rooted at the *contents* of `data/`) and uploads it to the `Data`
release, creating the release when it does not exist:

```bash
python scripts/pack_data.py                    # pack + upload (gh release upload --clobber)
python scripts/pack_data.py --no-upload        # only create the archive
python scripts/pack_data.py --list             # list the published assets
python scripts/pack_data.py --dry-run          # print the 7z/gh commands
python scripts/pack_data.py --level 5 --archive 20260101-120000.7z
```

Options: `--data-dir DIR`, `--archive-dir DIR` (default `dist/`), `--archive
NAME`, `--repo owner/name`, `--tag TAG`, `--level 0-9`, `--7z PATH`, `--gh PATH`,
`--keep-existing`, `--no-upload`, `--dry-run`, `--list`.

The upload is a release *asset*: it is not part of the git history, and
re-uploading the same name replaces the asset (`--clobber`). Both scripts use
only `gh` and 7-Zip, so they work on Windows, Linux and macOS.

## Data inventory

| path | used by |
|---|---|
| `data/bge-m3/` (`model.bgm`, `tokenizer.json`) | `example_embed_text`, `example_embed_dump_golden`, `example_embed_retrieval` |
| `data/ntc/material.ntc` + `material_manifest.json` + `input.png` | `luisa_ntc_cli`, `luisa_ntc_bench`, `luisa_ntc_test`, `luisa_ntc_viewer` |

`example_embed_mm_*` additionally need a packed `.mmegm` from the gated
`google/embeddinggemma-300m` checkpoint, and `example_ocr` needs a validated
`.lcocr` package that does not exist yet (the engine is fail-closed). See
`.kimix_cache/result/SUMMARY.md` for the honest state.

## Conventions

- C++23, `luisa::` containers in device-facing code, no RTTI/exceptions in
  library code.
- New shared example helpers go in `src/common/` and are included as
  `"common/<name>.h"`; include paths are set up by `scripts/gallery_build.lua`.
- A new example: add a target to `scripts/gallery_targets.json`, run
  `python scripts/gen_build.py`, drop the sources in the group directory, and
  give it a `headless` entry so `capture_gallery.py` can run it.
- Documentation: one page per target in `docs/examples/<target>.md` (template in
  `.kimix_cache/docs_style.md`), index generated, images in `docs/images/`.
- Do not commit `data/`, `build/`, `dist/`, `.kimix_cache/`, `options.lua`.

## LuisaCompute patches

The shared checkout is LuisaCompute at the submodule pin, and a fix that had to
go into LuisaCompute lives in `patches/` as a patch file so it survives a fresh
clone of that tree:

```bash
git -C LuisaCompute apply ../patches/luisa-native-shader-ptx-retry.patch
git -C LuisaCompute apply --check --reverse ../patches/luisa-native-shader-ptx-retry.patch  # is it already in?
```

(`LuisaCompute` here is the symlink to the shared checkout; use
`git -C C:/dev/LuisaCompute apply C:/dev/LuisaComputeGallery/patches/...` if the
shell cannot follow the link.)

`luisa-native-shader-ptx-retry.patch` makes the CUDA native-shader load path
retry once with a version-patched PTX when the installed driver is older than
the toolkit that produced it (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`). It is
currently applied.

`luisa-dx-default-warp-size-narrowing.patch` hands the Wave32 default warp
size to the HLSL codegen as an explicit `static_cast<uint8_t>(32u)`; the bare
`32u` narrowed inside `eastl::optional`'s in-place storage, which MSVC rejects
with C2397 (`src/backends/dx/DXApi/LCDevice.cpp`). It is currently applied.

`luisa-hlsl-std-span-include.patch` adds the missing `<span>` include to
`src/backends/common/hlsl/shader_compiler.cpp`; the DXIL WaveSize bitcode patch
uses `std::span` and fails to compile after the latest `origin/next` update
without the standard-header declaration. It is currently applied.

`luisa-vk-cooperative-extension-status.patch` adds
`VulkanExtensionStatusExt::cooperative_status()` (and makes the per-name
`is_supported`/`is_enabled`/`is_active` queries answer for those two names)
so the state of `VK_KHR_cooperative_matrix` / `VK_NV_cooperative_vector` is
readable at run time instead of only in the device log: the Vulkan backend
negotiates them outside the optional-extension registry, where every query
used to answer false. The reported fields also separate "enabled" from
"usable" (the SPIR-V codegen emits FLOAT32/FLOAT32/FLOAT32/FLOAT32 with
transpose=false only) and carry the number of enumerated cooperative-vector
configurations. It is currently applied.

`luisa-vk-pso-cache-identity-constexpr.patch` drops `constexpr` from
`ComputeShader::pipeline_cache_identity_for` (Vulkan): its body calls
`eastl::optional::value_or`, which is not `constexpr` in the bundled EASTL, so
MSVC rejects the function with C3615. The only caller builds the PSO cache
identity at run time, so nothing needs the constant-expression form. It is
currently applied.

`luisa-cuda-tile-ir-xmake.patch` gives the xmake build the optional CUDA 13.4+
native Tile path that only CMake had. It adds the `lc-cuda-tile-compiler` target
(`luisa_cuda_tile_compiler`, built from `cuda_tile_ir_compiler.cpp`, linked
against static NVRTC with the static CRT and `/STACK:16777216` on Windows) and
gates it on the toolkit that `cuda_sdkdir()`/`find_cuda()` resolves, not on a
guessed config value: `cuda_tile.h`, `CUDA_VERSION >= 13040` read out of that
toolkit's own `cuda.h`, `nvrtcGetTileIR` in its `nvrtc.h`, and a `tileiras` next
to `nvcc`. Only when all of that is present does `lc-backend-cuda` get
`LUISA_CUDA_TILE_IR_ENABLED=1`, `LUISA_CUDA_TILE_IR_ASSEMBLER=<tileiras>` and a
build-order dependency on the helper; otherwise the target degrades to a phony
no-op, so pre-13.4 toolkits keep the previous fail-closed behaviour and never
reference `nvrtcGetTileIR`. Ordinary DSL/native-shader compilation and the
`luisa_nvrtc` helper are untouched, and the helper lands in the ordinary target
directory next to `luisa-backend-cuda` and `luisa_nvrtc`
(`scripts/gallery_build.lua` depends on it there, like `luisa_nvrtc`). It is
currently applied.

luisa-cuda-multiprocessor-query.patch teaches the CUDA backend one more
DeviceInterface::query() key: multi_processor_count, answered with
`cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)` inside
`with_handle` and formatted as decimal text (`src/backends/cuda/cuda_device.cpp`).
A failing driver call returns an empty string and logs no warning -- "this device
cannot tell you" is a capability answer, not an error. It exists because the Qwen
engine's utilization evidence (blocks per SM: 176 blocks over 80 SMs = 2.2, 16 legacy
blocks = 0.2) and its split-KV grid bound need a real SM count instead of a guess; DX
and VK still answer nothing for that key, which the caps record reports as
unqueryable. It is currently applied.

`luisa-cuda-sparse-buffer-tile-offset.patch` fixes sparse-buffer tile mapping in
`CUDADevice::update_sparse_resources`: `cuMemMap` receives the tile's virtual
address (`buffer->device_address() + byte_offset`) and an offset of zero into
the one-tile physical allocation handle, rather than adding the logical byte
offset to the destination reservation while leaving the VA unoffset. The fix is
now present in the current `origin/next`; this patch is retained for older pins
and is not a separate local diff on the rebased checkout.

`luisa-cuda-tile-fp8-storage.patch`
(`src/backends/cuda/tile/cuda_tile_codegen.cpp`) FP8 STORAGE elements:
`ScalarType::FLOAT8_E4M3FN`/`FLOAT8_E5M2` map to the CUDA Tile scalar names
`__nv_fp8_e4m3`/`__nv_fp8_e5m2` in the value-type switch and in the
kernel-parameter allow-list, the compiled prologue adds `#include <cuda_fp8.h>`
(`cuda_tile.h` only forward-declares those structs), and FP8 joins FP16/BF16 in
the narrow-arithmetic rejection, so the only FP8 constructs that can be emitted
are a buffer parameter, a tile load, a `cast`/`select` and the storage side of
an MMA. FP16/BF16/FP32 behaviour, the direct-buffer ABI and the DX/VK Tile->XIR
route (which still rejects FP8 storage with an explicit diagnostic) are
unchanged. It is currently applied.

`luisa-cuda135-tile-worker-warps.patch` widens ONLY the experimental worker-warp
calibration hint: `LUISA_CUDA_TILE_WORKER_WARPS` now accepts `0, 1, 2, 4, 8, 16,
32` (`src/backends/cuda/tile/cuda_tile.cpp`) and the codegen admission in
`src/backends/cuda/tile/cuda_tile_codegen.cpp` accepts every power of two in
[1, 32] while keeping its SM89 calibration boundary, its
`[[cutile::hint(<sm>0, num_worker_warps_per_cta = N)]]` prologue and the exact
emitted source of 0/4/8. CUDA 13.5 widened `num_worker_warps_per_cta` from {4, 8}
to 1..32, so 1/2/16/32 need that toolkit: on 13.4 they are still passed straight
to NVRTC/tileiras and fail closed there with the toolkit's own diagnostic instead
of being dropped on the host (only 13.0 and 13.5 are installed on this machine,
so that 13.4 refusal is documented, not measured). Any other string is still
rejected with `requires 0, 1, 2, 4, 8, 16 or 32`. The streaming-scan arm
(`LUISA_CUDA_TILE_STREAMING_SCAN`) keeps its own 4/8-only check, so combining it
with 16/32 makes that realization fall back with its generic diagnostic instead
of being emitted. Defaults, the collective/partition/CUB cost modes, the emitted
hint text and the Qwen kernel numerics are untouched; the codegen hunk stops
before the `#include <cuda_fp8.h>` line that
`luisa-cuda-tile-fp8-storage.patch` rewrites, so the file applies both to a
pristine checkout and on top of that patch (`git apply --check` forward on
pristine, `git apply -R --check` on the current tree). It is currently applied.

Measured outcome of that widening (scratch CUDA 13.5 build,
`--prefill-gemm --format fp8 --cuda-tile-fp8 --cuda-tile-m 128 --cuda-tile-n 64`,
9 samples, RTX 4080 SUPER sm_89): the hypothesis did NOT hold. Against the
same-run native NVRTC rows the tile/native geomean over the 12 measurable rows is
2.56x at 8 worker warps (best, and a pre-existing value), 2.60x at the default 0,
2.88x at 16 and 3.28x at 32; the 64x64 tile gets worse as well (1.32x default,
1.92x at 8, 2.02x at 16). 1 and 2 are pathological (5.6x / 8.6x on the rows that
finished), and the only real effect of the larger values is compile time (12-shape
JIT 14.9 s at 0, 7.0 s at 16, 6.3 s at 32). So the widened range is a harmless,
now-available knob, NOT a fix for the slow large tile. Raw logs and the table:
`.kimix_cache/cuda135/tile_worker_warps_evidence.md` (scratch, not committed).

`luisa-cuda-tile-toolkit-env.patch` keeps `tileiras` and `nvJitLink` on the same
CUDA installation (`src/backends/cuda/tile/cuda_tile_ir.cpp`). On Windows the
backend derives the toolkit root from its configured `<toolkit>/bin/tileiras`
path and replaces `CUDA_PATH`/`CUDA_HOME` in the assembler's CHILD environment;
it never mutates the caller or system environment. A machine-wide `CUDA_PATH`
that still names an older toolkit therefore cannot mix nvJitLink into a CUDA
13.5 Tile build. It is currently applied.

Two operating limits of the native Tile route remain: (1) on Windows its
temporary directory still needs an ASCII path or an ASCII short-path alias;
(2) the toolkit rejects FP8 Tile types in PARTIAL (masked) tile loads below
sm_90 ("the '__nv_fp8_e5m2' and '__nv_fp8_e4m3' types are not supported in tile
code for architectures below sm_90"), so an FP8 Tile kernel whose weight tile
has a ragged N only compiles from sm_90 up.
`src/qwen/qwen/cuda_tile_gemm.cpp` probes the second fact at create time instead
of assuming it. The user-facing Qwen option is `example_qwen --version dsl-fp8
--cuda-tile-fp8` with `LUISA_CUDA_TILE_IR=1`; it is an opt-in experimental
prefill path, measured slower than the default NVRTC GEMM on sm_89.
