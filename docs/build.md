# Building the gallery

The gallery builds LuisaCompute from source. Both build systems locate it through
a single setting: `LC_DIR` for xmake, `GALLERY_LC_DIR` for CMake. By default it
is the git submodule in `thirdparty/LuisaCompute`; point it somewhere else to
build against an existing checkout.

```bash
git submodule update --init thirdparty/LuisaCompute
```

## xmake

```bash
xmake f -y -p windows -a x64 --toolchain=msvc -m release -c
xmake build -y                        # LuisaCompute + gallery + examples + tests
xmake build example_helloworld        # one target
xmake run example_helloworld cuda --headless
```

Machine-local configuration lives in `options.lua` (git-ignored, template:
`options.lua.template`), included at the very top of `xmake.lua`:

```lua
LC_DIR = "LuisaCompute"              -- a sibling clone or a symlink; default "thirdparty/LuisaCompute"
lc_options = {
    lc_cuda_backend = true,
    lc_dx_backend = true,  -- Windows
    lc_vk_backend = true,
    lc_enable_gui = true,
}
```

Every `lc_options` entry is applied with `set_config()`, so the precedence is
command line (`--lc_dx_backend=y`) > `options.lua` > the defaults in
`xmake.lua`. `xmake lua setup.lua` regenerates the file from `.env`.

Gallery-owned options:

| Option | Default | Meaning |
|---|---|---|
| `gallery_enable_tests` | true | build `tests/` (moved from LuisaCompute's test suite) |

`xmake show -l targets` lists the ~66 example targets; `xmake show -l options`
lists the LuisaCompute options.

## CMake

```bash
# 1. the submodule
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# 2. any external LuisaCompute checkout (sibling clone, CI tree, symlink)
cmake -S . -B build -G Ninja -DGALLERY_LC_MODE=external -DGALLERY_LC_DIR=D:/ws/LuisaCompute

# 3. an installed LuisaCompute SDK
cmake -S . -B build -G Ninja -DGALLERY_LC_MODE=package -DGALLERY_LC_DIR=<prefix>

cmake --build build --target example_helloworld
./build/bin/example_helloworld cuda --headless
```

`GALLERY_LC_DIR` (or the `GALLERY_LC_DIR` / `LC_DIR` environment variables)
implies `external` mode. Machine-local settings live in `cmake/options.cmake`
(git-ignored, template: `cmake/options.cmake.template`) and pre-seed cache
entries, so explicit `-D` flags win.

| Option | Default | Meaning |
|---|---|---|
| `GALLERY_LC_MODE` | `submodule` | `submodule`, `external` or `package` |
| `GALLERY_LC_DIR` | — | LuisaCompute source tree (external) or install prefix (package) |
| `GALLERY_BUILD_EXAMPLES` | ON | build `src/` |
| `GALLERY_BUILD_TESTS` | ON | build `tests/` |
| `GALLERY_BUILD_APP` | ON | build `app/gallery.exe` |
| `GALLERY_TEST_BACKEND` | `cuda` | backend used by `ctest -L headless` |
| `GALLERY_TEST_BACKENDS` | `GALLERY_TEST_BACKEND` | backends to register headless tests for, e.g. `"cuda;dx;vk"` |

LuisaCompute's own options keep their names (`-DLUISA_COMPUTE_ENABLE_CUDA=ON`,
`-DLUISA_COMPUTE_ENABLE_DX=OFF`, ...); the gallery does not shadow them.

### Backends

Every backend is built by LuisaCompute itself; the gallery only forwards the
options. What each one needs on Windows:

| Backend | xmake | CMake |
|---|---|---|
| `cuda` | CUDA toolkit (the gallery helper pulls in `lc-nvrtc`) | `-DLUISA_COMPUTE_ENABLE_CUDA=ON` |
| `dx` | downloads/extracts `dx_sdk_*.zip` into `LuisaCompute/SDKs/` and copies `dxcompiler.dll`, `dxil.dll`, `dstorage*.dll` next to the binaries | `-DLUISA_COMPUTE_ENABLE_DX=ON` plus `-DLUISA_COMPUTE_DX_SDK_DIR=<dir with those four DLLs>`, otherwise the configure downloads the SDK |
| `vk` | nothing beyond the GPU driver: the backend loads `vulkan-1.dll` through the bundled `volk`, and builds its own builtin kernels with the bundled glslang/SPIRV-Tools (`dxcompiler.dll` is only used for the optional HLSL compatibility path) | `-DLUISA_COMPUTE_ENABLE_VULKAN=ON` |
| `metal`, `metal4` | macOS only | `-DLUISA_COMPUTE_ENABLE_METAL=ON` |

Examples that need a backend only exist when it is enabled (`requires = {"dx"}`
in `scripts/gallery_targets.json`), so turning a backend on also adds its
examples and tests. This machine builds and runs `cuda`, `dx` and `vk`: see
`python scripts/headless_smoke.py --backend dx|vk`.


## Layout

```
thirdparty/LuisaCompute   git submodule (or LC_DIR -> anywhere)
src/<group>/              examples, moved out of LuisaCompute/examples
src/<group>/xmake.lua     generated from scripts/gallery_targets.json
src/<group>/CMakeLists.txt
tests/                    example-related tests, moved out of LuisaCompute/src/tests
gallery/                  GalleryRT shared library
app/                      gallery.exe viewer
scripts/gallery_targets.json   single source of truth for the example targets
scripts/gen_build.py           regenerates the per-group build files
scripts/headless_smoke.py      runs every example headless
cmake/                         CMake helpers + local options template
```

Adding or changing an example target means editing
`scripts/gallery_targets.json` and running `python scripts/gen_build.py`
(`--check` verifies the generated files are in sync). The target bodies live in
`scripts/gallery_build.lua` (xmake) and `cmake/gallery_examples.cmake` (CMake),
which are mirrors of each other.

## Headless runs

See [headless.md](headless.md): every example takes
`<example> <backend> --headless [--frames N] [--output-dir DIR]`, and
`python scripts/headless_smoke.py` runs the whole set.
