# LuisaComputeGallery

The gallery of [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute): the
examples that used to live in `LuisaCompute/examples` are maintained here, with
both xmake and CMake build configurations, headless (`--headless`) CLI support and
their tests.

Every example is documented in [docs/README.md](docs/README.md): purpose,
architecture, command line, and the picture it produces.

## Layout

```
thirdparty/LuisaCompute   LuisaCompute (git submodule; or point LC_DIR elsewhere)
src/<group>/              examples: rendering, simulation, gui, compute, coro, ml,
                          extension, interop, tensor, ntc, ocr, embedding,
                          embedding_mm, dx, ios
tests/                    the tests that exercise those examples
gallery/                  GalleryRT, the runtime library shared by the gallery app
app/                      gallery.exe viewer
scripts/                  gallery_targets.json (source of truth), gen_build.py,
                          headless_smoke.py, gallery_build.lua
cmake/                    CMake helpers + local options template
docs/                     build.md, headless.md
```

## Build

```bash
# xmake (LuisaCompute from the submodule)
xmake f -y -p windows -a x64 --toolchain=msvc -m release -c
xmake build -y

# xmake against any other checkout / symlink: options.lua (git-ignored)
echo 'LC_DIR = "LuisaCompute"' > options.lua

# CMake, submodule
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
# CMake, external LuisaCompute
cmake -S . -B build -G Ninja -DGALLERY_LC_MODE=external -DGALLERY_LC_DIR=D:/ws/LuisaCompute
```

See [docs/build.md](docs/build.md) for the full option list.

## Run an example

```bash
xmake run example_path_tracing cuda --headless --spp 1 --frames 1 --output-dir out
# or, with CMake
./build/bin/example_path_tracing cuda --headless --spp 1 --frames 1

# every example that can run without a window, on any enabled backend
python scripts/headless_smoke.py --backend cuda
python scripts/headless_smoke.py --backend dx
python scripts/headless_smoke.py --backend vk
cd build && ctest -L headless --output-on-failure
```

See [docs/headless.md](docs/headless.md) for the convention and the per-group
status (which demos need model data, a vendor SDK or a second backend).

## Adding an example

Add a target to `scripts/gallery_targets.json`, run
`python scripts/gen_build.py`, and drop the sources next to the generated
`src/<group>/xmake.lua` / `CMakeLists.txt`. Both build systems are generated from
the same manifest, so they cannot drift; `python scripts/gen_build.py --check`
fails if they do.

## Save the result picture

Any example that draws something can write it to a file you name, without a
window; the extension picks the container (png, jpg/jpeg, bmp, tga):

```
example_path_tracing dx --capture cornell.jpg --spp 64
example_mpm88 cuda --capture mpm88.png --frames 8
```

See [docs/capture.md](docs/capture.md).

## Example data

The examples that need weights or textures read them from the git-ignored
`data/` directory, which is published as a release archive:

```
python scripts/unpack_data.py    # download the newest package into data/
python scripts/pack_data.py      # pack data/ and publish a new package
```

The archive is attached to the `Data` release of this repository; both scripts
need `gh` and 7-Zip. See [AGENTS.md](AGENTS.md) for the options.
