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

