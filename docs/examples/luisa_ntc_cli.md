# luisa_ntc_cli

Compress a material into an NTEX v3 neural texture, inspect an existing `.ntc` container, or decompress one back to PNG or BCn DDS — the training and conversion half of the NTC pipeline.

| | |
|---|---|
| Group | `ntc` |
| Sources | `src/ntc/tools/luisa_ntc_cli.cpp`, `src/ntc/include/libntc/manifest.h`, `src/ntc/include/libntc/trainer.h`, `src/ntc/include/libntc/infer_engine.h`, `src/ntc/include/libntc/ntex_container.h`, `src/ntc/include/libntc/bcn_encoder.h` |
| Backends | dx, vk, cuda |
| Build | `xmake build luisa_ntc_cli` |
| Run | `luisa_ntc_cli --describe data/ntc/material.ntc` |

## Purpose

The pipeline in one executable: a JSON manifest lists the textures of a material, `-c` trains a latent feature grid plus a small MLP on the GPU and exports the quantised result into an NTEX v3 container, `--describe` reads that container back without touching a device, and `-D` runs the inference engine to reconstruct the channels as images or block-compressed textures.

## Architecture

- Manifest (`libntc/manifest.h`). `MaterialManifest::load` parses a JSON with a `textures` array; each entry has `fileName`, `name`, `isSRGB`, an optional `bcFormat` (`BC1`, `BC3`, `BC4`, `BC5`, `BC7`) and arbitrary `semantics`. The loader decodes the images with stb, packs the channels of all textures into one flattened reference buffer and generates the mip chain (`numMips`), because training is fitted per mip.
- Training (`-c`). `Trainer(device, stream, TrainingSettings)` fits the latent grid and the decoder MLP against that reference with analytical gradients and a fused Adam optimiser, entirely in C++ kernels — no Python or PyTorch in the loop. `export_container` then quantises the MLP into int8 and FP8 versions (QAT) and writes width, height, channel count, latent shape and per-texture descriptors. Progress is logged every `stepsPerIteration` steps (1000 by default) with MSE, PSNR and elapsed time.
- Container (`--describe`). `NtexContainer::load_from_file` parses the self-describing NTEX v3 header: dimensions, channel count, mip count, latent shape computed from `gridSizeScale`/`numFeatures`, the inference weight types present, and per texture the channel range, channel format, BCn format and colour space. No device is created for this action.
- Decompression (`-D`). `InferEngine::load_container` uploads the grid and the MLP weights, `decompress_mip_device` reconstructs full FP32 channels on the GPU, and `BCnEncoder` either transcodes them into `BC1`/`BC3`/`BC4`/`BC5`/`BC7` blocks and saves DDS files, or — with `-B none` — writes plain UNORM8 PNGs through the gallery `image_io` helper. Only the base mip is written.
- Defaults. `-c` without `-o` writes `output.ntc` next to the manifest; `-D` without `-i` writes into a `decompressed` directory next to the container; `-S`/`--trainingSteps` defaults to 10000.

## Usage

The sample container and its source come from `data/`, restored with `python scripts/unpack_data.py`
(`data/ntc/material.ntc`, plus `data/ntc/material_manifest.json` and the `input.png` it trains on).

```bash
luisa_ntc_cli --describe data/ntc/material.ntc
luisa_ntc_cli -c --loadManifest=data/ntc/material_manifest.json -o material.ntc --backend=dx --trainingSteps=10000
luisa_ntc_cli -D --loadCompressed=data/ntc/material.ntc -i out/png -B none --backend=dx
luisa_ntc_cli -D --loadCompressed=data/ntc/material.ntc -i out/dds --backend=dx
```

| Option | Meaning |
|---|---|
| `-d`, `--describe` | print the container summary; also the implicit action when only a `.ntc` path is given |
| `-c`, `--compress` | train from a manifest and write a container; the implicit action when only a `.json` path is given |
| `-D`, `--decompress` | reconstruct the base mip and write images |
| `--backend=<name>` | backend plugin, default `cuda`; not needed for `--describe` |
| `--loadManifest=<json>` | material manifest (`textures` array, see Architecture) |
| `--loadCompressed=<file.ntc>` | input container (a bare `*.ntc` positional works too) |
| `-o <path>`, `-o=<path>`, `--saveCompressed=<path>` | output container written by `-c` |
| `-i <dir>`, `-i=<dir>` | output directory for decompressed images |
| `-B <fmt>`, `-B=<fmt>` | `none` writes plain PNG instead of DDS; otherwise the container's own BCn format is used (`BC7` when it declares none) |
| `-S <n>`, `-S=<n>`, `--trainingSteps=<n>` | training steps, default 10000; this is the dominant cost — at 1280×720 a step takes about half a second |
| `--headless` | accepted for the uniform gallery CLI; this tool never opens a window |

## Result

`--describe` on the sample container:

```text
Dimensions: 1280x720, 4 channels, 11 mip level(s)
Base compression rate: --bitsPerPixel 5.000
Latent shape: --gridSizeScale 4 --numFeatures 16
Inference weights: Int8 [Y], FP8 [Y]
Latent compression: None
Textures:
0: albedo
```

followed by one block per texture with its channel range, channel format, BCn format and colour space. `-c` reports progress and the file it wrote:

```text
[info] Starting Real GPU NTC Training: 200 steps, 4 channels, resolution 1280x720
[info] Step 200/200: Real MSE = 0.000373, PSNR = 34.29 dB, Elapsed = 95.92s
[info] Saved compressed NTC package: material.ntc (632176 bytes)
```

`-D` writes one line per texture, e.g. `Saved image 'out/png/albedo.png': 1280x720 pixels, 4 channels, UNORM8.` for `-B none`, or `... albedo.dds: 1280x720 pixels, 1 mips, BCn.` for the transcoding path.

## See also

luisa_ntc_bench (decode throughput and BCn transcode speed), luisa_ntc_test (container round-trip, parity and training checks), luisa_ntc_viewer (interactive and offscreen preview of a container).
