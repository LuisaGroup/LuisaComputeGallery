# luisa_ntc_test

Four-test verification suite for NTC: container round-trip, decode parity against reference images, BCn transcoding and a short training run with QAT export.

| | |
|---|---|
| Group | `ntc` |
| Sources | `src/ntc/tools/luisa_ntc_test.cpp`, `src/ntc/include/libntc/ntex_container.h`, `src/ntc/include/libntc/infer_engine.h`, `src/ntc/include/libntc/bcn_encoder.h`, `src/ntc/include/libntc/trainer.h` |
| Backends | dx, vk, cuda |
| Build | `xmake build luisa_ntc_test` |
| Run | `luisa_ntc_test dx --ntc=data/ntc/material.ntc` |

## Purpose

The self-check of the whole NTC stack, including binary compatibility with NVIDIA's `ntc-cli`. It proves that a container written by this library describes itself correctly, that the inference engine reconstructs the same pixels as the official decoder, that every BCn encoder produces buffers of the exact expected size, and that the trainer can train and export a container end to end.

## Architecture

- Test 1, container round-trip. Loads the `.ntc`, asserts the container is self-describing (non-zero resolution, channels and latent features, at least one MLP version — the reference package carries the INT8 and FP8 pair), re-saves it as `test_roundtrip.ntc`, and, when `--official-cli` points at the official executable, runs `ntc-cli --describe` on the copy so an external parser has to accept it.
- Test 2, decode parity. `InferEngine::decompress_all_textures` reconstructs every texture of mip 0 and compares each against `<name>.png` in `--ref-dir` (the official decompression of the same package), requiring more than 50 dB PSNR. Without a reference directory the decode still runs and the comparison is reported as skipped.
- Test 3, BCn transcoding. Encodes the base mip to `BC1`, `BC3`, `BC4`, `BC5` and `BC7` buffers and asserts the exact block-aligned byte sizes, then writes `test_bc3.dds` and `test_bc7.dds` and checks the files are non-empty.
- Test 4, training and QAT export. Loads a manifest (default `data/ntc/material_manifest.json`), trains 200 steps with 12 latent features, saves `test_trained.ntc`, and — again with `--official-cli` — asks the official tool to describe the freshly trained file. Temporary `.ntc` and `.dds` files are deleted afterwards.
- Sample lookup. The container path comes from `--ntc=`, else `NTC_SAMPLE_NTC`, else the first `data/ntc/material.ntc` found next to or up to four levels above the executable; the manifest from `NTC_SAMPLE_MANIFEST`, else `data/ntc/material_manifest.json`; the official CLI from `--official-cli=` or `NTC_OFFICIAL_CLI`; the reference images from `--ref-dir=` or `NTC_REF_DIR`.

## Usage

The sample container and manifest come from `data/`, restored with `python scripts/unpack_data.py`
(`data/ntc/material.ntc` and `data/ntc/material_manifest.json`, trained from `data/ntc/input.png`).

```bash
luisa_ntc_test dx --ntc=data/ntc/material.ntc
luisa_ntc_test vk --ntc=data/ntc/material.ntc --headless
luisa_ntc_test cuda --ntc=data/ntc/material.ntc --ref-dir=official_png --official-cli=ntc-cli
```

| Argument | Meaning |
|---|---|
| `[backend]` | `cuda` (default), `dx` or `vk`; the only positional |
| `--ntc=<file.ntc>` | container to verify |
| `--official-cli=<path>` | official `ntc-cli` executable; enables the external describe checks in tests 1 and 4 |
| `--ref-dir=<dir>` | directory of official decompressed PNGs; enables the PSNR comparison in test 2 |
| `NTC_SAMPLE_NTC`, `NTC_SAMPLE_MANIFEST`, `NTC_OFFICIAL_CLI`, `NTC_REF_DIR` | environment overrides for the same inputs |
| `--headless` | no window is ever opened; keeps the gallery CLI uniform |

Only test 4 is slow: it is a real training run of 200 steps on the sample material, roughly two minutes at 1280×720.

## Result

Four blocks and a verdict; test 2 reports either PSNR per texture or the skipped comparison:

```text
[Test 1] Testing NTEX v3 Container Deserialization & Serialization...
 Sample: data/ntc/material.ntc (1280x720, 4 channel(s), 16 latent feature(s), 2 MLP version(s))
[Test 1 PASSED]
[Test 2] Testing Inference Engine Numerical Parity...
 -> 1 texture(s) decompressed; parity check skipped (set NTC_REF_DIR to the official reference PNGs to enable it)
[Test 4] Testing Training Engine & QAT Export...
[info] Step 200/200: Real MSE = 0.000291, PSNR = 35.36 dB, Elapsed = 106.41s
  ALL 4 VERIFICATION TESTS PASSED PERFECTLY!
```

A failed assertion aborts the run with the failing condition and a source location, and the process exits non-zero.

## See also

luisa_ntc_cli (the same container and training paths as tools), luisa_ntc_bench (throughput of the decode and transcode kernels), luisa_ntc_viewer (visual check of the decoded channels).
