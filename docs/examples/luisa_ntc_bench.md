# luisa_ntc_bench

Measures NTC decompression throughput per mip level and the speed of the GPU BCn transcoders on a `.ntc` container.

| | |
|---|---|
| Group | `ntc` |
| Sources | `src/ntc/tools/luisa_ntc_bench.cpp`, `src/ntc/include/libntc/infer_engine.h`, `src/ntc/include/libntc/bcn_encoder.h`, `src/ntc/include/libntc/ntex_container.h` |
| Backends | dx, vk, cuda |
| Build | `xmake build luisa_ntc_bench` |
| Run | `luisa_ntc_bench --ntc=data/ntc/material.ntc --backend=dx --iters=50` |

## Purpose

Answers the two questions a renderer asks about neural texture compression: how expensive is it to reconstruct the material, and how expensive is it to transcode it into something the texture unit can sample directly. Both are measured on the GPU only — the reconstructed channels never leave VRAM during the timed loop — so the numbers are the cost of the kernels, not of a readback.

## Architecture

- Load. `NtexContainer::load_from_file` parses the NTEX v3 container; `InferEngine::load_container` uploads the latent grid and the int8/FP8 MLP weights to the device.
- Decompression sweep. For up to four mips (`min(4, numColorMips)`), `benchmark_throughput(mip, w, h, iterations)` decodes the mip `iterations` times inside the engine and returns Mtexels/s; the tool divides the texel count by that rate to print the latency of one decode.
- Transcoding sweep. `InferEngine::decompress_mip_device` produces one full-resolution FP32 buffer, then `BCnEncoder::encode_to_gpu_buffer` compresses it with `BC1`, `BC3`, `BC4`, `BC5` and `BC7` after a warm-up encode, timing `iterations` encodes and converting to ms/texture and Mtexels/s.
- Inputs. `--ntc=<path>` (or a bare `.ntc` positional) is required; `NTC_SAMPLE_NTC` in the environment is used when the flag is absent. `--frames N` is accepted as a synonym for `--iters=N`, which is handy for the gallery's uniform CLI.

## Usage

The sample container comes from `data/`, restored with `python scripts/unpack_data.py` (`data/ntc/material.ntc`, a 1280×720 four-channel material).

```bash
luisa_ntc_bench --ntc=data/ntc/material.ntc --backend=dx --iters=50
luisa_ntc_bench --ntc=data/ntc/material.ntc --backend=cuda --iters=200
luisa_ntc_bench --ntc=data/ntc/material.ntc --backend=vk --frames 10 --headless
```

| Option | Meaning |
|---|---|
| `--ntc=<file.ntc>` | container to measure; a bare `*.ntc` argument works too |
| `--backend=<name>` | `cuda` (default), `dx` or `vk` |
| `--iters=<n>` | timed iterations per measurement, default 50 |
| `--frames N` | same as `--iters=N` |
| `--headless`, `--help` | no window is ever opened; `--help` prints the usage line |

Timing is wall-clock around a stream synchronise, so the numbers include dispatch overhead; keep `--iters` high enough that the per-iteration cost dominates it.

## Result

A banner with the container shape, then the two tables. Example output of a short `--iters=2` run on `dx` (the banner itself is elided):

```text
  Resolution: 1280x720 (4 channels)
  Backend: dx | Iterations: 2
--- 1. On-Load Decompression Throughput ---
Mip Level Resolution Throughput (MT/s) Latency (ms)
0         1280x720   114.11              8.08
--- 2. Pure GPU BCn Transcoding Performance ---
BC1 (RGB) Latency: 0.233 ms/texture (3954.5 Mtexels/s)
BC7 (RGBA) Latency: 0.226 ms/texture (4086.9 Mtexels/s)
Benchmark completed successfully!
```

The first table lists up to four mips, the second all five BCn formats. Absolute values are GPU-dependent; run the tool on the machine you care about and compare like with like.

## See also

luisa_ntc_cli (trains and decompresses the container this tool measures), luisa_ntc_test (correctness of the same decode and transcode kernels), luisa_ntc_viewer (a look at the reconstructed channels).
