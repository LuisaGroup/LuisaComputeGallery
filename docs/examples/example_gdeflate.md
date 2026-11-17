# example_gdeflate

A GDeflate (DirectStorage) compressor and decompressor written as two Luisa kernels, exercised by a round trip, hand-built Huffman tiles and malformed-stream corner cases.

| | |
|---|---|
| Group | `compute` |
| Sources | `src/compute/compact/main.cpp`, `src/compute/compact/gdeflate.cpp`, `src/compute/compact/gdeflate_kernels.h` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_gdeflate` |
| Run | `example_gdeflate dx --headless` |

## Purpose

Compression is an unusual fit for a GPU, which is exactly why it is instructive: one
32-lane warp decodes one tile with warp votes, scans and broadcasts instead of per-thread
state. The example shows how to drive `ByteBuffer`s from a small runtime class, how to
keep a bitstream (DirectStorage's tile stream) bit-compatible with the reference, and how
to test a device codec from both ends: crafted valid input and deliberately broken input.

## Architecture

- **The codec (`gdeflate.h` / `gdeflate.cpp`).** `GDeflateCodec` owns two compiled
  shaders — `Shader1D<ByteBuffer, ByteBuffer, uint, uint, uint, uint>` for compression and
  `Shader1D<ByteBuffer, Buffer<uint>, …>` for decompression — plus `compress()`,
  `decompress()`, `allocate_compressed()` and `allocate_uncompressed()`. The allocators
  size the buffers with `compress_bound()`, so a caller never has to know the format.
- **Stream layout.** An 8-byte `TileStream` header (id `4`, magic `0xFB`, tile count,
  last-tile size), a `uint` per tile as pointer table, then the tiles — DirectStorage's
  `tilestream.hlsl` layout. A tile is 32 KiB of input, so `compute_num_tiles()` and
  `kMaxTiles = 65535` bound the input size.
- **Dispatch shapes.** Compression launches `num_tiles * 256` threads — one 256-thread
  group per tile, each thread emitting its own output-word slots. Decompression launches
  `tiles * 32`: one warp per tile, because the format interleaves bits across 32
  sub-streams so each lane owns every 32nd bit. The output is bound as a `uint` view, since
  the decoder writes bytes atomically.
- **Kernels (`gdeflate_kernels.h`).** Tile-stream readers (`c_tilestream_construct`,
  `c_tilestream_get_params`), a `BitReader`, the warp helpers (`lane_vote`,
  `lane_broadcast`, `lane_scan`, `scan16_inclusive`), Huffman tables decoded into shared
  memory (`c_read_len_codes`, `kMaxSymbols = 320` = 288 literal/length + 32 distance
  codes), and the entry points `make_compress_kernel()` / `make_decompress_kernel()`.
  The compressor emits stored (uncompressed) DEFLATE blocks, so the printed ratio is about
  1.00x by design: this measures bitstream plumbing, not compression ratio.
- **Host validation.** Before any kernel work, `decompress()` reads the 8-byte header,
  rejects an unknown id/magic and clamps the dispatch to the tile count the stream
  actually declares, so a wrong `output_size` cannot read the pointer table out of bounds.
- **What `main.cpp` runs.** A random buffer (Mersenne Twister, seed 12345) is compressed,
  decompressed, byte-compared and timed (over `repeat` iterations); a host tile writer
  (`GDeflateTileWriter`) reproduces libdeflate's 32-stream bit assignment to build three
  crafted tiles — fixed-Huffman literals, fixed-Huffman with a length/distance match, one
  dynamic-Huffman tile — which the GPU decoder must decode exactly; finally
  `run_corner_case_tests()` feeds empty, truncated, wrong-size, bad-header and
  reserved-block streams and requires a clean rejection (no hang, no device loss).

## Usage

```bash
example_gdeflate dx
example_gdeflate dx 4194304 10
example_gdeflate cuda --headless --frames 5
```

| Option | Meaning |
|---|---|
| `<backend>` | `dx`, `vk` or `cuda` |
| `[size]` | uncompressed bytes to compress (default 1 MiB; `0` restores the default) |
| `[repeat]` | timed iterations, averaged in the report (default 1; `0` restores the default) |
| `--frames N` | overrides `[repeat]`, so the shared gallery flag bounds the same loop |
| `--headless`, `--cli`, `--output-dir DIR` | accepted for the uniform gallery CLI; the example never opens a window and writes nothing |

## Result

Example console output on `dx`, abbreviated:

```text
[info] GDeflate round-trip: PASSED
[info] input size: 1048576 bytes
[info] compressed size: 1052808 bytes (1.00x)
[info] compress time: 0.0902 ms
[info] decompress time: 0.2883 ms
[info] GDeflate fixed/dynamic Huffman tile tests: PASSED
[info] compact: success (1048576 byte(s), 1 iteration(s), headless=true).
```

The `[warning]` lines for rejected streams are the decoder validating its input, not failures.

## See also

`example_software_lbvh_sort_bench` (output order and byte identity matter there too),
`example_helloworld` (the minimal single-kernel counterpart).
