# example_dstorage

Streams a scratch file and a pinned host range into device buffers and into a 2D image through the DirectStorage extension, then compares every byte.

| | |
|---|---|
| Group | `extension` |
| Sources | `src/extension/dstorage.cpp`, `src/common/headless.h` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_dstorage` |
| Run | `example_dstorage dx --headless` |

## Purpose

`DStorageExt` is the "get bytes to the GPU without going through the host upload
path" extension, and it has four very different implementations: direct storage
queues on `dx` (`IDStorageQueue2`), a host-staged fallback on an internal `COPY`
stream on `vk`, file mapping plus pinned memory on `cuda`, MTLIO on `metal`. This
demo runs the same three transfers on whichever backend it is given and proves
they landed byte for byte, so it doubles as a capability probe.

## Architecture

- **Extension query.** `device.extension<DStorageExt>()`. A backend without the
  extension logs `does not provide DStorageExt; nothing to do.` and exits 0, so the
  demo is safe to run anywhere. Otherwise it prints what
  `supports_compression(DStorageCompression::None)` and `...::GDeflate` report.
- **Stream.** `DStorageStreamOption` asks for `source = AnySource` (a queue that
  serves both file- and memory-sourced reads) and a 4 MiB `staging_buffer_size`;
  `create_stream(option)` returns a `Stream` tagged `StreamTag::CUSTOM`. A null
  stream — no DirectStorage runtime — is a warning and a clean exit 0, not an
  error.
- **Two streams on purpose.** Only `copy_to` commands go through the dstorage
  stream; a plain `device.create_stream()` does every readback
  (`buffer.copy_to(luisa::span)`), so the round trip never runs through the
  extension's own queue.
- **File → buffer.** A 1 MiB file of `pattern_byte(i)` noise is written to a
  scratch path, opened with `open_file(path.string())` (a `DStorageFile`) and read
  with `dstorage_stream << file.copy_to(buffer) << synchronize()`. The result is
  compared with `bytes_equal`, i.e. `memcmp`.
- **File → image.** The source follows the texture layout:
  `dstorage_texture_row_pitch(PixelStorage::BYTE4, 100u)` rounds the 400-byte row up
  to 512 bytes (`D3D12_TEXTURE_DATA_PITCH_ALIGNMENT`), so the 100×32 file holds
  padded rows while the image must end up tightly packed — the check compares
  `image.view().size_bytes()` against the tight reference. The `vk` fallback
  repacks the 256-byte pitch into tight rows itself.
- **Pinned memory → buffer.** `pin_memory(ptr, size)` on a 64 KiB host vector
  yields a `DStorageFile` with the pinned tag, and the same `copy_to(buffer)` runs
  straight from page-locked memory.
- **Lifetime rule.** A `DStorageReadCommand` carries only an opaque handle, so the
  `DStorageFile` must outlive every read it produced; the example keeps it alive
  until the stream is synchronized, which is the pattern to copy.

## Usage

```bash
example_dstorage dx # window-less already; the whole demo is the console report
example_dstorage vk --headless
```

| Option | Meaning |
|---|---|
| `<backend>` | `dx`, `vk`, `cuda` (or `metal`): the backend whose `DStorageExt` is exercised |
| `--headless`, `--cli` | no window; accepted for the uniform gallery CLI — the example never opens one |
| `--frames N`, `--output-dir DIR` | consumed by the shared parser and unused: the work is three transfers |

Without a backend argument it logs its usage line and exits with 1.

## Result

Example console output on `dx`:

```text
[info] Direct storage on 'dx': None supported = true, GDeflate supported = true.
[info] [file -> buffer] 1048576 byte(s): OK.
[info] [file -> image 100x32 (512 B row pitch)]: OK.
[info] [pinned memory -> buffer] 65536 byte(s): OK.
[info] Direct-storage demo finished in 17.6846 ms with 0 failure(s).
[info] dstorage: PASSED (backend='dx', headless=true).
```

Any mismatch turns one line into `FAILED`, counts in `failure(s)` and makes the
program exit 1. On `vk` and `cuda` the first line reports
`GDeflate supported = false` — neither has a device-side decompressor.

## See also

`example_dml` (the other `extension` example), `example_gdeflate` (a GDeflate
compressor and decompressor written as Luisa kernels).
