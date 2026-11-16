# Gallery tests

These tests came here from `LuisaCompute/src/tests` together with the examples they
exercise (see `src/`): they include the example headers and their data paths are
compiled in, so they are only meaningful next to the moved sources.

## Layout

| Directory | Contents |
| --- | --- |
| `unit/core/` | host-only tests for the plan/option helpers in `src/common` |
| `unit/embedding/`, `integration/embedding/` | bge-m3 embedding engine (`src/embedding`) |
| `unit/ocr/`, `integration/ocr/` | OCR engine (`src/ocr`) |
| `benchmark/` | embedding benchmarks (built, not part of the test suite) |
| `ios/` | Apple/Metal4 iOS tests, **not wired** into this build |
| `data/` | local fixture prerequisites (not committed), see `data/README.md` |

## Build and run

xmake (targets are declared in `tests/xmake.lua`, enabled by `--gallery_enable_tests=y`):

```sh
xmake f --gallery_enable_tests=y ...      # any backend configuration
xmake build test_embed_tokenizer           # one target
xmake run test_ocr_ops_gemm vk             # device tests take the backend as argv[1]
```

CMake (targets + CTest entries are declared in `tests/CMakeLists.txt`,
`-DGALLERY_BUILD_TESTS=ON`, the default):

```sh
cmake -S . -B build -G Ninja
cmake --build build --target test_embed_tokenizer
ctest --test-dir build -L unit_ocr
```

The CTest labels are unchanged from LuisaCompute (`unit;unit_core`,
`unit;unit_embedding`, `integration;integration_embedding`, `unit;unit_ocr`,
`integration;integration_ocr`). Device tests take the backend as their first
argument; the embedding device tests use `LUISA_EMBED_DEVICE_BACKEND`
(default `fallback`), the OCR device tests only get a CTest entry when
`LUISA_OCR_DEVICE_BACKEND` names one.

## Fixtures

`unit/embedding`, `unit/ocr` and the device tests read model/golden data from
`tests/data/embedding` and `tests/data/ocr`. Those directories are local
prerequisites and are not part of the repository - see `data/README.md`.

## ios/

`ios/` holds the Apple-only Metal4 device-conformance and path-tracing tests and
is **not** wired into the gallery build: it needs the Apple toolchain (Xcode
clang with the Metal4 AIR backend) and, for the signed application, a physical
device. Its `CMakeLists.txt` is kept as-is from LuisaCompute for reference - it
still points at LuisaCompute paths (`src/backends/metal4`,
`examples/ios/metal4_path_tracing/Target.cmake`) and is not included by this
directory's build files.
