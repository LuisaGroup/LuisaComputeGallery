# Test fixtures (`tests/data/`)

`tests/data/embedding/` and `tests/data/ocr/` are **local fixture prerequisites**:
they hold the model/golden data the moved tests read (tokenizer models, packed
bge-m3 / OCR weights, safetensors slices, reference JSON, image goldens, ...).

They are **not committed** and are not created by the build:

- they are large binary blobs that belong to a data distribution, not to a source
  tree;
- the identical directories were git-ignored in LuisaCompute
  (`.gitignore: /src/tests/data/`), so they were never part of that repository
  either - a fresh checkout of LuisaCompute did not have them.

Create them (or symlink an existing copy) before running the affected tests:

```text
tests/data/embedding/    <- the former LuisaCompute/src/tests/data/embedding
tests/data/ocr/          <- the former LuisaCompute/src/tests/data/ocr
```

## Which tests need them

The paths are compiled into the binaries:

| Define | Value |
| --- | --- |
| `LUISA_EMBED_TEST_DATA_DIR` | `<repo>/tests/data/embedding` |
| `LUISA_OCR_TEST_DATA_DIR` | `<repo>/tests/data/ocr` |
| `LUISA_OCR_EXAMPLES_DIR` | `<repo>/src` (example assets, not a fixture) |

The host-only tests (`test_embed_tokenizer*`, `test_embed_chunker`,
`test_embed_package`, `test_embed_pack`, `test_embed_retrieval_*`,
`test_ocr_*` unit tests) and the device tests all consult that directory. Some of
them skip when a fixture is missing, others fail with a "cannot open"/"missing
golden" error - either way a green run requires the data.

`tests/data/` intentionally contains nothing but this README: keep it that way so
that `git status` stays clean on a machine that has the fixtures checked out.
