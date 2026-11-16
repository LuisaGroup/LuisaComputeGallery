-- GENERATED FILE - do not edit.
-- Source of truth: scripts/gallery_targets.json
-- Regenerate with: python scripts/gen_build.py

-- wildcards come from xmake; cmake lists ~90 files explicitly. xmake: lc-yyjson when bundled, add_packages('yyjson') when lc_yyjson_use_xrepo is on (make the dependency conditional on the yyjson_xrepo condition key); cmake: links luisa-compute-ext-yyjson-interface (PUBLIC). cmake also links luisa-compute-ext-stb (and luisa-compute-ext-yyjson-interface when that target exists); xmake relies on the stb-image dep. PUBLIC include dirs: $GROUP and $SRC
gallery_example("lc-example-ocr", {kind = "static", sources = {"ocr/*.cpp", "ocr/output/*.cpp", "ocr/runtime/*.cpp", "ocr/ops/*.cpp", "ocr/models/*.cpp"}, deps = {"lc-yyjson"}, includes = {"$LC/src/ext/yyjson/src", "$SRC", "$GROUP"}})
gallery_example("example_ocr", {sources = {"example_ocr.cpp"}, deps = {"lc-example-ocr"}, includes = {"$SRC", "$GROUP"}})
-- xmake: add_files('bench/*.cpp'), includes src/tests + src/tests/integration/ocr privately, and defines LUISA_OCR_BENCHMARK_SOURCE_REVISION from $LUISA_OCR_SOURCE_REVISION (default 'unknown', whitespace stripped) plus absolute LUISA_OCR_TEST_DATA_DIR / LUISA_OCR_EXAMPLES_DIR paths. cmake takes the revision from `git rev-parse HEAD` and the same absolute paths. The three defines must be regenerated from the new root; the JSON shows the token form and a safe default
gallery_example("lc-example-ocr-bench", {kind = "static", sources = {"bench/*.cpp"}, deps = {"lc-example-ocr"}, includes = {"$GROUP", "$TESTS/integration/ocr"}, defines = {"LUISA_OCR_BENCHMARK_SOURCE_REVISION=unknown", "LUISA_OCR_TEST_DATA_DIR=\"$TESTS/data/ocr\"", "LUISA_OCR_EXAMPLES_DIR=\"$SRC\""}})
gallery_example("example_ocr_bench", {sources = {"example_ocr_bench.cpp"}, deps = {"lc-example-ocr-bench"}, includes = {"$GROUP"}})
