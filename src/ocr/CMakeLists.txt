# GENERATED FILE - do not edit.
# Source of truth: scripts/gallery_targets.json
# Regenerate with: python scripts/gen_build.py

# wildcards come from xmake; cmake lists ~90 files explicitly. xmake: lc-yyjson when bundled, add_packages('yyjson') when lc_yyjson_use_xrepo is on (make the dependency conditional on the yyjson_xrepo condition key); cmake: links luisa-compute-ext-yyjson-interface (PUBLIC). cmake also links luisa-compute-ext-stb (and luisa-compute-ext-yyjson-interface when that target exists); xmake relies on the stb-image dep. PUBLIC include dirs: $GROUP and $SRC
gallery_target(lc-example-ocr
    KIND static
    SOURCES "ocr/*.cpp" "ocr/output/*.cpp" "ocr/runtime/*.cpp" "ocr/ops/*.cpp" "ocr/models/*.cpp"
    DEPS lc-yyjson
    INCLUDES "$LC/src/ext/yyjson/src" "$SRC" "$GROUP"
    NO_HEADLESS_TEST
)
gallery_target(example_ocr
    SOURCES "example_ocr.cpp"
    DEPS lc-example-ocr
    INCLUDES "$SRC" "$GROUP"
    HEADLESS
    HEADLESS_ARGS --list-models
)
# xmake: add_files('bench/*.cpp'), includes src/tests + src/tests/integration/ocr privately, and defines LUISA_OCR_BENCHMARK_SOURCE_REVISION from $LUISA_OCR_SOURCE_REVISION (default 'unknown', whitespace stripped) plus absolute LUISA_OCR_TEST_DATA_DIR / LUISA_OCR_EXAMPLES_DIR paths. cmake takes the revision from `git rev-parse HEAD` and the same absolute paths. The three defines must be regenerated from the new root; the JSON shows the token form and a safe default
gallery_target(lc-example-ocr-bench
    KIND static
    SOURCES "bench/*.cpp"
    DEPS lc-example-ocr
    INCLUDES "$GROUP" "$TESTS/integration/ocr"
    DEFINES LUISA_OCR_BENCHMARK_SOURCE_REVISION=unknown LUISA_OCR_TEST_DATA_DIR="$TESTS/data/ocr" LUISA_OCR_EXAMPLES_DIR="$SRC"
    NO_HEADLESS_TEST
)
gallery_target(example_ocr_bench
    SOURCES "example_ocr_bench.cpp"
    DEPS lc-example-ocr-bench
    INCLUDES "$GROUP"
    HEADLESS
    HEADLESS_ARGS ops
)
