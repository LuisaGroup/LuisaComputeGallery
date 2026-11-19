# GENERATED FILE - do not edit.
# Source of truth: scripts/gallery_targets.json
# Regenerate with: python scripts/gen_build.py

# Qwen3.5-4B text-only engine. Reuses the OCR tokenizer/detokenizer/status sources (compiled here, not linked from lc-example-ocr, to keep the dependency surface small); needs lc-tile for the Tile executor and lc-yyjson for the GGUF->tokenizer.json construction. QwenTokenizer is allocated/freed through Luisa's allocator with a custom deleter (luisa::delete_with_allocator), so it does not depend on EASTL's default_delete allocator configuration propagating through the static-lib dep chain. Includes: $GROUP for qwen/*.h, $SRC/ocr for ocr/*.h, yyjson headers.
gallery_target(lc-example-qwen
    KIND static
    SOURCES "qwen/*.cpp" "../ocr/ocr/tokenizer.cpp" "../ocr/ocr/detokenizer.cpp" "../ocr/ocr/status.cpp"
    DEPS lc-tile lc-yyjson
    INCLUDES "$GROUP" "$SRC" "$SRC/ocr" "$LC/src/ext/yyjson/src"
    NO_HEADLESS_TEST
)
gallery_target(example_qwen
    SOURCES "example_qwen.cpp"
    DEPS lc-example-qwen
    INCLUDES "$GROUP" "$SRC" "$SRC/ocr"
    HEADLESS
    HEADLESS_ARGS --list-versions
)
