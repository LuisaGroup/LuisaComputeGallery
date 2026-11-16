# GENERATED FILE - do not edit.
# Source of truth: scripts/gallery_targets.json
# Regenerate with: python scripts/gen_build.py

# wildcards come from xmake; cmake lists the 16 files explicitly. xmake: lc-yyjson when bundled, add_packages('yyjson') when lc_yyjson_use_xrepo is on (make the dependency conditional on the yyjson_xrepo condition key); cmake: links luisa-compute-ext-yyjson-interface (PUBLIC)
gallery_target(lc-example-embedding-mm
    KIND static
    SOURCES "embed/*.cpp" "embed/ops/*.cpp" "embed/models/*.cpp"
    DEPS lc-yyjson
    INCLUDES "$LC/src/ext/yyjson/src" "$SRC" "$GROUP"
    NO_HEADLESS_TEST
)
gallery_target(example_embed_mm_pack_model
    SOURCES "tools/pack_model.cpp"
    DEPS lc-example-embedding-mm
    INCLUDES "$SRC" "$GROUP"
    NO_HEADLESS_TEST
)
gallery_target(example_embed_mm_text
    SOURCES "example_embed_mm_text.cpp"
    DEPS lc-example-embedding-mm
    INCLUDES "$SRC" "$GROUP"
    NO_HEADLESS_TEST
)
gallery_target(example_embed_mm_test
    SOURCES "example_embed_mm_test.cpp"
    DEPS lc-example-embedding-mm
    INCLUDES "$SRC" "$GROUP"
    HEADLESS
)
gallery_target(example_embed_mm_vision
    SOURCES "example_embed_mm_vision.cpp"
    DEPS lc-example-embedding-mm
    INCLUDES "$SRC" "$GROUP"
    NO_HEADLESS_TEST
)
gallery_target(example_embed_mm_audio
    SOURCES "example_embed_mm_audio.cpp"
    DEPS lc-example-embedding-mm
    INCLUDES "$SRC" "$GROUP"
    NO_HEADLESS_TEST
)
gallery_target(example_embed_mm_video
    SOURCES "example_embed_mm_video.cpp"
    DEPS lc-example-embedding-mm
    INCLUDES "$SRC" "$GROUP"
    NO_HEADLESS_TEST
)
gallery_target(example_embed_mm_bench
    SOURCES "example_embed_mm_bench.cpp"
    DEPS lc-example-embedding-mm
    INCLUDES "$SRC" "$GROUP"
    NO_HEADLESS_TEST
)
