# GENERATED FILE - do not edit.
# Source of truth: scripts/gallery_targets.json
# Regenerate with: python scripts/gen_build.py

# wildcards come from xmake (embed/*.cpp, embed/kernels/*.cpp); cmake lists the 20 files explicitly. xmake: lc-yyjson when bundled, add_packages('yyjson') when lc_yyjson_use_xrepo is on (make the dependency conditional on the yyjson_xrepo condition key); cmake: links luisa-compute-ext-yyjson-interface (PUBLIC). cmake sets POSITION_INDEPENDENT_CODE on UNIX so the archive can be linked into example_embed_retrieval_lib; include dirs are PUBLIC in both ($GROUP and $SRC)
gallery_target(lc-example-embedding
    KIND static
    SOURCES "embed/*.cpp" "embed/kernels/*.cpp"
    DEPS lc-yyjson
    INCLUDES "$LC/src/ext/yyjson/src" "$SRC" "$GROUP"
    NO_HEADLESS_TEST
)
# both build files give it include dirs only - it does NOT link lc-example-embedding (it only links the luisa::compute umbrella in cmake)
gallery_target(example_embed_pack_model
    SOURCES "tools/pack_model.cpp"
    INCLUDES "$SRC" "$GROUP"
    NO_HEADLESS_TEST
)
# xmake: project_kind=shared, add_files('api/*.cpp') + add_headerfiles('api/*.h'); cmake: add_library(... SHARED api/example_embed_retrieval_lib.cpp) with the api/ dir PUBLIC. The EXAMPLE_EMBED_RETRIEVAL_LIB_BUILD define is set by both
gallery_target(example_embed_retrieval_lib
    KIND shared
    SOURCES "api/*.cpp"
    HEADERS "api/*.h"
    DEPS lc-example-embedding
    INCLUDES "$SRC" "$GROUP" "$GROUP/api"
    DEFINES EXAMPLE_EMBED_RETRIEVAL_LIB_BUILD
    NO_HEADLESS_TEST
)
# ws2_32 on windows only (xmake add_syslinks('ws2_32'), cmake target_link_libraries(... ws2_32) under if(WIN32)) - the generator should add the platform guard
gallery_target(example_embed_retrieval
    SOURCES "example_embed_retrieval.cpp"
    DEPS lc-example-embedding
    INCLUDES "$SRC" "$GROUP"
    LIBS ws2_32
    HEADLESS
    HEADLESS_ARGS --model $ROOT/data/bge-m3/model.bgm --tokenizer $ROOT/data/bge-m3/tokenizer.json
)
gallery_target(example_embed_dump_golden
    SOURCES "tools/dump_golden.cpp"
    DEPS lc-example-embedding
    INCLUDES "$SRC" "$GROUP"
    HEADLESS
    HEADLESS_ARGS $ROOT/data/bge-m3/model.bgm $ROOT/data/bge-m3/tokenizer.json hello --layers 0,1
)
gallery_target(example_embed_text
    SOURCES "example_embed_text.cpp"
    DEPS lc-example-embedding
    INCLUDES "$SRC" "$GROUP"
    HEADLESS
    HEADLESS_ARGS $ROOT/data/bge-m3/model.bgm $ROOT/data/bge-m3/tokenizer.json A photograph of a red apple on a wooden table
)
