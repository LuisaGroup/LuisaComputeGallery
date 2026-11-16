-- GENERATED FILE - do not edit.
-- Source of truth: scripts/gallery_targets.json
-- Regenerate with: python scripts/gen_build.py

-- wildcards come from xmake (embed/*.cpp, embed/kernels/*.cpp); cmake lists the 20 files explicitly. xmake: lc-yyjson when bundled, add_packages('yyjson') when lc_yyjson_use_xrepo is on (make the dependency conditional on the yyjson_xrepo condition key); cmake: links luisa-compute-ext-yyjson-interface (PUBLIC). cmake sets POSITION_INDEPENDENT_CODE on UNIX so the archive can be linked into example_embed_retrieval_lib; include dirs are PUBLIC in both ($GROUP and $SRC)
gallery_example("lc-example-embedding", {kind = "static", sources = {"embed/*.cpp", "embed/kernels/*.cpp"}, deps = {"lc-yyjson"}, includes = {"$LC/src/ext/yyjson/src", "$SRC", "$GROUP"}})
-- both build files give it include dirs only - it does NOT link lc-example-embedding (it only links the luisa::compute umbrella in cmake)
gallery_example("example_embed_pack_model", {sources = {"tools/pack_model.cpp"}, includes = {"$SRC", "$GROUP"}})
-- xmake: project_kind=shared, add_files('api/*.cpp') + add_headerfiles('api/*.h'); cmake: add_library(... SHARED api/example_embed_retrieval_lib.cpp) with the api/ dir PUBLIC. The EXAMPLE_EMBED_RETRIEVAL_LIB_BUILD define is set by both
gallery_example("example_embed_retrieval_lib", {kind = "shared", sources = {"api/*.cpp"}, headers = {"api/*.h"}, deps = {"lc-example-embedding"}, includes = {"$SRC", "$GROUP", "$GROUP/api"}, defines = {"EXAMPLE_EMBED_RETRIEVAL_LIB_BUILD"}})
-- ws2_32 on windows only (xmake add_syslinks('ws2_32'), cmake target_link_libraries(... ws2_32) under if(WIN32)) - the generator should add the platform guard
gallery_example("example_embed_retrieval", {sources = {"example_embed_retrieval.cpp"}, deps = {"lc-example-embedding"}, includes = {"$SRC", "$GROUP"}, syslibs = {"ws2_32"}})
gallery_example("example_embed_dump_golden", {sources = {"tools/dump_golden.cpp"}, deps = {"lc-example-embedding"}, includes = {"$SRC", "$GROUP"}})
gallery_example("example_embed_text", {sources = {"example_embed_text.cpp"}, deps = {"lc-example-embedding"}, includes = {"$SRC", "$GROUP"}})
