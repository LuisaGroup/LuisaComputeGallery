-- GENERATED FILE - do not edit.
-- Source of truth: scripts/gallery_targets.json
-- Regenerate with: python scripts/gen_build.py

-- wildcards come from xmake; cmake lists the 16 files explicitly. xmake: lc-yyjson when bundled, add_packages('yyjson') when lc_yyjson_use_xrepo is on (make the dependency conditional on the yyjson_xrepo condition key); cmake: links luisa-compute-ext-yyjson-interface (PUBLIC)
gallery_example("lc-example-embedding-mm", {kind = "static", sources = {"embed/*.cpp", "embed/ops/*.cpp", "embed/models/*.cpp"}, deps = {"lc-yyjson"}, includes = {"$LC/src/ext/yyjson/src", "$SRC", "$GROUP"}})
gallery_example("example_embed_mm_pack_model", {sources = {"tools/pack_model.cpp"}, deps = {"lc-example-embedding-mm"}, includes = {"$SRC", "$GROUP"}})
gallery_example("example_embed_mm_text", {sources = {"example_embed_mm_text.cpp"}, deps = {"lc-example-embedding-mm"}, includes = {"$SRC", "$GROUP"}})
gallery_example("example_embed_mm_test", {sources = {"example_embed_mm_test.cpp"}, deps = {"lc-example-embedding-mm"}, includes = {"$SRC", "$GROUP"}})
gallery_example("example_embed_mm_vision", {sources = {"example_embed_mm_vision.cpp"}, deps = {"lc-example-embedding-mm"}, includes = {"$SRC", "$GROUP"}})
gallery_example("example_embed_mm_audio", {sources = {"example_embed_mm_audio.cpp"}, deps = {"lc-example-embedding-mm"}, includes = {"$SRC", "$GROUP"}})
gallery_example("example_embed_mm_video", {sources = {"example_embed_mm_video.cpp"}, deps = {"lc-example-embedding-mm"}, includes = {"$SRC", "$GROUP"}})
gallery_example("example_embed_mm_bench", {sources = {"example_embed_mm_bench.cpp"}, deps = {"lc-example-embedding-mm"}, includes = {"$SRC", "$GROUP"}})
