-- GENERATED FILE - do not edit.
-- Source of truth: scripts/gallery_targets.json
-- Regenerate with: python scripts/gen_build.py

-- wildcards come from xmake; cmake lists the same 7 files explicitly. xmake: lc-yyjson when bundled, add_packages('yyjson') when lc_yyjson_use_xrepo is on (make the dependency conditional on the yyjson_xrepo condition key); cmake: links luisa-compute-ext-yyjson-interface (PUBLIC). cmake links luisa-compute-ext-stb-interface additionally; include dirs are PUBLIC in both
gallery_example("luisa-ntc", {kind = "static", sources = {"src/core/*.cpp", "src/infer/*.cpp", "src/bcn/*.cpp", "src/train/*.cpp"}, deps = {"lc-yyjson"}, includes = {"$GROUP/include", "$GROUP/include/libntc", "$SRC", "$LC/src/ext/yyjson/src", "$LC/src/ext/stb"}})
gallery_example("luisa_ntc_cli", {sources = {"tools/luisa_ntc_cli.cpp"}, deps = {"luisa-ntc"}, includes = {"$GROUP/include", "$GROUP/include/libntc", "$SRC", "$LC/src/ext/yyjson/src", "$LC/src/ext/stb"}})
gallery_example("luisa_ntc_bench", {sources = {"tools/luisa_ntc_bench.cpp"}, deps = {"luisa-ntc"}, includes = {"$GROUP/include", "$GROUP/include/libntc", "$SRC", "$LC/src/ext/yyjson/src", "$LC/src/ext/stb"}})
gallery_example("luisa_ntc_test", {sources = {"tools/luisa_ntc_test.cpp"}, deps = {"luisa-ntc"}, includes = {"$GROUP/include", "$GROUP/include/libntc", "$SRC", "$LC/src/ext/yyjson/src", "$LC/src/ext/stb"}})
-- built unconditionally by both build systems; when the GUI module is on it additionally links lc-gui (xmake: is_gui && lc_enable_gui; cmake: if(LUISA_COMPUTE_ENABLE_GUI)) and defines LUISA_ENABLE_GUI. This is the only ntc target with a GUI, so gui_dep is false but the conditional dep/define still has to be emitted for 'when: gui'
gallery_example("luisa_ntc_viewer", {sources = {"tools/luisa_ntc_viewer.cpp"}, deps = {"luisa-ntc"}, includes = {"$GROUP/include", "$GROUP/include/libntc", "$SRC", "$LC/src/ext/yyjson/src", "$LC/src/ext/stb"}, defines = {{value = "LUISA_ENABLE_GUI", when = "gui"}}})
