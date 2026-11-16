# GENERATED FILE - do not edit.
# Source of truth: scripts/gallery_targets.json
# Regenerate with: python scripts/gen_build.py

# wildcards come from xmake; cmake lists the same 7 files explicitly. xmake: lc-yyjson when bundled, add_packages('yyjson') when lc_yyjson_use_xrepo is on (make the dependency conditional on the yyjson_xrepo condition key); cmake: links luisa-compute-ext-yyjson-interface (PUBLIC). cmake links luisa-compute-ext-stb-interface additionally; include dirs are PUBLIC in both
gallery_target(luisa-ntc
    KIND static
    SOURCES "src/core/*.cpp" "src/infer/*.cpp" "src/bcn/*.cpp" "src/train/*.cpp"
    DEPS lc-yyjson
    INCLUDES "$GROUP/include" "$GROUP/include/libntc" "$SRC" "$LC/src/ext/yyjson/src" "$LC/src/ext/stb"
    NO_HEADLESS_TEST
)
gallery_target(luisa_ntc_cli
    SOURCES "tools/luisa_ntc_cli.cpp"
    DEPS luisa-ntc
    INCLUDES "$GROUP/include" "$GROUP/include/libntc" "$SRC" "$LC/src/ext/yyjson/src" "$LC/src/ext/stb"
    NO_HEADLESS_TEST
)
gallery_target(luisa_ntc_bench
    SOURCES "tools/luisa_ntc_bench.cpp"
    DEPS luisa-ntc
    INCLUDES "$GROUP/include" "$GROUP/include/libntc" "$SRC" "$LC/src/ext/yyjson/src" "$LC/src/ext/stb"
    NO_HEADLESS_TEST
)
gallery_target(luisa_ntc_test
    SOURCES "tools/luisa_ntc_test.cpp"
    DEPS luisa-ntc
    INCLUDES "$GROUP/include" "$GROUP/include/libntc" "$SRC" "$LC/src/ext/yyjson/src" "$LC/src/ext/stb"
    NO_HEADLESS_TEST
)
# built unconditionally by both build systems; when the GUI module is on it additionally links lc-gui (xmake: is_gui && lc_enable_gui; cmake: if(LUISA_COMPUTE_ENABLE_GUI)) and defines LUISA_ENABLE_GUI. This is the only ntc target with a GUI, so gui_dep is false but the conditional dep/define still has to be emitted for 'when: gui'
gallery_target(luisa_ntc_viewer
    SOURCES "tools/luisa_ntc_viewer.cpp"
    DEPS luisa-ntc
    INCLUDES "$GROUP/include" "$GROUP/include/libntc" "$SRC" "$LC/src/ext/yyjson/src" "$LC/src/ext/stb"
    COND_DEFINES "gui:LUISA_ENABLE_GUI"
    NO_HEADLESS_TEST
)
