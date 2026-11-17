# GENERATED FILE - do not edit.
# Source of truth: scripts/gallery_targets.json
# Regenerate with: python scripts/gen_build.py

gallery_target(example_helloworld
    SOURCES "helloworld.cpp"
    MIRROR_AS_TEST
    HEADLESS
)
# xmake + cmake (examples/CMakeLists.txt line 150)
gallery_target(example_bindless_mip
    SOURCES "bindless_mip.cpp"
    HEADLESS
)
gallery_target(example_cluster_launch_control
    SOURCES "cluster_launch_control.cpp"
    MIRROR_AS_TEST
    NO_HEADLESS_TEST
)
gallery_target(example_async_copy_prefetch
    SOURCES "async_copy_prefetch.cpp"
    MIRROR_AS_TEST
    HEADLESS
    SKIP_BACKENDS dx vk
)
# the 'lbvh/*.cpp' wildcard comes from xmake; cmake lists lbvh_storage.cpp, lbvh_sort.cpp, blas.cpp, tlas.cpp, software_lbvh.cpp explicitly. The LBVH sources are shared by the bench/test targets
gallery_target(example_software_lbvh
    SOURCES "software_lbvh.cpp" "lbvh/*.cpp"
    MIRROR_AS_TEST
    HEADLESS
)
# wildcards taken from xmake; cmake lists the same files explicitly. Deliberately NOT mirrored as a test (it reports numbers instead of asserting)
gallery_target(example_software_lbvh_bench
    SOURCES "lbvh/bench/benchmark_lbvh.cpp" "lbvh/*.cpp" "lbvh/bench/bench_harness.cpp" "lbvh/bench/bench_scenes.cpp" "lbvh/bench/bench_stats.cpp"
    HEADLESS
    HEADLESS_ARGS --scene worst --max-seconds 5 --iters 1 --warmup 0
)
# wildcards taken from xmake; cmake lists the same files explicitly; not mirrored as a test
gallery_target(example_software_lbvh_sort_bench
    SOURCES "lbvh/bench/sort_bench.cpp" "lbvh/*.cpp" "lbvh/bench/bench_harness.cpp" "lbvh/bench/bench_scenes.cpp" "lbvh/bench/bench_stats.cpp"
    HEADLESS
)
gallery_target(example_software_lbvh_test
    SOURCES "lbvh/test/lbvh_test.cpp" "lbvh/*.cpp"
    MIRROR_AS_TEST
    HEADLESS
)
# xmake builds it only with the GUI module enabled (gui_dep=true); the CMake file builds it outside the LUISA_COMPUTE_ENABLE_GUI guard
gallery_target(example_image_processing
    SOURCES "image_processing.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --frames 1
)
# xmake depends on lc-yyjson unless lc_yyjson_use_xrepo is on, in which case it add_packages('yyjson') instead (i.e. the dependency must be conditional on the yyjson_xrepo condition key); the cmake file adds no explicit link at all and gets yyjson through the luisa::compute umbrella. LUISA_ENABLE_GUI is added only when gui is on
gallery_target(example_native_shader
    SOURCES "native_shader_examples/native_shader.cpp" "native_shader_examples/native_shader_dispatch.cpp" "native_shader_examples/native_shader_runtime.cpp"
    DEPS lc-yyjson
    COND_DEFINES "gui:LUISA_ENABLE_GUI"
    HEADLESS
)
# minimal companion of example_native_shader (one TU, no dispatch document, no gui)
gallery_target(example_native_shader_minimal
    SOURCES "native_shader_examples/minimal.cpp"
    HEADLESS
)
# xmake add_files('*.cpp') in examples/compute/compact (no open(...) guard); cmake lists compute/compact/main.cpp + gdeflate.cpp explicitly. Only this one target comes from that directory (compute/compact has an xmake.lua but no CMakeLists.txt)
gallery_target(example_gdeflate
    SOURCES "compact/*.cpp"
    MIRROR_AS_TEST
    HEADLESS
)
# xmake add_files('*.cpp') in examples/compute/tokenize; cmake lists the same 10 files explicitly including test_tokenize.cpp and benchmark_tokenize.cpp. The cmake file deliberately adds NO extra target_link_libraries (linking the luisa-compute-vstl OBJECT library would bake the LMDB objects into the exe)
gallery_target(tokenizer
    SOURCES "tokenize/*.cpp"
    HEADLESS
)
