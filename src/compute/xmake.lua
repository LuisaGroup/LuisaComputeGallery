-- GENERATED FILE - do not edit.
-- Source of truth: scripts/gallery_targets.json
-- Regenerate with: python scripts/gen_build.py

gallery_example("example_helloworld", {sources = {"helloworld.cpp"}})
-- example_bindless_mip: no xmake target (cmake only; see scripts/gallery_targets.json)
gallery_example("example_cluster_launch_control", {sources = {"cluster_launch_control.cpp"}})
gallery_example("example_async_copy_prefetch", {sources = {"async_copy_prefetch.cpp"}})
-- the 'lbvh/*.cpp' wildcard comes from xmake; cmake lists lbvh_storage.cpp, lbvh_sort.cpp, blas.cpp, tlas.cpp, software_lbvh.cpp explicitly. The LBVH sources are shared by the bench/test targets
gallery_example("example_software_lbvh", {sources = {"software_lbvh.cpp", "lbvh/*.cpp"}})
-- wildcards taken from xmake; cmake lists the same files explicitly. Deliberately NOT mirrored as a test (it reports numbers instead of asserting)
gallery_example("example_software_lbvh_bench", {sources = {"lbvh/bench/benchmark_lbvh.cpp", "lbvh/*.cpp", "lbvh/bench/bench_harness.cpp", "lbvh/bench/bench_scenes.cpp", "lbvh/bench/bench_stats.cpp"}})
-- wildcards taken from xmake; cmake lists the same files explicitly; not mirrored as a test
gallery_example("example_software_lbvh_sort_bench", {sources = {"lbvh/bench/sort_bench.cpp", "lbvh/*.cpp", "lbvh/bench/bench_harness.cpp", "lbvh/bench/bench_scenes.cpp", "lbvh/bench/bench_stats.cpp"}})
gallery_example("example_software_lbvh_test", {sources = {"lbvh/test/lbvh_test.cpp", "lbvh/*.cpp"}})
-- xmake builds it only with the GUI module enabled (gui_dep=true); the CMake file builds it outside the LUISA_COMPUTE_ENABLE_GUI guard
gallery_example("example_image_processing", {sources = {"image_processing.cpp"}, requires = {"gui"}, gui_dep = true})
-- xmake depends on lc-yyjson unless lc_yyjson_use_xrepo is on, in which case it add_packages('yyjson') instead (i.e. the dependency must be conditional on the yyjson_xrepo condition key); the cmake file adds no explicit link at all and gets yyjson through the luisa::compute umbrella. LUISA_ENABLE_GUI is added only when gui is on
gallery_example("example_native_shader", {sources = {"native_shader_examples/native_shader.cpp", "native_shader_examples/native_shader_dispatch.cpp", "native_shader_examples/native_shader_runtime.cpp"}, deps = {"lc-yyjson"}, defines = {{value = "LUISA_ENABLE_GUI", when = "gui"}}})
-- minimal companion of example_native_shader (one TU, no dispatch document, no gui)
gallery_example("example_native_shader_minimal", {sources = {"native_shader_examples/minimal.cpp"}})
-- xmake add_files('*.cpp') in examples/compute/compact (no open(...) guard); cmake lists compute/compact/main.cpp + gdeflate.cpp explicitly. Only this one target comes from that directory (compute/compact has an xmake.lua but no CMakeLists.txt)
gallery_example("example_gdeflate", {sources = {"compact/*.cpp"}})
-- xmake add_files('*.cpp') in examples/compute/tokenize; cmake lists the same 10 files explicitly including test_tokenize.cpp and benchmark_tokenize.cpp. The cmake file deliberately adds NO extra target_link_libraries (linking the luisa-compute-vstl OBJECT library would bake the LMDB objects into the exe)
gallery_example("tokenizer", {sources = {"tokenize/*.cpp"}})
