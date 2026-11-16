-- GENERATED FILE - do not edit.
-- Source of truth: scripts/gallery_targets.json
-- Regenerate with: python scripts/gen_build.py

-- cmake also lists rendering/path_tracing_test.h as a source; the TU is shared with the iOS bundle and src/tests
gallery_example("example_path_tracing", {sources = {"path_tracing.cpp"}, headers = {"path_tracing_test.h"}, requires = {"gui"}, gui_dep = true})
gallery_example("example_path_tracing_camera", {sources = {"path_tracing_camera.cpp"}, requires = {"gui"}, gui_dep = true})
gallery_example("example_path_tracing_cutout", {sources = {"path_tracing_cutout.cpp"}, requires = {"gui"}, gui_dep = true})
gallery_example("example_path_tracing_hdr", {sources = {"path_tracing_hdr.cpp"}, requires = {"gui"}, gui_dep = true})
gallery_example("example_path_tracing_nested_callable", {sources = {"path_tracing_nested_callable.cpp"}, requires = {"gui"}, gui_dep = true})
gallery_example("example_path_tracing_ray_masks", {sources = {"path_tracing_ray_masks.cpp"}, requires = {"gui"}, gui_dep = true})
-- xmake embeds src/tests/SRGBToFourierEvenPacked.dat with the utils.bin2obj rule and defines LUISA_BIN_2_OBJ (srgb_to_fourier_embedded.hpp then uses _binary_*_dat_start/_end symbols). cmake instead builds luisa_embed_device_lib (custom target luisa-compute-example-spectrum-embed) and regenerates rendering/srgb_to_fourier_embedded.{cpp,h} INTO the source tree, adding them as sources; the checked-in srgb_to_fourier_embedded.{cpp,h,hpp} must move with this group. The .dat file itself lives in LuisaCompute/src/tests and does NOT move
gallery_example("example_path_tracing_spectrum", {sources = {"path_tracing_spectrum.cpp"}, defines = {"LUISA_BIN_2_OBJ"}, requires = {"gui"}, gui_dep = true, objcopy = {{file = "$LC/src/tests/SRGBToFourierEvenPacked.dat", ext = ".dat", define = "LUISA_BIN_2_OBJ", base = "srgb_to_fourier_embedded"}}})
gallery_example("example_photon_mapping", {sources = {"photon_mapping.cpp"}, requires = {"gui"}, gui_dep = true})
gallery_example("example_sdf_renderer", {sources = {"sdf_renderer.cpp"}, requires = {"gui"}, gui_dep = true})
gallery_example("example_blackhole", {sources = {"blackhole.cpp"}, requires = {"gui"}, gui_dep = true})
gallery_example("example_voxel_raytracer", {sources = {"voxel_raytracer.cpp"}, requires = {"gui"}, gui_dep = true})
gallery_example("example_procedural", {sources = {"procedural.cpp"}, requires = {"gui"}, gui_dep = true})
-- xmake builds it only with the GUI module enabled (gui_dep=true); the CMake file builds it outside the LUISA_COMPUTE_ENABLE_GUI guard
gallery_example("example_shader_toy", {sources = {"shader_toy.cpp"}, requires = {"gui"}, gui_dep = true})
-- usage: `shader_toy_spacex <backend> [<image>] [--offline] [--iterations N] [-c <ref.png>]`
gallery_example("example_shader_toy_spacex", {sources = {"shader_toy_spacex.cpp"}, requires = {"gui"}, gui_dep = true})
gallery_example("example_shader_visuals_present", {sources = {"shader_visuals_present.cpp"}, requires = {"gui"}, gui_dep = true})
-- xmake gates it on lc_enable_xir (and GUI); cmake gates it on TARGET luisa-compute-xir inside the GUI block. examples/ios/CMakeLists.txt also builds an iOS bundle from the same source
gallery_example("example_path_tracing_xir2ast", {sources = {"path_tracing_xir2ast.cpp"}, requires = {"gui", "xir"}, gui_dep = true})
-- xmake gates it on lc_enable_xir (and GUI); cmake gates it on TARGET luisa-compute-xir inside the GUI block. examples/ios/CMakeLists.txt also builds an iOS bundle from the same source
gallery_example("example_sdf_renderer_xir2ast", {sources = {"sdf_renderer_xir2ast.cpp"}, requires = {"gui", "xir"}, gui_dep = true})
-- xmake gates it on lc_enable_xir and adds lc-coro; cmake builds it unconditionally (no xir gate, no explicit coro target - the luisa::compute umbrella covers it). Also built as an iOS bundle
gallery_example("example_coro_sdf_renderer", {sources = {"coro_sdf_renderer.cpp"}, deps = {"lc-coro"}, requires = {"xir"}})
-- xmake gates it on lc_enable_xir and adds lc-coro; cmake builds it unconditionally. Also built as an iOS bundle
gallery_example("example_coro_path_tracing", {sources = {"coro_path_tracing.cpp"}, deps = {"lc-coro"}, requires = {"xir"}})
-- same TU as example_coro_path_tracing, only the compile definition differs (both build systems do this)
gallery_example("example_coro_path_tracing_wavefront", {sources = {"coro_path_tracing.cpp"}, deps = {"lc-coro"}, defines = {"LUISA_CORO_PATH_TRACING_SAMPLE_DISPATCH_DEFAULT=1"}, requires = {"xir"}})
