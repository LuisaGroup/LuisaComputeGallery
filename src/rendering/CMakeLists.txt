# GENERATED FILE - do not edit.
# Source of truth: scripts/gallery_targets.json
# Regenerate with: python scripts/gen_build.py

# cmake also lists rendering/path_tracing_test.h as a source; the TU is shared with the iOS bundle and src/tests
gallery_target(example_path_tracing
    SOURCES "path_tracing.cpp"
    HEADERS "path_tracing_test.h"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --spp 1 --frames 1
)
gallery_target(example_path_tracing_camera
    SOURCES "path_tracing_camera.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --spp 1 --frames 1
)
gallery_target(example_path_tracing_cutout
    SOURCES "path_tracing_cutout.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --spp 1 --frames 1
)
gallery_target(example_path_tracing_hdr
    SOURCES "path_tracing_hdr.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --spp 1 --frames 1
)
gallery_target(example_path_tracing_nested_callable
    SOURCES "path_tracing_nested_callable.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --spp 1 --frames 1
)
gallery_target(example_path_tracing_ray_masks
    SOURCES "path_tracing_ray_masks.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --spp 1 --frames 1
)
# xmake embeds src/tests/SRGBToFourierEvenPacked.dat with the utils.bin2obj rule and defines LUISA_BIN_2_OBJ (srgb_to_fourier_embedded.hpp then uses _binary_*_dat_start/_end symbols). cmake instead builds luisa_embed_device_lib (custom target luisa-compute-example-spectrum-embed) and regenerates rendering/srgb_to_fourier_embedded.{cpp,h} INTO the source tree, adding them as sources; the checked-in srgb_to_fourier_embedded.{cpp,h,hpp} must move with this group. The .dat file itself lives in LuisaCompute/src/tests and does NOT move
gallery_target(example_path_tracing_spectrum
    SOURCES "path_tracing_spectrum.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    EMBED_DAT "$LC/src/tests/SRGBToFourierEvenPacked.dat"
    EMBED_DAT_BASE srgb_to_fourier_embedded
    HEADLESS
    SKIP_BACKENDS cuda
)
gallery_target(example_photon_mapping
    SOURCES "photon_mapping.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --spp 1 --frames 1
)
gallery_target(example_sdf_renderer
    SOURCES "sdf_renderer.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --spp 1 --frames 1
)
gallery_target(example_blackhole
    SOURCES "blackhole.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --frames 1
)
gallery_target(example_voxel_raytracer
    SOURCES "voxel_raytracer.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --frames 1
)
gallery_target(example_procedural
    SOURCES "procedural.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --spp 1 --frames 1
)
# xmake builds it only with the GUI module enabled (gui_dep=true); the CMake file builds it outside the LUISA_COMPUTE_ENABLE_GUI guard
gallery_target(example_shader_toy
    SOURCES "shader_toy.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --frames 1
)
# usage: `shader_toy_spacex <backend> [<image>] [--offline] [--iterations N] [-c <ref.png>]`
gallery_target(example_shader_toy_spacex
    SOURCES "shader_toy_spacex.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --frames 1
)
gallery_target(example_shader_visuals_present
    SOURCES "shader_visuals_present.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --frames 1
)
# xmake gates it on lc_enable_xir (and GUI); cmake gates it on TARGET luisa-compute-xir inside the GUI block. examples/ios/CMakeLists.txt also builds an iOS bundle from the same source
gallery_target(example_path_tracing_xir2ast
    SOURCES "path_tracing_xir2ast.cpp"
    REQUIRES gui xir
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --spp 1 --frames 1
)
# xmake gates it on lc_enable_xir (and GUI); cmake gates it on TARGET luisa-compute-xir inside the GUI block. examples/ios/CMakeLists.txt also builds an iOS bundle from the same source
gallery_target(example_sdf_renderer_xir2ast
    SOURCES "sdf_renderer_xir2ast.cpp"
    REQUIRES gui xir
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --spp 1 --frames 1
)
# xmake gates it on lc_enable_xir and adds lc-coro; cmake builds it unconditionally (no xir gate, no explicit coro target - the luisa::compute umbrella covers it). Also built as an iOS bundle
gallery_target(example_coro_sdf_renderer
    SOURCES "coro_sdf_renderer.cpp"
    DEPS lc-coro
    REQUIRES xir
    HEADLESS
    HEADLESS_ARGS --frames 1
)
# xmake gates it on lc_enable_xir and adds lc-coro; cmake builds it unconditionally. Also built as an iOS bundle
gallery_target(example_coro_path_tracing
    SOURCES "coro_path_tracing.cpp"
    DEPS lc-coro
    REQUIRES xir
    HEADLESS
    HEADLESS_ARGS --spp 1 --frames 1
)
# same TU as example_coro_path_tracing, only the compile definition differs (both build systems do this)
gallery_target(example_coro_path_tracing_wavefront
    SOURCES "coro_path_tracing.cpp"
    DEPS lc-coro
    DEFINES LUISA_CORO_PATH_TRACING_SAMPLE_DISPATCH_DEFAULT=1
    REQUIRES xir
    HEADLESS
    HEADLESS_ARGS --spp 1 --frames 1
)
