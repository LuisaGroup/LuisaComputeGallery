# GENERATED FILE - do not edit.
# Source of truth: scripts/gallery_targets.json
# Regenerate with: python scripts/gen_build.py

# condition is a DISJUNCTION in both build systems (dx OR vk OR cuda OR metal backend enabled) which the AND-only 'requires' list cannot express: keep requires empty and gate on (dx||vk||cuda||metal). xmake: if lc_dx_backend or lc_vk_backend or lc_cuda_backend or lc_metal_backend
gallery_target(example_dstorage
    SOURCES "dstorage.cpp"
    REQUIRES_ANY dx vk cuda metal
    HEADLESS
)
# creates its own DX/DirectML device; argv[1] only tags the run
gallery_target(example_dml
    SOURCES "dml.cpp"
    REQUIRES dx
    HEADLESS
)
# headless path prints "not yet implemented" and exits
gallery_target(example_supersampling
    SOURCES "supersampling.cpp"
    REQUIRES dx gui
    GUI_DEP
    HEADLESS
)
# xmake: only when mode != debug and lc_enable_clangcxx, and it includes extension/clangcxx_compiler; cmake: only when LUISA_COMPUTE_ENABLE_CLANG_CXX, outside the GUI guard, and it does NOT build clangcxx_compiler ('xmake-only')
gallery_target(example_path_tracing_clangcxx
    SOURCES "path_tracing_clangcxx.cpp"
    DEPS lc-clangcxx
    REQUIRES gui clangcxx
    GUI_DEP
    NO_HEADLESS_TEST
)
gallery_target(example_cuda_lcub
    SOURCES "cuda_lcub.cpp"
    DEPS lc-compute-cuda-ext-lcub
    LIBS CUDA::cudart CUDA::cuda_driver
    REQUIRES cuda cuda_ext_lcub
    NO_HEADLESS_TEST
)
# cmake only - the matching xmake example_proj call is commented out in examples/xmake.lua ('-- example_proj("example_dx_supersampling"...)'). Sources include <ffx_fsr2.h>/<xess/xess_d3d12.h> behind #ifdef
gallery_target(example_dx_supersampling
    SOURCES "dx_supersampling.cpp"
    REQUIRES dx
    NO_HEADLESS_TEST
)
# cmake only (no xmake target)
gallery_target(example_fsr3
    SOURCES "fsr3.cpp"
    REQUIRES dx
    NO_HEADLESS_TEST
)
# clangcxx_compiler: no CMake target (xmake only; see scripts/gallery_targets.json)
