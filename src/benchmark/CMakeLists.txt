# GENERATED FILE - do not edit.
# Source of truth: scripts/gallery_targets.json
# Regenerate with: python scripts/gen_build.py

gallery_target(benchmark_vk_hlsl
    SOURCES "vk_hlsl_benchmark.cpp" "bench_compute.cpp" "bench_raytracing.cpp" "bench_physics.cpp" "bench_inference.cpp"
    HEADERS "bench_common.h"
    REQUIRES vk
    HEADLESS
    HEADLESS_ARGS --iters 3
)
gallery_target(spirv_inspect
    SOURCES "spirv_inspect.cpp"
    DEPS spirv-tools
    REQUIRES vk
    NO_HEADLESS_TEST
)
