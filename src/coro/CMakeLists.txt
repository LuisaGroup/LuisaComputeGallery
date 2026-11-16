# GENERATED FILE - do not edit.
# Source of truth: scripts/gallery_targets.json
# Regenerate with: python scripts/gen_build.py

# xmake gates these on lc_enable_xir and adds lc-coro (coro_example_proj wrapper); the cmake file builds them unconditionally and links no explicit coro target. CMake registers metal/metal4 integration tests for them
gallery_target(example_coro_external_stage_debug
    SOURCES "external_stage_debug.cpp"
    DEPS lc-coro
    REQUIRES xir
    HEADLESS
    HEADLESS_ARGS --frames 1
)
# xmake gates these on lc_enable_xir and adds lc-coro (coro_example_proj wrapper); the cmake file builds them unconditionally and links no explicit coro target. CMake registers metal/metal4 integration tests for them
gallery_target(example_coro_external_stage_neural_sdf
    SOURCES "external_stage_neural_sdf.cpp"
    DEPS lc-coro
    REQUIRES xir
    HEADLESS
    HEADLESS_ARGS --frames 1
)
# xmake gates these on lc_enable_xir and adds lc-coro (coro_example_proj wrapper); the cmake file builds them unconditionally and links no explicit coro target. CMake registers metal/metal4 integration tests for them
gallery_target(example_coro_on_demand_texture
    SOURCES "on_demand_texture.cpp"
    DEPS lc-coro
    REQUIRES xir
    HEADLESS
    HEADLESS_ARGS --frames 1
)
