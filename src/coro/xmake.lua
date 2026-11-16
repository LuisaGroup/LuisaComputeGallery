-- GENERATED FILE - do not edit.
-- Source of truth: scripts/gallery_targets.json
-- Regenerate with: python scripts/gen_build.py

-- xmake gates these on lc_enable_xir and adds lc-coro (coro_example_proj wrapper); the cmake file builds them unconditionally and links no explicit coro target. CMake registers metal/metal4 integration tests for them
gallery_example("example_coro_external_stage_debug", {sources = {"external_stage_debug.cpp"}, deps = {"lc-coro"}, requires = {"xir"}})
-- xmake gates these on lc_enable_xir and adds lc-coro (coro_example_proj wrapper); the cmake file builds them unconditionally and links no explicit coro target. CMake registers metal/metal4 integration tests for them
gallery_example("example_coro_external_stage_neural_sdf", {sources = {"external_stage_neural_sdf.cpp"}, deps = {"lc-coro"}, requires = {"xir"}})
-- xmake gates these on lc_enable_xir and adds lc-coro (coro_example_proj wrapper); the cmake file builds them unconditionally and links no explicit coro target. CMake registers metal/metal4 integration tests for them
gallery_example("example_coro_on_demand_texture", {sources = {"on_demand_texture.cpp"}, deps = {"lc-coro"}, requires = {"xir"}})
