-- GENERATED FILE - do not edit.
-- Source of truth: scripts/gallery_targets.json
-- Regenerate with: python scripts/gen_build.py

gallery_example("benchmark_vk_hlsl", {sources = {"vk_hlsl_benchmark.cpp", "bench_compute.cpp", "bench_raytracing.cpp", "bench_physics.cpp", "bench_inference.cpp"}, headers = {"bench_common.h"}, requires = {"vk"}})
gallery_example("spirv_inspect", {sources = {"spirv_inspect.cpp"}, deps = {"spirv-tools"}, requires = {"vk"}})
