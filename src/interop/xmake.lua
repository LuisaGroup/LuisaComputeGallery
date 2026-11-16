-- GENERATED FILE - do not edit.
-- Source of truth: scripts/gallery_targets.json
-- Regenerate with: python scripts/gen_build.py

-- creates its own CUDA and DX devices; argv[1] only tags the run
gallery_example("example_cuda_dx_interop", {sources = {"cuda_dx_interop.cpp"}, requires = {"dx_cuda_interop"}})
-- creates its own CUDA and VK devices; argv[1] only tags the run
gallery_example("example_cuda_vk_interop", {sources = {"cuda_vk_interop.cpp"}, requires = {"vk_cuda_interop", "cuda"}})
-- creates its own CUDA and VK devices; argv[1] only tags the run
gallery_example("example_cuda_vk_kernel_launch", {sources = {"cuda_vk_kernel_launch.cpp"}, requires = {"vk_cuda_interop", "cuda"}})
