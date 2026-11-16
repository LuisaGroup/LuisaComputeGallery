# GENERATED FILE - do not edit.
# Source of truth: scripts/gallery_targets.json
# Regenerate with: python scripts/gen_build.py

# creates its own CUDA and DX devices; argv[1] only tags the run
gallery_target(example_cuda_dx_interop
    SOURCES "cuda_dx_interop.cpp"
    REQUIRES dx_cuda_interop
    HEADLESS
)
# creates its own CUDA and VK devices; argv[1] only tags the run
gallery_target(example_cuda_vk_interop
    SOURCES "cuda_vk_interop.cpp"
    REQUIRES vk_cuda_interop cuda
    HEADLESS
)
# creates its own CUDA and VK devices; argv[1] only tags the run
gallery_target(example_cuda_vk_kernel_launch
    SOURCES "cuda_vk_kernel_launch.cpp"
    REQUIRES vk_cuda_interop cuda
    HEADLESS
)
