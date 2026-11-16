# GENERATED FILE - do not edit.
# Source of truth: scripts/gallery_targets.json
# Regenerate with: python scripts/gen_build.py

gallery_target(example_multi_head_attention
    SOURCES "multi_head_attention.cpp" "attention_kernels.cpp" "attention_host_data.cpp" "attention_cpu_reference.cpp" "attention_runner.cpp" "paged_attention.cpp"
    MIRROR_AS_TEST
    HEADLESS
)
