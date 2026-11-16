# GENERATED FILE - do not edit.
# Source of truth: scripts/gallery_targets.json
# Regenerate with: python scripts/gen_build.py

# creates its own DX device with a private Agility SDK copy; argv[1] only tags the run
gallery_target(example_dx_custom_sdk
    SOURCES "custom_sdk/custom_sdk.cpp"
    REQUIRES dx
    HEADLESS
)
