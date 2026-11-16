# GENERATED FILE - do not edit.
# Source of truth: scripts/gallery_targets.json
# Regenerate with: python scripts/gen_build.py

gallery_target(example_fire_simulation
    SOURCES "fire_simulation.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --frames 1
)
# xmake builds it only with the GUI module enabled (gui_dep=true); the CMake file builds it outside the LUISA_COMPUTE_ENABLE_GUI guard
gallery_target(example_game_of_life
    SOURCES "game_of_life.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --frames 2
)
gallery_target(example_mpm3d
    SOURCES "mpm3d.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --frames 1
)
gallery_target(example_mpm88
    SOURCES "mpm88.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --frames 1
)
# also includes ../common/nbody_render_plan.h (group-local shared header)
gallery_target(example_nbody_simulation
    SOURCES "nbody_simulation.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --frames 1
)
gallery_target(example_wave_equation
    SOURCES "wave_equation.cpp"
    REQUIRES gui
    GUI_DEP
    MIRROR_AS_TEST
    HEADLESS
    HEADLESS_ARGS --frames 1
)
