-- GENERATED FILE - do not edit.
-- Source of truth: scripts/gallery_targets.json
-- Regenerate with: python scripts/gen_build.py

gallery_example("example_fire_simulation", {sources = {"fire_simulation.cpp"}, requires = {"gui"}, gui_dep = true})
-- xmake builds it only with the GUI module enabled (gui_dep=true); the CMake file builds it outside the LUISA_COMPUTE_ENABLE_GUI guard
gallery_example("example_game_of_life", {sources = {"game_of_life.cpp"}, requires = {"gui"}, gui_dep = true})
gallery_example("example_mpm3d", {sources = {"mpm3d.cpp"}, requires = {"gui"}, gui_dep = true})
gallery_example("example_mpm88", {sources = {"mpm88.cpp"}, requires = {"gui"}, gui_dep = true})
-- also includes ../common/nbody_render_plan.h (group-local shared header)
gallery_example("example_nbody_simulation", {sources = {"nbody_simulation.cpp"}, requires = {"gui"}, gui_dep = true})
gallery_example("example_wave_equation", {sources = {"wave_equation.cpp"}, requires = {"gui"}, gui_dep = true})
