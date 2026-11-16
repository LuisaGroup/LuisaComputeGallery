-- GENERATED FILE - do not edit.
-- Source of truth: scripts/gallery_targets.json
-- Regenerate with: python scripts/gen_build.py

gallery_example("example_imgui", {sources = {"imgui.cpp"}, requires = {"gui"}, gui_dep = true})
-- built as a SHARED library in both build systems (xmake project_kind=shared, cmake add_library(... SHARED)); cmake additionally sets EXCLUDE_FROM_ALL and only builds it with CUDA enabled (needs CUDA + DX/VK interop, see gui/test_mnist.py). Never mirrored as a test
gallery_example("example_mnist", {kind = "shared", sources = {"mnist.cpp"}, requires = {"gui", "cuda"}, gui_dep = true})
gallery_example("example_swapchain", {sources = {"swapchain.cpp"}, requires = {"gui"}, gui_dep = true})
gallery_example("example_swapchain_static", {sources = {"swapchain_static.cpp"}, requires = {"gui"}, gui_dep = true})
-- cmake gates it on WIN32 AND LUISA_COMPUTE_ENABLE_DX inside the GUI block
gallery_example("example_win_hdr", {sources = {"win_hdr.cpp"}, requires = {"gui", "dx"}, gui_dep = true})
-- comdlg32 is needed on windows only for the native open/save dialogs (xmake add_syslinks, cmake target_link_libraries); the generator should add the platform guard
gallery_example("example_image_process", {sources = {"image_process/main.cpp", "image_process/image_process.cpp", "image_process/headless_test.cpp"}, syslibs = {"comdlg32"}, requires = {"gui"}, gui_dep = true})
-- the 'noise_*.cpp' wildcard comes from xmake add_files; the cmake file lists the same 17 files explicitly (noise_all.cpp + noise_*.cpp). LUISA_ENABLE_GUI is added only when the GUI module is enabled (both build systems); the target itself is built unconditionally
gallery_example("example_procedural_tex", {sources = {"procedural_tex/main.cpp", "procedural_tex/noise_*.cpp"}, defines = {{value = "LUISA_ENABLE_GUI", when = "gui"}}})
-- example_swapchain_wx: no xmake target (cmake only; see scripts/gallery_targets.json)
-- example_swapchain_qt: no xmake target (cmake only; see scripts/gallery_targets.json)
