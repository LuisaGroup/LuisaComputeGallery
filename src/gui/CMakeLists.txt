# GENERATED FILE - do not edit.
# Source of truth: scripts/gallery_targets.json
# Regenerate with: python scripts/gen_build.py

gallery_target(example_imgui
    SOURCES "imgui.cpp"
    REQUIRES gui
    GUI_DEP
    HEADLESS
    HEADLESS_ARGS --frames 1
)
# built as a SHARED library in both build systems (xmake project_kind=shared, cmake add_library(... SHARED)); cmake additionally sets EXCLUDE_FROM_ALL and only builds it with CUDA enabled (needs CUDA + DX/VK interop, see gui/test_mnist.py). Never mirrored as a test
gallery_target(example_mnist
    KIND shared
    SOURCES "mnist.cpp"
    REQUIRES gui cuda
    GUI_DEP
    NO_HEADLESS_TEST
)
gallery_target(example_swapchain
    SOURCES "swapchain.cpp"
    REQUIRES gui
    GUI_DEP
    HEADLESS
    HEADLESS_ARGS --frames 1
)
gallery_target(example_swapchain_static
    SOURCES "swapchain_static.cpp"
    REQUIRES gui
    GUI_DEP
    HEADLESS
    HEADLESS_ARGS --frames 1
)
# cmake gates it on WIN32 AND LUISA_COMPUTE_ENABLE_DX inside the GUI block
gallery_target(example_win_hdr
    SOURCES "win_hdr.cpp"
    REQUIRES gui dx
    GUI_DEP
    HEADLESS
)
# comdlg32 is needed on windows only for the native open/save dialogs (xmake add_syslinks, cmake target_link_libraries); the generator should add the platform guard
gallery_target(example_image_process
    SOURCES "image_process/main.cpp" "image_process/image_process.cpp" "image_process/headless_test.cpp"
    LIBS comdlg32
    REQUIRES gui
    GUI_DEP
    HEADLESS
    HEADLESS_ARGS --frames 1
)
# the 'noise_*.cpp' wildcard comes from xmake add_files; the cmake file lists the same 17 files explicitly (noise_all.cpp + noise_*.cpp). LUISA_ENABLE_GUI is added only when the GUI module is enabled (both build systems); the target itself is built unconditionally
gallery_target(example_procedural_tex
    SOURCES "procedural_tex/main.cpp" "procedural_tex/noise_*.cpp"
    COND_DEFINES "gui:LUISA_ENABLE_GUI"
    HEADLESS
    HEADLESS_ARGS --frames 1
)
# cmake only (no xmake target). cmake creates an INTERFACE target luisa-compute-gui-wx from the found wxWidgets (+GTK on unix) libs and links it; it is behind LUISA_COMPUTE_EXAMPLE_WX_INTEROP=ON and only exists when find_package(wxWidgets) succeeds
gallery_target(example_swapchain_wx
    SOURCES "swapchain_wx.cpp"
    LIBS luisa-compute-gui-wx
    REQUIRES gui wx
    REQUIRES_TARGET luisa-compute-gui-wx
    NO_HEADLESS_TEST
)
# cmake only (no xmake target), behind LUISA_COMPUTE_EXAMPLE_QT_INTEROP=ON; only configured when Qt6 Widgets/Core/Gui/GuiPrivate are found (skipped for clang on windows, needs the AGL framework on apple)
gallery_target(example_swapchain_qt
    SOURCES "swapchain_qt.cpp"
    LIBS Qt::Widgets Qt::Core Qt::Gui Qt::GuiPrivate
    REQUIRES gui qt
    REQUIRES_TARGET Qt::Widgets Qt::Core Qt::Gui Qt::GuiPrivate
    NO_HEADLESS_TEST
)
