# Optional GUI-toolkit interoperability examples.
#
# Two examples talk to a foreign window system (Qt6 and wxWidgets) instead of the
# plain GLFW window. They are opt-in: enable GALLERY_ENABLE_QT_INTEROP /
# GALLERY_ENABLE_WX_INTEROP and have the toolkit installed, otherwise the targets
# are skipped (see the `qt` / `wx` condition keys in cmake/gallery_examples.cmake).
#
# This mirrors the original LuisaCompute/examples/CMakeLists.txt blocks.

option(GALLERY_ENABLE_WX_INTEROP "Build the wxWidgets swapchain example (needs wxWidgets)" ON)
option(GALLERY_ENABLE_QT_INTEROP "Build the Qt6 swapchain example (needs Qt6)" ON)

if (LUISA_COMPUTE_ENABLE_GUI AND GALLERY_ENABLE_WX_INTEROP)
    find_package(wxWidgets COMPONENTS core base QUIET)
    if (wxWidgets_FOUND)
        message(STATUS "Found wxWidgets: ${wxWidgets_INCLUDE_DIRS}")
        if (NOT TARGET luisa-compute-gui-wx)
            add_library(luisa-compute-gui-wx INTERFACE)
            target_include_directories(luisa-compute-gui-wx INTERFACE ${wxWidgets_INCLUDE_DIRS})
            target_link_libraries(luisa-compute-gui-wx INTERFACE ${wxWidgets_LIBRARIES})
        endif ()
    endif ()
endif ()

if (LUISA_COMPUTE_ENABLE_GUI AND GALLERY_ENABLE_QT_INTEROP)
    if (WIN32 AND NOT MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(WARNING "Clang has trouble with Qt on Windows: the Qt example will not be built.")
    else ()
        find_package(Qt6 COMPONENTS Widgets Core Gui QUIET CONFIG)
        if (Qt6_FOUND)
            message(STATUS "Found Qt6: ${Qt6_DIR}")
        endif ()
    endif ()
endif ()
