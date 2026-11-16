# LuisaComputeGallery -- CMake build.
#
# Two ways to locate the LuisaCompute tree the gallery builds against:
#
#   cmake -S . -B build -G Ninja                     # 1. the git submodule
#                                                    #    thirdparty/LuisaCompute
#   cmake -S . -B build -G Ninja -DGALLERY_LC_MODE=external \
#         -DGALLERY_LC_DIR=D:/ws/repos/LuisaCompute  # 2. any external checkout
#
# A third mode, -DGALLERY_LC_MODE=package, consumes an installed LuisaCompute
# instead of building it from source. Machine-local settings (the CMake mirror of
# options.lua, git-ignored) may also be written to cmake/options.cmake; see
# cmake/options.cmake.template.
cmake_minimum_required(VERSION 3.26...3.29)
cmake_policy(VERSION 3.26)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW) # honour normal variables over option() defaults

project(LuisaComputeGallery LANGUAGES C CXX VERSION 0.1.0)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(GALLERY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Output layout, mirroring LuisaCompute's scripts/setup_output_dirs.cmake.
# Directory-scope properties do not propagate upwards, so a LuisaCompute that is
# added as a subdirectory cannot set this for the gallery: do it here, before any
# subdirectory is added, so every binary (examples, tests, gallery.exe) lands next
# to the luisa-*.dll runtime libraries and finds its backend plugins.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
if (WIN32)
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif ()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# machine-local options (git-ignored)
# ---------------------------------------------------------------------------
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/options.cmake")
    message(STATUS "Loading local options from cmake/options.cmake")
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/options.cmake")
endif ()

option(GALLERY_BUILD_EXAMPLES "Build the gallery examples (src/)" ON)
option(GALLERY_BUILD_TESTS "Build the gallery tests (tests/)" ON)
option(GALLERY_BUILD_APP "Build the gallery viewer application (app/)" ON)

# ---------------------------------------------------------------------------
# LuisaCompute
# ---------------------------------------------------------------------------
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/gallery_luisa.cmake")

# ---------------------------------------------------------------------------
# gallery targets
# ---------------------------------------------------------------------------
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/gallery_interop.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/gallery_examples.cmake")

if (GALLERY_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/CMakeLists.txt")
    add_subdirectory(src)
endif ()
if (GALLERY_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif ()
add_subdirectory(gallery)
if (GALLERY_BUILD_APP)
    add_subdirectory(app)
endif ()

message(STATUS "LuisaComputeGallery: LC mode = ${GALLERY_LC_MODE}, LC dir = ${GALLERY_LC_DIR}")
