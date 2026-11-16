# Locate LuisaCompute for the gallery build.
#
# Result variables (also usable as normal variables from cmake/options.cmake):
#   GALLERY_LC_MODE            submodule | external | package
#   GALLERY_LC_DIR             absolute source tree of LuisaCompute
#   GALLERY_LC_FROM_PACKAGE    TRUE when the targets come from find_package()
#   GALLERY_LC_INCLUDE_DIRS    include dirs of LuisaCompute needed by examples
#   GALLERY_LC_TEST_INCLUDE_DIRS  test support headers living in LuisaCompute
#
# The build also honours environment variables when no CMake value is given, so
# scripts can simply export them:
#   GALLERY_LC_DIR or LC_DIR   -> GALLERY_LC_DIR

set(_gallery_lc_submodule "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/LuisaCompute")

# Mode resolution: an explicit GALLERY_LC_MODE (from -D, cmake/options.cmake or a
# previous configure) always wins; otherwise a GALLERY_LC_DIR means "external" and
# the bare default is the submodule.
if (NOT GALLERY_LC_MODE)
    if (GALLERY_LC_DIR)
        set(_gallery_lc_mode_default "external")
    else ()
        set(_gallery_lc_mode_default "submodule")
    endif ()
    set(GALLERY_LC_MODE "${_gallery_lc_mode_default}" CACHE STRING
            "How to obtain LuisaCompute: submodule (thirdparty/LuisaCompute), external (GALLERY_LC_DIR), package (find_package)")
endif ()
set_property(CACHE GALLERY_LC_MODE PROPERTY STRINGS submodule external package)
set(GALLERY_LC_DIR "" CACHE PATH
        "LuisaCompute source tree for GALLERY_LC_MODE=external (or the install prefix for package)")

# --- environment fallback --------------------------------------------------
if (NOT GALLERY_LC_DIR)
    if (DEFINED ENV{GALLERY_LC_DIR} AND NOT "$ENV{GALLERY_LC_DIR}" STREQUAL "")
        set(GALLERY_LC_DIR "$ENV{GALLERY_LC_DIR}")
    elseif (DEFINED ENV{LC_DIR} AND NOT "$ENV{LC_DIR}" STREQUAL "")
        set(GALLERY_LC_DIR "$ENV{LC_DIR}")
    endif ()
endif ()
string(REPLACE "\\" "/" GALLERY_LC_DIR "${GALLERY_LC_DIR}")
# Relative paths are resolved against the gallery source tree (not the build dir),
# so `-DGALLERY_LC_DIR=LuisaCompute` and `set(GALLERY_LC_DIR "LuisaCompute")` in
# cmake/options.cmake mean "next to this repository".
if (GALLERY_LC_DIR AND NOT IS_ABSOLUTE "${GALLERY_LC_DIR}")
    get_filename_component(GALLERY_LC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${GALLERY_LC_DIR}" ABSOLUTE)
endif ()

set(GALLERY_LC_FROM_PACKAGE FALSE)

if (GALLERY_LC_MODE STREQUAL "submodule")
    set(GALLERY_LC_DIR "${_gallery_lc_submodule}")
    if (NOT EXISTS "${GALLERY_LC_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
                "LuisaCompute submodule not found at ${GALLERY_LC_DIR}.\n"
                "  git submodule update --init thirdparty/LuisaCompute\n"
                "or configure against an external tree:\n"
                "  cmake -S . -B build -DGALLERY_LC_MODE=external -DGALLERY_LC_DIR=<path>")
    endif ()
elseif (GALLERY_LC_MODE STREQUAL "external")
    if (NOT GALLERY_LC_DIR)
        message(FATAL_ERROR
                "GALLERY_LC_MODE=external needs GALLERY_LC_DIR (source tree of LuisaCompute), "
                "set with -DGALLERY_LC_DIR=<path>, the GALLERY_LC_DIR/LC_DIR environment variable "
                "or in cmake/options.cmake.")
    endif ()
    if (NOT EXISTS "${GALLERY_LC_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "GALLERY_LC_DIR (${GALLERY_LC_DIR}) is not a LuisaCompute source tree "
                "(no CMakeLists.txt).")
    endif ()
elseif (GALLERY_LC_MODE STREQUAL "package")
    find_package(LuisaCompute CONFIG REQUIRED)
    set(GALLERY_LC_FROM_PACKAGE TRUE)
    if (NOT GALLERY_LC_DIR)
        get_filename_component(GALLERY_LC_DIR "${LuisaCompute_DIR}/../../.." ABSOLUTE)
    endif ()
else ()
    message(FATAL_ERROR "Unknown GALLERY_LC_MODE '${GALLERY_LC_MODE}' "
            "(expected submodule, external or package).")
endif ()

if (NOT GALLERY_LC_FROM_PACKAGE)
    if (GALLERY_LC_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}" OR
            GALLERY_LC_DIR STREQUAL "${CMAKE_SOURCE_DIR}")
        message(FATAL_ERROR "GALLERY_LC_DIR must point at LuisaCompute, not at the gallery itself.")
    endif ()
    message(STATUS "Building LuisaCompute from source: ${GALLERY_LC_DIR}")
    # Building LuisaCompute in-tree keeps every option (LUISA_COMPUTE_ENABLE_*)
    # available to the caller: set them as normal variables from
    # cmake/options.cmake or pass -DLUISA_COMPUTE_...=... on the command line.
    add_subdirectory("${GALLERY_LC_DIR}" "${CMAKE_BINARY_DIR}/LuisaCompute")
else ()
    message(STATUS "Using installed LuisaCompute: ${LuisaCompute_DIR}")
endif ()

# --- include dirs shared by every gallery example --------------------------
set(GALLERY_LC_INCLUDE_DIRS "${GALLERY_LC_DIR}/include")
# Test-support headers that stayed in LuisaCompute (`ut/ut.hpp`, image writers,
# reference comparison helpers used by the moved examples).
set(GALLERY_LC_TEST_INCLUDE_DIRS
        "${GALLERY_LC_DIR}/src/tests"
        "${GALLERY_LC_DIR}/src/tests/common")
