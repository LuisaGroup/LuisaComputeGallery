# Helper functions shared by every example group CMakeLists in the gallery.
#
# The xmake mirror of this file is scripts/gallery_build.lua; the per-directory
# CMakeLists.txt / xmake.lua files are generated from scripts/gallery_targets.json
# by scripts/gen_build.py. Keep the two helpers in sync when a target gains a
# dependency, an include path or a new condition key.

include(CTest)

# Backend used by the registered headless smoke tests (`ctest -R headless`).
set(GALLERY_TEST_BACKEND "cuda" CACHE STRING
        "Backend passed to the headless example tests (cuda, dx, vk, metal, fallback, ...)")
# Backends the headless CTest entries are registered for. Defaults to
# GALLERY_TEST_BACKEND; set it to e.g. "cuda;dx;vk" to run every headless example on
# each enabled backend in one `ctest -L headless` invocation.
set(GALLERY_TEST_BACKENDS "" CACHE STRING
        "Backends to register headless tests for (empty: GALLERY_TEST_BACKEND)")
if (NOT GALLERY_TEST_BACKENDS)
    set(GALLERY_TEST_BACKENDS "${GALLERY_TEST_BACKEND}")
endif ()

# ---------------------------------------------------------------------------
# Condition keys -> CMake option variables (an empty entry means "always true").
# ---------------------------------------------------------------------------
set(_GALLERY_CONDITION_gui "LUISA_COMPUTE_ENABLE_GUI")
set(_GALLERY_CONDITION_xir "") # XIR is unconditional in the CMake build
set(_GALLERY_CONDITION_dx "LUISA_COMPUTE_ENABLE_DX")
set(_GALLERY_CONDITION_vk "LUISA_COMPUTE_ENABLE_VULKAN")
set(_GALLERY_CONDITION_cuda "LUISA_COMPUTE_ENABLE_CUDA")
set(_GALLERY_CONDITION_metal "LUISA_COMPUTE_ENABLE_METAL")
set(_GALLERY_CONDITION_clangcxx "LUISA_COMPUTE_ENABLE_CLANG_CXX")
set(_GALLERY_CONDITION_cuda_ext_lcub "LUISA_COMPUTE_ENABLE_CUDA_EXT_LCUB")
set(_GALLERY_CONDITION_dx_cuda_interop "LUISA_COMPUTE_ENABLE_DX_CUDA_INTEROP")
set(_GALLERY_CONDITION_vk_cuda_interop "LUISA_COMPUTE_ENABLE_VK_CUDA_INTEROP")
set(_GALLERY_CONDITION_yyjson_xrepo "LUISA_COMPUTE_USE_SYSTEM_YYJSON")
set(_GALLERY_CONDITION_qt "GALLERY_ENABLE_QT_INTEROP")
set(_GALLERY_CONDITION_wx "GALLERY_ENABLE_WX_INTEROP")

# xmake target name -> CMake target name (modules that are already part of
# luisa::compute can be listed here as well; linking them again is harmless).
set(_GALLERY_DEP_MAP
        lc-core luisa-compute-core
        lc-runtime luisa-compute-runtime
        lc-dsl luisa-compute-dsl
        lc-vstl luisa-compute-vstl
        lc-tile luisa-compute-tile
        lc-gui luisa-compute-gui
        lc-coro luisa-compute-coro
        lc-osl luisa-compute-osl
        lc-clangcxx luisa-compute-clangcxx
        lc-yyjson luisa-compute-ext-yyjson-interface
        stb-image luisa-compute-ext-stb-interface
        lc-compute-cuda-ext-lcub luisa-compute-cuda-ext-lcub)

function(_gallery_map_dep dep out_var)
    list(FIND _GALLERY_DEP_MAP "${dep}" _idx)
    if (_idx GREATER -1)
        math(EXPR _val_idx "${_idx} + 1")
        list(GET _GALLERY_DEP_MAP ${_val_idx} _mapped)
        set(${out_var} "${_mapped}" PARENT_SCOPE)
    else ()
        # Unknown name: a plain library (e.g. "ws2_32") or a target we do not
        # know about. Passing it through keeps the original behaviour.
        set(${out_var} "${dep}" PARENT_SCOPE)
    endif ()
endfunction()

# ---------------------------------------------------------------------------
# gallery_target(<name> [options])
#
#   KIND <binary|static|shared>       (default binary)
#   SOURCES <files...>                (globs allowed, resolved with CONFIGURE_DEPENDS)
#   HEADERS <files...>
#   DEPS <xmake-target-names...>      mapped to CMake targets
#   INCLUDES <dirs...>                $SRC/$LC/$GROUP tokens resolved here
#   DEFINES <defs...>
#   COND_DEFINES <"condition:def"...>
#   LIBS <libs...>                    linked on top of luisa::compute
#   REQUIRES <condition-keys...>      all must hold, otherwise the target is skipped
#   REQUIRES_ANY <condition-keys...>  at least one must hold
#   GUI_DEP                           defines LUISA_ENABLE_GUI
#   MIRROR_AS_TEST                    also build test_<name> from the same sources
#   EMBED_DAT <file> / EMBED_DAT_BASE <name>
#                                     generate <name>.cpp/.h from <file> with
#                                     luisa_embed_device_lib (mirrors the xmake
#                                     utils.bin2obj rule used by the example)
#   HEADLESS <args...>                extra argv for the headless ctest entry
#   SKIP_BACKENDS <backend...>        backends this example cannot run headless on
#                                     (a CUDA-only limitation does not keep the
#                                     DirectX/Vulkan runs out of CTest)
#   NO_HEADLESS_TEST                  do not register a headless ctest entry
# ---------------------------------------------------------------------------
function(gallery_target name)
    cmake_parse_arguments(GT
            "GUI_DEP;MIRROR_AS_TEST;NO_HEADLESS_TEST;HEADLESS"
            "KIND"
            "SOURCES;HEADERS;DEPS;INCLUDES;DEFINES;COND_DEFINES;LIBS;REQUIRES;REQUIRES_ANY;REQUIRES_TARGET;EMBED_DAT;EMBED_DAT_BASE;HEADLESS_ARGS;SKIP_BACKENDS"
            ${ARGN})

    # --- conditions --------------------------------------------------------
    foreach (_req IN LISTS GT_REQUIRES)
        set(_cond_var "${_GALLERY_CONDITION_${_req}}")
        if (_cond_var AND NOT ${_cond_var})
            return()
        endif ()
    endforeach ()
    if (GT_REQUIRES_ANY)
        set(_any_ok FALSE)
        foreach (_req IN LISTS GT_REQUIRES_ANY)
            set(_cond_var "${_GALLERY_CONDITION_${_req}}")
            if (NOT _cond_var OR ${_cond_var})
                set(_any_ok TRUE)
            endif ()
        endforeach ()
        if (NOT _any_ok)
            return()
        endif ()
    endif ()

    # --- required CMake targets (Qt::Widgets, luisa-compute-gui-wx, ...) ---
    foreach (_required IN LISTS GT_REQUIRES_TARGET)
        if (NOT TARGET ${_required})
            return()
        endif ()
    endforeach ()

    # --- sources -----------------------------------------------------------
    set(_sources "")
    foreach (_src IN LISTS GT_SOURCES)
        if (_src MATCHES "\\*")
            file(GLOB _globbed CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${_src}")
            list(APPEND _sources ${_globbed})
        else ()
            list(APPEND _sources "${CMAKE_CURRENT_SOURCE_DIR}/${_src}")
        endif ()
    endforeach ()
    foreach (_hdr IN LISTS GT_HEADERS)
        if (_hdr MATCHES "\\*")
            # same glob handling as above: "api/*.h" style header lists
            file(GLOB _globbed_headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${_hdr}")
            list(APPEND _sources ${_globbed_headers})
        else ()
            list(APPEND _sources "${CMAKE_CURRENT_SOURCE_DIR}/${_hdr}")
        endif ()
    endforeach ()

    # --- generated data blobs (xmake: utils.bin2obj) -----------------------
    set(_embed_includes "")
    if (GT_EMBED_DAT)
        set(_idx 0)
        list(LENGTH GT_EMBED_DAT_BASE _base_count)
        foreach (_dat IN LISTS GT_EMBED_DAT)
            if (_idx LESS _base_count)
                list(GET GT_EMBED_DAT_BASE ${_idx} _base)
            else ()
                # no explicit base: fall back to the file stem
                get_filename_component(_base "${_dat}" NAME_WE)
            endif ()
            math(EXPR _idx "${_idx} + 1")
            set(_out_dir "${CMAKE_CURRENT_BINARY_DIR}/embedded")
            set(_out_cpp "${_out_dir}/${_base}.cpp")
            set(_out_h "${_out_dir}/${_base}.h")
            if (_dat MATCHES "^\\$LC/")
                string(REPLACE "$LC" "${GALLERY_LC_DIR}" _dat "${_dat}")
            endif ()
            add_custom_command(
                    OUTPUT "${_out_cpp}" "${_out_h}"
                    COMMAND $<TARGET_FILE:luisa_embed_device_lib> "${_dat}"
                            --unsigned --prefix luisa_compute_
                            -o "${_out_cpp}" -h "${_out_h}"
                    DEPENDS luisa_embed_device_lib "${_dat}"
                    COMMENT "Embedding ${_dat}")
            list(APPEND _sources "${_out_cpp}")
            list(APPEND _embed_includes "${_out_dir}")
        endforeach ()
    endif ()

    # CMake's add_library() keywords are upper case ("SHARED", "STATIC", ...);
    # the manifest stores them lower case.
    if (NOT GT_KIND OR GT_KIND STREQUAL "binary")
        set(_kind "binary")
    else ()
        string(TOUPPER "${GT_KIND}" _kind)
    endif ()

    if (_kind STREQUAL "binary" AND GT_MIRROR_AS_TEST)
        string(REGEX REPLACE "^example_" "test_" _mirror "${name}")
    endif ()

    foreach (_variant IN ITEMS "${name}" "${_mirror}")
        if (NOT _variant)
            continue()
        endif ()
        if (_kind STREQUAL "binary")
            add_executable(${_variant} ${_sources})
        else ()
            add_library(${_variant} ${_kind} ${_sources})
        endif ()
        target_link_libraries(${_variant} PRIVATE luisa::compute)
        target_include_directories(${_variant} PRIVATE
                "${GALLERY_SOURCE_DIR}/src"
                "${CMAKE_CURRENT_SOURCE_DIR}"
                ${GALLERY_LC_INCLUDE_DIRS}
                ${GALLERY_LC_TEST_INCLUDE_DIRS}
                ${_embed_includes})
        foreach (_inc IN LISTS GT_INCLUDES)
            string(REPLACE "$SRC" "${GALLERY_SOURCE_DIR}/src" _inc "${_inc}")
            string(REPLACE "$LC" "${GALLERY_LC_DIR}" _inc "${_inc}")
            string(REPLACE "$GROUP" "${CMAKE_CURRENT_SOURCE_DIR}" _inc "${_inc}")
            string(REPLACE "$TESTS" "${GALLERY_SOURCE_DIR}/tests" _inc "${_inc}")
            if (NOT IS_ABSOLUTE "${_inc}")
                set(_inc "${CMAKE_CURRENT_SOURCE_DIR}/${_inc}")
            endif ()
            target_include_directories(${_variant} PRIVATE "${_inc}")
        endforeach ()

        if (GT_DEFINES)
            # resolve the path tokens the manifest uses for absolute defines
            set(_resolved_defines "")
            foreach (_def IN LISTS GT_DEFINES)
                string(REPLACE "$SRC" "${GALLERY_SOURCE_DIR}/src" _def "${_def}")
                string(REPLACE "$LC" "${GALLERY_LC_DIR}" _def "${_def}")
                string(REPLACE "$TESTS" "${GALLERY_SOURCE_DIR}/tests" _def "${_def}")
                string(REPLACE "$GROUP" "${CMAKE_CURRENT_SOURCE_DIR}" _def "${_def}")
                list(APPEND _resolved_defines "${_def}")
            endforeach ()
            target_compile_definitions(${_variant} PRIVATE ${_resolved_defines})
        endif ()
        foreach (_cdef IN LISTS GT_COND_DEFINES)
            if (_cdef MATCHES "^([^:]+):(.*)$")
                set(_cond "${CMAKE_MATCH_1}")
                set(_value "${CMAKE_MATCH_2}")
                set(_cond_var "${_GALLERY_CONDITION_${_cond}}")
                if (NOT _cond_var OR ${_cond_var})
                    target_compile_definitions(${_variant} PRIVATE "${_value}")
                endif ()
            endif ()
        endforeach ()
        if (GT_GUI_DEP AND LUISA_COMPUTE_ENABLE_GUI)
            target_compile_definitions(${_variant} PRIVATE LUISA_ENABLE_GUI)
        endif ()

        foreach (_dep IN LISTS GT_DEPS)
            if (_dep STREQUAL "lc-backends-dummy")
                continue()
            endif ()
            _gallery_map_dep("${_dep}" _mapped)
            if (TARGET "${_mapped}")
                target_link_libraries(${_variant} PRIVATE "${_mapped}")
            endif ()
        endforeach ()
        if (GT_LIBS)
            target_link_libraries(${_variant} PRIVATE ${GT_LIBS})
        endif ()

        if (_mirror AND _variant STREQUAL "${_mirror}")
            set_target_properties(${_variant} PROPERTIES FOLDER tests)
        else ()
            set_target_properties(${_variant} PROPERTIES FOLDER examples)
        endif ()
    endforeach ()

    # --- headless smoke test ----------------------------------------------
    if (GT_MIRROR_AS_TEST AND NOT GT_NO_HEADLESS_TEST AND _kind STREQUAL "binary")
        string(REGEX REPLACE "^example_" "test_" _test_target "${name}")
    else ()
        set(_test_target "${name}")
    endif ()
    if (NOT GT_NO_HEADLESS_TEST AND _kind STREQUAL "binary" AND GT_HEADLESS)
        gallery_add_headless_test(${_test_target} ARGS ${GT_HEADLESS_ARGS}
                SKIP_BACKENDS ${GT_SKIP_BACKENDS})
    endif ()
endfunction()

# gallery_add_headless_test(<target> [NAME <ctest-name>] [BACKEND <backend>]
#                           [LABELS <labels...>] [TIMEOUT <sec>] [ARGS <args...>])
#
# Registers a CTest entry that runs the example in headless mode: no window, no
# swapchain, bounded work, PNG/summary output under the build directory. This is
# the entry point for CI and for agents verifying a change from the command line
# (`ctest -R example_helloworld` or `ctest -L headless`).
function(gallery_add_headless_test target)
    cmake_parse_arguments(GT "" "NAME;BACKEND;TIMEOUT" "LABELS;ARGS;SKIP_BACKENDS" ${ARGN})
    if (NOT GT_TIMEOUT)
        set(GT_TIMEOUT 900)
    endif ()
    if (NOT GT_LABELS)
        set(GT_LABELS "headless")
    endif ()
    if (NOT TARGET ${target})
        return()
    endif ()
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/test-output")
    set(_backends "${GT_BACKEND}")
    if (NOT _backends)
        set(_backends "${GALLERY_TEST_BACKENDS}")
    endif ()
    list(LENGTH _backends _backend_count)
    foreach (_backend IN LISTS _backends)
        if (_backend IN_LIST GT_SKIP_BACKENDS)
            continue()
        endif ()
        if (GT_NAME)
            set(_test_name "${GT_NAME}")
        elseif (_backend_count GREATER 1)
            set(_test_name "headless_${target}_${_backend}")
        else ()
            set(_test_name "headless_${target}")
        endif ()
        add_test(NAME ${_test_name}
                COMMAND $<TARGET_FILE:${target}> ${_backend} --headless ${GT_ARGS})
        set_tests_properties(${_test_name} PROPERTIES
                LABELS "${GT_LABELS}"
                TIMEOUT ${GT_TIMEOUT}
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/test-output")
    endforeach ()
endfunction()

# gallery_test(<name> [options])
#
# Same as gallery_target, plus a CTest entry (respecting LABELS and ARGS), which
# is how the moved tests are registered.
function(gallery_test name)
    cmake_parse_arguments(GTS "NO_CTEST" "NAME;TIMEOUT" "LABELS;ARGS" ${ARGN})
    set(_forward ${GTS_UNPARSED_ARGUMENTS})
    gallery_target(${name} ${_forward})
    if (NOT TARGET ${name} OR GTS_NO_CTEST)
        return()
    endif ()
    set(_test_name "${name}")
    if (GTS_NAME)
        set(_test_name "${GTS_NAME}")
    endif ()
    if (NOT GTS_TIMEOUT)
        set(GTS_TIMEOUT 900)
    endif ()
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/test-output")
    add_test(NAME ${_test_name} COMMAND $<TARGET_FILE:${name}> ${GTS_ARGS})
    set_tests_properties(${_test_name} PROPERTIES
            LABELS "${GTS_LABELS}"
            TIMEOUT ${GTS_TIMEOUT}
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/test-output")
endfunction()

# Legacy aliases: hand written CMakeLists in this repository may still use them.
function(gallery_add_example name)
    gallery_target(${name} ${ARGN})
endfunction()

function(gallery_target_link name)
    target_link_libraries(${name} PRIVATE ${ARGN})
endfunction()

function(gallery_target_compile_definitions name)
    target_compile_definitions(${name} PRIVATE ${ARGN})
endfunction()

function(gallery_target_compile_options name)
    target_compile_options(${name} PRIVATE ${ARGN})
endfunction()
