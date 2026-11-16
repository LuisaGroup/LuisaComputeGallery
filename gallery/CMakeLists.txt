# GalleryRT: runtime library shared by the gallery application.
add_library(GalleryRT SHARED
        src/helper.cpp)

target_link_libraries(GalleryRT PUBLIC luisa::compute)
target_include_directories(GalleryRT PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(GalleryRT PRIVATE GALLERY_DLL_EXPORTS)

if (WIN32)
    # Same global ABI definition LuisaCompute uses (see its root CMakeLists).
    target_compile_definitions(GalleryRT PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif ()

if (LUISA_COMPUTE_ENABLE_SCCACHE)
    # inherited from the LuisaCompute launcher setup; nothing to do
endif ()

# The precompiled header is optional: it only exists to cut the cost of the
# LuisaCompute headers every translation unit pulls in.
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/__pch.h")
    target_precompile_headers(GalleryRT PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/__pch.h")
endif ()

set_target_properties(GalleryRT PROPERTIES FOLDER "gallery")
