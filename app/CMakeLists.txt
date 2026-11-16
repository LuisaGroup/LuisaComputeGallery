# gallery.exe -- the gallery viewer.
add_executable(gallery main.cpp)

target_link_libraries(gallery PRIVATE GalleryRT)
if (LUISA_COMPUTE_ENABLE_GUI)
    target_link_libraries(gallery PRIVATE luisa::compute)
endif ()

set_target_properties(gallery PROPERTIES FOLDER "gallery")
