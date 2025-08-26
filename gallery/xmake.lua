target("GalleryRT")
    _config_project({
        project_kind = "shared"
    })
    add_deps("lc-core", "lc-runtime", "lc-vstl", { public = true })
    add_deps("lc-dsl", "lc-ast", "lc-backends-dummy", "lc-gui", { public = true })
    add_defines("GALLERY_DLL_EXPORTS")

    add_includedirs("include", { public = true })
    add_headerfiles("include/**.h", "include/**.hpp")
    set_pcxxheader("src/__pch.h")
    add_files("src/**.cpp")
target_end()