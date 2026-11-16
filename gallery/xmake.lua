-- GalleryRT: the runtime library shared by the gallery application, linking the
-- LuisaCompute modules by their xmake target names.
target("GalleryRT")
    _config_project({
        project_kind = "shared"
    })
    -- lc-ast / lc-xir have no xmake targets of their own (they are compiled into
    -- lc-core / lc-runtime); these are the modules that exist in both build
    -- systems.
    add_deps("lc-core", "lc-runtime", "lc-vstl", { public = true })
    add_deps("lc-dsl", "lc-backends-dummy", { public = true })
    if has_config("lc_enable_gui") then
        add_deps("lc-gui", { public = true })
    end
    add_defines("GALLERY_DLL_EXPORTS")

    add_includedirs("include", { public = true })
    add_includedirs(path.join(LCPATH, "include"), { public = true })
    add_headerfiles("include/**.h", "include/**.hpp")
    set_pcxxheader("src/__pch.h")
    add_files("src/**.cpp")
target_end()
