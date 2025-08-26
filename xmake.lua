set_xmakever("3.0.1")
add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_languages("c++23")

includes("lc_options.generated.lua")
includes(lc_dir)

includes("gallery") -- GalleryRT.dll 
includes("app") -- gallery.exe 
