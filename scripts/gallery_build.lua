-- Shared xmake helpers for the gallery targets.
--
-- The examples were moved here from LuisaCompute/examples; their build logic is
-- expressed once, in this file, and driven by the generated per-directory
-- xmake.lua files (see scripts/gen_build.py). The CMake mirror of these helpers
-- is cmake/gallery_examples.cmake - keep the two in sync.
--
-- Tokens accepted in `includes` / `sources` / `objcopy` paths:
--   $SRC    this repository's src/ directory (the examples root)
--   $LC     the LuisaCompute source tree (global LC_DIR)
--   $TESTS  this repository's tests/ directory
--   $GROUP  the directory of the generated script (default for relative paths)

if not GALLERY_ROOT then
    raise("gallery_build.lua must be included after GALLERY_ROOT is set by the root xmake.lua")
end
GALLERY_SRC_DIR = path.unix(path.join(GALLERY_ROOT, "src"))
GALLERY_TESTS_DIR = path.unix(path.join(GALLERY_ROOT, "tests"))

-- Resolve a path token to an absolute, forward-slash path.
function gallery_path(p, group_dir)
    local s = tostring(p)
    s = s:gsub("^%$SRC", GALLERY_SRC_DIR):gsub("^%$LC", LCPATH):gsub("^%$TESTS", GALLERY_TESTS_DIR)
    if s:find("%$GROUP") then
        s = s:gsub("%$GROUP", path.unix(group_dir or os.scriptdir()))
    end
    if not path.is_absolute(s) then
        s = path.join(group_dir or os.scriptdir(), s)
    end
    return path.unix(path.absolute(s))
end

-- Condition keys -> LuisaCompute xmake config options.
local gallery_condition_config = {
    gui = "lc_enable_gui",
    xir = "lc_enable_xir",
    dx = "lc_dx_backend",
    vk = "lc_vk_backend",
    cuda = "lc_cuda_backend",
    metal = "lc_metal_backend",
    clangcxx = "lc_enable_clangcxx",
    cuda_ext_lcub = "lc_cuda_ext_lcub",
    dx_cuda_interop = "lc_dx_cuda_interop",
    vk_cuda_interop = "lc_vk_cuda_interop",
    yyjson_xrepo = "lc_yyjson_use_xrepo",
}

function gallery_condition_holds(key)
    local config = gallery_condition_config[key]
    if not config then
        raise(string.format("unknown gallery target condition '%s'", key))
    end
    if key == "clangcxx" then
        -- the clang-cxx module is not usable in debug builds (LuisaCompute's
        -- examples/xmake.lua gates it the same way)
        return has_config(config) and not is_mode("debug")
    end
    return has_config(config)
end

-- At least one of the keys must hold (used by examples that run on any GPU
-- backend, e.g. example_dstorage).
function gallery_requirements_any_hold(requires)
    if not requires or #requires == 0 then
        return true
    end
    for _, key in ipairs(requires) do
        if gallery_condition_holds(key) then
            return true
        end
    end
    return false
end

function gallery_requirements_hold(requires)
    for _, key in ipairs(requires or {}) do
        if not gallery_condition_holds(key) then
            return false
        end
    end
    return true
end

local gallery_cmake_module_for_xmake = {
    ["lc-core"] = "luisa-compute-core",
    ["lc-runtime"] = "luisa-compute-runtime",
    ["lc-dsl"] = "luisa-compute-dsl",
    ["lc-vstl"] = "luisa-compute-vstl",
    ["lc-tile"] = "luisa-compute-tile",
    ["lc-gui"] = "luisa-compute-gui",
    ["lc-coro"] = "luisa-compute-coro",
    ["lc-osl"] = "luisa-compute-osl",
    ["lc-clangcxx"] = "luisa-compute-clangcxx",
    ["lc-yyjson"] = "luisa-compute-yyjson",
    ["stb-image"] = "luisa-compute-ext-stb-interface",
    ["lc-compute-cuda-ext-lcub"] = "luisa-compute-cuda-ext-lcub",
}

-- gallery_module(name) -> the CMake name of a LuisaCompute module (mirror of the
-- generator's mapping, kept here for hand written targets).
function gallery_module(name)
    return gallery_cmake_module_for_xmake[name] or name
end

-- ---------------------------------------------------------------------------
-- gallery_target(name, options)
--
-- options:
--   kind        "binary" (default) | "static" | "shared" | "interface"
--   sources     list of source files (required)
--   headers     list of header files (installed/listed, optional)
--   deps        extra LuisaCompute/gallery target dependencies
--   includes    extra include directories (tokens above)
--   defines     list of strings, or {value = "X", when = "<condition key>"}
--   syslibs     system libraries, added only on the matching platform
--   requires    condition keys, all of which must hold for the target to exist
--   gui_dep     true -> link lc-gui and define LUISA_ENABLE_GUI
--   objcopy     list of {file = "<path>", ext = ".dat", define = "LUISA_BIN_2_OBJ"}
--   cxx_standard/compute_control/etc: forwarded to _config_project()
-- ---------------------------------------------------------------------------
function gallery_target(name, options)
    options = options or {}
    if not gallery_requirements_hold(options.requires) then
        return
    end
    if not gallery_requirements_any_hold(options.requires_any) then
        return
    end
    local group_dir = options.group_dir or os.scriptdir()
    local kind = options.kind or "binary"
    target(name)
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    local project_config = {
        project_kind = (kind == "binary") and "binary" or kind,
        batch_size = options.batch_size,
    }
    if options.exception then project_config.enable_exception = true end
    if options.cxx_standard then project_config.cxx_standard = options.cxx_standard end
    _config_project(project_config)

    local sources = {}
    for _, src in ipairs(options.sources or {}) do
        table.insert(sources, gallery_path(src, group_dir))
    end
    add_files(sources)
    if options.headers and #options.headers > 0 then
        local headers = {}
        for _, h in ipairs(options.headers) do
            table.insert(headers, gallery_path(h, group_dir))
        end
        add_headerfiles(headers)
    end

    -- include paths every example expects: "common/xxx.h" from src/, plus the
    -- test support headers and image writers that stayed in LuisaCompute.
    add_includedirs(GALLERY_SRC_DIR, {public = true})
    add_includedirs(path.join(LCPATH, "src/tests"), path.join(LCPATH, "src/tests/common"))
    for _, inc in ipairs(options.includes or {}) do
        add_includedirs(gallery_path(inc, group_dir))
    end

    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "stb-image")
    if has_config("lc_enable_gui") then
        -- LuisaCompute's example_proj links the GUI module for every example
        -- when it is available; only the gui_dep ones get LUISA_ENABLE_GUI.
        add_deps("lc-gui")
    end
    if options.gui_dep and has_config("lc_enable_gui") then
        add_defines("LUISA_ENABLE_GUI")
    end
    for _, dep in ipairs(options.deps or {}) do
        add_deps(dep)
    end
    -- Runtime helper executables that LuisaCompute loads from the binary
    -- directory (luisa_nvrtc.exe JIT-compiles CUDA shaders). Building them as a
    -- build-time-only dependency keeps `xmake build <example>` self-contained.
    if has_config("lc_cuda_backend") then
        add_deps("lc-nvrtc", {inherit = false, links = false})
        -- Native CUDA Tile IR (CUDA 13.4+): luisa-backend-cuda runs
        -- luisa_cuda_tile_compiler from its own directory, so it has to sit next
        -- to the backend binary - same wiring as luisa_nvrtc above. On toolkits
        -- without the Tile toolchain the LuisaCompute target degrades to a phony
        -- no-op target (src/backends/cuda/xmake.lua), so the dependency is safe
        -- to declare unconditionally.
        add_deps("lc-cuda-tile-compiler", {inherit = false, links = false})
    end

    for _, define in ipairs(options.defines or {}) do
        if type(define) == "table" then
            if not define.when or gallery_condition_holds(define.when) then
                add_defines(define.value:gsub("%$LC", LCPATH):gsub("%$SRC", GALLERY_SRC_DIR):gsub("%$TESTS", GALLERY_TESTS_DIR))
            end
        else
            add_defines(define:gsub("%$LC", LCPATH):gsub("%$SRC", GALLERY_SRC_DIR):gsub("%$TESTS", GALLERY_TESTS_DIR))
        end
    end

    for _, lib in ipairs(options.syslibs or {}) do
        add_syslinks(lib)
    end

    for _, obj in ipairs(options.objcopy or {}) do
        add_rules("utils.bin2obj", {extensions = {obj.ext or ".dat"}})
        add_files(gallery_path(obj.file, group_dir))
        if obj.define then add_defines(obj.define) end
    end

    target_end()
end

-- gallery_example(name, options) -- alias used by the generated scripts.
function gallery_example(name, options)
    gallery_target(name, options)
end

-- gallery_test(name, options) -- the test counterpart (tests/ directory).
-- Identical target shape; xmake has no ctest-style registration here, the
-- binaries are run by scripts/headless_smoke.py or by hand.
function gallery_test(name, options)
    options = options or {}
    options.requires = options.requires
    gallery_target(name, options)
end
