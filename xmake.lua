-- LuisaComputeGallery root build script (xmake).
--
-- Layout
--   thirdparty/LuisaCompute   LuisaCompute submodule (default LC_DIR)
--   src/                      examples moved out of LuisaCompute + their tests
--   gallery/                  GalleryRT runtime library
--   app/                      gallery.exe viewer
--
-- The LuisaCompute source tree is located through the global Lua variable
-- `LC_DIR`, which defaults to the submodule path and can be overridden in the
-- machine-local, git-ignored `options.lua` (see options.lua.template), e.g.
--
--     LC_DIR = "LuisaCompute"           -- a sibling clone or a symlink
--     LC_DIR = "D:/ws/repos/LuisaCompute"
--
set_xmakever("3.0.6")
add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_languages("c++23")

-- gallery-owned options (independent of LuisaCompute's lc_* set)
option("gallery_enable_tests", {default = true,
    description = "Build the gallery tests in tests/ (moved from LuisaCompute/src/tests).",
    showmenu = true})

-- ---------------------------------------------------------------------------
-- 1. local options (git-ignored): LC_DIR and lc_options
--
-- Mirrors LuisaCompute's own `scripts/options.lua` hook: the file is included
-- before anything else so both this script and LuisaCompute's root script see
-- the values.
-- ---------------------------------------------------------------------------
if os.isfile(path.join(os.scriptdir(), "options.lua")) then
    includes("options.lua")
end

-- ---------------------------------------------------------------------------
-- 2. locate LuisaCompute
--
-- `LC_DIR` stays a plain (relative or absolute) path: it is what users are
-- expected to edit. `LCPATH` is the absolute form used by every target below,
-- because xmake resolves relative include dirs against the *target script*
-- directory while we always mean "the LuisaCompute tree".
-- ---------------------------------------------------------------------------
LC_DIR = LC_DIR or "thirdparty/LuisaCompute"
LCPATH = path.unix(path.absolute(path.join(os.scriptdir(), LC_DIR)))
if not os.isdir(LCPATH) then
    raise(string.format([[
LuisaCompute was not found at "%s" (LC_DIR = "%s").

Check out the submodule:
    git submodule update --init thirdparty/LuisaCompute

or point LC_DIR at an existing LuisaCompute source tree in options.lua:
    echo 'LC_DIR = "LuisaCompute"' > options.lua
]], LCPATH, LC_DIR))
end
if not os.isfile(path.join(LCPATH, "xmake.lua")) then
    raise(string.format('"%s" does not look like a LuisaCompute source tree (no xmake.lua).', LCPATH))
end

-- ---------------------------------------------------------------------------
-- 3. LuisaCompute configuration handed over to its root xmake.lua
--
-- LuisaCompute's root script applies every entry of the global `lc_options`
-- table with set_config(), so these are defaults only: command line
-- `--lc_xxx=yyy` wins, then options.lua, then this table.
-- ---------------------------------------------------------------------------
lc_options = table.join({
    lc_enable_dsl = true,
    lc_enable_gui = true,
    lc_enable_imgui = true,
    -- LuisaCompute's own test/example/tutorial targets: the examples live in
    -- this repository now (src/), so keep their in-tree suites off.
    lc_enable_tests = false,
    lc_enable_py = false,
    lc_disable_win_message_box = true,
}, lc_options or {})

includes(LCPATH)

-- ---------------------------------------------------------------------------
-- 4. gallery build helpers (mirror of cmake/gallery_examples.cmake)
-- ---------------------------------------------------------------------------
GALLERY_ROOT = os.scriptdir()
includes(path.join(GALLERY_ROOT, "scripts", "gallery_build.lua"))

-- ---------------------------------------------------------------------------
-- 5. gallery targets
--
-- Guarded by LuisaCompute's `_lc_check_env` option: xmake evaluates this script
-- several times (environment probing first, targets last) and LuisaCompute only
-- creates its lc-* targets once that option is checked. Creating a target that
-- depends on lc-runtime before that would fail with
-- "unknown target(lc-runtime)", so mirror LuisaCompute's own gate.
-- ---------------------------------------------------------------------------
if has_config("_lc_check_env") then
    -- LuisaCompute's root script only pulls in its own `src` when its
    -- environment check has passed; by now it has. Asking for it again is a
    -- no-op (xmake memoizes includes), and it guarantees that the lc-* targets
    -- exist in *this* evaluation pass before any gallery target references
    -- them as a dependency.
    includes(path.join(LCPATH, "src"))
    includes("gallery") -- GalleryRT.dll
    includes("app")     -- gallery.exe
    if os.isdir(path.join(os.scriptdir(), "src")) then
        includes("src") -- examples (moved from LuisaCompute/examples)
    end
    if os.isdir(path.join(os.scriptdir(), "tests")) and has_config("gallery_enable_tests") then
        includes("tests") -- tests moved alongside the examples
    end
end
