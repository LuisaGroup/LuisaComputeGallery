#!/usr/bin/env python3
"""Generate the per-directory xmake / CMake build files for the gallery examples.

`scripts/gallery_targets.json` is the single source of truth: one entry per build
target, describing sources, dependencies, conditions and headless support.  This
script expands it into

    src/xmake.lua, src/CMakeLists.txt                    (aggregators)
    src/<group>/xmake.lua, src/<group>/CMakeLists.txt    (one call per target)

so the two build systems cannot drift apart.  The heavy lifting stays in
`scripts/gallery_build.lua` and `cmake/gallery_examples.cmake` (the mirrors of
each other); the generated files are just one call per target.

Usage:
    python scripts/gen_build.py            # (re)generate everything
    python scripts/gen_build.py --check    # fail if the generated files are stale
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "scripts" / "gallery_targets.json"
SRC = ROOT / "src"

# Groups that are moved verbatim instead of being generated: they need a foreign
# toolchain (Apple/iOS bundles) that the gallery build cannot configure yet.
VERBATIM_GROUPS = {"ios"}

HEADER_SUFFIXES = {".h", ".hpp", ".hxx", ".inl", ".inl.h"}

XMAKE_HEADER = """\
-- GENERATED FILE - do not edit.
-- Source of truth: scripts/gallery_targets.json
-- Regenerate with: python scripts/gen_build.py
"""

CMAKE_HEADER = """\
# GENERATED FILE - do not edit.
# Source of truth: scripts/gallery_targets.json
# Regenerate with: python scripts/gen_build.py
"""


def lua_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "{" + ", ".join(lua_value(v) for v in value) + "}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k} = {lua_value(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"unsupported value {value!r}")


def split_sources(target: dict) -> tuple[list[str], list[str]]:
    """Separate translation units from headers (both build systems treat them
    differently, and the upstream files list hybrid entries)."""
    sources, headers = [], list(target.get("headers") or [])
    for src in target.get("sources") or []:
        suffix = Path(src.split("*")[0]).suffix.lower()
        if suffix in HEADER_SUFFIXES:
            headers.append(src)
        else:
            sources.append(src)
    return sources, headers


def emit_xmake_target(target: dict) -> str:
    sources, headers = split_sources(target)
    opts: dict[str, object] = {}
    if target.get("kind", "binary") != "binary":
        opts["kind"] = target["kind"]
    opts["sources"] = sources
    if headers:
        opts["headers"] = headers
    deps = [d for d in target.get("deps") or [] if d != "lc-backends-dummy"]
    if deps:
        opts["deps"] = deps
    if target.get("includes"):
        opts["includes"] = target["includes"]
    defines = []
    for define in target.get("defines") or []:
        if isinstance(define, str):
            defines.append(define)
        elif define.get("when"):
            defines.append({"value": define["value"], "when": define["when"]})
        else:
            defines.append(define["value"])
    if defines:
        opts["defines"] = defines
    if target.get("syslibs"):
        opts["syslibs"] = target["syslibs"]
    if target.get("requires"):
        opts["requires"] = target["requires"]
    if target.get("requires_any"):
        opts["requires_any"] = target["requires_any"]
    if target.get("gui_dep"):
        opts["gui_dep"] = True
    if target.get("objcopy"):
        opts["objcopy"] = target["objcopy"]
    if target.get("cxx_standard"):
        opts["cxx_standard"] = target["cxx_standard"]

    lines = [f'gallery_example("{target["name"]}", {lua_value(opts)})']
    note = target.get("notes")
    if note:
        for chunk in str(note).splitlines():
            lines.insert(0, f"-- {chunk}")
    return "\n".join(lines)


def emit_cmake_target(target: dict) -> str:
    sources, headers = split_sources(target)
    lines: list[str] = []
    note = target.get("notes")
    if note:
        for chunk in str(note).splitlines():
            lines.append(f"# {chunk}")
    lines.append(f'gallery_target({target["name"]}')
    if target.get("kind", "binary") != "binary":
        lines.append(f"    KIND {target['kind']}")
    lines.append("    SOURCES " + " ".join(f'"{s}"' for s in sources))
    if headers:
        lines.append("    HEADERS " + " ".join(f'"{h}"' for h in headers))
    deps = [d for d in target.get("deps") or [] if d != "lc-backends-dummy"]
    if deps:
        lines.append("    DEPS " + " ".join(deps))
    if target.get("includes"):
        lines.append("    INCLUDES " + " ".join(f'"{i}"' for i in target["includes"]))
    # CMake embeds the blob as a luisa_compute_<name> array (EMBED_DAT) instead of
    # xmake's utils.bin2obj _binary_* symbols, so the bin2obj define is xmake-only.
    bin2obj_defines = {o.get("define") for o in (target.get("objcopy") or []) if o.get("define")}
    defines, cond_defines = [], []
    for define in target.get("defines") or []:
        if isinstance(define, str):
            defines.append(define) if define not in bin2obj_defines else None
        elif define.get("when"):
            cond_defines.append(f'{define["when"]}:{define["value"]}')
        else:
            defines.append(define["value"]) if define["value"] not in bin2obj_defines else None
    if defines:
        lines.append("    DEFINES " + " ".join(defines))
    if cond_defines:
        lines.append("    COND_DEFINES " + " ".join(f'"{d}"' for d in cond_defines))
    libs = list(dict.fromkeys((target.get("syslibs") or []) + (target.get("cmake_links") or [])))
    if libs:
        lines.append("    LIBS " + " ".join(libs))
    if target.get("requires"):
        lines.append("    REQUIRES " + " ".join(target["requires"]))
    if target.get("requires_any"):
        lines.append("    REQUIRES_ANY " + " ".join(target["requires_any"]))
    if target.get("requires_target"):
        lines.append("    REQUIRES_TARGET " + " ".join(target["requires_target"]))
    if target.get("gui_dep"):
        lines.append("    GUI_DEP")
    if target.get("mirror_as_test"):
        lines.append("    MIRROR_AS_TEST")
    if target.get("objcopy"):
        for obj in target["objcopy"]:
            lines.append(f"    EMBED_DAT \"{obj['file']}\"")
        for obj in target["objcopy"]:
            base = obj.get("base")
            if base:
                lines.append(f"    EMBED_DAT_BASE {base}")
    skip = target.get("headless_skip")
    if isinstance(skip, dict):
        # Partial skip: the example cannot run headless on a subset of the backends
        # (CUDA only, DirectX only, ...) but does run on the others, so register the
        # CTest entry with the unsupported backends filtered out.
        if isinstance(target.get("headless"), list):
            lines.append("    HEADLESS")
            if target["headless"]:
                lines.append("    HEADLESS_ARGS " + " ".join(target["headless"]))
            if skip:
                lines.append("    SKIP_BACKENDS " + " ".join(sorted(skip)))
        else:
            lines.append("    NO_HEADLESS_TEST")
    elif skip:
        lines.append("    NO_HEADLESS_TEST")
    elif isinstance(target.get("headless"), list):
        lines.append("    HEADLESS")
        if target["headless"]:
            lines.append("    HEADLESS_ARGS " + " ".join(target["headless"]))
    else:
        lines.append("    NO_HEADLESS_TEST")
    lines.append(")")
    return "\n".join(lines)


def generate_group(group: dict) -> dict[Path, str]:
    name = group["dir"]
    files: dict[Path, str] = {}
    xmake = [XMAKE_HEADER]
    cmake = [CMAKE_HEADER]
    for target in group["targets"]:
        systems = target.get("build_systems") or ["xmake", "cmake"]
        skipped = [s for s in ("xmake", "cmake") if s not in systems]
        if "xmake" in systems:
            xmake.append(emit_xmake_target(target))
        else:
            xmake.append(f"-- {target['name']}: no xmake target "
                         f"({'/'.join(systems)} only; see scripts/gallery_targets.json)")
        if "cmake" in systems:
            cmake.append(emit_cmake_target(target))
        else:
            cmake.append(f"# {target['name']}: no CMake target "
                         f"({'/'.join(systems)} only; see scripts/gallery_targets.json)")
    files[SRC / name / "xmake.lua"] = "\n".join(xmake) + "\n"
    files[SRC / name / "CMakeLists.txt"] = "\n".join(cmake) + "\n"
    return files


def generate_index(groups: list[dict]) -> dict[Path, str]:
    built = [g["dir"] for g in groups if g["dir"] not in VERBATIM_GROUPS]
    verbatim = [g["dir"] for g in groups if g["dir"] in VERBATIM_GROUPS]

    xmake = [XMAKE_HEADER, "-- Aggregator: every example group of src/.\n"]
    for name in built:
        xmake.append(f'includes("{name}")')
    if verbatim:
        xmake.append("")
        xmake.append("-- Moved verbatim, not wired into the gallery build (needs an Apple toolchain):")
        for name in verbatim:
            xmake.append(f"--   {name}/")
    files = {SRC / "xmake.lua": "\n".join(xmake) + "\n"}

    cmake = [CMAKE_HEADER, "# Aggregator: every example group of src/.\n"]
    for name in built:
        cmake.append(f"add_subdirectory({name})")
    if verbatim:
        cmake.append("")
        cmake.append("# Moved verbatim, not wired into the gallery build (needs an Apple toolchain):")
        for name in verbatim:
            cmake.append(f"#   {name}/")
    files[SRC / "CMakeLists.txt"] = "\n".join(cmake) + "\n"
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="only verify that the generated files match the manifest")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    groups = manifest["groups"]

    files: dict[Path, str] = {}
    for group in groups:
        if group["dir"] in VERBATIM_GROUPS:
            continue
        files.update(generate_group(group))
    files.update(generate_index(groups))

    stale = []
    for path, content in files.items():
        if args.check:
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            if existing != content:
                stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")

    if args.check:
        if stale:
            print("stale generated build files (run: python scripts/gen_build.py):")
            for path in stale:
                print("  " + str(path.relative_to(ROOT)))
            return 1
        print(f"all {len(files)} generated build files are up to date")
        return 0

    total = sum(len(g["targets"]) for g in groups if g["dir"] not in VERBATIM_GROUPS)
    print(f"generated {len(files)} files for {total} targets in "
          f"{len(groups) - len(VERBATIM_GROUPS)} groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
