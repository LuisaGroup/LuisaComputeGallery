#!/usr/bin/env python3
"""Run the gallery examples in headless mode (no window, no swapchain) and report.

This is the command an agent or a CI job uses to check the examples without a
desktop: every example accepts

    <example> <backend> --headless [--frames N] [--output-dir DIR] [example options]

and exits with status 0 on success.

Usage:
    python scripts/headless_smoke.py                        # every runnable example
    python scripts/headless_smoke.py --filter rendering     # only a group
    python scripts/headless_smoke.py --backend cuda -k 2    # two at a time
    python scripts/headless_smoke.py --list                 # what would run

The executable directory defaults to the xmake output; `--exe-dir build-cmake/bin`
works for the CMake build.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "scripts" / "gallery_targets.json"

DEFAULT_EXE_DIRS = (
    "build/windows/x64/release",
    "build/windows/x64/releasedbg",
    "build/windows/x64/debug",
    "build/linux/x86_64/release",
    "build/macosx/arm64/release",
    "build-cmake/bin",
)

# Per-example overrides. "args" are appended after `<exe> <backend> --headless`;
# "skip" excludes a target with a reason (needs model data, extra hardware, ...).
# The per-example headless arguments and skip reasons live in the manifest
# (scripts/gallery_targets.json: "headless" / "headless_skip"), which is also what
# the generated CMakeLists.txt registers with CTest, so there is exactly one place
# to describe how an example runs without a window.
SKIP_TIMEOUT = 600


def load_targets() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    targets = []
    for group in manifest["groups"]:
        for target in group["targets"]:
            if group["dir"] == "ios" or target.get("ios_only"):
                continue
            if target.get("kind", "binary") != "binary":
                continue
            targets.append({"group": group["dir"], **target})
    return targets


def skip_reason(target: dict, backend: str) -> str | None:
    """Reason this target cannot run on `backend`, or None.

    The manifest may spell "headless_skip" as a single string (the target cannot run
    headless at all: it needs model data, extra hardware, ...) or as an object mapping
    a backend name to that reason, so a target that only fails on, say, CUDA is still
    exercised on DirectX and Vulkan.
    """
    reason = target.get("headless_skip")
    if isinstance(reason, dict):
        return reason.get(backend)
    return reason


def backend_argv(backend: str, target: dict) -> list[str]:
    """How a target wants its backend on the command line.

    Most examples take it as argv[1] (`example_xxx cuda ...`), but a few CLI
    tools follow their own convention, recorded as "backend_style" in the
    manifest: "flag" is `--backend cuda`, "flag-equals" is `--backend=cuda`.
    """
    style = target.get("backend_style", "positional")
    if style == "flag":
        return ["--backend", backend]
    if style == "flag-equals":
        return [f"--backend={backend}"]
    return [backend]


def find_exe(exe_dir: Path, name: str) -> Path | None:
    for suffix in (".exe", ""):
        candidate = exe_dir / (name + suffix)
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe-dir", default=None,
                        help="directory holding the example binaries (default: detected)")
    parser.add_argument("--backend", default="cuda",
                        help="backend passed as argv[1] (default: cuda)")
    parser.add_argument("--filter", default=None, help="substring filter on the example name or group")
    parser.add_argument("--timeout", type=int, default=600, help="default per-example timeout (s)")
    parser.add_argument("-k", "--jobs", type=int, default=1, help="run this many examples concurrently")
    parser.add_argument("--output-dir", default=str(ROOT / ".kimix_cache" / "headless"))
    parser.add_argument("--include-skipped", action="store_true",
                        help="also run targets whose manifest entry says they cannot "
                             "run headless (used to probe a new backend)")
    parser.add_argument("--list", action="store_true", help="list the catalogue and exit")
    parser.add_argument("--json", default=None, help="write a JSON report to this path")
    args = parser.parse_args()

    exe_dir = Path(args.exe_dir) if args.exe_dir else None
    if exe_dir is None:
        for candidate in DEFAULT_EXE_DIRS:
            if (ROOT / candidate).is_dir():
                exe_dir = ROOT / candidate
                break
    if exe_dir is None or not exe_dir.is_dir():
        print("cannot find an executable directory, pass --exe-dir", file=sys.stderr)
        return 2

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    targets = load_targets()
    if args.filter:
        needle = args.filter.lower()
        targets = [t for t in targets
                   if needle in t["name"].lower() or needle in t["group"].lower()]

    if args.list:
        for target in targets:
            reason = skip_reason(target, args.backend)
            state = ("skip: " + reason) if reason \
                else "run --headless " + " ".join(target.get("headless") or [])
            print(f"{target['group']:12s} {target['name']:38s} {state}")
        return 0

    def run_one(target: dict) -> dict:
        name = target["name"]
        example_args = list(target.get("headless") or [])
        result = {"name": name, "group": target["group"], "status": "skip",
                  "reason": None, "seconds": 0.0, "command": ""}
        reason = skip_reason(target, args.backend)
        if reason and not args.include_skipped:
            result["reason"] = reason
            return result
        exe = find_exe(exe_dir, name)
        if exe is None:
            result.update(status="missing", reason=f"binary not found in {exe_dir}")
            return result
        # "$ROOT" in a manifest argument expands to the repository root, so a
        # target that needs weights can point at `$ROOT/data/...` regardless of
        # the working directory the runner uses.
        expanded = [a.replace("$ROOT", str(ROOT)) for a in example_args]
        argv = [str(exe), *backend_argv(args.backend, target), "--headless", *expanded]
        result["command"] = " ".join(argv)
        log = out_root / f"{name}.log"
        timeout = target.get("headless_timeout", args.timeout)
        started = time.time()
        try:
            with log.open("wb") as handle:
                completed = subprocess.run(argv, cwd=str(out_root), stdout=handle,
                                           stderr=subprocess.STDOUT, timeout=timeout,
                                           stdin=subprocess.DEVNULL)
            result["status"] = "ok" if completed.returncode == 0 else "fail"
            result["reason"] = "" if completed.returncode == 0 else f"exit code {completed.returncode}"
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["reason"] = f"no exit within {timeout}s"
        except OSError as error:  # pragma: no cover - environment problem
            result["status"] = "error"
            result["reason"] = str(error)
        result["seconds"] = round(time.time() - started, 1)
        return result


    results: list[dict] = []
    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(run_one, targets))
    else:
        for target in targets:
            result = run_one(target)
            results.append(result)
            print(f"[{result['status']:7s}] {result['name']:38s} {result['seconds']:6.1f}s {result['reason']}")

    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    print()
    print("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for result in results:
        if result["status"] in ("fail", "timeout", "error"):
            print(f"  FAILED {result['name']}: {result['reason']} (log: {out_root / (result['name'] + '.log')})")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if counts.get("fail", 0) == 0 and counts.get("timeout", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
