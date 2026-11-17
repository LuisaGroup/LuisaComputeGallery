#!/usr/bin/env python3
"""Run every gallery example headless and capture its result image.

The examples that draw something write their picture through
`--capture <path>` (see src/common/reference_compare.h and src/common/headless.h),
so one command per example is enough to produce the screenshots used by the
documents in `docs/`:

    python scripts/capture_gallery.py --backend dx
    python scripts/capture_gallery.py --backend dx --backend vk --backend cuda
    python scripts/capture_gallery.py --filter rendering --backend cuda
    python scripts/capture_gallery.py --list

Everything is driven by `scripts/gallery_targets.json`: `headless` holds the
example's arguments, `headless_skip` says why the target cannot run on that
backend. `$ROOT` in an argument is expanded to the repository root, so a target
that needs weights can point at `$ROOT/data/...`.

Output:
    <out-dir>/<name>_<backend>.jpg   the captured picture (when the example has one)
    <out-dir>/<name>_<backend>.log   full stdout/stderr of the run
    <out-dir>/report.json            what ran, what passed, what produced an image
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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


def expand(token: str) -> str:
    return token.replace("$ROOT", str(ROOT))


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


def locate_exe_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    for candidate in DEFAULT_EXE_DIRS:
        path = ROOT / candidate
        if path.is_dir():
            return path
    raise SystemExit("cannot find an executable directory, pass --exe-dir")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe-dir", default=None, help="directory with the example binaries")
    parser.add_argument("--backend", action="append", default=None,
                        help="backend to run (repeatable; default: dx)")
    parser.add_argument("--filter", default=None, help="substring filter on the target name or group")
    parser.add_argument("--out-dir", default=str(ROOT / ".kimix_cache" / "capture"),
                        help="where images/logs/report.json are written")
    parser.add_argument("--format", default="jpg", choices=("jpg", "png"),
                        help="capture container (default jpg)")
    parser.add_argument("--timeout", type=int, default=600, help="per-run timeout in seconds")
    parser.add_argument("--include-skipped", action="store_true",
                        help="run targets whose manifest entry says they cannot run")
    parser.add_argument("--list", action="store_true", help="list what would run and exit")
    parser.add_argument("--json", default=None, help="also write the report to this path")
    args = parser.parse_args()

    backends = args.backend or ["dx"]
    exe_dir = locate_exe_dir(args.exe_dir)
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    targets = load_targets()
    if args.filter:
        needle = args.filter.lower()
        targets = [t for t in targets
                   if needle in t["name"].lower() or needle in t["group"].lower()]
    if not targets:
        raise SystemExit("no target matched the filter")

    if args.list:
        for target in targets:
            for backend in backends:
                reason = skip_reason(target, backend)
                state = ("skip: " + reason) if reason else "run"
                print(f"{target['group']:12s} {target['name']:38s} {backend:5s} {state}")
        return 0

    results: list[dict] = []
    for backend in backends:
        for target in targets:
            name = target["name"]
            record = {"name": name, "group": target["group"], "backend": backend,
                      "status": "skip", "reason": None, "capture": None,
                      "seconds": 0.0, "command": ""}
            reason = skip_reason(target, backend)
            if reason and not args.include_skipped:
                record["reason"] = reason
                results.append(record)
                print(f"[{'skip':7s}] {name:38s} {backend:5s} {reason}")
                continue
            exe = find_exe(exe_dir, name)
            if exe is None:
                record.update(status="missing", reason="binary not built")
                results.append(record)
                print(f"[{'missing':7s}] {name:38s} {backend:5s} binary not built")
                continue
            capture = out_root / f"{name}_{backend}.{args.format}"
            if capture.exists():
                capture.unlink()
            example_args = [expand(a) for a in (target.get("headless") or [])]
            base = [str(exe), *backend_argv(backend, target)]
            # Most examples accept --capture, but a tool with its own strict
            # parser may not; those are retried without it (the run itself still
            # proves the example works, it just has no picture to store).
            attempts = ([*base, "--capture", str(capture), *example_args], base + example_args)
            log_path = out_root / f"{name}_{backend}.log"
            timeout = target.get("headless_timeout", args.timeout)
            started = time.time()
            record["command"] = " ".join(attempts[0])
            for attempt, argv in enumerate(attempts):
                try:
                    with log_path.open("wb") as log:
                        completed = subprocess.run(argv, cwd=str(out_root), stdout=log,
                                                   stderr=subprocess.STDOUT, timeout=timeout,
                                                   stdin=subprocess.DEVNULL)
                except subprocess.TimeoutExpired:
                    record["status"] = "timeout"
                    record["reason"] = f"no exit within {timeout}s"
                    break
                except OSError as error:  # pragma: no cover - environment problem
                    record["status"] = "error"
                    record["reason"] = str(error)
                    break
                if completed.returncode == 0:
                    record["status"] = "ok"
                    record["reason"] = None
                    break
                record["status"] = "fail"
                record["reason"] = f"exit code {completed.returncode}"
                if attempt == 0 and not capture.is_file():
                    record["command"] = " ".join(attempts[1])
                    continue
                break
            record["seconds"] = round(time.time() - started, 1)
            if capture.is_file():
                record["capture"] = str(capture.relative_to(out_root))
                record["capture_bytes"] = capture.stat().st_size
            results.append(record)
            detail = record["capture"] or "no image"
            print(f"[{record['status']:7s}] {name:38s} {backend:5s} "
                  f"{record['seconds']:6.1f}s {detail}"
                  + (f"  ({record['reason']})" if record["reason"] else ""))

    counts: dict[str, int] = {}
    for record in results:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    with_images = sum(1 for r in results if r["capture"])
    print()
    print("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
          + f", with image={with_images}")
    for record in results:
        if record["status"] in ("fail", "timeout", "error"):
            print(f"  FAILED {record['name']} ({record['backend']}): {record['reason']} "
                  f"(log: {out_root / (record['name'] + '_' + record['backend'] + '.log')})")

    report_path = out_root / "report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"report: {report_path}")
    return 0 if not (counts.get("fail") or counts.get("timeout")) else 1


if __name__ == "__main__":
    sys.exit(main())
