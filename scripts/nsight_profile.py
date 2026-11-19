#!/usr/bin/env python3
"""Headless NVIDIA Nsight Systems/Compute profiling for LuisaComputeGallery.

The script discovers the Windows Nsight CLIs, builds target command lines from
scripts/gallery_targets.json, runs each profiler sequentially, and saves both
binary reports and terminal-readable text reports. It never opens a GUI.

Examples:
    python scripts/nsight_profile.py doctor
    python scripts/nsight_profile.py list --filter rendering
    python scripts/nsight_profile.py profile example_path_tracing --backend cuda
    python scripts/nsight_profile.py profile example_path_tracing --mode both --dry-run
    python scripts/nsight_profile.py report .kimix_cache/nsight/<run>

Nsight Compute profiles CUDA kernels, so --mode ncu/both requires --backend
cuda. Nsight Systems also supports the dx and vk gallery backends. Profiling is
intentionally serial: concurrent GPU profilers would distort the measurements.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "scripts" / "gallery_targets.json"
DEFAULT_OUTPUT_DIR = ROOT / ".kimix_cache" / "nsight"
DEFAULT_NSYS = Path(
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.1"
    r"\target-windows-x64\nsys.exe"
)
DEFAULT_EXE_DIRS = (
    "build/windows/x64/release",
    "build/windows/x64/releasedbg",
    "build/windows/x64/debug",
    "build-cmake/bin",
)
NSYS_TRACES = {
    "cuda": "cuda,nvtx",
    "dx": "dx12,dx12-annotations,nvtx",
    "vk": "vulkan,vulkan-annotations,nvtx",
}
NSYS_REPORTS = {
    "cuda": "cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,"
            "cuda_gpu_mem_size_sum,nvtx_sum",
    "dx": "dx12_pix_sum,dx12_gpu_marker_sum,nvtx_sum",
    "vk": "vulkan_api_sum,vulkan_marker_sum,vulkan_gpu_marker_sum,nvtx_sum",
}


class ProfileError(RuntimeError):
    """A configuration, discovery, or output error that should be shown loudly."""


def load_targets(manifest_path: Path = MANIFEST) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    style = target.get("backend_style", "positional")
    if style == "flag":
        return ["--backend", backend]
    if style == "flag-equals":
        return [f"--backend={backend}"]
    if style == "positional":
        return [backend]
    raise ProfileError(f"{target['name']}: unknown backend_style {style!r}")


def application_argv(exe: Path, target: dict, backend: str,
                     extra_args: list[str] | None = None,
                     use_manifest_args: bool = True) -> list[str]:
    manifest_args = list(target.get("headless") or []) if use_manifest_args else []
    expanded = [arg.replace("$ROOT", str(ROOT)) for arg in manifest_args]
    return [str(exe), *backend_argv(backend, target), "--headless",
            *expanded, *(extra_args or [])]


def find_exe(exe_dir: Path, name: str) -> Path | None:
    for suffix in (".exe", ""):
        candidate = exe_dir / (name + suffix)
        if candidate.is_file():
            return candidate.resolve()
    return None


def locate_exe_dir(explicit: str | None, *, required: bool = True) -> Path:
    if explicit:
        path = Path(explicit).resolve()
        if required and not path.is_dir():
            raise ProfileError(f"executable directory does not exist: {path}")
        return path
    for candidate in DEFAULT_EXE_DIRS:
        path = ROOT / candidate
        if path.is_dir():
            return path.resolve()
    if required:
        raise ProfileError("cannot find gallery binaries; build a target with "
                           "`xmake build <target>` or pass --exe-dir")
    return (ROOT / DEFAULT_EXE_DIRS[0]).resolve()


def select_targets(targets: list[dict], names: list[str],
                   name_filter: str | None) -> list[dict]:
    if not names and not name_filter:
        raise ProfileError("select at least one target name or use --filter")
    by_name = {target["name"]: target for target in targets}
    selected: list[dict] = []
    for name in names:
        if name not in by_name:
            suggestion = difflib.get_close_matches(name, by_name, n=3)
            detail = f"; did you mean {', '.join(suggestion)}?" if suggestion else ""
            raise ProfileError(f"unknown gallery target {name!r}{detail}")
        if by_name[name] not in selected:
            selected.append(by_name[name])
    if name_filter:
        needle = name_filter.lower()
        for target in targets:
            if (needle in target["name"].lower() or needle in target["group"].lower()) \
                    and target not in selected:
                selected.append(target)
    if not selected:
        raise ProfileError(f"no target matched --filter {name_filter!r}")
    return selected


def _version_key(path: Path) -> tuple[int, ...]:
    match = re.search(r"(\d+(?:\.\d+)+)", path.name)
    return tuple(int(part) for part in match.group(1).split(".")) if match else (0,)


def _tool_from_root(tool: str, root: Path) -> Path | None:
    relative_paths = {
        "nsys": ("target-windows-x64/nsys.exe", "host-windows-x64/nsys.exe", "nsys.exe"),
        "ncu": ("target/windows-desktop-win7-x64/ncu.exe", "ncu.exe"),
    }
    for relative in relative_paths[tool]:
        candidate = root / relative
        if candidate.is_file():
            return candidate.resolve()
    return None


def discover_tool(tool: str, explicit: str | None = None,
                  *, required: bool = True) -> Path | None:
    """Find nsys/ncu without modifying PATH, preferring explicit configuration."""
    if tool not in ("nsys", "ncu"):
        raise ValueError(tool)
    env_names = (tool.upper(), f"NSIGHT_{'SYSTEMS' if tool == 'nsys' else 'COMPUTE'}_ROOT")
    candidates: list[str | Path] = []
    if explicit:
        candidates.append(explicit)
    else:
        candidates.extend(os.environ[name] for name in env_names if os.environ.get(name))
        on_path = shutil.which(tool)
        if on_path:
            candidates.append(on_path)
        if tool == "nsys":
            candidates.append(DEFAULT_NSYS)

    for value in candidates:
        candidate = Path(value).expanduser()
        resolved = _tool_from_root(tool, candidate) if candidate.is_dir() else candidate
        if resolved.is_file():
            return resolved.resolve()
        if len(candidates) == 1:
            on_path = shutil.which(str(value))
            if on_path:
                return Path(on_path).resolve()
    if explicit:
        if required:
            raise ProfileError(f"{tool} not found at explicit path: {explicit}")
        return None

    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    product = "Nsight Systems" if tool == "nsys" else "Nsight Compute"
    install_parent = program_files / "NVIDIA Corporation"
    if install_parent.is_dir():
        roots = sorted(install_parent.glob(f"{product} *"), key=_version_key, reverse=True)
        for root in roots:
            resolved = _tool_from_root(tool, root)
            if resolved:
                return resolved
    if required:
        raise ProfileError(
            f"cannot find {tool}; pass --{tool} PATH, set {env_names[0]}, or add it to PATH")
    return None


def tool_version(tool: Path) -> str:
    try:
        completed = subprocess.run([str(tool), "--version"], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, timeout=20,
                                   stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"version query failed: {error}"
    output = completed.stdout.strip()
    return output if completed.returncode == 0 else \
        f"version query exited {completed.returncode}: {output}"


def quote_command(argv: list[str]) -> str:
    return subprocess.list2cmdline([str(arg) for arg in argv])


def nsys_profile_argv(nsys: Path, app_argv: list[str], output_base: Path,
                      args: argparse.Namespace, backend: str) -> list[str]:
    argv = [str(nsys), "profile",
            f"--trace={args.trace or NSYS_TRACES[backend]}",
            f"--sample={args.sample}", f"--cpuctxsw={args.cpuctxsw}",
            f"--duration={args.duration:g}", f"--kill={args.kill}",
            "--show-output=true",
            f"--resolve-symbols={'true' if args.resolve_symbols else 'false'}",
            f"--output={output_base}"]
    if args.force:
        argv.append("--force-overwrite=true")
    if args.gpu_metrics:
        argv.append(f"--gpu-metrics-devices={'cuda-visible' if backend == 'cuda' else 'all'}")
    argv.extend(args.nsys_arg or [])
    argv.extend(app_argv)
    return argv


def nsys_stats_argv(nsys: Path, report: Path, sqlite: Path,
                    reports: str | None, backend: str, force: bool) -> list[str]:
    argv = [str(nsys), "stats", f"--report={reports or NSYS_REPORTS[backend]}",
            "--format=table", "--output=-", f"--sqlite={sqlite}",
            "--force-export=true"]
    if force:
        argv.append("--force-overwrite=true")
    argv.append(str(report))
    return argv


def ncu_profile_argv(ncu: Path, app_argv: list[str], output_base: Path,
                     args: argparse.Namespace) -> list[str]:
    argv = [str(ncu), "--export", str(output_base), "--set", args.ncu_set,
            "--target-processes", "application-only",
            "--clock-control", args.clock_control,
            "--cache-control", args.cache_control,
            "--launch-skip", str(args.launch_skip)]
    if not args.all_launches:
        argv.extend(["--launch-count", str(args.launch_count)])
    if args.kernel_name:
        argv.extend(["--kernel-name", args.kernel_name])
    if args.force:
        argv.append("--force-overwrite")
    argv.extend(args.ncu_arg or [])
    argv.extend(app_argv)
    return argv


def ncu_report_argv(ncu: Path, report: Path, page: str, csv: bool) -> list[str]:
    argv = [str(ncu), "--import", str(report), "--page", page]
    if csv:
        argv.append("--csv")
    return argv


def run_logged(argv: list[str], log_path: Path, cwd: Path, timeout: int,
               dry_run: bool) -> dict:
    record = {"argv": [str(arg) for arg in argv], "command": quote_command(argv),
              "log": str(log_path), "returncode": None, "seconds": 0.0,
              "status": "planned" if dry_run else "running"}
    print(f"  {record['command']}", flush=True)
    if dry_run:
        return record
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        with log_path.open("wb") as log:
            completed = subprocess.run(argv, cwd=str(cwd), stdout=log,
                                       stderr=subprocess.STDOUT, timeout=timeout,
                                       stdin=subprocess.DEVNULL, check=False)
        record["returncode"] = completed.returncode
        record["status"] = "ok" if completed.returncode == 0 else "fail"
    except subprocess.TimeoutExpired:
        record["status"] = "timeout"
    except OSError as error:
        record["status"] = "error"
        record["error"] = str(error)
    record["seconds"] = round(time.monotonic() - started, 3)
    print(f"  [{record['status']}] {log_path} ({record['seconds']:.1f}s)", flush=True)
    return record


def _require_report(operation: dict, report: Path, dry_run: bool) -> None:
    if not dry_run and operation["status"] == "ok" and not report.is_file():
        operation["status"] = "fail"
        operation["error"] = f"profiler exited successfully but did not create {report}"


def _should_generate_report(operation: dict, report: Path, no_report: bool,
                            dry_run: bool) -> bool:
    """Keep a target failure fatal, but salvage any report the profiler wrote.

    nsys propagates the target application's exit code. A crashing target can
    therefore produce both a useful .nsys-rep and a nonzero profiler exit code.
    """
    if no_report:
        return False
    if dry_run:
        return operation["status"] == "planned"
    return report.is_file()


def _prepare_run_root(output_dir: str, run_name: str | None, force: bool,
                      dry_run: bool) -> Path:
    safe_name = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    if safe_name in (".", "..") or not re.fullmatch(r"[A-Za-z0-9_.-]+", safe_name):
        raise ProfileError("--run-name may contain only letters, digits, '.', '_' and '-'")
    run_root = (Path(output_dir).resolve() / safe_name).resolve()
    if run_root.exists() and not force and not dry_run:
        raise ProfileError(f"run directory already exists: {run_root}; use --force or --run-name")
    if not dry_run:
        run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def command_list(args: argparse.Namespace) -> int:
    targets = load_targets()
    if args.filter:
        targets = select_targets(targets, [], args.filter)
    for target in targets:
        reason = skip_reason(target, args.backend)
        state = f"skip: {reason}" if reason and not args.include_skipped else "run"
        app_args = " ".join(target.get("headless") or [])
        print(f"{target['group']:12s} {target['name']:42s} {args.backend:4s} "
              f"{state:8s} {app_args}")
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    healthy = True
    for tool in ("nsys", "ncu"):
        explicit = getattr(args, tool)
        path = discover_tool(tool, explicit, required=False)
        if path is None:
            healthy = False
            print(f"{tool}: NOT FOUND (pass --{tool} PATH or set {tool.upper()})")
            continue
        print(f"{tool}: {path}")
        print(f"  {tool_version(path)}")
    try:
        exe_dir = locate_exe_dir(None, required=False)
        print(f"exe-dir: {exe_dir} ({'exists' if exe_dir.is_dir() else 'not built'})")
        print(f"targets: {len(load_targets())} binary entries in {MANIFEST}")
    except (OSError, KeyError, json.JSONDecodeError) as error:
        healthy = False
        print(f"manifest: ERROR: {error}")
    if os.name != "nt":
        healthy = False
        print("platform: ERROR: this toolkit drives the Windows Nsight CLIs")
    else:
        print("platform: Windows; no GUI or display is used")
    return 0 if healthy else 1


def command_profile(args: argparse.Namespace) -> int:
    if os.name != "nt" and not args.dry_run:
        raise ProfileError("profiling with this toolkit requires Windows")
    if args.mode in ("ncu", "both") and args.backend != "cuda":
        raise ProfileError("Nsight Compute profiles CUDA kernels; use --backend cuda "
                           "or --mode nsys for dx/vk")
    if args.duration < 0 or args.timeout <= 0 or args.launch_skip < 0 \
            or args.launch_count <= 0:
        raise ProfileError("duration/launch-skip must be >= 0; timeout/launch-count must be > 0")

    targets = select_targets(load_targets(), args.targets, args.filter)
    exe_dir = locate_exe_dir(args.exe_dir, required=not args.dry_run)
    tools: dict[str, Path] = {}
    for tool in ("nsys", "ncu"):
        needed = args.mode == "both" or args.mode == tool
        if needed:
            path = discover_tool(tool, getattr(args, tool), required=not args.dry_run)
            if path is None:
                path = Path(args.dry_run_tool_dir) / (
                    "nsys.exe" if tool == "nsys" else "ncu.exe")
            tools[tool] = path
    run_root = _prepare_run_root(args.output_dir, args.run_name, args.force, args.dry_run)
    print(f"run: {run_root}")
    results = []

    for target in targets:
        name = target["name"]
        result = {"name": name, "group": target["group"], "backend": args.backend,
                  "status": "planned" if args.dry_run else "ok", "operations": []}
        results.append(result)
        reason = skip_reason(target, args.backend)
        if reason and not args.include_skipped:
            result.update(status="skip", reason=reason)
            print(f"[skip] {name}: {reason}")
            continue
        exe = find_exe(exe_dir, name)
        if exe is None:
            if not args.dry_run:
                result.update(status="missing", reason=f"binary not found in {exe_dir}")
                print(f"[missing] {name}: build it with `xmake build {name}`")
                continue
            exe = exe_dir / f"{name}.exe"
        app_argv = application_argv(exe, target, args.backend, args.app_arg,
                                    not args.no_manifest_args)
        target_dir = run_root / f"{name}_{args.backend}"
        if not args.dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        result["application"] = app_argv
        print(f"[profile] {name} ({args.backend}, {args.mode})")

        if args.mode in ("nsys", "both"):
            nsys_base = target_dir / "system"
            nsys_report = nsys_base.with_suffix(".nsys-rep")
            operation = run_logged(
                nsys_profile_argv(tools["nsys"], app_argv, nsys_base, args, args.backend),
                target_dir / "nsys.profile.log", target_dir, args.timeout, args.dry_run)
            operation["kind"] = "nsys-profile"
            operation["report"] = str(nsys_report)
            result["operations"].append(operation)
            _require_report(operation, nsys_report, args.dry_run)
            if _should_generate_report(operation, nsys_report, args.no_report, args.dry_run):
                stats = run_logged(
                    nsys_stats_argv(tools["nsys"], nsys_report,
                                    target_dir / "system.sqlite", args.reports,
                                    args.backend, args.force),
                    target_dir / "system.stats.txt", target_dir, args.timeout, args.dry_run)
                stats["kind"] = "nsys-stats"
                result["operations"].append(stats)

        if args.mode in ("ncu", "both"):
            ncu_base = target_dir / "kernel"
            ncu_report = ncu_base.with_suffix(".ncu-rep")
            operation = run_logged(
                ncu_profile_argv(tools["ncu"], app_argv, ncu_base, args),
                target_dir / "ncu.profile.log", target_dir, args.timeout, args.dry_run)
            operation["kind"] = "ncu-profile"
            operation["report"] = str(ncu_report)
            result["operations"].append(operation)
            _require_report(operation, ncu_report, args.dry_run)
            if _should_generate_report(operation, ncu_report, args.no_report, args.dry_run):
                details = run_logged(
                    ncu_report_argv(tools["ncu"], ncu_report, args.page, args.csv),
                    target_dir / f"kernel.{args.page}.txt", target_dir,
                    args.timeout, args.dry_run)
                details["kind"] = "ncu-report"
                result["operations"].append(details)

        if not args.dry_run and any(op["status"] != "ok" for op in result["operations"]):
            result["status"] = "fail"

    summary_path = run_root / "summary.json"
    summary = {"run": str(run_root), "mode": args.mode, "backend": args.backend,
               "dry_run": args.dry_run, "tools": {key: str(value) for key, value in tools.items()},
               "results": results}
    print(f"summary: {summary_path}")
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    failed = any(result["status"] not in ("ok", "skip", "planned") for result in results)
    return 1 if failed else 0


def _collect_reports(inputs: list[str]) -> list[Path]:
    reports: list[Path] = []
    for value in inputs:
        path = Path(value).resolve()
        if path.is_dir():
            for pattern in ("*.nsys-rep", "*.ncu-rep"):
                reports.extend(path.rglob(pattern))
        elif path.is_file() and path.suffix in (".nsys-rep", ".ncu-rep"):
            reports.append(path)
        else:
            raise ProfileError(f"not an Nsight report or directory: {path}")
    unique = sorted(set(reports), key=str)
    if not unique:
        raise ProfileError("no .nsys-rep or .ncu-rep files found")
    return unique


def command_report(args: argparse.Namespace) -> int:
    if os.name != "nt" and not args.dry_run:
        raise ProfileError("reporting with the Windows Nsight CLIs requires Windows")
    reports = _collect_reports(args.inputs)
    tools: dict[str, Path] = {}
    for tool, suffix in (("nsys", ".nsys-rep"), ("ncu", ".ncu-rep")):
        if any(report.suffix == suffix for report in reports):
            path = discover_tool(tool, getattr(args, tool), required=not args.dry_run)
            if path is None:
                path = Path(args.dry_run_tool_dir) / (
                    "nsys.exe" if tool == "nsys" else "ncu.exe")
            tools[tool] = path
    results = []
    for report in reports:
        output_dir = Path(args.output_dir).resolve() if args.output_dir else report.parent
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        if report.suffix == ".nsys-rep":
            output = output_dir / f"{report.stem}.stats.txt"
            argv = nsys_stats_argv(tools["nsys"], report, output_dir / f"{report.stem}.sqlite",
                                   args.reports, args.backend, args.force)
        else:
            output = output_dir / f"{report.stem}.{args.page}.txt"
            argv = ncu_report_argv(tools["ncu"], report, args.page, args.csv)
        if output.exists() and not args.force and not args.dry_run:
            raise ProfileError(f"text report already exists: {output}; use --force")
        print(f"[report] {report}")
        operation = run_logged(argv, output, output_dir, args.timeout, args.dry_run)
        operation["input"] = str(report)
        results.append(operation)
    return 1 if any(result["status"] not in ("ok", "planned") for result in results) else 0


def _add_tool_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--nsys", help="nsys.exe path or Nsight Systems installation root")
    parser.add_argument("--ncu", help="ncu.exe path or Nsight Compute installation root")
    parser.add_argument("--dry-run", action="store_true",
                        help="print commands without creating files or running processes")
    parser.add_argument("--dry-run-tool-dir", default=r"C:\Nsight",
                        help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check Nsight tools and project setup")
    _add_tool_options(doctor)
    doctor.set_defaults(func=command_doctor)

    list_parser = subparsers.add_parser("list", help="list profileable manifest targets")
    list_parser.add_argument("--filter", help="substring match on target name or group")
    list_parser.add_argument("--backend", choices=tuple(NSYS_TRACES), default="cuda")
    list_parser.add_argument("--include-skipped", action="store_true")
    list_parser.set_defaults(func=command_list)

    profile = subparsers.add_parser(
        "profile", help="collect Nsight reports from headless gallery targets",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_tool_options(profile)
    profile.add_argument("targets", nargs="*", help="exact target names from gallery_targets.json")
    profile.add_argument("--filter", help="also select targets by name/group substring")
    profile.add_argument("--backend", choices=tuple(NSYS_TRACES), default="cuda")
    profile.add_argument("--mode", choices=("nsys", "ncu", "both"), default="nsys")
    profile.add_argument("--exe-dir", help="directory containing gallery executables")
    profile.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    profile.add_argument("--run-name", help="run subdirectory (default: local timestamp)")
    profile.add_argument("--force", action="store_true", help="overwrite an existing run")
    profile.add_argument("--include-skipped", action="store_true",
                         help="ignore manifest headless_skip reasons")
    profile.add_argument("--app-arg", action="append", default=[],
                         help="append one target argument; repeatable (use --app-arg=--foo)")
    profile.add_argument("--no-manifest-args", action="store_true",
                         help="do not append the manifest's default headless arguments")
    profile.add_argument("--timeout", type=int, default=900,
                         help="timeout for each profiler/report process in seconds")
    profile.add_argument("--no-report", action="store_true",
                         help="collect binary reports without generating text reports")

    nsys_group = profile.add_argument_group("Nsight Systems")
    nsys_group.add_argument("--trace", help="comma-separated providers (default: backend-specific)")
    nsys_group.add_argument("--reports", help="comma-separated nsys stats reports")
    nsys_group.add_argument("--duration", type=float, default=0.0,
                            help="collection seconds; 0 means until target exit (default: 0)")
    nsys_group.add_argument("--sample", choices=("none", "process-tree", "system-wide"),
                            default="none")
    nsys_group.add_argument("--cpuctxsw", choices=("none", "process-tree", "system-wide"),
                            default="none")
    nsys_group.add_argument("--kill", choices=("true", "false"), default="true",
                            help="terminate target when a profiling session ends (default: true)")
    nsys_group.add_argument("--resolve-symbols", action="store_true",
                            help="resolve sampled/backtrace symbols on Windows")
    nsys_group.add_argument("--gpu-metrics", action="store_true",
                            help="sample GPU metrics; support/permissions are system-dependent")
    nsys_group.add_argument("--nsys-arg", action="append", default=[],
                            help="append one advanced nsys profile argument; repeatable")

    ncu_group = profile.add_argument_group("Nsight Compute (CUDA only)")
    ncu_group.add_argument("--ncu-set", default="basic",
                           help="section set: basic, detailed, full, roofline, ... (default: basic)")
    ncu_group.add_argument("--launch-count", type=int, default=10,
                           help="maximum matching kernel launches (default: 10)")
    ncu_group.add_argument("--all-launches", action="store_true",
                           help="omit --launch-count; potentially very slow")
    ncu_group.add_argument("--launch-skip", type=int, default=0)
    ncu_group.add_argument("--kernel-name", help="exact name or regex:<expression>")
    ncu_group.add_argument("--clock-control", choices=("base", "none"), default="base")
    ncu_group.add_argument("--cache-control", choices=("all", "none"), default="all")
    ncu_group.add_argument("--page", choices=("details", "raw"), default="details")
    ncu_group.add_argument("--csv", action="store_true", help="format ncu text report as CSV")
    ncu_group.add_argument("--ncu-arg", action="append", default=[],
                           help="append one advanced ncu profile argument; repeatable")
    profile.set_defaults(func=command_profile)

    report = subparsers.add_parser("report", help="regenerate text from existing binary reports")
    _add_tool_options(report)
    report.add_argument("inputs", nargs="+", help=".nsys-rep/.ncu-rep files or directories")
    report.add_argument("--output-dir", help="text output directory (default: beside each report)")
    report.add_argument("--backend", choices=tuple(NSYS_TRACES), default="cuda",
                        help="selects the default nsys report set")
    report.add_argument("--reports", help="comma-separated nsys stats reports")
    report.add_argument("--page", choices=("details", "raw"), default="details")
    report.add_argument("--csv", action="store_true", help="format ncu text report as CSV")
    report.add_argument("--timeout", type=int, default=900)
    report.add_argument("--force", action="store_true")
    report.set_defaults(func=command_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ProfileError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
