#!/usr/bin/env python3
"""Download the newest data package from the "Data" release and unpack it.

`pack_data.py` publishes `<date-time>.7z` archives of `data/` to the `Data`
release of LuisaGroup/LuisaComputeGallery. This script fetches the newest of
them and extracts it into `data/`, overwriting files that are already there, so
a fresh clone can run the examples that need weights:

    python scripts/unpack_data.py                # newest package -> data/
    python scripts/unpack_data.py --list         # what is published?
    python scripts/unpack_data.py --dry-run      # print, do not touch data/
    python scripts/unpack_data.py --keep-archive # keep the downloaded .7z

The archive holds the *contents* of `data/` at its root, so extraction goes
straight into `data/` with 7-Zip's overwrite-all switch (`-aoa`).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_CACHE_DIR = ROOT / "dist"
DEFAULT_REPO = "LuisaGroup/LuisaComputeGallery"
DEFAULT_TAG = "Data"

WINDOWS_7Z_CANDIDATES = (
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
    r"C:\Program Files\7-Zip\7za.exe",
)


def log(message: str) -> None:
    print(message, flush=True)


def find_7z(explicit: str | None) -> str:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    for variable in ("LUISA_7Z", "SEVEN_ZIP", "7Z"):
        value = os.environ.get(variable)
        if value:
            candidates.append(value)
    for name in ("7z", "7za", "7zz"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    candidates.extend(WINDOWS_7Z_CANDIDATES)
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return str(path)
    raise SystemExit(
        "7-Zip was not found. Install it (https://7-zip.org/) or pass "
        "--7z <path-to-7z.exe> / set LUISA_7Z.")


def find_gh(explicit: str | None) -> str:
    candidates = [explicit] if explicit else []
    candidates.append(os.environ.get("GH", ""))
    candidates.append(shutil.which("gh") or "")
    candidates.append(r"C:\Program Files\GitHub CLI\gh.exe")
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise SystemExit("The GitHub CLI (gh) was not found; install it or pass --gh <path>.")


def run(command: list[str], *, dry_run: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    log(f"$ {printable}")
    if dry_run:
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(command, check=check, capture_output=True, text=True)


def list_assets(gh: str, repo: str, tag: str) -> list[dict]:
    """Assets of the release, newest first."""
    completed = subprocess.run(
        [gh, "release", "view", tag, "--repo", repo, "--json", "assets"],
        capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(
            f"cannot read release '{tag}' in {repo}: {completed.stderr.strip()}")
    import json
    assets = json.loads(completed.stdout or "{}").get("assets") or []
    assets = [a for a in assets if str(a.get("name", "")).lower().endswith(".7z")]
    assets.sort(key=lambda a: (a.get("createdAt") or "", a.get("name") or ""), reverse=True)
    return assets


def human_size(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="extraction target (default: <repo>/data)")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                        help="where the .7z is downloaded (default: <repo>/dist)")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="owner/name of the release repository")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="release tag to download from")
    parser.add_argument("--asset", default=None,
                        help="exact asset name instead of the newest one")
    parser.add_argument("--7z", dest="seven_zip", default=None, help="path to the 7z executable")
    parser.add_argument("--gh", dest="gh", default=None, help="path to the gh executable")
    parser.add_argument("--keep-archive", action="store_true", help="keep the downloaded archive")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("--list", action="store_true", help="list the published packages and exit")
    args = parser.parse_args()

    gh = find_gh(args.gh)
    assets = list_assets(gh, args.repo, args.tag)
    if not assets:
        raise SystemExit(
            f"release '{args.tag}' in {args.repo} has no .7z asset yet; "
            "publish one with scripts/pack_data.py")
    if args.list:
        for asset in assets:
            log(f"  {asset['name']:40s} {human_size(int(asset.get('size') or 0)):>12s}  "
                f"{asset.get('createdAt', '')}")
        return 0

    if args.asset:
        chosen = next((a for a in assets if a["name"] == args.asset), None)
        if chosen is None:
            raise SystemExit(f"'{args.asset}' is not published under tag '{args.tag}'")
    else:
        chosen = assets[0]
    log(f"newest package: {chosen['name']} "
        f"({human_size(int(chosen.get('size') or 0))}, {chosen.get('createdAt', '')})")

    seven_zip = find_7z(args.seven_zip)
    data_dir = Path(args.data_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / chosen["name"]

    if archive.is_file() and archive.stat().st_size == int(chosen.get("size") or 0):
        log(f"'{archive}' is already downloaded")
    else:
        run([gh, "release", "download", args.tag, "--repo", args.repo,
             "--pattern", chosen["name"], "--dir", str(cache_dir), "--clobber"],
            dry_run=args.dry_run)
    if not args.dry_run and not archive.is_file():
        raise SystemExit(f"'{archive}' was not downloaded")

    data_dir.mkdir(parents=True, exist_ok=True)
    # -aoa: overwrite all existing files, -y: assume yes on every prompt.
    extract = [seven_zip, "x", str(archive), f"-o{data_dir}", "-aoa", "-y", "-bsp1"]
    log(f"extracting into '{data_dir}' (existing files are overwritten)")
    if args.dry_run:
        log("$ " + " ".join(extract))
        return 0
    completed = subprocess.run(extract)
    if completed.returncode != 0:
        raise SystemExit(f"7z failed with exit code {completed.returncode}")

    if not args.keep_archive:
        try:
            archive.unlink()
            log(f"removed '{archive}' (pass --keep-archive to keep it)")
        except OSError as error:  # pragma: no cover - only a convenience
            log(f"could not remove '{archive}': {error}")
    log(f"data restored into '{data_dir}' from {chosen['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
