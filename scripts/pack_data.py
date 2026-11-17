#!/usr/bin/env python3
"""Pack `data/` into a timestamped .7z and publish it to the "Data" release.

The gallery examples that need weights (embedding/bge-m3, OCR packages, NTC
textures, ...) keep them in the git-ignored `data/` directory. This script turns
that directory into a single archive and uploads it with the GitHub CLI, so a
fresh clone can restore it with `scripts/unpack_data.py`.

    python scripts/pack_data.py                    # pack + upload
    python scripts/pack_data.py --no-upload        # only create the archive
    python scripts/pack_data.py --dry-run          # print what would happen
    python scripts/pack_data.py --list             # list the published assets

Archive name: `<YYYYmmdd-HHMMSS>.7z` (the moment packing started), release tag
`Data` in `LuisaGroup/LuisaComputeGallery` by default. The archive stores the
*contents* of `data/` at its root, so `unpack_data.py` can extract straight into
`data/`.

Requirements: 7-Zip (`7z`) and the GitHub CLI (`gh`), authenticated with a token
that may write releases.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_ARCHIVE_DIR = ROOT / "dist"
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
    """Locate the 7-Zip executable: --7z, $LUISA_7Z/$7Z, PATH, known installs."""
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


def archive_name() -> str:
    return _datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".7z"


def ensure_release(gh: str, repo: str, tag: str, *, dry_run: bool) -> None:
    view = run([gh, "release", "view", tag, "--repo", repo],
               dry_run=False, check=False)
    if view.returncode == 0:
        log(f"release '{tag}' already exists in {repo}")
        return
    log(f"creating release '{tag}' in {repo}")
    run([gh, "release", "create", tag, "--repo", repo, "--title", tag,
         "--notes", "Gallery example data packages (see scripts/pack_data.py)."],
        dry_run=dry_run)


def create_archive(seven_zip: str, data_dir: Path, out_path: Path,
                   level: int, dry_run: bool) -> Path:
    if not data_dir.is_dir():
        raise SystemExit(f"data directory '{data_dir}' does not exist.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    # "." keeps the archive rooted at the data directory contents.
    command = [seven_zip, "a", "-t7z", f"-mx{level}", "-mmt=on", "-bsp1",
               "-y", str(out_path), "."]
    log(f"packing '{data_dir}' -> '{out_path}' (this can take a while)")
    if dry_run:
        log("$ " + " ".join(command) + f"   (cwd={data_dir})")
        return out_path
    completed = subprocess.run(command, cwd=str(data_dir))
    if completed.returncode != 0:
        raise SystemExit(f"7z failed with exit code {completed.returncode}")
    return out_path


def human_size(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def list_assets(gh: str, repo: str, tag: str) -> int:
    completed = run([gh, "release", "view", tag, "--repo", repo,
                     "--json", "assets", "--jq",
                     ".assets[] | \"\\(.name) \\(.size) \\(.createdAt)\""],
                    dry_run=False, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"release '{tag}' not found in {repo}: {completed.stderr.strip()}")
    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    if not rows:
        log(f"release '{tag}' has no assets yet")
        return 0
    for row in rows:
        name, size, created = (row.split() + ["", "", ""])[:3]
        try:
            size_text = human_size(int(size))
        except ValueError:
            size_text = size
        log(f"  {name:40s} {size_text:>12s}  {created}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="directory to pack (default: <repo>/data)")
    parser.add_argument("--archive-dir", default=str(DEFAULT_ARCHIVE_DIR),
                        help="where the .7z is written (default: <repo>/dist)")
    parser.add_argument("--archive", default=None,
                        help="archive file name (default: <date-time>.7z)")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="owner/name of the release repository")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="release tag to upload to")
    parser.add_argument("--level", type=int, default=9, help="7-Zip compression level (0-9)")
    parser.add_argument("--7z", dest="seven_zip", default=None, help="path to the 7z executable")
    parser.add_argument("--gh", dest="gh", default=None, help="path to the gh executable")
    parser.add_argument("--no-upload", action="store_true", help="create the archive but do not publish it")
    parser.add_argument("--keep-existing", action="store_true",
                        help="do not overwrite an existing archive of the same name")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("--list", action="store_true", help="list the assets of the release and exit")
    args = parser.parse_args()

    gh = find_gh(args.gh)
    if args.list:
        return list_assets(gh, args.repo, args.tag)

    seven_zip = find_7z(args.seven_zip)
    data_dir = Path(args.data_dir).resolve()
    archive_dir = Path(args.archive_dir).resolve()
    out_path = archive_dir / (args.archive or archive_name())
    if out_path.exists() and args.keep_existing:
        log(f"'{out_path}' exists, keeping it")
    else:
        create_archive(seven_zip, data_dir, out_path, args.level, args.dry_run)

    if args.dry_run:
        log("dry run: nothing was uploaded")
        return 0
    if not out_path.is_file():
        raise SystemExit(f"'{out_path}' was not created")
    log(f"archive: {out_path} ({human_size(out_path.stat().st_size)})")
    if args.no_upload:
        log("--no-upload: skipping 'gh release upload'")
        return 0

    ensure_release(gh, args.repo, args.tag, dry_run=args.dry_run)
    upload = run([gh, "release", "upload", args.tag, str(out_path),
                  "--repo", args.repo, "--clobber"])
    if upload.returncode != 0:
        log(upload.stdout.rstrip())
        log(upload.stderr.rstrip())
        raise SystemExit(f"gh release upload failed with exit code {upload.returncode}")
    log(f"published {out_path.name} to "
        f"https://github.com/{args.repo}/releases/tag/{args.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
