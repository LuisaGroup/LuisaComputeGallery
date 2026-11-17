#!/usr/bin/env python3
"""Turn a `capture_gallery.py` run into the images referenced by `docs/`.

`capture_gallery.py` writes full-resolution JPEGs (one per example and backend)
into a scratch directory. The documents want something small enough to live in
the repository, so this script picks one picture per example, downscales it to
at most `--max-size` pixels on the long edge and writes
`docs/images/<example>.jpg`.

    python scripts/make_doc_images.py --capture-dir .kimix_cache/capture
    python scripts/make_doc_images.py --report .kimix_cache/capture/report.json --dry-run

Backend preference is dx, then vk, then cuda: the three produce the same picture,
dx is simply the first one the CI machine runs. `--all-backends` keeps
`<example>_<backend>.jpg` for every backend instead of one file per example.
Requires Pillow (`python -m pip install pillow`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND_PREFERENCE = ("dx", "vk", "cuda")


def log(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-dir", default=str(ROOT / ".kimix_cache" / "capture"))
    parser.add_argument("--report", default=None,
                        help="report.json (default: <capture-dir>/report.json)")
    parser.add_argument("--out-dir", default=str(ROOT / "docs" / "images"))
    parser.add_argument("--max-size", type=int, default=900,
                        help="longest edge of the stored image in pixels (default 900)")
    parser.add_argument("--quality", type=int, default=88, help="JPEG quality (default 88)")
    parser.add_argument("--all-backends", action="store_true",
                        help="keep one image per backend instead of one per example")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        from PIL import Image
    except ImportError:
        log("Pillow is missing: python -m pip install pillow")
        return 2

    capture_dir = Path(args.capture_dir).resolve()
    report_path = Path(args.report) if args.report else capture_dir / "report.json"
    if not report_path.is_file():
        log(f"no report at '{report_path}'; run scripts/capture_gallery.py first")
        return 2
    records = json.loads(report_path.read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir).resolve()
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    by_name: dict[str, dict[str, dict]] = {}
    for record in records:
        if not record.get("capture"):
            continue
        by_name.setdefault(record["name"], {})[record["backend"]] = record

    written = 0
    for name in sorted(by_name):
        per_backend = by_name[name]
        if args.all_backends:
            chosen = [(b, per_backend[b]) for b in BACKEND_PREFERENCE if b in per_backend]
        else:
            picked = next((per_backend[b] for b in BACKEND_PREFERENCE if b in per_backend),
                          next(iter(per_backend.values())))
            chosen = [(picked["backend"], picked)]
        for backend, record in chosen:
            source = capture_dir / record["capture"]
            if not source.is_file():
                log(f"missing {source}")
                continue
            target = out_dir / (f"{name}.jpg" if not args.all_backends
                                else f"{name}_{backend}.jpg")
            log(f"{source.name:50s} -> {target.name}")
            if args.dry_run:
                continue
            with Image.open(source) as image:
                image = image.convert("RGB")
                if max(image.size) > args.max_size:
                    scale = args.max_size / float(max(image.size))
                    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
                    image = image.resize(size, Image.LANCZOS)
                image.save(target, "JPEG", quality=args.quality, optimize=True)
            written += 1

    if args.dry_run:
        log("dry run: nothing written")
        return 0
    total = sum(f.stat().st_size for f in out_dir.glob("*.jpg"))
    log(f"wrote {written} image(s) into {out_dir} ({total / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
