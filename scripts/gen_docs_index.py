#!/usr/bin/env python3
"""Generate `docs/README.md` from the example pages and a capture report.

The index is derived, never hand written: one row per `docs/examples/*.md` page
with the page's own subtitle as the description, plus the backends that the
target actually ran on (from `capture_gallery.py`'s report.json) and a
thumbnail grid for the pages that have a picture.

    python scripts/gen_docs_index.py
    python scripts/gen_docs_index.py --report .kimix_cache/capture/report.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "scripts" / "gallery_targets.json"
EXAMPLES = ROOT / "docs" / "examples"
IMAGES = ROOT / "docs" / "images"
BACKEND_ORDER = ("dx", "vk", "cuda")

HEADER = """# LuisaComputeGallery examples

Every example in this repository is a self-contained program that shows one part
of [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute): the DSL, the
runtime, ray tracing, the coroutine module, tensor programming, and the higher
level engines (embedding, OCR, neural texture compression).

Each page below documents one runnable target: what it is for, how it is
architected, how to run it, and - when it draws something - the picture it
produces. Every picture in these pages was captured with the built-in capture
mode (`--capture result.jpg`), so any of them can be reproduced with one command.

- Build: [build.md](build.md) (`xmake build <target>`, or the CMake mirror).
- Run without a window: [headless.md](headless.md) (`<example> <backend> --headless`).
- Save the result picture: [capture.md](capture.md) (`--capture <file.png|jpg>`).
- Example data (weights, textures): `python scripts/unpack_data.py`;
  `python scripts/pack_data.py` publishes a new package.

The `Backends` column is copied from each page and lists the backends the target
really uses: `dx` (DirectX 12), `vk` (Vulkan) and `cuda` (NVIDIA). A page narrows
it when the target is not portable - an example that always creates a `dx` device,
or one whose feature only exists on some backends, says so in the same cell.

## Gallery
"""


def page_subtitle(name: str) -> str:
    path = EXAMPLES / f"{name}.md"
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    parts: list[str] = []
    started = False
    for line in lines:
        text = line.strip()
        if not text:
            if started:
                break
            continue
        if text[0] in "#|!-*`>":  # table, image, list, quote, heading, code
            if started:
                break
            continue
        started = True
        parts.append(text)
    subtitle = re.sub(r"\s+", " ", " ".join(parts)).strip()
    subtitle = subtitle.replace("|", "\\|")
    if len(subtitle) > 220:
        subtitle = subtitle[:217].rsplit(" ", 1)[0] + "..."
    return subtitle


BACKENDS_ROW = re.compile(r"^\|\s*Backends\s*\|\s*(.+?)\s*\|\s*$", re.MULTILINE)


def page_backends(name: str) -> str:
    """The `| Backends | ... |` row of a page, without the surrounding pipes."""
    path = EXAMPLES / f"{name}.md"
    if not path.is_file():
        return ""
    match = BACKENDS_ROW.search(path.read_text(encoding="utf-8"))
    return match.group(1).replace("|", "\\|") if match else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", default=str(ROOT / ".kimix_cache" / "capture" / "report.json"))
    parser.add_argument("--out", default=str(ROOT / "docs" / "README.md"))
    args = parser.parse_args()

    status: dict[str, dict[str, str]] = collections.defaultdict(dict)
    report = Path(args.report)
    if report.is_file():
        for record in json.loads(report.read_text(encoding="utf-8")):
            status[record["name"]][record["backend"]] = record["status"]
    else:
        print(f"warning: no capture report at {report}; backends will show as '-'")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out: list[str] = [HEADER]

    documented: list[tuple[str, str]] = []
    for group in manifest["groups"]:
        for target in group["targets"]:
            name = target["name"]
            if (EXAMPLES / f"{name}.md").is_file():
                documented.append((group["dir"], name))

    pictures = [(g, n) for g, n in documented if (IMAGES / f"{n}.jpg").is_file()]
    columns = 4
    for i in range(0, len(pictures), columns):
        row = pictures[i:i + columns]
        out.append("| " + " | ".join(
            f'<a href="examples/{n}.md"><img src="images/{n}.jpg" width="220"></a>'
            for _, n in row) + " |")
        out.append("| " + " | ".join(":-:" for _ in row) + " |")
        out.append("| " + " | ".join(f"[{n}](examples/{n}.md)" for _, n in row) + " |")
    out.append("")

    summary: list[tuple[str, int]] = []
    for group in manifest["groups"]:
        names = [n for g, n in documented if g == group["dir"]]
        if not names:
            continue
        out.append(f"\n## {group['dir']}\n")
        out.append("| Example | What it shows | Backends |")
        out.append("|---|---|---|")
        for name in names:
            backends = page_backends(name)
            if not backends:
                backends = ", ".join(
                    b for b in BACKEND_ORDER if status.get(name, {}).get(b) == "ok") or "-"
            out.append(f"| [{name}](examples/{name}.md) | {page_subtitle(name)} | {backends} |")
        summary.append((group["dir"], len(names)))

    out.append("\n## Summary\n")
    out.append("| Group | Pages |")
    out.append("|---|---|")
    for group, count in summary:
        out.append(f"| {group} | {count} |")
    out.append(f"| **total** | **{sum(c for _, c in summary)}** |")

    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(documented)} page(s), {len(pictures)} picture(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
