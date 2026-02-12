#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

# Import your tool (auto_crop_panels.py must be in same folder)
from auto_crop_panels import auto_crop_panels  # type: ignore


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def iter_images_from_args(args: list[str]) -> list[Path]:
    if not args:
        # No args: behave like run.bat -> all images in current folder
        return sorted(
            [p for p in Path(".").iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        )

    out: list[Path] = []
    for a in args:
        p = Path(a)

        # If it's a glob pattern (e.g. *.png)
        if any(ch in a for ch in "*?[]"):
            for m in glob.glob(a):
                mp = Path(m)
                if mp.is_file() and mp.suffix.lower() in IMAGE_EXTS:
                    out.append(mp)
            continue

        if p.is_dir():
            for child in p.rglob("*"):
                if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
                    out.append(child)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
        else:
            print(f"Skipping (not an image / not found): {a}")

    # De-dupe while preserving order
    seen = set()
    deduped = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            deduped.append(p)
    return deduped


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Run webcomic panel cropper on many images (OS-agnostic replacement for run.bat)."
    )
    ap.add_argument("paths", nargs="*", help="Files, folders, or globs. If empty: processes current folder.")
    ap.add_argument("--out", default="cropped", help="Output folder (created next to each input).")
    ap.add_argument("--debug", action="store_true", help="Write debug images.")
    ap.add_argument("--keep-debug", action="store_true", help="Do not delete debug images.")
    args = ap.parse_args(argv)

    images = iter_images_from_args(args.paths)
    if not images:
        print("No images found.")
        return 2

    total = 0
    for img_path in images:
        total += auto_crop_panels(
            str(img_path),
            out_root=args.out,
            debug=args.debug,
            keep_debug=args.keep_debug,
        )

    print(f"\nDone. Total panels exported across files: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
