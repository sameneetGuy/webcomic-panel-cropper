import argparse
import os
import sys

import cv2
import numpy as np


def auto_crop_panels(path: str, out_root: str = "cropped", debug: bool = False, keep_debug: bool = False) -> int:
    print(f"\n=== {path} ===")

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(" ! Could not read image")
        return 0

    basename = os.path.splitext(os.path.basename(path))[0]
    base_dir = os.path.dirname(path) or "."
    out_dir = os.path.join(base_dir, out_root)
    os.makedirs(out_dir, exist_ok=True)

    # --- Build BGR image + alpha for processing ---
    alpha = None
    if img.ndim == 2:
        h, w = img.shape
        bgr_for_debug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        gray = img
    else:
        h, w, channels = img.shape
        if channels == 4:
            b, g, r, a = cv2.split(img)
            alpha = a
            bgr_for_debug = cv2.merge((b, g, r))
        else:
            bgr_for_debug = img.copy()
        gray = cv2.cvtColor(bgr_for_debug, cv2.COLOR_BGR2GRAY)

    print(f" Size: {w}x{h}, alpha: {'yes' if alpha is not None else 'no'}")

    # --- Effective grayscale: transparent pixels treated as white background ---
    eff_gray = gray.copy()
    if alpha is not None:
        eff_gray[alpha <= 10] = 255

    debug_paths = []
    if debug:
        p = os.path.join(out_dir, f"{basename}_effgray.png")
        cv2.imwrite(p, eff_gray)
        debug_paths.append(p)
        print(f" -> saved effective grayscale: {p}")

    # --- Threshold: ink = dark pixels ---
    _, mask = cv2.threshold(eff_gray, 245, 255, cv2.THRESH_BINARY_INV)

    if debug:
        nonzero = cv2.countNonZero(mask)
        print(f" Ink pixels in mask: {nonzero} ({nonzero/(w*h):.3f} of image)")
        p = os.path.join(out_dir, f"{basename}_mask.png")
        cv2.imwrite(p, mask)
        debug_paths.append(p)
        print(f" -> saved ink mask: {p}")

    # --- Morphology: stabilize linework without fusing gutters too much ---
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if debug:
        p = os.path.join(out_dir, f"{basename}_processed.png")
        cv2.imwrite(p, processed)
        debug_paths.append(p)
        print(f" -> saved processed mask: {p}")

    # --- Find panel-ish rectangles ---
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f" Raw contours: {len(contours)}")

    rects = []
    total_pixels = w * h
    min_area = total_pixels * 0.01      # 1% of image
    min_w = w * 0.05                    # 5% of width
    min_h = h * 0.10                    # 10% of height

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < min_area:
            continue
        if cw < min_w or ch < min_h:
            continue
        rects.append((x, y, cw, ch))

    print(f" Rectangles after filter: {len(rects)}")
    if not rects:
        print(" ! No panel-like rectangles found, using full image as one panel.")
        rects = [(0, 0, w, h)]

    if debug:
        dbg = bgr_for_debug.copy()
        for i, (x, y, cw, ch) in enumerate(rects, start=1):
            cv2.rectangle(dbg, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
            cv2.putText(dbg, str(i), (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        p = os.path.join(out_dir, f"{basename}_debug_rects.png")
        cv2.imwrite(p, dbg)
        debug_paths.append(p)
        print(f" -> saved debug rect overlay: {p}")

    # --- Sort panels: top-to-bottom, then left-to-right with row grouping ---
    rects_sorted = sorted(rects, key=lambda r: r[1])
    row_tol = max(10, int(h * 0.05))

    rows = []
    for r in rects_sorted:
        x, y, cw, ch = r
        placed = False
        for row in rows:
            if abs(y - row[0]) < row_tol:
                row[1].append(r)
                placed = True
                break
        if not placed:
            rows.append([y, [r]])

    rows.sort(key=lambda r: r[0])
    panels = []
    for _, row_rects in rows:
        row_rects.sort(key=lambda r: r[0])
        panels.extend(row_rects)

    # --- Crop from original image (keeps transparency if present) ---
    for i, (x, y, cw, ch) in enumerate(panels, start=1):
        crop = img[y : y + ch, x : x + cw]
        out_path = os.path.join(out_dir, f"{basename}_panel{i}.png")
        cv2.imwrite(out_path, crop)
        print(f" -> panel {i}: {out_path}")

    # --- Cleanup debug files if requested ---
    if debug and (not keep_debug):
        for p in debug_paths:
            try:
                os.remove(p)
            except Exception as e:
                print(f" Could not delete {p}: {e}")

    print(f" Total panels exported: {len(panels)}")
    return len(panels)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Auto-crop webcomic panels from page images.")
    ap.add_argument("images", nargs="+", help="Input image(s): png/jpg/etc.")
    ap.add_argument("--out", default="cropped", help="Output folder (created next to input). Default: cropped")
    ap.add_argument("--debug", action="store_true", help="Write debug images.")
    ap.add_argument("--keep-debug", action="store_true", help="Do not delete debug images.")
    args = ap.parse_args(argv)

    total = 0
    for path in args.images:
        total += auto_crop_panels(path, out_root=args.out, debug=args.debug, keep_debug=args.keep_debug)
    return 0 if total > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
