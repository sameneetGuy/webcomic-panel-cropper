import sys
import os
import cv2
import numpy as np

DEBUG = True  # set False if you don't want extra images/logs


def auto_crop_panels(path, out_root="cropped"):
    print(f"\n=== {path} ===")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("  ! Could not read image")
        return

    basename = os.path.splitext(os.path.basename(path))[0]
    base_dir = os.path.dirname(path) or "."
    out_dir = os.path.join(base_dir, out_root)
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------
    # Figure out format and build a BGR image for debug
    # -------------------------------------------------
    if img.ndim == 2:
        # grayscale, no alpha
        h, w = img.shape
        channels = 1
        alpha = None
        bgr_for_debug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        h, w, channels = img.shape
        if channels == 4:
            b, g, r, a = cv2.split(img)
            alpha = a
            bgr_for_debug = cv2.merge((b, g, r))
        else:
            alpha = None
            bgr_for_debug = img.copy()

    print(f"  Size: {w}x{h}, channels: {channels}")

    # -------------------------------------------------
    # Build "effective grayscale" where transparent = white
    # -------------------------------------------------
    if img.ndim == 2:
        gray = img
        alpha_present = False
    else:
        gray = cv2.cvtColor(bgr_for_debug, cv2.COLOR_BGR2GRAY)
        alpha_present = alpha is not None

    if alpha_present:
        # any pixel with alpha <= 10 is treated as pure white (background/gutter)
        eff_gray = gray.copy()
        eff_gray[alpha <= 10] = 255
    else:
        eff_gray = gray

    if DEBUG:
        debug_gray_path = os.path.join(out_dir, f"{basename}_effgray.png")
        cv2.imwrite(debug_gray_path, eff_gray)
        print(f"  -> saved effective grayscale: {debug_gray_path}")

    # -------------------------------------------------
    # Threshold: dark stuff (borders, line art) = ink
    # -------------------------------------------------
    # High threshold (245) so only very dark-ish lines become ink
    _, mask = cv2.threshold(eff_gray, 245, 255, cv2.THRESH_BINARY_INV)

    if DEBUG:
        nonzero = cv2.countNonZero(mask)
        print(f"  Ink pixels in mask: {nonzero} ({nonzero/(w*h):.3f} of image)")
        debug_mask_path = os.path.join(out_dir, f"{basename}_mask.png")
        cv2.imwrite(debug_mask_path, mask)
        print(f"  -> saved ink mask: {debug_mask_path}")

    # -------------------------------------------------
    # Light morphology: keep borders solid, don’t fuse gutters
    # -------------------------------------------------
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if DEBUG:
        debug_proc_path = os.path.join(out_dir, f"{basename}_processed.png")
        cv2.imwrite(debug_proc_path, processed)
        print(f"  -> saved processed mask: {debug_proc_path}")

    # -------------------------------------------------
    # Contours = candidate panels (works even if gutters are slightly tilted)
    # -------------------------------------------------
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  Raw contours: {len(contours)}")

    rects = []
    total_pixels = w * h
    # very relaxed filters; we can tighten later if needed
    min_area_fraction = 0.01   # 1% of image
    min_w_fraction    = 0.05   # 5% of width
    min_h_fraction    = 0.10   # 10% of height

    min_area = total_pixels * min_area_fraction
    min_w = w * min_w_fraction
    min_h = h * min_h_fraction

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < min_area:
            continue
        if cw < min_w or ch < min_h:
            continue
        rects.append((x, y, cw, ch))

    print(f"  Rectangles after filter: {len(rects)}")

    if not rects:
        print("  ! No panel-like rectangles found, using full image as one panel.")
        rects = [(0, 0, w, h)]

    # Debug overlay with rectangles
    if DEBUG:
        dbg = bgr_for_debug.copy()
        for i, (x, y, cw, ch) in enumerate(rects, start=1):
            cv2.rectangle(dbg, (x, y), (x+cw, y+ch), (0, 0, 255), 2)
            cv2.putText(dbg, str(i), (x+5, y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        debug_rect_path = os.path.join(out_dir, f"{basename}_debug_rects.png")
        cv2.imwrite(debug_rect_path, dbg)
        print(f"  -> saved debug rect overlay: {debug_rect_path}")

    # -------------------------------------------------
    # Sort panels in reading order: top-to-bottom, then left-to-right
    # -------------------------------------------------
    rects_sorted = sorted(rects, key=lambda r: r[1])  # by y
    row_tol = max(10, int(h * 0.05))

    rows = []
    for r in rects_sorted:
        x, y, cw, ch = r
        assigned = False
        for row in rows:
            if abs(y - row[0]) < row_tol:
                row[1].append(r)
                assigned = True
                break
        if not assigned:
            rows.append([y, [r]])

    panels = []
    rows.sort(key=lambda r: r[0])
    for _, row_rects in rows:
        row_rects.sort(key=lambda r: r[0])  # by x
        panels.extend(row_rects)

    # -------------------------------------------------
    # Crop from original image (keeps transparency if present)
    # -------------------------------------------------
    for i, (x, y, cw, ch) in enumerate(panels, start=1):
        crop = img[y:y+ch, x:x+cw]
        out_path = os.path.join(out_dir, f"{basename}_panel{i}.png")
        cv2.imwrite(out_path, crop)
        print(f"  -> panel {i}: {out_path}")
    
    # -------------------------------------------------
    # Cleanup: delete debug images
    # -------------------------------------------------
    debug_suffixes = [
        "_mask.png",
        "_processed.png",
        "_debug_rects.png",
        "_effgray.png"
    ]

    for suf in debug_suffixes:
        p = os.path.join(out_dir, f"{basename}{suf}")
        if os.path.exists(p):
            try:
                os.remove(p)
                if DEBUG_TEXT:
                    print(f"  Deleted debug file: {p}")
            except Exception as e:
                print(f"  Could not delete {p}: {e}")

    
    print(f"  Total panels exported: {len(panels)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_crop_panels.py strip1.png [strip2.png ...]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        auto_crop_panels(arg)
