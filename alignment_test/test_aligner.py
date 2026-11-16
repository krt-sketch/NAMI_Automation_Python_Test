"""Run aligner.align_page on the test pages and save reference/aligned overlays.

The alignment window opens for each page; press q in it to continue.
Overlay colors: black = both dark (aligned), red = reference only, cyan = aligned scan only.
Usage (from the project directory): python alignment_test/test_aligner.py
"""
import sys
import time
from pathlib import Path

import cv2

PROJECT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from aligner import TEMPLATES_DIR, align_page  # noqa: E402

for page in ("page1", "page2", "page3"):
    t = time.perf_counter()
    result = align_page(PROJECT / "test_images" / f"{page}.jpg", page)
    print(f"{page}: {result.matches} matches, {result.inliers} inliers, {result.control_points} control points, "
          f"{time.perf_counter() - t:.1f}s including time waiting for q")
    ref = cv2.imread(str(TEMPLATES_DIR / page / "reference.png"), cv2.IMREAD_GRAYSCALE)
    overlay = cv2.merge([ref, ref, result.image])
    path = OUT / f"aligner_{page}_overlay.jpg"
    cv2.imwrite(str(path), cv2.resize(overlay, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    print(f"  wrote {path}")
