"""Probe 2: homography + locally re-matched ORB + TPS, on page2. Writes overlay images.

Overlay colors: black = both dark (aligned), red = reference only, cyan = aligned scan only.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
PAGE = sys.argv[3] if len(sys.argv) > 3 else "page2"

LOCAL_RADIUS = 40
REG = 100.0
GRID_STEP = 20
MAX_FIT_ERROR = 3.0

ref = cv2.imread(str(PROJECT / f"templates/{PAGE}/reference.png"), cv2.IMREAD_GRAYSCALE)
scan = cv2.imread(str(PROJECT / f"test_images/{PAGE}.jpg"), cv2.IMREAD_GRAYSCALE)
h, w = ref.shape
t0 = time.perf_counter()

# Global homography (same as aligner.py today).
scale = h / scan.shape[0]
small = cv2.resize(scan, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
orb = cv2.ORB_create(5000)
skp, sdes = orb.detectAndCompute(small, None)
rkp, rdes = orb.detectAndCompute(ref, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
pairs = bf.knnMatch(sdes, rdes, k=2)
good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
src = np.float32([skp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) / scale
dst = np.float32([rkp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
coarse = cv2.warpPerspective(scan, H, (w, h))
t1 = time.perf_counter()

# Local re-match: only reference keypoints within LOCAL_RADIUS (square window) are candidates.
akp, ades = orb.detectAndCompute(coarse, None)
apts = np.float32([k.pt for k in akp])
rpts = np.float32([k.pt for k in rkp])
near = np.zeros((len(apts), len(rpts)), np.uint8)
for i in range(0, len(apts), 1000):
    d = apts[i:i + 1000, None, :] - rpts[None, :, :]
    near[i:i + 1000] = (np.abs(d) < LOCAL_RADIUS).all(axis=2)
local = bf.knnMatch(ades, rdes, k=2, mask=near)
lm = [p[0] for p in local if len(p) == 1 or (len(p) == 2 and p[0].distance < 0.75 * p[1].distance)]
lm_single = sum(1 for p in local if len(p) == 1)
p_ref = np.float32([rkp[m.trainIdx].pt for m in lm])
p_al = np.float32([akp[m.queryIdx].pt for m in lm])
print(f"{PAGE}: homography inliers {int(mask.sum())}/{len(good)}; local matches {len(lm)} (single-candidate {lm_single})")
disp = np.linalg.norm(p_al - p_ref, axis=1)
print("local displacement px percentiles 50/90/99/max:", np.percentile(disp, [50, 90, 99, 100]).round(1))


def fit(p_ref, p_al):
    tps = cv2.createThinPlateSplineShapeTransformer(REG)
    tps.estimateTransformation(p_ref.reshape(1, -1, 2), p_al.reshape(1, -1, 2),
                               [cv2.DMatch(i, i, 0) for i in range(len(p_ref))])
    _, out = tps.applyTransformation(p_ref.reshape(1, -1, 2))
    return tps, np.linalg.norm(out.reshape(-1, 2) - p_al, axis=1)


tps, err = fit(p_ref, p_al)
print("fit error px percentiles 50/90/99/max:", np.percentile(err, [50, 90, 99, 100]).round(2))
keep = err < MAX_FIT_ERROR
tps, err = fit(p_ref[keep], p_al[keep])
print(f"refit on {keep.sum()} points; error 50/90/99/max:", np.percentile(err, [50, 90, 99, 100]).round(2))

# Coarse grid in reference coords -> coarse-aligned coords -> original scan coords.
gw, gh = w // GRID_STEP + 1, h // GRID_STEP + 1
gx, gy = np.meshgrid(np.linspace(0, w - 1, gw, dtype=np.float32), np.linspace(0, h - 1, gh, dtype=np.float32))
_, g = tps.applyTransformation(np.dstack([gx, gy]).reshape(1, -1, 2))
g = cv2.perspectiveTransform(g.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(gh, gw, 2)
# Upsample the grid to every reference pixel (corner-aligned, bilinear) using remap itself.
fx, fy = np.meshgrid(np.arange(w, dtype=np.float32) * (gw - 1) / (w - 1),
                     np.arange(h, dtype=np.float32) * (gh - 1) / (h - 1))
map_x = cv2.remap(np.ascontiguousarray(g[..., 0]), fx, fy, cv2.INTER_LINEAR)
map_y = cv2.remap(np.ascontiguousarray(g[..., 1]), fx, fy, cv2.INTER_LINEAR)
fine = cv2.remap(scan, map_x, map_y, cv2.INTER_LINEAR)
t2 = time.perf_counter()
print(f"time: homography {t1-t0:.2f}s, refine {t2-t1:.2f}s")

def overlay(img):
    return cv2.merge([ref, ref, img])


for name, img in (("homography", coarse), ("tps", fine)):
    ov = overlay(img)
    cv2.imwrite(str(OUT / f"{PAGE}_{name}_full.jpg"), cv2.resize(ov, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(OUT / f"{PAGE}_{name}_topright.jpg"), ov[150:850, 700:1700])

# Control point locations on the reference.
cp = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
for (x, y) in p_ref[keep]:
    cv2.circle(cp, (int(x), int(y)), 6, (0, 0, 255), -1)
cv2.imwrite(str(OUT / f"{PAGE}_control_points.jpg"), cv2.resize(cp, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
