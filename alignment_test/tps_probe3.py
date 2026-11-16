"""Probe 3: homography, then coarse-to-fine local ORB matching -> per-cell median -> TPS.

Overlay colors: black = both dark (aligned), red = reference only, cyan = aligned scan only.
Usage: python tps_probe3.py <project dir> <output dir> <page name>
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
PAGE = sys.argv[3]

RADII = [100, 30, 10]  # local match search window per pass, reference pixels
CELL = 60  # one control point per CELL x CELL block of the reference
MIN_PER_CELL = 3  # matches needed for a cell's median to count
NEIGHBORS = 8  # cells compared in the consistency check
MAX_DEVIATION = 8.0  # px a cell may differ from its neighbors' median offset
REG = 100.0  # TPS regularization
GRID_STEP = 20  # TPS is evaluated on this grid, then interpolated

ref = cv2.imread(str(PROJECT / f"templates/{PAGE}/reference.png"), cv2.IMREAD_GRAYSCALE)
scan = cv2.imread(str(PROJECT / f"test_images/{PAGE}.jpg"), cv2.IMREAD_GRAYSCALE)
h, w = ref.shape
t0 = time.perf_counter()

# --- Global homography (unchanged from aligner.py) ---
scale = h / scan.shape[0]
small = cv2.resize(scan, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
orb = cv2.ORB_create(5000)
skp, sdes = orb.detectAndCompute(small, None)
rkp, rdes = orb.detectAndCompute(ref, None)
rpts = np.float32([k.pt for k in rkp])
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
pairs = bf.knnMatch(sdes, rdes, k=2)
good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
src = np.float32([skp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) / scale
dst = np.float32([rkp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
coarse = cv2.warpPerspective(scan, H, (w, h))
print(f"{PAGE}: homography inliers {int(mask.sum())}/{len(good)}  ({time.perf_counter()-t0:.2f}s)")

# Reference grid for evaluating the TPS, and its corner-aligned upsampling coordinates.
gw, gh = w // GRID_STEP + 1, h // GRID_STEP + 1
gx, gy = np.meshgrid(np.linspace(0, w - 1, gw, dtype=np.float32), np.linspace(0, h - 1, gh, dtype=np.float32))
grid = np.dstack([gx, gy]).reshape(1, -1, 2)
ux, uy = np.meshgrid(np.arange(w, dtype=np.float32) * (gw - 1) / (w - 1),
                     np.arange(h, dtype=np.float32) * (gh - 1) / (h - 1))


def dense_map(tps):
    """Per-reference-pixel (map_x, map_y) into the homography-aligned image."""
    _, g = tps.applyTransformation(grid)
    g = g.reshape(gh, gw, 2)
    return (cv2.remap(np.ascontiguousarray(g[..., 0]), ux, uy, cv2.INTER_LINEAR),
            cv2.remap(np.ascontiguousarray(g[..., 1]), ux, uy, cv2.INTER_LINEAR))


def local_matches(img, radius):
    """ORB matches img -> reference, only between keypoints within `radius` of each other."""
    kp, des = orb.detectAndCompute(img, None)
    pts = np.float32([k.pt for k in kp])
    near = np.zeros((len(pts), len(rpts)), np.uint8)
    for i in range(0, len(pts), 1000):
        d = pts[i:i + 1000, None, :] - rpts[None, :, :]
        near[i:i + 1000] = (np.abs(d) < radius).all(axis=2)
    found = bf.knnMatch(des, rdes, k=2, mask=near)
    ms = [p[0] for p in found if len(p) == 1 or (len(p) == 2 and p[0].distance < 0.75 * p[1].distance)]
    return (np.float32([rkp[m.trainIdx].pt for m in ms]),  # reference point
            np.float32([kp[m.queryIdx].pt for m in ms]))   # same feature in img


def cell_medians(p, q):
    """One (reference point, offset) per CELL block: the median of its matches."""
    d = q - p
    cells = (p // CELL).astype(np.int32)
    key = cells[:, 1] * (w // CELL + 1) + cells[:, 0]
    cp, cd = [], []
    for k in np.unique(key):
        sel = key == k
        if sel.sum() >= MIN_PER_CELL:
            cp.append(np.median(p[sel], axis=0))
            cd.append(np.median(d[sel], axis=0))
    return np.float32(cp), np.float32(cd)


def consistent(cp, cd):
    """Keep cells whose offset is within MAX_DEVIATION of their neighbors' median offset."""
    dist = np.linalg.norm(cp[:, None] - cp[None], axis=2)
    np.fill_diagonal(dist, np.inf)
    nn = np.argsort(dist, axis=1)[:, :NEIGHBORS]
    dev = np.linalg.norm(cd - np.median(cd[nn], axis=1), axis=1)
    return dev < MAX_DEVIATION


tps = None
current = coarse
for radius in RADII:
    t = time.perf_counter()
    p, q = local_matches(current, radius)
    cp, cd = cell_medians(p, q)
    ok = consistent(cp, cd)
    cp, cd = cp[ok], cd[ok]
    # Feature found at cp+cd in `current`; its position in the homography-aligned image:
    target = cp + cd if tps is None else tps.applyTransformation((cp + cd).reshape(1, -1, 2))[1].reshape(-1, 2)
    tps = cv2.createThinPlateSplineShapeTransformer(REG)
    tps.estimateTransformation(cp.reshape(1, -1, 2), target.reshape(1, -1, 2),
                               [cv2.DMatch(i, i, 0) for i in range(len(cp))])
    _, fitted = tps.applyTransformation(cp.reshape(1, -1, 2))
    err = np.linalg.norm(fitted.reshape(-1, 2) - target, axis=1)
    map_x, map_y = dense_map(tps)
    current = cv2.remap(coarse, map_x, map_y, cv2.INTER_LINEAR)
    off = np.linalg.norm(cd, axis=1)
    print(f"  radius {radius:3d}: {len(p)} matches -> {ok.size} cells, {ok.sum()} consistent; "
          f"offset 50/90/max {np.percentile(off, 50):.1f}/{np.percentile(off, 90):.1f}/{off.max():.1f}px; "
          f"fit err max {err.max():.2f}px  ({time.perf_counter()-t:.2f}s)")

# Final single-resample from the original scan: reference -> TPS -> H^-1 -> scan.
_, g = tps.applyTransformation(grid)
g = cv2.perspectiveTransform(g.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(gh, gw, 2)
map_x = cv2.remap(np.ascontiguousarray(g[..., 0]), ux, uy, cv2.INTER_LINEAR)
map_y = cv2.remap(np.ascontiguousarray(g[..., 1]), ux, uy, cv2.INTER_LINEAR)
final = cv2.remap(scan, map_x, map_y, cv2.INTER_LINEAR)
print(f"  total {time.perf_counter()-t0:.2f}s")

for name, img in (("homography", coarse), ("tps", final)):
    ov = cv2.merge([ref, ref, img])
    cv2.imwrite(str(OUT / f"{PAGE}_{name}_full.jpg"), cv2.resize(ov, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(OUT / f"{PAGE}_{name}_topright.jpg"), ov[150:850, 700:1700])
vis = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
for (x, y), (dx, dy) in zip(cp, cd):
    cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
cv2.imwrite(str(OUT / f"{PAGE}_control_points.jpg"), cv2.resize(vis, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
