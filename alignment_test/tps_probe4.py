"""Probe 4: like probe 3, but with dense ORB features and block-wise local matching,
so small table text and grid corners also become control points.

Overlay colors: black = both dark (aligned), red = reference only, cyan = aligned scan only.
Usage: python tps_probe4.py <project dir> <output dir> <page name>
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
DENSE_FEATURES = 50000  # ORB features for the refinement passes
BLOCK = 200  # reference keypoints are matched block by block (speed only)
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
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
pairs = bf.knnMatch(sdes, rdes, k=2)
good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
src = np.float32([skp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) / scale
dst = np.float32([rkp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
coarse = cv2.warpPerspective(scan, H, (w, h))
print(f"{PAGE}: homography inliers {int(mask.sum())}/{len(good)}  ({time.perf_counter()-t0:.2f}s)")

# Dense reference features, computed once.
t = time.perf_counter()
dense = cv2.ORB_create(DENSE_FEATURES)
dkp, ddes = dense.detectAndCompute(ref, None)
dpts = np.float32([k.pt for k in dkp])
print(f"  dense reference features: {len(dkp)}  ({time.perf_counter()-t:.2f}s)")

# Reference grid for evaluating the TPS, and its corner-aligned upsampling coordinates.
gw, gh = w // GRID_STEP + 1, h // GRID_STEP + 1
gx, gy = np.meshgrid(np.linspace(0, w - 1, gw, dtype=np.float32), np.linspace(0, h - 1, gh, dtype=np.float32))
grid = np.dstack([gx, gy]).reshape(1, -1, 2)
ux, uy = np.meshgrid(np.arange(w, dtype=np.float32) * (gw - 1) / (w - 1),
                     np.arange(h, dtype=np.float32) * (gh - 1) / (h - 1))


def upsample(g):
    g = g.reshape(gh, gw, 2)
    return (cv2.remap(np.ascontiguousarray(g[..., 0]), ux, uy, cv2.INTER_LINEAR),
            cv2.remap(np.ascontiguousarray(g[..., 1]), ux, uy, cv2.INTER_LINEAR))


def local_matches(img, radius):
    """Match each reference keypoint to img keypoints within `radius` (square window).

    Reference keypoints are the queries, so every candidate of a keypoint is in the
    same block's train set and the ratio test sees all of them.
    """
    kp, des = dense.detectAndCompute(img, None)
    pts = np.float32([k.pt for k in kp])
    p_out, q_out = [], []
    for y0 in range(0, h, BLOCK):
        for x0 in range(0, w, BLOCK):
            qi = np.flatnonzero((dpts[:, 0] >= x0) & (dpts[:, 0] < x0 + BLOCK) &
                                (dpts[:, 1] >= y0) & (dpts[:, 1] < y0 + BLOCK))
            ti = np.flatnonzero((pts[:, 0] >= x0 - radius) & (pts[:, 0] < x0 + BLOCK + radius) &
                                (pts[:, 1] >= y0 - radius) & (pts[:, 1] < y0 + BLOCK + radius))
            if len(qi) == 0 or len(ti) == 0:
                continue
            near = (np.abs(dpts[qi, None, :] - pts[None, ti, :]) < radius).all(axis=2).astype(np.uint8)
            for pair in bf.knnMatch(ddes[qi], des[ti], k=2, mask=near):
                if len(pair) == 1 or (len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance):
                    p_out.append(dpts[qi[pair[0].queryIdx]])
                    q_out.append(pts[ti[pair[0].trainIdx]])
    return np.float32(p_out), np.float32(q_out)


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
    current = cv2.remap(coarse, *upsample(tps.applyTransformation(grid)[1]), cv2.INTER_LINEAR)
    off = np.linalg.norm(cd, axis=1)
    print(f"  radius {radius:3d}: {len(p)} matches -> {ok.size} cells, {ok.sum()} consistent; "
          f"offset 50/90/max {np.percentile(off, 50):.1f}/{np.percentile(off, 90):.1f}/{off.max():.1f}px; "
          f"fit err max {err.max():.2f}px  ({time.perf_counter()-t:.2f}s)")

# Final single resample from the original scan: reference -> TPS -> H^-1 -> scan.
g = tps.applyTransformation(grid)[1]
g = cv2.perspectiveTransform(g.reshape(-1, 1, 2), np.linalg.inv(H))
final = cv2.remap(scan, *upsample(g), cv2.INTER_LINEAR)
print(f"  total {time.perf_counter()-t0:.2f}s")

for name, img in (("homography", coarse), ("tps", final)):
    ov = cv2.merge([ref, ref, img])
    cv2.imwrite(str(OUT / f"p4_{PAGE}_{name}_full.jpg"), cv2.resize(ov, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(OUT / f"p4_{PAGE}_{name}_topright.jpg"), ov[150:850, 700:1700])
vis = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
for x, y in cp:
    cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
cv2.imwrite(str(OUT / f"p4_{PAGE}_control_points.jpg"), cv2.resize(vis, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
