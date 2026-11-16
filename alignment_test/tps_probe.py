"""Probe: TPS direction, and match residuals on page2 after the global homography."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(sys.argv[1])

# 1. Direction of applyTransformation: estimate(a -> b) then apply(a). Output == b or not?
rng = np.random.default_rng(0)
a = rng.uniform(0, 1000, (30, 2)).astype(np.float32)
b = a + np.float32([25, -10]) + rng.normal(0, 3, a.shape).astype(np.float32)
tps = cv2.createThinPlateSplineShapeTransformer(0)
matches = [cv2.DMatch(i, i, 0) for i in range(len(a))]
tps.estimateTransformation(a.reshape(1, -1, 2), b.reshape(1, -1, 2), matches)
_, out = tps.applyTransformation(a.reshape(1, -1, 2))
out = out.reshape(-1, 2)
print("max |apply(a) - b| =", float(np.abs(out - b).max()))
print("max |apply(a) - a| =", float(np.abs(out - a).max()))

# 2. Residuals of ratio-test matches against the RANSAC homography, for page2.
ref = cv2.imread(str(PROJECT / "templates/page2/reference.png"), cv2.IMREAD_GRAYSCALE)
scan = cv2.imread(str(PROJECT / "test_images/page2.jpg"), cv2.IMREAD_GRAYSCALE)
scale = ref.shape[0] / scan.shape[0]
small = cv2.resize(scan, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
orb = cv2.ORB_create(5000)
skp, sdes = orb.detectAndCompute(small, None)
rkp, rdes = orb.detectAndCompute(ref, None)
pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(sdes, rdes, k=2)
good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
src = np.float32([skp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) / scale
dst = np.float32([rkp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
res = np.linalg.norm(proj - dst.reshape(-1, 2), axis=1)
print("good", len(good), "inliers", int(mask.sum()))
for t in (5, 10, 20, 40, 80, 160):
    print(f"residual < {t}: {(res < t).sum()}")
# Where are the 5..80 px residual matches (the ones RANSAC dropped but might be the curl)?
mid = (res >= 5) & (res < 80)
d = dst.reshape(-1, 2)[mid]
print("5-80px residual points, ref coords (x,y) quartiles:")
print(np.percentile(d, [0, 25, 50, 75, 100], axis=0).round())

# 3. Timing of TPS on ~all points with residual < 80, evaluated on a 20px grid.
keep = res < 80
p_ref = d_all = dst.reshape(-1, 2)[keep]
p_al = proj[keep]
for reg in (0.0, 100.0, 10000.0):
    tps = cv2.createThinPlateSplineShapeTransformer(reg)
    m = [cv2.DMatch(i, i, 0) for i in range(len(p_ref))]
    t0 = time.perf_counter()
    tps.estimateTransformation(p_ref.reshape(1, -1, 2), p_al.reshape(1, -1, 2), m)
    t1 = time.perf_counter()
    gx, gy = np.meshgrid(np.linspace(0, ref.shape[1] - 1, 86), np.linspace(0, ref.shape[0] - 1, 111))
    grid = np.float32(np.dstack([gx, gy])).reshape(1, -1, 2)
    _, g = tps.applyTransformation(grid)
    t2 = time.perf_counter()
    _, at_pts = tps.applyTransformation(p_ref.reshape(1, -1, 2))
    fit = np.linalg.norm(at_pts.reshape(-1, 2) - p_al, axis=1)
    fit2 = np.linalg.norm(at_pts.reshape(-1, 2) - p_ref, axis=1)
    print(f"reg={reg}: n={len(p_ref)} estimate {t1-t0:.2f}s apply-grid {t2-t1:.2f}s "
          f"median|apply(ref)-aligned|={np.median(fit):.2f} median|apply(ref)-ref|={np.median(fit2):.2f}")
