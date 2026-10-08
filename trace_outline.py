"""Trace the fin silhouette from the reference photo into outline_mm.csv.

usage: trace_outline.py ref/photo.png outline_mm.csv
Scale comes from the Futures tab in the photo (657 px == 113 mm tab length).
Output coords: X along the tab (0 = front/V-notch end), Y up from the tab top.
"""
import sys
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

PHOTO, OUT = sys.argv[1], sys.argv[2]
TAB_X0_PX, TAB_X1_PX = 106.5, 763.5   # tab ends in the photo
TAB_TOP_PX = 935.0                    # tab top / fin root line
MM_PER_PX = 113.0 / (TAB_X1_PX - TAB_X0_PX)
HEAD_BOX_PX = (915, 225, 1145, 490)   # x0, y0, x1, y1: head, where shaded faces need a lower threshold

img = cv2.imread(PHOTO)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = ((hsv[..., 2] > 165) & (hsv[..., 1] < 75)).astype(np.uint8) * 255
# the shaded corona face and glans underside fall under the threshold: around the head only, also take
# dimmer plastic touching the fin (concrete is ~100, plastic ~190, the shaded face ~150-170)
x0, y0, x1, y1 = HEAD_BOX_PX
lo = np.zeros_like(mask)
lo[y0:y1, x0:x1] = ((hsv[y0:y1, x0:x1, 2] > 145) & (hsv[y0:y1, x0:x1, 1] < 75)) * 255
lo = cv2.morphologyEx(lo, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
mask |= lo & cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
mask[1018:] = 0                                    # tape measure
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
mask = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
# fill holes
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
c = gaussian_filter1d(c, 2.5, axis=0, mode="wrap")
c = c[::2]

xy = np.column_stack([(c[:, 0] - TAB_X0_PX) * MM_PER_PX, (TAB_TOP_PX - c[:, 1]) * MM_PER_PX])
np.savetxt(OUT, xy, fmt="%.3f", delimiter=",", header="x_mm,y_mm (tab top at y=0)")
print(f"{len(xy)} pts, {MM_PER_PX:.4f} mm/px, x {xy[:,0].min():.1f}..{xy[:,0].max():.1f}, "
      f"y {xy[:,1].min():.1f}..{xy[:,1].max():.1f}")
