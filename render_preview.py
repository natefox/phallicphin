"""Shaded top view of the STL next to the reference photo -> preview.png
usage: render_preview.py phallicphin.stl ref/photo.png preview.png"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection

m = trimesh.load(sys.argv[1])
v, n = m.vertices[m.faces], m.face_normals
L = np.array([-0.45, 0.55, 0.7]); L /= np.linalg.norm(L)
keep = n[:, 2] > -0.05
order = np.argsort(v[keep][:, :, 2].mean(1))
shade = (np.clip(n[keep] @ L, 0, 1) * 0.75 + 0.2)[order]
fig, (a0, a1) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 1.3]})
a0.imshow(plt.imread(sys.argv[2])[180:1060, 60:1180]); a0.axis("off"); a0.set_title("reference")
a1.add_collection(PolyCollection(v[keep][order][:, :, :2], facecolors=plt.cm.gray(shade), edgecolors="none", antialiased=False))
a1.set_xlim(-5, 185); a1.set_ylim(-18, 125); a1.set_aspect("equal"); a1.set_facecolor("#5a5f6a")
a1.set_title(f"model  {m.extents[0]:.0f} x {m.extents[1]:.0f} x {m.extents[2]:.1f} mm  (grid = 10 mm)")
a1.set_xticks(range(0, 181, 10)); a1.set_yticks(range(-10, 121, 10)); a1.grid(alpha=0.25); a1.tick_params(labelsize=7)
plt.tight_layout(); plt.savefig(sys.argv[3], dpi=90)
