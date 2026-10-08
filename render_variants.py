"""Shaded top views of fin STLs side by side.
usage: render_variants.py out.png a.stl [b.stl ...]"""
import sys, numpy as np, trimesh, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
fig, axs = plt.subplots(1, len(sys.argv) - 2, figsize=(9 * (len(sys.argv) - 2), 10))
L = np.array([-0.45, 0.55, 0.7]); L /= np.linalg.norm(L)
for ax, f in zip(np.atleast_1d(axs), sys.argv[2:]):
    m = trimesh.load(f); v, n = m.vertices[m.faces], m.face_normals
    k = n[:, 2] > -0.05; o = np.argsort(v[k][:, :, 2].mean(1))
    sh = np.clip(n[k] @ L, 0, 1) * 0.75 + 0.2
    ax.add_collection(PolyCollection(v[k][o][:, :, :2], facecolors=plt.cm.gray(sh[o]), edgecolors='none', antialiased=False))
    ax.set_xlim(-5, 185); ax.set_ylim(-18, 180); ax.set_aspect('equal'); ax.set_facecolor('#5a5f6a')
    ax.set_title(f"{f.rsplit('/', 1)[-1]}  {m.extents[0]:.0f} x {m.extents[1]:.0f} x {m.extents[2]:.1f} mm"); ax.grid(alpha=.25)
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=70)
