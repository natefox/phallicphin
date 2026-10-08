"""US-box (longboard single-fin) base tab, measured from a stock 9" longboard fin.

Coordinates: X along the board (front/nose end at X=0, fin rakes toward +X), Y up the fin,
the fin root line at Y=0 so the tab hangs below it (Y from -24.5 to 0). The tab spans
Z=0..BOX_T (lies flat on the bed); blades built on it should share the Z=0 back face.
"""
import numpy as np
from manifold3d import CrossSection, Manifold

BOX_L = 149.6          # tab length
BOX_H = 24.46          # tab height below the root line
BOX_T = 8.63           # tab thickness (fits a 3/8" box)
NOSE_H = 13.05         # front nose: full-height from X=0 down to Y=-NOSE_H...
NOSE_DROP = 0.3        # ...its underside dropping this much by X=NOSE_X
NOSE_X = 27.7
FRONT_C, FRONT_R = (22.8, -41.03), 28.12   # arc from the nose underside down to the flat bottom
REAR_C, REAR_R = (140.1, -14.93), 9.52     # rounded rear-bottom corner, concentric with the pin hole
REAR_TOP_X = 149.23    # rear face leans in slightly over its top 4 mm
PIN_D = 4.76           # 3/16" cross pin hole
SLOT_X = (9.3, 14.2)   # set-screw plate slot, through the tab height, centered in thickness
SLOT_W = 4.73
SEG = 64


def tab_profile():
    """Side outline (X, Y) of the tab, CCW."""
    bot = -BOX_H
    pts = [(0.0, -NOSE_H), (NOSE_X, -NOSE_H - NOSE_DROP)]
    cx, cy = FRONT_C
    a0 = np.arctan2(-NOSE_H - NOSE_DROP - cy, NOSE_X - cx)
    xb = cx + np.sqrt(FRONT_R**2 - (bot - cy) ** 2)            # where the arc meets the bottom
    a1 = np.arctan2(bot - cy, xb - cx)
    a1 = a1 if a1 < a0 else a1 - 2 * np.pi
    a1 = max(a1, -np.pi / 2)                                     # (bottom is tangent-ish; clamp)
    for a in np.linspace(a0, a1, 24)[1:]:
        pts.append((cx + FRONT_R * np.cos(a), max(cy + FRONT_R * np.sin(a), bot)))
    pts.append((REAR_C[0], bot))
    for a in np.linspace(-np.pi / 2, 0, 16)[1:]:
        pts.append((REAR_C[0] + REAR_R * np.cos(a), REAR_C[1] + REAR_R * np.sin(a)))
    pts += [(BOX_L, -4.2), (REAR_TOP_X, 0.0), (0.3, 0.0)]
    # drop near-duplicates from the arc/bottom join
    out = [pts[0]]
    for p in pts[1:]:
        if np.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 0.05:
            out.append(p)
    return np.array(out)


def tab(extend_up=0.0):
    """The tab as a Manifold, Z=0..BOX_T. extend_up raises its top edge into the blade (for a clean union)."""
    p = tab_profile().copy()
    p[np.isclose(p[:, 1], 0.0), 1] = extend_up
    cs = CrossSection([p.tolist()])
    if cs.area() < 0:
        cs = CrossSection([p[::-1].tolist()])
    body = Manifold.extrude(cs, BOX_T)
    pin = Manifold.cylinder(BOX_T + 2, PIN_D / 2, PIN_D / 2, SEG).translate((REAR_C[0], REAR_C[1], -1))
    slot = Manifold.cube((SLOT_X[1] - SLOT_X[0], BOX_H + 2, SLOT_W)).translate(
        (SLOT_X[0], -BOX_H - 1, (BOX_T - SLOT_W) / 2))
    return body - pin - slot


if __name__ == "__main__":
    import sys, trimesh
    m = tab().to_mesh()
    t = trimesh.Trimesh(m.vert_properties[:, :3], m.tri_verts)
    print("tab extents", np.round(t.extents, 2), "watertight", t.is_watertight)
    if len(sys.argv) > 1:
        t.export(sys.argv[1])
