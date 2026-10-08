"""boobphin: novelty Futures-base surf fin shaped like a breast in profile, nipple at the tip.

usage: breast.py [out.stl] [plan.png]
The leading edge is the upper slope, the full rounded trailing edge is the lower curve tucking into a
crease (inframammary fold) at the rear of the tab, and the nipple sticks out at the top-right tip where
phallicphin has its head, ringed by a slightly puffy areola.
Coords: X along the tab (0 = front V-notch end), Y up from the tab top, Z = thickness.
Flat-backed (Z=0 face flat, all relief on +Z): prints lying down, no supports.
"""
import sys
import numpy as np
from manifold3d import CrossSection, Manifold, Mesh
from scipy.interpolate import splev, splprep
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import Point, Polygon, box
from skimage.measure import marching_cubes
from skimage.draw import polygon as fill_poly

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = ARGS[0] if ARGS else "boobphin.stl"
PLAN = ARGS[1] if len(ARGS) > 1 else None
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."

# --- tab (Futures box fit), same as fin.py ---
TAB_W = 7.2
TAB_EDGE_R = 0.6
TAB_PROFILE = f"{HERE}/tab_profile_mm.csv"

# --- outline: closed spline through these (mm), CCW from the front of the root ---
BODY = [(4.0, -1.0), (60.0, -1.0), (106.0, -1.0),            # root (1 mm into the tab)
        (110.0, 2.5), (114.0, 5.0), (121.0, 6.0),              # crease: tucks in at the rear of the tab
        (134.0, 9.5), (147.0, 17.5), (158.0, 29.0), (166.0, 44.0), (170.5, 60.0),   # lower curve, full
        (171.0, 76.0), (168.0, 92.0), (162.5, 105.0), (156.0, 114.5),
        (147.0, 122.0),                                         # nipple sits on this corner
        (135.0, 120.5), (118.0, 114.5), (100.0, 104.5), (82.0, 91.0), (64.0, 75.0),   # upper slope = LE
        (45.0, 55.0), (29.0, 34.0), (15.0, 14.0)]
NIP_BASE = (152.5, 119.5)    # nipple: base center on the outline...
NIP_DIR = 40.0               # ...pointing this way (deg), roughly normal to the outline there
NIP_L, NIP_R = 6.5, 5.6      # sticks out this far, radius (capsule)
R_FILLET = 2.5               # fillet where the nipple meets the curve

# --- thickness (flat back) ---
T_MAX = TAB_W                # mound / root thickness
R_LE = 10.0                  # leading edge: rounded nose this wide
LE_K = 2.6
R_TE = 34.0                  # the lower curve: a soft, full roll-off this wide (rounded breast)
TE_K = 1.9
H_MIN = 0.9                  # edge thickness
TAPER = ((12.0, T_MAX), (40.0, 6.0))   # crest: full thickness at the root, easing to T_FLAT...
MOUND_C, MOUND_R = (124.0, 62.0), 66.0  # ...under a round mound, T_MAX at its center, falling as r^2
MOUND_EDGE = 5.0                        # ...to this at MOUND_R
AREOLA_C = (150.0, 116.5)    # areola center, just inside the nipple base
AREOLA_R = 17.0
AREOLA_H = 0.6               # rim step: raised this much, eased over...
AREOLA_EASE = 1.2            # ...this width
AREOLA_PUFF = 0.9            # ...then puffs up a further cone toward the nipple
BUMPS = 7                    # little Montgomery bumps round the areola
BUMP_R, BUMP_H = 1.3, 0.35
NIP_T = 7.6                  # nipple dome peak (stands proud of the areola)
NIP_WALL = 3.0               # nipple side wall height at its rim
NIP_Q = 2.2                  # dome profile 1 - r^Q

RES = 0.2


def tab():
    pts = np.loadtxt(TAB_PROFILE, delimiter=",")
    t = Manifold.extrude(CrossSection([pts.tolist()]), TAB_W + 2).translate((0, 0, -1))
    r = TAB_EDGE_R
    rr = CrossSection.batch_hull([CrossSection.circle(r, 24).translate(p) for p in
                                  [(-13.2 + r, r), (-13.2 + r, TAB_W - r)]] +
                                 [CrossSection.square((1, TAB_W)).translate((5, 0))])
    bar = Manifold.extrude(rr, 130).transform(np.array([[0, 0, 1, -5], [1, 0, 0, 0], [0, 1, 0, 0]], float))
    return t ^ bar


def smin(a, b, k):
    w = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - w * w * k / 4


def outline():
    p = np.array(BODY + [BODY[0]])
    tck, _ = splprep(p.T, s=0, per=1)
    body = Polygon(np.column_stack(splev(np.linspace(0, 1, 1200), tck))).buffer(0)
    a = np.radians(NIP_DIR)
    d = np.array([np.cos(a), np.sin(a)])
    b = np.array(NIP_BASE)
    nip = Point(*(b - 4 * d)).buffer(NIP_R, 64).union(Point(*(b + NIP_L * d)).buffer(NIP_R, 64)).convex_hull
    poly = body.union(nip)
    poly = poly.buffer(R_FILLET, 64).buffer(-R_FILLET, 64)         # fillet the nipple's root
    poly = poly.intersection(box(-10, -1.0, 300, 300))            # stop 1 mm into the tab
    return poly, nip, b + NIP_L * d * 0.45


def thickness_field(poly, nip, nip_c):
    x0, y0, x1, y1 = poly.bounds
    x0, y0 = x0 - 2, y0 - 2
    nx, ny = int((x1 + 2 - x0) / RES) + 1, int((y1 + 2 - y0) / RES) + 1
    X, Y = np.meshgrid(x0 + np.arange(nx) * RES, y0 + np.arange(ny) * RES)

    def raster(g):
        m = np.zeros((ny, nx), bool)
        c = np.array(g.exterior.coords)
        rr, cc = fill_poly((c[:, 1] - y0) / RES, (c[:, 0] - x0) / RES, m.shape)
        m[rr, cc] = True
        return m

    fm = raster(poly)
    # distance to the leading edge vs the rest: split the boundary at the top of the curve and at the root
    ring = np.array(poly.exterior.segmentize(0.2).coords)[:-1]
    le = (ring[:, 1] > 0.5) & (ring[:, 0] < 118) & (ring[:, 1] > ring[:, 0] * 0.6 - 30)   # upper slope
    from scipy.spatial import cKDTree
    pts = np.column_stack([X.ravel(), Y.ravel()])
    d_le = cKDTree(ring[le]).query(pts)[0].reshape(X.shape)
    te = ~le & (ring[:, 1] > 0.5)
    d_te = cKDTree(ring[te]).query(pts)[0].reshape(X.shape)
    # root needs no edge: the tab continues it

    crest = np.interp(Y, [TAPER[0][0], TAPER[1][0]], [TAPER[0][1], TAPER[1][1]])
    rm = np.hypot(X - MOUND_C[0], Y - MOUND_C[1]) / MOUND_R
    mound = T_MAX - (T_MAX - MOUND_EDGE) * rm ** 2
    crest = -smin(-crest, -mound, 1.0)        # smooth max
    e = np.clip(d_le / R_LE, 0, 1)
    h_le = H_MIN + (crest - H_MIN) * (1 - (1 - e) ** LE_K)
    e = np.clip(d_te / R_TE, 0, 1)
    h_te = H_MIN + (T_MAX - H_MIN) * (1 - (1 - e) ** TE_K)
    h = smin(smin(h_le, h_te, 1.0), crest, 0.8)

    # areola: puffy disc (plateau raised, eased at its rim), with a ring of small bumps
    r = np.hypot(X - AREOLA_C[0], Y - AREOLA_C[1])
    rise = AREOLA_H * np.clip((AREOLA_R - r) / AREOLA_EASE, 0, 1) ** 1.5
    rise += AREOLA_PUFF * np.clip(1 - r / AREOLA_R, 0, 1) ** 1.3
    for k in range(BUMPS):
        a = np.radians(200 + k * 360 / BUMPS)
        bx, by = AREOLA_C[0] + 0.78 * AREOLA_R * np.cos(a), AREOLA_C[1] + 0.78 * AREOLA_R * np.sin(a)
        rb = np.hypot(X - bx, Y - by)
        rise += BUMP_H * np.sqrt(np.clip(1 - (rb / BUMP_R) ** 2, 0, 1))
    h = h + rise * np.clip((h - H_MIN) / 1.5, 0, 1)

    # nipple: dome on a short side wall, peak at nip_c, out to the nipple's own outline
    nm = raster(nip.buffer(0.4))
    rn = np.hypot(X - nip_c[0], Y - nip_c[1]) / (NIP_R + 3.0)
    dome = NIP_WALL + (NIP_T - NIP_WALL) * (1 - np.clip(rn, 0, 1) ** NIP_Q)
    dome = np.where(rn < 1, dome, 0)
    h = np.where(nm | (rn < 1), np.maximum(h, dome), h)
    h = gaussian_filter(h, 0.6 / RES)          # soften any seams
    h = np.clip(h, H_MIN, T_MAX + 0.6)
    sdf = np.where(fm, distance_transform_edt(fm), -distance_transform_edt(~fm)) * RES
    return h, sdf, (x0, y0), fm


def body(poly, nip, nip_c):
    import trimesh
    h, sdf, (x0, y0), fm = thickness_field(poly, nip, nip_c)
    nz = int((h.max() + 2) / RES) + 1
    z = (-1 + np.arange(nz) * RES).astype(np.float32)
    vol = np.minimum(np.minimum(sdf.astype(np.float32)[..., None], h.astype(np.float32)[..., None] - z), z)
    vol = np.pad(vol, 1, constant_values=-1)
    vv, f, _, _ = marching_cubes(vol, 0.0, spacing=(RES, RES, RES))
    vv = vv - RES
    vv = np.column_stack([x0 + vv[:, 1], y0 + vv[:, 0], -1 + vv[:, 2]])
    m = trimesh.Trimesh(vv, f[:, ::-1], process=True)
    if m.volume < 0:
        m.invert()
    hin = np.where(fm, h, np.nan)
    return Manifold(Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                         tri_verts=np.asarray(m.faces, np.uint32))), (np.nanmin(hin), np.nanmax(hin))


if __name__ == "__main__":
    import trimesh
    poly, nip, nip_c = outline()
    fin, (hmin, hmax) = body(poly, nip, nip_c)
    part = (fin + tab()).simplify(0.01)
    mm = part.to_mesh()
    out = trimesh.Trimesh(mm.vert_properties[:, :3], mm.tri_verts)
    if out.body_count > 1:
        with np.errstate(invalid="ignore"):
            out = trimesh.util.concatenate([b for b in out.split(only_watertight=False) if abs(b.volume) > 1.0])
    print("thickness", round(hmin, 2), "..", round(hmax, 2), "watertight", out.is_watertight, "bodies",
          out.body_count, "extents", out.extents.round(1), "vol cm3", round(out.volume / 1000, 1))
    out.export(OUT)
    print("wrote", OUT)
    if PLAN:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(*np.array(poly.exterior.coords).T)
        ax.plot(*np.array(nip.exterior.coords).T, "--")
        ax.add_patch(plt.Circle(AREOLA_C, AREOLA_R, fill=False, color="r"))
        ax.plot(*np.loadtxt(TAB_PROFILE, delimiter=",").T, color="k")
        ax.set_aspect("equal"); ax.grid(); fig.savefig(PLAN, dpi=70)
