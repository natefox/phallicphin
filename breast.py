"""boobphin: novelty Futures-base surf fin shaped like a breast in profile, nipple at the tip.

usage: breast.py [out.stl] [plan.png]
The outline comes from a hand-drawn sketch (SKETCH): the chest lies along the Futures root, a long gentle
slope rises from the front of the tab to the peak, and a full round curve drops to the rear of the tab.
STRETCH exaggerates the height. A small nipple sits at the peak, pointing up, ringed by a slightly puffy
areola. The face is one smooth hill (no dips) that rounds off toward the edges.
Coords: X along the tab (0 = front V-notch end), Y up from the tab top, Z = thickness.
Flat-backed (Z=0 face flat, all relief on +Z): prints lying down, no supports.
"""
import sys
import numpy as np
from manifold3d import CrossSection, Manifold, Mesh
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

# --- outline: from a dotted-line sketch (px, chest along the bottom, nipple at the top), rear end first ---
SKETCH = [(929, 466), (935, 454), (942, 436), (945, 414), (944, 389), (940, 367), (933, 350), (925, 337),
          (915, 325), (904, 313), (892, 302), (879, 292), (863, 283), (848, 275), (836, 269), (822, 264),
          (809, 258), (796, 253), (783, 247), (769, 241), (755, 237), (741, 233), (728, 230), (714, 230),
          (700, 234), (687, 241), (675, 248), (662, 256), (650, 264), (638, 273), (628, 281), (616, 290),
          (601, 301), (586, 312), (575, 320), (566, 327), (559, 333), (552, 338), (545, 342), (538, 347),
          (531, 352), (521, 359), (509, 368), (495, 377), (482, 385), (469, 393), (457, 400), (440, 412),
          (410, 430), (380, 447), (355, 459), (340, 466)]
SK_BASE_Y = 466.0            # the sketch's chest line...
SK_X = (340.0, 935.0)        # ...whose span lands on the root
ROOT_X = (3.0, 108.0)        # (tab is 0..113)
STRETCH = 1.6                # height exaggeration over the sketch (1 = as drawn: only ~45 mm tall)
NIP_X = 714.0                # nipple: at the sketch's peak (px x)
NIP_L, NIP_R = -0.35, 3.2    # nipple: capsule end this far out from the outline, radius (sticks up NIP_L + NIP_R)
NIP_DIR = 88.0               # ...pointing this way (deg from +X): nearly straight up
R_FILLET = 2.5               # fillet where the nipple meets the curve
R_FOLD = 4.0                 # crease radius where the rear curve tucks into the root


def sk_to_mm(p):
    p = np.asarray(p, float)
    k = (ROOT_X[1] - ROOT_X[0]) / (SK_X[1] - SK_X[0])
    return np.column_stack([ROOT_X[0] + (p[:, 0] - SK_X[0]) * k, (SK_BASE_Y - p[:, 1]) * k * STRETCH])


def body_points():
    pts = sk_to_mm(SKETCH)[::-1]                  # root front -> peak -> rear
    pts[:, 1] = np.maximum(pts[:, 1], 0.0)
    return np.vstack([pts, [(pts[-1, 0], -1.0), (ROOT_X[0], -1.0)]])


def nipple_base():
    """(point, outward direction) of the outline's peak near NIP_X."""
    p = sk_to_mm(SKETCH)
    i = int(np.argmin(np.abs(np.array(SKETCH)[:, 0] - NIP_X) + 1e3 * (np.array(SKETCH)[:, 1] > 300)))
    return p[i], np.array([np.cos(np.radians(NIP_DIR)), np.sin(np.radians(NIP_DIR))])


def lobe():
    """Center and radius of a disc in the full rear lobe (the thickest part)."""
    p = sk_to_mm(SKETCH)
    top = p[:, 1].max()
    return np.array([(nipple_base()[0][0] + p[:, 0].max()) / 2, 0.45 * top]), 0.28 * top


# --- thickness (flat back) ---
T_MAX = TAB_W                # mound / root thickness
R_LE = 10.0                  # leading edge: rounded nose this wide
LE_K = 2.6
R_TE = 34.0                  # the lower curve: a soft, full roll-off this wide (rounded breast)
TE_K = 1.9
H_MIN = 0.9                  # edge thickness
# crest: T_MAX over a core (the convex hull of the root and a disc in the lower curve), easing down with
# distance from it. A convex core gives one smooth hill with no saddle between the root and the breast.
T_LOW = 4.6                  # crest thickness this far...
D_FALL = 48.0                # ...from the core (smoothstep)
AREOLA_IN = 3.0              # areola center: this far inside the nipple base
AREOLA_R = 11.5
AREOLA_H = 0.6               # rim step: raised this much, eased over...
AREOLA_EASE = 1.2            # ...this width
AREOLA_PUFF = 0.9            # ...then puffs up a further cone toward the nipple
BUMPS = 7                    # little Montgomery bumps round the areola
BUMP_R, BUMP_H = 1.3, 0.35
NIP_T = 6.6                  # nipple dome peak (stands proud of the areola)
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
    from scipy.interpolate import splev, splprep
    p = body_points()
    tck, _ = splprep(p[:-2].T, s=len(p) * 0.15)                          # smooth the hand-drawn line
    curve = np.column_stack(splev(np.linspace(0, 1, 800), tck))
    body = Polygon(np.vstack([curve, p[-2:]])).buffer(0)
    body = body.buffer(R_FOLD, 64).buffer(-R_FOLD, 64)                # round the crease at the fold
    b, d = nipple_base()
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
    le = (ring[:, 1] > 0.5) & (ring[:, 0] < nipple_base()[0][0] - 6)   # the slope in front of the nipple
    from scipy.spatial import cKDTree
    pts = np.column_stack([X.ravel(), Y.ravel()])
    d_le = cKDTree(ring[le]).query(pts)[0].reshape(X.shape)
    te = ~le & (ring[:, 1] > 0.5)
    d_te = cKDTree(ring[te]).query(pts)[0].reshape(X.shape)
    # root needs no edge: the tab continues it

    lc, lr = lobe()
    core = Polygon([(ROOT_X[0], -1.0), (ROOT_X[1], -1.0), (ROOT_X[1], 0.0), (ROOT_X[0], 0.0)]).union(
        Point(*lc).buffer(lr, 64)).convex_hull
    cm = raster(core)
    e = np.clip(distance_transform_edt(~cm) * RES / D_FALL, 0, 1)
    crest = T_MAX - (T_MAX - T_LOW) * e * e * (3 - 2 * e)
    e = np.clip(d_le / R_LE, 0, 1)
    h_le = H_MIN + (crest - H_MIN) * (1 - (1 - e) ** LE_K)
    e = np.clip(d_te / R_TE, 0, 1)
    h_te = H_MIN + (T_MAX - H_MIN) * (1 - (1 - e) ** TE_K)
    h = smin(smin(h_le, h_te, 1.0), crest, 0.8)

    # areola: puffy disc (plateau raised, eased at its rim), with a ring of small bumps
    nb, nd = nipple_base()
    ac = nb - AREOLA_IN * nd
    r = np.hypot(X - ac[0], Y - ac[1])
    rise = AREOLA_H * np.clip((AREOLA_R - r) / AREOLA_EASE, 0, 1) ** 1.5
    rise += AREOLA_PUFF * np.clip(1 - r / AREOLA_R, 0, 1) ** 1.3
    for k in range(BUMPS):
        a = np.radians(200 + k * 360 / BUMPS)
        bx, by = ac[0] + 0.78 * AREOLA_R * np.cos(a), ac[1] + 0.78 * AREOLA_R * np.sin(a)
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
        ax.add_patch(plt.Circle(nipple_base()[0], AREOLA_R, fill=False, color="r"))
        ax.plot(*np.loadtxt(TAB_PROFILE, delimiter=",").T, color="k")
        ax.set_aspect("equal"); ax.grid(); fig.savefig(PLAN, dpi=70)
