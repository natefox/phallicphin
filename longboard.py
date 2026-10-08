"""longboard: phallicphin as a 9" longboard single fin (US box), template-first.

usage: longboard.py [out.stl] [plan.png] [--sym]   (--sym: double-sided, exported standing on the tab)
The plan outline starts from a stock 9" longboard pivot fin (raked LE, rounded tip, chord, depth);
the anatomy from ref/photo.png is mapped onto it: the testicle lobes fill the rear root region (with the
gap notch on the trailing side and the little hook tooth at the rear of the tab), the shaft is the swept
blade (its underside running parallel to the leading edge), and the glans (corona rim + dome) is the tip.
Coords: X along the board (nose end of the tab = 0, rakes toward +X), Y up from the root line (tab below),
Z = thickness. Flat-backed (Z=0 face flat, all relief on +Z): prints lying down, no supports.
"""
import sys
import numpy as np

HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
sys.path.insert(0, HERE)
from boxbase import tab, BOX_T, BOX_L  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = ARGS[0] if ARGS else "phallicphin_longboard.stl"
PLAN = ARGS[1] if len(ARGS) > 1 else None
SYM = "--sym" in sys.argv        # double-sided foil centered on the tab (prints standing on the tab)

# --- plan outline -------------------------------------------------------------------------------------
# leading edge of the reference longboard fin (silhouette traced from its STL, root line at Y=0)
LE = [(22.8, 0.0), (24.9, 5.8), (27.3, 11.2), (29.6, 16.6), (31.9, 22.1), (34.2, 27.4), (36.5, 32.8),
      (38.9, 38.3), (41.2, 43.6), (43.8, 49.0), (46.5, 54.4), (49.2, 59.7), (51.9, 65.0), (54.8, 70.3),
      (57.6, 75.6), (60.7, 80.6), (63.7, 85.8), (66.9, 90.8), (70.0, 95.9), (73.3, 100.8), (76.7, 105.7),
      (80.3, 110.6), (83.8, 115.2), (87.4, 119.9), (91.2, 124.4), (95.1, 128.9), (99.0, 133.3),
      (102.9, 137.7), (107.0, 142.0), (111.2, 146.2), (115.5, 150.3), (119.9, 154.3), (124.4, 158.2),
      (128.9, 161.9), (133.5, 165.4), (138.2, 169.0), (143.0, 172.4), (148.0, 175.7), (153.0, 178.9),
      (158.2, 181.8), (163.2, 184.9), (168.6, 187.6), (174.0, 190.3), (179.4, 192.8), (184.7, 195.0),
      (190.3, 196.9), (195.8, 198.8), (201.4, 200.2), (207.0, 201.4), (212.7, 202.4), (218.5, 202.9)]
# shaft: its underside runs parallel to the LE at this width (mm, normal to the LE) vs the LE's height Y
W_SHAFT = [(70.0, 66.0), (110.0, 58.0), (150.0, 50.0), (180.0, 45.0), (200.0, 42.0)]
CORONA_LE_X = 192.0      # where the corona meets the leading edge (the head is everything beyond)
# glans in a local frame at the corona: a along the shaft axis, b across it (+ toward the LE side)
HEAD_L = 64.0            # corona to tip
HEAD_W = 60.0            # width at the corona rim (shaft neck is ~42: ~7 mm flare each side)
HEAD_B0 = 0.5           # head center offset across the axis (- = toward the underside)
HEAD_P = 2.6             # superellipse exponent of the dome outline (2 = ellipse, higher = boxier)
HEAD_TILT = -8.0        # head axis droop vs the LE tangent at the corona (deg), like the photo
HEAD_LIP = 2.0           # rim lip thickness behind the corona line (plan, mm)
R_FILLET = 2.0           # fillet the concave corona/neck corners
R_TIP = 2.0              # round the flare tips
# lower trailing edge: from the gap's inner end round the two testicle lobes to the hook tooth at the rear
# root (control points, spline-smoothed; drawn after the photo, scaled to longboard size)
GAP_TIP = (128.0, 86.0)  # inner end of the gap notch
R_GAP = 7.0              # its rounded end
GAP_JOIN = 20.0          # the shaft underside is followed down to this far (X) behind the gap tip
TE_STEP = 18.0           # spline control spacing along the shaft underside
LOBES = [((166.0, 66.0), 23.0), ((168.0, 33.0), 23.0)]   # two testicles: (center, radius)
LOBE_CLEFT = 4.0         # how far the cleft between them dips into the outline
HOOK = [(150.0, 10.0), (146.0, 5.0), (149.0, 0.0)]   # neck under the lower lobe, then the tooth at the root

# --- relief (flat back Z=0, everything on +Z) -----------------------------------------------------------
T_ROOT = 8.5             # plateau thickness at the root (<= BOX_T 8.63)
T_SHAFT = 4.6            # shaft crest by the head
TAPER_Y = (40.0, 175.0)  # crest tapers T_ROOT -> T_SHAFT between these heights (smoothstep)
R_LE = 13.0              # rounded leading edge band
LE_K = 2.6               # nose profile 1-(1-e)^K
R_TE = ((60.0, 36.0), (200.0, 28.0))   # trailing taper width vs X (linear)
TE_K = 1.25
H_MIN = 0.8              # trailing edge thickness
LAYER = 0.2
# plateau (full T): the root triangle of the photo, sharp apex under the gap
PLATEAU = [(40.0, 0.0), (52.0, 38.0), (72.0, 72.0), (98.0, 102.0), (110.0, 80.0), (117.0, 56.0),
           (119.0, 28.0), (116.0, 0.0)]
PLAT_SLOPE = 0.22
PLAT_EASE = 10.0
SMIN_K = 1.0
EDGE_CAP = 0.6
H_HOOK = 2.4             # min thickness of the hook tooth and the lobe neck by the tab end
LOBE_T = 7.8             # testicle dome peaks
LOBE_BASE = 2.6          # spherical-cap dome from this height at LOBE_SPREAD * radius
LOBE_SPREAD = 1.08
LOBE_ZONE = 1.75         # balls fully replace the body within r, blending out to LOBE_ZONE * r
R_LOBE_EDGE = 9.0        # rounded lobe edges: quarter-round over this width...
H_LOBE_EDGE = 1.4        # ...down to this edge thickness
# head: dome around HEAD_PEAK_A (fraction of HEAD_L along the axis), raised corona rim
T_HEAD = 7.4
H_CORONA = 5.6           # rim height on the corona side (~1.2 mm proud of the shaft)
D_RIM = 8.0
W_CORONA = 2.2
H_SKIRT = 2.4
HEAD_Q = 1.7
HEAD_PEAK_A = 0.45
DOME_P = 2.1            # dome contour exponent (rounder than the outline)
R_SKIRT = 2.5
H_WALL = 2.6
MEATUS = (1.6, 0.5)      # little dimple at the dome peak: radius, depth

RES = 0.25               # grid / voxel size, mm
FIT_BED = 250.0


def le_frame():
    from shapely.geometry import LineString
    le = LineString(LE)
    s = np.arange(0, le.length, 0.5)
    p = np.array([le.interpolate(d).coords[0] for d in s])
    from scipy.ndimage import gaussian_filter1d
    t = np.gradient(gaussian_filter1d(p, 6, axis=0, mode="nearest"), axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.column_stack([t[:, 1], -t[:, 0]])     # inward (toward the trailing side)
    return p, t, n


def spline(pts, n=400, s=0.0):
    from scipy.interpolate import splev, splprep
    tck, _ = splprep(np.array(pts, float).T, s=s, k=min(3, len(pts) - 1))
    return np.array(splev(np.linspace(0, 1, n), tck)).T


def outline():
    from shapely.geometry import Point, Polygon, box
    from shapely.ops import unary_union
    p, t, n = le_frame()
    ic = int(np.argmin(np.abs(p[:, 0] - CORONA_LE_X)))
    w = np.interp(p[:, 1], *zip(*W_SHAFT))
    under = p + w[:, None] * n                   # shaft underside
    # shaft underside from the corona down to where it passes over the gap tip
    lo = int(np.argmin(np.abs(under[:ic, 0] - (GAP_TIP[0] + GAP_JOIN))))
    # head frame
    C = p[ic] + (w[ic] / 2) * n[ic]
    a0 = np.arctan2(t[ic, 1], t[ic, 0]) + np.radians(HEAD_TILT)
    u, v = np.array([np.cos(a0), np.sin(a0)]), np.array([-np.sin(a0), np.cos(a0)])   # v toward the LE
    th = np.linspace(-np.pi / 2, np.pi / 2, 181)
    ca, sa = np.cos(th), np.sin(th)
    A = HEAD_L * np.sign(ca) * np.abs(ca) ** (2 / HEAD_P)
    B = HEAD_B0 + HEAD_W / 2 * np.sign(sa) * np.abs(sa) ** (2 / HEAD_P)
    head_pts = [C + a * u + b * v for a, b in zip(A, B)]
    head_pts = [C - HEAD_LIP * u + (HEAD_B0 - HEAD_W / 2) * v] + head_pts + [C - HEAD_LIP * u + (HEAD_B0 + HEAD_W / 2) * v]
    head = Polygon(head_pts).buffer(0)
    # trailing edge, one spline: shaft underside -> round gap end -> two lobes (with a cleft) -> hook tooth
    (c1, r1), (c2, r2) = LOBES
    g = np.array(GAP_TIP)
    k = np.arange(ic, lo, -int(TE_STEP / 0.5))
    gap = [g + R_GAP * np.array([np.cos(a), np.sin(a)]) for a in np.radians([115, 150, 180, 215, 250])]
    lobes = [(g[0] + 14, g[1] - R_GAP - 1.0), (c1[0] - 4, c1[1] + r1 - 0.5), (c1[0] + 10, c1[1] + r1 - 3),
             (c1[0] + r1 * 0.85, c1[1] + r1 * 0.45), (c1[0] + r1, c1[1] - 4),
             ((c1[0] + c2[0]) / 2 + max(r1, r2) - LOBE_CLEFT, (c1[1] + c2[1]) / 2),
             (c2[0] + r2, c2[1] + 2), (c2[0] + r2 * 0.75, c2[1] - r2 * 0.6), (c2[0] + 4, c2[1] - r2 + 1),
             (c2[0] - 10, c2[1] - r2 + 3)] + HOOK
    te = spline([tuple(q) for q in under[k]] + [tuple(q) for q in gap] + lobes, 1500)
    body = np.vstack([p[:ic + 1], te, [(BOX_L - 1, -1.0), (20.0, -1.0)]])
    poly = Polygon(body).buffer(0)
    poly = unary_union([poly, head]).buffer(0)
    # fillet the concave neck/corona corners, round the flare tips (head zone only)
    zone = Point(C).buffer(HEAD_W * 0.9)
    fix = poly.buffer(R_FILLET, 64).buffer(-R_FILLET, 64).buffer(-R_TIP, 64).buffer(R_TIP, 64)
    poly = poly.difference(zone).union(fix.intersection(zone)).buffer(0.01).buffer(-0.01)
    poly = poly.intersection(box(-50, -1.0, 400, 400))
    return poly, head.intersection(poly), (C, u, v)


def fit(xy, bed=FIT_BED):
    best = None
    for d in np.arange(0, 180, 0.25):
        r = np.radians(d)
        q = xy @ np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
        e = q.max(0) - q.min(0)
        if best is None or e.max() < best[1].max():
            best = (d, e)
    return best


def plan_png(poly, path, mesh=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from boxbase import tab_profile
    fig, ax = plt.subplots(figsize=(13, 12))
    try:
        import trimesh
        ref = trimesh.load(f"{HERE}/local/longboard/longboard-fin.STL")
        rv = ref.vertices[:, :2] - (0, 25.73)
        import cv2
        R, x0, y0 = 0.25, -5, -30
        im = np.zeros((int(270 / R), int(300 / R)), np.uint8)
        for tt in rv[ref.faces]:
            cv2.fillPoly(im, [np.round((tt - (x0, y0)) / R).astype(np.int32)], 1)
        cs, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)[:, 0, :] * R + (x0, y0)
        ax.plot(*np.vstack([c, c[:1]]).T, "--", color="#d0443a", lw=1.3, label="reference longboard fin")
    except Exception as e:  # reference is local-only
        print("no reference overlay:", e)
    tp = tab_profile()
    ax.fill(*tp.T, color="#9aa4b4", alpha=0.6, label="US box tab")
    ax.fill(*np.array(poly.exterior.coords).T, color="#3d6fb0", alpha=0.35)
    ax.plot(*np.array(poly.exterior.coords).T, color="#1d3f70", lw=1.6, label="longboard")
    allxy = np.vstack([tp, np.array(poly.exterior.coords)]) if mesh is None else mesh.vertices[:, :2]
    d, e = fit(allxy)
    r = np.radians(d)
    Rm = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
    q = allxy @ Rm
    mid = (q.max(0) + q.min(0)) / 2
    sq = mid + FIT_BED / 2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    ax.plot(*(sq @ Rm.T).T, color="#2a9d55", lw=1.4, label=f"250 mm bed square at {d:.2f} deg")
    ax.set_title(f"longboard plan: {np.ptp(allxy[:, 0]):.1f} x {np.ptp(allxy[:, 1]):.1f} mm; at {d:.2f} deg fits "
                 f"{e[0]:.1f} x {e[1]:.1f} (margin {FIT_BED - e.max():.1f} mm)")
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-20, 300, 10), minor=True); ax.set_yticks(np.arange(-40, 260, 10), minor=True)
    ax.set_xticks(np.arange(0, 300, 50)); ax.set_yticks(np.arange(-50, 260, 50))
    ax.grid(which="minor", alpha=0.15); ax.grid(which="major", alpha=0.45)
    ax.set_xlim(-15, 285); ax.set_ylim(-45, 250)
    ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close(fig)
    return d, e


def smin(a, b, k):
    """Smooth minimum: min(a, b) except within k of each other, where it rounds the seam."""
    w = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - w * w * k / 4


def thickness_field(poly, frame):
    import cv2
    from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d
    from shapely.geometry import LineString, Point, Polygon, box
    C, u, v = frame
    ring = np.array(poly.exterior.segmentize(0.1).coords)[:-1]
    x0, y0, x1, y1 = poly.bounds
    x0, y0 = x0 - 2, y0 - 2
    nx, ny = int((x1 + 2 - x0) / RES) + 1, int((y1 + 2 - y0) / RES) + 1
    Y = y0 + np.arange(ny)[:, None] * RES + 0 * np.arange(nx)[None, :]
    X = x0 + np.arange(nx)[None, :] * RES + 0 * Y
    g = lambda pts: np.round((np.asarray(pts) - (x0, y0)) / RES).astype(np.int32)

    def raster(p):
        im = np.zeros((ny, nx), np.uint8)
        cv2.fillPoly(im, [g(p.exterior.coords)], 1)
        return im.astype(bool)

    fm = raster(poly)
    sdf2 = np.where(fm, distance_transform_edt(fm) * RES - RES / 2, -(distance_transform_edt(~fm) * RES - RES / 2))
    # head = beyond the corona line (a > -HEAD_LIP in the head frame)
    A = (X - C[0]) * u[0] + (Y - C[1]) * u[1]
    Bc = (X - C[0]) * v[0] + (Y - C[1]) * v[1]
    hm = fm & (A > -HEAD_LIP) & (np.hypot(X - C[0], Y - C[1]) < 2 * HEAD_L)
    ra = (ring - C) @ u
    on_head = (ra > -HEAD_LIP - 0.2) & (np.hypot(*(ring - C).T) < 2 * HEAD_L)
    le = LineString(LE)
    on_le = np.array([le.distance(Point(q)) < 0.4 for q in ring]) & ~on_head & (ring[:, 1] > 0.3)
    on_te = ~on_le & ~on_head & (ring[:, 1] > 0.3)
    idx = np.arange(len(ring))

    def dist_to(sel):
        im = np.ones((ny, nx), np.uint8)
        q = g(ring[sel])
        for run in np.split(q, np.flatnonzero(np.diff(idx[sel]) > 1) + 1):
            if len(run) > 1:
                cv2.polylines(im, [run], False, 0, 1)
        return distance_transform_edt(im) * RES

    def smooth(d, r=3.0):   # round the medial-axis creases, keep the true distance near the edge
        w = np.clip(d / (2 * r), 0, 1)
        return w * gaussian_filter(d, r / RES) + (1 - w) * d

    d_le, d_te = smooth(dist_to(on_le)), smooth(dist_to(on_te))
    t = np.clip((Y - TAPER_Y[0]) / (TAPER_Y[1] - TAPER_Y[0]), 0, 1)
    T = T_ROOT + (T_SHAFT - T_ROOT) * t * t * (3 - 2 * t)
    r_te = np.interp(X, *zip(*R_TE))
    e_le, e_te = np.clip(d_le / R_LE, 0, 1), np.clip(d_te / r_te, 0, 1)
    h_le = (T - LAYER) * (1 - (1 - e_le) ** LE_K)
    h_te = H_MIN + (T - H_MIN) * (1 - (1 - e_te) ** TE_K)
    pm = raster(Polygon(PLATEAU).intersection(poly))
    d_p = distance_transform_edt(~pm) * RES
    dq = np.minimum(d_p, PLAT_EASE)
    h_pl = T - LAYER - PLAT_SLOPE * (dq * dq / (2 * PLAT_EASE) + np.maximum(d_p - PLAT_EASE, 0))
    h_out = np.minimum(np.minimum(h_le, -smin(-h_te, -h_pl, SMIN_K)), T - LAYER)
    # testicles: spherical-cap domes smooth-maxed onto the body; inside the lobe zone the trailing edge
    # gets a rounded (quarter-round) profile instead of the long foil taper, so they read as balls
    wl, hb = np.zeros_like(X), np.zeros_like(X)
    for (c, r) in LOBES:
        dc = np.hypot(X - c[0], Y - c[1])
        rr = np.clip(dc / (r * LOBE_SPREAD), 0, 1)
        hb = -smin(-hb, -(LOBE_BASE + (LOBE_T - LOBE_BASE) * np.sqrt(1 - rr ** 2)), 2 * SMIN_K)
        wl = np.maximum(wl, np.clip((LOBE_ZONE * r - dc) / ((LOBE_ZONE - 1) * r), 0, 1))
    wl = wl * wl * (3 - 2 * wl)
    h_out = h_out + wl * (hb - h_out)   # inside the lobes the balls replace the body
    wl = gaussian_filter(wl, 2.0 / RES)
    e_cap = np.clip(d_te / (EDGE_CAP * r_te), 0, 1)
    h_cap = H_MIN + (T_ROOT + SMIN_K - H_MIN) * (1 - (1 - e_cap) ** TE_K)
    e_l = np.clip(d_te / R_LOBE_EDGE, 0, 1)
    h_capl = H_LOBE_EDGE + (T_ROOT + SMIN_K - H_LOBE_EDGE) * np.sqrt(1 - (1 - e_l) ** 2)
    h_cap = h_cap + wl * (h_capl - h_cap)
    h = np.maximum(smin(np.where(pm, T, h_out), h_cap, SMIN_K), H_MIN)
    hx, hy = HOOK[0]
    w = np.clip((X - (hx - 8)) / 4, 0, 1) * np.clip((hy + 4 - Y) / 4, 0, 1)
    h = np.maximum(h, H_MIN + (H_HOOK - H_MIN) * w)
    h_body = h

    # glans: dome around the peak on a skirt wall, raised rim at the corona
    P = C + HEAD_PEAK_A * HEAD_L * u + HEAD_B0 * v
    head = poly.intersection(Polygon([C - HEAD_LIP * u - 100 * v, C - HEAD_LIP * u + 100 * v,
                                      C + 200 * u + 100 * v, C + 200 * u - 100 * v]))
    corona = LineString([C - HEAD_LIP * u - 60 * v, C - HEAD_LIP * u + 60 * v])
    # analytic superellipse dome in the head frame, peak at P; rim height raised near the corona
    ap = HEAD_PEAK_A * HEAD_L
    da = A - ap
    an = np.where(da > 0, da / (HEAD_L - ap), da / (ap + HEAD_LIP))
    bn = (Bc - HEAD_B0) / (HEAD_W / 2)
    rho = np.clip((np.abs(an) ** DOME_P + np.abs(bn) ** DOME_P) ** (1 / DOME_P), 0, 1)
    b = H_SKIRT + (H_CORONA - H_SKIRT) * np.clip(1 - (A + HEAD_LIP) / D_RIM, 0, 1) ** 1.5
    dh = b + (T_HEAD - b) * (1 - rho ** HEAD_Q)
    rad = np.hypot(X - P[0], Y - P[1])
    mr, md = MEATUS
    dh = dh - md * np.clip(1 - (rad / mr) ** 2, 0, 1) ** 2 * (rad < mr)
    d_hd = dist_to(on_head)
    e = np.clip(d_hd / R_SKIRT, 0, 1)
    dh = np.minimum(dh, H_WALL + (H_SKIRT + 0.8 - H_WALL) * np.sqrt(1 - (1 - e) ** 2) + 10 * np.maximum(d_hd - R_SKIRT, 0))
    im = np.ones((ny, nx), np.uint8)
    cv2.polylines(im, [g(corona.coords)], False, 0, 1)
    sc = np.clip(distance_transform_edt(im) * RES / W_CORONA, 0, 1)
    sc = sc * sc * (3 - 2 * sc)
    h = np.where(hm, h_body + (dh - h_body) * sc, h)
    _, (iy, ix) = distance_transform_edt(~fm, return_indices=True)
    h = gaussian_filter(h[iy, ix], 0.6 / RES)
    h = np.clip(h, H_MIN, BOX_T)
    return h, sdf2, (x0, y0), fm


def fin_body(poly, frame):
    import trimesh
    from manifold3d import Manifold, Mesh
    from skimage.measure import marching_cubes
    h, sdf2, (x0, y0), fm = thickness_field(poly, frame)
    nz = int((h.max() + 2) / RES) + 1
    z = (-1 + np.arange(nz) * RES).astype(np.float32)
    if SYM:   # same total thickness, split evenly about the tab's mid-plane
        hz = (h.astype(np.float32) / 2)[..., None] - np.abs(z - BOX_T / 2)
        vol = np.minimum(sdf2.astype(np.float32)[..., None], hz)
    else:
        vol = np.minimum(np.minimum(sdf2.astype(np.float32)[..., None], h.astype(np.float32)[..., None] - z), z)
    vol = np.pad(vol, 1, constant_values=-1)
    vv, f, _, _ = marching_cubes(vol, 0.0, spacing=(RES, RES, RES))
    vv = vv - RES
    vv = np.column_stack([x0 + vv[:, 1], y0 + vv[:, 0], -1 + vv[:, 2]])
    m = trimesh.Trimesh(vv, f[:, ::-1], process=True)
    if m.volume < 0:
        m.invert()
    hin = np.where(fm, h, np.nan)
    stats = dict(h_max=float(np.nanmax(hin)), h_min=float(np.nanmin(hin)))
    return Manifold(Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                         tri_verts=np.asarray(m.faces, np.uint32))), stats, (h, (x0, y0), fm)


if __name__ == "__main__":
    import trimesh
    poly, head, frame = outline()
    body, stats, _ = fin_body(poly, frame)
    print("fin status", body.status(), "vol", round(body.volume()), {k: round(v, 2) for k, v in stats.items()})
    part = (body + tab()).simplify(0.01)
    mm = part.to_mesh()
    out = trimesh.Trimesh(mm.vert_properties[:, :3], mm.tri_verts)
    if out.body_count > 1:
        with np.errstate(invalid="ignore"):
            out = trimesh.util.concatenate([b for b in out.split(only_watertight=False) if abs(b.volume) > 1.0])
    d, e = fit(out.vertices[:, :2])
    print("watertight", out.is_watertight, "bodies", out.body_count, "faces", len(out.faces),
          "bounds", out.bounds.round(2).tolist(), "vol cm3", round(out.volume / 1000, 1))
    print(f"fit: {d:.2f} deg -> {e[0]:.1f} x {e[1]:.1f} mm (margin {FIT_BED - e.max():.1f})")
    if SYM:   # stand it on the tab: Y up -> Z, tab bottom on the bed
        out.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        out.apply_translation(-out.bounds[0])
        print("upright extents", out.extents.round(1))
    out.export(OUT)
    print("wrote", OUT)
    if PLAN:
        plan_png(poly, PLAN, out)
