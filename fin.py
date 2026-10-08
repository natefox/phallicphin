"""phallicphin: novelty Futures-base surf fin, shape traced from ref/photo.png.

usage: fin.py [out.stl] [rake|upright]
  rake (default): as photographed, shaft + head sweep back. upright: shaft + head re-bent to arc upward.
Coords: X along the tab (0 = front V-notch end), Y up from the tab top, Z = thickness.
Built flat-backed (Z=0 face flat, all shaping on +Z) so it prints lying down with no
supports and layers running the length of the fin (strong at the root).
"""
import sys
import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold, Mesh
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import split
from skimage.measure import marching_cubes
import cv2

OUT = sys.argv[1] if len(sys.argv) > 1 else "phallicphin.stl"
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "rake"
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."

# --- tab (Futures box fit) ---
TAB_W = 7.2          # tab thickness; FuturesRearFinBase4mm = 7.2, Futures blank = 7.0
TAB_EDGE_R = 0.6     # rounding on the long bottom edges of the tab
TAB_PROFILE = f"{HERE}/tab_profile_mm.csv"   # side profile measured from a Futures rear-fin base

# --- fin body: foil-like relief, flat back (Z=0), all shaping on +Z ---
T_ROOT = TAB_W       # plateau thickness (root, blade, shaft crest)
T_SHAFT = 5.5        # crest thickness the shaft tapers to before the head
TAPER_Y = (55.0, 95.0)   # taper from T_ROOT to T_SHAFT between these heights
R_LE = 9.4           # leading edge (outer convex curve): nose this wide, rising to T - LAYER
LE_K = 2.8           # nose profile 1-(1-e)^K: steep (lines merge) to ~6 mm, then a gentle cluster
LAYER = 0.2          # one print layer of the original (terrace step)
R_TE = ((55.0, 34.0), (115.0, 24.0))   # trailing edges (gap, lobe, hook): taper width (mm) vs X, linear
TE_K = 1.2           # trailing taper exponent (1 = straight wedge, as the even contour spacing in the photo shows)
D_SMOOTH = 3.0       # blur (mm) on the edge-distance fields: rounds the ridge creases like the original
H_MIN = 0.8          # trailing edge thickness (no feather edges)
H_HOOK = 2.2         # min thickness of the hook tooth by the rear tab end (X > HOOK_X, Y < HOOK_Y - 4): strength
HOOK_X, HOOK_Y = 100.0, 8.0
# plateau at full thickness: the triangle (sharp apex) + the tongue into the lobe, traced from the photo's
# first layer contour (px, closed via the root): LE-side step line, apex, tongue upper edge, tip, lower edge
PLATEAU = [(190, 950), (198, 934), (225, 885), (252, 845), (285, 770), (322, 694), (385, 622), (445, 565),
           (453, 620), (477, 667), (510, 690), (563, 703), (630, 707), (700, 711), (750, 716), (775, 721),
           (783, 727), (776, 734), (740, 742), (700, 747), (660, 757), (633, 773), (610, 795), (598, 830),
           (600, 870), (608, 905), (615, 935), (615, 950)]
PLAT_SLOPE = 0.25    # fall-off away from the plateau, mm/mm (photo: ~0.7 mm per 0.2 mm layer)
PLAT_EASE = 9.0      # ...reached (quadratic ease-in) over this distance: photo's gentle first ~9 mm
SMIN_K = 1.0         # smooth min/max blend (mm of height, ~3 mm in plan at these slopes)
EDGE_CAP = 0.6       # ...capped near trailing edges by the taper squeezed to this fraction of its width
R_FILLET = 1.5       # head outline: fillet concave kinks (shaft-to-flare corners, glans underside)
R_TIP = 2.0          # ...and round the convex flare tips
# head: conical dome around HEAD_PEAK on a skirt wall, split off at the CORONA polyline
T_HEAD = 7.2         # dome peak
H_CORONA = 5.6       # dome edge height on the corona side: a rim standing ~1.2 mm proud of the shaft
D_RIM = 6.0          # ...fading to the skirt height over this distance round the flares (edge band caps the tips)
W_CORONA = 1.7       # the corona face slopes up from the shaft over this width (plan view)
H_SKIRT = 2.8        # dome edge height on the glans side
HEAD_Q = 1.4         # dome profile 1 - r^Q (1 = cone, 2 = paraboloid)
R_SKIRT = 2.0        # rounded edge band of the glans / flares...
H_WALL = 2.4         # ...ending on a side wall this tall (solid cap, rounded knobs)
# corona line + dome peak, traced from the photo (px -> mm, same scale as trace_outline.py)
S = 113.0 / 657.0
px = lambda x, y: ((x - 106.5) * S, (935.0 - y) * S)
CORONA = [px(935, 200), px(943, 245), px(946, 330), px(958, 400), px(978, 470), px(990, 520)]
HEAD_PEAK = px(990, 325)

RES = 0.2            # voxel size, mm

# --- upright variant: leading edge straightened, shaft + head re-bent to arc upward ---
# spine: the leading edge offset 12.5 mm inward (= shaft centerline up top), root to past the head tip (mm)
SPINE = [(16.8, 4.3), (21.6, 13.1), (26.5, 21.8), (31.6, 30.4), (37.0, 38.8), (42.9, 46.9), (49.2, 54.7),
         (55.8, 62.2), (62.8, 69.3), (70.3, 75.9), (78.3, 81.9), (86.7, 87.2), (95.7, 91.6), (105.1, 95.0),
         (114.8, 97.6), (124.5, 99.8), (134.4, 101.2), (144.4, 101.0), (150, 101), (160, 100.6),
         (175, 99.6), (200, 98)]
# heading (deg) the spine is re-bent to, at these arclengths along it (linear between, held after the last);
# before BEND_FROM the rake's own leading edge is kept, blending into the table over S_IN
BEND_S = [120]
BEND_DEG = [32.0]   # hold the leading edge's heading where it reaches ~(75, 95) instead of drooping
BEND_FROM, S_IN = 105.0, 20.0
W_IN, W_OUT = 20.0, 33.0   # carried fully within W_IN of the spine on the blade/lobe side, fading to 0 by W_OUT
# optional fuller back for a steeply bent shaft (BACK_FILL): back edge in RAKE mm (so it rides with the bend), from the
# gap tip across the gap to under the corona; the strip between it and the shaft is added to the outline (usual relief)
BACK_FILL = False
BACK_RAKE = [(88.2, 49.7), (88.6, 51.9), (89.4, 53.9), (90.7, 55.9), (92.4, 57.9), (94.4, 60.0), (96.7, 62.0),
             (99.3, 63.9), (102.0, 65.8), (104.7, 67.4), (107.4, 69.0), (109.9, 70.4), (112.3, 71.6), (114.4, 72.7),
             (116.5, 73.8), (118.5, 74.8), (120.5, 75.8), (122.6, 76.8), (124.8, 77.8), (127.2, 79.0), (130.5, 81.1),
             (134.4, 83.4), (138.3, 85.6), (142.5, 87.6)]
BACK_FILLET = 3.0    # round the joins where the new back edge meets the gap tip and the shaft


def tab():
    """Solid Futures tab: 113 x 13.2 side profile (V-notch front, set-screw slot rear),
    extruded TAB_W thick, with rounded long bottom edges."""
    pts = np.loadtxt(TAB_PROFILE, delimiter=",")                  # CCW, tab top at Y=0
    cs = CrossSection([pts.tolist()])
    t = Manifold.extrude(cs, TAB_W + 2).translate((0, 0, -1))   # bar below sets the Z faces
    # round the bottom long edges: intersect with a rounded-rect bar along X
    r = TAB_EDGE_R
    rr = CrossSection.batch_hull([CrossSection.circle(r, 24).translate(p) for p in
                                  [(-13.2 + r, r), (-13.2 + r, TAB_W - r)]] +
                                 [CrossSection.square((1, TAB_W)).translate((5, 0))])  # section in (Y, Z)
    bar = Manifold.extrude(rr, 130)                                    # length along z
    bar = bar.transform(np.array([[0, 0, 1, -5], [1, 0, 0, 0], [0, 1, 0, 0]], float))  # z->X, a->Y, b->Z
    return t ^ bar


def smin(a, b, k):
    """Smooth minimum (polynomial): equals min(a, b) except within k of each other, where it rounds the seam."""
    w = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - w * w * k / 4


def thickness_field(poly=None):
    if poly is None:
        poly = Polygon(np.loadtxt(f"{HERE}/outline_mm.csv", delimiter=",")).buffer(0)
    # head only: fillet the concave corners and round the flare tips (closing, then opening)
    zone = box(CORONA[0][0] - 8, 60, 300, 300)
    fix = poly.buffer(R_FILLET, 64).buffer(-R_FILLET, 64).buffer(-R_TIP, 64).buffer(R_TIP, 64)
    poly = poly.difference(zone).union(fix.intersection(zone)).buffer(0.01).buffer(-0.01)
    ring = np.array(poly.exterior.segmentize(0.1).coords)[:-1]
    outline = np.roll(ring, -int(np.argmax(ring[:, 0])), 0)       # start at the head tip
    fin = poly.intersection(box(-50, -1.0, 300, 300))     # fin above the tab, 1mm overlap
    corona = LineString(CORONA)
    parts = split(fin, corona)
    head = max(parts.geoms, key=lambda g: g.centroid.x)

    plat = Polygon([px(x, y) for x, y in PLATEAU]).intersection(fin)
    x0, y0, x1, y1 = fin.bounds
    x0, y0 = x0 - 2, y0 - 2
    nx, ny = int((x1 + 2 - x0) / RES) + 1, int((y1 + 2 - y0) / RES) + 1

    def raster(p):
        im = np.zeros((ny, nx), np.uint8)
        g = lambda ring: np.round((np.array(ring.coords) - (x0, y0)) / RES).astype(np.int32)
        cv2.fillPoly(im, [g(p.exterior)], 1)
        return im.astype(bool)

    fm, hm = raster(fin), raster(head)
    d_out = distance_transform_edt(~fm) * RES
    sdf2 = np.where(fm, distance_transform_edt(fm) * RES - RES / 2, -(d_out - RES / 2))

    # split the outline into leading edge (front root -> corona top) and trailing edge
    # (corona bottom -> gap -> lobe -> hook -> rear root); the root line and head are neither
    root = np.abs(outline[:, 1]) < 1.0
    i_front = np.flatnonzero(root)[np.argmin(outline[root, 0])]
    i_rear = np.flatnonzero(root & (outline[:, 0] < 130))[np.argmax(outline[root & (outline[:, 0] < 130), 0])]
    on_head = np.array([head.buffer(0.3).contains(Point(p)) for p in outline])
    idx = np.arange(len(outline))
    lo, hi = sorted((i_front, i_rear))
    between = (idx > lo) & (idx < hi)                  # the arc through the tab
    arc = ~between & ~on_head & (outline[:, 1] > -0.5)
    # the two non-tab arcs: the one touching i_front is the leading edge
    lead = arc & ((idx <= lo) if lo == i_front else (idx >= hi))
    trail = arc & ~lead

    def dist_to(sel):
        im = np.ones((ny, nx), np.uint8)
        q = np.round((outline[sel] - (x0, y0)) / RES).astype(np.int32)
        for run in np.split(q, np.flatnonzero(np.diff(idx[sel]) > 1) + 1):   # contiguous runs only
            cv2.polylines(im, [run], False, 0, 1)
        return distance_transform_edt(im) * RES

    def smooth(d):   # round the medial-axis creases, keep the true distance near the edge
        w = np.clip(d / (2 * D_SMOOTH), 0, 1)
        return w * gaussian_filter(d, D_SMOOTH / RES) + (1 - w) * d

    d_le, d_te = smooth(dist_to(lead)), smooth(dist_to(trail))

    Y = y0 + np.arange(ny)[:, None] * RES + 0 * np.arange(nx)[None, :]
    X = x0 + np.arange(nx)[None, :] * RES + 0 * Y
    t = np.clip((Y - TAPER_Y[0]) / (TAPER_Y[1] - TAPER_Y[0]), 0, 1)
    t = t * t * (3 - 2 * t)
    T = T_ROOT + (T_SHAFT - T_ROOT) * t
    r_te = np.interp(X, *zip(*R_TE))
    e_le, e_te = np.clip(d_le / R_LE, 0, 1), np.clip(d_te / r_te, 0, 1)
    h_le = (T - LAYER) * (1 - (1 - e_le) ** LE_K)               # steep nose up to the terrace
    h_te = H_MIN + (T - H_MIN) * (1 - (1 - e_te) ** TE_K)        # long foil-like taper (shaft, lobe body)
    # plateau (triangle + tongue) at full T; outside it the surface falls away from its boundary at a
    # steady PLAT_SLOPE (nested rings round the tongue, as in the photo) unless the edge taper is higher,
    # and nothing outside rises above the terrace level (T - LAYER)
    pm = raster(plat)
    d_p = distance_transform_edt(~pm) * RES
    dq = np.minimum(d_p, PLAT_EASE)
    h_pl = T - LAYER - PLAT_SLOPE * (dq * dq / (2 * PLAT_EASE) + np.maximum(d_p - PLAT_EASE, 0))  # quadratic ease-in
    h_out = np.minimum(np.minimum(h_le, -smin(-h_te, -h_pl, SMIN_K)), T - LAYER)   # smooth max: no seam
    # no trailing edge gets thickened by the plateau or its fall-off: cap everything with a steeper copy
    # of the edge taper (also keeps the tongue's gap side continuous instead of a cliff)
    e_cap = np.clip(d_te / (EDGE_CAP * r_te), 0, 1)
    h_cap = H_MIN + (T + SMIN_K - H_MIN) * (1 - (1 - e_cap) ** TE_K)   # tops out above T: smin leaves T alone
    h = np.maximum(smin(np.where(pm, T, h_out), h_cap, SMIN_K), H_MIN)
    w = np.clip((X - HOOK_X) / 4, 0, 1) * np.clip((HOOK_Y - Y) / 4, 0, 1)   # chunky hook tooth
    h = np.maximum(h, H_MIN + (H_HOOK - H_MIN) * w)
    h_body = h

    # head: cone-ish dome around HEAD_PEAK on a skirt wall (raised rim on the corona side),
    # r normalized by the head boundary distance along each ray
    ang = np.arctan2(Y - HEAD_PEAK[1], X - HEAD_PEAK[0])
    rad = np.hypot(Y - HEAD_PEAK[1], X - HEAD_PEAK[0])
    angs = np.linspace(-np.pi, np.pi, 721)
    hb = head.boundary
    Rb, base = [], []
    for a in angs:
        ray = LineString([HEAD_PEAK, (HEAD_PEAK[0] + 200 * np.cos(a), HEAD_PEAK[1] + 200 * np.sin(a))])
        ip = ray.intersection(hb)
        pts = [ip] if ip.geom_type == "Point" else list(getattr(ip, "geoms", []))
        p = min(pts, key=lambda p: np.hypot(p.x - HEAD_PEAK[0], p.y - HEAD_PEAK[1]))
        Rb.append(np.hypot(p.x - HEAD_PEAK[0], p.y - HEAD_PEAK[1]))
        base.append(H_SKIRT + (H_CORONA - H_SKIRT) * np.clip(1 - corona.distance(p) / D_RIM, 0, 1))  # rim wraps the flares
    base = gaussian_filter1d(np.array(base), 4, mode="wrap")
    r = np.clip(rad / np.interp(ang, angs, Rb), 0, 1)
    b = np.interp(ang, angs, base)
    dh = gaussian_filter(b + (T_HEAD - b) * (1 - r ** HEAD_Q), 1.0 / RES)   # smooth the ray seams
    # rounded edge band: the dome only ever falls toward the glans outline (and the flare tips)
    d_hd = dist_to(on_head)
    e = np.clip(d_hd / R_SKIRT, 0, 1)
    dh = np.minimum(dh, H_WALL + (H_SKIRT + 0.8 - H_WALL) * np.sqrt(1 - (1 - e) ** 2) + 10 * np.maximum(d_hd - R_SKIRT, 0))
    # corona face: slope up from the shaft surface over W_CORONA instead of a step
    im = np.ones((ny, nx), np.uint8)
    cv2.polylines(im, [np.round((np.array(CORONA) - (x0, y0)) / RES).astype(np.int32)], False, 0, 1)
    sc = np.clip(distance_transform_edt(im) * RES / W_CORONA, 0, 1)
    sc = sc * sc * (3 - 2 * sc)
    h = np.where(hm, h_body + (dh - h_body) * sc, h)
    # soften creases; pad outside the outline with the nearest inside value so nothing leaks in
    _, (iy, ix) = distance_transform_edt(~fm, return_indices=True)
    h = gaussian_filter(h[iy, ix], 0.6 / RES)
    return h, sdf2, (x0, y0)


def bend(xy):
    """Upright variant: forward-warp fin points (N x 2). Everything near the spine (leading edge, shaft,
    head) follows the spine re-bent to the BEND_S/BEND_DEG headings; the blade and lobe fade back to
    unmoved between W_IN and W_OUT. Applied to the fine marching-cubes mesh in XY only, so the relief
    rides along and the back stays flat."""
    from scipy.interpolate import splev, splprep
    from scipy.spatial import cKDTree
    tck, _ = splprep(np.array(SPINE, float).T, s=2.0)
    p = np.array(splev(np.linspace(0, 1, 4000), tck)).T
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    s = np.arange(0, seg[-1], 0.1)
    c0 = np.column_stack([np.interp(s, seg, p[:, 0]), np.interp(s, seg, p[:, 1])])
    phi0 = np.unwrap(np.arctan2(*np.gradient(c0, axis=0)[:, ::-1].T))
    ss = lambda u: (lambda u: u * u * (3 - 2 * u))(np.clip(u, 0, 1))
    target = np.radians(gaussian_filter1d(np.interp(s, BEND_S, BEND_DEG), 50, mode="nearest"))  # 5 mm smoothing
    phi1 = phi0 + (target - phi0) * ss((s - BEND_FROM) / S_IN)
    c1 = c0[0] + np.r_[[[0, 0]], np.cumsum(0.1 * np.column_stack([np.cos(phi1), np.sin(phi1)])[:-1], 0)]

    i = cKDTree(c0).query(xy)[1]
    d = xy - c0[i]
    t = d[:, 0] * np.cos(phi0[i]) + d[:, 1] * np.sin(phi0[i])
    n = -d[:, 0] * np.sin(phi0[i]) + d[:, 1] * np.cos(phi0[i])      # + toward the leading edge
    t = np.where(i == len(c0) - 1, np.maximum(t, 0), t)            # sub-sample offset; extrapolate past the tip
    moved = c1[i] + t[:, None] * np.column_stack([np.cos(phi1[i]), np.sin(phi1[i])]) + \
        n[:, None] * np.column_stack([-np.sin(phi1[i]), np.cos(phi1[i])])
    w = np.where(i > 0, 1 - ss((-n - W_IN) / (W_OUT - W_IN)), 0.0)
    return xy + w[:, None] * (moved - xy)


def unbend(q, iters=30):
    """Inverse of bend() for points q (N x 2): grid lookup, then Newton (finite-difference Jacobian)."""
    from scipy.spatial import cKDTree
    g = np.mgrid[-10:220:0.5, -5:160:0.5].reshape(2, -1).T          # seed: nearest forward-mapped grid point
    p, e = g[cKDTree(bend(g)).query(q)[1]].astype(float), 0.05
    for _ in range(iters):
        r = q - bend(p)
        jx = (bend(p + [e, 0]) - bend(p - [e, 0])) / (2 * e)
        jy = (bend(p + [0, e]) - bend(p - [0, e])) / (2 * e)
        det = jx[:, 0] * jy[:, 1] - jx[:, 1] * jy[:, 0]
        p += np.column_stack([jy[:, 1] * r[:, 0] - jy[:, 0] * r[:, 1], -jx[:, 1] * r[:, 0] + jx[:, 0] * r[:, 1]]) / det[:, None]
    return p, np.abs(q - bend(p)).max()


def upright_outline():
    """Rake outline plus the strip between the shaft and the BACK_RAKE curve (filleted joins)."""
    poly = Polygon(np.loadtxt(f"{HERE}/outline_mm.csv", delimiter=",")).buffer(0)
    if not BACK_FILL:
        return poly
    back = LineString(BACK_RAKE)
    strip = max((back.buffer(d, single_sided=True) for d in (18.0, -18.0)),   # the side that overlaps the shaft
                key=lambda g: g.intersection(poly).area)
    grown = poly.union(strip.intersection(poly.buffer(18.0)))
    # fillet the new concave joins only (closing), so the rest of the outline is untouched
    return grown.buffer(BACK_FILLET, 64).buffer(-BACK_FILLET, 64).intersection(poly.buffer(30)).union(grown)


def fin_body(variant="rake"):
    h, sdf2, (x0, y0) = thickness_field(upright_outline() if variant == "upright" else None)
    nz = int((h.max() + 2) / RES) + 1
    z = -1 + np.arange(nz) * RES
    # inside: within the outline, above the bed, below the surface
    vol = np.minimum(np.minimum(sdf2[..., None], h[..., None] - z[None, None, :]), z[None, None, :])
    vol = np.pad(vol, 1, constant_values=-1)
    v, f, _, _ = marching_cubes(vol, 0.0, spacing=(RES, RES, RES))
    v = v - RES
    v = np.column_stack([x0 + v[:, 1], y0 + v[:, 0], -1 + v[:, 2]])
    if variant == "upright":
        v[:, :2] = bend(v[:, :2])
    f = f[:, ::-1]
    m = trimesh.Trimesh(v, f, process=True)
    if m.volume < 0:
        m.invert()
    return Manifold(Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                         tri_verts=np.asarray(m.faces, np.uint32)))


if __name__ == "__main__":
    body = fin_body(VARIANT)
    print("fin status", body.status(), "vol", round(body.volume()))
    part = (body + tab()).simplify(0.01)
    mm = part.to_mesh()
    out = trimesh.Trimesh(mm.vert_properties[:, :3], mm.tri_verts)
    if out.body_count > 1:   # drop zero-volume slivers the union can leave on the root line
        with np.errstate(invalid="ignore"):
            out = trimesh.util.concatenate([b for b in out.split(only_watertight=False) if abs(b.volume) > 1.0])
    print("watertight", out.is_watertight, "faces", len(out.faces), "bounds", out.bounds.round(2).tolist(),
          "vol cm3", round(out.volume / 1000, 1))
    out.export(OUT)
    print("wrote", OUT)
