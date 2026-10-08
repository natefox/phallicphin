"""phallicphin: novelty Futures-base surf fin, shape traced from ref/photo.png.

usage: fin.py [out.stl]
Coords: X along the tab (0 = front V-notch end), Y up from the tab top, Z = thickness.
Built flat-backed (Z=0 face flat, all shaping on +Z) so it prints lying down with no
supports and layers running the length of the fin (strong at the root).
"""
import sys
import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold, Mesh
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import LineString, Polygon, box
from shapely.ops import split
from skimage.measure import marching_cubes
import cv2

OUT = sys.argv[1] if len(sys.argv) > 1 else "phallicphin.stl"
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."

# --- tab (Futures box fit) ---
TAB_W = 7.2          # tab thickness; FuturesRearFinBase4mm = 7.2, Futures blank = 7.0
TAB_EDGE_R = 0.6     # rounding on the long bottom edges of the tab
TAB_PROFILE = f"{HERE}/tab_profile_mm.csv"   # side profile measured from a Futures rear-fin base

# --- fin body ---
T_ROOT = TAB_W       # plateau thickness at the root
T_SHAFT = 5.0        # thickness the shaft tapers to before the head
TAPER_Y = (55.0, 95.0)   # taper from T_ROOT to T_SHAFT between these heights
R_EDGE = 12.0        # width of the rounded edge band
H_MIN = 0.8          # edge thickness (no feather edges)
T_HEAD = 7.2
HEAD_P = 3.0         # superellipse exponent of the head dome (higher = steeper rim)
# corona line + dome peak, traced from the photo (px -> mm, same scale as trace_outline.py)
S = 113.0 / 657.0
px = lambda x, y: ((x - 106.5) * S, (935.0 - y) * S)
CORONA = [px(935, 200), px(943, 245), px(946, 330), px(958, 400), px(978, 470), px(990, 520)]
HEAD_PEAK = px(990, 325)

RES = 0.2            # voxel size, mm


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


def thickness_field():
    outline = np.loadtxt(f"{HERE}/outline_mm.csv", delimiter=",")
    poly = Polygon(outline).buffer(0)
    fin = poly.intersection(box(-50, -1.0, 300, 300))     # fin above the tab, 1mm overlap
    corona = LineString(CORONA)
    parts = split(fin, corona)
    head = max(parts.geoms, key=lambda g: g.centroid.x)

    x0, y0, x1, y1 = fin.bounds
    x0, y0 = x0 - 2, y0 - 2
    nx, ny = int((x1 + 2 - x0) / RES) + 1, int((y1 + 2 - y0) / RES) + 1

    def raster(p):
        im = np.zeros((ny, nx), np.uint8)
        g = lambda ring: np.round((np.array(ring.coords) - (x0, y0)) / RES).astype(np.int32)
        cv2.fillPoly(im, [g(p.exterior)], 1)
        return im.astype(bool)

    fm, hm = raster(fin), raster(head)
    # edge rounding ignores the root line: measure edge distance as if the fin continued into the tab
    rx0, _, rx1, _ = fin.intersection(box(-50, -0.5, 300, 0.5)).bounds
    d_in = distance_transform_edt(raster(fin.union(box(rx0, y0 - 50, rx1, 0)))) * RES
    d_out = distance_transform_edt(~fm) * RES
    sdf2 = np.where(fm, distance_transform_edt(fm) * RES - RES / 2, -(d_out - RES / 2))

    Y = y0 + np.arange(ny)[:, None] * RES + 0 * np.arange(nx)[None, :]
    X = x0 + np.arange(nx)[None, :] * RES + 0 * Y
    t = np.clip((Y - TAPER_Y[0]) / (TAPER_Y[1] - TAPER_Y[0]), 0, 1)
    t = t * t * (3 - 2 * t)
    T = T_ROOT + (T_SHAFT - T_ROOT) * t
    e = np.clip(d_in / R_EDGE, 0, 1)
    h = H_MIN + (T - H_MIN) * np.sqrt(1 - (1 - e) ** 2)

    # head: dome around HEAD_PEAK, normalized by distance to the head boundary along each ray
    ang = np.arctan2(Y - HEAD_PEAK[1], X - HEAD_PEAK[0])
    rad = np.hypot(Y - HEAD_PEAK[1], X - HEAD_PEAK[0])
    angs = np.linspace(-np.pi, np.pi, 721)
    hb = head.boundary
    Rb = []
    for a in angs:
        ray = LineString([HEAD_PEAK, (HEAD_PEAK[0] + 200 * np.cos(a), HEAD_PEAK[1] + 200 * np.sin(a))])
        ip = ray.intersection(hb)
        pts = [ip] if ip.geom_type == "Point" else list(getattr(ip, "geoms", []))
        Rb.append(min(np.hypot(p.x - HEAD_PEAK[0], p.y - HEAD_PEAK[1]) for p in pts))
    r = rad / np.interp(ang, angs, Rb)
    dh = H_MIN + (T_HEAD - H_MIN) * np.clip(1 - np.clip(r, 0, 1) ** HEAD_P, 0, 1) ** (1 / HEAD_P)
    h = np.where(hm, np.maximum(h, dh), h)
    h = gaussian_filter(h, 0.6 / RES)      # soften the plateau creases a touch
    return h, sdf2, (x0, y0)


def fin_body():
    h, sdf2, (x0, y0) = thickness_field()
    nz = int((h.max() + 2) / RES) + 1
    z = -1 + np.arange(nz) * RES
    # inside: within the outline, above the bed, below the surface
    vol = np.minimum(np.minimum(sdf2[..., None], h[..., None] - z[None, None, :]), z[None, None, :])
    vol = np.pad(vol, 1, constant_values=-1)
    v, f, _, _ = marching_cubes(vol, 0.0, spacing=(RES, RES, RES))
    v = v - RES
    v = np.column_stack([x0 + v[:, 1], y0 + v[:, 0], -1 + v[:, 2]])
    f = f[:, ::-1]
    m = trimesh.Trimesh(v, f, process=True)
    if m.volume < 0:
        m.invert()
    return Manifold(Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                         tri_verts=np.asarray(m.faces, np.uint32)))


if __name__ == "__main__":
    body = fin_body()
    print("fin status", body.status(), "vol", round(body.volume()))
    part = (body + tab()).simplify(0.01)
    mm = part.to_mesh()
    out = trimesh.Trimesh(mm.vert_properties[:, :3], mm.tri_verts)
    print("watertight", out.is_watertight, "faces", len(out.faces), "bounds", out.bounds.round(2).tolist(),
          "vol cm3", round(out.volume / 1000, 1))
    out.export(OUT)
    print("wrote", OUT)
