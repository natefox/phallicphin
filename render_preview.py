"""Comparison renders of the STL against the reference photos.

usage: render_preview.py phallicphin.stl ref/photo.png preview.png [ref/photo_installed.png preview_installed.png]
preview.png: photo | model top-down at the photo's exact scale/position | model silhouette on the photo
preview_installed.png: installed photo | model above the root from the fitted camera | model silhouette on it
Software z-buffer, smooth shading, 0.2 mm layer steps drawn faintly (like the printed original).
"""
import sys
import cv2
import numpy as np
import trimesh

LAYER = 0.2
S = 657.0 / 113.0                  # photo px per mm (tab length)
X0, Y0 = 106.5, 935.0              # photo px of the front tab end / tab top
FIN_RGB = (244, 248, 250)    # BGR


def render(meshes, M, sc, c, W, H, light, bg, dist=0, pinhole=0):
    """Orthographic z-buffer render. meshes: [(trimesh, BGR, layer_lines)], M: 4x4 model -> camera
    (camera looks down -z), screen = c + sc * (x, -y), with perspective if dist (camera distance, mm),
    or a pinhole camera at the origin with focal length `pinhole` px (screen = c + f * (x, -y) / -z).
    Light is in camera space."""
    zb = np.full((H, W), -np.inf); img = np.tile(np.array(bg, float), (H, W, 1))
    hit_fin = np.zeros((H, W), bool)
    L = light / np.linalg.norm(light); Hv = (L + [0, 0, 1]) / np.linalg.norm(L + [0, 0, 1])
    for mi, (m, rgb, lines) in enumerate(meshes):
        v = (np.c_[m.vertices, np.ones(len(m.vertices))] @ M.T)[:, :3]
        cn = m.vertex_normals[m.faces]                                   # corner normals, (F, 3, 3)
        fn = np.repeat(m.face_normals[:, None], 3, 1)
        crease = (cn * fn).sum(-1) < np.cos(np.radians(35))             # keep sharp edges sharp
        cn = np.where(crease[..., None], fn, cn)
        n = cn @ M[:3, :3].T
        f = pinhole / -v[:, 2] if pinhole else (dist / (dist - v[:, 2]) if dist else 1.0) * sc
        s = np.c_[c[0] + f * v[:, 0], c[1] - f * v[:, 1], v[:, 2]]
        t = s[m.faces]
        lo = np.floor(t[:, :, :2].min(1)).astype(int); hi = np.ceil(t[:, :, :2].max(1)).astype(int)
        keep = (hi[:, 0] >= 0) & (lo[:, 0] < W) & (hi[:, 1] >= 0) & (lo[:, 1] < H) & \
               (((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])) < 0)  # front faces
        for k in np.flatnonzero(keep):
            (x0, y0), (x1, y1) = np.clip(lo[k], 0, [W - 1, H - 1]), np.clip(hi[k], 0, [W - 1, H - 1])
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            a, b, cc = t[k]
            d = (b[1] - cc[1]) * (a[0] - cc[0]) + (cc[0] - b[0]) * (a[1] - cc[1])
            if abs(d) < 1e-12:
                continue
            w0 = ((b[1] - cc[1]) * (gx - cc[0]) + (cc[0] - b[0]) * (gy - cc[1])) / d
            w1 = ((cc[1] - a[1]) * (gx - cc[0]) + (a[0] - cc[0]) * (gy - cc[1])) / d
            w2 = 1 - w0 - w1
            z = w0 * a[2] + w1 * b[2] + w2 * cc[2]
            sub = zb[y0:y1 + 1, x0:x1 + 1]
            msk = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6) & (z > sub)
            if not msk.any():
                continue
            sub[msk] = z[msk]
            w = np.stack([w0[msk], w1[msk], w2[msk]], -1)
            nn = w @ n[k]; nn /= np.linalg.norm(nn, axis=1, keepdims=True)
            g = 0.40 + 0.58 * np.clip(nn @ L, 0, 1) + 0.10 * np.clip(nn @ Hv, 0, 1) ** 40
            if lines:   # layer steps show on sloped top surfaces of the print
                mz = w @ m.vertices[m.faces[k], 2]
                nz = w @ cn[k, :, 2]
                g = g * np.where(((mz / LAYER) % 1 < 0.12) & (nz > 0.15) & (nz < 0.995), 0.88, 1.0)
            img[y0:y1 + 1, x0:x1 + 1][msk] = np.clip(g, 0, 1.1)[:, None] * rgb
            hit_fin[y0:y1 + 1, x0:x1 + 1][msk] = mi == 0
    return img.clip(0, 255).astype(np.uint8), hit_fin


def label(im, txt, x, col=(255, 255, 255)):
    cv2.putText(im, txt, (x + 20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 2, cv2.LINE_AA)


def face_view(m, photo, out):
    H, W = photo.shape[:2]
    img, hit = render([(m, FIN_RGB, True)], np.eye(4), S, (X0, Y0), W, H, np.array([-0.45, 0.55, 0.75]), (106, 100, 96))
    edge = cv2.morphologyEx(hit.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    ph = photo.copy(); ph[edge] = (0, 0, 255)
    y0, y1, x0, x1 = 180, 1020, 60, 1180
    row = np.concatenate([c[y0:y1, x0:x1] for c in (photo, img, ph)], 1)
    for i, t in enumerate(["reference photo", "model (same scale + position)", "model silhouette on photo"]):
        label(row, t, i * (x1 - x0))
    cv2.imwrite(out, cv2.resize(row, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))


# camera fitted to ref/photo_installed.png from landmarks (OpenCV pinhole, 444 x 550 px, ~6 px mean error)
CAM_F, CAM_C = 1300.0, (222.0, 275.0)
CAM_R = (-1.5979718229519455, 1.7862257996853492, -0.018038406746244472)   # Rodrigues
CAM_T = (48.31956815316952, 81.25569319712805, 500.4438811835799)


def installed_view(m, photo, out, k=2.0):
    """Model (same shaped face, tab cut off at the board) from the fitted camera, next to the installed photo."""
    H, W = int(photo.shape[0] * k), int(photo.shape[1] * k)
    ph = cv2.resize(photo, (W, H), interpolation=cv2.INTER_CUBIC)
    fin = m.slice_plane((0, 0.3, 0), (0, 1, 0), cap=True)            # tab is inside the fin box
    M = np.eye(4); M[:3, :3] = cv2.Rodrigues(np.array(CAM_R))[0]; M[:3, 3] = CAM_T
    M = np.diag([1.0, -1, -1, 1]) @ M                                  # OpenCV camera -> x right, y up, z to viewer
    img, hit = render([(fin, FIN_RGB, True)], M, 1.0, (CAM_C[0] * k, CAM_C[1] * k), W, H,
                      np.array([-0.35, 0.6, 0.75]), (200, 200, 200), pinhole=CAM_F * k)
    edge = cv2.morphologyEx(hit.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    ov = ph.copy(); ov[edge] = (0, 0, 255)
    row = np.concatenate([ph, img, ov], 1)
    for i, t in enumerate(["installed photo", "model, fitted camera", "model silhouette on photo"]):
        label(row, t, i * W, (30, 30, 30))
    cv2.imwrite(out, cv2.resize(row, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))


if __name__ == "__main__":
    m = trimesh.load(sys.argv[1])
    face_view(m, cv2.imread(sys.argv[2]), sys.argv[3])
    if len(sys.argv) > 5:
        installed_view(m, cv2.imread(sys.argv[4]), sys.argv[5])
