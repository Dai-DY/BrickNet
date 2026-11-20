"""Tiny dependency-free renderer (numpy z-buffer + PIL) for previews of LEGO models and descriptors.

Parts are drawn from their convex hulls (``geometry.PartShapes``) plus studs, in the descriptor grid frame (z up),
with Lambert shading and dark outlines between parts. Good enough to judge shape and proportions; use LDView /
BrickLink Studio / Blender for real renders.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull

import bricknet

from .colors import _PALETTE_HEX
from .geometry import PartShapes
from .recon.cameras import Camera
from .units import LDU_PER_PLATE, LDU_PER_STUD

_EXTRA_RGB = {"trans clear": (200, 220, 235), "trans brown": (60, 55, 50), "trans red": (200, 30, 30),
              "trans yellow": (240, 210, 40)}


def color_rgb(code: int) -> tuple[int, int, int]:
    name = bricknet.load_catalog().code_to_color.get(int(code), "")
    if name in _PALETTE_HEX:
        h = _PALETTE_HEX[name]
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    return _EXTRA_RGB.get(name, (160, 165, 170))


def _hull_tris(verts: np.ndarray) -> np.ndarray:
    hull = ConvexHull(verts)
    return verts[hull.simplices]


def _stud_tris(sections: int = 8, r: float = 6.0, h: float = 4.0) -> np.ndarray:
    """Stud prism in part-local LDraw coordinates (stud rises toward -Y), centred at the origin."""
    a = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    ring = np.stack([r * np.cos(a), np.zeros(sections), r * np.sin(a)], 1)
    top = ring + [0, -h, 0]
    tris = []
    for i in range(sections):
        j = (i + 1) % sections
        tris += [[ring[i], ring[j], top[j]], [ring[i], top[j], top[i]], [top[i], top[j], [0, -h, 0]]]
    return np.array(tris)


_STUD = _stud_tris()


def part_triangles(part_ids, colors, poses, shapes: PartShapes | None = None, studs: bool = True):
    """(tris (T,3,3) in the poses' frame, rgb (T,3), part index (T,))."""
    shapes = shapes or PartShapes()
    conns = bricknet.load_connectors()
    all_t, all_c, all_i = [], [], []
    for k, (pid, col, pose) in enumerate(zip(part_ids, colors, poses)):
        local = [_hull_tris(h) for h in shapes.hulls(int(pid)) if len(h) >= 4]
        if studs:
            for f in conns.get(int(pid), {}).get(("stud", None), ()):
                local.append(_STUD @ f[:3, :3].T + f[:3, 3])
        if not local:
            continue
        t = np.concatenate(local)
        t = t @ np.asarray(pose)[:3, :3].T + np.asarray(pose)[:3, 3]
        all_t.append(t)
        all_c.append(np.tile(color_rgb(col), (len(t), 1)))
        all_i.append(np.full(len(t), k))
    return np.concatenate(all_t), np.concatenate(all_c), np.concatenate(all_i)


def voxel_triangles(solid: np.ndarray, rgb_of_cell) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exposed faces of solid cells (stud x stud x plate boxes), grid frame."""
    size = np.array([LDU_PER_STUD, LDU_PER_STUD, LDU_PER_PLATE])
    pad = np.pad(solid, 1)
    tris, cols, ids = [], [], []
    quads = {  # axis, direction -> 4 corners of the unit face
        (0, 1): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], (0, -1): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        (1, 1): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)], (1, -1): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        (2, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], (2, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    }
    for n, (i, j, k) in enumerate(np.argwhere(solid)):
        for (axis, d), corners in quads.items():
            nb = [i + 1, j + 1, k + 1]
            nb[axis] += d
            if pad[tuple(nb)]:
                continue
            c = (np.array(corners) + [i, j, k]) * size
            tris += [c[[0, 1, 2]], c[[0, 2, 3]]]
            cols += [rgb_of_cell(i, j, k)] * 2
            ids += [n, n]
    return np.array(tris), np.array(cols), np.array(ids)


def rasterize(tris, rgb, ids, cam: Camera, background=(245, 245, 242), light=(-0.4, -0.6, 1.0),
              outline: bool = True) -> Image.Image:
    w, h = cam.width, cam.height
    zbuf = np.full((h, w), np.inf)
    img = np.empty((h, w, 3))
    img[:] = background
    idbuf = np.full((h, w), -1)
    light = np.asarray(light, float) / np.linalg.norm(light)
    flat = tris.reshape(-1, 3)
    uv, z = cam.project(flat)
    uv, z = uv.reshape(-1, 3, 2), z.reshape(-1, 3)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    to_cam = cam.center - tris.mean(1)
    normals *= np.sign((normals * to_cam).sum(1, keepdims=True))
    shade = 0.45 + 0.55 * np.clip(normals @ light, 0, 1)
    for t in range(len(tris)):
        if (z[t] <= 0).any():
            continue
        p = uv[t]
        x0, y0 = np.floor(p.min(0)).astype(int)
        x1, y1 = np.ceil(p.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
        if x0 > x1 or y0 > y1:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = p
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-9:
            continue
        l0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
        l1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
        l2 = 1 - l0 - l1
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not inside.any():
            continue
        depth = l0 * z[t, 0] + l1 * z[t, 1] + l2 * z[t, 2]
        sub = zbuf[y0 : y1 + 1, x0 : x1 + 1]
        win = inside & (depth < sub)
        sub[win] = depth[win]
        img[y0 : y1 + 1, x0 : x1 + 1][win] = rgb[t] * shade[t]
        idbuf[y0 : y1 + 1, x0 : x1 + 1][win] = ids[t]
    if outline:
        edge = np.zeros((h, w), bool)
        edge[:, 1:] |= idbuf[:, 1:] != idbuf[:, :-1]
        edge[1:, :] |= idbuf[1:, :] != idbuf[:-1, :]
        img[edge & (idbuf >= 0)] *= 0.55
        img[edge & (idbuf < 0)] = img[edge & (idbuf < 0)] * 0.55
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def orbit_camera(center, distance, azimuth_deg, elevation_deg, f=900.0, w=640, h=420) -> Camera:
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    eye = np.asarray(center) + distance * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return Camera.look_at("view", eye, center, f, w, h)
