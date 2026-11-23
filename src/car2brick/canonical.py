"""Front half: put a reconstructed car into the car frame and turn it into interior points.

``canonicalize`` finds a similarity transform (world -> car frame, LDU) from an unordered point set:
  1. PCA axes; up is the smallest axis (or the one closest to ``up_hint``, e.g. "camera centres are above the car")
     and its sign is chosen so the greenhouse (top band) is shorter than the body (bottom band);
  2. x is the longer remaining axis, y = z x x;
  3. a small yaw / lateral-offset search maximises mirror symmetry about y = 0;
  4. uniform scale so the car is ``target_width`` studs wide; z = 0 at the lowest point.

The forward direction (+x or -x) is not identifiable from shape alone and does not matter for generation: BrickNet
training cars appear in both orientations. Pass ``front_hint`` if you want a fixed convention.

``interior_points_from_surface`` turns surface samples (mesh, dense point cloud) into a filled lattice that is
robust to open windows / non-watertight meshes (closing + 3D fill + per-slice 2D fill).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .units import LDU_PER_STUD, transform_points

SUBCELL = np.array([10.0, 10.0, 4.0])  # matches GridSpec(sub=(2, 2, 2)): half stud x half stud x half plate


@dataclass
class Canonicalization:
    matrix: np.ndarray  # 4x4 similarity, world -> car frame (LDU)

    @property
    def scale(self) -> float:
        return float(np.cbrt(abs(np.linalg.det(self.matrix[:3, :3]))))

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return transform_points(self.matrix, pts)

    def invert(self, pts: np.ndarray) -> np.ndarray:
        return transform_points(np.linalg.inv(self.matrix), pts)


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _band_extent(p: np.ndarray, top: bool) -> float:
    h = p[:, 2]
    band = p[h >= np.quantile(h, 0.8)] if top else p[h <= np.quantile(h, 0.2)]
    if len(band) < 5:
        return 0.0
    return float(np.quantile(band[:, 0], 0.98) - np.quantile(band[:, 0], 0.02))


def _symmetry_error(p: np.ndarray, tree: cKDTree) -> float:
    mirrored = p * np.array([1.0, -1.0, 1.0])
    d, _ = tree.query(mirrored, k=1)
    return float(np.mean(np.minimum(d, np.quantile(d, 0.95))))


def canonicalize(
    points: np.ndarray,
    *,
    up_hint: np.ndarray | None = None,
    front_hint: np.ndarray | None = None,
    target_width: float = 8.0,
    symmetry_search_deg: float = 6.0,
    max_points: int = 20000,
    width_band: tuple[float, float] = (0.25, 0.75),
    seed: int = 0,
) -> Canonicalization:
    pts = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    c = pts.mean(0)
    X = pts - c
    w, v = np.linalg.eigh(np.cov(X.T))
    axes = v[:, np.argsort(w)[::-1]].T  # rows: largest -> smallest variance

    # up
    if up_hint is not None:
        hint = np.asarray(up_hint, float) / np.linalg.norm(up_hint)
        k = int(np.argmax(np.abs(axes @ hint)))
        up = axes[k] * np.sign(axes[k] @ hint)
        rest = [axes[i] for i in range(3) if i != k]
    else:
        up = axes[2]
        rest = [axes[0], axes[1]]
    x = rest[0] - (rest[0] @ up) * up
    x /= np.linalg.norm(x)
    if up_hint is None:
        R = np.stack([x, np.cross(up, x), up])
        P = X @ R.T
        if _band_extent(P, top=True) > _band_extent(P, top=False):
            up = -up
    if front_hint is not None and x @ np.asarray(front_hint, float) < 0:
        x = -x
    R = np.stack([x, np.cross(up, x), up])
    P = X @ R.T

    # mirror-symmetry refinement: yaw about z and lateral offset
    sub = P[rng.choice(len(P), min(len(P), 5000), replace=False)]
    best = (np.inf, 0.0, 0.0)
    for deg in np.arange(-symmetry_search_deg, symmetry_search_deg + 1e-9, 1.0):
        Q = sub @ _rot_z(np.radians(deg)).T
        off = 0.5 * (np.quantile(Q[:, 1], 0.02) + np.quantile(Q[:, 1], 0.98))
        Q = Q - np.array([0.0, off, 0.0])
        err = _symmetry_error(Q, cKDTree(Q))
        if err < best[0]:
            best = (err, np.radians(deg), off)
    R = _rot_z(best[1]) @ R
    P = X @ R.T - np.array([0.0, best[2], 0.0])

    # width at mid height: robust to the bottom flare of visual hulls (and to tyres / mirrors)
    h = P[:, 2]
    band = P[(h >= np.quantile(h, width_band[0])) & (h <= np.quantile(h, width_band[1]))]
    band = band if len(band) >= 20 else P
    width = np.quantile(band[:, 1], 0.997) - np.quantile(band[:, 1], 0.003)
    s = target_width * LDU_PER_STUD / max(width, 1e-9)
    P = P * s
    z0 = np.quantile(P[:, 2], 0.002)
    xc = 0.5 * (np.quantile(P[:, 0], 0.005) + np.quantile(P[:, 0], 0.995))

    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = -s * (R @ c) - s * np.array([0.0, best[2], 0.0]) - np.array([xc, 0.0, z0])
    return Canonicalization(M)


def interior_points_from_surface(surface_pts: np.ndarray, close_iters: int = 2) -> np.ndarray:
    """Surface samples (car frame, LDU) -> centres of filled half-stud x half-stud x half-plate cells.

    Not for visual hulls (already solid): this is for surfaces - meshes and dense point clouds."""
    pts = np.asarray(surface_pts, dtype=np.float64)
    lo = pts.min(0) - (close_iters + 2) * SUBCELL
    idx = np.floor((pts - lo) / SUBCELL).astype(int)
    shape = idx.max(0) + close_iters + 3
    vol = np.zeros(shape, dtype=bool)
    vol[tuple(idx.T)] = True
    if close_iters:
        vol = ndimage.binary_closing(vol, iterations=close_iters)
    filled = ndimage.binary_fill_holes(vol)
    # Slices of a car along any axis are (nearly) simply connected, but a non-watertight mesh (open windows,
    # missing panels) leaves some slices open, and a single open slice must not cut the car in two. So take the
    # union of 2D hole filling along all three axes: a cell is inside if it is enclosed in any of its slices.
    for axis in range(3):
        moved = np.moveaxis(vol, axis, 0)
        out = np.moveaxis(filled, axis, 0)
        for i in range(moved.shape[0]):
            out[i] |= ndimage.binary_fill_holes(moved[i])
    vol = filled
    cells = np.argwhere(vol)
    return lo + (cells + 0.5) * SUBCELL


def mesh_surface_points(mesh, count: int = 200000, seed: int = 0) -> np.ndarray:
    """Area-uniform surface samples of a trimesh.Trimesh (or Scene) in its own frame."""
    import trimesh

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry() if hasattr(mesh, "to_geometry") else mesh.dump(concatenate=True)
    pts, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return np.concatenate([np.asarray(pts), np.asarray(mesh.vertices)])


def mesh_descriptor(mesh, *, up_hint=None, front_hint=None, target_width: float = 8.0, spec=None, colors=(),
                    width_band: tuple[float, float] = (0.25, 0.75)):
    """Mesh (any frame, any scale) -> CarDescriptor. Returns (descriptor, canonicalization)."""
    from .descriptor import descriptor_from_points

    surf = mesh_surface_points(mesh)
    canon = canonicalize(surf, up_hint=up_hint, front_hint=front_hint, target_width=target_width,
                         width_band=width_band)
    inner = interior_points_from_surface(canon.apply(surf))
    return descriptor_from_points(inner, spec, center_y=0.0, colors=colors), canon
