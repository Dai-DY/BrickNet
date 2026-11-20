"""Silhouette carving: the most robust shape cue for shiny, texture-poor cars.

A point is kept when it projects inside the car mask in all views that see it, allowing ``max_misses`` views to
disagree (imperfect masks / poses). Views where the point falls outside the image are ignored.

``hull_descriptor`` runs the whole front half on calibrated views:
    coarse carve in the reconstruction frame -> canonicalize (up hint = mean camera position) ->
    fine carve on the descriptor's own half-stud lattice -> CarDescriptor.
"""

from __future__ import annotations

import numpy as np

from ..canonical import SUBCELL, Canonicalization, canonicalize
from ..descriptor import CarDescriptor, GridSpec, descriptor_from_points
from .cameras import Camera


def carve(points: np.ndarray, cameras: list[Camera], masks: list[np.ndarray], max_misses: int = 1,
          min_views: int = 3) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    misses = np.zeros(len(pts), dtype=np.int32)
    seen = np.zeros(len(pts), dtype=np.int32)
    for cam, mask in zip(cameras, masks):
        uv, z = cam.project(pts)
        h, w = mask.shape
        u = np.floor(uv[:, 0]).astype(np.int64)
        v = np.floor(uv[:, 1]).astype(np.int64)
        valid = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        inside = np.zeros(len(pts), dtype=bool)
        inside[valid] = mask[v[valid], u[valid]]
        seen += valid
        misses += valid & ~inside
    return (misses <= max_misses) & (seen >= min_views)


def _grid(lo, hi, n_cells: int) -> tuple[np.ndarray, float]:
    step = float(np.cbrt(np.prod(hi - lo) / n_cells))
    axes = [np.arange(lo[a] + step / 2, hi[a], step) for a in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3), step


def scene_bounds(cameras: list[Camera], masks: list[np.ndarray], sparse_points: np.ndarray | None = None):
    """Bounding box of the car: sparse points that land in most masks, else the cameras' convex region."""
    if sparse_points is not None and len(sparse_points) >= 50:
        keep = carve(sparse_points, cameras, masks, max_misses=max(1, len(cameras) // 5))
        p = sparse_points[keep] if keep.sum() >= 20 else sparse_points
        lo, hi = np.quantile(p, 0.01, axis=0), np.quantile(p, 0.99, axis=0)
        pad = 0.15 * (hi - lo)
        return lo - pad, hi + pad
    centers = np.array([c.center for c in cameras])
    mid = centers.mean(0)
    r = np.linalg.norm(centers - mid, axis=1).mean()
    return mid - r, mid + r


def trim_bottom_flare(pts_car: np.ndarray, band=(0.3, 0.7)) -> np.ndarray:
    """Eye-level views cannot carve the space beside the body near the ground, so the hull flares out at the
    bottom. Per x slice, clamp |y| below mid height to the slice's mid-height half width (+ one lattice step)."""
    z = pts_car[:, 2]
    z0, z1 = np.quantile(z, band[0]), np.quantile(z, band[1])
    xs = np.round(pts_car[:, 0] / SUBCELL[0]).astype(int)
    keep = np.ones(len(pts_car), dtype=bool)
    mid = (z >= z0) & (z <= z1)
    for x in np.unique(xs):
        sl = xs == x
        m = sl & mid
        if not m.any():
            continue
        lim = np.abs(pts_car[m, 1]).max() + SUBCELL[1]
        keep[sl & (z < z0) & (np.abs(pts_car[:, 1]) > lim)] = False
    return pts_car[keep]


def hull_descriptor(
    cameras: list[Camera],
    masks: list[np.ndarray],
    *,
    sparse_points: np.ndarray | None = None,
    spec: GridSpec | None = None,
    target_width: float = 8.0,
    coarse_cells: int = 200_000,
    max_misses: int = 1,
    front_hint: np.ndarray | None = None,
) -> tuple[CarDescriptor, Canonicalization, np.ndarray]:
    """Returns (descriptor, canonicalization, interior points in car frame)."""
    lo, hi = scene_bounds(cameras, masks, sparse_points)
    for _ in range(2):  # second pass re-carves inside the first hull's bounding box at a much finer step
        grid, step = _grid(lo, hi, coarse_cells)
        coarse = grid[carve(grid, cameras, masks, max_misses=max_misses)]
        if len(coarse) < 100:
            raise RuntimeError("visual hull is empty: check masks / camera poses / bounds")
        lo, hi = coarse.min(0) - 2 * step, coarse.max(0) + 2 * step
    up_hint = np.mean([c.center for c in cameras], axis=0) - coarse.mean(0)  # photographers stand above the car
    canon = canonicalize(coarse, up_hint=up_hint, front_hint=front_hint, target_width=target_width)

    # fine carve on a half-stud lattice expressed in the car frame
    car = canon.apply(coarse)
    clo, chi = car.min(0) - 2 * SUBCELL, car.max(0) + 2 * SUBCELL
    axes = [np.arange(clo[a] + SUBCELL[a] / 2, chi[a], SUBCELL[a]) for a in range(3)]
    lattice = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inside = carve(canon.invert(lattice), cameras, masks, max_misses=max_misses)
    pts = trim_bottom_flare(lattice[inside])
    desc = descriptor_from_points(pts, spec, center_y=0.0)
    return desc, canon, pts
