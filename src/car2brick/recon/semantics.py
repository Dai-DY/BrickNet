"""Colours and semantic labels for hull points, by projecting into the calibrated photos.

Visibility is approximated with a coarse z-buffer per view (points splatted into ``block`` x ``block`` pixel bins;
a point is visible when it is within ``depth_tol`` of the nearest point in its bin). Only surface points of the
hull are coloured.

Label masks come from any 2D segmenter (SAM2 with clicks, an open-vocabulary detector, manual painting): pass
``{2: [window masks per view], 3: [light masks per view]}`` aligned with ``cameras``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..canonical import SUBCELL
from .cameras import Camera


def surface_mask(points_car: np.ndarray) -> np.ndarray:
    """Points of a half-stud lattice (car frame) with at least one empty 6-neighbour."""
    lo = points_car.min(0)
    idx = np.round((points_car - lo) / SUBCELL).astype(int)
    vol = np.zeros(idx.max(0) + 1, dtype=bool)
    vol[tuple(idx.T)] = True
    inner = ndimage.binary_erosion(vol, border_value=0)
    return ~inner[tuple(idx.T)]


def visible(points: np.ndarray, cam: Camera, shape, block: int = 6, depth_tol: float | None = None) -> np.ndarray:
    uv, z = cam.project(points)
    h, w = shape
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    out = np.zeros(len(points), dtype=bool)
    if not ok.any():
        return out
    bu = (uv[ok, 0] // block).astype(np.int64)
    bv = (uv[ok, 1] // block).astype(np.int64)
    key = bv * (w // block + 1) + bu
    zmin = np.full(key.max() + 1, np.inf)
    np.minimum.at(zmin, key, z[ok])
    if depth_tol is None:
        depth_tol = 0.03 * float(np.median(z[ok]))
    out[np.nonzero(ok)[0]] = z[ok] <= zmin[key] + depth_tol
    return out


def project_colors(points_world, cameras: list[Camera], images: list[np.ndarray], masks: list[np.ndarray]):
    """Mean RGB per point over views where it is visible and inside the car mask; NaN when never seen."""
    acc = np.zeros((len(points_world), 3))
    n = np.zeros(len(points_world))
    for cam, img, mask in zip(cameras, images, masks):
        vis = visible(points_world, cam, mask.shape)
        uv, _ = cam.project(points_world[vis])
        u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
        inside = mask[v, u]
        idx = np.nonzero(vis)[0][inside]
        acc[idx] += img[v[inside], u[inside], :3]
        n[idx] += 1
    with np.errstate(invalid="ignore"):
        return acc / n[:, None]


def project_labels(points_world, cameras: list[Camera], label_masks: dict[int, list[np.ndarray]],
                   min_votes: int = 2, min_share: float = 0.5) -> np.ndarray:
    """Per-point label: the label voted by most views (>= min_votes and >= min_share of the views that see the
    point), else 1 (body)."""
    codes = sorted(label_masks)
    votes = np.zeros((len(points_world), len(codes)), dtype=np.int32)
    seen = np.zeros(len(points_world), dtype=np.int32)
    for ci, code in enumerate(codes):
        for cam, mask in zip(cameras, label_masks[code]):
            vis = visible(points_world, cam, mask.shape)
            if ci == 0:
                seen += vis
            uv, _ = cam.project(points_world[vis])
            votes[np.nonzero(vis)[0], ci] += mask[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    best = votes.argmax(1)
    out = np.ones(len(points_world), dtype=np.uint8)
    # majority of the views that see the point (visibility is approximate, so single votes are noise)
    hit = (votes.max(1) >= min_votes) & (votes.max(1) >= min_share * seen)
    out[hit] = np.array(codes, dtype=np.uint8)[best[hit]]
    return out


def wheels_from_dark_points(points_car: np.ndarray, rgb: np.ndarray, max_height_frac: float = 0.35,
                            max_luma: float = 70.0, min_points: int = 10) -> tuple[float, ...]:
    """Wheel-centre x positions (car frame, LDU) from dark (tyre-coloured) surface points near the ground.

    Eye-level photos cannot see under the car, so the hull fills the gap between the wheels and the occupancy-based
    detector finds nothing; the tyres are still visible as dark low regions on the sides.
    """
    ok = ~np.isnan(rgb).any(1)
    p, c = points_car[ok], rgb[ok]
    z = p[:, 2]
    # tyres are dark *and* unsaturated (a deep red body also has low luma)
    dark = (c.max(1) <= max_luma) & (np.ptp(c, axis=1) <= 30.0)
    side = np.abs(p[:, 1]) >= 0.6 * np.abs(p[:, 1]).max()  # tyres sit at the sides; the hull's underside does not
    low_dark = (z <= z.min() + max_height_frac * np.ptp(z)) & dark & side
    xs = p[low_dark, 0]
    if len(xs) < 2 * min_points:
        return ()
    # two peaks of the dark-point histogram along x, at least 30% of the car length apart
    length = np.ptp(p[:, 0])
    edges = np.arange(p[:, 0].min(), p[:, 0].max() + 10.0, 10.0)
    hist, _ = np.histogram(xs, bins=edges)
    hist = np.convolve(hist, np.ones(3) / 3, mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])
    first = int(np.argmax(hist))
    far = np.abs(centers - centers[first]) >= 0.3 * length
    if not far.any() or hist[far].max() < 0.2 * hist[first]:
        return ()
    second = int(np.flatnonzero(far)[np.argmax(hist[far])])
    peaks = sorted((centers[first], centers[second]))
    # refine each peak to the median of dark points within +-1.5 studs
    return tuple(float(np.median(xs[np.abs(xs - c) <= 30.0])) for c in peaks)
