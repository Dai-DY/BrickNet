"""Hybrid car (SDF + coloured mesh from ``python -m hybrid compose``) -> building target on the LEGO grid.

Scale: the car's body width maps to ``width_studs`` studs (mirrors ignored), so 1 stud = width / width_studs metres
and 1 plate = 0.4 stud. The car is described column by column (a column = one stud x one stud, vertical):
  top     (L*SUB, W*SUB)  highest point of the car over each sub-stud sample, in plates (NaN = no car)
  bottom  (L*SUB, W*SUB)  lowest point
  occ     (L, W)          the column belongs to the car (half its samples or more)
Columns are made left-right symmetric. Colours are looked up on the hybrid mesh (nearest vertex).
Wheels: axle x from the body car, tyre radius / width from the dark vertices around each axle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .parts import SUB


@dataclass
class Wheel:
    x: float  # axle position along the car, m
    radius: float  # tyre radius, m
    width: float  # tyre width, m


@dataclass
class Target:
    W: int
    L: int
    H: int
    stud: float  # metres per stud
    plate: float
    x0: float  # x (m) of cell 0's rear edge
    top: np.ndarray
    bottom: np.ndarray
    occ: np.ndarray
    wheels: list[Wheel]
    verts: np.ndarray  # hybrid mesh vertices (for colour lookup)
    vcol: np.ndarray
    name: str = ""

    def __post_init__(self):
        self._tree = cKDTree(self.verts)

    # grid <-> metres (car frame of the hybrid: x forward, y left, z up, ground 0)
    def cell_x(self, i):
        return self.x0 + (np.asarray(i, float) + 0.5) * self.stud

    def cell_y(self, j):
        return (np.asarray(j, float) + 0.5 - self.W / 2) * self.stud

    def color_at(self, pts: np.ndarray, k: int = 6) -> np.ndarray:
        d, idx = self._tree.query(pts, k=k)
        return self.vcol[idx].mean(1)

    def col_top(self) -> np.ndarray:
        """Per column: median top height (plates), NaN for empty columns."""
        return self._col_median(self.top)

    def col_bottom(self) -> np.ndarray:
        return self._col_median(self.bottom)

    def _col_median(self, a: np.ndarray) -> np.ndarray:
        a = a.reshape(self.L, SUB, self.W, SUB).transpose(0, 2, 1, 3).reshape(self.L, self.W, -1)
        out = np.full((self.L, self.W), np.nan)
        ok = ~np.isnan(a).all(2)
        out[ok] = np.nanmedian(a[ok], axis=1)
        return out


def _body_halfwidth(occ: np.ndarray, lo: np.ndarray, vs: float) -> float:
    """Median over the central 60 % of the length of each slice's half width (robust to mirrors and spoilers)."""
    X = occ.shape[0]
    ys = lo[1] + (np.arange(occ.shape[1]) + 0.5) * vs
    anyy = occ.any(2)
    has = np.nonzero(anyy.any(1))[0]
    a, b = has[0] + int(0.2 * len(has)), has[0] + int(0.8 * len(has))
    half = [np.abs(ys[anyy[i]]).max() + vs / 2 for i in range(a, b) if anyy[i].any()]
    return float(np.median(half)) if half else float(np.abs(ys).max())


def _tyre_top(occ, lo, vs, xa, half) -> float | None:
    """Tyre top at an axle: in the outer band of the car, the first gap above the ground (tyre, then air, then the
    fender). Median over the band; None when the fender hugs the tyre everywhere."""
    X, Y, Z = occ.shape
    ix = np.arange(int((xa - 0.04 - lo[0]) / vs), int((xa + 0.04 - lo[0]) / vs) + 1)
    ys = lo[1] + (np.arange(Y) + 0.5) * vs
    iy = np.nonzero((np.abs(ys) > 0.72 * half) & (np.abs(ys) < 0.95 * half))[0]
    z0 = int(0.12 / vs)
    tops = []
    for i in ix[(ix >= 0) & (ix < X)]:
        for j in iy:
            col = occ[i, j]
            if not col[: z0 + 1].any():
                continue
            empty = np.nonzero(~col[z0:])[0]
            if len(empty):
                tops.append((z0 + empty[0]) * vs + lo[2])
    tops = [t for t in tops if 0.3 < t < 1.3]
    return float(np.median(tops)) if len(tops) > 10 else None


def _wheels(verts, vcol, axles, half, occ=None, lo=None, vs=None) -> list[Wheel]:
    out = []
    dark = vcol.max(1) < 0.22
    for xa in axles:
        m = dark & (np.abs(verts[:, 0] - xa) < 0.55) & (np.abs(verts[:, 1]) > 0.55 * half) & (verts[:, 2] < 1.2)
        if m.sum() < 50:
            out.append(Wheel(float(xa), 0.33, 0.24))
            continue
        z = verts[m, 2]
        ay = np.abs(verts[m, 1])
        r = float(np.quantile(z, 0.98)) / 2
        top = _tyre_top(occ, lo, vs, xa, half) if occ is not None else None
        if top is not None:
            r = top / 2
        tyre = m & (verts[:, 2] < 2 * r)
        if tyre.sum() > 30:
            ay = np.abs(verts[tyre, 1])
        w = float(np.quantile(ay, 0.98) - np.quantile(ay, 0.05))
        out.append(Wheel(float(xa), float(np.clip(r, 0.2, 0.55)), float(np.clip(w, 0.15, 0.4))))
    return out


def load_target(hybrid_dir: Path, width_studs: int = 14) -> Target:
    import trimesh

    hybrid_dir = Path(hybrid_dir)
    z = np.load(hybrid_dir / "hybrid_sdf.npz")
    sdf, lo, vs = z["sdf"].astype(np.float32), z["lo"], float(z["vs"])
    occ = sdf < 0
    mesh = trimesh.load(hybrid_dir / "hybrid.glb", force="mesh")
    verts = np.asarray(mesh.vertices, float)
    vcol = np.asarray(mesh.visual.vertex_colors, float)[:, :3] / 255
    summary = json.loads((hybrid_dir / "summary.json").read_text())
    half = _body_halfwidth(occ, lo, vs)
    stud = 2 * half / width_studs
    plate = 0.4 * stud
    X, Y, Z = occ.shape
    has_x = np.nonzero(occ.any((1, 2)))[0]
    x_lo, x_hi = lo[0] + has_x[0] * vs, lo[0] + (has_x[-1] + 1) * vs
    L = int(np.ceil((x_hi - x_lo) / stud))
    x0 = (x_lo + x_hi) / 2 - L * stud / 2
    zs = lo[2] + (np.arange(Z) + 0.5) * vs
    anyz = occ.any(2)
    top_f = np.where(anyz, zs[Z - 1 - np.argmax(occ[:, :, ::-1], 2)] + vs / 2, np.nan)  # (X, Y) metres
    bot_f = np.where(anyz, zs[np.argmax(occ, 2)] - vs / 2, np.nan)
    us = x0 + (np.arange(L * SUB) + 0.5) * stud / SUB
    vs_ = (np.arange(width_studs * SUB) + 0.5) * stud / SUB - width_studs * stud / 2
    ix = np.clip(((us - lo[0]) / vs).astype(int), 0, X - 1)
    iy = np.clip(((vs_ - lo[1]) / vs).astype(int), 0, Y - 1)
    top = top_f[np.ix_(ix, iy)] / plate
    bottom = bot_f[np.ix_(ix, iy)] / plate
    # symmetric: average with the mirror image (NaN where either side is empty)
    top = (top + top[:, ::-1]) / 2
    bottom = (bottom + bottom[:, ::-1]) / 2
    frac = (~np.isnan(top)).reshape(L, SUB, width_studs, SUB).mean((1, 3))
    occ_col = frac >= 0.5
    H = int(np.ceil(np.nanmax(top))) + 1
    wheels = _wheels(verts, vcol, summary["body_axles"], half, occ, lo, vs)
    return Target(width_studs, L, H, stud, plate, x0, top, bottom, occ_col, wheels, verts, vcol,
                  name=hybrid_dir.name)
