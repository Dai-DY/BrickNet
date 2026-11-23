"""Source car -> watertight solid + signed distance field + landmarks, in the car frame (metres).

ShapeNet meshes are open (missing panels, single-sided glass, interior parts), so the solid is built by *depth
fusion*: render depth maps from many directions around the car (including from below) and mark every voxel that
some camera sees in front of the surface as empty; what no camera can see into is solid. This seals holes, drops the
interior, and keeps every concavity that is visible from somewhere (wheel arches, intakes, the pickup bed).

Landmarks: front / rear axle from the bottom profile (tyres touch the ground, the floor in between does not),
height and half-width profiles along x (used to match cross-sections at the seams).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from car2brick.recon.cameras import Camera

from .render import DEV, MeshRenderer


@dataclass
class SourceCar:
    sid: str
    name: str
    lo: np.ndarray  # world position of voxel (0, 0, 0)'s corner
    vs: float  # voxel size, m
    sdf: np.ndarray  # (X, Y, Z) float32, metres, negative inside
    axles: tuple[float, float]  # x of the front and rear axle
    tri: np.ndarray  # (T, 3, 3) mesh in the car frame (for figures)
    rgb: np.ndarray  # (T, 3)
    color_pts: np.ndarray  # (M, 3) visible surface samples
    color_rgb: np.ndarray  # (M, 3) their albedo
    scale: float = 1.0  # ShapeNet units -> metres
    meta: dict = field(default_factory=dict)

    @property
    def occ(self) -> np.ndarray:
        return self.sdf < 0

    def xs(self) -> np.ndarray:
        return self.lo[0] + (np.arange(self.sdf.shape[0]) + 0.5) * self.vs

    def profiles(self) -> dict[str, np.ndarray]:
        """Per x slice: top height, bottom height, half width (NaN where the slice is empty)."""
        occ = self.occ
        X, Y, Z = occ.shape
        ys = self.lo[1] + (np.arange(Y) + 0.5) * self.vs
        zs = self.lo[2] + (np.arange(Z) + 0.5) * self.vs
        anyz = occ.any(1)  # (X, Z)
        anyy = occ.any(2)  # (X, Y)
        has = anyz.any(1)
        top = np.where(has, zs[Z - 1 - np.argmax(anyz[:, ::-1], 1)] + self.vs / 2, np.nan)
        bot = np.where(has, zs[np.argmax(anyz, 1)] - self.vs / 2, np.nan)
        half = np.where(has, np.max(np.where(anyy, np.abs(ys)[None], 0), 1) + self.vs / 2, np.nan)
        return {"x": self.xs(), "top": top, "bottom": bot, "half": half}

    def extent_x(self) -> tuple[float, float]:
        has = np.nonzero(self.occ.any((1, 2)))[0]
        return self.lo[0] + has[0] * self.vs, self.lo[0] + (has[-1] + 1) * self.vs

    def save(self, path: Path):
        np.savez_compressed(path, sid=self.sid, name=self.name, lo=self.lo, vs=self.vs,
                            sdf=self.sdf.astype(np.float16), axles=np.array(self.axles),
                            tri=self.tri.astype(np.float32), rgb=self.rgb.astype(np.float16),
                            color_pts=self.color_pts.astype(np.float32),
                            color_rgb=(self.color_rgb * 255).astype(np.uint8), scale=self.scale)

    @classmethod
    def load(cls, path: Path) -> "SourceCar":
        z = np.load(path)
        return cls(str(z["sid"]), str(z["name"]), z["lo"], float(z["vs"]), z["sdf"].astype(np.float32),
                   tuple(float(a) for a in z["axles"]), z["tri"], z["rgb"].astype(np.float32), z["color_pts"],
                   z["color_rgb"].astype(np.float32) / 255, float(z["scale"]))


def fibonacci_dirs(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    a = np.pi * (1 + 5**0.5) * i
    return np.stack([r * np.cos(a), r * np.sin(a), z], 1)


def area_samples(tri: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    i = rng.choice(len(tri), n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n))[:, None], rng.random(n)[:, None]
    t = tri[i]
    return (1 - r1) * t[:, 0] + r1 * (1 - r2) * t[:, 1] + r1 * r2 * t[:, 2]


def normalise(v: np.ndarray, f: np.ndarray, width: float) -> tuple[np.ndarray, float]:
    """Ground at z = 0, centred in x and y, scaled so the body is ``width`` metres wide (mirrors ignored)."""
    p = area_samples(v[f], 200_000)
    lo, hi = v.min(0), v.max(0)
    c = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    w = 2 * np.quantile(np.abs(p[:, 1] - c[1]), 0.995)
    s = width / w
    return (v - c) * s, s


@torch.no_grad()
def fuse(rend: MeshRenderer, lo: np.ndarray, hi: np.ndarray, vs: float, n_views: int = 96, res: int = 640,
         color_per_view: int = 40_000, log=print):
    """Depth fusion -> (occupancy (X, Y, Z) bool, colour points, colour rgb)."""
    dims = tuple(int(d) for d in np.ceil((hi - lo) / vs))
    ax = [torch.arange(d, device=DEV, dtype=torch.float32) for d in dims]
    centre = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo) / 2)
    free = torch.zeros(dims, dtype=torch.bool, device=DEV)
    lo_t = torch.tensor(lo, dtype=torch.float32, device=DEV)
    pts, cols = [], []
    gen = torch.Generator(device=DEV).manual_seed(0)
    eps = 0.5 * vs
    for k, d in enumerate(fibonacci_dirs(n_views)):
        dist = 2.6 * radius
        up = (0.0, 0.0, 1.0) if abs(d[2]) < 0.98 else (1.0, 0.0, 0.0)
        f = (res / 2) / np.tan(np.arcsin(1.05 / 2.6))
        cam = Camera.look_at(f"fuse{k}", centre + dist * d, centre, f, res, res, up=up)
        r = rend.raster(cam, ss=2, pts_per_px=3.0, seed=k)
        D = r["depth"].view(1, 1, res * 2, res * 2)
        D = -F.max_pool2d(-D, 2)  # nearest surface in each pixel's 2x2 subsamples (no pinholes)
        D = -F.max_pool2d(-D, 3, stride=1, padding=1)[0, 0]  # 3x3 min filter: conservative near edges
        K = torch.tensor(cam.K, dtype=torch.float32, device=DEV)
        R = torch.tensor(cam.R, dtype=torch.float32, device=DEV)
        t = torch.tensor(cam.t, dtype=torch.float32, device=DEV)
        for i in range(0, dims[0], 64):  # x slabs in batches (the camera frames the whole box)
            xx, yy, zz = torch.meshgrid(ax[0][i : i + 64], ax[1], ax[2], indexing="ij")
            p = lo_t + (torch.stack([xx, yy, zz], -1).reshape(-1, 3) + 0.5) * vs
            pc = p @ R.T + t
            z = pc[:, 2]
            uvw = pc @ K.T
            u = torch.floor(uvw[:, 0] / z).long().clamp(0, res - 1)
            v = torch.floor(uvw[:, 1] / z).long().clamp(0, res - 1)
            free[i : i + 64] |= (z < D[v, u] - eps).view(xx.shape)
        idx = r["ok"].nonzero().squeeze(1)
        if len(idx):
            pick = idx[torch.randperm(len(idx), generator=gen, device=DEV)[:color_per_view]]
            pts.append(r["points"][pick].cpu().numpy())
            cols.append(rend.rgb[r["fid"][pick]].cpu().numpy())
        if (k + 1) % 32 == 0:
            log(f"    fused {k + 1}/{n_views} views")
    return (~free).cpu().numpy(), np.concatenate(pts), np.concatenate(cols)


def signed_distance(occ: np.ndarray, vs: float) -> np.ndarray:
    out = distance_transform_edt(~occ) - 0.5
    ins = distance_transform_edt(occ) - 0.5
    return (np.where(occ, -ins, out) * vs).astype(np.float32)


def find_axles(src: SourceCar, contact: float = 0.04, min_run: float = 0.05) -> tuple[float, float]:
    """Tyres are where the bottom of the car touches the ground; the front-most / rear-most contact runs (ignoring
    the outer 8% at each end, where low lips and diffusers live) are the axles."""
    pr = src.profiles()
    x, bot, top = pr["x"], pr["bottom"], pr["top"]
    h = np.nanmax(top)
    touch = np.nan_to_num(bot, nan=1e9) < contact * h
    x0, x1 = src.extent_x()
    inner = (x > x0 + 0.08 * (x1 - x0)) & (x < x1 - 0.08 * (x1 - x0))
    touch &= inner
    runs, start = [], None
    for i, tch in enumerate(np.append(touch, False)):
        if tch and start is None:
            start = i
        elif not tch and start is not None:
            if (i - start) * src.vs >= min_run:
                runs.append((x[start] + x[i - 1]) / 2)
            start = None
    if len(runs) < 2:  # skirted / odd model: typical overhangs
        L = x1 - x0
        return x1 - 0.2 * L, x0 + 0.22 * L
    return max(runs), min(runs)


def build_source(sid: str, name: str, v: np.ndarray, f: np.ndarray, rgb: np.ndarray, width: float = 1.95,
                 vs: float = 0.012, n_views: int = 96, res: int = 640, log=print) -> SourceCar:
    v, s = normalise(v, f, width)
    rend = MeshRenderer(v[f], rgb)
    lo, hi = v.min(0) - 4 * vs, v.max(0) + 4 * vs
    log(f"  [{name}] {len(f)} faces, {np.ptp(v, 0).round(2)} m, fusing at {vs * 100:.1f} cm")
    occ, cp, cc = fuse(rend, lo, hi, vs, n_views, res, log=log)
    src = SourceCar(sid, name, lo, vs, signed_distance(occ, vs), (0.0, 0.0), v[f].astype(np.float32),
                    rgb.astype(np.float32), cp, cc, s)
    src.axles = find_axles(src)
    log(f"    solid {occ.mean() * 100:.0f}% of the box, axles at x = {src.axles[0]:.2f} / {src.axles[1]:.2f} m "
        f"(wheelbase {src.axles[0] - src.axles[1]:.2f} m)")
    return src
