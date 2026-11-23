"""Compose a hybrid car: front of one car, body (cabin + wheelbase) of another, rear of a third.

Target frame = the body car's frame. Parts are placed by their axles: the front part's front axle goes onto the
body's front axle, the rear part's rear axle onto the body's rear axle, so each part keeps its own overhang,
nose / tail shape and wheels. Seams sit outside the wheel arches:

    rear part | body                                              | front part
    ----------|--o-------------------------------------------o----|----------
      seam_r  rear axle (body)                    front axle (body)  seam_f
              <- rear_cut ->                         <- front_cut ->

    seam_f = front axle - front_cut (behind the front wheels), seam_r = rear axle + rear_cut (ahead of the rear wheels)

``front_cut`` / ``rear_cut`` are distances from the axle *towards the middle of the car*, so by default the front
and rear parts bring their own wheels and arches. A part's shape is blended into the body over ``band`` metres
around its seam, with weights that sum to one (signed distances are interpolated: shapes morph rather than step).

Cross-section matching (``match`` in [0, 1]): a part's height and width are scaled by a factor that ramps from 1
at its far end to (body / part) at the seam, so a low nose meets a tall cab without a cliff. 0 keeps the part
exactly as it is, 1 matches the body completely at the seam.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .render import DEV
from .solid import SourceCar


@dataclass
class Plan:
    front_cut: float = 0.75  # m from the body's front axle towards the rear (seam behind the front arches)
    rear_cut: float = 0.75  # m from the body's rear axle towards the front (seam ahead of the rear arches)
    band: float = 0.35  # blend width, m
    match: float = 0.8  # cross-section matching strength
    fill: float = 1.0  # anti-pinch: mix in the union of the two shapes at the middle of a band (0 = pure morph)
    smooth: float = 0.8  # Gaussian sigma on the composite SDF, voxels
    vs: float | None = None  # output voxel size (default: the body's)


@dataclass
class Placement:
    """Maps target points into one source car: x shift, then a height / width scale that ramps along x."""

    src: SourceCar
    dx: float  # x_src = x_target + dx
    tip: float  # target x where the ramp is 0 (the part's far end)
    seam: float  # target x where the ramp is 1
    ry: float = 1.0  # width ratio body / part at the seam
    rz: float = 1.0
    match: float = 0.0

    def ramp(self, x):
        if self.tip == self.seam:
            return x * 0
        t = (self.tip - x) / (self.tip - self.seam)
        return t.clamp(0, 1) if torch.is_tensor(x) else np.clip(t, 0, 1)

    def scales(self, x):
        r = self.ramp(x)
        return 1 + self.match * (self.ry - 1) * r, 1 + self.match * (self.rz - 1) * r

    def to_src(self, p):
        """Target points (N, 3) -> source points (torch or numpy)."""
        sy, sz = self.scales(p[:, 0])
        cat = torch.stack if torch.is_tensor(p) else np.stack
        return cat([p[:, 0] + self.dx, p[:, 1] / sy, p[:, 2] / sz], 1)

    def to_target_x(self, x_src):
        return x_src - self.dx


def smoothstep(t):
    t = t.clamp(0, 1) if torch.is_tensor(t) else np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


class Hybrid:
    def __init__(self, front: SourceCar, body: SourceCar, rear: SourceCar, plan: Plan = Plan()):
        self.front, self.body, self.rear, self.plan = front, body, rear, plan
        fa, ra = body.axles
        self.seam_f, self.seam_r = fa - plan.front_cut, ra + plan.rear_cut
        if self.seam_f - self.seam_r < plan.band:
            raise ValueError("front and rear seams overlap: reduce front_cut / rear_cut / band")
        pb = body.profiles()
        f_lo, f_hi = front.extent_x()
        r_lo, r_hi = rear.extent_x()
        self.pf = Placement(front, front.axles[0] - fa, f_hi - (front.axles[0] - fa), self.seam_f, match=plan.match)
        self.pr = Placement(rear, rear.axles[1] - ra, r_lo - (rear.axles[1] - ra), self.seam_r, match=plan.match)
        self.pb = Placement(body, 0.0, 0.0, 0.0)
        for pl in (self.pf, self.pr):  # cross sections at the seam: body vs part
            ps = pl.src.profiles()
            xb, xs = pl.seam, pl.seam + pl.dx
            hb, wb = (np.interp(xb, pb["x"], np.nan_to_num(pb[k])) for k in ("top", "half"))
            hs, ws = (np.interp(xs, ps["x"], np.nan_to_num(ps[k])) for k in ("top", "half"))
            pl.rz, pl.ry = (hb / hs if hs > 0 else 1.0), (wb / ws if ws > 0 else 1.0)

    # -- weights -------------------------------------------------------------------------------------------------
    def weights(self, x):
        b = self.plan.band
        wf = smoothstep((x - (self.seam_f - b / 2)) / b)
        wr = smoothstep(((self.seam_r + b / 2) - x) / b)
        return {"front": wf, "body": 1 - wf - wr, "rear": wr}

    def parts(self) -> dict[str, Placement]:
        return {"front": self.pf, "body": self.pb, "rear": self.pr}

    # -- geometry ------------------------------------------------------------------------------------------------
    def bounds(self):
        f_hi = self.pf.src.extent_x()[1] - self.pf.dx
        r_lo = self.pr.src.extent_x()[0] - self.pr.dx
        b_lo, b_hi = self.body.extent_x()
        half, top = 0.0, 0.0
        for pl in self.parts().values():
            pr = pl.src.profiles()
            k = max(pl.ry, 1.0) if pl.match else 1.0
            kz = max(pl.rz, 1.0) if pl.match else 1.0
            half, top = max(half, np.nanmax(pr["half"]) * k), max(top, np.nanmax(pr["top"]) * kz)
        vs = self.plan.vs or self.body.vs
        lo = np.array([min(r_lo, b_lo), -half, 0.0]) - 4 * vs
        hi = np.array([max(f_hi, b_hi), half, top]) + 4 * vs
        return lo, hi, vs

    @staticmethod
    def _sampler(src: SourceCar):
        vol = torch.tensor(src.sdf, device=DEV)[None, None]  # (1, 1, X, Y, Z)
        lo = torch.tensor(src.lo, dtype=torch.float32, device=DEV)
        n = torch.tensor(src.sdf.shape, dtype=torch.float32, device=DEV)
        box_lo, box_hi = lo + 0.5 * src.vs, lo + (n - 0.5) * src.vs

        def sample(p):
            q = torch.maximum(torch.minimum(p, box_hi), box_lo)  # outside the grid: border value + distance
            g = 2 * ((q - lo) / src.vs - 0.5) / (n - 1) - 1
            grid = g[:, [2, 1, 0]].view(1, -1, 1, 1, 3)
            d = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True).view(-1)
            return d + (p - q).norm(dim=1)

        return sample

    @torch.no_grad()
    def field(self, log=print):
        """Composite SDF on the output grid -> (sdf (X, Y, Z), lo, vs)."""
        lo, hi, vs = self.bounds()
        dims = tuple(int(d) for d in np.ceil((hi - lo) / vs))
        log(f"  composite grid {dims} at {vs * 100:.1f} cm")
        samplers = {k: self._sampler(pl.src) for k, pl in self.parts().items()}
        ax = [torch.arange(d, device=DEV, dtype=torch.float32) for d in dims]
        lo_t = torch.tensor(lo, dtype=torch.float32, device=DEV)
        out = torch.empty(dims, device=DEV)
        for i in range(0, dims[0], 32):
            xx, yy, zz = torch.meshgrid(ax[0][i : i + 32], ax[1], ax[2], indexing="ij")
            p = lo_t + (torch.stack([xx, yy, zz], -1).reshape(-1, 3) + 0.5) * vs
            w = self.weights(p[:, 0])
            acc = torch.zeros(len(p), device=DEV)
            env = torch.full((len(p),), float("inf"), device=DEV)
            w2 = torch.zeros(len(p), device=DEV)
            for k, pl in self.parts().items():
                m = w[k] > 1e-6
                if m.any():
                    d = samplers[k](pl.to_src(p[m]))
                    acc[m] += w[k][m] * d
                    env[m] = torch.minimum(env[m], d)
                    w2 += w[k] ** 2
            # morphing two different cross sections pinches in the middle of the band (a pickup bed, high at the
            # sides, into an engine cover, high in the middle): blend towards their union there, 4 w (1 - w)
            lam = (2 * self.plan.fill * (1 - w2)).clamp(0, 1)
            out[i : i + 32] = ((1 - lam) * acc + lam * torch.where(torch.isinf(env), acc, env)).view(xx.shape)
        sdf = out.cpu().numpy()
        if self.plan.smooth > 0:
            sdf = gaussian_filter(sdf, self.plan.smooth)
        return sdf, lo, vs

    def mesh(self, log=print, field=None):
        """-> (vertices, faces, vertex rgb). ``field``: a precomputed (sdf, lo, vs) from ``field()``."""
        from skimage.measure import marching_cubes

        sdf, lo, vs = field if field is not None else self.field(log)
        v, f, _, _ = marching_cubes(np.pad(sdf, 1, constant_values=1.0), 0.0)
        v = lo + (v - 1 + 0.5) * vs
        tri = v[f]
        if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:  # signed volume: outward
            f = f[:, ::-1].copy()
        log(f"  mesh: {len(v)} vertices, {len(f)} triangles")
        return v, f, self.colors(v)

    def colors(self, v: np.ndarray, k: int = 4) -> np.ndarray:
        """Each vertex takes the colour of the nearest visible surface point of the source(s) it comes from."""
        out = np.zeros((len(v), 3))
        w = self.weights(v[:, 0])
        for name, pl in self.parts().items():
            m = w[name] > 1e-6
            if not m.any():
                continue
            tree = cKDTree(pl.src.color_pts)
            d, j = tree.query(pl.to_src(v[m]), k=k)
            wt = 1 / (d + 1e-4)
            c = (pl.src.color_rgb[j] * wt[..., None]).sum(1) / wt.sum(1, keepdims=True)
            out[m] += w[name][m, None] * c
        return np.clip(out, 0, 1)

    def summary(self) -> dict:
        def part(pl):
            return {"id": pl.src.sid, "name": pl.src.name, "dx": round(pl.dx, 3), "height_ratio": round(pl.rz, 3),
                    "width_ratio": round(pl.ry, 3)}

        return {"plan": asdict(self.plan), "seam_front_x": round(self.seam_f, 3), "seam_rear_x": round(self.seam_r, 3),
                "body_axles": [round(a, 3) for a in self.body.axles],
                "front": part(self.pf), "body": part(self.pb), "rear": part(self.pr)}
