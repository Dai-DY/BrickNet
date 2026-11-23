"""Space carving (Kutulakos & Seitz, "A Theory of Shape by Space Carving", IJCV 2000) on synthetic ShapeNet views.

Experiment: render a ShapeNet car from N random poses (perspective cameras, known intrinsics / extrinsics), then
recover its 3D shape from the images alone and compare with the mesh.

1. Render. Point-splat z-buffer renderer on the GPU (area-uniform samples, 2x supersampling). Shading is Lambert
   with lights fixed in the world, so a surface point has the same colour in every view - the photo-consistency
   assumption of space carving. ``--specular`` adds a Blinn-Phong highlight (view dependent, like car paint) to
   test what happens when that assumption breaks. ``--noise`` adds sensor noise.
2. Silhouette carving (visual hull) of a voxel grid: the initial volume, and the baseline.
3. Photo-consistency carving. Repeat until nothing changes: take the surface voxels of the current volume, build
   an item buffer per camera (each surface voxel is splatted with sub-voxel points, nearest wins), so every voxel
   knows exactly which pixels see it given the current shape; a voxel seen by >= ``min_views`` cameras whose colours
   disagree (std > ``tau``) cannot be on the surface and is carved. By the space carving theorem every carved voxel
   lies outside the photo hull, so the result converges to the photo hull (the largest photo-consistent shape).
   This is the multi-pass variant with exact visibility (as in Generalized Voxel Coloring) instead of plane sweeps.
4. Evaluate against the mesh: accuracy / completeness / Chamfer / F-score against the surface visible from the
   camera region (back-projected GT depth from many views), IoU against ShapeNet's ``model_normalized.solid.binvox``.

    python scripts/space_carving.py --random 3 --views 4,8,16,32
    python scripts/space_carving.py --ids 100715345ee54d7ae38b52b4ee9d36a3 --views 12 --specular 0.6
    python scripts/space_carving.py --prior outputs/car_prior/prior.npz --random 6 --views 4,8,16 --quorum 0.5

``--prior`` (build it with ``scripts/car_prior.py build``) adds a clean-up stage: left-right symmetry, SDF smoothing
and a PCA car shape prior, scored as meshes (``refine_<n>.jpg``, ``refined_*.obj``); see ``car_prior.py``.

Outputs per car: <out>/<id>/{inputs_<n>.jpg, compare_<n>.jpg, cameras_<n>.png, recon_<n>.ply, metrics_<n>.json},
plus <out>/summary.md. Units in the metrics are % of the car length.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baselines import outer_faces, read_obj  # noqa: E402

from car2brick.recon.cameras import Camera  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Y_UP_TO_Z_UP = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # ShapeNet is y-up; work z-up (car length along y)
LIGHTS = ((np.array([0.35, 0.5, 1.0]), 0.55), (np.array([-0.6, -0.3, 0.45]), 0.25))
AMBIENT = 0.3
ID_BITS = 26
ID_MASK = (1 << ID_BITS) - 1
Z_LEVELS = (1 << (63 - ID_BITS)) - 2
MAXKEY = torch.iinfo(torch.int64).max
BG = np.array([1.0, 1.0, 1.0])


# ---- cameras ------------------------------------------------------------------------------------------------------
def orbit_camera(name, az, el, dist, radius, res, target=(0.0, 0.0, 0.0), margin=1.15) -> Camera:
    a, e = np.radians(az), np.radians(el)
    target = np.asarray(target, float)
    eye = target + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    half = np.arcsin(min(0.98, margin * radius / dist))  # bounding sphere fits in the frame
    return Camera.look_at(name, eye, target, (res / 2) / np.tan(half), res, res)


def random_cameras(n, radius, res, rng, elev=(5.0, 50.0), dist=(2.0, 3.0), jitter=0.08) -> list[Camera]:
    cams = []
    for i in range(n):
        target = rng.normal(0, jitter * radius, 3)
        cams.append(orbit_camera(f"view{i:02d}", rng.uniform(0, 360), rng.uniform(*elev),
                                 rng.uniform(*dist) * radius, radius, res, target, margin=0.95 + 1.5 * jitter))
    return cams


def cam_tensors(cam: Camera, ss: int = 1):
    K = np.diag([ss, ss, 1.0]) @ cam.K
    return (torch.tensor(K, dtype=torch.float32, device=DEV), torch.tensor(cam.R, dtype=torch.float32, device=DEV),
            torch.tensor(cam.t, dtype=torch.float32, device=DEV))


# ---- z-buffer of points -------------------------------------------------------------------------------------------
def zbuffer_points(p, ids, K, R, t, W, H, zfar, zbuf):
    """Splat points (N, 3) with integer ids into ``zbuf`` (H*W int64, nearest wins; key = depth << ID_BITS | id)."""
    pc = p @ R.T + t
    z = pc[:, 2]
    uvw = pc @ K.T
    u = torch.floor(uvw[:, 0] / z).long()
    v = torch.floor(uvw[:, 1] / z).long()
    ok = (z > 1e-4) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    zq = (z[ok] / zfar * Z_LEVELS).long().clamp(0, Z_LEVELS)
    key = (zq << ID_BITS) | ids[ok]
    zbuf.scatter_reduce_(0, v[ok] * W + u[ok], key, reduce="amin")


def decode(zbuf, zfar):
    ok = zbuf != MAXKEY
    ids = zbuf & ID_MASK
    z = ((zbuf >> ID_BITS).double() + 0.5) / Z_LEVELS * zfar
    return ok, ids, z.float()


# ---- mesh renderer ------------------------------------------------------------------------------------------------
class MeshRenderer:
    """Triangles (T, 3, 3) + per-face colour (T, 3) -> images. ``emissive``: show the colours as is (no shading)."""

    def __init__(self, tri: np.ndarray, rgb: np.ndarray, specular: float = 0.0, shininess: float = 40.0,
                 emissive: bool = False):
        tri = torch.as_tensor(tri, dtype=torch.float32, device=DEV)
        n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
        a2 = n.norm(dim=1)
        ok = a2 > 1e-12
        self.tri, self.n, self.area = tri[ok], n[ok] / a2[ok, None], 0.5 * a2[ok]
        self.rgb = torch.as_tensor(rgb, dtype=torch.float32, device=DEV)[ok]
        assert len(self.tri) <= ID_MASK
        self.specular, self.shininess, self.emissive = specular, shininess, emissive
        self.lights = [(torch.tensor(d / np.linalg.norm(d), dtype=torch.float32, device=DEV), w) for d, w in LIGHTS]

    @torch.no_grad()
    def render(self, cam: Camera, ss: int = 2, pts_per_px: float = 4.0, chunk: int = 8_000_000, seed: int = 0):
        """-> dict(rgb (H,W,3) foreground colour, mask (H,W), coverage (H,W), display (H,W,3), points (M,3) world
        points of the visible surface at supersampled resolution)."""
        W, H = cam.width * ss, cam.height * ss
        K, R, t = cam_tensors(cam, ss)
        center = torch.tensor(cam.center, dtype=torch.float32, device=DEV)
        dist = (self.tri.mean(1) - center).norm(dim=1)
        zfar = float(dist.max()) * 1.5
        rho = pts_per_px * (float(K[0, 0]) / max(float(dist.min()), 1e-3)) ** 2  # samples / area, no holes
        gen = torch.Generator(device=DEV).manual_seed(seed)
        cnt = torch.floor(self.area * rho + torch.rand(len(self.area), generator=gen, device=DEV)).long()
        cum = cnt.cumsum(0)
        total = int(cum[-1])
        nchunk = total // chunk + 1
        edges = torch.searchsorted(cum, torch.linspace(0, total, nchunk + 1, device=DEV)[1:-1].long()).tolist()
        edges = [0, *edges, len(cnt)]
        zbuf = torch.full((H * W,), MAXKEY, dtype=torch.int64, device=DEV)
        for s, e in zip(edges[:-1], edges[1:]):
            if e <= s:
                continue
            ti = torch.repeat_interleave(torch.arange(s, e, device=DEV), cnt[s:e])
            r1 = torch.rand(len(ti), generator=gen, device=DEV).sqrt()[:, None]
            r2 = torch.rand(len(ti), generator=gen, device=DEV)[:, None]
            tr = self.tri[ti]
            p = (1 - r1) * tr[:, 0] + r1 * (1 - r2) * tr[:, 1] + r1 * r2 * tr[:, 2]
            zbuffer_points(p, ti, K, R, t, W, H, zfar, zbuf)
        ok, tid, z = decode(zbuf, zfar)
        pix = ok.nonzero().squeeze(1)
        tid, z = tid[pix], z[pix]
        uv1 = torch.stack([(pix % W).float() + 0.5, (pix // W).float() + 0.5, torch.ones_like(z)], 1)
        pc = (uv1 @ torch.linalg.inv(K).T) * z[:, None]
        pw = (pc - t) @ R  # R^T (pc - t)
        col = self._shade(tid, pw, center)
        img = torch.zeros(H * W, 3, device=DEV)
        img[pix] = col
        cov = ok.float().view(cam.height, ss, cam.width, ss).mean((1, 3))
        fg = img.view(cam.height, ss, cam.width, ss, 3).sum((1, 3)) / (cov * ss * ss).clamp_min(1e-6)[..., None]
        bg = torch.tensor(BG, dtype=torch.float32, device=DEV)
        disp = fg * cov[..., None] + bg * (1 - cov[..., None])
        return {"rgb": fg.clamp(0, 1), "mask": cov >= 0.5, "coverage": cov, "display": disp.clamp(0, 1), "points": pw}

    def _shade(self, tid, pw, center):
        base = self.rgb[tid]
        if self.emissive:
            return base
        n = self.n[tid]
        view = center - pw
        view = view / view.norm(dim=1, keepdim=True).clamp_min(1e-9)
        n = n * torch.sign((n * view).sum(1, keepdim=True) + 1e-12)  # two-sided: face the camera
        diff = torch.full_like(base[:, :1], AMBIENT)
        spec = torch.zeros_like(diff)
        for d, w in self.lights:
            diff = diff + w * (n @ d).clamp_min(0)[:, None]
            if self.specular > 0:
                h = d[None] + view
                h = h / h.norm(dim=1, keepdim=True)
                spec = spec + w * (n * h).sum(1, keepdim=True).clamp_min(0) ** self.shininess
        return (base * diff + self.specular * spec).clamp(0, 1)


# ---- voxels -------------------------------------------------------------------------------------------------------
_FACES = []  # (normal axis, sign) -> 4 corner offsets of the quad, counter-clockwise seen from outside
for ax in range(3):
    for sgn in (-1, 1):
        u, w = [a for a in range(3) if a != ax]
        quad = []
        for du, dw in ((0, 0), (1, 0), (1, 1), (0, 1)):
            c = np.zeros(3)
            c[ax] = 1.0 if sgn > 0 else 0.0
            c[u], c[w] = du, dw
            quad.append(c)
        _FACES.append((ax, sgn, np.array(quad)))


def voxel_triangles(occ: np.ndarray, lo: np.ndarray, vs: float, colors: np.ndarray | None = None):
    """Exposed faces of occupied voxels -> triangles (T, 3, 3) and per-triangle colour (from ``colors`` (X,Y,Z,3))."""
    pad = np.pad(occ, 1)
    tris, cols = [], []
    for ax, sgn, quad in _FACES:
        nb = np.roll(pad, -sgn, axis=ax)[1:-1, 1:-1, 1:-1]
        idx = np.argwhere(occ & ~nb)
        if not len(idx):
            continue
        corners = lo + (idx[:, None, :] + quad[None]) * vs  # (M, 4, 3)
        tris += [corners[:, [0, 1, 2]], corners[:, [0, 2, 3]]]
        c = colors[tuple(idx.T)] if colors is not None else np.full((len(idx), 3), 0.72)
        cols += [c, c]
    return np.concatenate(tris), np.concatenate(cols)


class Carver:
    def __init__(self, lo, hi, n_long: int, cams: list[Camera], imgs: list[np.ndarray], masks: list[np.ndarray]):
        self.vs = float((hi - lo).max() / n_long)
        self.dims = tuple(int(d) for d in np.ceil((hi - lo) / self.vs))
        self.lo = np.asarray(lo, float)
        self.lo_t = torch.tensor(self.lo, dtype=torch.float32, device=DEV)
        self.cams = cams
        self.cam_t = [cam_tensors(c) for c in cams]
        self.imgs = [torch.as_tensor(i, dtype=torch.float32, device=DEV).reshape(-1, 3) for i in imgs]
        self.masks = [torch.as_tensor(m, device=DEV).reshape(-1) for m in masks]
        # sub-voxel splat density: <= 0.5 px between samples in the closest view, so item buffers have no holes
        d_min = min(np.linalg.norm(c.center - (self.lo + hi) / 2) - np.linalg.norm(hi - lo) / 2 for c in cams)
        foot = max(c.K[0, 0] for c in cams) * self.vs / max(d_min, 1e-3)
        sub = int(np.clip(np.ceil(2 * foot), 3, 8))
        o = (torch.arange(sub, device=DEV, dtype=torch.float32) + 0.5) / sub - 0.5
        self.offsets = torch.stack(torch.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3) * self.vs
        self.zfar = max(np.linalg.norm(c.center - self.lo) + np.linalg.norm(hi - lo) for c in cams)

    def centers(self, idx: torch.Tensor) -> torch.Tensor:
        return self.lo_t + (idx.float() + 0.5) * self.vs

    @torch.no_grad()
    def silhouette(self) -> torch.Tensor:
        """Visual hull: keep voxels whose centre falls inside the mask in every view that sees it."""
        idx = torch.stack(torch.meshgrid(*[torch.arange(d, device=DEV) for d in self.dims], indexing="ij"), -1)
        c = self.centers(idx.reshape(-1, 3))
        miss = torch.zeros(len(c), dtype=torch.bool, device=DEV)
        seen = torch.zeros(len(c), dtype=torch.int32, device=DEV)
        for cam, (K, R, t), m in zip(self.cams, self.cam_t, self.masks):
            pc = c @ R.T + t
            uvw = pc @ K.T
            u = torch.floor(uvw[:, 0] / pc[:, 2]).long()
            v = torch.floor(uvw[:, 1] / pc[:, 2]).long()
            ok = (pc[:, 2] > 0) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
            inside = torch.zeros_like(ok)
            inside[ok] = m[v[ok] * cam.width + u[ok]]
            miss |= ok & ~inside
            seen += ok.int()
        return (~miss & (seen >= 2)).reshape(self.dims)

    @staticmethod
    def surface(occ: torch.Tensor) -> torch.Tensor:
        o = F.pad(occ[None, None].float(), (1, 1, 1, 1, 1, 1))[0, 0]
        full = (o[2:, 1:-1, 1:-1] * o[:-2, 1:-1, 1:-1] * o[1:-1, 2:, 1:-1] * o[1:-1, :-2, 1:-1]
                * o[1:-1, 1:-1, 2:] * o[1:-1, 1:-1, :-2])
        return occ & (full < 0.5)

    @torch.no_grad()
    def stats(self, occ: torch.Tensor):
        """Per surface voxel and camera: sum of the colours of the pixels it owns, their squared norm, pixel count."""
        sidx = self.surface(occ).nonzero()
        S, C, k = len(sidx), len(self.cams), len(self.offsets)
        c = self.centers(sidx)
        sums = torch.zeros(S * C, 3, device=DEV)
        sq = torch.zeros(S * C, device=DEV)
        cnt = torch.zeros(S * C, device=DEV)
        ids = torch.arange(S, device=DEV).repeat_interleave(k)
        step = max(1, 40_000_000 // k)
        for ci, (cam, (K, R, t), img, m) in enumerate(zip(self.cams, self.cam_t, self.imgs, self.masks)):
            zbuf = torch.full((cam.width * cam.height,), MAXKEY, dtype=torch.int64, device=DEV)
            for s in range(0, S, step):
                p = (c[s : s + step, None, :] + self.offsets[None]).reshape(-1, 3)
                zbuffer_points(p, ids[s * k : (s + step) * k], K, R, t, cam.width, cam.height, self.zfar, zbuf)
            ok, vid, _ = decode(zbuf, self.zfar)
            pix = (ok & m).nonzero().squeeze(1)  # background pixels carry no colour evidence
            row = vid[pix] * C + ci
            col = img[pix]
            sums.index_add_(0, row, col)
            sq.index_add_(0, row, (col * col).sum(1))
            cnt.index_add_(0, row, torch.ones(len(pix), device=DEV))
        return sidx, sums.view(S, C, 3), sq.view(S, C), cnt.view(S, C)

    @staticmethod
    def consistency(sums, sq, cnt, min_px: int, stat: str):
        """-> (std per voxel, number of views that see it, mean colour)."""
        seen = cnt >= min_px
        nv = seen.sum(1)
        m = sums / cnt.clamp_min(1)[..., None]
        if stat == "between":  # std of the per-view mean colours, every view weighted equally
            w = seen.float()
            mean = (m * w[..., None]).sum(1) / w.sum(1).clamp_min(1)[:, None]
            var = (((m - mean[:, None]) ** 2).sum(-1) * w).sum(1) / w.sum(1).clamp_min(1) / 3
        else:  # "pooled": std of all pixels (K&S), includes the colour spread inside each footprint
            w = seen.float()
            n = (cnt * w).sum(1).clamp_min(1)
            mean = (sums * w[..., None]).sum(1) / n[:, None]
            var = ((sq * w).sum(1) / n - (mean * mean).sum(1)).clamp_min(0) / 3
        return var.sqrt(), nv, mean

    @staticmethod
    def robust_agreement(sums, cnt, tau: float, min_px: int) -> torch.Tensor:
        """Share of the views that see a voxel whose mean colour is within ``tau`` of the median over views."""
        seen = cnt >= min_px
        m = sums / cnt.clamp_min(1)[..., None]
        ref = torch.where(seen[..., None], m, torch.nan).nanmedian(1).values
        dev = ((m - ref[:, None]) ** 2).sum(-1).sqrt() / 3**0.5
        return ((dev < tau) & seen).sum(1).float() / seen.sum(1).clamp_min(1)

    @torch.no_grad()
    def photo_carve(self, occ, tau=0.06, min_views=2, min_px=2, stat="between", max_iter=100, quorum=0.0,
                    max_depth=0.0, log=print):
        """``quorum`` > 0: robust test, carve only when fewer than this share of the views agree with the median
        colour (instead of the strict std test). ``max_depth`` > 0: never carve deeper than this many voxels below
        the initial (visual hull) surface, which stops a wrong carve from tunnelling through a thin-shelled car."""
        occ = occ.clone()
        history = []
        carvable = None
        if max_depth > 0:
            from scipy.ndimage import distance_transform_edt

            carvable = torch.as_tensor(distance_transform_edt(occ.cpu().numpy()) <= max_depth, device=DEV)
        for it in range(max_iter):
            sidx, sums, sq, cnt = self.stats(occ)
            sigma, nv, _ = self.consistency(sums, sq, cnt, min_px, stat)
            if quorum > 0:
                bad = (nv >= min_views) & (self.robust_agreement(sums, cnt, tau, min_px) < quorum)
            else:
                bad = (nv >= min_views) & (sigma > tau)
            if carvable is not None:
                bad &= carvable[tuple(sidx.T)]
            nb = int(bad.sum())
            history.append({"iter": it, "surface": len(sidx), "carved": nb, "occupied": int(occ.sum())})
            if it % 5 == 0 or nb == 0:
                log(f"    iter {it:3d}: surface {len(sidx):7d}  carved {nb:6d}  occupied {int(occ.sum()):8d}")
            if nb == 0:
                break
            occ[tuple(sidx[bad].T)] = False
        return occ, history

    @torch.no_grad()
    def colors(self, occ, min_px=1, stat="between") -> np.ndarray:
        sidx, sums, sq, cnt = self.stats(occ)
        _, nv, mean = self.consistency(sums, sq, cnt, min_px, stat)
        out = np.full((*self.dims, 3), 0.5, np.float32)
        s = sidx.cpu().numpy()
        out[tuple(s.T)] = np.where((nv > 0).cpu().numpy()[:, None], mean.cpu().numpy(), 0.5)
        return out


# ---- evaluation ---------------------------------------------------------------------------------------------------
def surface_points(occ: np.ndarray, lo, vs) -> np.ndarray:
    s = Carver.surface(torch.as_tensor(occ)).numpy()
    return lo + (np.argwhere(s) + 0.5) * vs


def surface_metrics(rec: np.ndarray, gt: np.ndarray, length: float, f_tau: float = 0.02) -> dict:
    d_acc = cKDTree(gt).query(rec)[0] / length
    d_com = cKDTree(rec).query(gt)[0] / length
    prec, rcl = float((d_acc < f_tau).mean()), float((d_com < f_tau).mean())
    return {"accuracy": 100 * float(d_acc.mean()), "completeness": 100 * float(d_com.mean()),
            "chamfer": 50 * float(d_acc.mean() + d_com.mean()), "precision": prec, "recall": rcl,
            "fscore": 2 * prec * rcl / max(prec + rcl, 1e-9)}


def binvox_occupancy(path: Path, carver: Carver, shift: np.ndarray) -> np.ndarray | None:
    """GT solid voxels resampled at the carving grid's voxel centres."""
    if not path.is_file():
        return None
    with open(path, "rb") as fh:
        dims = trans = scale = None
        while True:
            line = fh.readline().strip()
            if line.startswith(b"dim"):
                dims = [int(x) for x in line.split()[1:]]
            elif line.startswith(b"translate"):
                trans = np.array([float(x) for x in line.split()[1:]])
            elif line.startswith(b"scale"):
                scale = float(line.split()[1])
            elif line == b"data":
                break
        raw = np.frombuffer(fh.read(), dtype=np.uint8)
    vals, counts = raw[::2], raw[1::2]
    vox = np.repeat(vals, counts).astype(bool).reshape(dims)  # binvox order: x, z, y (y fastest)
    vox = vox.transpose(0, 2, 1)  # -> x, y, z
    idx = np.stack(np.meshgrid(*[np.arange(d) for d in carver.dims], indexing="ij"), -1).reshape(-1, 3)
    p_world = carver.lo + (idx + 0.5) * carver.vs
    p_norm = (p_world + shift) @ Y_UP_TO_Z_UP  # inverse of the z-up rotation and centring
    g = np.floor((p_norm - trans) / scale * dims[0]).astype(int)
    ok = np.all((g >= 0) & (g < dims[0]), axis=1)
    out = np.zeros(len(idx), bool)
    out[ok] = vox[g[ok, 0], g[ok, 1], g[ok, 2]]
    return out.reshape(carver.dims)


def iou(a, b) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


# ---- figures ------------------------------------------------------------------------------------------------------
def to_img(x) -> Image.Image:
    a = x.cpu().numpy() if torch.is_tensor(x) else x
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def labelled(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 + 7 * len(text), 16], fill=(255, 255, 255))
    d.text((4, 2), text, fill=(0, 0, 0))
    return img


def grid(images: list[Image.Image], cols: int) -> Image.Image:
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, im in enumerate(images):
        out.paste(im, ((i % cols) * w, (i // cols) * h))
    return out


def camera_plot(path: Path, cams: list[Camera], pts: np.ndarray, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = np.array([cam.center for cam in cams])
    fwd = np.array([cam.R[2] for cam in cams])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    sub = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 5000), replace=False)]
    for ax, (i, j), name in zip(axes, ((0, 1), (1, 2)), ("top (x-y)", "side (y-z)")):
        ax.scatter(sub[:, i], sub[:, j], s=0.3, c="0.6")
        ax.quiver(c[:, i], c[:, j], fwd[:, i], fwd[:, j], color="C3", scale=12, width=0.004)
        ax.scatter(c[:, i], c[:, j], c="C3", s=12)
        for k, p in enumerate(c):
            ax.annotate(str(k), (p[i], p[j]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_aspect("equal")
        ax.set_title(name)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_ply(path: Path, pts: np.ndarray, rgb: np.ndarray):
    rgb8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    with open(path, "w") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\n"
                 "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(pts, rgb8):
            fh.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}\n")


# ---- experiment ---------------------------------------------------------------------------------------------------
def load_car(obj: Path):
    v, f, rgb, alpha = read_obj(obj)
    keep = outer_faces(v, f)
    f, rgb, alpha = f[keep], rgb[keep], alpha[keep]
    v = v @ Y_UP_TO_Z_UP.T
    shift = (v.min(0) + v.max(0)) / 2
    v = v - shift
    glass = alpha < 0.9  # see-through glass rendered as dark tinted opaque (what windows look like in photos)
    rgb = np.where(glass[:, None], 0.3 * rgb + 0.05, rgb)
    return v, f, rgb, shift


def gt_visible_points(rend: MeshRenderer, radius: float, elev, n: int = 48, res: int = 256) -> np.ndarray:
    """The surface the camera region can observe: back-projected GT depth from many views over the same elevations."""
    pts = []
    for i in range(n):
        el = elev[0] + (elev[1] - elev[0]) * ((i * 0.618034) % 1.0)
        cam = orbit_camera("gt", 360.0 * i / n, el, 2.5 * radius, radius, res)
        p = rend.render(cam, ss=1, pts_per_px=3.0, seed=i)["points"]
        pts.append(p[torch.randperm(len(p), device=DEV)[:8000]].cpu().numpy())
    return np.concatenate(pts)


def run_car(sid: str, name: str, args, out_root: Path) -> list[dict]:
    obj = Path(args.shapenet) / sid / "models" / "model_normalized.obj"
    v, f, rgb, shift = load_car(obj)
    length = float(np.ptp(v, 0).max())
    radius = float(np.linalg.norm(np.ptp(v, 0)) / 2)
    out = out_root / sid
    out.mkdir(parents=True, exist_ok=True)
    rend = MeshRenderer(v[f], rgb, specular=args.specular)
    gray = MeshRenderer(v[f], np.full((len(f), 3), 0.72))
    print(f"[{sid}] {name}: {len(f)} faces, length {length:.3f}")
    gt_pts = gt_visible_points(rend, radius, args.elev)
    lo, hi = v.min(0) - 0.05 * length, v.max(0) + 0.05 * length
    rng = np.random.default_rng(args.seed)
    all_cams = random_cameras(max(args.views), radius, args.res, rng, args.elev, args.dist)
    shots = []
    for i, cam in enumerate(all_cams):
        r = rend.render(cam, seed=100 + i)
        img = r["rgb"].cpu().numpy()
        if args.noise > 0:
            img = np.clip(img + rng.normal(0, args.noise, img.shape), 0, 1)
        shots.append((img, r["mask"].cpu().numpy(), r["display"].cpu().numpy()))
    novel = [orbit_camera(f"novel{az}", az, el, 2.4 * radius, radius, 320) for az, el in
             ((35, 20), (100, 12), (215, 25), (300, 60))]
    gt_novel = [to_img(gray.render(c)["display"]) for c in novel]
    rows = []
    for n in args.views:
        t0 = time.time()
        cams, imgs, masks = all_cams[:n], [s[0] for s in shots[:n]], [s[1] for s in shots[:n]]
        car = Carver(lo, hi, args.grid, cams, imgs, masks)
        hull = car.silhouette()
        print(f"  {n} views: grid {car.dims}, voxel {car.vs / length * 100:.2f}% of length, "
              f"hull {int(hull.sum())} voxels")
        occ, hist = car.photo_carve(hull, tau=args.tau, min_views=args.min_views, min_px=args.min_px,
                                    stat=args.stat, max_iter=args.max_iter, quorum=args.quorum,
                                    max_depth=args.max_depth)
        torch.cuda.synchronize() if DEV.type == "cuda" else None
        secs = time.time() - t0
        hull_np, occ_np = hull.cpu().numpy(), occ.cpu().numpy()
        cols = car.colors(occ, stat=args.stat)
        gt_solid = binvox_occupancy(obj.parent / "model_normalized.solid.binvox", car, shift)
        row = {"id": sid, "name": name, "views": n, "tau": args.tau, "quorum": args.quorum,
               "max_depth": args.max_depth,
               "specular": args.specular, "noise": args.noise, "elev": list(args.elev),
               "grid": list(car.dims), "voxel_pct": 100 * car.vs / length, "seconds": round(secs, 1),
               "iterations": len(hist), "hull_voxels": int(hull_np.sum()), "carved_voxels": int(occ_np.sum())}
        for key, o in (("hull", hull_np), ("carve", occ_np)):
            row[key] = surface_metrics(surface_points(o, car.lo, car.vs), gt_pts, length)
            if gt_solid is not None:
                row[key]["iou"] = iou(o, gt_solid)
                row[key]["volume_ratio"] = float(o.sum() / max(gt_solid.sum(), 1))
        (out / f"metrics_{n}.json").write_text(json.dumps({**row, "history": hist}, indent=1))
        rows.append(row)
        print(f"    chamfer {row['hull']['chamfer']:.2f} -> {row['carve']['chamfer']:.2f} (% length), "
              f"F@2% {row['hull']['fscore']:.3f} -> {row['carve']['fscore']:.3f}"
              + (f", IoU {row['hull']['iou']:.3f} -> {row['carve']['iou']:.3f}" if gt_solid is not None else "")
              + f", {secs:.1f}s")

        # figures
        sp = surface_points(occ_np, car.lo, car.vs)
        s = Carver.surface(occ).cpu().numpy()
        write_ply(out / f"recon_{n}.ply", sp, cols[s])
        ins = [labelled(to_img(s[2]).resize((256, 256)), f"view {i}") for i, s in enumerate(shots[:n])]
        grid(ins, min(8, n)).save(out / f"inputs_{n}.jpg", quality=90)
        hr = MeshRenderer(*voxel_triangles(hull_np, car.lo, car.vs))
        cr = MeshRenderer(*voxel_triangles(occ_np, car.lo, car.vs))
        cc = MeshRenderer(*voxel_triangles(occ_np, car.lo, car.vs, cols), emissive=True)
        tiles = []
        for c, g in zip(novel, gt_novel):
            tiles += [labelled(g, "GT mesh"), labelled(to_img(hr.render(c)["display"]), "visual hull"),
                      labelled(to_img(cr.render(c)["display"]), "space carving"),
                      labelled(to_img(cc.render(c)["display"]), "space carving (colour)")]
        grid(tiles, 4).save(out / f"compare_{n}.jpg", quality=90)
        camera_plot(out / f"cameras_{n}.png", cams, gt_pts, f"{name[:40]} - {n} views")
        if args.prior_model is not None:
            row["refine"] = refine_and_report(occ_np, car, args, gt_pts, gt_solid, length, novel, gt_novel, out, n,
                                              cr)
            (out / f"metrics_{n}.json").write_text(json.dumps({**row, "history": hist}, indent=1))
    return rows


def sample_surface(tri: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    i = rng.choice(len(tri), count, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(count))[:, None], rng.random(count)[:, None]
    t = tri[i]
    return (1 - r1) * t[:, 0] + r1 * (1 - r2) * t[:, 1] + r1 * r2 * t[:, 2]


def write_obj(path: Path, v: np.ndarray, f: np.ndarray):
    with open(path, "w") as fh:
        fh.writelines(f"v {a:.5f} {b:.5f} {c:.5f}\n" for a, b, c in v)
        fh.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f)


REFINE_LABELS = {"carve": "carve (mesh)", "sym": "+ symmetry", "smooth": "+ smoothing", "prior": "PCA prior fit",
                 "fused": "prior & hull (final)"}


def refine_and_report(occ_np, car, args, gt_pts, gt_solid, length, novel, gt_novel, out, n, hull_render) -> dict:
    """Symmetry / smoothing / shape prior on the carved volume; meshes are scored like the voxels."""
    import car_prior as cp

    res = cp.refine(occ_np, car.lo, car.vs, args.prior_model, smooth=args.smooth, alpha=args.alpha, lam=args.lam)
    metrics, renders = {}, {}
    for key, fld in res["fields"].items():
        verts, faces = cp.field_mesh(fld, car.lo, car.vs)
        tri = verts[faces]
        m = surface_metrics(sample_surface(tri, 60000), gt_pts, length)
        if gt_solid is not None:
            o = fld < 0
            m["iou"] = iou(o, gt_solid)
            m["volume_ratio"] = float(o.sum() / max(gt_solid.sum(), 1))
        metrics[key] = m
        renders[key] = MeshRenderer(tri, np.full((len(tri), 3), 0.72))
        if key in ("smooth", "fused"):
            write_obj(out / f"refined_{key}_{n}.obj", verts, faces)
    metrics["flipped"] = res["flipped"]
    keys = [k for k in ("smooth", "prior", "fused") if k in renders]
    tiles = []
    for c, g in zip(novel, gt_novel):
        tiles += [labelled(g, "GT mesh"), labelled(to_img(hull_render.render(c)["display"]), "space carving")]
        tiles += [labelled(to_img(renders[k].render(c)["display"]), REFINE_LABELS[k]) for k in keys]
    grid(tiles, 2 + len(keys)).save(out / f"refine_{n}.jpg", quality=90)
    print("    refine: " + ", ".join(f"{k} chamfer {metrics[k]['chamfer']:.2f}"
                                    + (f" IoU {metrics[k]['iou']:.3f}" if "iou" in metrics[k] else "")
                                    for k in res["fields"]) + (" (front/back flipped)" if res["flipped"] else ""))
    return metrics


def pick(args) -> list[tuple[str, str]]:
    root = Path(args.shapenet)
    rows = [r for r in csv.DictReader(open(args.index)) if r.get("v2_has_mesh") == "1"] if Path(args.index).is_file() \
        else [{"id": p.name, "name": ""} for p in sorted(root.iterdir())]
    names = {r["id"]: r.get("name", "") for r in rows}
    if args.ids:
        if args.prior_model is not None:
            leak = [i for i in args.ids.split(",") if i not in set(args.prior_model.heldout)]
            if leak:
                print(f"[warn] in the prior's training set (results are optimistic): {leak}")
        return [(i, names.get(i, "")) for i in args.ids.split(",")]
    ok = [r for r in rows if (root / r["id"] / "models" / "model_normalized.obj").is_file()]
    if args.prior_model is not None:  # never test on cars the prior was built from
        held = set(args.prior_model.heldout)
        ok = [r for r in ok if r["id"] in held]
    sel = np.random.default_rng(args.seed).choice(len(ok), args.random, replace=False)
    return [(ok[i]["id"], ok[i].get("name", "")) for i in sel]


def main(argv=None):
    data = Path(__file__).resolve().parents[2] / "shape2brick" / "data"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapenet", default=str(data / "raw" / "ShapeNetCore" / "02958343"))
    ap.add_argument("--index", default=str(data / "cars" / "index" / "shapenet.csv"))
    ap.add_argument("--ids", default="", help="comma-separated ShapeNet ids (else --random)")
    ap.add_argument("--random", type=int, default=1, help="number of random cars")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--views", default="8", help="comma-separated view counts, e.g. 4,8,16 (nested camera sets)")
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument("--grid", type=int, default=128, help="voxels along the longest side")
    ap.add_argument("--tau", type=float, default=0.06, help="photo-consistency threshold (colour std, 0-1 RGB)")
    ap.add_argument("--stat", default="between", choices=["between", "pooled"])
    ap.add_argument("--quorum", type=float, default=0.0,
                    help="robust test: carve when fewer than this share of views agree with the median colour (0 = "
                         "strict std test)")
    ap.add_argument("--max-depth", type=float, default=0.0, help="max carve depth below the visual hull, voxels")
    ap.add_argument("--min-views", type=int, default=2)
    ap.add_argument("--min-px", type=int, default=2, help="pixels a voxel must own in a view to count as seen")
    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--elev", default="5,50", help="camera elevation range in degrees")
    ap.add_argument("--dist", default="2,3", help="camera distance range in bounding-sphere radii")
    ap.add_argument("--specular", type=float, default=0.0, help="Blinn-Phong highlight strength (0 = Lambert)")
    ap.add_argument("--noise", type=float, default=0.0, help="Gaussian pixel noise std (0-1 RGB)")
    ap.add_argument("--prior", default="", help="car_prior.py model (.npz): refine the result with symmetry, "
                    "smoothing and the PCA shape prior; --random then picks held-out cars")
    ap.add_argument("--smooth", type=float, default=0.5, help="Gaussian sigma on the SDF, voxels")
    ap.add_argument("--alpha", type=float, default=0.2, help="prior fit: cost of staying inside the hull")
    ap.add_argument("--lam", type=float, default=3.0, help="prior fit: weight of the coefficient prior")
    ap.add_argument("--out", default="outputs/space_carving")
    a = ap.parse_args(argv)
    a.views = sorted(int(x) for x in a.views.split(","))
    a.elev = tuple(float(x) for x in a.elev.split(","))
    a.dist = tuple(float(x) for x in a.dist.split(","))
    a.prior_model = None
    if a.prior:
        from car_prior import CarPrior

        a.prior_model = CarPrior(Path(a.prior))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid, name in pick(a):
        rows += run_car(sid, name, a, out)
    lines = ["| car | views | chamfer hull | chamfer carve | F@2% hull | F@2% carve | IoU hull | IoU carve | s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        h, c = r["hull"], r["carve"]
        lines.append(f"| {(r['name'] or r['id'])[:30]} | {r['views']} | {h['chamfer']:.2f} | {c['chamfer']:.2f} | "
                     f"{h['fscore']:.3f} | {c['fscore']:.3f} | {h.get('iou', float('nan')):.3f} | "
                     f"{c.get('iou', float('nan')):.3f} | {r['seconds']} |")
    ref = [r for r in rows if "refine" in r]
    if ref:
        keys = [k for k in ("carve", "sym", "smooth", "prior", "fused") if k in ref[0]["refine"]]
        lines += ["", "| car | views | " + " | ".join(keys) + " |  (meshes; chamfer % length / IoU)",
                  "|---|---|" + "---|" * len(keys)]
        for r in ref:
            cell = [f"{r['refine'][k]['chamfer']:.2f} / {r['refine'][k].get('iou', float('nan')):.3f}" for k in keys]
            lines.append(f"| {(r['name'] or r['id'])[:30]} | {r['views']} | " + " | ".join(cell) + " |")
    with open(out / "summary.md", "a") as fh:
        fh.write(f"\n## tau={a.tau} stat={a.stat} quorum={a.quorum} max_depth={a.max_depth} specular={a.specular} "
                 f"noise={a.noise} elev={a.elev}\n\n")
        fh.write("\n".join(lines) + "\n")
    with open(out / "summary.jsonl", "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
