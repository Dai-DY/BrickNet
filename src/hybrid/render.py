"""GPU point-splat rasteriser for triangle meshes (perspective cameras), used for solid fusion and for figures.

Triangles are sampled area-uniformly, densely enough that the front surface has no holes, and z-buffered with
``scatter_reduce(amin)`` on a key that packs quantised depth and triangle id. Shading is Lambert with fixed world
lights, two-sided (normals face the camera).
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

from car2brick.recon.cameras import Camera

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ID_BITS = 26
ID_MASK = (1 << ID_BITS) - 1
Z_LEVELS = (1 << (63 - ID_BITS)) - 2
MAXKEY = torch.iinfo(torch.int64).max
LIGHTS = ((np.array([0.35, 0.5, 1.0]), 0.55), (np.array([-0.6, -0.3, 0.45]), 0.25), (np.array([0.0, 0.0, -1.0]), 0.1))
AMBIENT = 0.3


def orbit(az: float, el: float, dist: float, target, res=(512, 384), fov_half: float | None = None,
          radius: float | None = None) -> Camera:
    """Camera on a sphere around ``target`` (degrees; az = 0 looks from the front, +x)."""
    a, e = np.radians(az), np.radians(el)
    target = np.asarray(target, float)
    eye = target + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    if fov_half is None:
        fov_half = np.arcsin(min(0.98, 1.05 * radius / dist))
    w, h = res
    up = (0.0, 0.0, 1.0) if abs(np.sin(e)) < 0.98 else (1.0, 0.0, 0.0)
    return Camera.look_at("orbit", eye, target, (min(w, h) / 2) / np.tan(fov_half), w, h, up=up)


class MeshRenderer:
    def __init__(self, tri: np.ndarray, rgb: np.ndarray):
        tri = torch.as_tensor(np.asarray(tri), dtype=torch.float32, device=DEV)
        n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
        a2 = n.norm(dim=1)
        ok = a2 > 1e-14
        self.tri, self.n, self.area = tri[ok], n[ok] / a2[ok, None], 0.5 * a2[ok]
        self.rgb = torch.as_tensor(np.asarray(rgb), dtype=torch.float32, device=DEV)[ok]
        assert len(self.tri) <= ID_MASK
        self.lights = [(torch.tensor(d / np.linalg.norm(d), dtype=torch.float32, device=DEV), w) for d, w in LIGHTS]

    @torch.no_grad()
    def raster(self, cam: Camera, ss: int = 2, pts_per_px: float = 4.0, chunk: int = 8_000_000, seed: int = 0):
        """-> dict(ok, fid, depth (camera z), points (world)) per pixel at ``ss`` x supersampling, flattened."""
        W, H = cam.width * ss, cam.height * ss
        K = torch.tensor(np.diag([ss, ss, 1.0]) @ cam.K, dtype=torch.float32, device=DEV)
        R = torch.tensor(cam.R, dtype=torch.float32, device=DEV)
        t = torch.tensor(cam.t, dtype=torch.float32, device=DEV)
        center = torch.tensor(cam.center, dtype=torch.float32, device=DEV)
        dist = (self.tri.mean(1) - center).norm(dim=1)
        zfar = float(dist.max()) * 1.5
        rho = pts_per_px * (float(K[0, 0]) / max(float(dist.min()), 1e-3)) ** 2
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
            pc = p @ R.T + t
            z = pc[:, 2]
            uvw = pc @ K.T
            u = torch.floor(uvw[:, 0] / z).long()
            v = torch.floor(uvw[:, 1] / z).long()
            ok = (z > 1e-4) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            zq = (z[ok] / zfar * Z_LEVELS).long().clamp(0, Z_LEVELS)
            zbuf.scatter_reduce_(0, v[ok] * W + u[ok], (zq << ID_BITS) | ti[ok], reduce="amin")
        ok = zbuf != MAXKEY
        fid = torch.where(ok, zbuf & ID_MASK, torch.full_like(zbuf, -1))
        depth = torch.where(ok, ((zbuf >> ID_BITS).double() + 0.5).float() / Z_LEVELS * zfar,
                            torch.full_like(zbuf, float("inf"), dtype=torch.float32))
        pix = torch.arange(H * W, device=DEV)
        uv1 = torch.stack([(pix % W).float() + 0.5, (pix // W).float() + 0.5, torch.ones(H * W, device=DEV)], 1)
        pc = (uv1 @ torch.linalg.inv(K).T) * torch.where(ok, depth, torch.zeros_like(depth))[:, None]
        pw = (pc - t) @ R
        return {"ok": ok, "fid": fid, "depth": depth, "points": pw, "W": W, "H": H, "center": center}

    def shade(self, fid, pw, center):
        base = self.rgb[fid]
        n = self.n[fid]
        view = center - pw
        view = view / view.norm(dim=1, keepdim=True).clamp_min(1e-9)
        n = n * torch.sign((n * view).sum(1, keepdim=True) + 1e-12)
        diff = torch.full_like(base[:, :1], AMBIENT)
        for d, w in self.lights:
            diff = diff + w * (n @ d).clamp_min(0)[:, None]
        rim = 0.12 * (1 - (n * view).sum(1, keepdim=True).clamp(0, 1)) ** 2  # a little edge light
        return (base * diff + rim).clamp(0, 1)

    @torch.no_grad()
    def render(self, cam: Camera, ss: int = 2, bg=(1.0, 1.0, 1.0), emissive: bool = False) -> np.ndarray:
        r = self.raster(cam, ss)
        col = torch.zeros(r["H"] * r["W"], 3, device=DEV)
        idx = r["ok"].nonzero().squeeze(1)
        col[idx] = self.rgb[r["fid"][idx]] if emissive else self.shade(r["fid"][idx], r["points"][idx], r["center"])
        b = torch.tensor(bg, dtype=torch.float32, device=DEV)
        col[~r["ok"]] = b
        img = col.view(cam.height, ss, cam.width, ss, 3).mean((1, 3))
        return img.clamp(0, 1).cpu().numpy()


# ---- figure helpers -----------------------------------------------------------------------------------------------
def to_img(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def label(img: Image.Image, text: str, size: int = 14) -> Image.Image:
    d = ImageDraw.Draw(img)
    try:
        from PIL import ImageFont

        font = ImageFont.load_default(size=size)
    except TypeError:  # old Pillow
        font = None
    d.text((8, 6), text, fill=(20, 20, 20), font=font)
    return img


def grid(images: list[Image.Image], cols: int, bg=(255, 255, 255)) -> Image.Image:
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (cols * w, rows * h), bg)
    for i, im in enumerate(images):
        out.paste(im, ((i % cols) * w, (i // cols) * h))
    return out
