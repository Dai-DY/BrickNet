"""Car shape prior for cleaning up a visual hull / space carving result: symmetry, smoothing and a PCA shape model.

Everything lives on a *bbox-normalised* grid: a car's bounding box (x width, y length, z height, z up) is mapped to
[-0.9, 0.9]^3 and sampled at ``DIMS`` cells; the feature is a truncated signed distance (negative inside, +-1 at
``TRUNC`` cells). Normalising by the box removes size and aspect, so the PCA only models shape.

Refinement of a hull (an *outer bound* of the car: the true car is inside it):
  carve   the input voxels as a mesh (marching cubes of their TSDF), the baseline
  sym     hull intersected with its mirror image (cars are left-right symmetric) - still an outer bound. The
          plane is searched for (most volume kept), not taken from the bbox: hulls bulge unevenly
  smooth  Gaussian-smoothed SDF of ``sym`` -> marching cubes (no staircase)
  prior   PCA model fitted to ``sym`` with an asymmetric loss: bulging *out* of the hull costs ``1``, staying
          *inside* it costs ``alpha`` (hulls bulge over concavities such as wheel arches and windscreens, the
          prior knows they are there). Plus a Gaussian prior on the coefficients (``lam``). Both front/back
          orientations are tried; the better fit wins.
  fused   intersection of ``prior`` and ``sym`` (keep the prior's concavities, never leave the hull), smoothed

    python scripts/car_prior.py build --out outputs/car_prior/prior.npz       # ~1-2 min, held-out split saved

Used by ``space_carving.py --prior outputs/car_prior/prior.npz``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

DIMS = (40, 96, 36)  # x width, y length, z height
BOX = 0.9  # the car's bbox maps to [-BOX, BOX]
TRUNC = 3.0  # cells
Y_UP_TO_Z_UP = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- grid & features ----------------------------------------------------------------------------------------------
def grid_coords(dims=DIMS) -> np.ndarray:
    """Cell centres in normalised coordinates, (X, Y, Z, 3) in [-1, 1]."""
    ax = [-1 + (np.arange(n) + 0.5) * 2 / n for n in dims]
    return np.stack(np.meshgrid(*ax, indexing="ij"), -1)


def to_world(g: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (lo + hi) / 2 + g / BOX * (hi - lo) / 2


def to_norm(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (p - (lo + hi) / 2) / ((hi - lo) / 2) * BOX


def norm_to_index(g: np.ndarray, dims=DIMS) -> np.ndarray:
    """Normalised coordinates -> fractional cell indices (for map_coordinates)."""
    return (g + 1) / 2 * np.array(dims) - 0.5


def tsdf(occ: np.ndarray) -> np.ndarray:
    """Occupancy -> truncated signed distance in [-1, 1] (negative inside, zero level on the cell faces)."""
    if not occ.any():
        return np.ones(occ.shape, np.float32)
    out = distance_transform_edt(~occ) - 0.5
    ins = distance_transform_edt(occ) - 0.5
    d = np.where(occ, -ins, out)
    return np.clip(d / TRUNC, -1, 1).astype(np.float32)


def mirror(f: np.ndarray) -> np.ndarray:
    return f[::-1]  # x is the width axis; the grid is centred on the bbox


def flip_front(f: np.ndarray) -> np.ndarray:
    return f[:, ::-1]


# ---- ShapeNet solids ----------------------------------------------------------------------------------------------
def read_binvox(path: Path):
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
    vox = np.repeat(raw[::2], raw[1::2]).astype(bool).reshape(dims).transpose(0, 2, 1)  # binvox xzy -> xyz
    return vox, trans, scale


def solid_feature(path: Path) -> np.ndarray | None:
    """ShapeNet solid voxels -> bbox-normalised TSDF feature (z-up frame of ``space_carving.load_car``)."""
    vox, trans, scale = read_binvox(path)
    n = vox.shape[0]
    idx = np.argwhere(vox)
    if len(idx) < 1000:
        return None
    pn = trans + (idx + 0.5) / n * scale
    p = pn @ Y_UP_TO_Z_UP.T
    half = scale / n / 2
    lo, hi = p.min(0) - half, p.max(0) + half
    ext = hi - lo
    if not (0.25 < ext[0] / ext[1] < 0.8 and 0.15 < ext[2] / ext[1] < 0.8):  # not car-like (or not aligned)
        return None
    pw = to_world(grid_coords(), lo, hi).reshape(-1, 3)
    g = np.floor(((pw @ Y_UP_TO_Z_UP) - trans) / scale * n).astype(int)
    ok = np.all((g >= 0) & (g < n), axis=1)
    occ = np.zeros(len(pw), bool)
    occ[ok] = vox[g[ok, 0], g[ok, 1], g[ok, 2]]
    occ = occ.reshape(DIMS)
    if occ.mean() < 0.15:  # hollow solids (non-watertight source meshes)
        return None
    f = tsdf(occ)
    return 0.5 * (f + mirror(f))


def heldout(sid: str) -> bool:
    return int(sid[:2], 16) % 10 == 0  # ~10% test split, stable across runs


def build(root: Path, out: Path, k: int = 128):
    ids = sorted(p.name for p in root.iterdir())
    feats, keep, t0 = [], [], time.time()
    for i, sid in enumerate(ids):
        if heldout(sid):
            continue
        p = root / sid / "models" / "model_normalized.solid.binvox"
        if not p.is_file():
            continue
        f = solid_feature(p)
        if f is not None:
            feats.append(f.ravel())
            keep.append(sid)
        if i % 500 == 0:
            print(f"  {i}/{len(ids)} read, {len(keep)} kept, {time.time() - t0:.0f}s")
    X = torch.tensor(np.stack(feats), device=DEV)
    mean = X.mean(0)
    X -= mean
    G = X @ X.T  # Gram trick: N x N instead of D x D
    lam, V = torch.linalg.eigh(G)
    lam, V = lam.flip(0)[:k].clamp_min(1e-6), V.flip(1)[:, :k]
    U = (X.T @ V) / lam.sqrt()  # D x k, orthonormal columns
    sigma = (lam / (len(X) - 1)).sqrt()
    var = float(lam.sum() / torch.trace(G))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, mean=mean.cpu().numpy(), U=U.cpu().numpy().astype(np.float32),
                        sigma=sigma.cpu().numpy(), dims=np.array(DIMS), ids=np.array(keep),
                        heldout=np.array([s for s in ids if heldout(s)]))
    print(f"prior: {len(keep)} cars, {k} components explain {100 * var:.1f}% of the variance -> {out}")


# ---- fitting ------------------------------------------------------------------------------------------------------
class CarPrior:
    def __init__(self, path: Path):
        z = np.load(path)
        assert tuple(z["dims"]) == DIMS
        self.mean = torch.tensor(z["mean"], device=DEV)
        self.U = torch.tensor(z["U"], device=DEV)
        self.sigma = torch.tensor(z["sigma"], device=DEV)
        self.heldout = [str(s) for s in z["heldout"]]

    def fit(self, h: np.ndarray, alpha: float = 0.2, lam: float = 3.0, k: int | None = None, iters: int = 15):
        """IRLS for the asymmetric weighted least squares. Returns (fitted TSDF, loss)."""
        U = self.U[:, :k] if k else self.U
        sig = self.sigma[: U.shape[1]]
        ht = torch.tensor(h.ravel(), device=DEV)
        rhs0 = ht - self.mean
        w = torch.ones_like(ht)
        reg = lam * torch.diag(1 / sig**2)
        for _ in range(iters):
            A = U.T @ (U * w[:, None]) + reg
            z = torch.linalg.solve(A, U.T @ (w * rhs0))
            r = self.mean + U @ z - ht
            w = torch.where(r < 0, 1.0, alpha)  # r < 0: model more inside-ish than the hull says = bulges out
        loss = float((w * r * r).sum() + (z * z / sig**2).sum() * lam)
        return (self.mean + U @ z).clamp(-1, 1).reshape(DIMS).cpu().numpy(), loss


def occ_feature(occ: np.ndarray, lo_grid: np.ndarray, vs: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Voxel occupancy on a world grid (origin ``lo_grid``, voxel ``vs``) -> TSDF feature for the bbox [lo, hi]."""
    pw = to_world(grid_coords(), lo, hi).reshape(-1, 3)
    g = np.floor((pw - lo_grid) / vs).astype(int)
    ok = np.all((g >= 0) & (g < np.array(occ.shape)), axis=1)
    o = np.zeros(len(pw), bool)
    o[ok] = occ[g[ok, 0], g[ok, 1], g[ok, 2]]
    return tsdf(o.reshape(DIMS))


def symmetry_plane(occ: np.ndarray, search: float = 4.0, step: float = 0.25) -> float:
    """x (in voxel index units, continuous) of the left-right mirror plane that keeps the most of the volume: the
    hull bulges unevenly on the two sides, so its bbox centre is not the car's plane."""
    idx = np.argwhere(occ)
    c0 = (idx[:, 0].min() + idx[:, 0].max() + 1) / 2
    best, best_x = -1, c0
    for dx in np.arange(-search, search + 1e-9, step):
        x0 = c0 + dx
        mx = np.floor(2 * x0 - (idx[:, 0] + 0.5)).astype(int)
        ok = (mx >= 0) & (mx < occ.shape[0])
        keep = int(occ[mx[ok], idx[ok, 1], idx[ok, 2]].sum())
        if keep > best:
            best, best_x = keep, x0
    return float(best_x)


def mirror_world(f: np.ndarray, x0: float, cval: float = 1.0) -> np.ndarray:
    """Field mirrored about the plane x = x0 (voxel index units)."""
    ii = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float64) for n in f.shape], indexing="ij"))
    ii[0] = 2 * x0 - 1 - ii[0]  # cell i (centre i + 0.5) <-> 2 x0 - (i + 0.5) - 0.5
    return map_coordinates(f, ii, order=1, mode="constant", cval=cval).astype(np.float32)


def refine(occ: np.ndarray, lo_grid: np.ndarray, vs: float, prior: CarPrior | None, smooth: float = 0.5,
           alpha: float = 0.2, lam: float = 3.0) -> dict:
    """All fields are TSDFs on the *input* voxel grid (so no resolution is lost); only the prior fit runs on the
    normalised grid. -> {'fields': {name: field}, 'flipped', 'plane'}"""
    h = tsdf(occ)
    x0 = symmetry_plane(occ)
    sym = np.maximum(h, mirror_world(h, x0))  # intersection with the mirror image (still an outer bound)
    fields = {"carve": h, "sym": sym, "smooth": gaussian_filter(sym, smooth)}
    flipped = False
    if prior is not None:
        idx = np.argwhere(sym < 0)
        lo, hi = lo_grid + idx.min(0) * vs, lo_grid + (idx.max(0) + 1) * vs
        half = max(x0 - idx[:, 0].min(), idx[:, 0].max() + 1 - x0) * vs  # bbox centred on the mirror plane
        lo[0], hi[0] = lo_grid[0] + x0 * vs - half, lo_grid[0] + x0 * vs + half
        hf = occ_feature(sym < 0, lo_grid, vs, lo, hi)
        (fa, la), (fb, lb) = prior.fit(hf, alpha, lam), prior.fit(flip_front(hf), alpha, lam)
        flipped = lb < la
        fit = flip_front(fb) if flipped else fa
        centers = lo_grid + (np.argwhere(np.ones(occ.shape, bool)) + 0.5) * vs
        pw = sample_field(fit, centers, lo, hi).reshape(occ.shape).astype(np.float32)
        fields["prior"] = pw
        fields["fused"] = gaussian_filter(np.maximum(pw, sym), smooth)
    return {"fields": fields, "flipped": flipped, "plane": x0}


def sample_field(f: np.ndarray, pw: np.ndarray, lo, hi) -> np.ndarray:
    """Normalised-grid TSDF at world points (outside the grid = outside the car)."""
    c = norm_to_index(to_norm(pw, lo, hi)).T
    return map_coordinates(f, c, order=1, mode="constant", cval=1.0)


def field_mesh(f: np.ndarray, lo_grid: np.ndarray, vs: float):
    """Zero level set of a field on a world voxel grid -> (vertices in world, faces)."""
    from skimage.measure import marching_cubes

    g = np.pad(f, 1, constant_values=1.0)
    v, fa, _, _ = marching_cubes(g, 0.0)
    return lo_grid + (v - 1 + 0.5) * vs, fa


def main(argv=None):
    data = Path(__file__).resolve().parents[2] / "shape2brick" / "data"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--shapenet", default=str(data / "raw" / "ShapeNetCore" / "02958343"))
    b.add_argument("--out", default="outputs/car_prior/prior.npz")
    b.add_argument("--components", type=int, default=128)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(Path(a.shapenet), Path(a.out), a.components)


if __name__ == "__main__":
    main()
