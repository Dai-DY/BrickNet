"""ShapeNet cars: lookup by name, OBJ + MTL reading, conversion to the car frame (x forward, y left, z up)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parents[3] / "shape2brick" / "data"
SHAPENET_ROOT = _DATA / "raw" / "ShapeNetCore" / "02958343"
SHAPENET_INDEX = _DATA / "cars" / "index" / "shapenet.csv"

# ShapeNet normalised models: y up, front towards -z, x towards the car's right  ->  car frame
SN_TO_CAR = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def names(index: Path = SHAPENET_INDEX) -> dict[str, str]:
    if not Path(index).is_file():
        return {}
    return {r["id"]: r.get("name", "") for r in csv.DictReader(open(index)) if r.get("v2_has_mesh") == "1"}


def search(keyword: str, root: Path = SHAPENET_ROOT) -> list[tuple[str, str]]:
    kw = keyword.lower()
    return [(i, n) for i, n in names().items() if kw in n.lower() and (root / i / "models").is_dir()]


def _read_mtl(path: Path) -> dict[str, tuple[np.ndarray, float]]:
    out, name = {}, None
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "newmtl":
            name = " ".join(t[1:])
            out[name] = (np.array([0.6, 0.6, 0.6]), 1.0)
        elif name and t[0] == "Kd" and len(t) >= 4:
            out[name] = (np.array([float(v) for v in t[1:4]]), out[name][1])
        elif name and t[0] in ("d", "Tr") and len(t) >= 2:
            d = float(t[1])
            out[name] = (out[name][0], d if t[0] == "d" else 1.0 - d)
    return out


def read_obj(path: Path):
    """-> vertices (V, 3), triangles (F, 3), per-face diffuse colour (F, 3) in [0, 1], per-face opacity (F,)."""
    mats: dict[str, tuple[np.ndarray, float]] = {}
    verts, faces, fmat = [], [], []
    cur = None
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("v "):
            verts.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            for k in range(1, len(idx) - 1):
                faces.append([idx[0], idx[k], idx[k + 1]])
                fmat.append(cur)
        elif line.startswith("usemtl"):
            cur = line.split(maxsplit=1)[1].strip() if len(line.split()) > 1 else None
        elif line.startswith("mtllib"):
            mats.update(_read_mtl(Path(path).parent / line.split(maxsplit=1)[1].strip()))
    grey = (np.array([0.6, 0.6, 0.6]), 1.0)
    rgb = np.array([mats.get(m, grey)[0] for m in fmat], dtype=np.float64).reshape(-1, 3)
    alpha = np.array([mats.get(m, grey)[1] for m in fmat], dtype=np.float64)
    return np.asarray(verts, np.float64), np.asarray(faces, np.int64), np.clip(rgb, 0, 1), alpha


def dedupe_faces(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """ShapeNet surfaces are often modelled twice with opposite winding (paint outside, black inside): keep one face
    per coincident group, the one facing away from the model centre. Boolean mask over faces."""
    tri = v[f]
    key = np.round(np.sort(tri, axis=1).reshape(len(f), -1), 5)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = (n * (tri.mean(1) - v.mean(0))).sum(1)
    order = np.lexsort((-out, inv))
    keep = np.zeros(len(f), bool)
    first = np.ones(len(order), bool)
    first[1:] = inv[order[1:]] != inv[order[:-1]]
    keep[order[first]] = True
    return keep


def load_shapenet(sid: str, root: Path = SHAPENET_ROOT, flip: bool = False):
    """-> (vertices in the car frame, unscaled, faces, face rgb). ``flip``: the model faces backwards (rare in
    ShapeNet; turn it 180 degrees about z)."""
    v, f, rgb, alpha = read_obj(Path(root) / sid / "models" / "model_normalized.obj")
    keep = dedupe_faces(v, f)
    f, rgb, alpha = f[keep], rgb[keep], alpha[keep]
    v = v @ SN_TO_CAR.T
    if flip:
        v[:, :2] *= -1
    glass = alpha < 0.9  # see-through glass: dark tinted and opaque (closes the cabin, looks like glass in photos)
    rgb = np.where(glass[:, None], 0.3 * rgb + 0.05, rgb)
    return v, f, rgb
