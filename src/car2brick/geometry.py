"""Part geometry for voxelizing LEGO models.

Every part is approximated by one or more convex hulls in its LDraw-local frame (LDU, -Y up). Three backends,
tried in order by ``PartShapes(backend="auto")``:

* ``cvx``       BrickNet's convex-decomposed colliders, ``<BRICKNET_DATA>/cvx/<stem>.cvx.obj``
                (``python -m bricknet fetch-meshes``). Best fidelity.
* ``ldraw``     the LDraw part library (``parts/``, ``p/``); one hull of all of a part's vertices.
* ``connector`` always available: the hull of the part's connector positions from BrickNet's bundled labels,
                widened by half a stud around studs/holes and padded to one plate thick when flat. Coarse, but
                good enough for stud-resolution occupancy of bricks, plates, tiles and slopes.

``interior_points`` returns a lattice of points inside the hulls, which the descriptor bins into its grid.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

import bricknet

from .units import LDU_PER_PLATE


def bricknet_data_dir() -> Path:
    """Same lookup as ``bricknet.collision.data_dir`` without importing pybullet."""
    env = os.environ.get("BRICKNET_DATA")
    if env:
        return Path(env)
    import platformdirs

    return Path(platformdirs.user_data_dir("bricknet"))

# lattice spacing (LDU) for interior sampling; finer than the descriptor's supersample cell (10 x 10 x 4)
SAMPLE_STEP = np.array([4.0, 2.0, 4.0])  # LDraw-local x, y (vertical), z


def _hull_equations(verts: np.ndarray) -> np.ndarray | None:
    try:
        return ConvexHull(verts).equations
    except (QhullError, ValueError):
        return None


def _parse_obj_groups(path: Path) -> list[np.ndarray]:
    """Vertex arrays per ``o`` group of an OBJ (faces are irrelevant for convex hulls)."""
    groups: list[list[list[float]]] = [[]]
    for line in path.read_text().splitlines():
        if line.startswith("o ") and groups[-1]:
            groups.append([])
        elif line.startswith("v "):
            groups[-1].append([float(v) for v in line.split()[1:4]])
    return [np.array(g) for g in groups if len(g) >= 4]


class LDrawLibrary:
    """Minimal recursive LDraw .dat reader: collects triangle/quad/line vertices through type-1 references."""

    SUBDIRS = ("parts", "p", "parts/s", "p/48", "p/8")

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        if not (self.root / "parts").is_dir():
            raise FileNotFoundError(f"not an LDraw library (no parts/): {self.root}")

    def _find(self, name: str) -> Path | None:
        name = name.replace("\\", "/").lower()
        for sub in self.SUBDIRS:
            p = self.root / sub / name
            if p.is_file():
                return p
        return None

    @lru_cache(maxsize=None)
    def vertices(self, name: str) -> np.ndarray:
        path = self._find(name)
        if path is None:
            return np.empty((0, 3))
        pts = []
        for line in path.read_text(errors="replace").splitlines():
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "1" and len(tok) >= 15:
                v = np.array(tok[2:14], dtype=np.float64)
                rot, t = v[3:].reshape(3, 3), v[:3]
                sub = self.vertices(" ".join(tok[14:]))
                if len(sub):
                    pts.append(sub @ rot.T + t)
            elif tok[0] in ("2", "3", "4", "5"):
                n = {"2": 2, "3": 3, "4": 4, "5": 2}[tok[0]]
                pts.append(np.array(tok[2 : 2 + 3 * n], dtype=np.float64).reshape(n, 3))
        return np.concatenate(pts) if pts else np.empty((0, 3))


class PartShapes:
    def __init__(self, backend: str = "auto", ldraw_dir: str | None = None, cvx_dir: str | None = None):
        if backend not in ("auto", "cvx", "ldraw", "connector"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self.catalog = bricknet.load_catalog()
        self.cvx_dir = Path(cvx_dir) if cvx_dir else bricknet_data_dir() / "cvx"
        ldraw_dir = ldraw_dir or os.environ.get("LDRAW_DIR")
        self.ldraw = LDrawLibrary(ldraw_dir) if ldraw_dir and Path(ldraw_dir).is_dir() else None

    # ---- backends -------------------------------------------------------------------------------------------
    def _cvx_hulls(self, part_id: int) -> list[np.ndarray] | None:
        path = self.cvx_dir / f"{self.catalog.id_to_stem[part_id]}.cvx.obj"
        return _parse_obj_groups(path) if path.is_file() else None

    def _ldraw_hulls(self, part_id: int) -> list[np.ndarray] | None:
        if self.ldraw is None:
            return None
        v = self.ldraw.vertices(self.catalog.id_to_stem[part_id] + ".dat")
        return [v] if len(v) >= 4 else None

    def _studded_slope_hull(self, part_id: int) -> np.ndarray | None:
        """Wedge for slopes whose studs sit on part of the footprint (45/33/18 degree bricks): full height under
        the stud row, ramping down to a 4 LDU lip at the far edge. None for anything else."""
        frames = bricknet.load_connectors().get(part_id, {})
        studs = np.array([f[:3, 3] for f in frames.get(("stud", None), ())]).reshape(-1, 3)
        holes = np.array([f[:3, 3] for f in frames.get(("hole", None), ())]).reshape(-1, 3)
        if not len(studs) or len(holes) <= len(studs) or np.ptp(studs[:, 1]) > 0.5:
            return None
        holes = holes[holes[:, 1] >= holes[:, 1].max() - 0.5]
        bottom, top = holes[0, 1], studs[0, 1]
        if bottom - top < 16:  # only brick-height slopes
            return None
        for a, b in ((2, 0), (0, 2)):  # a: slope axis, b: across
            if np.ptp(studs[:, a]) > 0.5:
                continue
            lo_side = holes[:, a] < studs[:, a].min() - 0.5
            hi_side = holes[:, a] > studs[:, a].max() + 0.5
            if lo_side.any() == hi_side.any():
                continue
            far = holes[:, a].min() - 10.0 if lo_side.any() else holes[:, a].max() + 10.0
            near = studs[0, a] + (-10.0 if lo_side.any() else 10.0)
            back = studs[0, a] + (10.0 if lo_side.any() else -10.0)
            w0, w1 = holes[:, b].min() - 10.0, holes[:, b].max() + 10.0
            pts = []
            for w in (w0, w1):
                for (pa, py) in ((back, top), (near, top), (back, bottom), (far, bottom), (far, bottom - 4.0)):
                    p = np.zeros(3)
                    p[a], p[b], p[1] = pa, w, py
                    pts.append(p)
            return np.array(pts)
        return None

    def _connector_hulls(self, part_id: int) -> list[np.ndarray]:
        wedge = self._studded_slope_hull(part_id)
        if wedge is not None:
            return [wedge]
        frames = bricknet.load_connectors().get(part_id, {})
        pts = []
        for (sub, _pol), group in frames.items():
            for f in group:
                p = f[:3, 3]
                if sub in ("stud", "hole", "tube", "open", "post"):
                    # widen to the stud cell around the connector, in the connector's own plane
                    ax, az = f[:3, 0] * 10.0, f[:3, 2] * 10.0
                    pts += [p + ax + az, p + ax - az, p - ax + az, p - ax - az]
                else:
                    pts.append(p)
        if not pts:
            return [np.array([[-10, 0, -10], [10, 0, -10], [-10, 0, 10], [10, 0, 10], [0, 8, 0]], dtype=float)]
        pts = np.array(pts)
        for axis in range(3):  # pad flat sets to one plate thickness so the hull has volume
            if np.ptp(pts[:, axis]) < 4.0:
                shift = np.zeros(3)
                shift[axis] = -LDU_PER_PLATE if axis == 1 else 4.0
                pts = np.concatenate([pts, pts + shift])
        return [pts]

    @lru_cache(maxsize=None)
    def hulls(self, part_id: int) -> tuple[np.ndarray, ...]:
        order = {"auto": ("cvx", "ldraw", "connector")}.get(self.backend, (self.backend,))
        for name in order:
            h = getattr(self, f"_{name}_hulls")(part_id)
            if h:
                return tuple(h)
        return tuple(self._connector_hulls(part_id))

    @lru_cache(maxsize=None)
    def local_interior(self, part_id: int) -> np.ndarray:
        """Lattice points inside the part (local LDU). Always returns at least the hull centroids."""
        out = []
        for verts in self.hulls(part_id):
            eq = _hull_equations(verts)
            lo, hi = verts.min(0), verts.max(0)
            axes = [np.arange(lo[a] + SAMPLE_STEP[a] / 2, hi[a], SAMPLE_STEP[a]) for a in range(3)]
            if eq is None or any(len(a) == 0 for a in axes):
                out.append(verts.mean(0, keepdims=True))
                continue
            grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
            inside = (grid @ eq[:, :3].T + eq[:, 3] <= 1e-6).all(1)
            out.append(grid[inside] if inside.any() else verts.mean(0, keepdims=True))
        return np.concatenate(out)

    def interior_points(self, part_ids, transforms) -> np.ndarray:
        """World-space interior points of a placed model (same frame as ``transforms``)."""
        chunks = []
        for pid, mat in zip(part_ids, transforms):
            loc = self.local_interior(int(pid))
            chunks.append(loc @ np.asarray(mat)[:3, :3].T + np.asarray(mat)[:3, 3])
        return np.concatenate(chunks) if chunks else np.empty((0, 3))

