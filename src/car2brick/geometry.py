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

from .units import LDU_PER_PLATE, LDU_PER_STUD


def bricknet_data_dir() -> Path:
    """Same lookup as ``bricknet.collision.data_dir`` without importing pybullet."""
    env = os.environ.get("BRICKNET_DATA")
    if env:
        return Path(env)
    import platformdirs

    return Path(platformdirs.user_data_dir("bricknet"))

# lattice spacing (LDU) for interior sampling; finer than the descriptor's supersample cell (10 x 10 x 4)
SAMPLE_STEP = np.array([4.0, 2.0, 4.0])  # LDraw-local x, y (vertical), z
LATTICE_OFFSET = np.array([0.0, 1.0, 0.0])
STUD_RADIUS, STUD_HEIGHT = 6.5, 4.5  # LDU, with a little slack around the 6 x 4 LDU stud


def _drop_studs(pts: np.ndarray, part_id: int) -> np.ndarray:
    """Remove the points inside the part's studs (cylinders on top of each stud connector)."""
    frames = bricknet.load_connectors().get(part_id, {}).get(("stud", None), ())
    if not len(frames) or not len(pts):
        return pts
    keep = np.ones(len(pts), dtype=bool)
    for f in frames:
        axis = f[:3, :3] @ np.array([0.0, -1.0, 0.0])  # a stud rises along its frame's -Y
        d = pts - f[:3, 3]
        h = d @ axis
        r = np.linalg.norm(d - h[:, None] * axis, axis=1)
        keep &= ~((h > 0) & (h <= STUD_HEIGHT) & (r <= STUD_RADIUS))
    return pts[keep] if keep.any() else pts


def _fill_under_holes(pts: np.ndarray, part_id: int) -> np.ndarray:
    """Lattice points (``SAMPLE_STEP`` multiples + ``LATTICE_OFFSET``) -> the same points plus, above every downward
    anti-stud ("hole" connector), the column from that hole up to the part's material (LDraw local: +y is down).

    The hollow underside of a brick / plate / tile / studded slope is exactly where the studs of the part below go,
    so in a built model it is volume. Using the labelled holes (not the geometry: BrickNet's colliders leave out
    thin walls, e.g. the long sides of a 2x4 brick) also keeps open undersides open: an arch has holes under its
    legs only."""
    if len(pts) < 2:
        return pts
    holes = [f for (sub, _), group in bricknet.load_connectors().get(part_id, {}).items() if sub == "hole"
             for f in group]
    holes = [f for f in holes if abs((f[:3, :3] @ np.array([0.0, -1.0, 0.0]))[1]) > 0.99]  # vertical ones only
    if not holes:
        return pts
    # global lattice indices (off-lattice centroid fallbacks of tiny hulls snap to the nearest node); the box also
    # spans the holes, which sit at the bottom face below the lowest sampled row
    gi = np.round((pts - LATTICE_OFFSET) / SAMPLE_STEP).astype(int)
    lo, hi = pts.min(0), pts.max(0)
    hp = np.array([f[:3, 3] for f in holes])
    reach = np.array([LDU_PER_STUD / 2, 0.0, LDU_PER_STUD / 2])
    g0 = np.minimum(gi.min(0), np.floor((hp - reach - LATTICE_OFFSET) / SAMPLE_STEP).min(0).astype(int))
    g1 = np.maximum(gi.max(0), np.ceil((hp + reach - LATTICE_OFFSET) / SAMPLE_STEP).max(0).astype(int))
    vol = np.zeros(g1 - g0 + 1, dtype=bool)
    vol[tuple((gi - g0).T)] = True
    cx, cy, cz = (LATTICE_OFFSET[a] + (g0[a] + np.arange(vol.shape[a])) * SAMPLE_STEP[a] for a in range(3))
    has = vol.any(1)
    top = np.where(has, vol.argmax(1), vol.shape[1])  # first filled row from the top, per (x, z) column
    rows = np.arange(vol.shape[1])[None, :, None]
    for hx, hy, hz in hp:
        # a stud cell, edges included (a lattice row may sit between two holes), but not past the part's sides
        ix = np.flatnonzero((np.abs(cx - hx) <= LDU_PER_STUD / 2 + 1e-6) & (cx >= lo[0]) & (cx <= hi[0]))
        iz = np.flatnonzero((np.abs(cz - hz) <= LDU_PER_STUD / 2 + 1e-6) & (cz >= lo[2]) & (cz <= hi[2]))
        if not len(ix) or not len(iz):
            continue
        col_top = top[np.ix_(ix, iz)][:, None, :]
        vol[np.ix_(ix, np.arange(vol.shape[1]), iz)] |= (rows >= col_top) & (cy[None, :, None] < hy)
    return LATTICE_OFFSET + (g0 + np.argwhere(vol)) * SAMPLE_STEP


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
        if self.backend == "auto":
            # studded slopes: the analytic wedge is their exact envelope, while the colliders' thin top step under the
            # stud row falls between lattice points (the part comes out a half plate short and lopsided)
            wedge = self._studded_slope_hull(part_id)
            if wedge is not None:
                return (wedge,)
        order = {"auto": ("cvx", "ldraw", "connector")}.get(self.backend, (self.backend,))
        for name in order:
            h = getattr(self, f"_{name}_hulls")(part_id)
            if h:
                return tuple(h)
        return tuple(self._connector_hulls(part_id))

    @lru_cache(maxsize=None)
    def local_interior(self, part_id: int) -> np.ndarray:
        """Lattice points inside the part's envelope (local LDU); the hull centroids if no lattice point is inside.

        Real bricks, plates, tiles and studded slopes are hollow underneath (walls and tubes only): a brick fills
        about 37 % of its box, below the descriptor's 55 % solid threshold, so models built from real geometry
        would look hollow while the same model from connector hulls - or a real car - looks solid. In a built model
        that cavity is taken by the studs of the part below, so it counts as volume: the columns above the part's
        downward anti-studs are filled up to its material (``_fill_under_holes``). Open undersides (arches) have no
        anti-studs there and stay open.

        For the same reason studs are left out: they sit inside the part above (counted there), and as a separate
        half-plate of volume on top they would block any part from being stacked in a sub-cell grid."""
        hulls = self.hulls(part_id)
        pts = np.concatenate([self._hull_lattice(v) for v in hulls])
        if not len(pts):  # every hull thinner than the lattice: fall back to their centroids
            return np.array([np.asarray(v).mean(0) for v in hulls])
        return _fill_under_holes(_drop_studs(pts, part_id), part_id)

    @staticmethod
    def _hull_lattice(verts: np.ndarray) -> np.ndarray:
        eq = _hull_equations(verts)
        lo, hi = verts.min(0), verts.max(0)
        # one lattice for all hulls of a part, so their samples line up; its points sit strictly inside stud cells
        # (x, z: multiples of 4 -> -8..8 across a 20 LDU stud) and plate layers (y: odd -> 1, 3, 5, 7 in 0..8)
        axes = [np.arange(np.floor(lo[a] / SAMPLE_STEP[a]) * SAMPLE_STEP[a] + LATTICE_OFFSET[a], hi[a],
                          SAMPLE_STEP[a]) for a in range(3)]
        axes = [ax[(ax > lo[a]) & (ax < hi[a])] for a, ax in enumerate(axes)]
        if eq is None or any(len(a) == 0 for a in axes):
            return np.empty((0, 3))  # thinner than the lattice (walls, tube rims): covered by its neighbours
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        inside = (grid @ eq[:, :3].T + eq[:, 3] <= 1e-6).all(1)
        return grid[inside]

    def interior_points(self, part_ids, transforms) -> np.ndarray:
        """World-space interior points of a placed model (same frame as ``transforms``)."""
        chunks = []
        for pid, mat in zip(part_ids, transforms):
            loc = self.local_interior(int(pid))
            chunks.append(loc @ np.asarray(mat)[:3, :3].T + np.asarray(mat)[:3, 3])
        return np.concatenate(chunks) if chunks else np.empty((0, 3))

