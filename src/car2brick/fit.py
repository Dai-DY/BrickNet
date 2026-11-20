"""B2: part-by-part fitting baseline (non-learned).

Builds the model bottom-up, one part at a time, choosing at every step the library part, rotation and position that
best fits the target shape, under hard building rules. Unlike B0/B1 it can use bricks and slopes, so sloped hoods,
windscreens and tails are approximated by slanted faces instead of plate staircases.

Rules
-----
Hard constraints (a placement is never made if it breaks one):
  H1 grid        parts sit on the stud grid, studs up, rotated in 90-degree steps (no SNOT in this baseline);
  H2 no overlap  a part may not share any half-stud x half-stud x half-plate sub-cell with placed parts;
  H3 support     at least one of the part's bottom holes must sit on a free stud of a part directly below, unless
                 the target is empty under the whole footprint (the part is then a "base" piece, e.g. the floor
                 between the wheels, and must be tied in by parts placed on top of it);
  H4 symmetry    every placement is mirrored across the car's centre plane (the mirror is placed with it);
  H5 courses     brick-height parts (bricks, studded slopes) may only start on the shared courses k_ref, k_ref+3,
                 ... where k_ref is the body floor (first layer with at least half of the widest layer's area);
                 other heights are made up with plates. Without this, neighbouring regions end up one plate out of
                 step and can never be tied together by a part on top.

Score of a candidate (sum over the sub-cells the part would fill):
  +1           sub-cell inside the target,
  -0.5         sub-cell just outside the target (within ``band`` sub-cells: tolerance for the coarse target),
  -lambda_out  sub-cell clearly outside the target (default 3: overflow is worse than a missing cell),
  +bridge x volume  for each *extra* connected group the part joins (staggered joints; keeps the model in one
                   piece instead of separate columns),
and a candidate is only eligible when score >= min_fill x (part volume): parts must be mostly inside the shape.

Order: layers bottom to top; within a layer, best score first (bigger parts win ties because they collect more +1).
After fitting: plates whose whole top stays exposed become tiles of the same size (as in B1); parts covering window
cells take the glass colour (trans black), the rest the main colour.

Slope geometry: studded slopes (45 / 33 / 18 degree, 1 or 2 wide) use an analytic wedge when no meshes are
installed (``geometry.PartShapes``: high end = the stud row, 4 LDU lip), so they work without any mesh files.
Cheese slopes need real geometry (BrickNet colliders or the LDraw library) and are only enabled when it is
installed. Curved slopes are not used yet (irregular bottoms).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

import bricknet

from .descriptor import CarDescriptor
from .geometry import PartShapes
from .units import CAR_FROM_LD, LDU_PER_PLATE, LDU_PER_STUD

SUB = np.array([10.0, 10.0, 4.0])  # sub-cell size (LDU), GridSpec(sub=(2, 2, 2))

PLATES = ["3024", "3023b", "3623", "3710", "3666", "3460", "3022", "3021", "3020", "3795", "3034"]
BRICKS = ["3005", "3004", "3622", "3010", "3003", "3002", "3001"]
STUDDED_SLOPES = ["3040b", "3039", "4286", "3298", "60477", "30363"]  # 45 2x1/2x2, 33 3x1/3x2, 18 4x1/4x2
GEOMETRY_SLOPES = ["54200", "85984"]  # cheese slopes: need real meshes
TILES = {(1, 1): "3070b", (1, 2): "3069b", (1, 4): "2431", (1, 6): "6636", (1, 8): "4162", (2, 2): "3068b",
         (2, 4): "87079", (2, 6): "69729"}


@dataclass
class Template:
    part_id: int
    yaw: int  # quarter turns about the vertical axis
    kind: str  # plate / brick / slope / tile
    mask: np.ndarray  # bool (sx, sy, sz) sub-cells, origin at the footprint corner, bottom at z = 0
    holes: np.ndarray  # (n, 2) cell offsets of bottom holes
    studs: np.ndarray  # (m, 3) cell offsets (di, dj, plate level of the stud's top face)
    pose: np.ndarray  # 4x4: part-local LDraw -> grid frame, for a placement at cell (0, 0, 0)
    mirror: int = -1  # index of the mirror-image template (same part), -1 if none

    @property
    def cells(self) -> tuple[int, int]:
        return self.mask.shape[0] // 2, self.mask.shape[1] // 2


@dataclass
class Placed:
    template: int
    i: int
    j: int
    k: int
    color: int = 4


@dataclass
class FitResult:
    placements: list[Placed]
    templates: list[Template]
    stats: dict = field(default_factory=dict)

    def poses(self) -> list[np.ndarray]:
        out = []
        for p in self.placements:
            m = self.templates[p.template].pose.copy()
            m[:3, 3] += [p.i * LDU_PER_STUD, p.j * LDU_PER_STUD, p.k * LDU_PER_PLATE]
            out.append(m)
        return out

    @property
    def part_ids(self) -> list[int]:
        return [self.templates[p.template].part_id for p in self.placements]

    @property
    def colors(self) -> list[int]:
        return [p.color for p in self.placements]


def _ry(quarters: int) -> np.ndarray:
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][quarters % 4]
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _connectors(pid: int, sub: str) -> np.ndarray:
    v = bricknet.load_connectors()[pid].get((sub, None), ())
    return np.array([f[:3, 3] for f in v]) if v else np.zeros((0, 3))


def _bottom_holes(pid: int) -> np.ndarray:
    h = _connectors(pid, "hole")
    if not len(h):
        return h
    h = h[h[:, 1] >= h[:, 1].max() - 0.5]  # bottom face only
    ref = h[np.lexsort((h[:, 2], h[:, 0]))[0]]
    on_grid = (np.abs(((h[:, [0, 2]] - ref[[0, 2]]) / LDU_PER_STUD) % 1.0) < 1e-6).all(1)  # skip half-stud holes
    return h[on_grid]


def build_templates(shapes: PartShapes | None = None, use_slopes: bool = True) -> list[Template]:
    shapes = shapes or PartShapes()
    cat = bricknet.load_catalog()
    real_geometry = shapes.backend in ("cvx", "ldraw") or (
        shapes.backend == "auto" and (shapes.cvx_dir.is_dir() or shapes.ldraw is not None)
    )
    entries = [(s, "plate") for s in PLATES] + [(s, "brick") for s in BRICKS]
    if use_slopes:
        entries += [(s, "slope") for s in STUDDED_SLOPES]
        if real_geometry:
            entries += [(s, "slope") for s in GEOMETRY_SLOPES]
    templates: list[Template] = []
    seen = set()
    for stem, kind in entries:
        if stem not in cat.stem_to_id:
            continue
        pid = cat.stem_to_id[stem]
        local = shapes.local_interior(pid)  # studded slopes get an analytic wedge in the connector fallback
        holes, studs = _bottom_holes(pid), _connectors(pid, "stud")
        if not len(holes) or not len(local):
            continue
        for yaw in range(4):
            R = CAR_FROM_LD @ _ry(yaw)
            P, H, S = local @ R.T, holes @ R.T, studs @ R.T
            t = np.zeros(3)
            t[:2] = LDU_PER_STUD / 2 - H[:, :2].min(0)
            t[2] = -H[:, 2].min()
            P, H, S = P + t, H + t, S + t
            P = P[P[:, 2] > -1e-6]
            idx = np.floor(P / SUB + 1e-6).astype(int)
            if (idx < 0).any():
                continue
            mask = np.zeros(idx.max(0) + 1, dtype=bool)
            mask[tuple(idx.T)] = True
            # pad to whole cells
            shape = [mask.shape[0] + mask.shape[0] % 2, mask.shape[1] + mask.shape[1] % 2, mask.shape[2]]
            full = np.zeros(shape, dtype=bool)
            full[: mask.shape[0], : mask.shape[1], : mask.shape[2]] = mask
            hole_cells = np.unique(np.floor(H[:, :2] / LDU_PER_STUD).astype(int), axis=0)
            stud_cells = np.unique(np.column_stack([np.floor(S[:, :2] / LDU_PER_STUD),
                                                    np.round(S[:, 2] / LDU_PER_PLATE)]).astype(int), axis=0) \
                if len(S) else np.zeros((0, 3), int)
            key = (pid, full.tobytes(), full.shape, hole_cells.tobytes())
            if key in seen:  # e.g. a plate turned by 180 degrees
                continue
            seen.add(key)
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = R, t
            templates.append(Template(pid, yaw, kind, full, hole_cells, stud_cells, pose))
    # mirror pairs (reflection across the centre plane flips the j axis); wedge sampling is not exactly
    # symmetric, so match by overlap
    for ta in templates:
        best, best_iou = -1, 0.85
        for b, tb in enumerate(templates):
            if ta.part_id != tb.part_id or ta.mask.shape != tb.mask.shape:
                continue
            m = ta.mask[:, ::-1]
            iou = (m & tb.mask).sum() / max((m | tb.mask).sum(), 1)
            if iou > best_iou:
                best, best_iou = b, iou
        ta.mirror = best
    return templates


def _score_maps(weight, tp: Template, k: int, ni: int, nj: int) -> np.ndarray:
    """Score of placing ``tp`` with its corner at every cell (i, j) of layer k (vectorised over positions)."""
    sx, sy, sz = tp.mask.shape
    z0 = 2 * k
    out = np.zeros((ni, nj))
    for a, b, c in np.argwhere(tp.mask):
        out += weight[a : a + 2 * ni : 2, b : b + 2 * nj : 2, z0 + c]
    return out


class _Groups:
    """Union-find over placements (connected groups of the model so far)."""

    def __init__(self):
        self.parent: list[int] = []

    def add(self) -> int:
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def fit(
    desc: CarDescriptor,
    *,
    shapes: PartShapes | None = None,
    templates: list[Template] | None = None,
    lambda_out: float = 3.0,
    band: int = 1,
    min_fill: float = 0.6,
    bridge: float = 0.5,
    cap_tiles: bool = True,
    use_slopes: bool = True,
    max_parts: int = 1500,
) -> FitResult:
    from scipy import ndimage

    tps = templates or build_templates(shapes, use_slopes=use_slopes)
    target = desc.symmetrized().fine_target()
    nx, ny, nz = target.shape
    L, W, H = nx // 2, ny // 2, nz // 2
    near = ndimage.binary_dilation(target, iterations=band) if band else target
    weight = np.where(target, 1.0, np.where(near, -0.5, -lambda_out))
    # pad so that parts sticking out of the grid read as "clearly outside"
    pad = 32
    weight = np.pad(weight, ((0, pad), (0, pad), (0, pad)), constant_values=-lambda_out)
    occ = np.zeros_like(weight, dtype=bool)
    stud_owner = -np.ones((L + 16, W + 16, H + 16), dtype=int)  # placement index owning a free stud
    placed: list[Placed] = []
    groups = _Groups()
    ks = np.nonzero(target.any((0, 1)))[0]
    if not len(ks):
        return FitResult([], tps, {"parts": 0})
    k_min = int(ks.min()) // 2
    cell_solid = target.reshape(L, 2, W, 2, H, 2).any((1, 3, 5))
    area = cell_solid.sum((0, 1))
    k_ref = int(np.nonzero(area >= 0.5 * area.max())[0][0])

    def can_place(t: Template, i, j, k) -> bool:
        sx, sy, sz = t.mask.shape
        if i < 0 or j < 0:
            return False
        region = occ[2 * i : 2 * i + sx, 2 * j : 2 * j + sy, 2 * k : 2 * k + sz]
        return region.shape == t.mask.shape and not (region & t.mask).any()

    def is_base(t: Template, i, j, k) -> bool:
        fi, fj = t.cells
        return not cell_solid[i : i + fi, j : j + fj, k - 1].any()

    def owners_below(t: Template, i, j, k) -> set[int]:
        if k == k_min:
            return set()
        return {int(o) for di, dj in t.holes if (o := stud_owner[i + di, j + dj, k]) >= 0}

    def do_place(ti: int, i: int, j: int, k: int, below: set[int]) -> None:
        t = tps[ti]
        sx, sy, sz = t.mask.shape
        occ[2 * i : 2 * i + sx, 2 * j : 2 * j + sy, 2 * k : 2 * k + sz] |= t.mask
        idx = groups.add()
        for o in below:
            groups.union(idx, o)
        for di, dj in t.holes:  # studs underneath are now used
            if stud_owner[i + di, j + dj, k] >= 0:
                stud_owner[i + di, j + dj, k] = -1
        for di, dj, dk in t.studs:
            stud_owner[i + di, j + dj, k + dk] = idx
        placed.append(Placed(ti, i, j, k))

    for k in range(k_min, H):
        heap = []
        on_course = k >= k_ref and (k - k_ref) % 3 == 0
        for ti, t in enumerate(tps):
            if t.mask.shape[2] > 2 and not on_course:  # H5: brick-height parts only on shared courses
                continue
            fi, fj = t.cells
            ni, nj = L - fi + 1, W - fj + 1
            if ni <= 0 or nj <= 0 or 2 * k + t.mask.shape[2] > nz + pad:
                continue
            score = _score_maps(weight, t, k, ni, nj)
            vol = t.mask.sum()
            ok = score >= min_fill * vol
            if k > k_min:  # H3: a stud below, or nothing below in the target (base piece)
                sup = np.zeros((ni, nj), dtype=bool)
                for di, dj in t.holes:
                    sup |= stud_owner[di : di + ni, dj : dj + nj, k] >= 0
                below = cell_solid[:, :, k - 1]
                base = np.ones((ni, nj), dtype=bool)
                for di in range(fi):
                    for dj in range(fj):
                        base &= ~below[di : di + ni, dj : dj + nj]
                ok &= sup | base
            for i, j in np.argwhere(ok):
                # bridging bonus with the groups as they stand at the start of the layer
                roots = {groups.find(o) for o in owners_below(t, i, j, k)} if bridge and k > k_min else ()
                prio = score[i, j] + bridge * vol * max(len(roots) - 1, 0)
                heapq.heappush(heap, (-prio, ti, int(i), int(j)))
        while heap and len(placed) < max_parts:
            _, ti, i, j = heapq.heappop(heap)
            t = tps[ti]
            if not can_place(t, i, j, k):
                continue
            below = owners_below(t, i, j, k)
            if k > k_min and not below and not is_base(t, i, j, k):
                continue
            # H4: mirror partner
            mi, mj = -1, -1
            if t.mirror >= 0:
                fi, fj = t.cells
                mi, mj = i, W - j - fj
            same = t.mirror == ti and mj == j
            if t.mirror >= 0 and not same:
                tm = tps[t.mirror]
                if not can_place(tm, mi, mj, k) or (
                    k > k_min and not owners_below(tm, mi, mj, k) and not is_base(tm, mi, mj, k)
                ):
                    continue  # keep the model symmetric: skip placements whose mirror is impossible
            elif t.mirror < 0:
                continue
            do_place(ti, i, j, k, below)
            if not same:
                do_place(t.mirror, mi, mj, k, owners_below(tps[t.mirror], mi, mj, k))

    _finish(desc, placed, tps, occ, cap_tiles)
    n_groups = len({groups.find(a) for a in range(len(groups.parent))})
    covered = (occ[:nx, :ny, :nz] & target).sum() / max(target.sum(), 1)
    overflow = (occ[:nx, :ny, :nz] & ~near).sum() / max(occ.sum(), 1)
    stats = {"parts": len(placed), "groups": n_groups, "k_ref": k_ref, "target_covered": round(float(covered), 3),
             "overflow": round(float(overflow), 3),
             "kinds": {kd: sum(tps[p.template].kind == kd for p in placed)
                       for kd in ("plate", "brick", "slope", "tile")}}
    return FitResult(placed, tps, stats)


def _finish(desc: CarDescriptor, placed: list[Placed], tps: list[Template], occ: np.ndarray, cap_tiles: bool):
    cat = bricknet.load_catalog()
    main = cat.color_to_code.get(desc.colors[0], 4) if desc.colors else 4
    glass = cat.color_to_code.get("trans brown", 40)  # LDraw 40 = trans black
    labels = desc.symmetrized().labels
    tile_ids = {k: cat.stem_to_id[s] for k, s in TILES.items() if s in cat.stem_to_id}
    extra: dict[tuple, int] = {}
    for p in placed:
        t = tps[p.template]
        fi, fj = t.cells
        # colour: windows -> glass
        if labels is not None:
            lab = labels[p.i : p.i + fi, p.j : p.j + fj, p.k : p.k + t.mask.shape[2] // 2]
            p.color = glass if lab.size and (lab == 2).mean() > 0.5 else main
        else:
            p.color = main
        # B1-style tile capping for plates whose top stays exposed
        if cap_tiles and t.kind == "plate":
            top = occ[2 * p.i : 2 * (p.i + fi), 2 * p.j : 2 * (p.j + fj), 2 * (p.k + 1)]
            key = tuple(sorted((fi, fj)))
            if not top.any() and key in tile_ids:
                yaw = t.yaw
                ck = (key, yaw)
                if ck not in extra:
                    tile_tps = [x for x in build_tile_templates(tile_ids[key]) if x.cells == t.cells]
                    if not tile_tps:
                        continue
                    tps.append(tile_tps[0])
                    extra[ck] = len(tps) - 1
                p.template = extra[ck]


def build_tile_templates(pid: int) -> list[Template]:
    out = []
    holes = _bottom_holes(pid)
    for yaw in range(2):
        R = CAR_FROM_LD @ _ry(yaw)
        H = holes @ R.T
        t = np.zeros(3)
        t[:2] = LDU_PER_STUD / 2 - H[:, :2].min(0)
        t[2] = -H[:, 2].min()
        H = H + t
        cells = np.unique(np.floor(H[:, :2] / LDU_PER_STUD).astype(int), axis=0)
        fi, fj = cells.max(0) + 1
        mask = np.ones((2 * fi, 2 * fj, 2), dtype=bool)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = R, t
        out.append(Template(pid, yaw, "tile", mask, cells, np.zeros((0, 3), int), pose))
    return out


def fit_ldr(desc: CarDescriptor, **kw) -> tuple[str, FitResult]:
    from .evaluate import poses_to_ldr

    res = fit(desc, **kw)
    return poses_to_ldr(res.part_ids, res.colors, res.poses(), "car2brick B2 fit"), res
