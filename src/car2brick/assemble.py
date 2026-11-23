"""B3: module-wise kNN assembly baseline.

Every library car is cut into modules by rules on its parts (``label_parts``):

    chassis   non-wheel parts at or below the floor (first layer with half of the widest layer's area)
    wheels_a  wheel parts (tyre / rim / hub) of the axle at smaller x, with the small low parts joined to them
              (holders, axles);  wheels_b  the other axle
    glass     transparent parts with a large interior (windows; small ones are lights and stay with the body)
    wing      parts above a gap in their column (an open slot under them) in the outer 35 % of the length
    end_a     the rest of the body before the first axle (+1.5 studs);  end_b  after the second axle
    cabin     the body between them

"a" / "b" are the two ends by x: front and back cannot be told apart from shape, so every library car is also
indexed driving the other way (rotated 180 degrees about the vertical axis), which swaps them.

The same regions are marked on the target descriptor (``Frame`` / ``regions``: its profile, wheel positions, floor
and colour classes). Each module is retrieved on its own: among all library cars of the target's width (both
directions), score = IoU between the module's cells and the target region x (0.7 + 0.3 x fraction of the overlap
whose colour class agrees), after aligning the module on its anchor (wheel centre for wheels / ends, wheelbase centre
for chassis / cabin / glass, the nearer car end for the wing; body modules also on the floor height) and a shift
search of +-1 cell in x and z.

Assembly in the order chassis, cabin, ends, wheels, glass, wing: for each module the top-k candidates are tried at
their anchor and +-1 cell in x and z; a part is dropped when it would overlap parts already placed (sub-cell
collision grid), lies mostly outside the target's allowed volume, or - modules of different cars meet on the stud
grid but are not always tied together - is not joined to the model built so far (BrickNet's connection inference
on the LDraw parts). The placement with the best shape score x kept-volume fraction wins, so the result is one
connected piece; a module with no joinable candidate is left out. Body parts are then recoloured with the target's
colour class of the cells they cover.
"""

from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

import bricknet

from .colors import CLASS_NAMES, CLASSES, GLASS, LIGHT, class_of_name, code
from .descriptor import CarDescriptor, column_filled
from .evaluate import poses_to_ldr
from .geometry import PartShapes
from .lego import color_name, is_wheel
from .units import CAR_FROM_LD, LDU_PER_PLATE, LDU_PER_STUD, homogeneous

MODULES = ("chassis", "wheels_a", "wheels_b", "cabin", "end_a", "end_b", "glass", "wing")
MODULE_NAMES = {"base": "整车底", "chassis": "底盘", "wheels_a": "轮胎A", "wheels_b": "轮胎B", "cabin": "座舱",
                "end_a": "车头/尾A", "end_b": "车头/尾B", "glass": "玻璃", "wing": "尾翼"}
MODULE_COLORS = {"base": 15, "chassis": 72, "wheels_a": 0, "wheels_b": 0, "cabin": 1, "end_a": 4, "end_b": 14,
                 "glass": 27, "wing": 26}  # LDraw codes for the "coloured by module" view
CELL = np.array([LDU_PER_STUD, LDU_PER_STUD, LDU_PER_PLATE])
SUB = 2  # collision grid: sub-cells per cell along each axis
END_ZONE = 0.35  # wing search zone: this fraction of the length at either end
MIN_CELL_POINTS = 12  # interior lattice points for a part to count as filling a cell (~100 in a full cell)
COLOR_WEIGHT = 0.3
HOLDER_MAX_CELLS = 8  # a wheel holder / axle joined to a wheel is at most this many cells
ORDER = ("chassis", "cabin", "end_a", "end_b", "wheels_a", "wheels_b", "glass", "wing")  # assembly order


# ---- car frame and regions ----------------------------------------------------------------------------------------
@dataclass
class Frame:
    """Anchors of one car in its grid (cells): wheel centres (continuous x, sorted), floor layer, solid extent."""

    wheels: tuple[float, float]
    floor: int
    x0: int
    x1: int
    y0: int
    y1: int

    @property
    def mid(self) -> float:
        return 0.5 * (self.wheels[0] + self.wheels[1])

    def flipped(self, L: int, W: int) -> "Frame":
        return Frame((L - self.wheels[1], L - self.wheels[0]), self.floor, L - self.x1, L - self.x0, W - self.y1,
                     W - self.y0)


def car_frame(P: np.ndarray, wheels=()) -> Frame | None:
    """Frame of a profile (bool L, W, H); without two wheel positions they are put at 20 % / 80 % of the length."""
    idx = np.argwhere(P)
    if not len(idx):
        return None
    (x0, y0, _), (x1, y1, _) = idx.min(0), idx.max(0) + 1
    area = P.sum((0, 1))
    floor = int(np.argmax(area >= 0.5 * area.max()))
    if len(wheels) == 2:
        w = tuple(sorted(float(v) for v in wheels))
    else:
        w = (x0 + 0.2 * (x1 - x0), x0 + 0.8 * (x1 - x0))
    return Frame(w, floor, int(x0), int(x1), int(y0), int(y1))


def upper_cells(P: np.ndarray) -> np.ndarray:
    """Solid cells above a gap in their column (everything over the column's lowest run)."""
    first = np.argmax(P, axis=2)
    has = P.any(2)
    z = np.arange(P.shape[2])[None, None, :]
    after = z > first[..., None]
    gap = after & ~P  # empty cells above the first solid one
    # first empty cell above the lowest run; solid cells beyond it are above a gap
    g = np.where(gap.any(2), np.argmax(gap, axis=2), P.shape[2])
    return P & (z > g[..., None]) & has[..., None]


def regions(P: np.ndarray, fr: Frame, C: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Cell masks of the modules on a profile ``P`` (and colour classes ``C`` for glass)."""
    L, W, H = P.shape
    x = np.arange(L)[:, None, None] + 0.5
    y = np.arange(W)[None, :, None]
    z = np.arange(H)[None, None, :]
    wa, wb = fr.wheels
    low = z <= fr.floor
    n = fr.x1 - fr.x0
    ends = (x < fr.x0 + END_ZONE * n) | (x >= fr.x1 - END_ZONE * n)
    wing = upper_cells(P) & ends & ~low
    glass = P & (C == GLASS) if C is not None else np.zeros_like(P)
    body = P & ~low & ~wing & ~glass
    outer = (y < fr.y0 + 2) | (y >= fr.y1 - 2)
    near_wheel = z <= fr.floor + 2
    return {
        "chassis": P & low,
        "wheels_a": P & near_wheel & outer & (np.abs(x - wa) <= 2.5),
        "wheels_b": P & near_wheel & outer & (np.abs(x - wb) <= 2.5),
        "cabin": body & (x >= wa + 1.5) & (x <= wb - 1.5),
        "end_a": body & (x < wa + 1.5),
        "end_b": body & (x > wb - 1.5),
        "glass": glass,
        "wing": wing,
    }


def anchor(module: str, tgt: Frame, src: Frame) -> tuple[int, int]:
    """(dx, dz) in cells that puts a module of ``src`` onto the matching place of ``tgt``."""
    if module in ("wheels_a", "end_a"):
        dx = tgt.wheels[0] - src.wheels[0]
    elif module in ("wheels_b", "end_b"):
        dx = tgt.wheels[1] - src.wheels[1]
    else:  # chassis, cabin, glass (the wing's x is set per candidate by ``_wing_anchor``)
        dx = tgt.mid - src.mid
    dz = 0 if module in ("chassis", "wheels_a", "wheels_b") else tgt.floor - src.floor
    return int(round(dx)), int(dz)


def _wing_anchor(cells: np.ndarray, tgt: Frame, src: Frame) -> int:
    cx = cells[:, 0].mean() + 0.5
    near_b = abs(cx - src.x1) < abs(cx - src.x0)
    return int(round(tgt.x1 - src.x1)) if near_b else int(round(tgt.x0 - src.x0))


# ---- library index ------------------------------------------------------------------------------------------------
def part_cells(points: np.ndarray, shape) -> np.ndarray:
    """Cells (k, 3) a part fills, from its interior points in the grid frame."""
    idx = np.floor(points / CELL).astype(int)
    idx = idx[((idx >= 0) & (idx < np.array(shape))).all(1)]
    if not len(idx):
        return np.zeros((0, 3), np.int16)
    cells, cnt = np.unique(idx, axis=0, return_counts=True)
    keep = cnt >= MIN_CELL_POINTS
    if not keep.any():
        keep = cnt == cnt.max()
    return cells[keep].astype(np.int16)


def label_parts(P: np.ndarray, fr: Frame, cells: list[np.ndarray], wheel: np.ndarray, glass: np.ndarray) -> np.ndarray:
    """Module index (into MODULES) of every part; see the module docstring for the rules."""
    reg = regions(P, fr)
    wing = reg["wing"]
    labels = np.zeros(len(cells), dtype=np.int8)
    for i, c in enumerate(cells):
        cx = c[:, 0].mean() + 0.5 if len(c) else fr.mid
        if wheel[i]:
            m = "wheels_a" if cx < fr.mid else "wheels_b"
        elif glass[i]:
            m = "glass"
        elif len(c) and wing[tuple(c.T)].mean() >= 0.6:
            m = "wing"
        elif not len(c) or c[:, 2].max() <= fr.floor:
            m = "chassis"
        elif cx < fr.wheels[0] + 1.5:
            m = "end_a"
        elif cx > fr.wheels[1] - 1.5:
            m = "end_b"
        else:
            m = "cabin"
        labels[i] = MODULES.index(m)
    return labels


GRAPH_FIXED = ("wheels_a", "wheels_b", "glass")  # several separate pieces by nature (two wheels, two windows)


def refine_labels(lab: np.ndarray, edges: np.ndarray, cells: list[np.ndarray], P: np.ndarray, fr: Frame,
                  iters: int = 10, reach: int = 2) -> np.ndarray:
    """Make the position labels follow the connection graph, so that a module is a sub-assembly of the car rather
    than a slab cut out of it:

    * a part joined more to another module than to its own moves there (label propagation; only when the part
      lies within ``reach`` cells of that module's region, so the cabin does not swallow the car);
    * a piece of a module that is not connected to the module's largest piece joins the neighbouring module it
      has most connections to.

    Wheels and glass keep their labels (two wheels / several windows are separate pieces by nature)."""
    n = len(lab)
    lab = lab.copy()
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        if a != b:
            adj[a].append(int(b))
            adj[b].append(int(a))
    fixed_ids = [MODULES.index(m) for m in GRAPH_FIXED]
    fixed = np.isin(lab, fixed_ids)
    reg = regions(P, fr)
    near = {MODULES.index(m): ndimage.binary_dilation(r, iterations=reach) for m, r in reg.items()}

    def fits(p: int, m: int) -> bool:
        c = cells[p]
        return not len(c) or near[m][tuple(c.T)].mean() >= 0.5

    for _ in range(iters):
        changed = 0
        for p in range(n):
            if fixed[p] or not adj[p]:
                continue
            cnt = Counter(int(lab[q]) for q in adj[p] if not fixed[q])
            if not cnt:
                continue
            b, nb = cnt.most_common(1)[0]
            if b != lab[p] and nb > cnt.get(int(lab[p]), 0) and fits(p, b):
                lab[p] = b
                changed += 1
        for m in range(len(MODULES)):
            if m in fixed_ids:
                continue
            members = set(np.nonzero(lab == m)[0].tolist())
            pieces, seen = [], set()
            for s0 in members:
                if s0 in seen:
                    continue
                stack, piece = [s0], []
                seen.add(s0)
                while stack:
                    u = stack.pop()
                    piece.append(u)
                    for v in adj[u]:
                        if v in members and v not in seen:
                            seen.add(v)
                            stack.append(v)
                pieces.append(piece)
            if len(pieces) < 2:
                continue
            pieces.sort(key=len, reverse=True)
            for piece in pieces[1:]:
                cnt = Counter(int(lab[v]) for u in piece for v in adj[u] if lab[v] != m and not fixed[v])
                if cnt:
                    lab[piece] = cnt.most_common(1)[0][0]
                    changed += 1
        if not changed:
            break
    return lab


@dataclass
class ModCar:
    source: str
    width: int
    shape: tuple[int, int, int]
    part_ids: np.ndarray  # (n,) int32
    colors: np.ndarray  # (n,) int32 LDraw codes
    poses: np.ndarray  # (n, 4, 4) float32, grid frame
    part_cls: np.ndarray  # (n,) int8 colour class
    frames: list  # [Frame, Frame]: as stored / driving the other way
    labels: list  # [(n,) int8, (n,) int8]
    mods: list  # [{module: (cells (k, 3) int16, classes (k,) int8)}, ...] per direction
    solid: list = field(default_factory=list)  # [column-filled profile, flipped]
    pcells: np.ndarray | None = None  # (m, 3) int16: cells of every part, concatenated (as stored, not flipped)
    poff: np.ndarray | None = None  # (n + 1,) offsets into pcells

    def cells_of(self, j: int, flip: bool) -> np.ndarray:
        c = self.pcells[self.poff[j] : self.poff[j + 1]]
        if flip:
            c = np.stack([self.shape[0] - 1 - c[:, 0], self.shape[1] - 1 - c[:, 1], c[:, 2]], 1)
        return c.astype(np.int32)


def flip_pose(L: int, W: int) -> np.ndarray:
    f = np.diag([-1.0, -1.0, 1.0, 1.0])
    f[0, 3], f[1, 3] = L * LDU_PER_STUD, W * LDU_PER_STUD
    return f


def index_car(source: str, desc: CarDescriptor, ldr_text: str, shapes: PartShapes) -> ModCar | None:
    g = bricknet.parse_ldr(ldr_text)
    if not g.part_ids:
        return None
    shape = desc.spec.shape
    L, W, _ = shape
    poses = np.array([homogeneous(CAR_FROM_LD) @ np.asarray(t) for t in g.transforms])
    pids = np.array(g.part_ids, np.int32)
    cols = np.array(g.colors, np.int32)
    names = [color_name(c) for c in cols]
    pts = [shapes.interior_points([p], [t]) for p, t in zip(pids, poses)]
    cells = [part_cells(p, shape) for p in pts]
    wheel = np.array([is_wheel(int(p)) for p in pids])
    glass = np.array(["trans" in n and len(p) > 40 for n, p in zip(names, pts)])
    cls = np.array([LIGHT if "trans" in n and not gl else class_of_name(n) for n, gl in zip(names, glass)], np.int8)
    P = desc.profile()
    fr = car_frame(P, desc.wheels)
    if fr is None:
        return None
    # wheel holders / axles (small, low parts joined to a wheel) travel with the wheels, so a wheel set brings what
    # holds it instead of arriving as loose tyres
    holder = np.zeros(len(pids), bool)
    for ea, eb in zip(g.edges["a"], g.edges["b"]):  # parse_ldr: structured array of contacts (a, b, ...)
        for a, b in ((int(ea), int(eb)), (int(eb), int(ea))):
            if wheel[a] and not wheel[b] and len(cells[b]) and len(cells[b]) <= HOLDER_MAX_CELLS \
                    and cells[b][:, 2].max() <= fr.floor + 2:
                holder[b] = True
    wheel = wheel | holder
    edges = np.stack([np.asarray(g.edges["a"], int), np.asarray(g.edges["b"], int)], 1) if len(g.edges) \
        else np.zeros((0, 2), int)
    frames, labels, mods, solids = [], [], [], []
    for flip in (False, True):
        f = fr.flipped(L, W) if flip else fr
        Pf = P[::-1, ::-1] if flip else P
        cf = [np.stack([L - 1 - c[:, 0], W - 1 - c[:, 1], c[:, 2]], 1).astype(np.int16) if flip else c
              for c in cells]
        lab = refine_labels(label_parts(Pf, f, cf, wheel, glass), edges, cf, Pf, f)
        md = {}
        for mi, m in enumerate(MODULES):
            sel = np.nonzero(lab == mi)[0]
            if not len(sel):
                continue
            cc = np.concatenate([cf[i] for i in sel])
            kk = np.concatenate([np.full(len(cf[i]), cls[i], np.int8) for i in sel])
            u, first = np.unique(cc, axis=0, return_index=True)
            md[m] = (u.astype(np.int16), kk[first])
        frames.append(f)
        labels.append(lab)
        mods.append(md)
        solids.append(column_filled(Pf))
    off = np.concatenate([[0], np.cumsum([len(c) for c in cells])]).astype(np.int64)
    pc = np.concatenate(cells).astype(np.int16) if len(cells) else np.zeros((0, 3), np.int16)
    return ModCar(source, int(desc.size()[1]), shape, pids, cols, poses.astype(np.float32), cls, frames, labels,
                  mods, solids, pc, off)


class ModuleLibrary:
    def __init__(self, cars: list[ModCar]):
        self.cars = cars
        self.by_width: dict[int, list[int]] = {}
        for i, c in enumerate(cars):
            self.by_width.setdefault(c.width, []).append(i)

    def save(self, path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self.cars, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "ModuleLibrary":
        with open(path, "rb") as fh:
            return cls(pickle.load(fh))


# ---- retrieval and assembly ---------------------------------------------------------------------------------------
@dataclass
class Choice:
    module: str
    car: int  # index into the library
    flip: bool
    dx: int
    dz: int
    score: float
    iou: float
    color: float
    kept: int = 0
    total: int = 0


def score_cells(cells: np.ndarray, cls: np.ndarray, R: np.ndarray, Rhit: np.ndarray, C: np.ndarray, dx: int,
                dz: int) -> tuple[float, float, float]:
    """(score, iou, colour agreement) of module cells shifted by (dx, 0, dz) against target region ``R``
    (``Rhit``: R grown by one cell along x inside the profile, so a module may spill over a section cut)."""
    s = cells.astype(np.int32) + np.array([dx, 0, dz])
    ok = ((s >= 0) & (s < np.array(R.shape))).all(1)
    s, k = s[ok], cls[ok]
    hit = Rhit[tuple(s.T)] if len(s) else np.zeros(0, bool)
    nh = int(hit.sum())
    iou = nh / max(int(R.sum()) + len(cells) - nh, 1)
    col = float((k[hit] == C[tuple(s[hit].T)]).mean()) if nh else 0.0
    return iou * (1 - COLOR_WEIGHT + COLOR_WEIGHT * col), iou, col


def target_view(target: CarDescriptor):
    P = target.profile()
    C = target.cell_classes(P)
    fr = car_frame(P, target.wheels)
    return P, C, fr


def retrieve(lib: ModuleLibrary, target: CarDescriptor, k: int = 8, exclude=frozenset(), dedup: float = 0.9,
             width_tol: int = 0) -> tuple[dict[str, list[Choice]], dict]:
    """Top-k candidates per module; ``exclude`` sources and (``dedup``) cars whose whole outer shape matches the
    target almost exactly (same model stored twice), so a target cannot be rebuilt from its own copy."""
    P, C, fr = target_view(target)
    reg = regions(P, fr, C)
    grow = np.zeros((3, 1, 1), bool)
    grow[:, 0, 0] = True
    W = target.size()[1]
    pool = [i for w, ids in lib.by_width.items() if abs(w - W) <= width_tol for i in ids
            if lib.cars[i].source not in exclude]
    tsolid = column_filled(P)
    dropped = []
    if dedup:
        keep = []
        for i in pool:
            c = lib.cars[i]
            best = 0.0
            for o in (0, 1):
                dx = int(round(fr.mid - c.frames[o].mid))
                s = np.roll(c.solid[o], dx, axis=0)
                best = max(best, (s & tsolid).sum() / max((s | tsolid).sum(), 1))
            (dropped if best >= dedup else keep).append(i)
        pool = keep
    out: dict[str, list[Choice]] = {}
    for m in MODULES:
        R = reg[m]
        if not R.any():
            out[m] = []
            continue
        Rhit = ndimage.binary_dilation(R, grow) & P
        cands = []
        for i in pool:
            car = lib.cars[i]
            for o in (0, 1):
                if m not in car.mods[o]:
                    continue
                cells, cls = car.mods[o][m]
                dx0, dz0 = anchor(m, fr, car.frames[o])
                if m == "wing":
                    dx0 = _wing_anchor(cells, fr, car.frames[o])
                zs = (0,) if m in ("chassis", "wheels_a", "wheels_b") else (-1, 0, 1)
                best = None
                for ddx in (-1, 0, 1):
                    for ddz in zs:
                        sc = score_cells(cells, cls, R, Rhit, C, dx0 + ddx, dz0 + ddz)
                        if best is None or sc[0] > best[0][0]:
                            best = (sc, dx0 + ddx, dz0 + ddz)
                (s, iou, col), dx, dz = best
                cands.append(Choice(m, i, bool(o), dx, dz, s, iou, col))
        cands.sort(key=lambda c: -c.score)
        out[m] = cands[:k]
    return out, {"pool": len(pool), "pool_ids": pool, "dedup_dropped": [lib.cars[i].source for i in dropped]}


@dataclass
class Assembly:
    part_ids: list
    colors: list
    poses: list
    modules: list  # module name per part
    choices: list  # Choice per module used
    info: dict


def assemble(lib: ModuleLibrary, target: CarDescriptor, shapes: PartShapes | None = None, k: int = 12,
             exclude=frozenset(), dedup: float = 0.9, max_collide: float = 0.2, max_outside: float = 0.5,
             recolor: bool = True, connected: bool = True, glass_fallback: bool = True) -> Assembly:
    """Put the retrieved modules together (order ``ORDER``). ``connected``: try every candidate at its anchor
    +-1 cell in x and z and keep only the parts that end up joined to the model built so far (BrickNet's connection
    inference), so the result is one piece; the value of a placement is its shape score x the fraction of its volume
    that is kept."""
    shapes = shapes or PartShapes()
    cands, info = retrieve(lib, target, k=k, exclude=exclude, dedup=dedup)
    P, C, fr = target_view(target)
    reg = regions(P, fr, C)
    grow = np.zeros((3, 1, 1), bool)
    grow[:, 0, 0] = True
    fine_shape = np.array(P.shape) * SUB
    occ = np.zeros(fine_shape, bool)
    allowed = target.allowed(margin=(1, 1, 2)) | P
    fine_cell = CELL / SUB
    out = Assembly([], [], [], [], [], info)
    info["missing"] = []

    def filtered(car: ModCar, sel, poses):
        """Parts that neither collide nor stick out: [(j, pose, fine cells, volume)], total volume."""
        keep, vol = [], 0
        local = np.zeros(fine_shape, bool)
        for j, pose in zip(sel, poses):
            pts = shapes.interior_points([int(car.part_ids[j])], [pose])
            f = np.floor(pts / fine_cell).astype(int)
            inb = ((f >= 0) & (f < fine_shape)).all(1)
            f = np.unique(f[inb], axis=0) if inb.any() else np.zeros((0, 3), int)
            n = max(len(f), 1)
            vol += n
            if not len(f):
                continue
            collide = (occ[tuple(f.T)] | local[tuple(f.T)]).mean()
            outside = 1.0 - allowed[tuple((f // SUB).T)].mean()
            if collide <= max_collide and outside <= max_outside:
                keep.append((j, pose, f, n))
                local[tuple(f.T)] = True
        return keep, vol

    def attached(car: ModCar, keep):
        """The parts of ``keep`` joined to the model so far (or, for the first module, its largest piece)."""
        if not keep:
            return keep
        pids = out.part_ids + [int(car.part_ids[j]) for j, *_ in keep]
        poses = out.poses + [p for _, p, *_ in keep]
        g = bricknet.parse_ldr(poses_to_ldr(pids, [4] * len(pids), poses))
        if len(g.part_ids) != len(pids):
            return keep
        n0 = len(out.part_ids)
        if n0:
            main = next(c for c in g.components if 0 in c)
        else:
            main = max(g.components, key=len)
        return [kp for i, kp in enumerate(keep) if n0 + i in main]

    # second pass: a module that could not be joined may fit once its neighbours are in place
    todo = list(ORDER)
    for m in todo + [None] + list(ORDER):
        if m is None:
            todo = list(info["missing"])
            info["missing"] = []
            continue
        if m not in todo or any(c.module == m for c in out.choices):
            continue
        options = list(cands[m])
        # the same module of a donor already used, where that donor's other module sits: its original connections
        # to that module are kept (e.g. the wheels of the car the ends came from)
        seen = {(c.car, c.flip, c.dx, c.dz) for c in options}
        for c in out.choices:
            key = (c.car, c.flip, c.dx, c.dz)
            if key not in seen and m in lib.cars[c.car].mods[int(c.flip)] and (m != "glass" or glass_fallback):
                seen.add(key)
                options.append(Choice(m, c.car, c.flip, c.dx, c.dz, 0.0, 0.0, 0.0))
        R = reg[m]
        Rhit = ndimage.binary_dilation(R, grow) & P
        best = None
        for ch in options:
            car = lib.cars[ch.car]
            mi = MODULES.index(m)
            sel = np.nonzero(car.labels[int(ch.flip)] == mi)[0]
            if not len(sel):
                continue
            base = car.poses[sel].astype(np.float64)
            if ch.flip:
                base = flip_pose(car.shape[0], car.shape[1]) @ base
            cells, cls = car.mods[int(ch.flip)][m]
            offsets = [(ch.dx + a, ch.dz + b) for a in (0, -1, 1) for b in (0, -1, 1)] if connected \
                else [(ch.dx, ch.dz)]
            for dx, dz in offsets:
                sc = score_cells(cells, cls, R, Rhit, C, dx, dz) if R.any() else (1e-3, 0.0, 0.0)
                sc = (max(sc[0], 1e-3), sc[1], sc[2])  # a joinable module that misses its region still beats none
                if best is not None and sc[0] <= best[0]:
                    continue  # the value is at most the shape score: cannot win
                poses = base.copy()
                poses[:, 0, 3] += dx * LDU_PER_STUD
                poses[:, 2, 3] += dz * LDU_PER_PLATE
                keep, vol = filtered(car, sel, poses)
                if connected:
                    keep = attached(car, keep)
                value = sc[0] * sum(kp[3] for kp in keep) / max(vol, 1)
                if keep and (best is None or value > best[0]):
                    best = (value, Choice(m, ch.car, ch.flip, dx, dz, *sc), keep)
        if best is None:
            info["missing"].append(m)
            continue
        _, ch, keep = best
        car = lib.cars[ch.car]
        ch.kept, ch.total = len(keep), int((car.labels[int(ch.flip)] == MODULES.index(m)).sum())
        out.choices.append(ch)
        for j, pose, f, _ in keep:
            occ[tuple(f.T)] = True
            col = int(car.colors[j])
            if recolor and m not in ("wheels_a", "wheels_b", "glass") and car.part_cls[j] not in (GLASS, LIGHT):
                col = _target_color(f // SUB, P, C, col)
            out.part_ids.append(int(car.part_ids[j]))
            out.colors.append(col)
            out.poses.append(pose)
            out.modules.append(m)
    return out


def _target_color(cells: np.ndarray, P: np.ndarray, C: np.ndarray, default: int) -> int:
    """LDraw colour of the target's majority colour class over ``cells`` (the part's own colour if none)."""
    if not len(cells):
        return default
    inside = P[tuple(cells.T)]
    cls = [int(v) for v in C[tuple(cells[inside].T)] if v >= 0 and v not in (GLASS, LIGHT)]
    if not cls:
        return default
    return code(CLASS_NAMES[CLASSES[Counter(cls).most_common(1)[0][0]]])


# ---- B3b: retrieve a whole car, then swap modules ------------------------------------------------------------------
MIN_GAIN = 0.005  # a swap must raise the objective by this much
GLASS_WEIGHT = 0.15


@dataclass
class Placed:
    pid: int
    col: int
    pose: np.ndarray
    cells: np.ndarray  # target-grid cells (in bounds)
    fine: np.ndarray  # collision sub-cells (in bounds)
    module: str
    car: int
    j: int
    cls: int


def _place(lib: ModuleLibrary, i: int, flip: bool, dx: int, dz: int, sel, shapes: PartShapes, cache: dict,
           shape: np.ndarray) -> list[Placed]:
    """Parts ``sel`` of library car ``i`` (driving the other way if ``flip``) moved by (dx, 0, dz) cells."""
    car = lib.cars[i]
    fine_shape = shape * SUB
    F = flip_pose(car.shape[0], car.shape[1]) if flip else np.eye(4)
    lab = car.labels[int(flip)]
    out = []
    for j in sel:
        pose = F @ car.poses[j].astype(np.float64)
        key = (i, flip, int(j))
        if key not in cache:
            pts = shapes.interior_points([int(car.part_ids[j])], [pose])
            cache[key] = np.unique(np.floor(pts / (CELL / SUB)).astype(np.int32), axis=0) if len(pts) \
                else np.zeros((0, 3), np.int32)
        f = cache[key] + np.array([dx * SUB, 0, dz * SUB])
        f = f[((f >= 0) & (f < fine_shape)).all(1)]
        c = car.cells_of(int(j), flip) + np.array([dx, 0, dz])
        c = c[((c >= 0) & (c < shape)).all(1)]
        pose = pose.copy()
        pose[0, 3] += dx * LDU_PER_STUD
        pose[2, 3] += dz * LDU_PER_PLATE
        out.append(Placed(int(car.part_ids[j]), int(car.colors[j]), pose, c, f, MODULES[lab[j]], i, int(j),
                          int(car.part_cls[j])))
    return out


def objective(parts: list[Placed], P: np.ndarray, G: np.ndarray) -> float:
    """Mean of two shape IoUs with the target profile - closed cavities filled (keeps the gap under a wing) and
    every column filled (outer shape, as the evaluation scores it) - weighted by how much of the target's glass the
    model's glass covers (when the target has glass)."""
    S = np.zeros(P.shape, bool)
    Sg = np.zeros(P.shape, bool)
    for q in parts:
        if len(q.cells):
            S[tuple(q.cells.T)] = True
            if q.module == "glass":
                Sg[tuple(q.cells.T)] = True
    S = ndimage.binary_fill_holes(S)
    Sc, Pc = column_filled(S), column_filled(P)
    iou = 0.5 * ((S & P).sum() / max((S | P).sum(), 1) + (Sc & Pc).sum() / max((Sc | Pc).sum(), 1))
    if not G.any():
        return float(iou)
    glass = (ndimage.binary_dilation(Sg) & G).sum() / G.sum()
    return float(iou * (1 - GLASS_WEIGHT + GLASS_WEIGHT * glass))


def _largest_piece(parts: list[Placed]) -> list[Placed]:
    if not parts:
        return parts
    g = bricknet.parse_ldr(poses_to_ldr([q.pid for q in parts], [4] * len(parts), [q.pose for q in parts]))
    if len(g.part_ids) != len(parts):
        return parts
    main = max(g.components, key=len)
    return [parts[i] for i in sorted(main)]


def base_candidates(lib: ModuleLibrary, pool, P: np.ndarray, fr: Frame, n: int = 3) -> list[tuple]:
    """Best whole cars for the target: (iou, car, flip, dx), one entry per car, by column-filled IoU."""
    T = column_filled(P)
    best: dict[int, tuple] = {}
    for i in pool:
        c = lib.cars[i]
        for o in (0, 1):
            d0 = int(round(fr.mid - c.frames[o].mid))
            for dx in range(d0 - 3, d0 + 4):
                s = np.roll(c.solid[o], dx, axis=0)
                iou = (s & T).sum() / max((s | T).sum(), 1)
                if i not in best or iou > best[i][0]:
                    best[i] = (float(iou), i, bool(o), dx)
    return sorted(best.values(), key=lambda t: -t[0])[:n]


def swap_assemble(lib: ModuleLibrary, target: CarDescriptor, shapes: PartShapes | None = None, k: int = 8,
                  bases: int = 5, rounds: int = 2, exclude=frozenset(), dedup: float = 0.9,
                  max_collide: float = 0.2, max_outside: float = 0.5, recolor: bool = True) -> Assembly:
    """B3b: start from each of the ``bases`` best whole cars; then, module by module (``ORDER``, ``rounds``
    sweeps), try to replace the base's module with one of its k retrieved candidates (at the anchor +-1 cell in x
    and z), to drop it, or - when the base has none - to add one. The incoming module comes whole (only parts that
    stick out of the target are left behind); parts of the model in its way are removed. A change is kept only if
    the model, reduced to its largest connected piece, scores higher on ``objective``; so the result is always one
    piece. The best of the bases wins."""
    shapes = shapes or PartShapes()
    cands, info = retrieve(lib, target, k=k, exclude=exclude, dedup=dedup)
    P, C, fr = target_view(target)
    shape = np.array(P.shape)
    G = P & (C == GLASS)
    allowed = target.allowed(margin=(1, 1, 2)) | P
    cache: dict = {}
    results = []
    for biou, bi, bflip, bdx in base_candidates(lib, info["pool_ids"], P, fr, bases):
        car = lib.cars[bi]
        model = _largest_piece(_place(lib, bi, bflip, bdx, 0, range(len(car.part_ids)), shapes, cache, shape))
        J = objective(model, P, G)
        log = [{"module": "base", "name": MODULE_NAMES["base"], "source": car.source, "flip": bflip, "dx": bdx,
                "dz": 0, "objective": round(J, 4), "whole_car_iou": round(biou, 3), "kept": len(model),
                "total": len(car.part_ids)}]
        for _ in range(rounds):
            improved = False
            for m in ORDER:
                rest = [q for q in model if q.module != m]
                had = len(rest) < len(model)
                occ = np.zeros(shape * SUB, bool)
                best = (J + MIN_GAIN, None, None)
                options = ([None] if had else []) + [
                    (c, c.dx + a, c.dz + b) for c in cands[m] for a in (0, -1, 1) for b in (0, -1, 1)]
                for opt in options:
                    new, keep_rest = [], rest
                    if opt is not None:
                        ch, dx, dz = opt
                        sel = np.nonzero(lib.cars[ch.car].labels[int(ch.flip)] == MODULES.index(m))[0]
                        new = [q for q in _place(lib, ch.car, ch.flip, dx, dz, sel, shapes, cache, shape)
                               if len(q.fine) and 1.0 - allowed[tuple((q.fine // SUB).T)].mean() <= max_outside]
                        if not new:
                            continue
                        # the incoming module comes whole: parts of the model in its way make room for it
                        inc = np.zeros_like(occ)
                        for q in new:
                            inc[tuple(q.fine.T)] = True
                        keep_rest = [q for q in rest if not len(q.fine) or inc[tuple(q.fine.T)].mean() <= max_collide]
                    trial = keep_rest + new
                    if objective(trial, P, G) <= best[0]:  # the connected part can only score lower
                        continue
                    trial = _largest_piece(trial)
                    jt = objective(trial, P, G)
                    if jt > best[0]:
                        best = (jt, trial, opt)
                if best[1] is not None:
                    jt, model, opt = best
                    entry = {"module": m, "name": MODULE_NAMES[m], "objective": round(jt, 4),
                             "gain": round(jt - J, 4)}
                    if opt is None:
                        entry |= {"source": None, "action": "removed"}
                    else:
                        ch, dx, dz = opt
                        kept = sum(q.car == ch.car and q.module == m for q in model)
                        entry |= {"source": lib.cars[ch.car].source, "flip": ch.flip, "dx": dx, "dz": dz,
                                  "evicted": len(rest) - sum(q.car != ch.car or q.module != m for q in model),
                                  "action": "replaced" if had else "added", "kept": kept,
                                  "total": int((lib.cars[ch.car].labels[int(ch.flip)] == MODULES.index(m)).sum())}
                    log.append(entry)
                    J = jt
                    improved = True
            if not improved:
                break
        results.append((J, model, log))
    if not results:
        return Assembly([], [], [], [], [], info | {"log": []})
    J, model, log = max(results, key=lambda r: r[0])
    out = Assembly([], [], [], [], [], info | {"log": log, "objective": J,
                                               "bases": [(r[2][0]["source"], round(r[0], 4)) for r in results]})
    out.info["part_sources"] = [lib.cars[q.car].source for q in model]
    for q in model:
        col = q.col
        if recolor and q.module not in ("wheels_a", "wheels_b", "glass") and q.cls not in (GLASS, LIGHT):
            col = _target_color(q.fine // SUB, P, C, col)
        out.part_ids.append(q.pid)
        out.colors.append(col)
        out.poses.append(q.pose)
        out.modules.append(q.module)
    return out


def describe(lib: ModuleLibrary, asm: Assembly) -> list[dict]:
    """Where every module came from (for reports)."""
    if "log" in asm.info:
        return asm.info["log"]
    return [{"module": c.module, "name": MODULE_NAMES[c.module], "source": lib.cars[c.car].source, "flip": c.flip,
             "dx": c.dx, "dz": c.dz, "score": round(c.score, 3), "iou": round(c.iou, 3), "color": round(c.color, 3),
             "kept": c.kept, "total": c.total} for c in asm.choices]


def build_index(library_dir, workers: int = 8, limit: int | None = None) -> ModuleLibrary:
    """Index every car of a ``build_car_library`` folder (``index.jsonl``, ``desc/``, ``ldr/``)."""
    import json
    import multiprocessing as mp

    root = Path(library_dir)
    rows = [json.loads(ln) for ln in open(root / "index.jsonl")]
    if limit:
        rows = rows[:limit]
    jobs = [(str(root), r["source"]) for r in rows]
    with mp.get_context("fork").Pool(workers, initializer=_init_worker) as pool:
        cars = [c for c in pool.imap(_index_job, jobs, chunksize=8) if c is not None]
    return ModuleLibrary(cars)


_W: dict = {}


def _init_worker():
    _W["shapes"] = PartShapes()


def _index_job(job):
    root, source = job
    try:
        desc = CarDescriptor.load(Path(root) / "desc" / f"{source}.npz")
        return index_car(source, desc, (Path(root) / "ldr" / f"{source}.ldr").read_text(), _W["shapes"])
    except Exception as e:  # one malformed car must not stop the index
        print(f"[skip] {source}: {type(e).__name__}: {e}", flush=True)
        return None
