"""The bridge between the two halves: a coarse, stud-resolution description of a car.

Both a real car (mesh / visual hull, after ``canonical.canonicalize``) and a LEGO model (``from_ldr`` /
``from_graph``) are reduced to the *same* ``CarDescriptor`` through ``descriptor_from_points``, which bins
interior points of the object (car frame, LDU) into a grid of stud x stud x plate cells.

The descriptor records how much of each cell is filled, the colour class of each cell's visible surface, where the
wheels are, and the main colours. ``to_text`` serialises it (half width: the car is mirror-symmetric) as the
conditioning prefix of the generator; ``perturb`` is the training-time augmentation that makes LEGO-derived
descriptors look like real-car ones.

Text format ``v2`` (default) - one line per x row, one cell per half-width column from the centre outward::

    17 1k1y5 3y4+11k1 5y2 .

A cell lists the solid segments of that column bottom-up, joined by ``+``; a segment is its bottom plate followed by
colour runs (class letter + number of plates, ``colors.CLASSES``: ``t`` glass, ``l`` light, ...). ``3y4+11k1``:
yellow from plate 3 to 6, a gap, a black plate at 11 (a spoiler or a windscreen frame). Only gaps open to the
outside are kept: cavities closed inside the body (LEGO cars are hollow, a visual hull is not) are filled
(``profile``). Thin parts (a tile-thick wing) count when they are separated from the body by a gap.

``v1`` (the first training runs): ``top[w|l]/bottom`` per column, no colours, no gaps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import ndimage

from .colors import CLASS_INDEX, CLASSES, GLASS, LIGHT, class_of_name
from .units import LDU_PER_PLATE, LDU_PER_STUD

LABELS = {0: "", 1: "", 2: "w", 3: "l"}  # 1 = body, 2 = window, 3 = light; only 2 and 3 appear in text
LABEL_CODES = {"w": 2, "l": 3}
V2_CELL_RE = re.compile(r"\d+(?:[a-z]\d+)+(?:\+\d+(?:[a-z]\d+)+)*")


@dataclass(frozen=True)
class GridSpec:
    length: int = 32  # cells (studs) along x
    width: int = 12  # cells (studs) along y, symmetric about the centre plane
    height: int = 24  # cells (plates) along z
    sub: tuple[int, int, int] = (2, 2, 2)  # supersampling per cell when binning points

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.length, self.width, self.height)

    @property
    def cell(self) -> np.ndarray:
        return np.array([LDU_PER_STUD, LDU_PER_STUD, LDU_PER_PLATE])

    def subcell_centers(self) -> np.ndarray:
        """Grid-frame (LDU) centres of all supersample cells, shape (L*sx, W*sy, H*sz, 3)."""
        n = np.array(self.shape) * np.array(self.sub)
        step = self.cell / np.array(self.sub)
        axes = [(np.arange(n[a]) + 0.5) * step[a] for a in range(3)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), -1)


@dataclass
class CarDescriptor:
    occupancy: np.ndarray  # (L, W, H) float32 fill fraction
    spec: GridSpec = field(default_factory=GridSpec)
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))  # car-frame LDU of the grid corner
    wheels: tuple[float, ...] = ()  # wheel-centre x positions, in cells from the grid start
    colors: tuple[str, ...] = ()  # BrickNet colour names, main first
    labels: np.ndarray | None = None  # (L, W, H) uint8 semantic labels, optional
    fine: np.ndarray | None = None  # (L*sx, W*sy, H*sz) bool supersampled occupancy, optional (fitting baseline)
    surface: np.ndarray | None = None  # (L, W, H) int8 colour class of the cell's visible surface, -1 unknown

    def fine_target(self) -> np.ndarray:
        """Supersampled solid mask; falls back to repeating solid cells when ``fine`` is unknown."""
        if self.fine is not None:
            return self.fine
        sx, sy, sz = self.spec.sub
        return self.solid().repeat(sx, 0).repeat(sy, 1).repeat(sz, 2)

    # ---- basic views ------------------------------------------------------------------------------------------
    def solid(self, thr: float = 0.55) -> np.ndarray:
        return self.occupancy >= thr

    def extent(self, thr: float = 0.55) -> tuple[slice, slice, slice] | None:
        idx = np.argwhere(self.solid(thr))
        if not len(idx):
            return None
        lo, hi = idx.min(0), idx.max(0) + 1
        return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))

    def size(self, thr: float = 0.55) -> tuple[int, int, int]:
        ext = self.extent(thr)
        return (0, 0, 0) if ext is None else tuple(s.stop - s.start for s in ext)

    def iou(self, other: "CarDescriptor", thr: float = 0.55, filled: bool = False) -> float:
        """Voxel IoU. ``filled``: compare outer shapes, each column filled from its lowest to its highest solid cell
        (LEGO cars are hollow inside - open cockpits, empty bodies - while a descriptor from a mesh is solid)."""
        a, b = self.solid(thr), other.solid(thr)
        if filled:
            a, b = column_filled(a), column_filled(b)
        union = (a | b).sum()
        return float((a & b).sum() / union) if union else 1.0

    def symmetrized(self) -> "CarDescriptor":
        occ = 0.5 * (self.occupancy + self.occupancy[:, ::-1])
        lab = None if self.labels is None else np.maximum(self.labels, self.labels[:, ::-1])
        fine = None if self.fine is None else self.fine | self.fine[:, ::-1]
        surf = None
        if self.surface is not None:
            a, b = self.surface, self.surface[:, ::-1]
            half = self.spec.width // 2
            right = np.where(a >= 0, a, b)[:, half:]  # outer side first, its mirror where unknown
            surf = np.concatenate([right[:, ::-1], right], 1).astype(np.int8)
        return replace(self, occupancy=occ.astype(np.float32), labels=lab, fine=fine, surface=surf)

    def profile(self, thr: float = 0.55, thin: float = 0.2) -> np.ndarray:
        """Solid cells as written in the text: cells filled at least ``thr``, plus thinner ones (``thin``) that are
        separated from those by a gap above and below (a wing made of tiles), with closed cavities filled - only
        gaps that are open to the outside remain, as in a visual hull."""
        occ = self.symmetrized().occupancy
        strong = occ >= thr
        near = strong.copy()
        near[:, :, 1:] |= strong[:, :, :-1]
        near[:, :, :-1] |= strong[:, :, 1:]
        solid = strong | ((occ >= thin) & ~near)
        return ndimage.binary_fill_holes(solid)

    def cell_classes(self, solid: np.ndarray) -> np.ndarray:
        """Colour class of every solid cell: its own visible surface; otherwise (inside the body) the nearest known
        cell above in its column, else below; else the main colour."""
        main = class_of_name(self.colors[0]) if self.colors else CLASS_INDEX["s"]
        known = self.symmetrized().surface if self.surface is not None else np.full(solid.shape, -1, np.int8)
        known = np.where(solid, known, -1)
        out = np.full(solid.shape, -1, dtype=np.int64)
        for i, j in np.argwhere(solid.any(2)):
            col, k = known[i, j], np.nonzero(solid[i, j])[0]
            vals = col[k]
            if (vals >= 0).any():
                pos = k[vals >= 0]
                for kk in k:
                    above = pos[pos >= kk]
                    out[i, j, kk] = col[above.min()] if len(above) else col[pos.max()]
            else:
                out[i, j, k] = main
        return out

    def allowed(self, margin=(1, 1, 2), thr: float = 0.3) -> np.ndarray:
        """Cells a generated part may occupy: the (loosely thresholded) shape dilated by ``margin`` cells."""
        mask = self.solid(thr)
        for axis, m in enumerate(margin):
            if m:
                st = np.zeros((3, 3, 3), dtype=bool)
                st[1, 1, 1] = True
                idx = [1, 1, 1]
                for d in (0, 2):
                    idx[axis] = d
                    st[tuple(idx)] = True
                mask = ndimage.binary_dilation(mask, st, iterations=m)
        return mask

    # ---- text serialisation -----------------------------------------------------------------------------------
    def half_heightmap(self, thr: float = 0.55):
        """(top, bottom, label) per (x row, half-width column from the centre outward); -1 where empty."""
        solid = self.symmetrized().solid(thr)
        half = solid[:, self.spec.width // 2 :, :]
        has = half.any(2)
        top = np.where(has, self.spec.height - 1 - np.argmax(half[:, :, ::-1], axis=2), -1)
        bottom = np.where(has, np.argmax(half, axis=2), -1)
        lab = np.zeros(top.shape, dtype=np.uint8)
        if self.labels is not None:
            labs = np.maximum(self.labels, self.labels[:, ::-1])[:, self.spec.width // 2 :, :]
            ii, jj = np.nonzero(has)
            lab[ii, jj] = labs[ii, jj, top[ii, jj]]
        return top, bottom, lab

    def segments(self, max_segments: int = 3) -> tuple[np.ndarray, list]:
        """(profile, per (row, half column) list of (bottom, top) runs); at most ``max_segments`` per column (the
        smallest gaps are closed first)."""
        solid = self.profile()
        half = self.spec.width // 2
        out = []
        for i in range(solid.shape[0]):
            row = []
            for j in range(half, solid.shape[1]):
                k = np.nonzero(solid[i, j])[0]
                runs = []
                for kk in k:
                    if runs and kk == runs[-1][1] + 1:
                        runs[-1][1] = kk
                    else:
                        runs.append([kk, kk])
                while len(runs) > max_segments:
                    g = int(np.argmin([runs[n + 1][0] - runs[n][1] for n in range(len(runs) - 1)]))
                    runs[g : g + 2] = [[runs[g][0], runs[g + 1][1]]]
                    solid[i, j, runs[g][0] : runs[g][1] + 1] = True
                    solid[i, 2 * half - 1 - j, runs[g][0] : runs[g][1] + 1] = True
                row.append([tuple(r) for r in runs])
            out.append(row)
        return solid, out

    def to_text(self, style: str | None = None, thr: float = 0.55, fmt: str = "v2") -> str:
        if fmt == "v1":
            return self._to_text_v1(style, thr)
        solid, segs = self.segments()
        cls = self.cell_classes(solid)
        half = self.spec.width // 2
        L, W, H = (0, 0, 0) if not solid.any() else tuple(int(v) for v in np.ptp(np.argwhere(solid), 0) + 1)
        lines = [f"<{style}>"] if style else []
        lines.append(f"car {L}x{W}x{H}")
        if self.wheels:
            lines.append("wheels " + " ".join(f"{w:g}" for w in self.wheels))
        if self.colors:
            lines.append("colors " + ", ".join(self.colors))
        for i, row in enumerate(segs):
            if not any(row):
                continue
            cells = []
            for j, runs in enumerate(row):
                if not runs:
                    cells.append(".")
                    continue
                parts = []
                for b, t in runs:
                    c = cls[i, half + j, b : t + 1]
                    txt, n = str(b), 1
                    for kk in range(1, len(c) + 1):
                        if kk < len(c) and c[kk] == c[kk - 1]:
                            n += 1
                            continue
                        txt += f"{CLASSES[c[kk - 1]]}{n}"
                        n = 1
                    parts.append(txt)
                cells.append("+".join(parts))
            while cells and cells[-1] == ".":
                cells.pop()
            lines.append(f"{i} " + " ".join(cells))
        return "\n".join(lines)

    def _to_text_v1(self, style: str | None = None, thr: float = 0.55) -> str:
        top, bottom, lab = self.half_heightmap(thr)
        rows = np.nonzero((top >= 0).any(1))[0]
        L, W, H = self.size(thr)
        lines = []
        if style:
            lines.append(f"<{style}>")
        lines.append(f"car {L}x{W}x{H}")
        if self.wheels:
            lines.append("wheels " + " ".join(f"{w:g}" for w in self.wheels))
        if self.colors:
            lines.append("colors " + ", ".join(self.colors))
        for i in rows:
            cols = []
            for j in range(top.shape[1]):
                cols.append("." if top[i, j] < 0 else f"{top[i, j]}{LABELS[int(lab[i, j])]}/{bottom[i, j]}")
            while cols and cols[-1] == ".":
                cols.pop()
            lines.append(f"{i} " + " ".join(cols))
        return "\n".join(lines)

    @classmethod
    def from_text(cls, text: str, spec: GridSpec | None = None) -> "CarDescriptor":
        """Inverse of ``to_text`` up to its resolution (filled columns between bottom and top)."""
        spec = spec or GridSpec()
        occ = np.zeros(spec.shape, dtype=np.float32)
        labels = np.zeros(spec.shape, dtype=np.uint8)
        surface = np.full(spec.shape, -1, dtype=np.int8)
        wheels, colors = (), ()
        half = spec.width // 2
        for line in text.strip().splitlines():
            tok = line.split()
            if not tok or tok[0].startswith("<") or tok[0] == "car":
                continue
            if tok[0] == "wheels":
                wheels = tuple(float(t) for t in tok[1:])
            elif tok[0] == "colors":
                colors = tuple(c.strip() for c in line[len("colors ") :].split(","))
            elif tok[0].isdigit():
                i = int(tok[0])
                for j, cell in enumerate(tok[1:]):
                    if i < spec.length and j < half and V2_CELL_RE.fullmatch(cell):
                        for seg in cell.split("+"):
                            k = int(re.match(r"\d+", seg)[0])
                            for c, n in re.findall(r"([a-z])(\d+)", seg):
                                for kk in range(k, min(k + int(n), spec.height)):
                                    for jj in (half + j, half - 1 - j):
                                        occ[i, jj, kk] = 1.0
                                        surface[i, jj, kk] = CLASS_INDEX[c]
                                k += int(n)
                        continue
                    m = re.fullmatch(r"(\d+)([a-z]?)/(\d+)", cell)
                    if m is None or i >= spec.length or j >= half:
                        continue
                    t, lab, b = int(m[1]), m[2], int(m[3])
                    for jj in (half + j, half - 1 - j):
                        occ[i, jj, b : t + 1] = 1.0
                        if lab:
                            labels[i, jj, t] = LABEL_CODES[lab]
        return cls(occ, spec, np.zeros(3), wheels, colors, labels if labels.any() else None,
                   surface=surface if (surface >= 0).any() else None)

    # ---- augmentation -----------------------------------------------------------------------------------------
    def perturb(self, rng: np.random.Generator, strength: float = 1.0) -> "CarDescriptor":
        """Training-time noise: blur + threshold jitter, random 1-cell dilation/erosion, column height jitter,
        wheel jitter. Keeps mirror symmetry."""
        occ = self.occupancy.astype(np.float32)
        sigma = rng.uniform(0.0, 0.8) * strength
        if sigma > 0.05:
            occ = ndimage.gaussian_filter(occ, sigma=(sigma, sigma, sigma * 2.5))
        solid = occ >= rng.uniform(0.5 - 0.15 * strength, 0.5 + 0.15 * strength)
        u = rng.uniform()
        st = np.ones((3, 3, 1), dtype=bool)
        if u < 0.25 * strength:
            solid = ndimage.binary_dilation(solid, st)
        elif u < 0.5 * strength:
            eroded = ndimage.binary_erosion(solid, st)
            solid = eroded if eroded.sum() > 0.5 * solid.sum() else solid
        # per-column top jitter of +-1 plate on a fraction of columns
        cols = np.argwhere(solid.any(2))
        pick = cols[rng.uniform(size=len(cols)) < 0.1 * strength]
        for i, j in pick:
            ks = np.nonzero(solid[i, j])[0]
            top = ks.max()
            if rng.uniform() < 0.5 and top + 1 < solid.shape[2]:
                solid[i, j, top + 1] = True
            elif top > ks.min():
                solid[i, j, top] = False
        # parts above a gap (wings, frames) and thin parts would not survive the (vertical) blur: keep them, mostly
        sym = self.symmetrized().occupancy
        prof = self.profile()
        started = np.maximum.accumulate(prof, axis=2)
        above_gap = prof & np.maximum.accumulate(started & ~prof, axis=2)
        keep = above_gap | (prof & (sym > 0) & (sym < 0.55))
        if keep.any() and rng.uniform() < 0.85:
            solid |= keep
        wheels = tuple(round(w + rng.uniform(-0.5, 0.5) * strength, 1) for w in self.wheels)
        surface = self.surface
        if surface is not None:
            surface = surface.copy()
            # photos without window / light masks: glass reads as dark paint, lamps as the body colour
            if rng.uniform() < 0.35 * strength:
                surface[surface == GLASS] = CLASS_INDEX["k"] if rng.uniform() < 0.6 else CLASS_INDEX["g"]
            if rng.uniform() < 0.3 * strength:
                surface[surface == LIGHT] = class_of_name(self.colors[0]) if self.colors else -1
            surface[rng.uniform(size=surface.shape) < 0.05 * strength] = -1  # unseen cells: inherit from above
        out = replace(self, occupancy=solid.astype(np.float32), wheels=wheels, fine=None, surface=surface)
        return out.symmetrized()

    # ---- io ---------------------------------------------------------------------------------------------------
    def save(self, path) -> None:
        np.savez_compressed(
            path,
            occupancy=self.occupancy,
            spec=np.array([*self.spec.shape, *self.spec.sub]),
            origin=self.origin,
            wheels=np.array(self.wheels, dtype=np.float64),
            colors=np.array(self.colors, dtype=str),
            labels=self.labels if self.labels is not None else np.zeros(0, dtype=np.uint8),
            fine=np.packbits(self.fine) if self.fine is not None else np.zeros(0, dtype=np.uint8),
            surface=self.surface if self.surface is not None else np.zeros(0, dtype=np.int8),
        )

    @classmethod
    def load(cls, path) -> "CarDescriptor":
        d = np.load(path, allow_pickle=False)
        s = [int(v) for v in d["spec"]]
        spec = GridSpec(*s[:3], tuple(s[3:]))
        labels = d["labels"] if d["labels"].size else None
        fine = None
        if "fine" in d.files and d["fine"].size:
            n = np.array(spec.shape) * np.array(spec.sub)
            fine = np.unpackbits(d["fine"])[: n.prod()].reshape(n).astype(bool)
        surface = d["surface"] if "surface" in d.files and d["surface"].size else None
        return cls(
            d["occupancy"], spec, d["origin"], tuple(d["wheels"].tolist()), tuple(d["colors"].tolist()), labels, fine,
            surface,
        )


def column_filled(solid: np.ndarray) -> np.ndarray:
    """Fill every vertical column between its lowest and highest solid cell."""
    up = np.maximum.accumulate(solid, axis=2)
    down = np.maximum.accumulate(solid[:, :, ::-1], axis=2)[:, :, ::-1]
    return up & down


def detect_wheels(occupancy: np.ndarray, thr: float = 0.25, tol: int = 0) -> tuple[float, ...]:
    """Wheel-centre x positions from the ground-contact profile: columns whose lowest filled cell is within
    ``tol`` plates of the global minimum form runs; the two outermost substantial runs are the axles.
    Applied identically to LEGO and real-car descriptors so both domains agree."""
    solid = occupancy >= thr
    has = solid.any((1, 2))
    if not has.any():
        return ()
    rows = solid.any(1)  # (L, H): filled anywhere across the width
    low = np.where(rows.any(1), np.argmax(rows, axis=1), 10**6)  # lowest filled plate per x row
    contact = low <= low.min() + tol
    runs, start = [], None
    for i, c in enumerate(np.append(contact, False)):
        if c and start is None:
            start = i
        elif not c and start is not None:
            runs.append((start, i))
            start = None
    if len(runs) < 2:
        return ()
    span = np.nonzero(has)[0]
    if sum(b - a for a, b in runs) > 0.8 * (span[-1] - span[0] + 1):
        return ()  # flat-bottomed: no distinguishable wheels
    first, last = runs[0], runs[-1]
    return tuple(round((a + b) / 2, 1) for a, b in (first, last))


def descriptor_from_points(
    points: np.ndarray,
    spec: GridSpec | None = None,
    *,
    center_y: float | None = None,
    colors: tuple[str, ...] = (),
    point_labels: np.ndarray | None = None,
    point_classes: np.ndarray | None = None,
    color_points: np.ndarray | None = None,
    color_classes: np.ndarray | None = None,
    min_count: int = 1,
    snap: float | None = None,
) -> CarDescriptor:
    """Bin interior points (car frame, LDU, z up) into a descriptor.

    The grid is placed so that the object's lowest point is at z = 0, its lateral centre (``center_y``, default
    the midpoint of the y extent) is on the grid's centre plane, and it is centred along x by a whole number of
    studs (stud boundaries of a LEGO model stay aligned with cell boundaries).

    ``snap`` (LEGO models): round the lateral centre to a multiple of ``snap`` and the origin to whole LDU, so that
    parts on the LDraw grid get integer grid poses (the ``@root`` line stores integers; a fractional origin would
    shift the decoded model and flip anchors that sit on cell boundaries).

    Surface colours (``colors.CLASSES`` indices, -1 unknown), either
    * ``point_classes`` for the interior ``points`` (LEGO: the part's colour) - a cell takes the majority class of
      its sub-cells on the outside of the object (next to empty space connected to the grid border), so parts
      hidden inside the body do not count; or
    * ``color_points`` / ``color_classes``: surface samples (mesh, photos) - majority class per cell.
    """
    spec = spec or GridSpec()
    pts = np.asarray(points, dtype=np.float64)
    if not len(pts):
        return CarDescriptor(np.zeros(spec.shape, np.float32), spec)
    cell = spec.cell
    lo, hi = pts.min(0), pts.max(0)
    cy = 0.5 * (lo[1] + hi[1]) if center_y is None else center_y
    if snap:
        cy = round(cy / snap) * snap
    pad_x = np.floor((spec.length * cell[0] - (hi[0] - lo[0])) / 2 / cell[0]) * cell[0]
    origin = np.array([lo[0] - max(pad_x, 0.0), cy - spec.width * cell[1] / 2, lo[2]])
    if snap:
        origin = np.round(origin)

    sub = np.array(spec.sub)
    n = np.array(spec.shape) * sub
    # points on a lattice with the sub-cell pitch (``interior_points_from_surface``) sit exactly on bin centres
    # relative to ``origin``; without the epsilon, round-off sends a whole slice into its neighbour (an empty plate
    # layer that splits the car in two, next to a doubled one)
    idx = np.floor((pts - origin) / (cell / sub) + 1e-6).astype(np.int64)
    keep = ((idx >= 0) & (idx < n)).all(1)
    idx = idx[keep]
    counts = np.zeros(n, dtype=np.int32)
    np.add.at(counts, tuple(idx.T), 1)
    fine = counts >= min_count
    occ = fine.reshape(spec.length, sub[0], spec.width, sub[1], spec.height, sub[2]).mean((1, 3, 5))

    labels = None
    if point_labels is not None:
        lab = np.asarray(point_labels)[keep]
        labels = np.zeros(spec.shape, dtype=np.uint8)
        cidx = idx // sub
        for code in np.unique(lab):
            if code <= 1:
                continue
            m = lab == code
            labels[tuple(cidx[m].T)] = code
    occ = occ.astype(np.float32)
    surface = None
    n_cls = len(CLASSES)
    if color_points is not None and color_classes is not None:
        cp = np.asarray(color_points, dtype=np.float64)
        cc = np.asarray(color_classes)
        ci = np.floor((cp - origin) / cell + 1e-6).astype(np.int64)
        ok = ((ci >= 0) & (ci < np.array(spec.shape))).all(1) & (cc >= 0)
        surface = _majority(ci[ok], cc[ok], spec.shape, n_cls)
    elif point_classes is not None:
        pc = np.asarray(point_classes)[keep]
        ok = pc >= 0
        counts_c = np.zeros((*n, n_cls), dtype=np.int32)
        np.add.at(counts_c, (*idx[ok].T, pc[ok]), 1)
        exterior = ~ndimage.binary_fill_holes(fine)
        exposed = fine & ndimage.binary_dilation(exterior)
        fi = np.argwhere(exposed)
        fcls = counts_c[tuple(fi.T)].argmax(1)
        has = counts_c[tuple(fi.T)].sum(1) > 0
        surface = _majority(fi[has] // sub, fcls[has], spec.shape, n_cls)
    return CarDescriptor(occ, spec, origin, detect_wheels(occ), tuple(colors), labels, fine, surface)


def with_surface_samples(desc: CarDescriptor, points: np.ndarray, classes: np.ndarray) -> CarDescriptor:
    """``desc`` with its surface colours from coloured surface samples (grid placement as in ``desc``)."""
    pts, cls = np.asarray(points, dtype=np.float64), np.asarray(classes)
    ci = np.floor((pts - desc.origin) / desc.spec.cell + 1e-6).astype(np.int64)
    ok = ((ci >= 0) & (ci < np.array(desc.spec.shape))).all(1) & (cls >= 0)
    return replace(desc, surface=_majority(ci[ok], cls[ok], desc.spec.shape, len(CLASSES)))


def _majority(cells: np.ndarray, classes: np.ndarray, shape, n_cls: int) -> np.ndarray:
    """Most frequent class per cell (-1 where none)."""
    counts = np.zeros((*shape, n_cls), dtype=np.int32)
    np.add.at(counts, (*np.asarray(cells).T, np.asarray(classes)), 1)
    return np.where(counts.sum(-1) > 0, counts.argmax(-1), -1).astype(np.int8)
