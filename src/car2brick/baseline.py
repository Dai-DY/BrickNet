"""Non-learned baselines: descriptor -> LDraw model.

B0  ``greedy_plates``: every solid cell of the (symmetrised) descriptor becomes plate volume; each plate layer is
    covered greedily with the largest plates that fit, alternating the preferred direction between layers so
    joints are staggered and layers interlock.
B1  ``cap_tiles=True``: plates whose whole top is exposed are swapped for tiles of the same size (studless look).

Part footprints and orientation are read from BrickNet's connector labels (studs on top / holes underneath), so
no part geometry files are needed. Output is LDraw text in the car frame mapped by ``units.LD_FROM_CAR``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage

import bricknet

from .descriptor import CarDescriptor
from .units import LD_FROM_CAR, LDU_PER_PLATE, LDU_PER_STUD

PLATE_NAMES = [
    "plate 2x10", "plate 2x8", "plate 2x6", "plate 2x4", "plate 1x10", "plate 1x8", "plate 2x3", "plate 1x6",
    "plate 2x2", "plate 1x4", "plate 1x3", "plate 1x2", "plate 1x1",
]
TILE_STEMS = {(1, 1): "3070b", (1, 2): "3069b", (1, 4): "2431", (1, 6): "6636", (1, 8): "4162", (2, 2): "3068b",
              (2, 4): "87079", (2, 6): "69729"}


@dataclass(frozen=True)
class Footprint:
    part_id: int
    nx: int  # studs along local LDraw X
    nz: int  # studs along local LDraw Z
    center: tuple[float, float]  # local (x, z) of the stud-grid centre


@lru_cache(maxsize=None)
def footprint(part_id: int) -> Footprint:
    studs = bricknet.load_connectors()[part_id].get(("stud", None)) or bricknet.load_connectors()[part_id].get(
        ("hole", None)
    )
    if not studs:
        raise ValueError(f"part {part_id} has no studs/holes")
    xyz = np.array([f[:3, 3] for f in studs])
    nx = int(round(np.ptp(xyz[:, 0]) / LDU_PER_STUD)) + 1
    nz = int(round(np.ptp(xyz[:, 2]) / LDU_PER_STUD)) + 1
    c = 0.5 * (xyz.min(0) + xyz.max(0))
    return Footprint(part_id, nx, nz, (float(c[0]), float(c[2])))


@lru_cache(maxsize=1)
def plate_library() -> dict[tuple[int, int], int]:
    """(short, long) studs -> part id; every plate is long along its local X (asserted)."""
    cat = bricknet.load_catalog()
    lib = {}
    for name in PLATE_NAMES:
        pid = cat.name_to_id[name]
        fp = footprint(pid)
        assert fp.nx >= fp.nz, (name, fp)
        lib[(fp.nz, fp.nx)] = pid
    return lib


@lru_cache(maxsize=1)
def tile_library() -> dict[tuple[int, int], int]:
    cat = bricknet.load_catalog()
    return {k: cat.stem_to_id[s] for k, s in TILE_STEMS.items() if s in cat.stem_to_id}


@dataclass
class Placement:
    part_id: int
    color: int
    i0: int  # first cell along x
    j0: int  # first cell along y
    k: int  # plate layer (bottom at k * 8 LDU)
    di: int  # cells along x
    dj: int  # cells along y

    def ldraw_matrix(self, origin=np.zeros(3)) -> np.ndarray:
        """4x4 LDraw-world transform. Plates are long along local X; rotate 90 deg about Y when long along y."""
        fp = footprint(self.part_id)
        along_x = self.di == fp.nx and self.dj == fp.nz
        rot_local = np.eye(3) if along_x else np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
        center_car = np.array(
            [(self.i0 + self.di / 2) * LDU_PER_STUD, (self.j0 + self.dj / 2) * LDU_PER_STUD,
             (self.k + 1) * LDU_PER_PLATE]
        ) + origin
        m = np.eye(4)
        m[:3, :3] = rot_local
        m[:3, 3] = LD_FROM_CAR @ center_car - rot_local @ np.array([fp.center[0], 0.0, fp.center[1]])
        return m


def _anchor_floating(free: np.ndarray, anchor: np.ndarray, lib, rng=None) -> list[tuple[int, int, int, int, int]]:
    """Cover cells that have nothing above or below (``free & ~anchor``) first, each with the largest free
    rectangle that also covers an anchored cell, so the part grips the rest of the model. Updates ``free``."""
    sizes = sorted(lib, key=lambda s: -s[0] * s[1])
    out = []
    cells = np.argwhere(free & ~anchor)
    if rng is not None:
        cells = cells[rng.permutation(len(cells))]
    for i, j in cells:
        if not free[i, j]:
            continue
        fits = []
        for short, long_ in sizes:
            for di, dj in {(long_, short), (short, long_)}:
                for i0 in range(i - di + 1, i + 1):
                    for j0 in range(j - dj + 1, j + 1):
                        if i0 < 0 or j0 < 0:
                            continue
                        block = free[i0 : i0 + di, j0 : j0 + dj]
                        if block.shape == (di, dj) and block.all() and anchor[i0 : i0 + di, j0 : j0 + dj].any():
                            fits.append((lib[(short, long_)], i0, j0, di, dj))
            if len(fits) >= (3 if rng is not None else 1):
                break
        if not fits:
            continue  # no anchored rectangle fits: the plain cover takes it
        pid, i0, j0, di, dj = fits[0] if rng is None else fits[int(rng.integers(len(fits)))]
        free[i0 : i0 + di, j0 : j0 + dj] = False
        out.append((pid, int(i0), int(j0), di, dj))
    return out


def _cover_layer(mask: np.ndarray, prefer_x: bool, lib, rng: np.random.Generator | None = None,
                 anchor: np.ndarray | None = None) -> list[tuple[int, int, int, int, int]]:
    """Greedy rectangle cover of one 2D layer. Returns (part_id, i0, j0, di, dj).

    ``anchor``: cells that connect vertically (something above or below); the others are covered first by parts
    that reach an anchored cell (see ``_anchor_floating``).
    With ``rng``: cells are visited in random order and each picks randomly among the (up to) three largest fitting
    rectangles in a random orientation - used to search for covers that connect better between layers."""
    sizes = sorted(lib, key=lambda s: -s[0] * s[1])
    free = mask.copy()
    out = _anchor_floating(free, anchor, lib, rng) if anchor is not None and lib else []
    cells = np.argwhere(free)
    if rng is not None:
        cells = cells[rng.permutation(len(cells))]
    for i, j in cells:
        if not free[i, j]:
            continue
        fits = []
        for short, long_ in sizes:
            px = prefer_x if rng is None else bool(rng.integers(2))
            dims = [(long_, short), (short, long_)] if px else [(short, long_), (long_, short)]
            for di, dj in dims:
                if rng is not None:  # random visiting order: the cell may be any corner of the rectangle
                    i0, j0 = i - rng.integers(di), j - rng.integers(dj)
                else:
                    i0, j0 = i, j
                if i0 < 0 or j0 < 0:
                    continue
                block = free[i0 : i0 + di, j0 : j0 + dj]
                if block.shape == (di, dj) and block.all():
                    fits.append((lib[(short, long_)], int(i0), int(j0), di, dj))
                    break
            if fits and (rng is None or len(fits) >= 3):
                break
        if not fits:  # a corner guess missed: fall back to the rectangle anchored at the cell (1x1 always fits)
            fits = _cover_layer(free[i : i + 1, j : j + 1], prefer_x, lib)
            fits = [(p, int(i), int(j), di, dj) for p, _, _, di, dj in fits]
        pid, i0, j0, di, dj = fits[0] if rng is None else fits[int(rng.integers(len(fits)))]
        free[i0 : i0 + di, j0 : j0 + dj] = False
        out.append((pid, i0, j0, di, dj))
    return out


def plate_components(placements: list["Placement"]) -> list[list[int]]:
    """Connected components of a layered plate model: parts connect when they overlap in xy on adjacent layers
    (studs of the lower part into the tubes of the upper one). Largest first."""
    occ = {}
    for n, p in enumerate(placements):
        for i in range(p.i0, p.i0 + p.di):
            for j in range(p.j0, p.j0 + p.dj):
                occ[(i, j, p.k)] = n
    parent = list(range(len(placements)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (i, j, k), n in occ.items():
        m = occ.get((i, j, k + 1))
        if m is not None:
            parent[find(n)] = find(m)
    groups: dict[int, list[int]] = {}
    for n in range(len(placements)):
        groups.setdefault(find(n), []).append(n)
    return sorted(groups.values(), key=len, reverse=True)


def greedy_plates(desc: CarDescriptor, color: int | None = None, cap_tiles: bool = True,
                  thr: float = 0.55, repair_steps: int = 400, seed: int = 0) -> list[Placement]:
    """B0 / B1 placements. Layers are covered greedily (alternating direction); then, while the model falls apart,
    a local search re-covers random layers next to the stray pieces and keeps changes that do not add pieces
    (fewer parts breaks ties). Plates only connect to the layers above / below, so two regions that touch only
    side by side need a different cover to be bridged."""
    if color is None:
        cat = bricknet.load_catalog()
        color = cat.color_to_code.get(desc.colors[0], 4) if desc.colors else 4
    solid = desc.symmetrized().solid(thr)
    lib = plate_library()
    tiles = tile_library() if cap_tiles else {}
    layers = [k for k in range(solid.shape[2]) if solid[:, :, k].any()]

    def cover(k, rng=None):
        layer = solid[:, :, k]
        above = solid[:, :, k + 1] if k + 1 < solid.shape[2] else np.zeros_like(layer)
        below = solid[:, :, k - 1] if k > 0 else np.zeros_like(layer)
        # B1: cells with nothing on top get tiles (studless look); the rest are plates. A tile only grips the layer
        # below, so overhanging cells (nothing below either) stay plates, and so do the exposed cells right next to
        # them: a plate needs room to reach from the overhang back onto a supported cell.
        exposed = layer & ~above if tiles else np.zeros_like(layer)
        overhang = exposed & ~below
        exposed &= ~ndimage.binary_dilation(overhang)
        out = [Placement(pid, color, i0, j0, k, di, dj)
               for pid, i0, j0, di, dj in _cover_layer(exposed, prefer_x=(k % 2 == 0), lib=tiles, rng=rng)]
        out += [Placement(pid, color, i0, j0, k, di, dj)
                for pid, i0, j0, di, dj in _cover_layer(layer & ~exposed, prefer_x=(k % 2 == 0), lib=lib, rng=rng,
                                                        anchor=above | below)]
        return out

    per_layer = {k: cover(k) for k in layers}

    def flat(pl):
        return [p for k in layers for p in pl[k]]

    def score(pl):
        comps = plate_components(flat(pl))
        return len(comps), sum(len(c) for c in comps[1:]), sum(len(v) for v in pl.values())

    best = score(per_layer)
    rng = np.random.default_rng(seed)
    for _ in range(repair_steps):
        if best[0] == 1:
            break
        placements = flat(per_layer)
        comps = plate_components(placements)
        stray = {placements[n].k for c in comps[1:] for n in c}
        k = int(rng.choice(sorted({j for s in stray for j in (s - 1, s, s + 1) if j in per_layer})))
        trial = dict(per_layer)
        trial[k] = cover(k, rng)
        s = score(trial)
        if s <= best:
            per_layer, best = trial, s
    return flat(per_layer)


def placements_to_ldr(placements: list[Placement], title: str = "car2brick baseline") -> str:
    stems = bricknet.load_catalog().id_to_stem
    lines = [f"0 {title}", "0 Name: baseline.ldr"]
    for p in placements:
        m = p.ldraw_matrix()
        nums = " ".join(f"{v:.6g}" for v in [*m[:3, 3], *m[:3, :3].flat])
        lines.append(f"1 {p.color} {nums} {stems[p.part_id]}.dat")
    return "\n".join(lines) + "\n"


def baseline_ldr(desc: CarDescriptor, cap_tiles: bool = True) -> str:
    return placements_to_ldr(greedy_plates(desc, cap_tiles=cap_tiles))
