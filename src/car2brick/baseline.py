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


def _cover_layer(mask: np.ndarray, prefer_x: bool, lib) -> list[tuple[int, int, int, int, int]]:
    """Greedy rectangle cover of one 2D layer. Returns (part_id, i0, j0, di, dj)."""
    sizes = sorted(lib, key=lambda s: -s[0] * s[1])
    free = mask.copy()
    out = []
    for i, j in np.argwhere(mask):
        if not free[i, j]:
            continue
        for short, long_ in sizes:
            dims = [(long_, short), (short, long_)] if prefer_x else [(short, long_), (long_, short)]
            for di, dj in dims:
                block = free[i : i + di, j : j + dj]
                if block.shape == (di, dj) and block.all():
                    free[i : i + di, j : j + dj] = False
                    out.append((lib[(short, long_)], int(i), int(j), di, dj))
                    break
            else:
                continue
            break
    return out


def greedy_plates(desc: CarDescriptor, color: int | None = None, cap_tiles: bool = True,
                  thr: float = 0.55) -> list[Placement]:
    if color is None:
        cat = bricknet.load_catalog()
        color = cat.color_to_code.get(desc.colors[0], 4) if desc.colors else 4
    solid = desc.symmetrized().solid(thr)
    lib = plate_library()
    tiles = tile_library() if cap_tiles else {}
    placements = []
    for k in range(solid.shape[2]):
        layer = solid[:, :, k]
        if not layer.any():
            continue
        above = solid[:, :, k + 1] if k + 1 < solid.shape[2] else np.zeros_like(layer)
        exposed = layer & ~above if tiles else np.zeros_like(layer)
        # B1: cells with nothing on top get tiles (studless look); the rest are plates
        for pid, i0, j0, di, dj in _cover_layer(exposed, prefer_x=(k % 2 == 0), lib=tiles):
            placements.append(Placement(pid, color, i0, j0, k, di, dj))
        for pid, i0, j0, di, dj in _cover_layer(layer & ~exposed, prefer_x=(k % 2 == 0), lib=lib):
            placements.append(Placement(pid, color, i0, j0, k, di, dj))
    return placements


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
