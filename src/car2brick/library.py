"""Library of LEGO cars (from BrickNet graphs) and the retrieval baseline.

Upright fix: BrickNet components are not always stored stud-up (about one in six roots lies on its side or upside
down), while the rest of car2brick assumes LDraw -Y is up. ``stud_up`` takes a majority vote over the directions of
all stud connectors (SNOT parts are a minority in any car) and ``upright`` rotates the model so that direction
becomes -Y.

Retrieval baseline: for a target descriptor, score every library car of a similar width by the IoU of their
symmetrised, column-filled solids (outer shape; LEGO cars are hollow inside) after the best shift along x (and the
better of the two driving directions), plus a small term for wheel positions; return the best car moved into the
target's grid. LEGO cannot be rescaled, so a target is only compared with cars within ``width_tol`` studs of its
own width.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import bricknet

from .descriptor import CarDescriptor, column_filled
from .lego import is_wheel
from .units import AXIS_ROTATIONS, LDU_PER_PLATE, LDU_PER_STUD

_LOCAL_STUD_UP = np.array([0.0, -1.0, 0.0])  # a stud connector rises along its frame's -Y


def stud_up(part_ids, transforms) -> np.ndarray:
    """Unit axis (LDraw world) that most studs point along."""
    conns = bricknet.load_connectors()
    votes = np.zeros(6)  # +x -x +y -y +z -z
    for pid, t in zip(part_ids, transforms):
        frames = conns.get(int(pid), {}).get(("stud", None), ())
        if not len(frames):
            continue
        rot = np.asarray(t)[:3, :3]
        for f in frames:
            d = rot @ (f[:3, :3] @ _LOCAL_STUD_UP)
            a = int(np.argmax(np.abs(d)))
            votes[2 * a + (d[a] < 0)] += 1
    k = int(np.argmax(votes))
    axis = np.zeros(3)
    axis[k // 2] = -1.0 if k % 2 else 1.0
    return axis


def upright(transforms, up: np.ndarray) -> list[np.ndarray]:
    """Rotate a model (about the origin) so that ``up`` becomes LDraw up (-Y)."""
    target = np.array([0.0, -1.0, 0.0])
    rot = next(r for r in AXIS_ROTATIONS if np.allclose(r @ up, target))
    m = np.eye(4)
    m[:3, :3] = rot
    return [m @ np.asarray(t, dtype=np.float64) for t in transforms]


def rough_size(part_ids, transforms) -> dict:
    """Cheap size from part origins (upright model): studs / plates, + wheel part count."""
    o = np.array([np.asarray(t)[:3, 3] for t in transforms])
    ext_x, ext_z = np.ptp(o[:, 0]), np.ptp(o[:, 2])
    return {"length": float(max(ext_x, ext_z) / LDU_PER_STUD + 1), "width": float(min(ext_x, ext_z) / LDU_PER_STUD + 1),
            "height": float(np.ptp(o[:, 1]) / LDU_PER_PLATE + 1), "wheels": int(sum(is_wheel(p) for p in part_ids))}


def car_like(s: dict, n_parts: int, min_wheels: int = 3) -> bool:
    """The survey's filter: 5-11 studs wide, long, car proportions (height / width 0.5-0.8 in LDU), <= 600 parts."""
    w = s["width"]
    return (s["wheels"] >= min_wheels and 20 <= n_parts <= 600 and 5 <= round(w) <= 11 and s["length"] >= 1.5 * w
            and 0.5 <= s["height"] * LDU_PER_PLATE / (w * LDU_PER_STUD) <= 0.8)


# ---- library ------------------------------------------------------------------------------------------------------
@dataclass
class LibraryCar:
    source: str
    width: int
    descriptor: CarDescriptor
    ldr_path: Path
    meta: dict

    def ldr(self) -> str:
        return self.ldr_path.read_text()


class CarLibrary:
    """``<dir>/index.jsonl`` + ``<dir>/desc/<source>.npz`` + ``<dir>/ldr/<source>.ldr`` (grid frame)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.cars: list[LibraryCar] = []
        for line in open(self.root / "index.jsonl"):
            m = json.loads(line)
            d = CarDescriptor.load(self.root / "desc" / f"{m['source']}.npz")
            self.cars.append(LibraryCar(m["source"], int(m["size"][1]), d, self.root / "ldr" / f"{m['source']}.ldr", m))
        self._solid = [column_filled(c.descriptor.symmetrized().solid()) for c in self.cars]

    def __len__(self):
        return len(self.cars)

    def retrieve(self, target: CarDescriptor, k: int = 5, width_tol: int = 1, exclude: set[str] = frozenset(),
                 wheel_weight: float = 0.02) -> list[dict]:
        """Top-k matches: {"car", "score", "iou", "shift", "flip"}; shift / flip map the car into the target grid."""
        tgt = column_filled(target.symmetrized().solid())  # outer shapes: LEGO cars are hollow inside
        tw = target.size()[1]
        twh = np.array(sorted(target.wheels)) if len(target.wheels) == 2 else None
        out = []
        for car, sol in zip(self.cars, self._solid):
            if car.source in exclude or abs(car.width - tw) > width_tol:
                continue
            best = None
            for flip in (False, True):
                s = sol[::-1] if flip else sol
                for shift in range(-4, 5):
                    m = np.roll(s, shift, axis=0)
                    inter = np.logical_and(m, tgt).sum()
                    union = np.logical_or(m, tgt).sum()
                    iou = inter / max(union, 1)
                    score = iou
                    if twh is not None and len(car.descriptor.wheels) == 2:
                        L = s.shape[0]
                        w = np.array(sorted(car.descriptor.wheels))
                        w = np.sort(L - w) if flip else w
                        score -= wheel_weight * float(np.abs(w + shift - twh).sum())
                    if best is None or score > best["score"]:
                        best = {"car": car, "score": float(score), "iou": float(iou), "shift": shift, "flip": flip}
            out.append(best)
        out.sort(key=lambda r: -r["score"])
        return out[:k]


def place(match: dict, target: CarDescriptor) -> tuple[list[int], list[int], np.ndarray]:
    """The matched car's (part_ids, colors, grid poses) in the target's grid frame."""
    car: LibraryCar = match["car"]
    g = bricknet.parse_ldr(car.ldr())
    from .units import CAR_FROM_LD, homogeneous

    poses = np.array([homogeneous(CAR_FROM_LD) @ np.asarray(t) for t in g.transforms])  # LDraw -> grid frame
    spec = car.descriptor.spec
    L = spec.length * LDU_PER_STUD
    if match["flip"]:  # drive the other way: rotate 180 deg about the vertical axis through the grid centre
        flip = np.diag([-1.0, -1.0, 1.0, 1.0])
        flip[0, 3], flip[1, 3] = L, spec.width * LDU_PER_STUD
        poses = np.array([flip @ p for p in poses])
    poses[:, 0, 3] += match["shift"] * LDU_PER_STUD
    return list(g.part_ids), list(g.colors), poses
