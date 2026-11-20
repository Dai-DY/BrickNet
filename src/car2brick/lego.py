"""LEGO side of the bridge: LDraw models / BrickNet graphs -> car frame -> CarDescriptor."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

import bricknet

from .descriptor import CarDescriptor, GridSpec, descriptor_from_points
from .geometry import PartShapes
from .units import homogeneous

WHEEL_RE = re.compile(r"\b(wheel|tyre|tire|rim|hub)\b")
# car = (X, Z, -Y) when the length runs along LDraw X; car = (Z, -X, -Y) when it runs along LDraw Z
_CAR_FROM_LD_X = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
_CAR_FROM_LD_Z = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])


@dataclass
class LegoCar:
    part_ids: tuple[int, ...]
    colors: tuple[int, ...]
    transforms: list[np.ndarray]  # LDraw world, LDU
    descriptor: CarDescriptor
    grid_from_ld: np.ndarray  # 4x4: LDraw world -> descriptor grid frame (LDU, z up, origin at grid corner)

    def grid_poses(self) -> np.ndarray:
        return np.array([self.grid_from_ld @ t for t in self.transforms])


def is_wheel(part_id: int) -> bool:
    return bool(WHEEL_RE.search(bricknet.load_catalog().id_to_name[part_id]))


def color_name(code: int) -> str:
    return bricknet.load_catalog().code_to_color.get(int(code), str(code))


def dominant_colors(part_ids, colors, weights, k: int = 2) -> tuple[str, ...]:
    """Most-used colours by volume, ignoring wheels and transparent parts."""
    cnt: Counter = Counter()
    for pid, col, w in zip(part_ids, colors, weights):
        name = color_name(col)
        if is_wheel(pid) or name.startswith("trans"):
            continue
        cnt[name] += w
    return tuple(c for c, _ in cnt.most_common(k))


def car_frame_from_ld(points_ld: np.ndarray) -> np.ndarray:
    """3x3 rotation LDraw world -> car frame: z = -Y (LDraw is -Y up); x = the longer horizontal axis."""
    ext_x, ext_z = np.ptp(points_ld[:, 0]), np.ptp(points_ld[:, 2])
    return _CAR_FROM_LD_X if ext_x >= ext_z else _CAR_FROM_LD_Z


def lego_car(part_ids, colors, transforms, shapes: PartShapes | None = None, spec: GridSpec | None = None) -> LegoCar:
    shapes = shapes or PartShapes()
    transforms = [np.asarray(t, dtype=np.float64) for t in transforms]
    per_part = [shapes.interior_points([pid], [t]) for pid, t in zip(part_ids, transforms)]
    pts_ld = np.concatenate(per_part)
    rot = car_frame_from_ld(pts_ld)
    pts_car = pts_ld @ rot.T
    # semantic labels: transparent parts are glass; large glass -> window (2), small -> light (3)
    labels = []
    for pid, col, p in zip(part_ids, colors, per_part):
        trans = color_name(col).startswith("trans")
        labels.append(np.full(len(p), (2 if len(p) > 40 else 3) if trans else 1, dtype=np.uint8))
    cols = dominant_colors(part_ids, colors, [len(p) for p in per_part])
    desc = descriptor_from_points(pts_car, spec, colors=cols, point_labels=np.concatenate(labels))
    grid_from_ld = homogeneous(np.eye(3), -desc.origin) @ homogeneous(rot)
    return LegoCar(tuple(part_ids), tuple(colors), transforms, desc, grid_from_ld)


def lego_car_from_graph(graph: bricknet.Graph, component: int | None = 0, **kw) -> LegoCar:
    """Uses stored transforms when the graph is rooted, otherwise realises the component with decode_graph."""
    transforms = graph.transforms if graph.transforms is not None else bricknet.decode_graph(graph)
    idx = list(range(len(graph.part_ids))) if component is None else sorted(graph.components[component])
    return lego_car(
        [graph.part_ids[i] for i in idx], [graph.colors[i] for i in idx], [transforms[i] for i in idx], **kw
    )


def lego_car_from_ldr(text: str, **kw) -> LegoCar:
    return lego_car_from_graph(bricknet.parse_ldr(text), component=None, **kw)


def vehicle_stats(graph: bricknet.Graph, shapes: PartShapes | None = None) -> dict:
    """Cheap filter features for picking cars out of BrickNet: wheel count, part count, width/length in studs."""
    comp = sorted(graph.components[0]) if graph.components else []
    pids = [graph.part_ids[i] for i in comp]
    n_wheels = sum(is_wheel(p) for p in pids)
    out = {"parts": len(pids), "wheels": n_wheels, "width": 0.0, "length": 0.0, "height": 0.0}
    if n_wheels < 3 or len(pids) < 20:
        return out
    transforms = graph.transforms if graph.transforms is not None else bricknet.decode_graph(graph)
    origins = np.array([transforms[i][:3, 3] for i in comp])
    ext_x, ext_z = np.ptp(origins[:, 0]), np.ptp(origins[:, 2])
    out["length"] = float(max(ext_x, ext_z) / 20.0 + 1)
    out["width"] = float(min(ext_x, ext_z) / 20.0 + 1)
    out["height"] = float(np.ptp(origins[:, 1]) / 8.0 + 1)
    return out
