"""Scoring generated / baseline models against a target descriptor, and LDraw export."""

from __future__ import annotations

from collections import Counter

import numpy as np

import bricknet

from .descriptor import CarDescriptor, descriptor_from_points
from .geometry import PartShapes
from .lego import color_name, is_wheel
from .units import LD_FROM_CAR, homogeneous


def poses_to_ldr(part_ids, colors, grid_poses, title: str = "car2brick") -> str:
    """Grid-frame poses -> LDraw world (``LD = (x, -z, y)``)."""
    stems = bricknet.load_catalog().id_to_stem
    to_ld = homogeneous(LD_FROM_CAR)
    lines = [f"0 {title}", "0 Name: car2brick.ldr"]
    for pid, col, pose in zip(part_ids, colors, grid_poses):
        m = to_ld @ np.asarray(pose)
        nums = " ".join(f"{round(v, 4) + 0.0:.10g}" for v in [*m[:3, 3], *m[:3, :3].flat])
        lines.append(f"1 {col} {nums} {stems[pid]}.dat")
    return "\n".join(lines) + "\n"


def grid_occupancy(part_ids, grid_poses, like: CarDescriptor, shapes: PartShapes) -> CarDescriptor:
    """Occupancy of a model *in the target's grid* (no re-centering), for IoU against ``like``."""
    pts = shapes.interior_points(part_ids, grid_poses)
    spec = like.spec
    sub = np.array(spec.sub)
    n = np.array(spec.shape) * sub
    idx = np.floor(pts / (spec.cell / sub)).astype(int)
    idx = idx[((idx >= 0) & (idx < n)).all(1)]
    fine = np.zeros(n, dtype=bool)
    fine[tuple(idx.T)] = True
    occ = fine.reshape(spec.length, sub[0], spec.width, sub[1], spec.height, sub[2]).mean((1, 3, 5))
    return CarDescriptor(occ.astype(np.float32), spec)


def recolor(part_ids, colors, target: tuple[str, ...]) -> list[int]:
    """Map the generated model's most-used body colours onto the target's main / second colours."""
    if not target:
        return list(colors)
    cat = bricknet.load_catalog()
    use = Counter(c for p, c in zip(part_ids, colors) if not is_wheel(p) and not color_name(c).startswith("trans"))
    mapping = {}
    for (src, _), dst in zip(use.most_common(len(target)), target):
        if dst in cat.color_to_code:
            mapping[src] = cat.color_to_code[dst]
    return [mapping.get(c, c) for c in colors]


def evaluate(part_ids, grid_poses, target: CarDescriptor, shapes: PartShapes | None = None,
             collisions: bool = True, colors=None) -> dict:
    shapes = shapes or PartShapes()
    out = {"parts": len(part_ids)}
    if not part_ids:
        return out | {"iou": 0.0, "components": 0}
    gen = grid_occupancy(part_ids, grid_poses, target, shapes)
    out["iou"] = gen.iou(target)
    out["iou_filled"] = gen.iou(target, filled=True)
    out["size"] = gen.size()
    graph = bricknet.parse_ldr(poses_to_ldr(part_ids, colors or [4] * len(part_ids), grid_poses))
    out["components"] = len(graph.components)
    out["largest_component"] = len(graph.components[0]) / len(part_ids)
    if collisions:
        try:
            from bricknet.collision import colliding_pairs

            pairs = colliding_pairs(list(part_ids), list(grid_poses))
            out["collisions"] = None if pairs is None else len(pairs)
        except Exception:
            out["collisions"] = None
    return out


def evaluate_ldr(ldr_text: str, target: CarDescriptor, shapes: PartShapes | None = None) -> dict:
    """Score an arbitrary LDraw car (e.g. a baseline) by re-deriving its own descriptor and comparing shapes
    after alignment (both centred the same way)."""
    from .lego import lego_car_from_ldr

    lc = lego_car_from_ldr(ldr_text, shapes=shapes)
    graph = bricknet.parse_ldr(ldr_text)
    return {
        "parts": len(graph.part_ids),
        "iou": lc.descriptor.iou(target),
        "iou_filled": lc.descriptor.iou(target, filled=True),
        "size": lc.descriptor.size(),
        "components": len(graph.components),
    }


__all__ = ["poses_to_ldr", "grid_occupancy", "recolor", "evaluate", "evaluate_ldr", "descriptor_from_points"]
