"""B3 module-wise assembly: module labels, retrieval across library cars, one connected result."""

import numpy as np

from car2brick.assemble import MODULES, ModuleLibrary, assemble, describe, index_car
from car2brick.evaluate import evaluate_ldr, poses_to_ldr
from car2brick.geometry import PartShapes
from car2brick.lego import lego_car_from_ldr


def test_assemble_from_a_copy(baseline_car):
    shapes = PartShapes()
    lc = lego_car_from_ldr(baseline_car, shapes=shapes)
    ldr = poses_to_ldr(lc.part_ids, lc.colors, lc.grid_poses())
    cars = [index_car(s, lc.descriptor, ldr, shapes) for s in ("a", "b")]
    labels = {MODULES[k] for k in cars[0].labels[0]}
    # (the B0 car has no wheel parts; graph refinement may merge a small end into its neighbour)
    assert "chassis" in labels and len(labels & {"cabin", "end_a", "end_b"}) >= 2
    assert len(cars[0].labels[1]) == len(lc.part_ids)  # driving the other way: every part labelled too
    # a module follows the connection graph: one piece
    import bricknet

    g = bricknet.parse_ldr(ldr)
    for m in labels - {"wheels_a", "wheels_b", "glass"}:
        sel = set(np.nonzero(cars[0].labels[0] == MODULES.index(m))[0].tolist())
        reach, stack = set(), [min(sel)]
        while stack:
            u = stack.pop()
            reach.add(u)
            stack += [int(v) for a, b in zip(g.edges["a"], g.edges["b"]) for x, v in ((a, b), (b, a))
                      if x == u and v in sel and v not in reach]
        assert reach == sel, m
    lib = ModuleLibrary(cars)
    asm = assemble(lib, lc.descriptor, shapes, exclude={"a"}, dedup=0.0)
    assert {d["source"] for d in describe(lib, asm)} == {"b"}  # the excluded car is never used
    out = poses_to_ldr(asm.part_ids, asm.colors, asm.poses)
    ev = evaluate_ldr(out, lc.descriptor, shapes)
    assert ev["components"] == 1 and ev["iou_filled"] > 0.8
    assert len(asm.part_ids) <= len(lc.part_ids) and np.all(np.isfinite(np.array(asm.poses)))
    # a library with only the target's own copy (deduplicated away) gives nothing
    assert not assemble(ModuleLibrary(cars[:1]), lc.descriptor, shapes, exclude=set(), dedup=0.9).part_ids


def test_swap_assemble_keeps_one_piece(baseline_car):
    from car2brick.assemble import swap_assemble

    shapes = PartShapes()
    lc = lego_car_from_ldr(baseline_car, shapes=shapes)
    ldr = poses_to_ldr(lc.part_ids, lc.colors, lc.grid_poses())
    lib = ModuleLibrary([index_car(s, lc.descriptor, ldr, shapes) for s in ("a", "b")])
    asm = swap_assemble(lib, lc.descriptor, shapes, exclude={"a"}, dedup=0.0, bases=2)
    log = describe(lib, asm)
    assert log[0]["module"] == "base" and log[0]["source"] == "b"
    ev = evaluate_ldr(poses_to_ldr(asm.part_ids, asm.colors, asm.poses), lc.descriptor, shapes)
    assert ev["components"] == 1 and ev["iou_filled"] > 0.8
