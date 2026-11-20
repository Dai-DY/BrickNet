"""Training/generation text format: BrickNet path text + car2brick additions.

A training example is ``caption + "\\n" + path`` exactly as in BrickNet's SFT recipe (``scripts/train.py sft``), where

* caption = ``CarDescriptor.to_text(style)``: style tag (``<official>`` / ``<moc8>`` / ``<other>``), size, wheels,
  colours and the half-width height map;
* path    = BrickNet path text with one extra *anchor* line after every part:

      a plate 2x8 | red
      @root 140 80 8 r0          <- root part: grid-frame origin (LDU) + axis-aligned rotation code
      b plate 2x4 | red
      a stud stud c hole a 0     <- BrickNet edge line (unchanged)
      @ 7 4 2                    <- the new part's grid cell (x stud, y stud, z plate) of its origin

Anchors ground the relative (connection-based) sequence in the descriptor's grid, so the model can compare
"where am I" with "what shape is wanted here". At generation time anchors are *computed* by the decoder and
appended, never sampled, so they are always correct. ``strip_anchors`` recovers plain BrickNet path text.

Build order: a priority walk over the connection graph that grows the model bottom-up and front-to-back
(chassis first), optionally following the instruction ``STEP`` order of the source file.
"""

from __future__ import annotations

import heapq
import itertools
import re

import numpy as np

import bricknet
from bricknet.graph import _adjacency, _build_tree

from .units import AXIS_ROTATIONS, LDU_PER_PLATE, LDU_PER_STUD, rotation_code

ROOT_RE = re.compile(r"^@root (-?\d+) (-?\d+) (-?\d+) r(\d+)$")
ANCHOR_RE = re.compile(r"^@ (-?\d+) (-?\d+) (-?\d+)$")


def order_tree(
    graph: bricknet.Graph,
    grid_poses: np.ndarray,
    component: int = 0,
    *,
    steps: list[int] | None = None,
    rng: np.random.Generator | None = None,
    jitter: float = 0.0,
) -> tuple[bricknet.Tree, list[int]]:
    """Spanning tree of one component in chassis-first order; also returns the graph node visit order
    (``order[0]`` is the root, whose grid pose is the ``root_pose`` for ``anchored_text``).

    Priority of a part = (instruction step, plate layer of its origin, stud x, stud y) with optional random jitter
    (``jitter`` in cells) for augmentation. The root is the first part in that order whose rotation is
    axis-aligned in the grid frame (so it can be written as an ``@root`` line).
    """
    comp = set(graph.components[component])
    adj = _adjacency(graph.edges, comp)
    rng = rng or np.random.default_rng(0)

    def key(n: int):
        p = grid_poses[n][:3, 3]
        noise = rng.uniform(-jitter, jitter, 3) if jitter else np.zeros(3)
        return (
            steps[n] if steps is not None else 0,
            round(p[2] / LDU_PER_PLATE + noise[2]),
            round(p[0] / LDU_PER_STUD + noise[0]),
            round(p[1] / LDU_PER_STUD + noise[1]),
        )

    keys = {n: key(n) for n in comp}
    roots = sorted(comp, key=lambda n: keys[n])
    root = next((n for n in roots if rotation_code(grid_poses[n][:3, :3]) is not None), roots[0])

    order, parent_of, seen = [root], {}, {root}
    counter = itertools.count()
    heap = [(keys[v], next(counter), root, ei, fwd, v) for v, ei, fwd in adj.get(root, [])]
    heapq.heapify(heap)
    while heap:
        _, _, parent, ei, fwd, child = heapq.heappop(heap)
        if child in seen:
            continue
        seen.add(child)
        parent_of[child] = (parent, ei, fwd)
        order.append(child)
        for v, ei2, fwd2 in adj.get(child, []):
            if v not in seen:
                heapq.heappush(heap, (keys[v], next(counter), child, ei2, fwd2, v))
    return _build_tree(graph, order, parent_of), order


def realize(tree: bricknet.Tree, root_pose: np.ndarray) -> np.ndarray:
    """Poses (n, 4, 4) of a tree's parts in the frame of ``root_pose`` (from the quantized edges)."""
    rel = bricknet.decode_graph(bricknet.tree_to_graph(tree))  # root (index 0) at identity
    return np.array([root_pose @ r for r in rel])


def root_line(pose: np.ndarray) -> str:
    code = rotation_code(pose[:3, :3])
    if code is None:
        raise ValueError("root rotation is not axis-aligned")
    x, y, z = (int(round(v)) for v in pose[:3, 3])
    return f"@root {x} {y} {z} r{code}"


def anchor_line(pose: np.ndarray) -> str:
    # small epsilon: poses composed step by step and via decode_graph differ by ~1e-12 on cell boundaries
    x, y, z = pose[:3, 3] + 1e-6
    return f"@ {int(np.floor(x / LDU_PER_STUD))} {int(np.floor(y / LDU_PER_STUD))} {int(np.floor(z / LDU_PER_PLATE))}"


def parse_root(line: str) -> np.ndarray | None:
    m = ROOT_RE.match(line.strip())
    if m is None or int(m[4]) >= len(AXIS_ROTATIONS):
        return None
    pose = np.eye(4)
    pose[:3, :3] = AXIS_ROTATIONS[int(m[4])]
    pose[:3, 3] = [float(m[1]), float(m[2]), float(m[3])]
    return pose


def anchored_text(tree: bricknet.Tree, root_pose: np.ndarray) -> str:
    base = bricknet.serialize_tree(tree).rstrip("\n").split("\n")
    poses = realize(tree, root_pose)
    out = [base[0], root_line(poses[0])]
    for i in range(1, len(tree.parts)):
        out += [base[2 * i - 1], base[2 * i], anchor_line(poses[i])]
    return "\n".join(out) + "\n"


def strip_anchors(text: str) -> str:
    lines = [ln for ln in text.split("\n") if not ln.startswith("@")]
    return "\n".join(lines)


def prompt(caption: str) -> str:
    """Generation prefix, matching the SFT collator's ``caption + "\\n" + path``."""
    return caption.rstrip("\n") + "\n"
