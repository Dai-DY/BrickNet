"""Back half: sequence format, constrained decoding, dataset building."""

import json

import numpy as np
import pytest

import bricknet
from car2brick.decode import ConstrainedDecoder, DecodeConfig
from car2brick.evaluate import evaluate, poses_to_ldr, recolor
from car2brick.ldraw_io import flatten
from car2brick.lego import lego_car_from_graph
from car2brick.sequence import anchored_text, order_tree, parse_root, prompt, realize, strip_anchors
from car2brick.units import AXIS_ROTATIONS, rotation_code


@pytest.fixture(scope="module")
def car(baseline_car):
    g = bricknet.parse_ldr(baseline_car)
    lc = lego_car_from_graph(g, component=None)
    gp = lc.grid_poses()
    tree, order = order_tree(g, gp, 0)
    return g, lc, gp, tree, order, anchored_text(tree, gp[order[0]])


def test_rotation_codes():
    assert len(AXIS_ROTATIONS) == 24
    for k, R in enumerate(AXIS_ROTATIONS):
        assert rotation_code(R) == k
    assert rotation_code(np.array([[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1]])) is None


def test_anchored_text_roundtrip(car):
    g, lc, gp, tree, order, text = car
    lines = text.splitlines()
    assert lines[1].startswith("@root ") and parse_root(lines[1]) is not None
    assert sum(ln.startswith("@ ") for ln in lines) == len(g.part_ids) - 1
    parsed = bricknet.parse_sample(strip_anchors(text))
    assert parsed.error is None and len(parsed.tree.parts) == len(g.part_ids)
    poses = realize(parsed.tree, gp[order[0]])
    assert np.abs(poses[:, :3, 3] - gp[order][:, :3, 3]).max() < 1e-6


def test_chassis_first_order(car):
    g, lc, gp, tree, order, text = car
    z = gp[order][:, 2, 3]
    # the walk may only go up once its frontier is exhausted, so the first parts are on the lowest layer
    assert z[0] == z.min() and np.mean(z[: len(z) // 4]) < np.mean(z[-len(z) // 4 :])


class Scripted:
    """Replays a known sequence; optionally offers a bad candidate first."""

    def __init__(self, text, bad_root=False):
        L = text.rstrip("\n").split("\n")
        self.steps = [(L[0], L[1])] + [(L[2 + 3 * i], L[3 + 3 * i]) for i in range((len(L) - 2) // 3)]
        self.bad_root = bad_root

    def __call__(self, prefix, n, n_lines):
        k = sum(1 for ln in prefix.split("\n") if ln.startswith("@"))
        if k >= len(self.steps):
            return [None] * n
        node, nxt = self.steps[k]
        cands = ["not a part | red\nbad"]
        if k == 0 and self.bad_root:
            cands.append(node + "\n@root 9000 9000 9000 r0")  # far outside the target
        return cands + [node + "\n" + nxt]


def test_decoder_reproduces_and_rejects(car):
    g, lc, gp, tree, order, text = car
    res = ConstrainedDecoder(Scripted(text, bad_root=True), config=DecodeConfig(candidates=3)).run(lc.descriptor)
    assert res.text == text
    assert res.stats["stop"] == "eos" and res.stats["outside"] >= 1 and res.stats["parse_fail"] >= 1
    ev = evaluate(res.part_ids, res.poses, lc.descriptor, colors=res.colors, collisions=False)
    assert ev["iou"] == 1.0 and ev["components"] == 1
    ldr = poses_to_ldr(res.part_ids, res.colors, res.poses)
    assert len(bricknet.parse_ldr(ldr).components) == 1


def test_recolor():
    cat = bricknet.load_catalog()
    plate = cat.name_to_id["plate 2x4"]
    out = recolor([plate, plate, plate], [4, 4, 0], ("blue", "white"))
    assert out == [cat.color_to_code["blue"]] * 2 + [cat.color_to_code["white"]]


MPD = """0 FILE main.ldr
0 Name: main.ldr
1 16 0 0 0 1 0 0 0 1 0 0 0 1 base.ldr
0 STEP
1 4 0 -8 0 1 0 0 0 1 0 0 0 1 3020.dat
1 4 0 -16 0 1 0 0 0 1 0 0 0 1 notapart.dat
0 NOFILE
0 FILE base.ldr
1 16 0 0 0 1 0 0 0 1 0 0 0 1 3020.dat
1 0 40 8 0 1 0 0 0 1 0 0 0 1 3020.dat
0 NOFILE
"""


def test_flatten_mpd():
    flat = flatten(MPD)
    assert len(flat.lines) == 3 and flat.steps == [0, 0, 1]
    assert flat.skipped == {"notapart": 1}
    assert flat.lines[0].split()[1] == "71"  # main colour 16 resolved
    assert len(bricknet.parse_ldr(flat.text).components) == 1


def test_build_data(tmp_path, baseline_car):
    import argparse

    from car2brick.dataset import add_args, build

    src = tmp_path / "cars"
    src.mkdir()
    (src / "car_a.ldr").write_text(baseline_car)
    (src / "car_b.ldr").write_text(baseline_car)
    (tmp_path / "official.txt").write_text("car_a\n")
    ap = argparse.ArgumentParser()
    add_args(ap)
    a = ap.parse_args(["--ldr-dir", str(src), "--out", str(tmp_path / "data"), "--min-wheels", "0",
                       "--official", str(tmp_path / "official.txt"), "--orders", "2", "--augs", "2",
                       "--val-frac", "0"])
    counts = build(a)
    assert counts["kept"] == 2 and counts["examples"] == 8
    paths = [json.loads(x) for x in (tmp_path / "data" / "paths_train.jsonl").read_text().splitlines()]
    caps = [json.loads(x) for x in (tmp_path / "data" / "captions_train.jsonl").read_text().splitlines()]
    assert len(paths) == len(caps) == 8
    # BrickNet's sft_pairs: k-th caption of a source pairs with sample_index k
    by_src = {}
    for c in caps:
        by_src.setdefault(c["id"], []).append(c["caption"])
    for p in paths:
        cap = by_src[p["source"]][p["sample_index"]]
        assert cap.startswith("<official>" if p["source"] == "car_a" else "<moc8>")
        assert bricknet.parse_sample(strip_anchors(p["path"])).error is None
    assert prompt("x") == "x\n"
    assert (tmp_path / "data" / "descriptors" / "car_a.npz").exists()
