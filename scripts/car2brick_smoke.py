#!/usr/bin/env python3
"""End-to-end smoke test of the back half without downloads or a GPU.

1. Build a box car descriptor and its B1 baseline LDraw model (stands in for a dataset car).
2. Turn it into car2brick training text (descriptor caption + anchored path), several build orders.
3. Train a tiny Qwen3-architecture model on CPU until it memorises the sequences.
4. Generate with HFProposer + ConstrainedDecoder and report IoU / connectivity.

It checks that the training format and the generation loop agree (prompt boundary, line stopping, anchors).

    python scripts/car2brick_smoke.py --steps 600
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "car2brick"))

import bricknet  # noqa: E402
import tiny_lm  # noqa: E402
from conftest import box_car_points  # noqa: E402

from car2brick.baseline import baseline_ldr  # noqa: E402
from car2brick.decode import ConstrainedDecoder, DecodeConfig, HFProposer  # noqa: E402
from car2brick.descriptor import descriptor_from_points  # noqa: E402
from car2brick.evaluate import evaluate, poses_to_ldr  # noqa: E402
from car2brick.lego import lego_car_from_graph  # noqa: E402
from car2brick.sequence import anchored_text, order_tree  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--orders", type=int, default=1)
    ap.add_argument("--out", default=None, help="write the generated .ldr here")
    a = ap.parse_args()

    desc = descriptor_from_points(box_car_points(), colors=("red", "black"))
    graph = bricknet.parse_ldr(baseline_ldr(desc))
    lc = lego_car_from_graph(graph, component=None)
    gp = lc.grid_poses()
    rng = np.random.default_rng(0)
    caption = lc.descriptor.to_text("official")
    pairs = []
    for r in range(a.orders):
        tree, order = order_tree(graph, gp, 0, rng=rng, jitter=0.0 if r == 0 else 1.5)
        pairs.append((caption, anchored_text(tree, gp[order[0]])))

    tok = tiny_lm.build_tokenizer([c + "\n" + p for c, p in pairs] * 4, vocab_size=800)
    n_tok = len(tok(pairs[0][0] + "\n" + pairs[0][1]).input_ids)
    print(f"{len(graph.part_ids)} parts, {n_tok} tokens per example")
    model = tiny_lm.build_model(len(tok), hidden=128, layers=3)
    t = time.time()
    tiny_lm.train(model, tok, pairs, steps=a.steps, lr=3e-3, log_every=100)
    print(f"trained in {time.time() - t:.0f}s")

    proposer = HFProposer(model, tok, temperature=0.7, top_k=5, max_new_tokens=48)
    dec = ConstrainedDecoder(proposer, config=DecodeConfig(candidates=4, max_parts=200))
    t = time.time()
    res = dec.run(lc.descriptor, caption=caption)
    ev = evaluate(res.part_ids, res.poses, lc.descriptor, colors=res.colors, collisions=False)
    print(f"generated in {time.time() - t:.0f}s: {res.stats}")
    print(f"eval: {ev}")
    print("exact reproduction:", res.text in [p for _, p in pairs])
    if a.out:
        Path(a.out).write_text(poses_to_ldr(res.part_ids, res.colors, res.poses, "smoke"))


if __name__ == "__main__":
    main()
