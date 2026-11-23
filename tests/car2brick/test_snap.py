"""Target-first sequences, snap enumeration and the KV-cached snap decoder."""

import numpy as np
import pytest

import bricknet
from car2brick.lego import lego_car_from_graph
from car2brick.sequence import anchored_text, order_tree, parse_anchor, strip_anchors
from car2brick.snapdecode import SnapConfig, SnapDecoder


@pytest.fixture(scope="module")
def gold(baseline_car):
    g = bricknet.parse_ldr(baseline_car)
    lc = lego_car_from_graph(g, component=None)
    gp = lc.grid_poses()
    tree, order = order_tree(g, gp, 0)
    return lc, gp[order], anchored_text(tree, gp[order[0]], target_first=True)


class ScriptedLM:
    """Replays a gold target-first text: proposes its next line, scores its next line highest."""

    def __init__(self, gold_text: str, wrong_first: bool = False):
        self.gold = gold_text.rstrip("\n").split("\n")
        self.wrong_first = wrong_first  # propose an off-shape target once per step before the right one
        self._tried = set()

    def reset(self, head):
        self.text = ""

    def append(self, text):
        self.text += text

    def checkpoint(self):
        return self.text

    def restore(self, cp):
        self.text = cp

    def _next(self):
        done = self.text.count("\n")
        assert self.text == "".join(line + "\n" for line in self.gold[:done]), "decoder left the gold path"
        return done

    def sample_line(self):
        done = self._next()
        if done >= len(self.gold):
            return None
        line = self.gold[done]
        if self.wrong_first and line.startswith("@ ") and done not in self._tried:
            self._tried.add(done)
            return "@ 0 0 30"  # outside the shape
        return line

    def choose(self, lines, allow_eos=False, greedy=False):
        done = self._next()
        if done >= len(self.gold):
            return None if allow_eos else 0
        return lines.index(self.gold[done]) if self.gold[done] in lines else 0


def test_target_first_text(gold):
    _, poses, text = gold
    lines = text.rstrip("\n").split("\n")
    assert lines[1].startswith("@root") and lines[2].startswith("@ ")  # target, then node, then edge
    assert parse_anchor(lines[2]) is not None and parse_anchor("b plate 2x4 | red") is None
    assert bricknet.parse_sample(strip_anchors(text)).error is None


@pytest.mark.parametrize("wrong_first", [False, True])
def test_snap_decoder_reproduces_the_model(gold, wrong_first):
    lc, poses, text = gold
    lm = ScriptedLM(text, wrong_first)
    res = SnapDecoder(lm, config=SnapConfig(collisions=False, greedy_edges=True)).run(lc.descriptor)
    assert res.text == text  # every connection line was among the enumerated candidates
    assert np.allclose(np.array(res.poses), poses, atol=1e-6)
    assert res.stats["stop"] == "eos" and res.stats["target_out"] == (len(poses) - 1 if wrong_first else 0)


def test_cached_lm_matches_full_forward():
    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, str(Path(__file__).parent))
    import tiny_lm

    from car2brick.snapdecode import CachedLM

    text = "car 20x8x12\nwheels 9.5 22.5\na plate 2x8 | red\n@root 160 80 8 r0\n@ 7 4 2\n"
    tok = tiny_lm.build_tokenizer([text] * 20, vocab_size=300)
    model = tiny_lm.build_model(len(tok))
    lm = CachedLM(model, tok, temperature=1.0)
    head, rest = "car 20x8x12\nwheels 9.5 22.5\n", ["a plate 2x8 | red", "@root 160 80 8 r0", "@ 7 4 2"]
    lm.reset(head)
    lm.append(rest[0] + "\n")
    n = lm.n
    lm.sample_line()
    assert lm.n == n  # sampling leaves the cache as it was
    got = lm.score(rest[1:2])[0]
    # reference: one forward pass over the whole text
    ids = tok(head + rest[0] + "\n" + rest[1] + "\n", add_special_tokens=False).input_ids
    ctx = len(tok(head + rest[0] + "\n", add_special_tokens=False).input_ids)
    with torch.inference_mode():
        lp = torch.log_softmax(model(torch.tensor([ids])).logits[0].float(), -1)
    ref = float(sum(lp[i - 1, ids[i]] for i in range(ctx, len(ids))))
    assert abs(got - ref) < 1e-3
    lm.append(rest[1] + "\n")
    assert lm.score(rest[2:])[0] < 0


def test_cached_lm_choose_samples_the_restricted_trie():
    """``choose`` normalizes over the allowed tokens at each branch of the lines' prefix trie (constrained
    decoding); compare its frequencies with that distribution computed from full forward passes."""
    import sys
    from collections import Counter
    from pathlib import Path

    import torch

    sys.path.insert(0, str(Path(__file__).parent))
    import tiny_lm

    from car2brick.snapdecode import CachedLM

    torch.manual_seed(0)
    lines = ["@ 7 4 2", "@ 7 4 3", "@ 12 1 0", "a stud hole b stud a 90"]
    tok = tiny_lm.build_tokenizer(["\n".join(lines) + "\n"] * 20, vocab_size=300)
    model = tiny_lm.build_model(len(tok))
    lm = CachedLM(model, tok, temperature=1.0)
    head = "car 20x8x12\n"
    lm.reset(head)
    n = lm.n

    ctx = tok(head, add_special_tokens=False).input_ids
    enc = [tok(ln + "\n", add_special_tokens=False).input_ids for ln in lines]
    expect = []
    for ids in enc:
        with torch.inference_mode():
            logits = model(torch.tensor([ctx + ids])).logits[0, len(ctx) - 1 :].float()
        p = 1.0
        for pos, t in enumerate(ids):
            branch = sorted({e[pos] for e in enc if len(e) > pos and e[:pos] == ids[:pos]})
            if len(branch) > 1:
                p *= float(torch.softmax(logits[pos, branch], 0)[branch.index(t)])
            if sum(e[: pos + 1] == ids[: pos + 1] for e in enc) == 1:
                break
        expect.append(p)
    assert abs(sum(expect) - 1) < 1e-5

    counts = Counter(lm.choose(lines) for _ in range(600))
    assert lm.n == n  # the cache is left as it was
    assert np.abs(np.array([counts[i] for i in range(len(lines))]) / 600 - expect).max() < 0.08
    assert lm.choose(lines, greedy=True) is not None
