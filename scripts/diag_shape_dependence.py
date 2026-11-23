#!/usr/bin/env python3
"""Does the model look at the descriptor? Teacher-forced loss with the right descriptor vs another car's.

For validation sequences, compute the per-line loss of the path given (a) its own descriptor and (b) the
descriptor of a different car (a fixed derangement). Losses are grouped by part ordinal (root, 1-2, 3-9, 10-29,
30+) and line kind (target / anchor, node, edge). A model that follows the shape has a clearly higher loss with the
wrong descriptor; if a group shows no gap, generation in that range ignores the shape.

Works for both layouts: anchors after the part (``@`` line after node + edge) and target-first (``@`` before).

    python scripts/diag_shape_dependence.py --model Qwen/Qwen3-0.6B --lora kulits/BrickNet-0.6B-PT \
        --lora outputs/runs/c2b_pilot_06b/checkpoint-350 --data outputs/data/c2b_v1 --n 64
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

GROUPS = ((0, 0, "root"), (1, 2, "1-2"), (3, 9, "3-9"), (10, 29, "10-29"), (30, 10**9, "30+"))


def group_of(k: int) -> str:
    return next(name for lo, hi, name in GROUPS if lo <= k <= hi)


def line_meta(path: str) -> list[tuple[int, str]]:
    """(part ordinal, kind) per line of a path text."""
    out, part = [], 0
    lines = path.rstrip("\n").split("\n")
    target_first = len(lines) > 2 and lines[2].startswith("@ ")
    for i, ln in enumerate(lines):
        if i < 2:
            out.append((0, "root"))
            continue
        j = i - 2
        part = j // 3 + 1
        kind = ("target", "node", "edge")[j % 3] if target_first else ("node", "edge", "anchor")[j % 3]
        out.append((part, kind))
    return out


@torch.inference_mode()
def line_losses(model, tok, caption: str, path: str, max_len: int) -> list[float]:
    nl = tok("\n", add_special_tokens=False).input_ids
    ctx = tok(caption, add_special_tokens=False).input_ids + nl
    ids, spans = list(ctx), []
    for ln in path.rstrip("\n").split("\n"):
        t = tok(ln + "\n", add_special_tokens=False).input_ids
        spans.append((len(ids), len(ids) + len(t)))
        ids += t
    ids = ids[:max_len]
    x = torch.tensor([ids], device=model.device)
    hidden = model.model(input_ids=x).last_hidden_state[0, :-1]
    tgt = torch.tensor(ids[1:], device=model.device)
    losses = []
    for s in range(0, len(tgt), 1024):  # full-vocabulary logits of a long sequence would take several GB at once
        logits = model.lm_head(hidden[s : s + 1024]).float()
        losses.append(torch.logsumexp(logits, -1) - logits.gather(1, tgt[s : s + 1024, None])[:, 0])
    tok_loss = torch.cat(losses).cpu().numpy()  # loss of token i+1
    out = []
    for a, b in spans:
        if b > len(ids):
            break
        out.append(float(tok_loss[a - 1 : b - 1].mean()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", action="append", default=[])
    ap.add_argument("--data", required=True, help="build-data folder (uses paths_val / captions_val)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--min-parts", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    for ad in a.lora:
        model = PeftModel.from_pretrained(model, ad).merge_and_unload()

    data = Path(a.data)
    caps: dict[str, list[str]] = defaultdict(list)
    for ln in open(data / "captions_val.jsonl"):
        r = json.loads(ln)
        caps[r["id"]].append(r["caption"])
    rows = [json.loads(ln) for ln in open(data / "paths_val.jsonl")]
    # one clean example per car (sample_index 0: first build order, unperturbed descriptor)
    rows = [r for r in rows if r["sample_index"] == 0 and r["path"].count("\n") >= 3 * a.min_parts]
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in rng.permutation(len(rows))[: a.n]]
    other = np.roll(np.arange(len(rows)), 1)  # derangement: each car gets its neighbour's descriptor

    acc = defaultdict(lambda: [[], []])  # (group, kind) -> [right losses, wrong losses]
    for i, r in enumerate(rows):
        meta = line_meta(r["path"])
        right = line_losses(model, tok, caps[r["source"]][0], r["path"], a.max_len)
        wrong = line_losses(model, tok, caps[rows[other[i]]["source"]][0], r["path"], a.max_len)
        for (k, kind), lr, lw in zip(meta, right, wrong):
            acc[(group_of(k), kind)][0].append(lr)
            acc[(group_of(k), kind)][1].append(lw)
        print(f"{i + 1}/{len(rows)}", end="\r", flush=True)
    print()
    report = []
    for _, _, g in GROUPS:
        for kind in ("root", "target", "node", "edge", "anchor"):
            if (g, kind) not in acc:
                continue
            r_, w_ = map(np.array, acc[(g, kind)])
            report.append({"parts": g, "line": kind, "n": len(r_), "loss_right": round(float(r_.mean()), 4),
                           "loss_wrong": round(float(w_.mean()), 4), "gap": round(float((w_ - r_).mean()), 4)})
            print(f"parts {g:>6s} {kind:7s} n={len(r_):6d}  right {r_.mean():.4f}  wrong {w_.mean():.4f}  "
                  f"gap {(w_ - r_).mean():+.4f}")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
