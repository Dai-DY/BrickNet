"""Shape-constrained autoregressive decoding.

The language model proposes the next part (node line + edge line); the decoder parses it, places it by composing
the connection transform onto its parent, and accepts it only if

  * it parses (known part / colour / connector, valid mating),
  * at most ``outside_tol`` of its volume falls outside the allowed region (target shape dilated by ``margin``),
  * it does not collide with already placed parts (BrickNet colliders, when pybullet + colliders are installed).

Accepted steps get their anchor line computed and appended, so the model always sees true positions. Rejected
proposals are resampled (``candidates`` per round, up to ``patience`` rounds); generation stops when most
candidates end the sequence (EOS) or ``max_parts`` is reached.

``Proposer`` abstracts the model: ``HFProposer`` wraps a Hugging Face causal LM (BrickNet base + LoRA adapters);
tests use a scripted proposer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

import bricknet
from bricknet.core import FixedEdge, Part
from bricknet.graph import _EDGE_DT, _edge_matrix, _tree_edge_row
from bricknet.tree import ParseError, parse_edge, parse_node

from .descriptor import CarDescriptor
from .geometry import PartShapes
from .sequence import anchor_line, parse_root, prompt, root_line


class Proposer(Protocol):
    def __call__(self, prefix: str, n: int, n_lines: int) -> list[str | None]:
        """``n`` sampled continuations of ``prefix``, each cut after ``n_lines`` lines; None = ended (EOS)."""


@dataclass
class DecodeConfig:
    max_parts: int = 500
    candidates: int = 8
    patience: int = 3
    outside_tol: float = 0.15
    margin: tuple[int, int, int] = (1, 1, 2)
    collisions: bool = True
    eos_fraction: float = 0.5  # stop when at least this fraction of a round's candidates end the sequence


@dataclass
class DecodeResult:
    text: str  # anchored path text (without the prompt)
    part_ids: list[int] = field(default_factory=list)
    colors: list[int] = field(default_factory=list)
    poses: list[np.ndarray] = field(default_factory=list)  # grid frame
    stats: dict = field(default_factory=dict)


def child_relative(edge, parent_pid: int, child_pid: int) -> np.ndarray:
    """Child pose in the parent's frame for one tree edge (same maths as BrickNet's collision-free walk)."""
    row = np.array([_tree_edge_row(edge, 0, 1, parent_pid, child_pid)], dtype=_EDGE_DT)[0]
    a_is_parent = row["a"] == 0
    M = _edge_matrix(row, parent_pid if a_is_parent else child_pid, child_pid if a_is_parent else parent_pid)
    return M if a_is_parent else np.linalg.inv(M)


class _Collisions:
    def __init__(self, enabled: bool):
        self.scene = None
        if enabled:
            try:
                from bricknet import collision

                self.scene = collision.CollisionScene()
            except Exception as e:  # pybullet or colliders missing
                print(f"[car2brick] collision checks disabled: {e}")

    def ok(self, pid: int, pose: np.ndarray, parent_index: int | None, fixed: bool) -> bool:
        if self.scene is None:
            return True
        try:
            hits = self.scene.check(pid, pose)
        except (FileNotFoundError, RuntimeError):
            return True
        if fixed:  # rigid mounts may overlap their parent
            hits = [h for h in hits if h != parent_index]
        return not hits

    def add(self, pid: int, pose: np.ndarray) -> None:
        if self.scene is not None:
            try:
                self.scene.add(pid, pose)
            except (FileNotFoundError, RuntimeError):
                pass

    def close(self):
        if self.scene is not None:
            self.scene.close()


class ConstrainedDecoder:
    def __init__(self, proposer: Proposer, shapes: PartShapes | None = None, config: DecodeConfig | None = None):
        self.proposer = proposer
        self.shapes = shapes or PartShapes()
        self.cfg = config or DecodeConfig()
        self.catalog = bricknet.load_catalog()

    # ---- geometry checks ----------------------------------------------------------------------------------------
    def outside_fraction(self, pid: int, pose: np.ndarray, allowed: np.ndarray, cell: np.ndarray) -> float:
        pts = self.shapes.local_interior(pid) @ pose[:3, :3].T + pose[:3, 3]
        idx = np.floor(pts / cell).astype(int)
        inside_grid = ((idx >= 0) & (idx < np.array(allowed.shape))).all(1)
        ok = np.zeros(len(pts), dtype=bool)
        ok[inside_grid] = allowed[tuple(idx[inside_grid].T)]
        return 1.0 - ok.mean()

    # ---- main loop --------------------------------------------------------------------------------------------
    def run(self, desc: CarDescriptor, style: str | None = "official", caption: str | None = None) -> DecodeResult:
        cfg = self.cfg
        caption = caption if caption is not None else desc.to_text(style)
        head = prompt(caption)
        allowed = desc.allowed(cfg.margin)
        cell = desc.spec.cell
        res = DecodeResult(text="")
        parts: list[Part] = []
        col = _Collisions(cfg.collisions)
        stats = {"proposed": 0, "parse_fail": 0, "outside": 0, "collision": 0, "stop": "max_parts"}

        def try_root(cands):
            for c in cands:
                if c is None:
                    continue
                lines = c.split("\n")
                try:
                    part = parse_node(lines[0], self.catalog)
                except ParseError:
                    stats["parse_fail"] += 1
                    continue
                pose = parse_root(lines[1]) if len(lines) > 1 else None
                if pose is None:
                    stats["parse_fail"] += 1
                    continue
                if self.outside_fraction(part.part_id, pose, allowed, cell) > cfg.outside_tol:
                    stats["outside"] += 1
                    continue
                return part, pose, lines[0]
            return None

        def try_child(cands):
            for c in cands:
                if c is None:
                    continue
                lines = c.split("\n")
                if len(lines) < 2:
                    stats["parse_fail"] += 1
                    continue
                try:
                    part = parse_node(lines[0], self.catalog)
                    edge = parse_edge(lines[1], parts, part.part_id, self.catalog)
                    rel = child_relative(edge, parts[edge.parent].part_id, part.part_id)
                except (ParseError, KeyError, IndexError):
                    stats["parse_fail"] += 1
                    continue
                pose = res.poses[edge.parent] @ rel
                if self.outside_fraction(part.part_id, pose, allowed, cell) > cfg.outside_tol:
                    stats["outside"] += 1
                    continue
                if not col.ok(part.part_id, pose, edge.parent, isinstance(edge, FixedEdge)):
                    stats["collision"] += 1
                    continue
                return part, pose, lines[0], lines[1]
            return None

        try:
            # root: node line + @root line
            for _ in range(cfg.patience):
                cands = self.proposer(head, cfg.candidates, 2)
                stats["proposed"] += len(cands)
                got = try_root(cands)
                if got:
                    break
            else:
                stats["stop"] = "no_root"
                res.stats = stats
                return res
            part, pose, node = got
            pose_line = root_line(pose)
            parts.append(part)
            res.part_ids.append(part.part_id), res.colors.append(part.color), res.poses.append(pose)
            res.text += f"{node}\n{pose_line}\n"
            col.add(part.part_id, pose)

            while len(parts) < cfg.max_parts:
                accepted = None
                for _ in range(cfg.patience):
                    cands = self.proposer(head + res.text, cfg.candidates, 2)
                    stats["proposed"] += len(cands)
                    if sum(c is None for c in cands) >= cfg.eos_fraction * len(cands):
                        stats["stop"] = "eos"
                        break
                    accepted = try_child(cands)
                    if accepted:
                        break
                if accepted is None:
                    if stats["stop"] != "eos":
                        stats["stop"] = "stuck"
                    break
                part, pose, node, edge_line = accepted
                parts.append(part)
                res.part_ids.append(part.part_id), res.colors.append(part.color), res.poses.append(pose)
                res.text += f"{node}\n{edge_line}\n{anchor_line(pose)}\n"
                col.add(part.part_id, pose)
        finally:
            col.close()
        stats["parts"] = len(parts)
        res.stats = stats
        return res


class HFProposer:
    """Sampling proposer over a Hugging Face causal LM (e.g. Qwen3-0.6B + BrickNet PT + car2brick LoRA).

    Each call re-encodes the prefix (simple and robust; a KV-cache across steps is a possible speed-up).
    """

    def __init__(self, model, tokenizer, *, temperature: float = 1.0, top_k: int = 20, top_p: float = 0.95,
                 max_new_tokens: int = 128):
        import torch

        self.torch = torch
        self.model = model.eval()
        self.tok = tokenizer
        self.kw = dict(do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p,
                       max_new_tokens=max_new_tokens)
        self.eos = tokenizer.eos_token_id

    @classmethod
    def load(cls, base: str, adapters: list[str] = (), dtype: str = "bfloat16", device: str | None = None, **kw):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(base, dtype=getattr(torch, dtype)).to(device)
        if adapters:
            from peft import PeftModel

            for a in adapters:
                model = PeftModel.from_pretrained(model, a).merge_and_unload()
        return cls(model, AutoTokenizer.from_pretrained(base), **kw)

    def _stopper(self, prompt_len: int, n_lines: int):
        from transformers import StoppingCriteria

        tok, eos = self.tok, self.eos
        torch = self.torch

        class Lines(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                gen = input_ids[:, prompt_len:]
                done = []
                for row in gen:
                    text = tok.decode(row, skip_special_tokens=False)
                    done.append(text.count("\n") >= n_lines or bool((row == eos).any()))
                return torch.tensor(done, device=input_ids.device)

        return Lines()

    def __call__(self, prefix: str, n: int, n_lines: int) -> list[str | None]:
        from transformers import StoppingCriteriaList

        torch = self.torch
        ids = self.tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                ids, num_return_sequences=n, pad_token_id=self.eos, eos_token_id=self.eos,
                stopping_criteria=StoppingCriteriaList([self._stopper(ids.shape[1], n_lines)]), **self.kw,
            )
        results: list[str | None] = []
        for row in out[:, ids.shape[1] :]:
            hit = (row == self.eos).nonzero()
            cut = int(hit[0]) if len(hit) else len(row)
            text = self.tok.decode(row[:cut], skip_special_tokens=True)
            lines = text.split("\n")
            if len(lines) <= n_lines and len(hit):  # ended before completing the requested lines
                results.append(None)
            else:
                results.append("\n".join(lines[:n_lines]))
        return results

    def loglik(self, prefix: str, continuation: str) -> float:
        """Mean token log-probability of ``continuation`` given ``prefix`` (style score for self-training)."""
        torch = self.torch
        p = self.tok(prefix, add_special_tokens=False).input_ids
        c = self.tok(continuation, add_special_tokens=False).input_ids
        ids = torch.tensor([p + c], device=self.model.device)
        with torch.inference_mode():
            logits = self.model(ids).logits[0, len(p) - 1 : -1].float()
        lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(c, device=ids.device)[:, None])
        return float(lp.mean())
