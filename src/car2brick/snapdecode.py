"""Target-first decoding with program-chosen connections ("snap"), on a KV-cached language model.

Sequences built with ``build-data --target-first`` write every part (after the root) as

    @ i j k                    target cell of the part's origin          <- sampled from the model
    b plate 2x10 | red         the part                                  <- sampled from the model
    a stud stud m hole f 270   how it attaches to an earlier part        <- enumerated here, ranked by the model

Each step: sample a target (rejected if it lies outside the allowed region), sample a part, then enumerate every
connection of that part to the parts already placed (``snap.enumerate_candidates``) whose result has its origin in
the target cell (``target_tol`` cells of slack), stays inside the allowed region and does not collide. One of those
connection lines is sampled from the model restricted to them (``CachedLM.choose``, token by token over their prefix
trie). If the part fits nowhere in the target cell, placements up to ``fallback_tol`` cells away are offered and
the model picks one of their cells, which replaces its target line. Failures back off: another part for the same
target (a part that already failed there is not re-checked),
then another target - sampled from the model restricted to the in-shape cells not tried yet at this step - and
after ``target_attempts`` targets the sequence ends. The model sampling the end of the sequence (EOS) instead of a
target ``eos_patience`` times also ends it.

Why: the model sees the shape when it picks *where* the next part goes, and never has to invert the kinematics to
get it there (see ``snap``; in the shape2brick experiments this was what made generations follow the target).

``CachedLM`` keeps the KV cache of the accepted text, so a step only runs the new tokens; candidates are sampled
one at a time and the cache is cropped back afterwards, so memory does not grow with the number of candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import bricknet
from bricknet.core import FixedEdge
from bricknet.tree import ParseError, parse_node

from .decode import DecodeResult, _Collisions, child_relative
from .descriptor import CarDescriptor
from .geometry import PartShapes
from .sequence import anchor_line, parse_anchor, parse_root, prompt, root_line
from .snap import ConnIndex, edge_line, enumerate_candidates, make_edge
from .units import LDU_PER_PLATE, LDU_PER_STUD

CELL = np.array([LDU_PER_STUD, LDU_PER_STUD, LDU_PER_PLATE])
MATCH_LDU = 0.5  # slack of the target-cell test (see ``cell_distance``)


def cell_distance(pose: np.ndarray, tgt) -> int:
    """Cells between the cell of ``pose``'s origin and ``tgt`` (max over axes). An origin within ``MATCH_LDU`` of a
    cell boundary counts as in both cells: anchors of the first target-first data had float noise on boundaries
    (``anchor_line`` used a 1e-6 epsilon) and were computed from unquantized edges, which can differ from the path
    text by a fraction of an LDU, so either side may be what the model learnt."""
    x = pose[:3, 3]
    lo = np.floor((x - MATCH_LDU) / CELL)
    hi = np.floor((x + MATCH_LDU) / CELL)
    t = np.asarray(tgt, float)
    return int(np.maximum(np.maximum(lo - t, t - hi), 0).max())


class CachedLM:
    """A causal LM with a KV cache of the accepted text.

    ``reset(text)`` / ``append(text)`` extend the cache; ``sample_line()`` samples one line and leaves the cache as it
    was; ``score(lines)`` returns each line's total log-probability (newline included) as a continuation;
    ``choose(lines)`` samples one of the given lines; ``checkpoint()`` / ``restore(cp)`` roll back appended text.

    Text is tokenized chunk by chunk. Every chunk starts at a line start and Qwen's pre-tokenizer splits on newlines,
    so this gives the same tokens as tokenizing the whole sequence at once (the training layout).
    """

    def __init__(self, model, tokenizer, *, temperature: float = 1.0, top_k: int = 20, top_p: float = 0.95,
                 max_line_tokens: int = 64, chunk: int = 2048):
        import torch

        self.torch = torch
        self.model = model.eval()
        self.tok = tokenizer
        self.eos = tokenizer.eos_token_id
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.max_line_tokens, self.chunk = max_line_tokens, chunk
        self.cache = None
        self.n = 0
        self.logits = None  # next-token logits after the cached text
        # token ids whose text contains a newline (a line ends there); decoding ~150k ids once takes a second
        self._newline = {i for i in range(len(tokenizer)) if "\n" in tokenizer.decode([i])}

    @classmethod
    def load(cls, base: str, adapters=(), dtype: str = "bfloat16", device: str | None = None, **kw):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(base, dtype=getattr(torch, dtype)).to(device)
        if adapters:
            from peft import PeftModel

            for a in adapters:
                model = PeftModel.from_pretrained(model, a).merge_and_unload()
        return cls(model, AutoTokenizer.from_pretrained(base), **kw)

    # ---- cache --------------------------------------------------------------------------------------------------
    def _forward(self, ids: list[int]):
        """Run ``ids`` on top of the cache; returns logits (len(ids), vocab) as float32."""
        torch = self.torch
        out = []
        for s in range(0, len(ids), self.chunk):
            x = torch.tensor([ids[s : s + self.chunk]], device=self.model.device)
            with torch.inference_mode():
                o = self.model(input_ids=x, past_key_values=self.cache, use_cache=True)
            self.cache = o.past_key_values
            out.append(o.logits[0].float())
        self.n += len(ids)
        return torch.cat(out)

    def _encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def reset(self, text: str) -> None:
        from transformers import DynamicCache

        self.cache, self.n = DynamicCache(), 0
        self.logits = self._forward(self._encode(text))[-1]

    def append(self, text: str) -> None:
        self.logits = self._forward(self._encode(text))[-1]

    def checkpoint(self):
        return self.n, self.logits

    def restore(self, cp) -> None:
        n, logits = cp
        if self.n != n:
            self.cache.crop(n - self.n)  # transformers 5: a negative count drops that many tokens from the end
            self.n = n
        self.logits = logits

    # ---- sampling / scoring -------------------------------------------------------------------------------------
    def _sample(self, logits):
        torch = self.torch
        if self.temperature <= 0:
            return int(logits.argmax())
        logits = logits / self.temperature
        if self.top_k:
            kth = torch.topk(logits, min(self.top_k, logits.numel())).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = torch.softmax(logits, -1)
        if self.top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            cut = torch.cumsum(sp, 0) - sp > self.top_p
            sp[cut] = 0
            probs = torch.zeros_like(probs).scatter(0, si, sp)
        return int(torch.multinomial(probs / probs.sum(), 1))

    def _is_newline(self, t: int) -> bool:
        return t in self._newline

    def sample_line(self) -> str | None:
        """One sampled line (without the newline); None if the model ends the sequence. The cache is unchanged."""
        cp = self.checkpoint()
        toks, logits = [], self.logits
        try:
            for _ in range(self.max_line_tokens):
                t = self._sample(logits)
                if t == self.eos:
                    return None
                toks.append(t)
                if self._is_newline(t):
                    break
                logits = self._forward([t])[-1]
            return self.tok.decode(toks).split("\n")[0]
        finally:
            self.restore(cp)

    def score(self, lines: list[str]) -> list[float]:
        """Total log-probability of each ``line + "\\n"`` as the continuation of the cached text."""
        torch = self.torch
        cp = self.checkpoint()
        out = []
        for line in lines:
            ids = self._encode(line + "\n")
            logits = torch.cat([self.logits[None], self._forward(ids[:-1])]) if len(ids) > 1 else self.logits[None]
            lp = torch.log_softmax(logits, -1)[torch.arange(len(ids)), torch.tensor(ids, device=logits.device)]
            out.append(float(lp.sum()))
            self.restore(cp)
        return out

    def choose(self, lines: list[str], allow_eos: bool = False, greedy: bool = False) -> int | None:
        """Sample one of ``lines`` token by token from the model's distribution restricted to those lines (a prefix
        trie); returns its index, or None if ``allow_eos`` and the model ends the sequence. The cache is unchanged.

        Costs one forward per token of the chosen line (forced runs share one call), however many lines there are."""
        torch = self.torch
        if not lines:
            return None
        if len(lines) == 1 and not allow_eos:
            return 0
        enc = self._encode_lines(lines)
        alive = list(range(len(lines)))
        cp = self.checkpoint()
        logits, pos, forced = self.logits, 0, []
        try:
            while True:
                if pos >= min(len(enc[i]) for i in alive):  # duplicate lines (or one a prefix of another)
                    return alive[0]
                nxt = {}
                for i in alive:
                    nxt.setdefault(enc[i][pos], []).append(i)
                if allow_eos and pos == 0:
                    nxt.setdefault(self.eos, None)
                if len(nxt) == 1:
                    t = next(iter(nxt))
                else:
                    if forced:  # run the tokens all remaining lines share before branching
                        logits = self._forward(forced)[-1]
                        forced = []
                    ids = torch.tensor(list(nxt), device=logits.device)
                    sub = logits[ids]
                    if greedy or self.temperature <= 0:
                        t = int(ids[int(sub.argmax())])
                    else:
                        t = int(ids[int(torch.multinomial(torch.softmax(sub / self.temperature, -1), 1))])
                if t == self.eos and nxt[t] is None:
                    return None
                alive = nxt[t]
                if len(alive) == 1:
                    return alive[0]
                forced.append(t)
                pos += 1
        finally:
            self.restore(cp)

    def _encode_lines(self, lines: list[str]) -> list[list[int]]:
        cache = self.__dict__.setdefault("_line_ids", {})
        if len(cache) > 200_000:
            cache.clear()
        new = [ln for ln in dict.fromkeys(lines) if ln not in cache]
        if new:
            for ln, ids in zip(new, self.tok([ln + "\n" for ln in new], add_special_tokens=False)["input_ids"]):
                cache[ln] = ids
        return [cache[ln] for ln in lines]


@dataclass
class SnapConfig:
    max_parts: int = 500
    target_attempts: int = 12  # targets tried per step before giving up (after the first: untried in-shape cells)
    node_attempts: int = 4  # parts tried per target
    max_candidates: int = 256  # distinct feasible placements offered to the model
    max_lines: int = 32  # ways of writing one placement (stud pairs, parents) offered to the model
    target_tol: int = 0  # cells of slack between a placement's origin cell and the target (per axis)
    fallback_tol: int = 1  # if nothing fits the target: placements this close, written with their own cell
    outside_tol: float = 0.15
    margin: tuple[int, int, int] = (1, 1, 2)
    collisions: bool = True
    eos_patience: int = 2
    root_attempts: int = 16
    greedy_edges: bool = False  # take the best-scored connection instead of sampling


class SnapDecoder:
    def __init__(self, lm, shapes: PartShapes | None = None, config: SnapConfig | None = None):
        self.lm = lm
        self.shapes = shapes or PartShapes()
        self.cfg = config or SnapConfig()
        self.catalog = bricknet.load_catalog()
        self.index = ConnIndex()
        self.rng = np.random.default_rng(0)

    def outside_fraction(self, pid: int, pose: np.ndarray, allowed: np.ndarray) -> float:
        pts = self.shapes.local_interior(pid) @ pose[:3, :3].T + pose[:3, 3]
        idx = np.floor(pts / CELL).astype(int)
        inside_grid = ((idx >= 0) & (idx < np.array(allowed.shape))).all(1)
        ok = np.zeros(len(pts), dtype=bool)
        ok[inside_grid] = allowed[tuple(idx[inside_grid].T)]
        return 1.0 - ok.mean()

    def _target_ok(self, tgt, allowed) -> bool:
        i, j, k = tgt
        if not (0 <= i < allowed.shape[0] and 0 <= j < allowed.shape[1] and 0 <= k < allowed.shape[2]):
            return False
        # the origin of a part sits on its top face: accept a target cell that is allowed or rests on an allowed cell
        return bool(allowed[i, j, k] or (k > 0 and allowed[i, j, k - 1]))

    def _feasible(self, pid: int, tgt, res: DecodeResult, allowed, col, tol: int = 0) -> list[tuple[str, np.ndarray]]:
        """(edge line, pose) for connections of ``pid`` landing within ``tol`` cells of ``tgt`` that are feasible.

        One placement can usually be written many ways (any overlapping stud pair, or another parent), and the model
        has learnt BrickNet's own choice, so every line of a placement is kept for ranking; the cap applies to
        distinct placements (``max_candidates``) and to lines per placement (``max_lines``)."""
        cfg = self.cfg
        target = (np.array(tgt) + 0.5) * CELL
        radius = float(((tol + 0.5) * CELL).max()) + MATCH_LDU
        placed = list(zip(res.part_ids, res.poses))
        poses: dict[tuple, list] = {}  # placement -> [ok?, pose, lines]
        for cand in enumerate_candidates(self.index, placed, pid, np.zeros(3), target, radius):
            ppid = res.part_ids[cand.parent]
            edge = make_edge(self.index, cand, len(res.part_ids), ppid, pid)
            pose = res.poses[cand.parent] @ child_relative(edge, ppid, pid)
            if cell_distance(pose, tgt) > tol:
                continue
            key = tuple(np.round(pose[:3, :], 2).ravel())
            if key not in poses:
                if sum(v[0] for v in poses.values()) >= cfg.max_candidates:
                    continue
                ok = self.outside_fraction(pid, pose, allowed) <= cfg.outside_tol and \
                    col.ok(pid, pose, cand.parent, isinstance(edge, FixedEdge))
                poses[key] = [ok, pose, []]
            entry = poses[key]
            line = edge_line(edge)
            if entry[0] and line not in entry[2] and len(entry[2]) < cfg.max_lines:
                entry[2].append(line)
        return [(line, pose) for ok, pose, lines in poses.values() if ok for line in lines]

    def _moved(self, pid, tgt, res, allowed, col, cp) -> list[tuple[str, np.ndarray]]:
        """Fallback when ``pid`` fits nowhere in the target cell: feasible placements up to ``fallback_tol`` cells
        away. The model picks one of their cells (as a target line, in place of its own), so the text stays in the
        training layout (anchor = the part's actual cell); on return the cache holds that target line and the
        candidates of the chosen cell are returned."""
        cands = self._feasible(pid, tgt, res, allowed, col, self.cfg.fallback_tol)
        if not cands:
            return []
        by_cell: dict[str, list] = {}
        for c in cands:
            by_cell.setdefault(anchor_line(c[1]), []).append(c)
        lines = list(by_cell)
        self.lm.restore(cp)
        tl = lines[self.lm.choose(lines)]
        self.lm.append(tl + "\n")
        return by_cell[tl]

    def run(self, desc: CarDescriptor, style: str | None = "official", caption: str | None = None) -> DecodeResult:
        cfg, lm = self.cfg, self.lm
        caption = caption if caption is not None else desc.to_text(style)
        allowed = desc.allowed(cfg.margin)
        res = DecodeResult(text="")
        stats = {"targets": 0, "parse_fail": 0, "target_out": 0, "snap_empty": 0, "moved": 0, "stop": "max_parts"}
        col = _Collisions(cfg.collisions)
        lm.reset(prompt(caption))
        try:
            # root: node line + @root line
            for _ in range(cfg.root_attempts):
                node = lm.sample_line()
                try:
                    part = parse_node(node or "", self.catalog)
                except ParseError:
                    stats["parse_fail"] += 1
                    continue
                cp = lm.checkpoint()
                lm.append(node + "\n")
                pose = parse_root(lm.sample_line() or "")
                if pose is None or self.outside_fraction(part.part_id, pose, allowed) > cfg.outside_tol:
                    stats["parse_fail" if pose is None else "target_out"] += 1
                    lm.restore(cp)
                    continue
                rl = root_line(pose)
                lm.append(rl + "\n")
                res.part_ids.append(part.part_id), res.colors.append(part.color), res.poses.append(pose)
                res.text += f"{node}\n{rl}\n"
                col.add(part.part_id, pose)
                break
            else:
                stats["stop"] = "no_root"
                return res

            cells = np.argwhere(allowed | np.pad(allowed, ((0, 0), (0, 0), (1, 0)))[:, :, :-1])
            target_lines = [f"@ {i} {j} {k}" for i, j, k in cells]
            eos = 0
            while len(res.part_ids) < cfg.max_parts:
                accepted, tried = False, set()
                for attempt in range(cfg.target_attempts):
                    if attempt == 0:  # the model's own choice ...
                        tl = lm.sample_line()
                    else:  # ... then its choice among the in-shape cells not tried yet at this step
                        left = [t for t in target_lines if t not in tried]
                        k = lm.choose(left, allow_eos=True)
                        tl = None if k is None else left[k]
                    if tl is None:
                        eos += 1
                        if eos >= cfg.eos_patience:
                            break
                        continue
                    stats["targets"] += 1
                    tried.add(tl)
                    tgt = parse_anchor(tl)
                    if tgt is None:
                        stats["parse_fail"] += 1
                        continue
                    if not self._target_ok(tgt, allowed):
                        stats["target_out"] += 1
                        continue
                    cp = lm.checkpoint()
                    lm.append(tl + "\n")
                    failed = set()
                    for _ in range(cfg.node_attempts):
                        node = lm.sample_line()
                        if node in failed:
                            stats["snap_empty"] += 1
                            continue
                        try:
                            part = parse_node(node or "", self.catalog)
                        except ParseError:
                            stats["parse_fail"] += 1
                            continue
                        cands = self._feasible(part.part_id, tgt, res, allowed, col, cfg.target_tol)
                        if not cands and cfg.fallback_tol > cfg.target_tol:
                            cands = self._moved(part.part_id, tgt, res, allowed, col, cp)
                            if cands:
                                tl = anchor_line(cands[0][1])
                                stats["moved"] += 1
                        if not cands:
                            stats["snap_empty"] += 1
                            failed.add(node)
                            continue
                        lm.append(node + "\n")
                        k = lm.choose([c[0] for c in cands], greedy=cfg.greedy_edges)
                        line, pose = cands[k]
                        lm.append(line + "\n")
                        res.part_ids.append(part.part_id), res.colors.append(part.color), res.poses.append(pose)
                        res.text += f"{tl}\n{node}\n{line}\n"
                        col.add(part.part_id, pose)
                        accepted = True
                        break
                    if accepted:
                        break
                    lm.restore(cp)
                if not accepted:
                    stats["stop"] = "eos" if eos >= cfg.eos_patience else "stuck"
                    break
                eos = 0
        finally:
            col.close()
        stats["parts"] = len(res.part_ids)
        res.stats = stats
        return res
