"""Build car2brick SFT data from BrickNet graphs and/or LDraw files.

Output (in ``out_dir``) uses exactly the file layout BrickNet's ``scripts/train.py sft`` reads, so the original
trainer works unchanged (``scripts/train_car2brick.py`` only adjusts lengths / iterations):

    paths_train.jsonl     {"source", "round": 0, "sample_index": k, "component_index": 0, "path": anchored text}
    captions_train.jsonl  {"id": source, "caption": descriptor text}   (k-th caption of a source <-> sample_index k)
    paths_val.jsonl / captions_val.jsonl
    descriptors/<source>.npz      clean (unperturbed) descriptor, for evaluation
    index.jsonl           one row per kept car: source, style, stats

Every kept car yields ``orders x augs`` examples: ``orders`` build orders (chassis-first walk, jittered after the
first) times ``augs`` conditions (the clean descriptor, then perturbed ones).

Style tags: ``official`` for ids in ``--official`` (a text file of source ids or file stems) or whose BrickNet
caption mentions Speed Champions; ``moc8`` for other cars 7-9 studs wide; ``other`` otherwise.
"""

from __future__ import annotations

import argparse
import json
import lzma
import re
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np

import bricknet

from .geometry import PartShapes
from .ldraw_io import flatten
from .lego import is_wheel, lego_car
from .library import car_like, rough_size, stud_up, upright
from .sequence import anchored_text, order_tree

OFFICIAL_WORDS = ("speed champions", "speed champion")


def _open_text(path: str | Path):
    path = str(path)
    return lzma.open(path, "rt", encoding="utf-8") if path.endswith(".xz") else open(path, encoding="utf-8")


def load_caption_groups(path: str | Path, per_model: int) -> list[list[str]]:
    """BrickNet captions file -> list (npz row order) of caption lists."""
    rows = [json.loads(line)["caption"] for line in _open_text(path)]
    return [rows[i : i + per_model] for i in range(0, len(rows), per_model)]


def iter_sources(graphs: list[str], ldr_dirs: list[str], captions: list[str | None], per_model: int):
    """Yields (source id, graph, steps or None, caption text)."""
    for gi, gpath in enumerate(graphs):
        caps = load_caption_groups(captions[gi], per_model) if gi < len(captions) and captions[gi] else None
        tag = Path(gpath).name.split(".")[0]
        for i, g in enumerate(bricknet.load_graphs(gpath)):
            yield f"{tag}-{i:06d}", g, None, " ".join(caps[i]) if caps and i < len(caps) else ""
    for d in ldr_dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.suffix.lower() not in (".ldr", ".mpd", ".dat"):
                continue
            flat = flatten(p.read_text(errors="replace"))
            if not flat.lines:
                continue
            yield p.stem, bricknet.parse_ldr(flat.text), flat.steps, ""


FIGURE_RE = re.compile(r"^(minifig|figure|bigfig)\b")


def is_figure(part_id: int) -> bool:
    """Minifigures, figures and their accessories (not part of the car)."""
    return bool(FIGURE_RE.match(bricknet.load_catalog().id_to_name[part_id].lower()))


def drop_figures(graph: bricknet.Graph, steps=None):
    """The graph without figure parts (edges and components rebuilt), and the matching instruction steps."""
    keep = np.array([not is_figure(p) for p in graph.part_ids], dtype=bool)
    if keep.all():
        return graph, steps
    from bricknet.graph import _components_from_edges

    transforms = graph.transforms if graph.transforms is not None else bricknet.decode_graph(graph)
    new = np.cumsum(keep) - 1
    e = graph.edges[keep[graph.edges["a"]] & keep[graph.edges["b"]]].copy()
    e["a"], e["b"] = new[e["a"]], new[e["b"]]
    idx = np.nonzero(keep)[0]
    out = bricknet.Graph(tuple(graph.part_ids[i] for i in idx), tuple(graph.colors[i] for i in idx),
                         np.asarray(transforms)[idx], e, _components_from_edges(len(idx), e["a"], e["b"]))
    return out, (None if steps is None else [steps[i] for i in idx])


def keep_car(graph: bricknet.Graph, min_wheels: int, width: tuple[float, float], max_parts: int) -> bool:
    if not graph.components:
        return False
    comp = graph.components[0]
    if not 20 <= len(comp) <= max_parts:
        return False
    return sum(is_wheel(graph.part_ids[i]) for i in comp) >= min_wheels


def style_of(source: str, caption: str, width: int, official: set[str]) -> str:
    if source in official or any(w in caption.lower() for w in OFFICIAL_WORDS):
        return "official"
    return "moc8" if 7 <= width <= 9 else "other"


def _sanitize_colors(graph: bricknet.Graph) -> bricknet.Graph:
    known = bricknet.load_catalog().code_to_color
    if all(c in known for c in graph.colors):
        return graph
    return replace(graph, colors=tuple(c if c in known else 71 for c in graph.colors))


def split_of(source: str, val_frac: float, seed: int = 0) -> str:
    """Train / val split from a hash of the car id, so a car stays on the same side when the data is rebuilt
    with other options (a draw from the shared generator moved 10 % of the cars between rebuilds)."""
    h = zlib.crc32(f"{seed}:{source}".encode()) / 2**32
    return "val" if h < val_frac else "train"


def build(args) -> dict:
    out = Path(args.out)
    (out / "descriptors").mkdir(parents=True, exist_ok=True)
    shapes = PartShapes(args.shapes)
    official = set()
    if args.official:
        official = {ln.strip() for ln in Path(args.official).read_text().splitlines() if ln.strip()}
    rng = np.random.default_rng(args.seed)
    files = {s: (open(out / f"paths_{s}.jsonl", "w"), open(out / f"captions_{s}.jsonl", "w")) for s in ("train", "val")}
    index = open(out / "index.jsonl", "w")
    counts = {"seen": 0, "kept": 0, "examples": 0, "errors": 0}
    try:
        for source, graph, steps, caption in iter_sources(args.graphs, args.ldr_dir, args.captions, args.per_model):
            counts["seen"] += 1
            if args.limit and counts["kept"] >= args.limit:
                break
            if args.drop_figures:
                try:
                    graph, steps = drop_figures(graph, steps)
                except Exception as e:
                    counts["errors"] += 1
                    print(f"[skip] {source}: {type(e).__name__}: {e}")
                    continue
            if not keep_car(graph, args.min_wheels, (args.min_width, args.max_width), args.max_parts):
                continue
            try:
                graph = _sanitize_colors(graph)
                comp = sorted(graph.components[0])
                T = graph.transforms if graph.transforms is not None else bricknet.decode_graph(graph)
                pids, cols = [graph.part_ids[j] for j in comp], [graph.colors[j] for j in comp]
                # stand the car stud-up (about one BrickNet component in six lies on its side / upside down); the
                # rotation is shared by all parts, so the connection tree and its relative edges are unchanged
                tr = upright([T[j] for j in comp], stud_up(pids, [T[j] for j in comp]))
                if args.car_filter and not car_like(rough_size(pids, tr), len(pids), args.min_wheels):
                    continue
                lc = lego_car(pids, cols, tr, shapes=shapes)
                L, W, H = lc.descriptor.size()
                if not (args.min_width <= W <= args.max_width) or L < 1.5 * W:
                    continue
                gp_all = np.zeros((len(graph.part_ids), 4, 4))
                gp_all[comp] = lc.grid_poses()
                comp_steps = steps if steps is not None else None
                style = style_of(source, caption, W, official)
                split = split_of(source, args.val_frac, args.seed)
                k = 0
                fp, fc = files[split]
                for r in range(args.orders):
                    tree, order = order_tree(graph, gp_all, 0, steps=comp_steps, rng=rng,
                                             jitter=0.0 if r == 0 else args.jitter)
                    text = anchored_text(tree, gp_all[order[0]], target_first=args.target_first)
                    for a in range(args.augs):
                        desc = lc.descriptor if a == 0 else lc.descriptor.perturb(rng, args.noise)
                        fp.write(json.dumps({"source": source, "round": 0, "sample_index": k, "component_index": 0,
                                             "path": text}) + "\n")
                        fc.write(json.dumps({"id": source, "caption": desc.to_text(style)}) + "\n")
                        k += 1
            except Exception as e:  # a malformed model must not stop a long run
                counts["errors"] += 1
                print(f"[skip] {source}: {type(e).__name__}: {e}")
                continue
            lc.descriptor.save(out / "descriptors" / f"{source}.npz")
            index.write(json.dumps({"source": source, "style": style, "split": split, "size": [L, W, H],
                                    "parts": len(comp), "wheels": list(lc.descriptor.wheels)}) + "\n")
            counts["kept"] += 1
            counts["examples"] += k
            if counts["kept"] % 50 == 0:
                print(counts, flush=True)
    finally:
        for fp, fc in files.values():
            fp.close(), fc.close()
        index.close()
    print(counts)
    return counts


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--graphs", nargs="*", default=[], help="BrickNet graph .npz files (decompressed)")
    ap.add_argument("--captions", nargs="*", default=[], help="captions .jsonl(.xz) aligned with --graphs (optional)")
    ap.add_argument("--per-model", type=int, default=5, help="captions per model in the captions files")
    ap.add_argument("--ldr-dir", nargs="*", default=[], help="folders of .ldr/.mpd cars (e.g. official sets)")
    ap.add_argument("--official", help="text file of source ids / file stems to tag <official>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--orders", type=int, default=2)
    ap.add_argument("--augs", type=int, default=3)
    ap.add_argument("--jitter", type=float, default=1.5)
    ap.add_argument("--target-first", action="store_true",
                    help="anchor line before each part (a predicted target, for snap decoding) instead of after it")
    ap.add_argument("--noise", type=float, default=1.0, help="descriptor perturbation strength")
    ap.add_argument("--min-wheels", type=int, default=3)
    ap.add_argument("--min-width", type=int, default=5)
    ap.add_argument("--max-width", type=int, default=11)
    ap.add_argument("--max-parts", type=int, default=600)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many kept cars (0 = all)")
    ap.add_argument("--shapes", default="auto", choices=["auto", "cvx", "ldraw", "connector"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-figures", dest="drop_figures", action="store_false",
                    help="keep minifigures (dropped by default: they are not part of the car)")
    ap.add_argument("--no-car-filter", dest="car_filter", action="store_false",
                    help="skip the survey filter (5-11 wide, height/width 0.5-0.8, see library.car_like)")
