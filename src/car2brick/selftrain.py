"""Self-training on real-car shapes (ShapeNet): generate -> filter -> add to the training set.

For every mesh: descriptor (``canonical.mesh_descriptor``) -> ``samples`` constrained generations -> keep the best
one that is a single connected component, collision-free (when colliders are available) and has IoU >= ``min_iou``;
ties are broken by the model's own style score (mean log-likelihood under the ``<official>`` tag).

Kept examples are written in the same jsonl layout as ``dataset.build`` (paths / captions), so the next SFT round
simply concatenates them with the LEGO-derived data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .canonical import mesh_descriptor
from .decode import ConstrainedDecoder, DecodeConfig, HFProposer
from .evaluate import evaluate, poses_to_ldr
from .geometry import PartShapes

UP_AXES = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def find_meshes(root: str | Path, pattern: str) -> list[Path]:
    return sorted(Path(root).rglob(pattern))


def run(args) -> dict:
    import trimesh

    out = Path(args.out)
    (out / "ldr").mkdir(parents=True, exist_ok=True)
    shapes = PartShapes(args.shapes)
    proposer = HFProposer.load(args.model, args.lora, temperature=args.temperature)
    decoder = ConstrainedDecoder(proposer, shapes, DecodeConfig(max_parts=args.max_parts, candidates=args.candidates))
    fp, fc = open(out / "paths_selftrain.jsonl", "a"), open(out / "captions_selftrain.jsonl", "a")
    log = open(out / "selftrain_log.jsonl", "a")
    meshes = find_meshes(args.meshes, args.pattern)[: args.limit or None]
    kept = 0
    for mi, path in enumerate(meshes):
        source = f"st-{path.parent.name if path.stem == 'model_normalized' else path.stem}"
        try:
            mesh = trimesh.load(path, force="mesh")
            desc, _ = mesh_descriptor(mesh, up_hint=UP_AXES.get(args.up))
        except Exception as e:
            print(f"[skip] {path}: {e}")
            continue
        caption = desc.to_text(args.style)
        best = None
        for s in range(args.samples):
            res = decoder.run(desc, caption=caption)
            ev = evaluate(res.part_ids, res.poses, desc, shapes, collisions=True, colors=res.colors)
            ok = ev.get("components") == 1 and not ev.get("collisions") and ev.get("iou", 0) >= args.min_iou
            score = (ok, round(ev.get("iou", 0.0), 2), proposer.loglik(caption + "\n", res.text) if ok else -1e9)
            log.write(json.dumps({"source": source, "sample": s, **{k: v for k, v in ev.items() if k != "size"},
                                  "stats": res.stats}) + "\n")
            if best is None or score > best[0]:
                best = (score, res, ev)
        if best and best[0][0]:
            _, res, ev = best
            fp.write(json.dumps({"source": source, "round": 0, "sample_index": 0, "component_index": 0,
                                 "path": res.text}) + "\n")
            fc.write(json.dumps({"id": source, "caption": caption}) + "\n")
            (out / "ldr" / f"{source}.ldr").write_text(poses_to_ldr(res.part_ids, res.colors, res.poses, source))
            kept += 1
        print(f"[{mi + 1}/{len(meshes)}] {source}: kept={bool(best and best[0][0])} total_kept={kept}", flush=True)
    fp.close(), fc.close(), log.close()
    return {"meshes": len(meshes), "kept": kept}


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--meshes", required=True, help="root folder of car meshes (e.g. ShapeNetCore 02958343)")
    ap.add_argument("--pattern", default="model_normalized.obj")
    ap.add_argument("--up", default="y", choices=["x", "y", "z", "auto"], help="mesh up axis (ShapeNet: y)")
    ap.add_argument("--model", required=True, help="base HF model, e.g. Qwen/Qwen3-0.6B")
    ap.add_argument("--lora", action="append", default=[], help="adapters in order: BrickNet PT, then car2brick")
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", default="official")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--max-parts", type=int, default=500)
    ap.add_argument("--min-iou", type=float, default=0.45)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shapes", default="auto", choices=["auto", "cvx", "ldraw", "connector"])
