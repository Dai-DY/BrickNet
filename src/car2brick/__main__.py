"""car2brick command line.

Front half
  photos2desc   photos (+ masks / SAM2) -> descriptor          (COLMAP or an existing COLMAP text model)
  mesh2desc     any car mesh -> descriptor                     (2DGS / TRELLIS / ShapeNet ...)
  ldr2desc      LDraw car -> descriptor                        (the LEGO side of the bridge)
  showdesc      print a descriptor as conditioning text

Back half
  baseline      descriptor -> LDraw (B0 greedy plates, B1 + tiles, B2 part-by-part fitting)
  build-data    BrickNet graphs / LDraw folders -> SFT jsonl
  generate      descriptor -> LDraw with the fine-tuned model + constrained decoding
  selftrain     ShapeNet meshes -> generated examples for the next SFT round
  eval          score an LDraw model against a descriptor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _desc_out(desc, out: str | None, style: str | None = None):
    if out:
        desc.save(out)
        Path(out).with_suffix(".txt").write_text(desc.to_text(style) + "\n")
        print(f"saved {out} (+ .txt)")
    print(desc.to_text(style))


def cmd_photos2desc(a):
    from .recon.pipeline import photos_to_descriptor

    labels = {}
    for item in a.labels or []:
        name, d = item.split("=", 1)
        labels[{"window": 2, "light": 3}[name]] = d
    desc = photos_to_descriptor(
        a.images, a.workspace, colmap_model=a.colmap_model, image_dir=a.image_dir, mask_dir=a.masks,
        label_dirs=labels or None, sam2=(a.sam2_ckpt, a.sam2_cfg) if a.sam2_ckpt else None, matcher=a.matcher,
        target_width=a.width, max_misses=a.max_misses,
    )
    _desc_out(desc, a.out, a.style)


def cmd_mesh2desc(a):
    import trimesh

    from .canonical import mesh_descriptor

    up = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}.get(a.up)
    desc, _ = mesh_descriptor(trimesh.load(a.mesh, force="mesh"), up_hint=up, target_width=a.width,
                              colors=tuple(a.colors or ()))
    _desc_out(desc, a.out, a.style)


def cmd_ldr2desc(a):
    from .ldraw_io import flatten
    from .lego import lego_car_from_ldr

    lc = lego_car_from_ldr(flatten(Path(a.ldr).read_text(errors="replace")).text)
    _desc_out(lc.descriptor, a.out, a.style)


def cmd_showdesc(a):
    from .descriptor import CarDescriptor

    print(CarDescriptor.load(a.desc).to_text(a.style))


def cmd_baseline(a):
    from .baseline import baseline_ldr
    from .descriptor import CarDescriptor

    desc = CarDescriptor.load(a.desc)
    method = "b0" if a.no_tiles else a.method
    if method == "b2":
        from .fit import fit_ldr

        ldr, res = fit_ldr(desc, lambda_out=a.lambda_out, min_fill=a.min_fill, use_slopes=not a.no_slopes)
        print(res.stats)
    else:
        ldr = baseline_ldr(desc, cap_tiles=method == "b1")
    Path(a.out).write_text(ldr)
    print(f"wrote {a.out} ({ldr.count(chr(10) + '1 ')} parts)")


def cmd_build_data(a):
    from .dataset import build

    build(a)


def cmd_generate(a):
    from .decode import ConstrainedDecoder, DecodeConfig, HFProposer
    from .descriptor import CarDescriptor
    from .evaluate import evaluate, poses_to_ldr, recolor
    from .geometry import PartShapes

    desc = CarDescriptor.load(a.desc)
    shapes = PartShapes(a.shapes)
    proposer = HFProposer.load(a.model, a.lora, temperature=a.temperature)
    dec = ConstrainedDecoder(proposer, shapes, DecodeConfig(max_parts=a.max_parts, candidates=a.candidates))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in range(a.samples):
        res = dec.run(desc, a.style)
        ev = evaluate(res.part_ids, res.poses, desc, shapes, colors=res.colors)
        cols = recolor(res.part_ids, res.colors, desc.colors) if a.recolor else res.colors
        (out / f"sample-{s:02d}.ldr").write_text(poses_to_ldr(res.part_ids, cols, res.poses, f"sample {s}"))
        (out / f"sample-{s:02d}.txt").write_text(res.text)
        rows.append({"sample": s, **{k: v for k, v in ev.items()}, "stats": res.stats})
        print(json.dumps(rows[-1], default=str), flush=True)
    ok = [r for r in rows if r.get("components") == 1]
    best = max(ok or rows, key=lambda r: r.get("iou", 0))
    print(f"best: sample-{best['sample']:02d}.ldr iou={best.get('iou', 0):.3f}")
    (out / "summary.json").write_text(json.dumps(rows, indent=1, default=str))


def cmd_selftrain(a):
    from .selftrain import run

    print(run(a))


def cmd_eval(a):
    from .descriptor import CarDescriptor
    from .evaluate import evaluate_ldr
    from .ldraw_io import flatten

    print(evaluate_ldr(flatten(Path(a.ldr).read_text(errors="replace")).text, CarDescriptor.load(a.desc)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m car2brick", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("photos2desc")
    p.add_argument("--images", help="input images (COLMAP is run on them)")
    p.add_argument("--workspace", help="COLMAP workspace")
    p.add_argument("--colmap-model", help="existing COLMAP *text* model (skips COLMAP)")
    p.add_argument("--image-dir", help="images matching --colmap-model (undistorted)")
    p.add_argument("--masks", help="car masks, one per image (same stem)")
    p.add_argument("--labels", nargs="*", help="window=DIR light=DIR label masks")
    p.add_argument("--sam2-ckpt"), p.add_argument("--sam2-cfg")
    p.add_argument("--matcher", default="exhaustive", choices=["exhaustive", "sequential"])
    p.add_argument("--width", type=float, default=8.0)
    p.add_argument("--max-misses", type=int, default=1)
    p.add_argument("--style", default="official")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_photos2desc)

    p = sub.add_parser("mesh2desc")
    p.add_argument("mesh")
    p.add_argument("--up", default="auto", choices=["auto", "x", "y", "z"])
    p.add_argument("--width", type=float, default=8.0)
    p.add_argument("--colors", nargs="*", help="BrickNet colour names, main first, e.g. red black")
    p.add_argument("--style", default="official")
    p.add_argument("--out")
    p.set_defaults(func=cmd_mesh2desc)

    p = sub.add_parser("ldr2desc")
    p.add_argument("ldr")
    p.add_argument("--style", default=None)
    p.add_argument("--out")
    p.set_defaults(func=cmd_ldr2desc)

    p = sub.add_parser("showdesc")
    p.add_argument("desc")
    p.add_argument("--style", default="official")
    p.set_defaults(func=cmd_showdesc)

    p = sub.add_parser("baseline")
    p.add_argument("desc")
    p.add_argument("--out", required=True)
    p.add_argument("--method", default="b1", choices=["b0", "b1", "b2"],
                   help="b0 greedy plates, b1 + tiles, b2 part-by-part fitting with bricks and slopes")
    p.add_argument("--no-tiles", action="store_true", help="same as --method b0")
    p.add_argument("--lambda-out", type=float, default=3.0, help="b2: penalty per sub-cell outside the target")
    p.add_argument("--min-fill", type=float, default=0.6, help="b2: minimum score per unit of part volume")
    p.add_argument("--no-slopes", action="store_true", help="b2: plates and bricks only")
    p.set_defaults(func=cmd_baseline)

    from .dataset import add_args as data_args

    p = sub.add_parser("build-data")
    data_args(p)
    p.set_defaults(func=cmd_build_data)

    p = sub.add_parser("generate")
    p.add_argument("desc")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--lora", action="append", default=[], help="adapters in order (BrickNet PT, car2brick SFT)")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--max-parts", type=int, default=500)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--style", default="official")
    p.add_argument("--recolor", action="store_true", help="map generated colours onto the descriptor's colours")
    p.add_argument("--shapes", default="auto", choices=["auto", "cvx", "ldraw", "connector"])
    p.set_defaults(func=cmd_generate)

    from .selftrain import add_args as st_args

    p = sub.add_parser("selftrain")
    st_args(p)
    p.set_defaults(func=cmd_selftrain)

    p = sub.add_parser("eval")
    p.add_argument("ldr")
    p.add_argument("desc")
    p.set_defaults(func=cmd_eval)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
