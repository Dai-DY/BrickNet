#!/usr/bin/env python3
"""Generate cars with a trained car2brick model and score / render them.

Targets: validation LEGO cars (``--data``: their clean descriptors; the original car is shown as reference) and / or
ShapeNet descriptors from ``scripts/run_baselines.py`` (``--shapenet``: its B2 result is shown as reference).

    python scripts/eval_generate.py --model Qwen/Qwen3-1.7B --lora kulits/BrickNet-1.7B-PT \
        --lora outputs/runs/c2b_tf_17b/checkpoint-500 --data outputs/data/c2b_v2_tf --n-val 6 \
        --shapenet outputs/baselines/shapenet --n-shapenet 6 --out outputs/runs/c2b_tf_17b/gen_500
Outputs: <out>/<name>.ldr, <out>/summary.jsonl, <out>/sheet_*.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from car2brick.descriptor import CarDescriptor  # noqa: E402
from car2brick.evaluate import evaluate_ldr, poses_to_ldr, recolor  # noqa: E402
from car2brick.geometry import PartShapes  # noqa: E402
from car2brick.snapdecode import CachedLM, SnapConfig, SnapDecoder  # noqa: E402


def targets(a) -> list[tuple[str, Path, str | None]]:
    """(name, descriptor path, reference LDraw path or None)."""
    out = []
    if a.data:
        d = Path(a.data)
        val = [json.loads(ln) for ln in open(d / "index.jsonl")]
        val = [r for r in val if r["split"] == "val" and 6 <= r["size"][1] <= 10]
        rng = np.random.default_rng(0)
        # --sources first (the same cars as an earlier run), then random validation cars up to --n-val in all
        pick = [r for s in a.sources or [] for r in val if r["source"] == s]
        rest = [val[i] for i in rng.permutation(len(val)) if val[i]["source"] not in (a.sources or [])]
        pick += rest[: max(0, a.n_val - len(pick))]
        for r in pick:
            ref = Path(a.library) / "ldr" / f"{r['source']}.ldr"
            out.append((f"val-{r['source']}", d / "descriptors" / f"{r['source']}.npz",
                        str(ref) if ref.exists() else None))
    if a.shapenet:
        rows = [json.loads(ln) for ln in open(Path(a.shapenet) / "summary.jsonl")]
        for r in rows[: a.n_shapenet]:
            d = Path(a.shapenet) / r["label"]
            out.append((f"sn-{r['label']}", d / "desc.npz", str(d / "b2.ldr")))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", action="append", default=[])
    ap.add_argument("--data", default=None)
    ap.add_argument("--library", default="outputs/car_library")
    ap.add_argument("--n-val", type=int, default=6)
    ap.add_argument("--sources", nargs="*", default=None, help="validation car ids to use instead of a random pick")
    ap.add_argument("--shapenet", default=None)
    ap.add_argument("--n-shapenet", type=int, default=6)
    ap.add_argument("--style", default="moc8")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-parts", type=int, default=400)
    ap.add_argument("--target-tol", type=int, default=0)
    ap.add_argument("--desc-format", default="v2", choices=("v1", "v2"),
                    help="descriptor text the model was trained on")
    ap.add_argument("--samples", type=int, default=1, help="generations per car (name suffix _s<i>)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    lm = CachedLM.load(a.model, a.lora, temperature=a.temperature)
    shapes = PartShapes()
    dec = SnapDecoder(lm, shapes, SnapConfig(max_parts=a.max_parts, target_tol=a.target_tol))
    rows = []
    with open(out / "summary.jsonl", "w") as fh:
        for (car, desc_path, ref), i in ((t_, i) for t_ in targets(a) for i in range(a.samples)):
            name = car if a.samples == 1 else f"{car}_s{i}"
            torch.manual_seed(a.seed * 1000 + i)
            desc = CarDescriptor.load(desc_path)
            t = time.time()
            res = dec.run(desc, a.style, caption=desc.to_text(a.style, fmt=a.desc_format))
            cols = recolor(res.part_ids, res.colors, desc.colors)
            ldr = poses_to_ldr(res.part_ids, cols, res.poses, name)
            (out / f"{name}.ldr").write_text(ldr)
            (out / f"{name}.txt").write_text(res.text)
            ev = evaluate_ldr(ldr, desc) if res.part_ids else {"parts": 0, "iou": 0.0, "iou_filled": 0.0,
                                                               "components": 0}
            row = {"name": name, "car": car, "sample": i, "desc": str(desc_path), "size": list(desc.size()),
                   "seconds": round(time.time() - t, 1),
                   "parts": ev["parts"], "iou": round(float(ev["iou"]), 3),
                   "iou_filled": round(float(ev["iou_filled"]), 3), "components": ev["components"],
                   "stats": res.stats, "ref": ref}
            if ref:
                rv = evaluate_ldr(Path(ref).read_text(), desc)
                row["ref_iou_filled"] = round(float(rv["iou_filled"]), 3)
                row["ref_parts"] = rv["parts"]
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(json.dumps(row), flush=True)
    for key in ("iou_filled", "parts", "components", "seconds"):
        print(f"{key}: mean {np.mean([r[key] for r in rows]):.3f}")
    report(rows, out)
    if not a.no_render:
        render(rows, out, as_text=a.desc_format == "v2")


def report(rows, out: Path):
    """Per car: mean / best / spread of the filled IoU over its samples; how the runs stopped."""
    from collections import Counter, defaultdict

    by = defaultdict(list)
    for r in rows:
        by[r["car"]].append(r)
    cars = []
    for car, rs in by.items():
        v = [r["iou_filled"] for r in rs]
        cars.append({"car": car, "n": len(v), "mean": round(float(np.mean(v)), 3), "best": max(v), "worst": min(v),
                     "parts": round(float(np.mean([r["parts"] for r in rs])), 1),
                     "stops": dict(Counter(r["stats"]["stop"] for r in rs)), "ref": rs[0].get("ref_iou_filled")})
        print(f"{car:28s} mean {cars[-1]['mean']:.3f} best {max(v):.3f} worst {min(v):.3f} stops {cars[-1]['stops']}")
    agg = {"cars": cars, "mean_of_means": round(float(np.mean([c["mean"] for c in cars])), 3),
           "mean_of_best": round(float(np.mean([c["best"] for c in cars])), 3),
           "stops": dict(Counter(r["stats"]["stop"] for r in rows))}
    print(f"mean IoU {agg['mean_of_means']}  best-of-n {agg['mean_of_best']}  stops {agg['stops']}")
    (out / "report.json").write_text(json.dumps(agg, indent=1))


def render(rows, out: Path, per_sheet: int = 4, as_text: bool = True):
    from PIL import Image, ImageDraw, ImageFont
    from run_baselines import Renderer

    from car2brick.units import LDU_PER_PLATE, LDU_PER_STUD

    rnd = Renderer(260)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 16)
    for s in range(0, len(rows), per_sheet):
        chunk = rows[s : s + per_sheet]
        panels = []
        for r in chunk:
            desc = CarDescriptor.load(r["desc"])
            row = []
            ext = desc.extent()
            center = np.array([(e.start + e.stop) / 2 for e in ext]) * np.array([LDU_PER_STUD, LDU_PER_STUD,
                                                                                    LDU_PER_PLATE])
            half = 0.62 * desc.size()[0] * LDU_PER_STUD
            t, c = rnd.voxel_triangles(desc, as_text=as_text)
            row.append((f"目标描述 {'×'.join(map(str, r['size']))}", rnd.draw(t, c, center, half)))
            t, c = rnd.ldr_triangles((out / f"{r['name']}.ldr").read_text())
            title = (f"生成 {r['parts']}件 外形IoU {r['iou_filled']:.2f} 连通块 {r['components']} "
                     f"{r['seconds']:.0f}s")
            row.append((title, rnd.draw(t, c, center, half) if len(t) else np.full((260, 520, 3), 255, np.uint8)))
            if r.get("ref"):
                t, c = rnd.ldr_triangles(Path(r["ref"]).read_text())
                kind = "原车" if r["name"].startswith("val-") else "B2"
                title = f"{kind} {r['ref_parts']}件 外形IoU {r['ref_iou_filled']:.2f}"
                row.append((title, rnd.draw(t, c, center, half)))
            panels.append((r["name"], row))
        pw, ph = panels[0][1][0][1].shape[1], panels[0][1][0][1].shape[0]
        cols = max(len(p[1]) for p in panels)
        sheet = Image.new("RGB", (cols * (pw + 8), len(panels) * (ph + 34)), "white")
        d = ImageDraw.Draw(sheet)
        for i, (name, row) in enumerate(panels):
            for j, (title, img) in enumerate(row):
                x, y = j * (pw + 8), i * (ph + 34)
                d.text((x + 4, y + 2), f"{name} | {title}" if j == 0 else title, fill=(20, 20, 20), font=font)
                sheet.paste(Image.fromarray(img), (x, y + 30))
        sheet.save(out / f"sheet_{s // per_sheet:02d}.jpg", quality=88)



if __name__ == "__main__":
    main()
