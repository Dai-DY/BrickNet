#!/usr/bin/env python3
"""B3 module-wise kNN assembly baseline (``car2brick.assemble``): index the car library, rebuild targets from
modules of other cars, score and render them.

    python scripts/run_assemble.py --build-index --library outputs/car_library        # once (~3 min, 16 workers)
    python scripts/run_assemble.py --data outputs/data/c2b_v3_tf --sources pt-169087 --n-val 8 \
        --shapenet outputs/baselines/shapenet --n-shapenet 4 --out outputs/baselines/assemble

Same targets as ``eval_generate.py``. A validation car is never rebuilt from itself: its own source is excluded, and
so is every library car whose outer shape matches the target with IoU >= ``--dedup`` (copies of the same model).
Outputs: <out>/<name>.ldr, <name>_modules.ldr (coloured by module), summary.jsonl (with each module's source),
sheet_*.jpg.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from car2brick.assemble import (MODULE_COLORS, ModuleLibrary, assemble, build_index, describe,  # noqa: E402
                                swap_assemble)
from car2brick.descriptor import CarDescriptor  # noqa: E402
from car2brick.evaluate import evaluate_ldr, poses_to_ldr  # noqa: E402
from car2brick.geometry import PartShapes  # noqa: E402
from eval_generate import targets  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default="outputs/car_library")
    ap.add_argument("--index", default=None, help="module index (default <library>/modules.pkl)")
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--data", default=None)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--n-val", type=int, default=6)
    ap.add_argument("--shapenet", default=None)
    ap.add_argument("--n-shapenet", type=int, default=6)
    ap.add_argument("--mode", default="swap", choices=("swap", "modules"),
                    help="swap: best whole car, then replace modules (B3b); modules: build from modules only (B3a)")
    ap.add_argument("--bases", type=int, default=5, help="swap: whole cars to start from")
    ap.add_argument("--k", type=int, default=None, help="candidates per module (default swap 8, modules 12)")
    ap.add_argument("--floating", action="store_true", help="keep parts that are not joined to the model")
    ap.add_argument("--dedup", type=float, default=0.9)
    ap.add_argument("--no-recolor", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    index = Path(a.index or Path(a.library) / "modules.pkl")
    if a.build_index:
        t = time.time()
        lib = build_index(a.library, workers=a.workers)
        lib.save(index)
        print(f"indexed {len(lib.cars)} cars in {time.time() - t:.0f}s -> {index}")
    if not a.out:
        return
    lib = ModuleLibrary.load(index)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shapes = PartShapes()
    rows = []
    with open(out / "summary.jsonl", "w") as fh:
        for name, desc_path, ref in targets(a):
            desc = CarDescriptor.load(desc_path)
            own = {name.split("-", 1)[1]} if name.startswith("val-") else set()
            t = time.time()
            if a.mode == "swap":
                asm = swap_assemble(lib, desc, shapes, k=a.k or 8, bases=a.bases, exclude=own, dedup=a.dedup,
                                    recolor=not a.no_recolor)
            else:
                asm = assemble(lib, desc, shapes, k=a.k or 12, exclude=own, dedup=a.dedup, recolor=not a.no_recolor,
                               connected=not a.floating)
            ldr = poses_to_ldr(asm.part_ids, asm.colors, asm.poses, f"assembled {name}")
            (out / f"{name}.ldr").write_text(ldr)
            by_mod = [MODULE_COLORS[m] for m in asm.modules]
            (out / f"{name}_modules.ldr").write_text(poses_to_ldr(asm.part_ids, by_mod, asm.poses, name))
            ev = evaluate_ldr(ldr, desc) if asm.part_ids else {"parts": 0, "iou_filled": 0.0, "components": 0}
            row = {"name": name, "desc": str(desc_path), "size": list(desc.size()),
                   "seconds": round(time.time() - t, 1),
                   "parts": ev["parts"], "iou": round(float(ev.get("iou", 0.0)), 3),
                   "iou_filled": round(float(ev["iou_filled"]), 3), "components": ev["components"],
                   "modules": describe(lib, asm), "pool": asm.info["pool"],
                   "dedup_dropped": asm.info["dedup_dropped"], "missing": asm.info.get("missing", []),
                   "bases": asm.info.get("bases"), "ref": ref}
            if ref:
                rv = evaluate_ldr(Path(ref).read_text(), desc)
                row["ref_iou_filled"] = round(float(rv["iou_filled"]), 3)
                row["ref_parts"] = rv["parts"]
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            mods = " ".join(f"{d['name']}{'' if d.get('action') in (None, 'replaced') else '(' + d['action'] + ')'}"
                            f"={d['source']}" for d in row["modules"])
            print(f"{name}: {row['parts']} parts, IoU {row['iou_filled']:.3f}, {row['components']} comp | {mods}",
                  flush=True)
    for key in ("iou_filled", "parts", "components", "seconds"):
        print(f"{key}: mean {np.mean([r[key] for r in rows]):.3f}")
    if not a.no_render:
        render(rows, out)


def render(rows, out: Path, per_sheet: int = 4):
    from PIL import Image, ImageDraw, ImageFont
    from run_baselines import Renderer

    from car2brick.units import LDU_PER_PLATE, LDU_PER_STUD

    rnd = Renderer(240)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18)
    legend = "灰底盘 黑轮胎 蓝座舱 红端A 黄端B 绿玻璃 紫尾翼"  # MODULE_COLORS
    for s in range(0, len(rows), per_sheet):
        panels = []
        for r in rows[s : s + per_sheet]:
            desc = CarDescriptor.load(r["desc"])
            ext = desc.extent()
            center = np.array([(e.start + e.stop) / 2 for e in ext]) * np.array([LDU_PER_STUD, LDU_PER_STUD,
                                                                                    LDU_PER_PLATE])
            half = 0.62 * desc.size()[0] * LDU_PER_STUD
            row = [(f"{r['name']} 目标 {'×'.join(map(str, r['size']))}",
                    rnd.draw(*rnd.voxel_triangles(desc, as_text=True), center, half))]
            for fn, title in ((f"{r['name']}.ldr", f"拼装 {r['parts']}件 IoU {r['iou_filled']:.2f} "
                                                    f"连通块 {r['components']}"),
                              (f"{r['name']}_modules.ldr", "按模块着色: " + legend)):
                t, c = rnd.ldr_triangles((out / fn).read_text())
                row.append((title, rnd.draw(t, c, center, half) if len(t) else np.full((240, 480, 3), 255, np.uint8)))
            if r.get("ref"):
                t, c = rnd.ldr_triangles(Path(r["ref"]).read_text())
                kind = "原车" if r["name"].startswith("val-") else "B2"
                row.append((f"{kind} {r['ref_parts']}件 IoU {r['ref_iou_filled']:.2f}", rnd.draw(t, c, center, half)))
            panels.append(row)
        pw, ph = panels[0][0][1].shape[1], panels[0][0][1].shape[0]
        ncol = max(len(p) for p in panels)
        sheet = Image.new("RGB", (ncol * (pw + 8), len(panels) * (ph + 34)), "white")
        d = ImageDraw.Draw(sheet)
        for i, row in enumerate(panels):
            for j, (title, img) in enumerate(row):
                x, y = j * (pw + 8), i * (ph + 34)
                d.text((x + 4, y + 4), title, fill=(20, 20, 20), font=font)
                sheet.paste(Image.fromarray(img), (x, y + 30))
        sheet.save(out / f"sheet_{s // per_sheet:02d}.jpg", quality=88)


if __name__ == "__main__":
    main()
