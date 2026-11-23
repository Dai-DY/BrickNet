"""Build the retrieval library of LEGO cars from BrickNet graphs (largest component of each model).

Each candidate is stood stud-up (``library.upright``), filtered with the survey's rule (``library.car_like``), and
turned into a descriptor with real part geometry. Output (BrickNet-derived: keep it local, do not redistribute):

    <out>/index.jsonl          source, split, size [L, W, H], parts, wheels, colors, rough stats, upright axis
    <out>/desc/<source>.npz    descriptor
    <out>/ldr/<source>.ldr     the car in its descriptor grid frame (``evaluate.poses_to_ldr``)

    python scripts/build_car_library.py --graphs sft.npz pt.npz --out outputs/car_library --workers 16
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

import bricknet
from car2brick.evaluate import poses_to_ldr
from car2brick.geometry import PartShapes
from car2brick.lego import is_wheel, lego_car
from car2brick.library import car_like, rough_size, stud_up, upright

_G: dict = {}


def candidates(graphs) -> list[int]:
    out = []
    for i, g in enumerate(graphs):
        if not g.components:
            continue
        comp = g.components[0]
        if 20 <= len(comp) <= 600 and sum(is_wheel(g.part_ids[j]) for j in comp) >= 3:
            out.append(i)
    return out


def work(i: int):
    g, tag, out = _G["graphs"][i], _G["tag"], _G["out"]
    source = f"{tag}-{i:06d}"
    try:
        comp = sorted(g.components[0])
        T = g.transforms if g.transforms is not None else bricknet.decode_graph(g)
        pids = [g.part_ids[j] for j in comp]
        cols = [g.colors[j] for j in comp]
        known = bricknet.load_catalog().code_to_color
        cols = [c if c in known else 71 for c in cols]
        up = stud_up(pids, [T[j] for j in comp])
        tr = upright([T[j] for j in comp], up)
        s = rough_size(pids, tr)
        if not car_like(s, len(pids)):
            return None
        lc = lego_car(pids, cols, tr, shapes=_G["shapes"])
        L, W, H = lc.descriptor.size()
        if not (5 <= W <= 11 and L >= 1.5 * W):
            return None
        lc.descriptor.save(out / "desc" / f"{source}.npz")
        (out / "ldr" / f"{source}.ldr").write_text(poses_to_ldr(pids, cols, lc.grid_poses(), source))
        return {"source": source, "split": tag, "index": i, "size": [L, W, H], "parts": len(pids),
                "wheels": list(lc.descriptor.wheels), "colors": list(lc.descriptor.colors), "rough": s,
                "up": up.tolist()}
    except Exception as e:  # malformed graphs must not stop the run
        return {"source": source, "error": f"{type(e).__name__}: {e}"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/car_library")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="candidates per graph file (0 = all), for trial runs")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "desc").mkdir(parents=True, exist_ok=True)
    (out / "ldr").mkdir(parents=True, exist_ok=True)
    counts = {"candidates": 0, "kept": 0, "errors": 0}
    with open(out / "index.jsonl", "w") as index, open(out / "errors.jsonl", "w") as errors:
        for path in a.graphs:
            t = time.time()
            tag = Path(path).name.split(".")[0]
            graphs = bricknet.load_graphs(path)
            cand = candidates(graphs)
            if a.limit:
                cand = cand[: a.limit]
            print(f"{tag}: {len(graphs)} graphs, {len(cand)} candidates ({time.time() - t:.0f}s)", flush=True)
            counts["candidates"] += len(cand)
            _G.update(graphs=graphs, tag=tag, out=out, shapes=PartShapes())
            with mp.get_context("fork").Pool(a.workers) as pool:  # workers share the loaded graphs (copy on write)
                for n, row in enumerate(pool.imap_unordered(work, cand, chunksize=16)):
                    if row is None:
                        pass
                    elif "error" in row:
                        counts["errors"] += 1
                        errors.write(json.dumps(row) + "\n")
                    else:
                        counts["kept"] += 1
                        index.write(json.dumps(row) + "\n")
                    if (n + 1) % 1000 == 0:
                        print(f"  {tag} {n + 1}/{len(cand)} {counts} ({time.time() - t:.0f}s)", flush=True)
            del graphs
            _G.clear()
            print(f"{tag} done {counts} ({time.time() - t:.0f}s)", flush=True)
    print("DONE", counts, flush=True)


if __name__ == "__main__":
    main()
