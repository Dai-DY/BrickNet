"""python -m brickcar build <hybrid dir> [--width-studs 14]

Turns a hybrid car (``python -m hybrid compose`` output folder) into a LEGO model built from real parts. Outputs in
<out>/<name>_w<width>/: model.ldr (open in BrickLink Studio / LDView / LeoCAD), bom.csv, views.jpg (four views of
the real part meshes), compare.jpg (hybrid car vs LEGO, same cameras), info.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hybrid.render import MeshRenderer, grid, label, orbit, to_img

from .build import Config, Designer
from .model import save
from .target import load_target


def views(lo, hi, res=(640, 420)):
    c = (lo + hi) / 2
    c[2] = (hi[2] - lo[2]) * 0.42
    rad = float(np.linalg.norm(hi - lo) / 2)
    return [(n, orbit(az, el, 2.5 * rad, c, res, radius=rad)) for n, az, el in
            (("front 3/4", 35, 16), ("side", 90, 5), ("rear 3/4", 215, 18), ("top", 140, 58))]


def cmd_build(a):
    t0 = time.time()
    tgt = load_target(Path(a.hybrid), a.width_studs)
    k = 20 / tgt.stud
    print(f"target {tgt.name}: {tgt.L} x {tgt.W} studs, {tgt.H} plates, 1 stud = {tgt.stud * 100:.1f} cm "
          f"(scale 1:{tgt.stud / 0.008:.0f}), wheels " +
          ", ".join(f"r {w.radius:.2f} m w {w.width:.2f} m" for w in tgt.wheels))
    d = Designer(tgt, Config(flip_wheels=a.flip_wheels)).run()
    out = Path(a.out) / f"{tgt.name}_w{a.width_studs}"
    save(d.model, out, tgt.name)
    print(f"design: {d.info['parts']} parts, {d.info['components']} connected group(s) "
          f"(largest {d.info['largest_component']}), caps {d.info['caps']}, {time.time() - t0:.1f}s")
    tri, rgb = d.model.triangles()
    lego = MeshRenderer(tri, rgb)
    hv = np.c_[(tgt.verts[:, 0] - tgt.x0) * k, (tgt.verts[:, 1] + tgt.W * tgt.stud / 2) * k, tgt.verts[:, 2] * k]
    import trimesh

    hm = trimesh.load(Path(a.hybrid) / "hybrid.glb", force="mesh")
    f = np.asarray(hm.faces)
    hyb = MeshRenderer(hv[f], tgt.vcol[f].mean(1))
    v = tri.reshape(-1, 3)
    cams = views(np.minimum(v.min(0), hv.min(0)), np.maximum(v.max(0), hv.max(0)))
    grid([label(to_img(lego.render(c)), n) for n, c in cams], 2).save(out / "views.jpg", quality=92)
    tiles = []
    for n, c in cams[:3]:
        tiles += [label(to_img(hyb.render(c)), f"hybrid - {n}"), label(to_img(lego.render(c)), f"LEGO - {n}")]
    grid(tiles, 2).save(out / "compare.jpg", quality=92)
    if not a.no_turntable:
        lo, hi = v.min(0), v.max(0)
        c = (lo + hi) / 2
        c[2] = (hi[2] - lo[2]) * 0.42
        rad = float(np.linalg.norm(hi - lo) / 2)
        frames = [to_img(lego.render(orbit(az, 18, 2.4 * rad, c, (520, 340), radius=rad))) for az in range(0, 360, 10)]
        frames[0].save(out / "turntable.gif", save_all=True, append_images=frames[1:], duration=90, loop=0)
    info = {**d.info, "scale": f"1:{tgt.stud / 0.008:.0f}", "stud_cm": round(tgt.stud * 100, 2),
            "seconds": round(time.time() - t0, 1)}
    (out / "info.json").write_text(json.dumps(info, indent=1, default=float))
    print(f"-> {out} ({time.time() - t0:.1f}s)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m brickcar", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("hybrid", help="hybrid output folder (with hybrid_sdf.npz, hybrid.glb, summary.json)")
    b.add_argument("--width-studs", type=int, default=14)
    b.add_argument("--out", default="outputs/brickcar")
    b.add_argument("--flip-wheels", action="store_true", help="turn the wheels' decorated face inwards/outwards")
    b.add_argument("--no-turntable", action="store_true")
    a = ap.parse_args(argv)
    cmd_build(a)


if __name__ == "__main__":
    main()
