"""python -m hybrid <command>

    search   find ShapeNet cars by name                 python -m hybrid search veyron
    source   build (and cache) one source solid, figure python -m hybrid source e6ec389325de7ae5becf71e2e014ff6f
    compose  front + body + rear -> hybrid car           python -m hybrid compose --front <id> --body <id> --rear <id>

A car is given as a ShapeNet id, optionally with ``:flip`` when the model faces backwards.
Outputs of ``compose`` (in <out>/<name>/): sources.jpg, layout.png, hybrid.jpg, turntable.gif, hybrid.glb,
hybrid.ply, summary.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .compose import Hybrid, Plan
from .io import load_shapenet, names, search
from .render import MeshRenderer, grid, label, orbit, to_img
from .solid import SourceCar, build_source

PART_LABEL = {"front": "FRONT", "body": "BODY", "rear": "REAR"}
PART_COLOR = {"front": "#1f77b4", "body": "#555555", "rear": "#d62728"}


def get_source(spec: str, args, log=print) -> SourceCar:
    sid, _, flag = spec.partition(":")
    flip = flag == "flip"
    cache = Path(args.cache) / f"{sid}{'_flip' if flip else ''}_{args.width:.2f}_{args.voxel * 1000:.0f}mm.npz"
    if cache.is_file() and not args.rebuild:
        return SourceCar.load(cache)
    v, f, rgb = load_shapenet(sid, flip=flip)
    src = build_source(sid, names().get(sid, sid), v, f, rgb, width=args.width, vs=args.voxel, log=log)
    cache.parent.mkdir(parents=True, exist_ok=True)
    src.save(cache)
    return src


def view_set(lo, hi, res=(640, 420)):
    c = (lo + hi) / 2
    c[2] = (hi[2] - lo[2]) * 0.4
    rad = float(np.linalg.norm(hi - lo) / 2)
    return [(name, orbit(az, el, 2.6 * rad, c, res, radius=rad)) for name, az, el in
            (("front 3/4", 35, 14), ("side", 90, 4), ("rear 3/4", 215, 16), ("top", 140, 55))]


def sources_figure(hy: Hybrid, path: Path):
    tiles = []
    for name, pl in hy.parts().items():
        src = pl.src
        xt = pl.to_target_x(src.tri.mean(1)[:, 0])
        used = hy.weights(xt)[name] > 0.5
        rgb = np.where(used[:, None], src.rgb, 0.25 * src.rgb + 0.68)  # unused part greyed out
        lo, hi = src.tri.reshape(-1, 3).min(0), src.tri.reshape(-1, 3).max(0)
        img = MeshRenderer(src.tri, rgb).render(view_set(lo, hi, (520, 340))[0][1])
        tiles.append(label(to_img(img), f"{PART_LABEL[name]}: {src.name[:40]}"))
    grid(tiles, 3).save(path, quality=92)


def layout_figure(hy: Hybrid, v: np.ndarray, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3.6))
    # hybrid side silhouette from the mesh vertices (x, z), filled per x bin
    xb = np.linspace(v[:, 0].min(), v[:, 0].max(), 400)
    i = np.clip(np.digitize(v[:, 0], xb) - 1, 0, len(xb) - 2)
    top = np.full(len(xb) - 1, np.nan)
    bot = np.full(len(xb) - 1, np.nan)
    np.fmax.at(top, i, v[:, 2])
    np.fmin.at(bot, i, v[:, 2])
    xc = (xb[1:] + xb[:-1]) / 2
    ax.fill_between(xc, bot, top, color="0.85", label="hybrid")
    for name, pl in hy.parts().items():
        pr = pl.src.profiles()
        xt = pl.to_target_x(pr["x"])
        sz = pl.scales(xt)[1] if pl.match else 1.0
        ax.plot(xt, pr["top"] * sz, color=PART_COLOR[name], lw=1.3, label=f"{PART_LABEL[name]}: {pl.src.name[:28]}")
    for x, s in ((hy.seam_f, "seam"), (hy.seam_r, "seam")):
        ax.axvspan(x - hy.plan.band / 2, x + hy.plan.band / 2, color="orange", alpha=0.25)
        ax.text(x, 0.02, s, ha="center", fontsize=8)
    for a in hy.body.axles:
        ax.plot([a], [0.0], "k^", ms=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, forward ->)")
    ax.set_ylabel("z (m)")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cmd_compose(a):
    t0 = time.time()
    out = Path(a.out) / (a.name or "hybrid")
    out.mkdir(parents=True, exist_ok=True)
    print("sources")
    front, body, rear = (get_source(s, a) for s in (a.front, a.body, a.rear))
    plan = Plan(front_cut=a.front_cut, rear_cut=a.rear_cut, band=a.band, match=a.match, smooth=a.smooth, fill=a.fill)
    hy = Hybrid(front, body, rear, plan)
    print("compose")
    sdf, sdf_lo, sdf_vs = hy.field()
    np.savez_compressed(out / "hybrid_sdf.npz", sdf=sdf.astype(np.float16), lo=sdf_lo, vs=sdf_vs,
                        body_axles=np.array(hy.body.axles))  # the solid, for brickcar
    v, f, col = hy.mesh(field=(sdf, sdf_lo, sdf_vs))
    fcol = col[f].mean(1)
    rend = MeshRenderer(v[f], fcol)
    lo, hi = v.min(0), v.max(0)
    print(f"  hybrid: {np.ptp(v, 0).round(2)} m, {time.time() - t0:.1f}s")

    import trimesh

    tm = trimesh.Trimesh(v, f, vertex_colors=(np.c_[col, np.ones(len(col))] * 255).astype(np.uint8), process=False)
    tm.export(out / "hybrid.glb")
    tm.export(out / "hybrid.ply")
    sources_figure(hy, out / "sources.jpg")
    layout_figure(hy, v, out / "layout.png")
    tiles = [label(to_img(rend.render(cam)), n) for n, cam in view_set(lo, hi)]
    grid(tiles, 2).save(out / "hybrid.jpg", quality=92)
    if not a.no_turntable:
        c = (lo + hi) / 2
        c[2] = (hi[2] - lo[2]) * 0.4
        rad = float(np.linalg.norm(hi - lo) / 2)
        frames = [to_img(rend.render(orbit(az, 15, 2.6 * rad, c, (480, 300), radius=rad)))
                  for az in range(0, 360, 10)]
        frames[0].save(out / "turntable.gif", save_all=True, append_images=frames[1:], duration=90, loop=0)
    summary = {**hy.summary(), "size_m": np.ptp(v, 0).round(3).tolist(), "vertices": len(v), "triangles": len(f),
               "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"-> {out}  ({time.time() - t0:.1f}s)")


def cmd_source(a):
    src = get_source(a.car, a)
    v = src.tri.reshape(-1, 3)
    tiles = [label(to_img(MeshRenderer(src.tri, src.rgb).render(cam)), n) for n, cam in view_set(v.min(0), v.max(0))]
    out = Path(a.out) / "sources"
    out.mkdir(parents=True, exist_ok=True)
    grid(tiles, 2).save(out / f"{src.sid}.jpg", quality=90)
    print(f"{src.name}: axles {src.axles}, -> {out / (src.sid + '.jpg')}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m hybrid", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("keyword")

    def common(p):
        p.add_argument("--out", default="outputs/hybrid")
        p.add_argument("--cache", default="outputs/hybrid/cache")
        p.add_argument("--width", type=float, default=1.95, help="every source is scaled to this body width (m)")
        p.add_argument("--voxel", type=float, default=0.012, help="solid / SDF resolution (m)")
        p.add_argument("--rebuild", action="store_true", help="ignore cached solids")

    p = sub.add_parser("source")
    p.add_argument("car")
    common(p)
    c = sub.add_parser("compose")
    c.add_argument("--front", required=True)
    c.add_argument("--body", required=True)
    c.add_argument("--rear", required=True)
    c.add_argument("--name", default="")
    c.add_argument("--front-cut", type=float, default=Plan.front_cut, help="seam, m behind the front axle")
    c.add_argument("--rear-cut", type=float, default=Plan.rear_cut, help="seam, m ahead of the rear axle")
    c.add_argument("--band", type=float, default=Plan.band, help="blend width at each seam (m)")
    c.add_argument("--match", type=float, default=Plan.match, help="cross-section matching strength 0..1")
    c.add_argument("--smooth", type=float, default=Plan.smooth, help="SDF smoothing (voxels)")
    c.add_argument("--fill", type=float, default=Plan.fill, help="anti-pinch union at the band centres 0..1")
    c.add_argument("--no-turntable", action="store_true")
    common(c)
    a = ap.parse_args(argv)
    if a.cmd == "search":
        for sid, n in search(a.keyword):
            print(sid, n)
    elif a.cmd == "source":
        cmd_source(a)
    else:
        cmd_compose(a)


if __name__ == "__main__":
    main()
