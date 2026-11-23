"""Training-free baselines on ShapeNet cars: mesh -> descriptor -> B0 / B1 / B2 / retrieval -> metrics + sheets.

For every car: ``mesh_descriptor`` (front half on a clean mesh), main / secondary colour from the OBJ materials
(area-weighted, glass ignored), B0 (plates) and B1 (plates + tiles), then IoU / part count / connected components.
Sheets show, per car: input mesh | descriptor | B0 | B1, each as an oblique and a side view, all in the
descriptor grid frame and at one scale per car. LEGO models are drawn with BrickNet's GLB part meshes using the
torch rasteriser from the sibling ``shape2brick`` project (must be importable in the environment).

    python scripts/run_baselines.py --shapenet <ShapeNetCore.v2/02958343> --index <shapenet.csv> \
        --out outputs/baselines/shapenet --pick default
Outputs: <out>/<name>/{desc.npz,desc.txt,b0.ldr,b1.ldr,metrics.json}, <out>/summary.{jsonl,md}, <out>/sheet_*.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import bricknet
from car2brick.baseline import baseline_ldr
from car2brick.canonical import mesh_descriptor
from car2brick.colors import _PALETTE_HEX, GLASS, class_of_rgb, palette, quantize
from car2brick.descriptor import with_surface_samples
from car2brick.evaluate import evaluate_ldr, poses_to_ldr, recolor
from car2brick.units import CAR_FROM_LD, LDU_PER_PLATE, LDU_PER_STUD

# (label, keyword in the ShapeNet model name): sports cars, sedans, hatchbacks, SUVs / off-road, pickups, classics
DEFAULT_PICK = [
    ("ferrari_f430", "Ferrari F430"), ("bugatti_veyron", "Bugatti Veyron"), ("nissan_370z", "Nissan 370z"),
    ("audi_tt", "Audi TT"), ("porsche_911", "Porsche 911"), ("lamborghini", "Lamborghini"),
    ("corvette", "Corvette"), ("camaro", "Camaro"), ("mustang", "Mustang"), ("ford_gt", "Ford GT"),
    ("bmw_m6", "BMW M6"), ("audi_a6", "Audi A6"), ("honda_civic", "Honda Civic"), ("ford_focus", "Ford Focus"),
    ("prius", "Prius"), ("mini_cooper", "Mini Cooper"), ("vw_beetle", "Beetle"), ("jeep_wrangler", "Jeep Wrangler"),
    ("hummer_h1", "Hummer H1"), ("audi_q7", "Audi Q7"), ("land_rover", "Land Rover"), ("f150", "F-150"),
    ("cadillac", "Cadillac"), ("smart", "Smart"),
]


# ---- OBJ with materials -----------------------------------------------------------------------------------------
def read_mtl(path: Path) -> dict[str, tuple[np.ndarray, float]]:
    out, name = {}, None
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "newmtl":
            name = " ".join(t[1:])
            out[name] = (np.array([0.6, 0.6, 0.6]), 1.0)
        elif name and t[0] == "Kd" and len(t) >= 4:
            out[name] = (np.array([float(v) for v in t[1:4]]), out[name][1])
        elif name and t[0] in ("d", "Tr") and len(t) >= 2:
            d = float(t[1])
            out[name] = (out[name][0], d if t[0] == "d" else 1.0 - d)
    return out


def read_obj(path: Path):
    """-> vertices (V,3), faces (F,3) triangulated, per-face rgb (F,3) in [0,1], per-face opacity (F,)."""
    mats: dict[str, tuple[np.ndarray, float]] = {}
    verts, faces, fmat = [], [], []
    cur = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("v "):
            verts.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            for k in range(1, len(idx) - 1):
                faces.append([idx[0], idx[k], idx[k + 1]])
                fmat.append(cur)
        elif line.startswith("usemtl"):
            cur = line.split(maxsplit=1)[1].strip() if len(line.split()) > 1 else None
        elif line.startswith("mtllib"):
            mats.update(read_mtl(path.parent / line.split(maxsplit=1)[1].strip()))
    names = sorted({m for m in fmat if m is not None})
    table = {m: mats.get(m, (np.array([0.6, 0.6, 0.6]), 1.0)) for m in names}
    rgb = np.array([table[m][0] if m in table else (0.6, 0.6, 0.6) for m in fmat], dtype=np.float64)
    alpha = np.array([table[m][1] if m in table else 1.0 for m in fmat], dtype=np.float64)
    return np.asarray(verts, np.float64), np.asarray(faces, np.int64), np.clip(rgb, 0, 1), alpha


def outer_faces(v, f):
    """ShapeNet surfaces are mostly modelled twice with opposite winding (paint outside, often black inside), which
    z-fights when rendered and drowns the paint colour. Keep one face per coincident group: the one whose normal
    points away from the model centre. Returns a boolean mask over faces."""
    tri = v[f]
    key = np.round(np.sort(tri, axis=1).reshape(len(f), -1), 5)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = (n * (tri.mean(1) - v.mean(0))).sum(1)
    order = np.lexsort((-out, inv))  # per group, most outward first
    keep = np.zeros(len(f), bool)
    first = np.ones(len(order), bool)
    first[1:] = inv[order[1:]] != inv[order[:-1]]
    keep[order[first]] = True
    return keep


def shell_faces(v, f, depth: float = 0.06):
    """Faces near the outer shell (within ``depth`` x model width of the convex hull): drops seats, dashboards and
    engine parts that would otherwise vote in the body colour."""
    from scipy.spatial import ConvexHull

    eq = ConvexHull(v).equations  # n . x + b <= 0 inside, |n| = 1
    c = v[f].mean(1)
    d = -(c @ eq[:, :3].T + eq[:, 3]).max(1)  # depth below the hull surface
    return d <= depth * float(np.ptp(v, 0).min())


def body_colors(v, f, rgb, alpha, k: int = 2, min_share: float = 0.08) -> tuple[str, ...]:
    """Area-weighted dominant LEGO colours of the opaque faces."""
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    keep = alpha > 0.9
    if not keep.any():
        return ()
    names, _ = palette()
    w = np.bincount(quantize(rgb[keep] * 255), weights=area[keep], minlength=len(names))
    order = np.argsort(w)[::-1]
    out = [names[order[0]]]
    out += [names[i] for i in order[1:k] if w[i] >= min_share * w.sum()]
    return tuple(out)


def coloured_samples(v, f, rgb, count: int = 60000, seed: int = 0, return_faces: bool = False):
    """Area-uniform surface samples with their face colour (0-255) (and face index)."""
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(f), count, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(count)), rng.random(count)
    w = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
    out = (tri[idx] * w[:, :, None]).sum(1), rgb[idx] * 255
    return (*out, idx) if return_faces else out


# ---- rendering ----------------------------------------------------------------------------------------------------
class Renderer:
    """Everything is drawn in the descriptor grid frame (x length, y width, z up; LDU)."""

    VIEWS = ((-50.0, 22.0), (0.0, 0.0))  # oblique (front-left, from above), side

    def __init__(self, res: int = 300):
        from shape2brick.ldraw.shape import PartMeshCache
        from shape2brick.render import ldraw_colors

        self.res = res
        self.cache = PartMeshCache()
        ld = Path(__file__).resolve().parents[2] / "shape2brick" / "data" / "raw" / "ldraw" / "LDConfig.ldr"
        self.colors = ldraw_colors(str(ld))
        self.cat = bricknet.load_catalog()

    def draw(self, tri_grid: np.ndarray, rgb: np.ndarray, center: np.ndarray, half: float) -> np.ndarray:
        from shape2brick.render import render_triangles

        p = (tri_grid.reshape(-1, 3) - center) / half
        yup = np.stack([p[:, 0], p[:, 2], -p[:, 1]], 1).reshape(-1, 3, 3).astype(np.float32)
        views = tuple((az, el, 1.0) for az, el in self.VIEWS)
        imgs, _ = render_triangles(yup, rgb.astype(np.float32), views=views, res=self.res)
        return np.concatenate(list(imgs), 1)

    def ldr_triangles(self, ldr_text: str):
        from shape2brick.render import assemble_triangles

        g = bricknet.parse_ldr(ldr_text)
        stems = [self.cat.id_to_stem[p] for p in g.part_ids]
        tri, rgb = assemble_triangles(stems, [np.asarray(t, float) for t in g.transforms], list(g.colors),
                                      self.cache, self.colors)
        return (tri.reshape(-1, 3) @ CAR_FROM_LD.T).reshape(-1, 3, 3), rgb  # baseline LDraw = grid frame

    @staticmethod
    def voxel_triangles(desc, as_text: bool = False):
        """Cells of a descriptor. ``as_text``: what the v2 text says - its profile (with gaps) coloured by class."""
        from car2brick.render import voxel_triangles

        if as_text:
            from car2brick.colors import _CLASS_PROTOTYPES, CLASSES

            solid, _ = desc.segments()
            cls = desc.cell_classes(solid)
            hexes = {**{c: h[0] for c, h in _CLASS_PROTOTYPES.items()}, "t": "9DC3F7", "l": "F5E050"}
            rgb = {i: np.array([int(hexes[c][n : n + 2], 16) for n in (0, 2, 4)]) / 255.0
                   for i, c in enumerate(CLASSES)}
            t, col, _ = voxel_triangles(solid, lambda i, j, k: rgb[int(cls[i, j, k])])
            return t, np.asarray(col, np.float64)
        solid = desc.symmetrized().solid()
        main = desc.colors[0] if desc.colors else "light bluish gray"
        c = np.array([int(_PALETTE_HEX.get(main, "A0A5A9")[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0
        wheel_cells = set()
        for wx in desc.wheels:
            for i in (int(np.floor(wx)), int(np.ceil(wx)) - 1):
                wheel_cells.add(i)

        def cell_rgb(i, j, k):
            return (0.25, 0.25, 0.28) if (i in wheel_cells and k <= 4) else c

        t, col, _ = voxel_triangles(solid, cell_rgb)
        return t, np.asarray(col, np.float64)


# ---- main ---------------------------------------------------------------------------------------------------------
def pick_models(index: Path, root: Path, picks) -> list[tuple[str, str, str]]:
    rows = [r for r in csv.DictReader(open(index)) if r.get("v2_has_mesh") == "1" and r.get("name")]
    out = []
    for label, kw in picks:
        m = [r for r in rows if kw.lower() in r["name"].lower() and (root / r["id"] / "models").is_dir()]
        if m:
            out.append((label, m[0]["id"], m[0]["name"]))
        else:
            print(f"[skip] no ShapeNet model matches {kw!r}")
    return out


METHODS = (("b0", "B0 铺板"), ("b1", "B1 铺板+tile"), ("b2", "B2 逐件拟合"), ("ret", "检索"))


def run_one(label, sid, name, root: Path, out: Path, lib=None) -> dict:
    d = out / label
    d.mkdir(parents=True, exist_ok=True)
    obj = root / sid / "models" / "model_normalized.obj"
    v, f, rgb, alpha = read_obj(obj)
    import trimesh

    from car2brick.recon.semantics import wheels_from_dark_points

    keep = outer_faces(v, f)
    f, rgb, alpha = f[keep], rgb[keep], alpha[keep]
    shell = shell_faces(v, f)
    # colours are approximate: dark trim / tinted glass modelled as opaque can still win (e.g. a red Mini)
    colors = body_colors(v, f[shell], rgb[shell], alpha[shell])
    desc, canon = mesh_descriptor(trimesh.Trimesh(v, f, process=False), up_hint=(0, 1, 0), colors=colors)
    # tyres by colour, as in the photo pipeline: the filled interior hides the gap under the car
    pts, cols, fi = coloured_samples(v, f[shell], rgb[shell], return_faces=True)
    wx = wheels_from_dark_points(canon.apply(pts), cols)
    if wx:
        desc.wheels = tuple(float(round((x - desc.origin[0]) / LDU_PER_STUD, 1)) for x in wx)
    # surface colour classes of the outer shell; see-through faces are glass
    cls = np.where(alpha[shell][fi] < 0.9, GLASS, class_of_rgb(cols))
    desc = with_surface_samples(desc, canon.apply(pts), cls)
    desc.save(d / "desc.npz")
    (d / "desc.txt").write_text(desc.to_text("official") + "\n")
    row = {"label": label, "shapenet_id": sid, "name": name, "size": list(desc.size()),
           "wheels": list(desc.wheels), "colors": list(colors)}
    for key, tiles in (("b0", False), ("b1", True)):
        ldr = baseline_ldr(desc, cap_tiles=tiles)
        (d / f"{key}.ldr").write_text(ldr)
        ev = evaluate_ldr(ldr, desc)
        row[key] = {"parts": ev["parts"], "iou": round(float(ev["iou"]), 3),
                    "iou_filled": round(float(ev["iou_filled"]), 3), "components": ev["components"]}
    from car2brick.fit import fit_ldr

    ldr, res = fit_ldr(desc)
    (d / "b2.ldr").write_text(ldr)
    ev = evaluate_ldr(ldr, desc)
    row["b2"] = {"parts": ev["parts"], "iou": round(float(ev["iou"]), 3),
                 "iou_filled": round(float(ev["iou_filled"]), 3), "components": ev["components"],
                 "kinds": res.stats["kinds"]}
    if lib is not None:
        from car2brick.library import place

        top = lib.retrieve(desc, k=5)
        pids, cols, poses = place(top[0], desc)
        ldr = poses_to_ldr(pids, recolor(pids, cols, desc.colors), poses, f"retrieved {top[0]['car'].source}")
        (d / "ret.ldr").write_text(ldr)
        ev = evaluate_ldr(ldr, desc)
        row["ret"] = {"parts": ev["parts"], "iou": round(float(ev["iou"]), 3),
                      "iou_filled": round(float(ev["iou_filled"]), 3), "components": ev["components"],
                      "source": top[0]["car"].source, "match_iou": round(top[0]["iou"], 3),
                      "top5": [(m["car"].source, round(m["iou"], 3)) for m in top]}
    (d / "metrics.json").write_text(json.dumps(row, indent=1, ensure_ascii=False))
    # keep what the sheet needs
    row["_mesh"] = (canon.apply(v) - desc.origin, f, rgb, alpha)
    row["_desc"] = desc
    return row


def sheet_panels(row, rnd: Renderer, d: Path) -> list[tuple[str, np.ndarray]]:
    desc = row["_desc"]
    L, W, H = desc.size()
    ext = desc.extent()
    center = np.array([(s.start + s.stop) / 2 for s in ext]) * np.array([LDU_PER_STUD, LDU_PER_STUD, LDU_PER_PLATE])
    half = 0.62 * max(L * LDU_PER_STUD, 1.0)
    v, f, rgb, alpha = row["_mesh"]
    shade = np.where(alpha[:, None] > 0.9, rgb, rgb * 0.5 + 0.35)  # glass drawn lighter
    panels = [(f"ShapeNet：{row['name'][:34]}", rnd.draw(v[f], shade, center, half))]
    t, c = rnd.voxel_triangles(desc)
    panels.append((f"描述 {L}×{W}×{H}  轮 {', '.join(f'{w:g}' for w in desc.wheels)}  {'/'.join(desc.colors)}",
                   rnd.draw(t, c, center, half)))
    for key, title in METHODS:
        if key not in row:
            continue
        t, c = rnd.ldr_triangles((d / row["label"] / f"{key}.ldr").read_text())
        m = row[key]
        extra = f"  {m['source']}" if "source" in m else f"  连通块 {m['components']}"
        panels.append((f"{title}  {m['parts']}件  IoU {m['iou']:.2f}/{m['iou_filled']:.2f}{extra}",
                       rnd.draw(t, c, center, half)))
    return panels


def write_sheets(rows, rnd: Renderer, out: Path, per_sheet: int = 4):
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 17)
    paths = []
    for s in range(0, len(rows), per_sheet):
        chunk = rows[s : s + per_sheet]
        all_panels = [sheet_panels(r, rnd, out) for r in chunk]
        pw, ph = all_panels[0][0][1].shape[1], all_panels[0][0][1].shape[0]
        cols = len(all_panels[0])
        gap, title_h = 8, 26
        sheet = Image.new("RGB", (cols * (pw + gap), len(chunk) * (ph + title_h + gap)), "white")
        draw = ImageDraw.Draw(sheet)
        for r, panels in enumerate(all_panels):
            for c, (title, img) in enumerate(panels):
                x, y = c * (pw + gap), r * (ph + title_h + gap)
                draw.text((x + 6, y + 3), title, fill=(20, 20, 20), font=font)
                sheet.paste(Image.fromarray(img), (x, y + title_h))
                draw.rectangle([x, y, x + pw - 1, y + title_h + ph - 1], outline=(215, 215, 215))
        p = out / f"sheet_{s // per_sheet:02d}.jpg"
        sheet.save(p, quality=88)
        paths.append(p)
        print(f"saved {p}", flush=True)
    return paths


def write_summary(rows, out: Path):
    with open(out / "summary.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
    keys = [k for k, _ in METHODS if rows and k in rows[0]]
    head = " | ".join(f"{k.upper()} parts / IoU / filled / comps" for k in keys)
    lines = ["| car | size (L×W×H) | wheels | colors | " + head
             + " |", "|---" * (4 + len(keys)) + "|"]
    for r in rows:
        cells = [f"{r[k]['parts']} / {r[k]['iou']:.2f} / {r[k]['iou_filled']:.2f} / {r[k]['components']}" for k in keys]
        lines.append(f"| {r['label']} | {'×'.join(map(str, r['size']))} | {', '.join(f'{w:g}' for w in r['wheels'])} "
                     f"| {'/'.join(r['colors'])} | " + " | ".join(cells) + " |")
    ok = [r for r in rows]
    if ok:
        lines.append("")
        for key in keys:
            iou = np.mean([r[key]["iou"] for r in ok])
            iouf = np.mean([r[key]["iou_filled"] for r in ok])
            one = np.mean([r[key]["components"] == 1 for r in ok])
            parts = np.median([r[key]["parts"] for r in ok])
            lines.append(f"- {key.upper()}: mean IoU {iou:.3f} (filled {iouf:.3f}), single component {one:.0%}, "
                         f"median parts {parts:.0f}")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    data = Path(__file__).resolve().parents[2] / "shape2brick" / "data"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapenet", default=str(data / "raw" / "ShapeNetCore" / "02958343"))
    ap.add_argument("--index", default=str(data / "cars" / "index" / "shapenet.csv"))
    ap.add_argument("--out", default="outputs/baselines/shapenet")
    ap.add_argument("--pick", default="default", help="'default' or comma-separated ShapeNet ids")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--res", type=int, default=300)
    ap.add_argument("--library", default="outputs/car_library",
                    help="car library for the retrieval baseline ('' = off)")
    a = ap.parse_args(argv)
    root, out = Path(a.shapenet), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lib = None
    if a.library and (Path(a.library) / "index.jsonl").is_file():
        from car2brick.library import CarLibrary

        lib = CarLibrary(a.library)
        print(f"library: {len(lib)} cars", flush=True)
    if a.pick == "default":
        models = pick_models(Path(a.index), root, DEFAULT_PICK)
    else:
        models = [(sid[:8], sid, sid) for sid in a.pick.split(",")]
    rows = []
    for label, sid, name in models:
        t = time.time()
        try:
            rows.append(run_one(label, sid, name, root, out, lib))
        except Exception as e:  # keep going; report at the end
            print(f"[fail] {label} {sid}: {type(e).__name__}: {e}", flush=True)
            continue
        r = rows[-1]
        res = " ".join(f"{k.upper()} {r[k]['iou']:.2f}/{r[k]['iou_filled']:.2f}/{r[k]['components']}"
                       for k, _ in METHODS if k in r)
        print(f"{label:16s} {'x'.join(map(str, r['size'])):9s} IoU/filled/comps: {res} ({time.time() - t:.1f}s)",
              flush=True)
    write_summary(rows, out)
    if not a.no_render:
        write_sheets(rows, Renderer(a.res), out)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
