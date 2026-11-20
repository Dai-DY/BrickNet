#!/usr/bin/env python3
"""Preview images of the whole pipeline on the synthetic test car (no downloads, no GPU).

    python scripts/car2brick_demo.py --out docs/car2brick

Writes:
  1_input_photos.png      4 of the rendered input views (the "real car")
  2_descriptor.png        the stud x stud x plate descriptor recovered from the photos (windows marked)
  3_baseline_b0.png       B0: greedy plates
  4_baseline_b1.png       B1: plates + tiles on exposed tops
  5_fit_b2.png            B2: part-by-part fitting (bricks + slopes) on the same descriptor
  6_sloped_b1_vs_b2.png   B1 vs B2 on a car with a sloped nose / windscreen / fastback
  overview.png            all of the above in one sheet
  descriptor.txt          the conditioning text the generator would see
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "car2brick"))

import bricknet  # noqa: E402
import synthetic as S  # noqa: E402
from conftest import sloped_car_points  # noqa: E402

from car2brick.baseline import baseline_ldr  # noqa: E402
from car2brick.descriptor import descriptor_from_points  # noqa: E402
from car2brick.fit import fit_ldr  # noqa: E402
from car2brick.lego import lego_car_from_ldr  # noqa: E402
from car2brick.recon.masks import save_masks  # noqa: E402
from car2brick.recon.pipeline import photos_to_descriptor  # noqa: E402
from car2brick.render import orbit_camera, part_triangles, rasterize, voxel_triangles  # noqa: E402


def two_views(tris, rgb, ids, center, dist):
    a = rasterize(tris, rgb, ids, orbit_camera(center, dist, -125, 28))
    b = rasterize(tris, rgb, ids, orbit_camera(center, dist, -90, 4))
    return hstack([a, b])


def render_ldr(ldr, dist):
    lc = lego_car_from_ldr(ldr)
    gp = lc.grid_poses()
    t, c, i = part_triangles(lc.part_ids, lc.colors, gp)
    mid = 0.5 * (gp[:, :3, 3].min(0) + gp[:, :3, 3].max(0))
    return two_views(t, c, i, mid, dist)


def summary(ldr):
    g = bricknet.parse_ldr(ldr)
    return f"({len(g.part_ids)} parts, {len(g.components)} connected component{'s' if len(g.components) > 1 else ''})"


def hstack(imgs, gap=8, bg=(255, 255, 255)):
    w = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new("RGB", (w, max(i.height for i in imgs)), bg)
    x = 0
    for i in imgs:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


def titled(img, title, pad=34):
    out = Image.new("RGB", (img.width, img.height + pad), (255, 255, 255))
    out.paste(img, (0, pad))
    ImageDraw.Draw(out).text((8, 10), title, fill=(30, 30, 30))
    return out


def vstack(imgs, gap=10):
    w = max(i.width for i in imgs)
    out = Image.new("RGB", (w, sum(i.height for i in imgs) + gap * (len(imgs) - 1)), (255, 255, 255))
    y = 0
    for i in imgs:
        out.paste(i, (0, y))
        y += i.height + gap
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/car2brick")
    ap.add_argument("--views", type=int, default=36)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. synthetic "real car" photos + masks + COLMAP model; window masks from the cabin
    root = Path(tempfile.mkdtemp())
    model, img_dir, mask_dir = S.write_scene(root, n=a.views)
    body, cabin, wheels = S.car_mesh()
    win_masks = [S.rasterize(c, [(cabin, (0, 0, 0))], c.width, c.height)[0] for c in S.ring_cameras(a.views)]
    # the cabin's lower half is body, only its upper part is glass
    names = [c.name for c in S.ring_cameras(a.views)]
    save_masks(win_masks, names, root / "windows")
    photos = [Image.open(img_dir / names[k]) for k in (3, 9, 15, 27)]
    p1 = titled(hstack([p.resize((p.width * 2 // 3, p.height * 2 // 3)) for p in photos]),
                f"1. input: {a.views} photos of the car (4 shown) + car masks")
    p1.save(out / "1_input_photos.png")

    # 2. front half -> descriptor
    desc = photos_to_descriptor(colmap_model=model, image_dir=img_dir, mask_dir=mask_dir,
                                label_dirs={2: root / "windows"})
    (out / "descriptor.txt").write_text(desc.to_text("official") + "\n")
    solid = desc.symmetrized().solid()
    labels = desc.labels if desc.labels is not None else np.zeros(solid.shape, np.uint8)
    body_rgb = (180, 20, 20)

    def cell_rgb(i, j, k):
        return (70, 90, 110) if labels[i, j, k] == 2 else body_rgb

    vt, vc, vi = voxel_triangles(solid, cell_rgb)
    ext = desc.extent()
    center = np.array([(s.start + s.stop) / 2 for s in ext]) * np.array([20.0, 20.0, 8.0])
    dist = 820.0
    L, W, H = desc.size()
    p2 = titled(two_views(vt, vc, vi, center, dist),
                f"2. descriptor from photos: {L}x{W} studs x {H} plates, wheels at x={list(desc.wheels)}, "
                f"colors={list(desc.colors)} (blue-grey = window cells)")
    p2.save(out / "2_descriptor.png")

    # 3/4/5. baselines on the descriptor from the photos
    panels = [p1, p2]
    builds = (("3_baseline_b0.png", "3. B0 baseline: greedy plates", baseline_ldr(desc, cap_tiles=False)),
              ("4_baseline_b1.png", "4. B1 baseline: plates + tiles on exposed tops", baseline_ldr(desc)),
              ("5_fit_b2.png", "5. B2 baseline: part-by-part fitting (bricks, slopes, plates, tiles)",
               fit_ldr(desc)[0]))
    for name, title, ldr in builds:
        (out / name.replace(".png", ".ldr")).write_text(ldr)
        p = titled(render_ldr(ldr, dist), f"{title}  {summary(ldr)}")
        p.save(out / name)
        panels.append(p)
    vstack(panels).save(out / "overview.png")

    # 6. where B2 differs: sloped nose / windscreen / fastback
    sloped = descriptor_from_points(sloped_car_points(), colors=("blue", "black"))
    rows = []
    for title, ldr in (("B1", baseline_ldr(sloped)), ("B2", fit_ldr(sloped)[0])):
        rows.append(titled(render_ldr(ldr, dist), f"{title} on a sloped car  {summary(ldr)}"))
    vstack(rows).save(out / "6_sloped_b1_vs_b2.png")
    print(f"wrote {out}/overview.png")
    print(desc.to_text("official"))


if __name__ == "__main__":
    main()
