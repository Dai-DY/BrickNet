"""Synthetic fixtures: a box-and-cylinder car mesh, ring cameras, rasterised masks/images, a COLMAP text model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from car2brick.recon.cameras import Camera


def car_mesh():
    import trimesh

    body = trimesh.creation.box((4.5, 1.9, 0.75))
    body.apply_translation((0, 0, 0.15 + 0.375))
    cabin = trimesh.creation.box((2.2, 1.6, 0.5))
    cabin.apply_translation((0.3, 0, 0.9 + 0.25))
    # small triangles so the painter's-algorithm rasteriser orders faces correctly
    body = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(body.vertices, body.faces, 0.15))
    cabin = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(cabin.vertices, cabin.faces, 0.15))
    wheels = []
    for x in (-1.4, 1.4):
        for y in (-0.8, 0.8):
            cyl = trimesh.creation.cylinder(radius=0.33, height=0.25, sections=24)
            cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0)))
            cyl.apply_translation((x, 1.05 * y, 0.33))  # tyres flush with the body sides
            wheels.append(cyl)
    return body, cabin, wheels


def ring_cameras(n=24, radius=7.0, height=1.6, f=420.0, w=480, h=360):
    cams = []
    for k in range(n):
        a = 2 * np.pi * k / n
        eye = (radius * np.cos(a), radius * np.sin(a), height + 0.6 * np.sin(3 * a))
        cams.append(Camera.look_at(f"{k:04d}.png", eye, (0, 0, 0.6), f, w, h))
    return cams


def rasterize(cam: Camera, meshes_colors, w, h):
    """Painter's-algorithm rasterisation (far triangles first): returns (mask bool, rgb uint8)."""
    tris = []
    for mesh, color in meshes_colors:
        v = np.asarray(mesh.vertices)
        uv, z = cam.project(v)
        for f in np.asarray(mesh.faces):
            if (z[f] > 0).all():
                tris.append((z[f].mean(), [tuple(p) for p in uv[f]], color))
    tris.sort(key=lambda t: -t[0])
    img = Image.new("RGB", (w, h), (90, 140, 90))
    msk = Image.new("L", (w, h), 0)
    di, dm = ImageDraw.Draw(img), ImageDraw.Draw(msk)
    for _, poly, color in tris:
        di.polygon(poly, fill=color)
        dm.polygon(poly, fill=255)
    return np.array(msk) > 127, np.array(img)


def write_scene(root: Path, n=24):
    """Images, masks and a COLMAP text model of the synthetic car. Returns (model_dir, image_dir, mask_dir)."""
    body, cabin, wheels = car_mesh()
    cams = ring_cameras(n)
    img_dir, mask_dir, model_dir = root / "images", root / "masks", root / "sparse_txt"
    for d in (img_dir, mask_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)
    scene = [(body, (180, 10, 10)), (cabin, (175, 15, 15))] + [(wm, (20, 20, 25)) for wm in wheels]
    cam_lines, img_lines = [], []
    for k, cam in enumerate(cams):
        mask, rgb = rasterize(cam, scene, cam.width, cam.height)
        Image.fromarray(rgb).save(img_dir / cam.name)
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_dir / cam.name)
        K = cam.K
        cam_lines.append(f"{k + 1} PINHOLE {cam.width} {cam.height} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}")
        q = _rot_to_qvec(cam.R)
        img_lines.append(f"{k + 1} {' '.join(map(str, q))} {' '.join(map(str, cam.t))} {k + 1} {cam.name}")
        img_lines.append("")
    (model_dir / "cameras.txt").write_text("# cameras\n" + "\n".join(cam_lines) + "\n")
    (model_dir / "images.txt").write_text("# images\n" + "\n".join(img_lines) + "\n")
    return model_dir, img_dir, mask_dir


def _rot_to_qvec(R):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [w, x, y, z]
