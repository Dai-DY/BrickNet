"""Front half end to end: photos (or a video) -> CarDescriptor (+ colours, labels).

    python -m car2brick photos2desc --images frames/ --workspace ws/ --masks masks/ --out car.npz

Steps: COLMAP (skipped when ``--colmap-model`` points to an existing text model) -> masks (folder or SAM2) ->
visual hull in the car frame -> colours / labels by projection -> descriptor.
Alternatively reconstruct a mesh with any external method and use ``mesh2desc``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import colors as palette
from ..descriptor import CarDescriptor, GridSpec, descriptor_from_points
from ..units import LDU_PER_STUD
from .cameras import read_colmap_model
from .masks import load_masks, segment_with_sam2
from .semantics import project_colors, project_labels, surface_mask, wheels_from_dark_points
from .visual_hull import hull_descriptor


def _load_images(image_dir: Path, names: list[str]) -> list[np.ndarray]:
    from PIL import Image

    return [np.asarray(Image.open(image_dir / n).convert("RGB")) for n in names]


def photos_to_descriptor(
    images: str | Path | None = None,
    workspace: str | Path | None = None,
    *,
    colmap_model: str | Path | None = None,
    image_dir: str | Path | None = None,
    mask_dir: str | Path | None = None,
    label_dirs: dict[int, str | Path] | None = None,
    sam2: tuple[str, str] | None = None,
    matcher: str = "exhaustive",
    spec: GridSpec | None = None,
    target_width: float = 8.0,
    max_misses: int = 1,
) -> CarDescriptor:
    if colmap_model is None:
        from .colmap import run_colmap

        if images is None or workspace is None:
            raise ValueError("need images + workspace, or colmap_model")
        colmap_model = run_colmap(images, workspace, matcher=matcher)
        image_dir = Path(workspace) / "dense" / "images"
    image_dir = Path(image_dir or images)
    cameras, sparse = read_colmap_model(colmap_model)
    names = [c.name for c in cameras]

    if mask_dir is not None:
        masks = load_masks(mask_dir, names)
    elif sam2 is not None:
        seg_names, seg_masks = segment_with_sam2(image_dir, checkpoint=sam2[0], model_cfg=sam2[1])
        by_name = dict(zip(seg_names, seg_masks))
        masks = [by_name[n] for n in names]
    else:
        raise ValueError("need mask_dir or sam2=(checkpoint, config)")

    desc, canon, pts_car = hull_descriptor(
        cameras, masks, sparse_points=sparse, spec=spec, target_width=target_width, max_misses=max_misses
    )

    # colours from surface points; labels (windows / lights) when label masks are given
    surf = surface_mask(pts_car)
    world = canon.invert(pts_car)
    imgs = _load_images(image_dir, names)
    rgb = project_colors(world[surf], cameras, imgs, masks)
    labels = np.ones(len(pts_car), dtype=np.uint8)
    if label_dirs:
        lm = {code: load_masks(d, names) for code, d in label_dirs.items()}
        labels[surf] = project_labels(world[surf], cameras, lm)
    body_rgb = rgb[(labels[surf] == 1) & ~np.isnan(rgb).any(1)]
    cols = palette.dominant(body_rgb)
    out = descriptor_from_points(pts_car, spec or desc.spec, center_y=0.0, colors=cols, point_labels=labels)
    # the hull cannot see under the car, so prefer tyre colour over the occupancy contact profile
    wx = wheels_from_dark_points(pts_car[surf], rgb)
    if wx:
        out.wheels = tuple(float(round((x - out.origin[0]) / LDU_PER_STUD, 1)) for x in wx)
    return out
