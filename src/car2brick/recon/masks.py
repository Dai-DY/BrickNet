"""Foreground (car) masks: load from disk, or segment a frame sequence with SAM2.

Mask files are matched to images by stem (``0001.jpg`` -> ``0001.png``); any non-zero pixel is foreground.
``clean_mask`` keeps the largest connected component and fills holes (windows seen through the car).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage


def clean_mask(mask: np.ndarray, close: int = 3) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if close:
        m = ndimage.binary_closing(m, iterations=close)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(m)


def load_masks(mask_dir: str | Path, image_names: list[str]) -> list[np.ndarray]:
    from PIL import Image

    mask_dir = Path(mask_dir)
    out = []
    for name in image_names:
        stem = Path(name).stem
        cands = [p for p in mask_dir.glob(stem + ".*")]
        if not cands:
            raise FileNotFoundError(f"no mask for {name} in {mask_dir}")
        out.append(clean_mask(np.array(Image.open(cands[0]).convert("L")) > 127))
    return out


def save_masks(masks: list[np.ndarray], names: list[str], out_dir: str | Path) -> None:
    from PIL import Image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for m, name in zip(masks, names):
        Image.fromarray((np.asarray(m) * 255).astype(np.uint8)).save(out / (Path(name).stem + ".png"))


def segment_with_sam2(
    frame_dir: str | Path,
    checkpoint: str,
    model_cfg: str,
    *,
    point: tuple[float, float] | None = None,
    device: str = "cuda",
) -> tuple[list[str], list[np.ndarray]]:
    """Track the car through an ordered frame folder with the SAM2 video predictor.

    The car is prompted with one positive click on the first frame (default: image centre). Requires
    ``pip install sam2`` (https://github.com/facebookresearch/sam2) and a checkpoint. Frames must be JPEGs named so
    that sorting gives temporal order (``colmap.extract_frames`` does this).
    """
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor

    frame_dir = Path(frame_dir)
    names = sorted(p.name for p in frame_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    w, h = Image.open(frame_dir / names[0]).size
    pt = np.array([point or (w / 2, h / 2)], dtype=np.float32)
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    masks: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        state = predictor.init_state(video_path=str(frame_dir))
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, points=pt, labels=np.array([1], np.int32))
        for idx, _obj_ids, logits in predictor.propagate_in_video(state):
            masks[idx] = clean_mask((logits[0, 0] > 0).cpu().numpy())
    return names, [masks[i] for i in range(len(names))]
