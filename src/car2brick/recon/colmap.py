"""Run COLMAP's sparse reconstruction + undistortion from the command line and export a text model.

    run_colmap(images, workspace) -> workspace/dense/sparse_txt  (PINHOLE cameras, undistorted images in
                                                                    workspace/dense/images)

Requires the ``colmap`` binary on PATH. For walk-around videos use ``matcher="sequential"``. If COLMAP fails on a
shiny car, a learned SfM (VGGT / MASt3R-SfM) can export the same text format; ``read_colmap_model`` then works
unchanged.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def extract_frames(video: str | Path, out_dir: str | Path, fps: float = 2.0) -> Path:
    """Frames from a walk-around video with ffmpeg (``fps`` frames per second)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-i", str(video), "-vf", f"fps={fps}", "-q:v", "2", str(out / "%05d.jpg")])
    return out


def run_colmap(
    images: str | Path,
    workspace: str | Path,
    *,
    matcher: str = "exhaustive",
    use_gpu: bool = True,
    single_camera: bool = True,
    colmap: str = "colmap",
) -> Path:
    if shutil.which(colmap) is None:
        raise FileNotFoundError(f"{colmap!r} not found on PATH")
    images, ws = Path(images), Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    db, sparse, dense = ws / "database.db", ws / "sparse", ws / "dense"
    sparse.mkdir(exist_ok=True)
    gpu = "1" if use_gpu else "0"
    _run([colmap, "feature_extractor", "--database_path", str(db), "--image_path", str(images),
          "--ImageReader.single_camera", "1" if single_camera else "0", "--SiftExtraction.use_gpu", gpu])
    _run([colmap, f"{matcher}_matcher", "--database_path", str(db), "--SiftMatching.use_gpu", gpu])
    _run([colmap, "mapper", "--database_path", str(db), "--image_path", str(images), "--output_path", str(sparse)])
    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    if not models:
        raise RuntimeError("COLMAP mapper produced no model")
    # the largest model is the one with the biggest images.bin
    best = max(models, key=lambda p: (p / "images.bin").stat().st_size if (p / "images.bin").exists() else 0)
    _run([colmap, "image_undistorter", "--image_path", str(images), "--input_path", str(best),
          "--output_path", str(dense), "--output_type", "COLMAP"])
    txt = dense / "sparse_txt"
    txt.mkdir(exist_ok=True)
    _run([colmap, "model_converter", "--input_path", str(dense / "sparse"), "--output_path", str(txt),
          "--output_type", "TXT"])
    return txt
