"""Pinhole cameras and a reader for COLMAP's text model format (cameras.txt / images.txt / points3D.txt).

Distortion is ignored: run ``colmap image_undistorter`` first (``colmap.run_colmap`` does), which rewrites the
model as PINHOLE cameras for undistorted images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    name: str  # image file name
    K: np.ndarray  # 3x3 intrinsics
    R: np.ndarray  # 3x3 world -> camera rotation
    t: np.ndarray  # world -> camera translation
    width: int
    height: int

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(uv pixel coordinates (N, 2), depth (N,)) of world points."""
        cam = np.asarray(pts) @ self.R.T + self.t
        z = cam[:, 2]
        uvw = cam @ self.K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = uvw[:, :2] / uvw[:, 2:3]
        return uv, z

    @classmethod
    def look_at(cls, name, eye, target, f, width, height, up=(0.0, 0.0, 1.0)) -> "Camera":
        """Synthetic camera (tests / previews): OpenCV convention, +z forward, +y down."""
        eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
        fwd = target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd])
        K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
        return cls(name, K, R, -R @ eye, width, height)


def _qvec_to_rot(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _intrinsics(model: str, params: list[float]) -> np.ndarray:
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        f, cx, cy = params[:3]
        fx = fy = f
    elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"):
        fx, fy, cx, cy = params[:4]
    else:
        raise ValueError(f"unsupported COLMAP camera model {model}")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def read_colmap_model(model_dir: str | Path) -> tuple[list[Camera], np.ndarray]:
    """Cameras (sorted by image name) and sparse points (N, 3) from a COLMAP *text* model directory."""
    model_dir = Path(model_dir)
    intr = {}
    for line in (model_dir / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tok = line.split()
        intr[int(tok[0])] = (int(tok[2]), int(tok[3]), _intrinsics(tok[1], [float(v) for v in tok[4:]]))
    cams = []
    lines = [ln for ln in (model_dir / "images.txt").read_text().splitlines() if not ln.startswith("#")]
    for i in range(0, len(lines), 2):  # every image has a pose line followed by a 2D-points line
        tok = lines[i].split()
        if len(tok) < 10:
            continue
        q, t, cam_id, name = [float(v) for v in tok[1:5]], np.array(tok[5:8], float), int(tok[8]), tok[9]
        w, h, K = intr[cam_id]
        cams.append(Camera(name, K, _qvec_to_rot(q), t, w, h))
    pts = []
    p3d = model_dir / "points3D.txt"
    if p3d.is_file():
        for line in p3d.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                pts.append([float(v) for v in line.split()[1:4]])
    return sorted(cams, key=lambda c: c.name), np.array(pts).reshape(-1, 3)
