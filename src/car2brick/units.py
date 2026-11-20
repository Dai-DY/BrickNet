"""Units and frames.

LDraw units (LDU): 1 stud pitch = 20 LDU, 1 plate = 8 LDU, 1 brick = 24 LDU. LDraw world axes have -Y up.

The *car frame* used everywhere in car2brick is right-handed, in LDU:
    x: along the car's length (grid index i grows with x)
    y: lateral, y = 0 on the symmetry plane
    z: up, z = 0 at the lowest point of the car (tyre contact)

``CAR_FROM_LD`` maps LDraw world coordinates to the car frame for a model whose length runs along LDraw X;
``LD_FROM_CAR`` is its inverse (LD = (x, -z, y)).
"""

import itertools

import numpy as np

LDU_PER_STUD = 20.0
LDU_PER_PLATE = 8.0
LDU_PER_BRICK = 24.0

# LD = (x, -z, y)  <=>  car = (X, Z, -Y)
LD_FROM_CAR = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
CAR_FROM_LD = LD_FROM_CAR.T


def homogeneous(rot: np.ndarray, trans=(0.0, 0.0, 0.0)) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = trans
    return m


def transform_points(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ mat[:3, :3].T + mat[:3, 3]


def _axis_rotations() -> np.ndarray:
    """The 24 proper rotations that map coordinate axes onto coordinate axes, in a fixed order."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            for r, (c, s) in enumerate(zip(perm, signs)):
                m[r, c] = s
            if np.linalg.det(m) > 0:
                out.append(m)
    return np.array(out)


AXIS_ROTATIONS = _axis_rotations()


def rotation_code(rot: np.ndarray, tol: float = 1e-3) -> int | None:
    """Index of ``rot`` in AXIS_ROTATIONS, or None when it is not axis-aligned."""
    d = np.abs(AXIS_ROTATIONS - np.asarray(rot)[None]).max(axis=(1, 2))
    k = int(np.argmin(d))
    return k if d[k] < tol else None
