import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))  # for `import synthetic`

from car2brick.descriptor import descriptor_from_points  # noqa: E402


def box_car_points(length=400.0, width=160.0, cabin=(140.0, 300.0), wheel_x=(70.0, 330.0)):
    """Interior lattice of a body + cabin + wheels car, car frame (LDU)."""
    g = np.stack(
        np.meshgrid(np.arange(0, length, 5.0), np.arange(-width / 2, width / 2, 5.0), np.arange(0, 120, 2.0),
                    indexing="ij"), -1
    ).reshape(-1, 3) + [2.5, 2.5, 1.0]
    body = (g[:, 2] >= 8) & (g[:, 2] < 56)
    wheels = ((np.abs(g[:, 0] - wheel_x[0]) < 30) | (np.abs(g[:, 0] - wheel_x[1]) < 30)) & (g[:, 2] < 8)
    cab = (g[:, 0] > cabin[0]) & (g[:, 0] < cabin[1]) & (g[:, 2] >= 56) & (g[:, 2] < 96) & (np.abs(g[:, 1]) < 60)
    return g[body | wheels | cab]


def sloped_car_points():
    """A car with a sloped nose, raked windscreen and fastback (car frame, LDU)."""
    g = np.stack(
        np.meshgrid(np.arange(0, 400, 5.0), np.arange(-80, 80, 5.0), np.arange(0, 120, 2.0), indexing="ij"), -1
    ).reshape(-1, 3) + [2.5, 2.5, 1.0]
    x, y, z = g.T
    body_top = np.where(x < 80, 32 + (56 - 32) * x / 80, 56)  # nose rises from 32 to 56 LDU over 4 studs
    body = (z >= 8) & (z < body_top)
    wheels = ((np.abs(x - 70) < 30) | (np.abs(x - 330) < 30)) & (z < 8)
    roof = np.interp(x, [120, 180, 260, 380], [56, 96, 96, 60])  # windscreen, roof, fastback
    cab = (x > 120) & (x < 380) & (z >= 56) & (z < roof) & (np.abs(y) < 60)
    return g[body | wheels | cab]


@pytest.fixture(scope="session")
def sloped_desc():
    return descriptor_from_points(sloped_car_points(), colors=("blue", "black"))


@pytest.fixture(scope="session")
def box_desc():
    return descriptor_from_points(box_car_points(), colors=("red", "black"))


@pytest.fixture(scope="session")
def baseline_car(box_desc):
    from car2brick.baseline import baseline_ldr

    return baseline_ldr(box_desc)
