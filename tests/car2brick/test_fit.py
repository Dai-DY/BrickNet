"""B2 part-by-part fitting baseline."""

import numpy as np
import pytest

import bricknet
from car2brick.fit import build_templates, fit, fit_ldr
from car2brick.geometry import PartShapes
from car2brick.lego import lego_car_from_ldr


@pytest.fixture(scope="module")
def templates():
    return build_templates(PartShapes("connector"))


def test_templates(templates):
    cat = bricknet.load_catalog()
    by_part = {}
    for t in templates:
        by_part.setdefault(cat.id_to_stem[t.part_id], []).append(t)
    assert len(by_part["3020"]) == 2  # a plate has two distinct quarter turns
    assert len(by_part["3040b"]) == 4  # a slope has four
    assert all(t.mirror >= 0 for t in templates)
    slope = by_part["3040b"][0]
    # wedge: full brick height under the stud, lower at the far end
    assert slope.mask.shape[2] == 6 and slope.mask.sum() < slope.mask.size
    plate = by_part["3020"][0]
    assert plate.mask.all() and plate.mask.shape[2] == 2 and len(plate.studs) == 8


def test_wedge_geometry():
    shapes = PartShapes("connector")
    pid = bricknet.load_catalog().stem_to_id["3040b"]
    hull = shapes.hulls(pid)[0]
    assert np.allclose(hull.min(0), [-10, 0, -30]) and np.allclose(hull.max(0), [10, 24, 10])
    pts = shapes.local_interior(pid)
    low_end = pts[pts[:, 2] < -25]
    assert low_end[:, 1].min() > 16  # only the lip near the bottom at the low end (LDraw y grows down)


def test_fit_box_car_is_connected_and_exact(box_desc, templates):
    ldr, res = fit_ldr(box_desc, templates=templates)
    g = bricknet.parse_ldr(ldr)
    assert len(g.components) == 1
    assert res.stats["target_covered"] > 0.98 and res.stats["overflow"] < 0.02
    assert lego_car_from_ldr(ldr).descriptor.iou(box_desc) > 0.95


def test_fit_uses_slopes_and_stays_symmetric(sloped_desc, templates):
    res = fit(sloped_desc, templates=templates)
    assert res.stats["kinds"]["slope"] >= 2 and res.stats["groups"] == 1
    poses = np.array(res.poses())
    centre = sloped_desc.spec.width * 20 / 2
    ys = np.sort(np.round(poses[:, 1, 3] - centre, 3))
    assert np.allclose(ys, -ys[::-1])  # every part has a mirror partner


def test_fit_without_slopes(sloped_desc, templates):
    res = fit(sloped_desc, templates=[t for t in templates if t.kind != "slope"])
    assert res.stats["kinds"]["slope"] == 0 and res.stats["target_covered"] > 0.9
