"""Car library (upright fix, retrieval baseline) and the baseline / descriptor fixes found on ShapeNet."""

import json

import numpy as np

import bricknet
from car2brick.baseline import Placement, greedy_plates, plate_components
from car2brick.descriptor import CarDescriptor, descriptor_from_points
from car2brick.evaluate import evaluate_ldr, poses_to_ldr
from car2brick.lego import lego_car_from_ldr
from car2brick.library import CarLibrary, place, stud_up, upright


def _model(ldr):
    g = bricknet.parse_ldr(ldr)
    return list(g.part_ids), list(g.colors), [np.asarray(t, float) for t in g.transforms]


def test_upright_restores_rotated_models(baseline_car):
    pids, _, tr = _model(baseline_car)
    assert np.allclose(stud_up(pids, tr), [0, -1, 0])
    flips = {
        "upside down": np.diag([1.0, -1.0, -1.0]),
        "on its side": np.array([[0, -1.0, 0], [1.0, 0, 0], [0, 0, 1.0]]),
        "nose down": np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]),
    }
    for name, rot in flips.items():
        m = np.eye(4)
        m[:3, :3] = rot
        moved = [m @ t for t in tr]
        up = stud_up(pids, moved)
        assert np.allclose(up, rot @ [0, -1, 0]), name
        assert np.allclose(stud_up(pids, upright(moved, up)), [0, -1, 0]), name


def _write_library(root, cars):
    (root / "desc").mkdir(parents=True)
    (root / "ldr").mkdir()
    with open(root / "index.jsonl", "w") as fh:
        for name, ldr in cars.items():
            lc = lego_car_from_ldr(ldr)
            lc.descriptor.save(root / "desc" / f"{name}.npz")
            (root / "ldr" / f"{name}.ldr").write_text(poses_to_ldr(lc.part_ids, lc.colors, lc.grid_poses(), name))
            fh.write(json.dumps({"source": name, "size": list(lc.descriptor.size())}) + "\n")


def test_retrieval_finds_the_matching_car(tmp_path, box_desc, baseline_car):
    from conftest import box_car_points

    from car2brick.baseline import baseline_ldr

    short = baseline_ldr(descriptor_from_points(box_car_points(length=300.0, cabin=(60.0, 160.0), wheel_x=(60, 240))))
    _write_library(tmp_path, {"same": baseline_car, "short": short})
    lib = CarLibrary(tmp_path)
    top = lib.retrieve(box_desc, k=2)
    assert [m["car"].source for m in top] == ["same", "short"]
    assert top[0]["iou"] > 0.95 > top[1]["iou"]
    # placed into the target grid, the retrieved car scores like the original
    pids, cols, poses = place(top[0], box_desc)
    assert evaluate_ldr(poses_to_ldr(pids, cols, poses), box_desc)["iou"] > 0.95
    # a car facing the other way is matched by flipping it
    rev = CarDescriptor(box_desc.occupancy[::-1].copy(), box_desc.spec, box_desc.origin,
                        tuple(box_desc.spec.length - w for w in box_desc.wheels), box_desc.colors)
    best = lib.retrieve(rev, k=1)[0]
    assert best["car"].source == "same" and best["flip"] and best["iou"] > 0.95
    assert lib.retrieve(box_desc, k=1, exclude={"same"})[0]["car"].source == "short"


def test_lattice_points_fill_every_layer():
    """Points on sub-cell centres must not alias into a neighbouring bin (an empty layer split cars in two)."""
    idx = np.stack(np.meshgrid(np.arange(30), np.arange(16), np.arange(24), indexing="ij"), -1).reshape(-1, 3)
    pts = 0.37 + (idx + 0.5) * np.array([10.0, 10.0, 4.0])
    d = descriptor_from_points(pts)
    assert d.size() == (15, 8, 12)
    assert d.solid()[d.extent()].all()


def _box(x0, x1, y0, y1, z0, z1):
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=float)


def test_hollow_part_counts_as_solid():
    """A real brick is a shell (top, walls, studs; BrickNet's colliders even drop its long walls). Its volume must
    match the solid connector approximation, and an arch's open underside must stay open."""
    from car2brick.geometry import PartShapes, _drop_studs, _fill_under_holes

    cat = bricknet.load_catalog()
    conn = PartShapes("connector")

    def interior(stem, hulls):
        pid = cat.stem_to_id[stem]
        pts = np.concatenate([PartShapes._hull_lattice(h) for h in hulls])
        return _fill_under_holes(_drop_studs(pts, pid), pid), conn.local_interior(pid)

    studs = [_box(x - 6, x + 6, -4, 0, z - 6, z + 6) for x in (-30, -10, 10, 30) for z in (-10, 10)]
    shell = [_box(-40, 40, 0, 4, -20, 20), _box(-40, -36, 4, 24, -20, 20), _box(36, 40, 4, 24, -20, 20)]
    real, solid = interior("3001", shell + studs)  # brick 2x4 without its long walls
    assert len(real) == len(solid)
    legs = [_box(-40, -20, 0, 24, -10, 10), _box(20, 40, 0, 24, -10, 10), _box(-20, 20, 0, 8, -10, 10)]
    arch, _ = interior("3659", legs)  # arch 1x4: top beam on two legs
    assert not (np.abs(arch[:, 0]) < 16).__and__(arch[:, 1] > 12).any()  # nothing under the beam


def test_lego_grid_poses_are_integral(baseline_car):
    lc = lego_car_from_ldr(baseline_car)
    t = lc.grid_poses()[:, :3, 3]
    assert np.allclose(t, np.round(t))


def test_plate_components():
    a = Placement(0, 4, 0, 0, 0, 2, 4)
    b = Placement(0, 4, 1, 0, 1, 2, 4)  # overlaps a from above
    c = Placement(0, 4, 4, 0, 0, 2, 4)  # beside a on the same layer, nothing above it: not connected
    assert [sorted(x) for x in plate_components([a, b, c])] == [[0, 1], [2]]


def test_b1_overhang_stays_connected(box_desc):
    """A one-plate-thick overhang (nothing below, nothing above) must not become a floating tile."""
    occ = box_desc.occupancy.copy()
    ext = box_desc.extent()
    back = ext[0].stop - 1
    top = int(np.flatnonzero(box_desc.solid()[back].any(0)).max())  # deck height at the back
    occ[back + 1 : back + 3, ext[1], top] = 1.0  # a spoiler sticking out from the deck
    d = CarDescriptor(occ, box_desc.spec, box_desc.origin, box_desc.wheels, box_desc.colors)
    for tiles in (False, True):
        assert len(plate_components(greedy_plates(d, cap_tiles=tiles))) == 1
