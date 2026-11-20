"""The descriptor is the bridge: LEGO and real-car shapes must map to the same representation."""

import numpy as np

import bricknet
from car2brick.baseline import greedy_plates, plate_library, tile_library
from car2brick.descriptor import CarDescriptor, GridSpec, detect_wheels
from car2brick.lego import lego_car_from_ldr


def test_descriptor_basics(box_desc):
    assert box_desc.size() == (20, 8, 12)
    assert box_desc.wheels == (9.5, 22.5)  # wheel contact runs at x = 40..100 and 300..360 LDU
    assert box_desc.colors == ("red", "black")
    sym = box_desc.symmetrized()
    assert np.allclose(sym.occupancy, sym.occupancy[:, ::-1])


def test_text_roundtrip(box_desc):
    text = box_desc.to_text("official")
    assert text.splitlines()[0] == "<official>"
    assert "wheels 9.5 22.5" in text
    back = CarDescriptor.from_text(text)
    assert back.iou(box_desc) == 1.0
    assert back.wheels == box_desc.wheels and back.colors == box_desc.colors


def test_perturb_keeps_shape_roughly(box_desc):
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = box_desc.perturb(rng)
        assert 0.5 < p.iou(box_desc) <= 1.0
        assert np.allclose(p.occupancy, p.occupancy[:, ::-1])


def test_libraries_from_connectors():
    lib = plate_library()
    assert lib[(2, 4)] == bricknet.load_catalog().stem_to_id["3020"]
    assert set(tile_library()) >= {(1, 2), (2, 2), (2, 4)}


def test_baseline_is_connected_and_matches(box_desc, baseline_car):
    graph = bricknet.parse_ldr(baseline_car)
    assert len(graph.components) == 1
    lc = lego_car_from_ldr(baseline_car)  # LEGO side of the bridge
    assert lc.descriptor.iou(box_desc) > 0.95
    assert lc.descriptor.size() == box_desc.size()


def test_tiles_cap_exposed_tops(box_desc):
    pl = greedy_plates(box_desc, cap_tiles=True)
    names = [bricknet.load_catalog().id_to_name[p.part_id] for p in pl]
    assert any(n.startswith("tile") for n in names)


def test_detect_wheels_flat_bottom():
    occ = np.zeros(GridSpec().shape, np.float32)
    occ[5:25, 2:10, 0:6] = 1
    assert detect_wheels(occ) == ()
