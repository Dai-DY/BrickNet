"""Text format v2: segments with colour runs; open gaps kept, closed cavities filled; colour classes."""

import numpy as np

from car2brick.colors import CLASS_INDEX, GLASS, class_of_name, class_of_rgb
from car2brick.descriptor import CarDescriptor, GridSpec, descriptor_from_points, with_surface_samples


def _box_points(lo, hi, step=(5.0, 5.0, 2.0)):
    axes = [np.arange(a + s / 2, b, s) for a, b, s in zip(lo, hi, step)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def _car(cavity=False):
    """A 16 x 6 stud body, 4 plates high (hollow inside if ``cavity``), with a one-plate wing on two struts at the
    back, two plates above the deck."""
    body = _box_points((0, 0, 0), (320, 120, 32))
    if cavity:
        inside = ((body > [40, 20, 8]) & (body < [280, 100, 24])).all(1)
        body = body[~inside]
    struts = np.concatenate([_box_points((10, y, 32), (20, y + 10, 48)) for y in (10, 100)])
    wing = _box_points((0, 0, 48), (40, 120, 56))
    pts = np.concatenate([body, struts, wing])
    cls = np.concatenate([np.full(len(body), CLASS_INDEX["r"]), np.full(len(struts) + len(wing), CLASS_INDEX["k"])])
    return descriptor_from_points(pts, GridSpec(), colors=("red", "black"), point_classes=cls)


def test_wing_gap_kept_and_cavity_filled():
    d = _car(cavity=True)
    text = d.to_text("official")
    rows = {int(ln.split()[0]): ln.split()[1:] for ln in text.split("\n")[3:]}
    first = min(rows)
    assert "+" in rows[first][0] and rows[first][0].endswith("k1")  # body, gap, black wing plate
    middle = rows[first + 8]
    assert all("+" not in c for c in middle) and middle[0] == "0r4"  # the closed cavity is filled
    solid, _ = d.segments()
    assert solid.sum() == _car(cavity=False).segments()[0].sum()


def test_v2_round_trip_and_colours():
    d = _car()
    text = d.to_text("moc8")
    back = CarDescriptor.from_text(text, d.spec)
    assert back.to_text("moc8") == text
    assert back.surface is not None and (back.surface == CLASS_INDEX["k"]).any()
    assert d.to_text(fmt="v1").split("\n")[0].startswith("car")  # the old format is still available


def test_colour_classes():
    assert class_of_name("trans clear") == GLASS and class_of_name("dark bluish grey") == CLASS_INDEX["g"]
    assert class_of_name("reddish brown") == CLASS_INDEX["n"]
    assert class_of_name("light bluish grey") == CLASS_INDEX["s"]
    assert list(class_of_rgb([[250, 200, 10], [20, 20, 25], [np.nan, 0, 0]])) == [CLASS_INDEX["y"], CLASS_INDEX["k"], -1]


def test_surface_samples_and_perturb():
    d = _car()
    pts = _box_points((0, 0, 30), (320, 120, 32))  # samples on the deck, painted blue
    e = with_surface_samples(d, pts, np.full(len(pts), CLASS_INDEX["b"]))
    assert (e.surface == CLASS_INDEX["b"]).sum() > 0
    p = d.perturb(np.random.default_rng(0))
    assert p.surface is not None and "+" in p.to_text()  # the wing survives the augmentation (mostly)


def test_drop_figures():
    import bricknet

    from car2brick.dataset import drop_figures

    ldr = "\n".join([
        "1 4 0 0 0 1 0 0 0 1 0 0 0 1 3020.dat",  # plate 2x4
        "1 4 0 -8 0 1 0 0 0 1 0 0 0 1 3022.dat",  # plate 2x2 on top
        "1 1 0 -16 0 1 0 0 0 1 0 0 0 1 3815.dat",  # minifig hips on top of that
    ])
    g = bricknet.parse_ldr(ldr)
    g2, steps = drop_figures(g, [1, 2, 3])
    assert len(g2.part_ids) == 2 and steps == [1, 2] and len(g2.components) == 1
    assert len(g2.edges) == len(g.edges) - sum(1 for e in g.edges if 2 in (e["a"], e["b"]))
