"""Front half on a synthetic scene: rendered masks/images + a COLMAP text model -> descriptor."""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("PIL")

import synthetic as S  # noqa: E402
from car2brick.canonical import canonicalize, mesh_descriptor  # noqa: E402
from car2brick.recon.cameras import read_colmap_model  # noqa: E402
from car2brick.recon.pipeline import photos_to_descriptor  # noqa: E402


@pytest.fixture(scope="module")
def reference():
    body, cabin, wheels = S.car_mesh()
    ref, _ = mesh_descriptor(trimesh.util.concatenate([body, cabin] + wheels), up_hint=(0, 0, 1))
    return ref


def _best_iou(d, ref):
    flipped = type(d)(d.occupancy[::-1].copy(), d.spec)  # forward direction is not identifiable
    return max(d.iou(ref), flipped.iou(ref))


def test_colmap_reader(tmp_path):
    model, _, _ = S.write_scene(tmp_path, n=6)
    cams, pts = read_colmap_model(model)
    truth = S.ring_cameras(6)
    assert len(cams) == 6 and pts.shape == (0, 3)
    for c, t in zip(cams, truth):
        assert np.allclose(c.R, t.R, atol=1e-6) and np.allclose(c.center, t.center, atol=1e-6)


def test_canonicalize_is_pose_and_scale_invariant(reference):
    from scipy.spatial.transform import Rotation

    body, cabin, wheels = S.car_mesh()
    car = trimesh.util.concatenate([body, cabin] + wheels)
    M = np.eye(4)
    M[:3, :3] = Rotation.from_euler("xyz", [30, -50, 170], degrees=True).as_matrix() * 2.7
    M[:3, 3] = (3, -1, 5)
    car.apply_transform(M)
    d, canon = mesh_descriptor(car)
    assert d.size()[1] == 8
    assert _best_iou(d, reference) > 0.75
    assert isinstance(canonicalize(np.random.default_rng(0).normal(size=(500, 3))).scale, float)


def test_photos_to_descriptor(tmp_path, reference):
    model, img, msk = S.write_scene(tmp_path, n=36)
    d = photos_to_descriptor(colmap_model=model, image_dir=img, mask_dir=msk)
    L, W, H = d.size()
    assert W == 8 and 16 <= L <= 20 and 13 <= H <= 16
    assert d.colors[0] == "red"
    assert len(d.wheels) == 2
    ext = d.extent()[0]
    # true axles are ~19% of the length in from each end
    rel = sorted((w - ext.start) / (ext.stop - ext.start) for w in d.wheels)
    assert abs(rel[0] - 0.19) < 0.1 and abs(rel[1] - 0.81) < 0.1
    assert _best_iou(d, reference) > 0.6
