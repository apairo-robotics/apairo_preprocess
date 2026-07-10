import numpy as np
import pytest
from apairo.core.config import CameraIntrinsics
from apairo.core.sample import Sample

from apairo_preprocess.projection.lidar_to_camera import LidarCameraProjection


K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
SIZE = (100, 100)  # (height, width)
IDENTITY = np.eye(4)


def _project(xyz, **kwargs):
    defaults = dict(intrinsics=K, image_size=SIZE, extrinsics=IDENTITY)
    defaults.update(kwargs)
    proc = LidarCameraProjection(**defaults)
    lidar_key = kwargs.get("lidar_key", "lidar")
    sample = Sample(data={lidar_key: np.asarray(xyz, dtype=np.float32)})
    return proc(sample)


# ------------------------------------------------------------------
# Constructor / API
# ------------------------------------------------------------------

def test_default_keys():
    proc = LidarCameraProjection(intrinsics=K, image_size=SIZE, extrinsics=IDENTITY)
    assert proc.input_keys == ["lidar"]
    assert proc.sources == ["lidar"]
    assert proc.timestamps_from == "lidar"
    assert proc.output_key == "lidar_uv"


def test_custom_keys():
    proc = LidarCameraProjection(
        intrinsics=K,
        image_size=SIZE,
        extrinsics=IDENTITY,
        lidar_key="velodyne_0",
        output_key="lidar_uv_left",
    )
    assert proc.input_keys == ["velodyne_0"]
    assert proc.sources == ["velodyne_0"]
    assert proc.output_key == "lidar_uv_left"


def test_intrinsics_shape_rejected():
    with pytest.raises(ValueError):
        LidarCameraProjection(intrinsics=np.eye(4), image_size=SIZE, extrinsics=IDENTITY)


def test_extrinsics_shape_rejected():
    with pytest.raises(ValueError):
        LidarCameraProjection(intrinsics=K, image_size=SIZE, extrinsics=np.eye(3))


def test_image_size_rejected():
    with pytest.raises(ValueError):
        LidarCameraProjection(intrinsics=K, image_size=(0, 100), extrinsics=IDENTITY)


def test_distortion_length_rejected():
    with pytest.raises(ValueError):
        LidarCameraProjection(
            intrinsics=K, image_size=SIZE, extrinsics=IDENTITY, distortion=[0.1, 0.2]
        )


# ------------------------------------------------------------------
# CameraIntrinsics input
# ------------------------------------------------------------------

def test_camera_intrinsics_supplies_k_size_distortion():
    cam = CameraIntrinsics(
        K=K, distortion=np.zeros(5), model="plumb_bob", width=100, height=100
    )
    proc = LidarCameraProjection(intrinsics=cam, extrinsics=IDENTITY)
    sample = Sample(data={"lidar": np.array([[0.0, 0.0, 5.0]], dtype=np.float32)})
    np.testing.assert_allclose(proc(sample)[0], [50.0, 50.0, 5.0], atol=1e-5)


def test_camera_intrinsics_matches_bare_k():
    d = [0.1, 0.01, 0.001, 0.002, 0.0]
    cam = CameraIntrinsics(K=K, distortion=np.array(d), width=100, height=100)
    xyz = np.array([[1.0, -2.0, 10.0]], dtype=np.float32)
    from_cam = LidarCameraProjection(intrinsics=cam, extrinsics=IDENTITY)
    from_k = LidarCameraProjection(
        intrinsics=K, extrinsics=IDENTITY, image_size=SIZE, distortion=d
    )
    sample = Sample(data={"lidar": xyz})
    np.testing.assert_allclose(from_cam(sample), from_k(sample), atol=1e-6)


def test_explicit_image_size_overrides_camera_intrinsics():
    cam = CameraIntrinsics(K=K, width=100, height=100)
    proc = LidarCameraProjection(intrinsics=cam, extrinsics=IDENTITY, image_size=(50, 55))
    # u = 60 is inside the recorded 100-wide image but outside the 55-wide override.
    sample = Sample(data={"lidar": np.array([[1.0, 0.0, 10.0]], dtype=np.float32)})
    assert np.isnan(proc(sample)[0]).all()


def test_camera_intrinsics_without_size_rejected():
    cam = CameraIntrinsics(K=K)
    with pytest.raises(ValueError):
        LidarCameraProjection(intrinsics=cam, extrinsics=IDENTITY)


def test_bare_k_without_image_size_rejected():
    with pytest.raises(ValueError):
        LidarCameraProjection(intrinsics=K, extrinsics=IDENTITY)


def test_unsupported_distortion_model_rejected():
    cam = CameraIntrinsics(
        K=K, distortion=np.ones(8), model="rational_polynomial", width=100, height=100
    )
    with pytest.raises(ValueError):
        LidarCameraProjection(intrinsics=cam, extrinsics=IDENTITY)


# ------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------

def test_principal_axis_hits_image_center():
    out = _project([[0.0, 0.0, 5.0]])
    np.testing.assert_allclose(out[0], [50.0, 50.0, 5.0], atol=1e-5)


def test_offset_point():
    # x/z = 0.1 -> u = 100 * 0.1 + 50; y/z = -0.2 -> v = 100 * -0.2 + 50
    out = _project([[1.0, -2.0, 10.0]])
    np.testing.assert_allclose(out[0], [60.0, 30.0, 10.0], atol=1e-5)


def test_behind_camera_is_nan():
    out = _project([[0.0, 0.0, -5.0]])
    assert np.isnan(out[0]).all()


def test_out_of_bounds_is_nan():
    # x/z = 1.0 -> u = 150, outside a 100-wide image.
    out = _project([[5.0, 0.0, 5.0]])
    assert np.isnan(out[0]).all()


def test_output_is_row_aligned_float32():
    xyz = [[0.0, 0.0, 5.0], [0.0, 0.0, -1.0], [1.0, -2.0, 10.0]]
    out = _project(xyz)
    assert out.shape == (3, 3)
    assert out.dtype == np.float32
    assert np.isfinite(out[0]).all()
    assert np.isnan(out[1]).all()
    assert np.isfinite(out[2]).all()


def test_extra_lidar_columns_ignored():
    out = _project([[0.0, 0.0, 5.0, 0.7]])  # (N, 4) with intensity
    np.testing.assert_allclose(out[0], [50.0, 50.0, 5.0], atol=1e-5)


def test_extrinsics_applied():
    # Camera 5 m behind the lidar origin along the optical axis: a point at
    # the lidar origin lands on the principal point at depth 5.
    T = np.eye(4)
    T[2, 3] = 5.0
    out = _project([[0.0, 0.0, 0.0]], extrinsics=T)
    np.testing.assert_allclose(out[0], [50.0, 50.0, 5.0], atol=1e-5)


def test_empty_scan():
    out = _project(np.zeros((0, 3)))
    assert out.shape == (0, 3)


def test_all_behind_camera():
    out = _project([[0.0, 0.0, -1.0], [1.0, 1.0, -2.0]])
    assert np.isnan(out).all()


# ------------------------------------------------------------------
# Distortion
# ------------------------------------------------------------------

def test_zero_distortion_matches_pinhole():
    xyz = [[1.0, -2.0, 10.0], [0.5, 0.5, 4.0]]
    plain = _project(xyz)
    zeroed = _project(xyz, distortion=[0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(plain, zeroed, atol=1e-6)


def test_four_coefficient_distortion_accepted():
    # (k1, k2, p1, p2) without k3 is the common ROS camera_info form.
    out = _project([[0.0, 0.0, 5.0]], distortion=[0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(out[0], [50.0, 50.0, 5.0], atol=1e-5)


def test_radial_distortion_moves_point_outward():
    # k1 > 0 pushes points away from the principal point; on-axis is unmoved.
    xyz = [[1.0, 0.0, 10.0]]
    plain = _project(xyz)
    distorted = _project(xyz, distortion=[0.1, 0.0, 0.0, 0.0, 0.0])
    # x = 0.1, r2 = 0.01 -> u = 100 * 0.1 * (1 + 0.1 * 0.01) + 50
    np.testing.assert_allclose(distorted[0, 0], 60.01, atol=1e-5)
    assert distorted[0, 0] > plain[0, 0]
    np.testing.assert_allclose(distorted[0, 1], plain[0, 1], atol=1e-6)
