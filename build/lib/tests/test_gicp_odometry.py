import numpy as np
import pytest
from apairo.core.sample import Sample

o3d = pytest.importorskip("open3d", reason="open3d not installed")

from apairo_preprocess.odometry.gicp import GICPOdometry


def _sample(xyz):
    return Sample(data={"lidar": np.asarray(xyz, dtype=np.float32)})


def _random_cloud(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-10.0, 10.0, (n, 3)).astype(np.float32)


def test_output_shape_dtype():
    proc = GICPOdometry()
    out = proc.process(_sample(_random_cloud()))
    assert out.shape == (4, 4)
    assert out.dtype == np.float64


def test_first_frame_is_identity():
    proc = GICPOdometry()
    out = proc.process(_sample(_random_cloud()))
    np.testing.assert_array_almost_equal(out, np.eye(4))


def test_output_is_a_copy():
    # Mutating the returned array must not affect internal state.
    proc = GICPOdometry()
    out = proc.process(_sample(_random_cloud()))
    out[:] = 99.0
    out2 = proc.process(_sample(_random_cloud()))
    assert not np.all(out2 == 99.0)


def test_second_frame_returns_valid_se3():
    proc = GICPOdometry()
    cloud = _random_cloud()
    proc.process(_sample(cloud))
    out = proc.process(_sample(cloud))
    # Bottom row must be [0, 0, 0, 1]
    np.testing.assert_array_almost_equal(out[3], [0.0, 0.0, 0.0, 1.0])
    # Rotation block must be orthogonal
    R = out[:3, :3]
    np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=5)
