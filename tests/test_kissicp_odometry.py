import numpy as np
import pytest
from apairo.core.sample import Sample

pytest.importorskip("kiss_icp", reason="kiss-icp not installed")

from apairo_preprocess.odometry.kissicp import KissICPOdometry


def _sample(xyz):
    return Sample(data={"lidar": np.asarray(xyz, dtype=np.float32)})


def _random_cloud(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-10.0, 10.0, (n, 3)).astype(np.float32)


def test_output_shape_dtype():
    proc = KissICPOdometry()
    out = proc.process(_sample(_random_cloud()))
    assert out.shape == (4, 4)
    assert out.dtype == np.float64


def test_first_frame_is_identity():
    proc = KissICPOdometry()
    out = proc.process(_sample(_random_cloud()))
    np.testing.assert_array_almost_equal(out, np.eye(4))


def test_output_is_a_copy():
    proc = KissICPOdometry()
    out = proc.process(_sample(_random_cloud()))
    out[:] = 99.0
    out2 = proc.process(_sample(_random_cloud()))
    assert not np.all(out2 == 99.0)


def test_second_frame_returns_valid_se3():
    proc = KissICPOdometry()
    cloud = _random_cloud()
    proc.process(_sample(cloud))
    out = proc.process(_sample(cloud))
    np.testing.assert_array_almost_equal(out[3], [0.0, 0.0, 0.0, 1.0])
    R = out[:3, :3]
    np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=5)


def test_deskew_reads_timestamp_column():
    # With deskew=True, column 3 of the point cloud is used as per-point timestamp.
    # Verify no crash and correct output shape.
    proc = KissICPOdometry(deskew=True)
    rng = np.random.default_rng(0)
    pc = rng.uniform(-10.0, 10.0, (500, 4)).astype(np.float32)
    pc[:, 3] = np.linspace(0.0, 0.1, 500)  # timestamps in [0, 0.1 s]
    out = proc.process(_sample(pc))
    assert out.shape == (4, 4)
