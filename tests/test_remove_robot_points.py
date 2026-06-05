import numpy as np
import pytest
from apairo.core.sample import Sample

from apairo_preprocess.pointcloud.robot_filter import RemoveRobotPoints


def _sample(xyz, lidar_key="lidar"):
    pc = np.asarray(xyz, dtype=np.float32)
    if pc.ndim == 1:
        pc = pc.reshape(1, -1)
    if pc.shape[1] == 3:
        pc = np.column_stack([pc, np.zeros(len(pc), dtype=np.float32)])
    return Sample(data={lidar_key: pc})


# ------------------------------------------------------------------
# Constructor / class attributes
# ------------------------------------------------------------------

def test_default_output_key():
    assert RemoveRobotPoints().output_key == "lidar_no_robot"


def test_default_output_loader():
    assert RemoveRobotPoints().output_loader == "npys"


def test_custom_output_key():
    assert RemoveRobotPoints(output_key="filtered").output_key == "filtered"


def test_custom_lidar_key():
    proc = RemoveRobotPoints(lidar_key="velodyne")
    assert proc.input_keys == ["velodyne"]
    assert proc.sources == ["velodyne"]


def test_invalid_shape():
    with pytest.raises(ValueError):
        RemoveRobotPoints(shape="sphere")


def test_invalid_x_range():
    with pytest.raises(ValueError):
        RemoveRobotPoints(shape="box", x_range=(1.0, 0.0))


def test_invalid_y_range():
    with pytest.raises(ValueError):
        RemoveRobotPoints(shape="box", y_range=(0.5, 0.5))


def test_invalid_z_range_box():
    with pytest.raises(ValueError):
        RemoveRobotPoints(shape="box", z_range=(1.0, -1.0))


def test_invalid_z_range_cylinder():
    with pytest.raises(ValueError):
        RemoveRobotPoints(shape="cylinder", radius=1.0, z_range=(2.0, 1.0))


def test_invalid_radius():
    with pytest.raises(ValueError):
        RemoveRobotPoints(shape="cylinder", radius=0.0)
    with pytest.raises(ValueError):
        RemoveRobotPoints(shape="cylinder", radius=-0.5)


# ------------------------------------------------------------------
# Output shape and dtype
# ------------------------------------------------------------------

def test_output_dtype():
    xyz = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
    out = RemoveRobotPoints().process(_sample(xyz))
    assert out.dtype == np.float64


def test_output_preserves_extra_channels():
    # 5-channel point cloud (XYZ + intensity + ring)
    pc = np.ones((10, 5), dtype=np.float32)
    pc[:, :3] = 50.0  # far from origin — all kept
    sample = Sample(data={"lidar": pc})
    out = RemoveRobotPoints().process(sample)
    assert out.shape[1] == 5


def test_no_points_removed_when_outside():
    xyz = np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]], dtype=np.float32)
    proc = RemoveRobotPoints(shape="box", x_range=(-1.0, 1.0),
                             y_range=(-1.0, 1.0), z_range=(-1.0, 1.0))
    out = proc.process(_sample(xyz))
    assert out.shape[0] == 2


def test_all_points_removed_when_inside():
    xyz = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float32)
    proc = RemoveRobotPoints(shape="box", x_range=(-1.0, 1.0),
                             y_range=(-1.0, 1.0), z_range=(-1.0, 1.0))
    out = proc.process(_sample(xyz))
    assert out.shape[0] == 0


def test_empty_output_shape_has_correct_columns():
    xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    proc = RemoveRobotPoints(shape="box", x_range=(-1.0, 1.0),
                             y_range=(-1.0, 1.0), z_range=(-1.0, 1.0))
    out = proc.process(_sample(xyz))
    assert out.ndim == 2
    assert out.shape[1] == 4  # XYZ + extra channel added by _sample


# ------------------------------------------------------------------
# Box shape correctness
# ------------------------------------------------------------------

def test_box_boundary_inclusive():
    # Points exactly on the boundary should be removed.
    proc = RemoveRobotPoints(shape="box", x_range=(-1.0, 1.0),
                             y_range=(-1.0, 1.0), z_range=(-1.0, 1.0))
    on_boundary = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    out = proc.process(_sample(on_boundary))
    assert out.shape[0] == 0


def test_box_only_filters_xyz_axes():
    xyz = np.array([
        [0.0, 0.0, 0.0],   # inside — removed
        [2.0, 0.0, 0.0],   # outside X — kept
        [0.0, 2.0, 0.0],   # outside Y — kept
        [0.0, 0.0, 2.0],   # outside Z — kept
    ], dtype=np.float32)
    proc = RemoveRobotPoints(shape="box", x_range=(-1.0, 1.0),
                             y_range=(-1.0, 1.0), z_range=(-1.0, 1.0))
    out = proc.process(_sample(xyz))
    assert out.shape[0] == 3


def test_box_negative_coordinates():
    xyz = np.array([[-0.5, -0.5, -0.5]], dtype=np.float32)
    proc = RemoveRobotPoints(shape="box", x_range=(-1.0, 1.0),
                             y_range=(-1.0, 1.0), z_range=(-1.0, 1.0))
    out = proc.process(_sample(xyz))
    assert out.shape[0] == 0


def test_box_asymmetric_range():
    # Robot box shifted forward: x in [0, 2], y in [-0.5, 0.5], z in [-1, 0]
    proc = RemoveRobotPoints(shape="box", x_range=(0.0, 2.0),
                             y_range=(-0.5, 0.5), z_range=(-1.0, 0.0))
    inside = np.array([[1.0, 0.0, -0.5]], dtype=np.float32)
    outside = np.array([[-0.1, 0.0, -0.5], [1.0, 1.0, -0.5]], dtype=np.float32)
    assert proc.process(_sample(inside)).shape[0] == 0
    assert proc.process(_sample(outside)).shape[0] == 2


# ------------------------------------------------------------------
# Cylinder shape correctness
# ------------------------------------------------------------------

def test_cylinder_removes_center_point():
    xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    proc = RemoveRobotPoints(shape="cylinder", radius=1.0, z_range=(-1.0, 1.0))
    assert proc.process(_sample(xyz)).shape[0] == 0


def test_cylinder_keeps_radially_outside():
    xyz = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
    proc = RemoveRobotPoints(shape="cylinder", radius=1.0, z_range=(-1.0, 1.0))
    assert proc.process(_sample(xyz)).shape[0] == 1


def test_cylinder_keeps_vertically_outside():
    xyz = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
    proc = RemoveRobotPoints(shape="cylinder", radius=1.0, z_range=(-1.0, 1.0))
    assert proc.process(_sample(xyz)).shape[0] == 1


def test_cylinder_boundary_on_radius_inclusive():
    # Point at exactly radius — considered inside.
    xyz = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    proc = RemoveRobotPoints(shape="cylinder", radius=1.0, z_range=(-1.0, 1.0))
    assert proc.process(_sample(xyz)).shape[0] == 0


def test_cylinder_mixed_inside_outside():
    xyz = np.array([
        [0.0, 0.0, 0.0],    # inside — removed
        [0.5, 0.5, 0.0],    # inside (dist_xy ≈ 0.707 < 1.0) — removed
        [1.5, 0.0, 0.0],    # outside radius — kept
        [0.0, 0.0, 2.0],    # outside Z — kept
    ], dtype=np.float32)
    proc = RemoveRobotPoints(shape="cylinder", radius=1.0, z_range=(-1.0, 1.0))
    out = proc.process(_sample(xyz))
    assert out.shape[0] == 2


# ------------------------------------------------------------------
# Idempotency and statefulness
# ------------------------------------------------------------------

def test_stateless_same_input_same_output():
    proc = RemoveRobotPoints(shape="box", x_range=(-1.0, 1.0),
                             y_range=(-1.0, 1.0), z_range=(-1.0, 1.0))
    xyz = np.random.default_rng(42).uniform(-3, 3, (100, 3)).astype(np.float32)
    s = _sample(xyz)
    out1 = proc.process(s)
    out2 = proc.process(s)
    np.testing.assert_array_equal(out1, out2)
