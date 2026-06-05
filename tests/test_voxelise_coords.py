import numpy as np
import pytest
from apairo.core.sample import Sample

from apairo_preprocess.pointcloud.voxelise import VoxeliseCoords, VoxelisePointCloud


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
    assert VoxeliseCoords().output_key == "voxel_coords"


def test_default_output_loader():
    assert VoxeliseCoords().output_loader == "npys"


def test_custom_output_key():
    assert VoxeliseCoords(output_key="coords").output_key == "coords"


def test_custom_lidar_key():
    proc = VoxeliseCoords(lidar_key="velodyne")
    assert proc.input_keys == ["velodyne"]
    assert proc.sources == ["velodyne"]


def test_invalid_voxel_size():
    with pytest.raises(ValueError):
        VoxeliseCoords(voxel_size=0.0)
    with pytest.raises(ValueError):
        VoxeliseCoords(voxel_size=-1.0)


def test_invalid_max_range():
    with pytest.raises(ValueError):
        VoxeliseCoords(max_range=-5.0)


# ------------------------------------------------------------------
# Output shape and dtype
# ------------------------------------------------------------------

def test_output_dtype():
    xyz = np.array([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]], dtype=np.float32)
    out = VoxeliseCoords(voxel_size=1.0).process(_sample(xyz))
    assert out.dtype == np.int32


def test_output_shape_columns():
    xyz = np.random.default_rng(0).uniform(-5, 5, (100, 3)).astype(np.float32)
    out = VoxeliseCoords(voxel_size=0.5).process(_sample(xyz))
    assert out.ndim == 2
    assert out.shape[1] == 3


def test_at_most_as_many_voxels_as_points():
    xyz = np.random.default_rng(1).uniform(-5, 5, (200, 3)).astype(np.float32)
    out = VoxeliseCoords(voxel_size=0.5).process(_sample(xyz))
    assert out.shape[0] <= 200


# ------------------------------------------------------------------
# Correctness
# ------------------------------------------------------------------

def test_coords_equal_floor_division():
    # Place one point per voxel at exact voxel centres — coords must be predictable.
    voxel_size = 1.0
    xyz = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [2.5, 0.5, 0.5]], dtype=np.float32)
    out = VoxeliseCoords(voxel_size=voxel_size).process(_sample(xyz))
    expected = np.floor(xyz / voxel_size).astype(np.int32)
    # np.unique sorts rows, so sort expected the same way.
    expected_sorted = expected[np.lexsort(expected.T[::-1])]
    np.testing.assert_array_equal(out, expected_sorted)


def test_two_points_same_voxel_yield_one_coord():
    voxel_size = 1.0
    # Both points fall inside voxel (0, 0, 0).
    xyz = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]], dtype=np.float32)
    out = VoxeliseCoords(voxel_size=voxel_size).process(_sample(xyz))
    assert out.shape == (1, 3)
    np.testing.assert_array_equal(out[0], [0, 0, 0])


def test_negative_coords():
    voxel_size = 1.0
    xyz = np.array([[-0.5, -0.5, -0.5]], dtype=np.float32)
    out = VoxeliseCoords(voxel_size=voxel_size).process(_sample(xyz))
    np.testing.assert_array_equal(out[0], [-1, -1, -1])


def test_coords_are_unique_rows():
    rng = np.random.default_rng(7)
    xyz = rng.uniform(-3, 3, (500, 3)).astype(np.float32)
    out = VoxeliseCoords(voxel_size=0.5).process(_sample(xyz))
    unique_rows = np.unique(out, axis=0)
    assert unique_rows.shape == out.shape


# ------------------------------------------------------------------
# max_range filter
# ------------------------------------------------------------------

def test_max_range_excludes_far_points():
    xyz = np.array([[1.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32)
    out = VoxeliseCoords(voxel_size=1.0, max_range=10.0).process(_sample(xyz))
    assert out.shape[0] == 1
    np.testing.assert_array_equal(out[0], [1, 0, 0])


def test_max_range_none_keeps_all():
    rng = np.random.default_rng(3)
    xyz = rng.uniform(-10, 10, (300, 3)).astype(np.float32)
    without_filter = VoxeliseCoords(voxel_size=1.0).process(_sample(xyz))
    with_large_filter = VoxeliseCoords(voxel_size=1.0, max_range=1000.0).process(_sample(xyz))
    assert without_filter.shape == with_large_filter.shape


def test_empty_after_max_range():
    xyz = np.array([[50.0, 0.0, 0.0]], dtype=np.float32)
    out = VoxeliseCoords(voxel_size=1.0, max_range=1.0).process(_sample(xyz))
    assert out.shape == (0, 3)


# ------------------------------------------------------------------
# Alignment with VoxelisePointCloud
# ------------------------------------------------------------------

def test_n_voxels_matches_voxelise_point_cloud():
    rng = np.random.default_rng(99)
    xyz = rng.uniform(-5, 5, (400, 3)).astype(np.float32)
    voxel_size, max_range = 0.5, 7.0

    coords_out = VoxeliseCoords(voxel_size=voxel_size, max_range=max_range).process(_sample(xyz))
    pc_out = VoxelisePointCloud(voxel_size=voxel_size, max_range=max_range).process(_sample(xyz))

    assert coords_out.shape[0] == pc_out.shape[0]
