"""Regression tests for the 0.3.0 correctness audit.

Each test states the physical expectation first; all of them failed, or
raised, before the fix they pin.
"""

import numpy as np
import pytest
from apairo.core.sample import Sample

from apairo_preprocess import (
    BEVRasterisation,
    GroundHeightFromLabels,
    GroundSegmentationRANSAC,
    ImageMaskFromPointLabels,
    LidarCameraProjection,
    TraversabilityFromTrajectory,
    VoxeliseCoords,
    VoxeliseLabels,
    VoxelisePointCloud,
)

# ------------------------------------------------------------------
# Voxelisation
# ------------------------------------------------------------------

_SIX = np.array(
    [[0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0], [1.1, 0, 0], [1.2, 0, 0], [1.3, 0, 0]]
)


def test_first_keeps_the_first_point_in_input_order():
    pc = np.array(
        [[0.1, 0.1, 0.1, 100.0], [0.5, 0.5, 0.5, 200.0], [0.9, 0.9, 0.9, 300.0]]
    )
    out = VoxelisePointCloud(voxel_size=1.0, reduction="first")(
        Sample(data={"lidar": pc})
    )
    np.testing.assert_array_equal(out, pc[:1])


def test_random_reduction_is_reproducible_with_a_seed():
    rng = np.random.default_rng(0)
    pc = rng.uniform(0, 3, (400, 4))
    s = Sample(data={"lidar": pc})
    a = VoxelisePointCloud(voxel_size=1.0, reduction="random", seed=7)(s)
    b = VoxelisePointCloud(voxel_size=1.0, reduction="random", seed=7)(s)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("aggregation", ["max", "majority"])
@pytest.mark.parametrize(
    "labels, expected",
    [
        ([0, 0, 0, -1, -1, -1], [0, -1]),
        ([1, 1, -1, -1, -1, -1], [1, -1]),
        ([-1, -1, -1, 0, 0, 0], [-1, 0]),
    ],
)
def test_negative_labels_are_aggregated_like_any_other(aggregation, labels, expected):
    s = Sample(data={"lidar": _SIX, "labels": np.array(labels, dtype=np.int32)})
    out = VoxeliseLabels(voxel_size=1.0, aggregation=aggregation)(s)
    assert out.tolist() == expected


def test_large_label_ids_do_not_allocate_a_table_of_their_size():
    # A panoptic id (instance << 16 | class) used to size a bincount table.
    labels = np.array([19660840, 19660840, 3, 7, 7, 3 << 20], dtype=np.int64)
    s = Sample(data={"lidar": _SIX, "labels": labels})
    out = VoxeliseLabels(voxel_size=1.0, aggregation="majority")(s)
    assert out.tolist() == [19660840, 7]


def test_nan_points_are_dropped_by_every_voxeliser():
    pc = np.array([[0.1, 0.1, 0.1], [np.nan, np.nan, np.nan], [2.1, 0.1, 0.1]])
    s = Sample(data={"lidar": pc})
    assert VoxeliseCoords(voxel_size=1.0)(s).tolist() == [[0, 0, 0], [2, 0, 0]]
    np.testing.assert_allclose(VoxelisePointCloud(voxel_size=1.0)(s), pc[[0, 2]])


# ------------------------------------------------------------------
# Traversability from trajectory
# ------------------------------------------------------------------


def _frames(poses, xyz):
    xyz = np.asarray(xyz, dtype=np.float32)
    return iter([Sample(data={"lidar": xyz, "poses": p}) for p in poses])


def _line(n, step):
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n) * step
    return poses


@pytest.mark.parametrize(
    "footprint", [{"robot_radius": 0.75}, {"robot_length": 1.0, "robot_width": 0.6}]
)
def test_pose_height_puts_the_window_on_the_ground_under_a_sensor_pose(footprint):
    # Lidar odometry poses are the sensor's, 1 m above flat ground: the ground
    # points under the future path sit at z = -1 in every pose frame.
    h = 1.0
    xyz = [[1, 0, -h], [2, 0, -h], [3, 0.2, -h], [5, 0, -h], [3, 6, -h]]
    out = TraversabilityFromTrajectory(pose_height=h, **footprint)(
        _frames(_line(6, 1.0), xyz)
    )
    assert out[0].tolist() == [1, 1, 1, 1, 0]
    # Without it the window sits around the sensor and nothing is ever labelled.
    blind = TraversabilityFromTrajectory(**footprint)(_frames(_line(6, 1.0), xyz))
    assert int(np.concatenate(list(blind)).sum()) == 0


@pytest.mark.parametrize("norm", [np.inf, 2])
def test_near_exclusion_runs_with_its_documented_norms(norm):
    out = TraversabilityFromTrajectory(
        robot_radius=0.5, near_exclusion_radius=1.0, near_exclusion_norm=norm
    )(_frames(_line(4, 2.0), [[2.0, 0, 0], [0.2, 0, 0]]))
    assert out[0].tolist() == [1, 0]


@pytest.mark.parametrize(
    "footprint", [{"robot_radius": 0.5}, {"robot_length": 1.0, "robot_width": 0.6}]
)
def test_a_nan_point_is_not_traversable_and_does_not_abort(footprint):
    xyz = [[2.0, 0, 0], [np.nan, np.nan, np.nan], [2.0, 5.0, 0]]
    out = TraversabilityFromTrajectory(**footprint)(_frames(_line(4, 2.0), xyz))
    assert out[0].tolist() == [1, 0, 0]


@pytest.mark.parametrize(
    "footprint", [{"robot_radius": 0.5}, {"robot_length": 1.0, "robot_width": 1.0}]
)
def test_height_is_tested_against_every_future_pose_not_the_nearest(footprint):
    # The path drives over the point, then crosses the same spot 3 m higher
    # and slightly closer in XY: the first passage makes it traversable.
    p = np.array(
        [[0, 0, 0], [2.0, 0.10, 0], [4, 0, 1.5], [2.0, 0.05, 3.0], [0, 0, 3.0]], float
    )
    poses = np.tile(np.eye(4), (len(p), 1, 1))
    poses[:, :3, 3] = p
    out = TraversabilityFromTrajectory(**footprint)(_frames(poses, [[2.0, 0, 0]]))
    assert out[0].tolist() == [1]


def test_an_empty_sequence_returns_no_rows():
    assert len(TraversabilityFromTrajectory()(iter([]))) == 0


# ------------------------------------------------------------------
# Projection
# ------------------------------------------------------------------


def test_points_beyond_the_distortion_model_do_not_fold_back_into_the_image():
    # k1 = -0.30: the radial polynomial stops being monotonic at
    # r^2 = 1/(3|k1|); points beyond it used to land inside the image.
    k = np.array([[400.0, 0, 320], [0, 400, 240], [0, 0, 1]])
    pts = np.array(
        [[0.05, 0.0, 10.0], [1.82, 0.0, 1.0], [3.0, 3.0, 2.0]], dtype=np.float32
    )
    proj = LidarCameraProjection(
        k, np.eye(4), image_size=(480, 640), distortion=[-0.30, 0, 0, 0, 0]
    )
    uv = proj(Sample(data={"lidar": pts}))
    assert np.isfinite(uv[0]).all()
    assert np.isnan(uv[1:]).all()
    mask = ImageMaskFromPointLabels(image_size=(480, 640), radius=3)(
        Sample(data={"trav_traj": np.array([1, 0, 0], np.uint8), "lidar_uv": uv})
    )
    assert mask[240, 322] == 1


def test_an_undistorted_projection_is_unchanged_by_the_limit():
    k = np.array([[400.0, 0, 320], [0, 400, 240], [0, 0, 1]])
    pts = np.array([[0.05, 0.0, 10.0], [7.0, 0.0, 10.0]], dtype=np.float32)
    uv = LidarCameraProjection(k, np.eye(4), image_size=(480, 640))(
        Sample(data={"lidar": pts})
    )
    np.testing.assert_allclose(uv[0], [322.0, 240.0, 10.0], atol=1e-4)
    np.testing.assert_allclose(uv[1], [600.0, 240.0, 10.0], atol=1e-4)


def test_fractional_labels_are_refused_by_the_image_mask():
    uv = np.array([[10.0, 10.0, 5.0]])
    with pytest.raises(ValueError, match="integers"):
        ImageMaskFromPointLabels(image_size=(32, 32))(
            Sample(data={"trav_traj": np.array([0.5]), "lidar_uv": uv})
        )


# ------------------------------------------------------------------
# BEV
# ------------------------------------------------------------------


@pytest.mark.parametrize("resolution", [0.4, 0.3, 0.35])
def test_bev_keeps_exactly_the_points_inside_the_bounds(resolution):
    # Bounds that are not a multiple of the resolution: the grid is rounded
    # up to cover them, and a point outside the bounds is still left out.
    b = BEVRasterisation(
        bounds=(0, 1.0, 0, 1.0),
        resolution=resolution,
        values_key="v",
        reduce="max",
        background=-1,
    )
    pts = np.array([[0.95, 0.5], [0.5, 0.05], [1.04, 0.5]])
    g = b(Sample(data={"lidar": pts, "v": np.array([1.0, 2.0, 3.0])}))
    assert b.shape[0] * resolution >= 1.0 and b.shape[1] * resolution >= 1.0
    assert sorted(g[g != -1].tolist()) == [1.0, 2.0]


@pytest.mark.parametrize("reduce, expected", [("max", 4.0), ("mean", 4.0)])
def test_bev_ignores_a_nan_value_next_to_a_real_one(reduce, expected):
    b = BEVRasterisation(
        bounds=(0, 2, 0, 1),
        resolution=1.0,
        values_key="v",
        reduce=reduce,
        background=0.0,
    )
    pts = np.array([[0.5, 0.5], [0.5, 0.5], [1.5, 0.5]])
    g = b(Sample(data={"lidar": pts, "v": np.array([4.0, np.nan, 7.0])}))
    assert g.tolist() == [[expected, 7.0]]


def test_bev_keeps_a_negative_maximum():
    # A cell whose only values are negative used to keep the background.
    b = BEVRasterisation(
        bounds=(-2, 2, -1, 1), resolution=1.0, values_key="v", reduce="max"
    )
    pts = np.array([[-1.5, 0.5], [1.5, -0.5], [1.5, -0.5]])
    g = b(Sample(data={"lidar": pts, "v": np.array([3.0, -2.0, -5.0])}))
    assert g[0, 0] == 3.0 and g[1, 3] == -2.0


def test_bev_cell_edges_do_not_drift_with_float_division():
    b = BEVRasterisation(
        bounds=(0, 1, 0, 1), resolution=0.1, values_key="v", reduce="max", background=-1
    )
    pts = np.array([[0.3, 0.95], [0.6, 0.95], [0.7, 0.95]])
    g = b(Sample(data={"lidar": pts, "v": np.array([1.0, 2.0, 3.0])}))
    assert np.flatnonzero(g[0] != -1).tolist() == [3, 6, 7]


# ------------------------------------------------------------------
# Ground, priors: empty or tiny clouds
# ------------------------------------------------------------------


def _empty_cloud():
    return VoxelisePointCloud(voxel_size=0.1, max_range=1.0)(
        Sample(data={"lidar": np.full((10, 3), 30.0)})
    )


def test_ransac_on_fewer_than_three_points_labels_nothing_as_ground_plane():
    empty = _empty_cloud()
    assert empty.shape == (0, 3)
    assert GroundSegmentationRANSAC()(Sample(data={"voxelised": empty})).shape == (0,)
    two = np.array([[0.0, 0, 0], [1, 0, 0]])
    assert GroundSegmentationRANSAC()(Sample(data={"voxelised": two})).shape == (2,)


def test_ground_height_on_an_empty_cloud_is_empty():
    out = GroundHeightFromLabels("g")(
        Sample(data={"voxelised": _empty_cloud(), "g": np.empty(0, np.uint8)})
    )
    assert out.shape == (0,)


def test_csf_on_an_empty_cloud_is_empty():
    pytest.importorskip("CSF", reason="cloth-simulation-filter not installed")
    from apairo_preprocess import GroundSegmentationCSF

    out = GroundSegmentationCSF()(Sample(data={"voxelised": _empty_cloud()}))
    assert out.shape == (0,)


# ------------------------------------------------------------------
# Odometry: one world origin per sequence
# ------------------------------------------------------------------


def _cloud(seed):
    return np.random.default_rng(seed).uniform(-10, 10, (500, 3)).astype(np.float32)


def test_gicp_start_sequence_resets_the_pose():
    pytest.importorskip("open3d", reason="open3d not installed")
    from apairo_preprocess import GICPOdometry

    proc = GICPOdometry()
    proc(Sample(data={"lidar": _cloud(0)}))
    proc(Sample(data={"lidar": _cloud(0) + np.float32([0.3, 0, 0])}))
    proc.start_sequence()
    np.testing.assert_allclose(proc(Sample(data={"lidar": _cloud(1)})), np.eye(4))


def test_kissicp_start_sequence_resets_the_pose():
    pytest.importorskip("kiss_icp", reason="kiss-icp not installed")
    from apairo_preprocess import KissICPOdometry

    proc = KissICPOdometry()
    proc(Sample(data={"lidar": _cloud(0)}))
    proc(Sample(data={"lidar": _cloud(0) + np.float32([0.3, 0, 0])}))
    proc.start_sequence()
    np.testing.assert_allclose(
        proc(Sample(data={"lidar": _cloud(1)})), np.eye(4), atol=1e-9
    )
