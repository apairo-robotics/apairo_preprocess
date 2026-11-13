"""Delivered poses on the lidar clock: interpolation, validity, labelling.

Each test states the physical expectation first.
"""

import numpy as np
import pytest
from apairo.core.sample import Sample
from scipy.spatial.transform import Rotation

from apairo_preprocess import (
    ReferencePoses,
    TraversabilityFromTrajectory,
    pose_validity,
    reference_poses,
)


def _rows(t, x, y, z, yaw):
    """Rows [x y z qx qy qz qw] of a body yawing in the plane."""
    q = Rotation.from_euler("z", np.asarray(yaw, dtype=float).reshape(-1, 1)).as_quat()
    return np.column_stack([x, y, z, q])


def _true_pose(x, y, z, yaw):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


# ------------------------------------------------------------------ interpolation


def test_a_straight_line_is_interpolated_exactly():
    # 5 m/s along x, rows at 50 Hz, scans at 10 Hz in between the rows.
    t = np.arange(0, 2.0001, 0.02)
    rows = _rows(t, 5 * t, 0 * t, 0 * t, 0 * t)
    scans = np.arange(0.005, 1.9, 0.1)
    poses, valid = reference_poses(rows, t, scans, relative=False)
    assert valid.all()
    np.testing.assert_allclose(poses[:, 0, 3], 5 * scans, atol=1e-9)
    np.testing.assert_allclose(
        poses[:, :3, :3], np.tile(np.eye(3), (len(scans), 1, 1)), atol=1e-12
    )


def test_a_turn_is_interpolated_on_position_and_heading():
    # Constant yaw rate: both position (lerp between close rows) and heading
    # (slerp) match the truth at mid-row times.
    t = np.arange(0, 2.0001, 0.02)
    yaw = 0.5 * t
    rows = _rows(t, np.cos(yaw), np.sin(yaw), 0 * t, yaw)
    scans = np.array([0.31, 0.97, 1.53])
    poses, valid = reference_poses(rows, t, scans, relative=False)
    assert valid.all()
    got_yaw = np.arctan2(poses[:, 1, 0], poses[:, 0, 0])
    np.testing.assert_allclose(got_yaw, 0.5 * scans, atol=1e-9)
    # lerp of a circle's chord: error below r * (dtheta)^2 / 8 = 1.25e-5 m
    np.testing.assert_allclose(poses[:, 0, 3], np.cos(0.5 * scans), atol=2e-5)


def test_the_clock_offset_shifts_the_lookup():
    # The pose clock runs 0.05 s behind the lidar's: the pose of a scan at t
    # is the row at t - 0.05.
    t = np.arange(0, 2.0001, 0.02)
    rows = _rows(t, 5 * t, 0 * t, 0 * t, 0 * t)
    scans = np.array([0.5, 1.0])
    poses, _ = reference_poses(rows, t, scans, clock_offset=-0.05, relative=False)
    np.testing.assert_allclose(poses[:, 0, 3], 5 * (scans - 0.05), atol=1e-9)


def test_utm_scale_positions_keep_their_millimetres_relative_to_the_first():
    # Northing 4.48e6 m: relative to the first scan, in float64, the 1 mm
    # steps survive; a float32 cast of the absolute value would not.
    t = np.arange(0, 1.0001, 0.02)
    x = 4.48e6 + 0.001 * np.arange(len(t))
    rows = _rows(t, x, -6.03e5 + 0 * t, 300 + 0 * t, 0 * t)
    poses, valid = reference_poses(rows, t, t[:10])
    assert valid.all()
    np.testing.assert_allclose(poses[:, 0, 3], 0.001 * np.arange(10), atol=1e-7)
    np.testing.assert_allclose(poses[0], np.eye(4), atol=1e-12)
    assert np.ptp(poses[:, 0, 3].astype(np.float32)) > 0.008


def test_an_extrinsic_is_composed_on_the_right():
    # Body at (10, 0, 0) facing +y; sensor 1 m ahead of the body, 2 m up:
    # in the world it sits at (10, 1, 2), facing +y.
    t = np.array([0.0, 0.1])
    rows = _rows(
        t, np.array([10.0, 10.0]), np.zeros(2), np.zeros(2), np.full(2, np.pi / 2)
    )
    sensor = np.eye(4)
    sensor[:3, 3] = [1.0, 0.0, 2.0]
    poses, _ = reference_poses(
        rows, t, np.array([0.05]), extrinsic=sensor, relative=False
    )
    np.testing.assert_allclose(poses[0, :3, 3], [10.0, 1.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(
        poses[0, :3, :3], Rotation.from_euler("z", np.pi / 2).as_matrix(), atol=1e-12
    )


# ---------------------------------------------------------------------- validity


def test_scans_inside_a_hole_in_the_rows_have_no_pose():
    # Rows stop for 0.5 s: the scans inside are invalid, not interpolated
    # across the hole; their pose is NaN.
    t = np.concatenate([np.arange(0, 1.0, 0.02), np.arange(1.5, 2.5, 0.02)])
    rows = _rows(t, 5 * t, 0 * t, 0 * t, 0 * t)
    scans = np.array([0.5, 1.2, 1.3, 2.0])
    poses, valid = reference_poses(rows, t, scans, max_gap=0.1)
    assert valid.tolist() == [True, False, False, True]
    assert np.isnan(poses[1:3]).all() and np.isfinite(poses[[0, 3]]).all()


def test_nothing_is_extrapolated_outside_the_recorded_span():
    t = np.arange(1.0, 2.0001, 0.02)
    rows = _rows(t, t, 0 * t, 0 * t, 0 * t)
    _, valid = reference_poses(rows, t, np.array([0.9, 1.5, 2.1]))
    assert valid.tolist() == [False, True, False]


def test_a_repeated_row_is_dropped_and_a_clock_going_back_is_refused():
    t = np.array([0.0, 0.02, 0.02, 0.04])
    rows = _rows(t, t, 0 * t, 0 * t, 0 * t)
    _, valid = reference_poses(rows, t, np.array([0.01, 0.03]))
    assert valid.all()
    with pytest.raises(ValueError, match="backwards"):
        reference_poses(rows, np.array([0.0, 0.02, 0.01, 0.04]), np.array([0.01]))


def test_a_jump_beyond_the_top_speed_invalidates_both_scans_of_the_step():
    # 5 m/s, except one row 5 m off: the step into it and the step out of it
    # both exceed 15 m/s; the scan before, the jumped one and the one after
    # are invalid.
    t = np.arange(0, 2.0001, 0.1)
    x = 5 * t
    x[10] += 5.0
    rows = _rows(t, x, 0 * t, 0 * t, 0 * t)
    poses, valid = reference_poses(rows, t, t, max_speed=15.0)
    assert np.flatnonzero(~valid).tolist() == [9, 10, 11]
    assert np.isnan(poses[~valid]).all()


def test_pose_validity_needs_a_positive_top_speed_and_checks_turns():
    t = np.arange(0, 1.0001, 0.1)
    yaw = np.zeros_like(t)
    yaw[5] = 1.0  # a 1 rad flick in 0.1 s
    poses = np.stack([_true_pose(ti, 0, 0, a) for ti, a in zip(t, yaw)])
    assert pose_validity(poses, t, max_speed=15.0).all()
    assert np.flatnonzero(
        ~pose_validity(poses, t, max_speed=15.0, max_yaw_rate=2.0)
    ).tolist() == [4, 5, 6]
    with pytest.raises(ValueError):
        pose_validity(poses, t, max_speed=0.0)


def test_a_scan_repeated_under_the_same_stamp_is_not_a_jump():
    # Real recordings repeat a scan under the same stamp: no time elapses,
    # there is no speed to check, and both copies keep their pose.
    t = np.arange(0, 1.0001, 0.02)
    rows = _rows(t, 5 * t, 0 * t, 0 * t, 0 * t)
    scans = np.array([0.1, 0.2, 0.2, 0.3])
    poses, valid = reference_poses(rows, t, scans, max_speed=15.0)
    assert valid.all()
    np.testing.assert_array_equal(poses[1], poses[2])
    with pytest.raises(ValueError, match="backwards"):
        pose_validity(poses, np.array([0.1, 0.2, 0.15, 0.3]), max_speed=15.0)


def test_reference_poses_class_keeps_its_configuration():
    rp = ReferencePoses(clock_offset=-0.05, max_gap=0.1, max_speed=20.0)
    assert rp.get_config()["clock_offset"] == -0.05
    t = np.arange(0, 1.0001, 0.02)
    rows = _rows(t, t, 0 * t, 0 * t, 0 * t)
    a = rp(rows, t, np.array([0.5]))
    b = reference_poses(rows, t, np.array([0.5]), clock_offset=-0.05, max_speed=20.0)
    np.testing.assert_array_equal(a[0], b[0])


# ----------------------------------------------------------------- labelling


def _frames(poses, valid, xyz, valid_key="valid"):
    xyz = np.asarray(xyz, dtype=np.float32)
    return iter(
        [
            Sample(data={"lidar": xyz, "poses": p, valid_key: np.bool_(v)})
            for p, v in zip(poses, valid)
        ]
    )


def _line(n, step):
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n) * step
    return poses


def test_a_non_finite_pose_without_valid_key_is_an_error():
    poses = _line(4, 1.0)
    poses[2] = np.nan
    frames = iter(
        [
            Sample(data={"lidar": np.zeros((1, 3), np.float32), "poses": p})
            for p in poses
        ]
    )
    with pytest.raises(ValueError, match="valid_key"):
        TraversabilityFromTrajectory(robot_radius=0.5)(frames)


def test_an_invalid_scan_cuts_the_look_ahead_and_is_marked_unknown():
    # Straight line, 1 m steps, a point 2 m ahead of each scan, on the path.
    # Scan 1 is invalid. From scan 0 the point lies on scan 2, but the
    # look-ahead stops at scan 1: not traversable. Scan 1's own points are
    # unknown (255). From scan 2 the point lies on scan 4: traversable.
    poses = _line(6, 1.0)
    poses[1] = np.nan
    valid = [True, False, True, True, True, True]
    xyz = [[2.0, 0.0, 0.0]]
    lab = TraversabilityFromTrajectory(
        robot_radius=0.3, valid_key="valid", unknown_label=255
    )(_frames(poses, valid, xyz))
    assert lab[0].tolist() == [0]
    assert lab[1].tolist() == [255]
    assert lab[2].tolist() == [1]


def test_without_unknown_label_an_invalid_scan_is_zero():
    poses = _line(4, 1.0)
    poses[1] = np.nan
    lab = TraversabilityFromTrajectory(robot_radius=0.3, valid_key="valid")(
        _frames(poses, [True, False, True, True], [[1.0, 0.0, 0.0]])
    )
    assert lab[1].tolist() == [0]


def test_unknown_label_must_not_collide_with_the_labels():
    for bad in (0, 1, 256):
        with pytest.raises(ValueError):
            TraversabilityFromTrajectory(valid_key="valid", unknown_label=bad)


def test_valid_key_is_an_input_and_a_source():
    p = TraversabilityFromTrajectory(valid_key="ins_valid")
    assert p.input_keys == ["lidar", "poses", "ins_valid"]
    assert p.sources == ["lidar", "poses", "ins_valid"]


def test_end_to_end_delivered_rows_give_the_labels_of_the_true_poses():
    # A drive with a turn, recorded by a 50 Hz INS 1.2 m above the ground,
    # scans at 10 Hz on a lidar clock 0.05 s ahead: labelling the poses
    # rebuilt from the rows gives exactly the labels of the true poses.
    t = np.arange(0, 12.0001, 0.02)
    yaw = np.where(t < 6, 0.0, 0.3 * (t - 6))
    x = np.cumsum(np.r_[0, 3.0 * np.cos(yaw[:-1]) * 0.02])
    y = np.cumsum(np.r_[0, 3.0 * np.sin(yaw[:-1]) * 0.02])
    rows = _rows(t, x, y, np.full_like(t, 1.2), yaw)
    scans = np.arange(0.1, 11.0, 0.1)
    poses, valid = reference_poses(rows, t, scans, clock_offset=-0.05, max_speed=10.0)
    assert valid.all()

    ts = scans - 0.05
    truth = np.stack(
        [
            _true_pose(
                np.interp(s, t, x), np.interp(s, t, y), 1.2, np.interp(s, t, yaw)
            )
            for s in ts
        ]
    )
    truth = np.linalg.inv(truth[0]) @ truth
    rng = np.random.default_rng(0)
    ground = np.column_stack(
        [rng.uniform(0, 12, 400), rng.uniform(-2, 2, 400), np.full(400, -1.2)]
    ).astype(np.float32)
    kw = dict(robot_radius=0.6, pose_height=1.2, forward_distance=15.0)
    a = TraversabilityFromTrajectory(**kw)(
        iter([Sample(data={"lidar": ground, "poses": p}) for p in poses])
    )
    b = TraversabilityFromTrajectory(**kw)(
        iter([Sample(data={"lidar": ground, "poses": p}) for p in truth])
    )
    assert int(np.asarray(b).sum()) > 0
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
