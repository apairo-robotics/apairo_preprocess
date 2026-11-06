import numpy as np
import pytest
from apairo.core.sample import Sample

from apairo_preprocess.traversability.from_trajectory import TraversabilityFromTrajectory


def _straight_line_poses(n, step=2.0):
    """N poses along the X axis, one every `step` metres."""
    poses = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n) * step
    return poses


def _sample(xyz, poses_key="poses"):
    pose = np.eye(4, dtype=np.float64)
    return Sample(data={"lidar": np.asarray(xyz, dtype=np.float32), poses_key: pose})


def _samples(poses, xyz_per_frame=None, poses_key="poses"):
    """Build a list of samples aligned with the given pose array."""
    n = len(poses)
    result = []
    for i in range(n):
        xyz = xyz_per_frame[i] if xyz_per_frame is not None else np.zeros((5, 3), dtype=np.float32)
        result.append(Sample(data={"lidar": np.asarray(xyz, dtype=np.float32), poses_key: poses[i]}))
    return result


def _proc(poses, **kwargs):
    """Run a TraversabilityFromTrajectory on a fixed pose sequence and return all outputs."""
    proc = TraversabilityFromTrajectory(**kwargs)
    samples = _samples(poses)
    return proc(iter(samples))


# ------------------------------------------------------------------
# Constructor / API
# ------------------------------------------------------------------

def test_default_input_keys():
    proc = TraversabilityFromTrajectory()
    assert proc.input_keys == ["lidar", "poses"]
    assert proc.sources == ["lidar", "poses"]


def test_custom_keys():
    proc = TraversabilityFromTrajectory(lidar_key="velodyne", poses_key="kissicp_poses")
    assert "velodyne" in proc.input_keys
    assert "kissicp_poses" in proc.input_keys
    assert "velodyne" in proc.sources
    assert "kissicp_poses" in proc.sources


def test_custom_output_key():
    proc = TraversabilityFromTrajectory(output_key="custom_trav")
    assert proc.output_key == "custom_trav"


def test_default_output_key():
    assert TraversabilityFromTrajectory().output_key == "trav_traj"


# ------------------------------------------------------------------
# Algorithm correctness
# ------------------------------------------------------------------

def test_output_shape_and_dtype():
    poses = _straight_line_poses(5)
    out = _proc(poses)
    assert out.dtype == np.uint8
    assert out.shape == (5, 5)  # 5 frames x 5 points (default zeros)


def test_point_on_future_pose_is_traversable():
    step = 2.0
    poses = _straight_line_poses(5, step)
    proc = TraversabilityFromTrajectory(robot_radius=0.5, height_min=-0.5, height_max=0.5)
    # Frame 0: robot at (0,0,0); point at (step, 0, 0) is on the next footprint.
    xyz = np.array([[step, 0.0, 0.0]], dtype=np.float32)
    samples = [
        Sample(data={"lidar": xyz, "poses": poses[i]})
        for i in range(len(poses))
    ]
    out = proc(iter(samples))
    assert out[0, 0] == 1


def test_point_outside_xy_radius_not_traversable():
    step = 2.0
    poses = _straight_line_poses(5, step)
    proc = TraversabilityFromTrajectory(robot_radius=0.5)
    xyz = np.array([[step, 10.0, 0.0]], dtype=np.float32)
    samples = [Sample(data={"lidar": xyz, "poses": poses[i]}) for i in range(len(poses))]
    out = proc(iter(samples))
    assert out[0, 0] == 0


def test_height_filter():
    step = 2.0
    poses = _straight_line_poses(5, step)
    proc = TraversabilityFromTrajectory(robot_radius=1.0, height_min=-0.3, height_max=0.5)
    samples_high = [Sample(data={"lidar": np.array([[step, 0.0, 1.0]], dtype=np.float32), "poses": poses[i]}) for i in range(len(poses))]
    samples_low = [Sample(data={"lidar": np.array([[step, 0.0, -0.5]], dtype=np.float32), "poses": poses[i]}) for i in range(len(poses))]
    samples_ok = [Sample(data={"lidar": np.array([[step, 0.0, 0.2]], dtype=np.float32), "poses": poses[i]}) for i in range(len(poses))]
    assert proc(iter(samples_high))[0, 0] == 0
    assert proc(iter(samples_low))[0, 0] == 0
    assert proc(iter(samples_ok))[0, 0] == 1


def test_last_frame_all_zeros():
    n = 4
    poses = _straight_line_poses(n)
    proc = TraversabilityFromTrajectory()
    out = _proc(poses)
    np.testing.assert_array_equal(out[n - 1], np.zeros(5, dtype=np.uint8))


def test_forward_window_limits_look_ahead():
    step = 2.0
    n = 5
    poses = _straight_line_poses(n, step)
    proc = TraversabilityFromTrajectory(robot_radius=0.5, forward_window=1)
    xyz = np.array([[3 * step, 0.0, 0.0]], dtype=np.float32)
    samples = [Sample(data={"lidar": xyz, "poses": poses[i]}) for i in range(n)]
    out = proc(iter(samples))
    # Frame 0 with window=1 only sees poses[1]; point at 3*step is out of window.
    assert out[0, 0] == 0


def test_forward_distance_caps_look_ahead_by_path_length():
    step = 2.0
    poses = _straight_line_poses(5, step)  # x = 0, 2, 4, 6, 8
    # Seen from frame 0 the point sits on the pose at x = 4, i.e. 4 m of path away.
    xyz = np.array([[2 * step, 0.0, 0.0]], dtype=np.float32)
    samples = lambda: iter(
        [Sample(data={"lidar": xyz, "poses": poses[i]}) for i in range(5)]
    )
    kwargs = dict(robot_radius=0.5, height_min=-0.5, height_max=0.5)

    assert TraversabilityFromTrajectory(**kwargs)(samples())[0, 0] == 1
    # 3 m of path from frame 0 only reaches the pose at x=2: x=4 is out.
    proc = TraversabilityFromTrajectory(**kwargs, forward_distance=3.0)
    assert proc(samples())[0, 0] == 0
    # The cap is inclusive: 4 m of path reaches the pose the point sits on.
    proc = TraversabilityFromTrajectory(**kwargs, forward_distance=4.0)
    assert proc(samples())[0, 0] == 1


def test_sequence_gap_prevents_cross_sequence_look_ahead():
    step = 2.0
    poses = np.tile(np.eye(4, dtype=np.float64), (6, 1, 1))
    poses[0, 0, 3] = 0.0
    poses[1, 0, 3] = step
    poses[2, 0, 3] = step * 2
    poses[3, 0, 3] = 100.0
    poses[4, 0, 3] = 100.0 + step
    poses[5, 0, 3] = 100.0 + step * 2

    proc = TraversabilityFromTrajectory(robot_radius=0.5, sequence_gap=5.0)
    xyz = np.array([[100.0, 0.0, 0.0]], dtype=np.float32)
    samples = [Sample(data={"lidar": xyz, "poses": poses[i]}) for i in range(6)]
    out = proc(iter(samples))
    # Frame 2 is the last in seq 1; point at x=100 (seq 2) must NOT be traversable.
    assert out[2, 0] == 0


def test_compact_3x4_poses_rejected():
    # Pose conversion is apairo_transform's job (PoseTo4x4); the preprocessor
    # demands strict (4, 4) and says so.
    poses_34 = _straight_line_poses(5)[:, :3, :]  # (N, 3, 4)
    proc = TraversabilityFromTrajectory()
    samples = [Sample(data={"lidar": np.zeros((5, 3), dtype=np.float32), "poses": poses_34[i]}) for i in range(len(poses_34))]
    with pytest.raises(ValueError, match="PoseTo4x4"):
        proc(iter(samples))


def test_ragged_scans_return_per_frame_rows():
    # Real lidar scans have a variable point count: the result is one row per
    # frame (object array), each row aligned with its scan.
    poses = _straight_line_poses(3)
    sizes = [5, 3, 7]
    xyz_per_frame = [np.zeros((s, 3), dtype=np.float32) for s in sizes]
    proc = TraversabilityFromTrajectory()
    out = proc(iter(_samples(poses, xyz_per_frame=xyz_per_frame)))
    assert out.dtype == object
    assert [len(row) for row in out] == sizes
    assert all(row.dtype == np.uint8 for row in out)


# ------------------------------------------------------------------
# Oriented (heading-aligned rectangle) footprint
# ------------------------------------------------------------------

def _yaw_poses(n, step=2.0, yaw=0.0):
    """N poses spaced `step` apart along a straight line at heading `yaw`."""
    poses = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    c, s = np.cos(yaw), np.sin(yaw)
    poses[:, 0, 0], poses[:, 0, 1] = c, -s
    poses[:, 1, 0], poses[:, 1, 1] = s, c
    for i in range(n):
        poses[i, 0, 3] = i * step * c
        poses[i, 1, 3] = i * step * s
    return poses


def test_length_and_width_must_be_given_together():
    with pytest.raises(ValueError, match="robot_length and robot_width"):
        TraversabilityFromTrajectory(robot_length=1.05)
    with pytest.raises(ValueError, match="robot_length and robot_width"):
        TraversabilityFromTrajectory(robot_width=0.725)


def test_oriented_point_ahead_on_centerline_is_traversable():
    step = 2.0
    poses = _straight_line_poses(5, step)
    proc = TraversabilityFromTrajectory(
        robot_length=1.0, robot_width=0.6, height_min=-0.5, height_max=0.5
    )
    xyz = np.array([[step, 0.0, 0.0]], dtype=np.float32)
    samples = [Sample(data={"lidar": xyz, "poses": poses[i]}) for i in range(len(poses))]
    out = proc.process(iter(samples))
    assert out[0, 0] == 1


def test_oriented_rejects_lateral_offset_a_circle_would_accept():
    # robot_radius=0.9 (diameter 1.8m) would accept a 0.7m lateral offset; a
    # heading-aligned box that is only 0.6m wide (half_width=0.3) must not.
    step = 2.0
    poses = _straight_line_poses(5, step)
    lateral_offset = 0.7
    xyz = np.array([[step, lateral_offset, 0.0]], dtype=np.float32)
    samples = [Sample(data={"lidar": xyz, "poses": poses[i]}) for i in range(len(poses))]

    circle = TraversabilityFromTrajectory(robot_radius=0.9, height_min=-0.5, height_max=0.5)
    assert circle.process(iter(samples))[0, 0] == 1

    box = TraversabilityFromTrajectory(
        robot_length=1.0, robot_width=0.6, height_min=-0.5, height_max=0.5
    )
    assert box.process(iter(samples))[0, 0] == 0


def test_oriented_accepts_lateral_offset_within_half_width():
    step = 2.0
    poses = _straight_line_poses(5, step)
    xyz = np.array([[step, 0.25, 0.0]], dtype=np.float32)
    samples = [Sample(data={"lidar": xyz, "poses": poses[i]}) for i in range(len(poses))]
    proc = TraversabilityFromTrajectory(
        robot_length=1.0, robot_width=0.6, height_min=-0.5, height_max=0.5
    )
    assert proc.process(iter(samples))[0, 0] == 1


def test_oriented_uses_pose_rotation_for_heading_on_a_turn():
    # A 90-degree turn: the second pose's box is oriented along +Y, not +X.
    # A point that sits laterally in world-X but longitudinally in the turned
    # frame must only be accepted once the box is heading-aligned.
    poses = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
    poses[0] = _yaw_poses(1, yaw=0.0)[0]
    poses[1] = _yaw_poses(1, yaw=np.pi / 2)[0]
    poses[1, 0, 3] = 2.0  # pose 1 is 2m ahead on +X, but facing +Y

    proc = TraversabilityFromTrajectory(
        robot_length=1.0, robot_width=0.4, height_min=-0.5, height_max=0.5
    )
    # Point 0.15m to the +X side of pose 1 (within half_width if box faces +Y,
    # since local-y there maps to world-x displacement from pose 1's origin).
    xyz = np.array([[2.15, 0.0, 0.0]], dtype=np.float32)
    samples = [Sample(data={"lidar": xyz, "poses": poses[0]}), Sample(data={"lidar": xyz, "poses": poses[1]})]
    out = proc.process(iter(samples))
    assert out[0, 0] == 1  # 0.15m lateral to pose1's local frame is inside half_width=0.2

    xyz_far = np.array([[2.35, 0.0, 0.0]], dtype=np.float32)
    samples_far = [Sample(data={"lidar": xyz_far, "poses": poses[0]}), Sample(data={"lidar": xyz_far, "poses": poses[1]})]
    out_far = proc.process(iter(samples_far))
    assert out_far[0, 0] == 0  # 0.35m lateral is outside half_width=0.2


def test_oriented_height_filter_still_applies():
    step = 2.0
    poses = _straight_line_poses(5, step)
    proc = TraversabilityFromTrajectory(
        robot_length=1.0, robot_width=0.6, height_min=-0.3, height_max=0.5
    )
    samples_high = [Sample(data={"lidar": np.array([[step, 0.0, 1.0]], dtype=np.float32), "poses": poses[i]}) for i in range(len(poses))]
    samples_ok = [Sample(data={"lidar": np.array([[step, 0.0, 0.2]], dtype=np.float32), "poses": poses[i]}) for i in range(len(poses))]
    assert proc.process(iter(samples_high))[0, 0] == 0
    assert proc.process(iter(samples_ok))[0, 0] == 1


def test_oriented_ragged_scans_return_per_frame_rows():
    poses = _straight_line_poses(3)
    sizes = [5, 3, 7]
    xyz_per_frame = [np.zeros((s, 3), dtype=np.float32) for s in sizes]
    proc = TraversabilityFromTrajectory(robot_length=1.0, robot_width=0.6)
    out = proc.process(iter(_samples(poses, xyz_per_frame=xyz_per_frame)))
    assert out.dtype == object
    assert [len(row) for row in out] == sizes
    assert all(row.dtype == np.uint8 for row in out)
