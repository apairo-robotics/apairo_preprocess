import numpy as np
import pytest
from apairo.core.sample import Sample

from apairo_preprocess.traversability.from_trajectory import TraversabilityFromTrajectory


def _straight_line_poses(n, step=2.0):
    """N poses along the X axis, one every `step` metres."""
    poses = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n) * step
    return poses


def _sample(xyz):
    return Sample(data={"lidar": np.asarray(xyz, dtype=np.float32)})


def test_output_dtype_and_shape():
    poses = _straight_line_poses(5)
    proc = TraversabilityFromTrajectory(poses)
    out = proc.process(_sample(np.zeros((10, 3))))
    assert out.dtype == np.uint8
    assert out.shape == (10,)


def test_point_on_future_pose_is_traversable():
    # Frame 0: robot at (0,0,0), next poses at (2,0,0), (4,0,0), …
    # A point at (2,0,0) in sensor frame (= world, identity pose) is exactly on the next footprint.
    step = 2.0
    poses = _straight_line_poses(5, step)
    proc = TraversabilityFromTrajectory(poses, robot_radius=0.5, height_min=-0.5, height_max=0.5)
    out = proc.process(_sample([[step, 0.0, 0.0]]))
    assert out[0] == 1


def test_point_outside_xy_radius_is_not_traversable():
    step = 2.0
    poses = _straight_line_poses(5, step)
    proc = TraversabilityFromTrajectory(poses, robot_radius=0.5)
    # Laterally far from the trajectory
    out = proc.process(_sample([[step, 10.0, 0.0]]))
    assert out[0] == 0


def test_height_filter_too_high():
    step = 2.0
    poses = _straight_line_poses(5, step)
    proc = TraversabilityFromTrajectory(poses, robot_radius=1.0, height_min=-0.3, height_max=0.5)
    # Within XY radius but above height_max
    out = proc.process(_sample([[step, 0.0, 1.0]]))
    assert out[0] == 0


def test_height_filter_too_low():
    step = 2.0
    poses = _straight_line_poses(5, step)
    proc = TraversabilityFromTrajectory(poses, robot_radius=1.0, height_min=-0.3, height_max=0.5)
    out = proc.process(_sample([[step, 0.0, -0.5]]))
    assert out[0] == 0


def test_height_filter_valid():
    step = 2.0
    poses = _straight_line_poses(5, step)
    proc = TraversabilityFromTrajectory(poses, robot_radius=1.0, height_min=-0.3, height_max=0.5)
    out = proc.process(_sample([[step, 0.0, 0.2]]))
    assert out[0] == 1


def test_last_frame_returns_all_zeros():
    n = 4
    poses = _straight_line_poses(n)
    proc = TraversabilityFromTrajectory(poses)
    pc = np.zeros((5, 3), dtype=np.float32)
    sample = _sample(pc)
    for _ in range(n - 1):
        proc.process(sample)
    out = proc.process(sample)
    np.testing.assert_array_equal(out, np.zeros(5, dtype=np.uint8))


def test_poses_are_consumed_in_order():
    # Two frames: robot goes from (0,0,0) to (2,0,0).
    # Frame 0 should see future pose at (2,0,0) as traversable.
    # Frame 1 has no future poses → all zeros.
    step = 2.0
    poses = _straight_line_poses(2, step)
    proc = TraversabilityFromTrajectory(poses, robot_radius=0.5, height_min=-0.5, height_max=0.5)

    out0 = proc.process(_sample([[step, 0.0, 0.0]]))
    out1 = proc.process(_sample([[step, 0.0, 0.0]]))

    assert out0[0] == 1   # future trajectory covers this point
    assert out1[0] == 0   # no future poses left


def test_forward_window_limits_look_ahead():
    # With a window of 1, only the immediately next pose is considered.
    step = 2.0
    n = 5
    poses = _straight_line_poses(n, step)
    proc = TraversabilityFromTrajectory(poses, robot_radius=0.5, forward_window=1)
    # A point at 3*step should NOT be traversable (out of window)
    out = proc.process(_sample([[3 * step, 0.0, 0.0]]))
    assert out[0] == 0


def test_sequence_gap_prevents_cross_sequence_look_ahead():
    # Two sequences: frames 0-2 at x=0,2,4 and frames 3-5 at x=100,102,104.
    # The jump of 96 m triggers a sequence boundary.
    # Frame 2 (end of seq 1) should have no future look-ahead within its sequence.
    step = 2.0
    poses = np.tile(np.eye(4, dtype=np.float64), (6, 1, 1))
    poses[0, 0, 3] = 0.0
    poses[1, 0, 3] = step
    poses[2, 0, 3] = step * 2
    poses[3, 0, 3] = 100.0
    poses[4, 0, 3] = 100.0 + step
    poses[5, 0, 3] = 100.0 + step * 2

    proc = TraversabilityFromTrajectory(poses, robot_radius=0.5, sequence_gap=5.0)
    # consume frames 0 and 1
    proc.process(_sample([[0.0, 0.0, 0.0]]))
    proc.process(_sample([[0.0, 0.0, 0.0]]))
    # frame 2: last in seq 1, a point at x=100 (seq 2) must NOT be traversable
    out = proc.process(_sample([[100.0, 0.0, 0.0]]))
    assert out[0] == 0
