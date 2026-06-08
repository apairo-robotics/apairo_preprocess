"""Trajectory-distance preprocessor for Weakly Informed PUNCE.

Computes the Euclidean distance (metres) from each voxelised point to the
nearest waypoint on the robot's recorded trajectory.  Points that lie close
to the path the robot actually drove are likely traversable.

The output float32 array of shape (N,) is consumed by
:class:`WeaklyInformedPUNCELoss` to compute the Gaussian prior

    π(p) = exp(−d(p, traj)² / (2σ²))

All trajectory waypoints must be passed at construction time
(as world-frame positions) so the preprocessor can build a single KD-tree
shared across frames.

Typical usage in a preprocess script::

    ds = Rellis3DDataset(root, keys=["poses"]).transform("poses", PoseTo4x4())
    all_poses = np.stack([ds[i].data["poses"] for i in range(len(ds))])
    trajectory = all_poses[:, :3, 3]          # (N_frames, 3) world positions

    Rellis3DDataset.run_preprocess(
        TrajectoryDistance(trajectory=trajectory),
        root,
    )
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.spatial import KDTree

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample


def _to_4x4(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim == 1 and pose.shape[0] == 16:
        pose = pose.reshape(4, 4)
    if pose.shape == (3, 4):
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
        pose = np.vstack([pose, bottom])
    if pose.shape != (4, 4):
        raise ValueError(f"Pose must be (3, 4) or (4, 4), got {pose.shape}")
    return pose


class TrajectoryDistance(FramePreprocessor):
    """Per-voxel Euclidean distance to the nearest trajectory waypoint.

    For each frame, voxelised points are transformed to the world frame using
    the current pose, then queried against a KD-tree built from all trajectory
    waypoints.

    Args:
        trajectory:    (M, 3) array of robot positions in world frame.
                       Typically ``all_poses[:, :3, 3]`` from the sequence.
        voxelised_key: Input channel for the voxelised point cloud.
        poses_key:     Input channel for the current-frame pose (4x4 or 3x4).
        output_key:    Override the default channel name
                       ``"trajectory_distance"``.
    """

    output_key: ClassVar[str] = "trajectory_distance"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["voxelised", "poses"]
    timestamps_from: ClassVar[str] = "voxelised"
    sources: ClassVar[list[str]] = ["voxelised", "poses"]

    def __init__(
        self,
        trajectory: np.ndarray,
        voxelised_key: str = "voxelised",
        poses_key: str = "poses",
        output_key: str | None = None,
    ) -> None:
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] != 3:
            raise ValueError(
                f"trajectory must be (M, 3), got {traj.shape}"
            )
        self._tree = KDTree(traj)
        self._voxelised_key = voxelised_key
        self._poses_key = poses_key
        self.input_keys = [voxelised_key, poses_key]
        self.sources = [voxelised_key, poses_key]
        if output_key is not None:
            self.output_key = output_key

    def process(self, sample: Sample) -> np.ndarray:
        pc = np.asarray(sample.data[self._voxelised_key], dtype=np.float64)
        pose = _to_4x4(np.asarray(sample.data[self._poses_key]))

        n = len(pc)
        xyz_sensor = pc[:, :3]
        xyz_h = np.column_stack([xyz_sensor, np.ones(n)])
        xyz_world = (pose @ xyz_h.T).T[:, :3]

        dist, _ = self._tree.query(xyz_world)
        return dist.astype(np.float32)
