"""Trajectory-based traversability ground truth.

For each scan, a point is labelled traversable (1) if it lies within the
robot's 2-D footprint projected along any **future** pose — i.e. positions
the robot has not yet visited.

Poses are read from the dataset via ``poses_key`` (declared in
``input_keys``).  The full trajectory is available in ``process()`` because
:class:`SequencePreprocessor` receives an iterator over all frames at once.

Accepted pose formats (per frame):
  - ``(4, 4)`` float64 — standard homogeneous transform T_world_sensor
  - ``(3, 4)`` float64 — compact form, homogeneous row appended automatically

Typical usage::

    Goose3DDataset.run_preprocess(
        TraversabilityFromTrajectory(poses_key="kissicp_poses", robot_radius=0.75),
        split_dir,
    )
"""

from __future__ import annotations

from typing import ClassVar, Iterator

import numpy as np
from scipy.spatial import KDTree

from apairo.core.preprocessor import SequencePreprocessor
from apairo.core.sample import Sample
from apairo_transform import RangeFilter


def _to_4x4(poses: np.ndarray) -> np.ndarray:
    """Normalise a pose array to (N, 4, 4) float64.

    Accepts (N, 4, 4) or (N, 3, 4); the compact form has the homogeneous row
    ``[0, 0, 0, 1]`` appended automatically.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim == 3 and poses.shape[1:] == (3, 4):
        n = poses.shape[0]
        bottom = np.zeros((n, 1, 4), dtype=np.float64)
        bottom[:, 0, 3] = 1.0
        poses = np.concatenate([poses, bottom], axis=1)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(
            f"poses must be (N, 4, 4) or (N, 3, 4), got {poses.shape}"
        )
    return poses


class TraversabilityFromTrajectory(SequencePreprocessor):
    """Label each point traversable if it lies in the robot's forward footprint.

    Poses and point clouds are loaded from the dataset via ``input_keys``.
    Sequence boundary detection and look-ahead are computed in ``process()``
    where the full trajectory is available.

    Args:
        lidar_key:             Input channel for point cloud data.
        poses_key:             Input channel for per-frame poses (4x4 or 3x4).
        robot_radius:          Half-width of the robot footprint in XY (metres).
        height_min:            Minimum point height relative to the nearest robot
                               position to be traversable (metres, ≤ 0).
        height_max:            Maximum point height relative to the nearest robot
                               position (metres, ≥ 0).
        forward_window:        Maximum number of future poses to look ahead.
                               ``None`` (default) uses the entire remaining trajectory.
        sequence_gap:          Distance threshold (metres) to detect sequence
                               boundaries and avoid look-ahead across discontinuous
                               sessions.
        near_exclusion_radius: Sensor-frame distance below which a point is never
                               labelled traversable, suppressing dynamic objects
                               (e.g. humans) close to the robot.  The shape of the
                               exclusion zone is controlled by ``near_exclusion_norm``.
                               ``0.0`` disables the exclusion (default).
        near_exclusion_norm:   Norm order for the exclusion distance, forwarded to
                               :class:`~apairo_transform.RangeFilter`.  Use
                               ``np.inf`` (L∞) for a cube, ``2`` for a sphere.
                               Defaults to ``np.inf``.
        output_key:            Override the default output channel name ``"trav_traj"``.
    """

    output_key: ClassVar[str] = "trav_traj"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["lidar", "poses"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["lidar", "poses"]

    def __init__(
        self,
        lidar_key: str = "lidar",
        poses_key: str = "poses",
        robot_radius: float = 0.75,
        height_min: float = -0.3,
        height_max: float = 0.5,
        forward_window: int | None = None,
        sequence_gap: float = 5.0,
        near_exclusion_radius: float = 0.0,
        near_exclusion_norm: float = np.inf,
        output_key: str | None = None,
    ) -> None:
        self._lidar_key = lidar_key
        self._poses_key = poses_key
        self._robot_radius = robot_radius
        self._height_min = height_min
        self._height_max = height_max
        self._forward_window = forward_window
        self._sequence_gap = sequence_gap
        self._near_filter = (
            RangeFilter(min=near_exclusion_radius, max=None, norm=near_exclusion_norm)
            if near_exclusion_radius > 0
            else None
        )

        self.input_keys = [lidar_key, poses_key]
        self.sources = [lidar_key, poses_key]
        if output_key is not None:
            self.output_key = output_key

    def process(self, frames: Iterator[Sample]) -> np.ndarray:
        all_samples = list(frames)
        n = len(all_samples)

        poses = _to_4x4(np.stack([s.data[self._poses_key] for s in all_samples]))

        # Sequence boundary detection — done here where poses are available.
        positions = poses[:, :3, 3]
        dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        boundaries = np.where(dists > self._sequence_gap)[0]
        ends = np.concatenate([boundaries, [n - 1]])
        seq_end = ends[np.searchsorted(ends, np.arange(n))]

        results = []
        for idx, sample in enumerate(all_samples):
            pc = np.asarray(sample.data[self._lidar_key])
            xyz_sensor = pc[:, :3].astype(np.float64)
            n_pts = len(xyz_sensor)

            T = poses[idx]
            xyz_h = np.column_stack([xyz_sensor, np.ones(n_pts)])
            xyz_world = (T @ xyz_h.T).T[:, :3]

            seq_end_idx = int(seq_end[idx]) + 1
            end = min(
                idx + 1 + self._forward_window
                if self._forward_window is not None
                else seq_end_idx,
                seq_end_idx,
            )
            future_pos = poses[idx + 1 : end, :3, 3]

            if len(future_pos) == 0:
                results.append(np.zeros(n_pts, dtype=np.uint8))
                continue

            tree = KDTree(future_pos[:, :2])
            # workers=-1 deadlocks when called from a ThreadPoolExecutor (e.g. viewer)
            dist_xy, nn_idx = tree.query(xyz_world[:, :2], k=1)
            dz = xyz_world[:, 2] - future_pos[nn_idx, 2]

            trav = (
                (dist_xy < self._robot_radius)
                & (dz >= self._height_min)
                & (dz <= self._height_max)
            )
            if self._near_filter is not None:
                trav &= self._near_filter.compute_mask(xyz_sensor)

            results.append(trav.astype(np.uint8))

        return np.stack(results)
