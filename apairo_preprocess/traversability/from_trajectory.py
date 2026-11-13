"""Trajectory-based traversability ground truth.

For each scan, a point is labelled traversable (1) if it lies within the
robot's 2-D footprint projected along any **future** pose — i.e. positions
the robot has not yet visited.

Poses are read from the dataset via ``poses_key`` (declared in
``input_keys``).  The full trajectory is available in ``__call__`` because
:class:`SequencePreprocessor` receives an iterator over all frames at once.

Poses must be ``(4, 4)`` float64 homogeneous matrices.  Apply
``ds = ds.transform(poses_key, PoseTo4x4())`` from ``apairo_transform`` beforehand
if your poses are stored in another format (quaternion, Euler, compact 3×4).

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

class _OrderNorm:
    """``(N, 3) -> (N,)`` norm of a given order (``2`` a sphere, ``np.inf`` a
    cube): the callable :class:`~apairo_transform.RangeFilter` takes."""

    def __init__(self, order: float) -> None:
        self.order = float(order)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points, ord=self.order, axis=1)

    def __repr__(self) -> str:
        return f"_OrderNorm({self.order})"


class TraversabilityFromTrajectory(SequencePreprocessor):
    """Label each point traversable if it lies in the robot's forward footprint.

    Poses and point clouds are loaded from the dataset via ``input_keys``.
    Sequence boundary detection and look-ahead are computed in ``__call__``
    where the full trajectory is available.

    Args:
        lidar_key:             Input channel for point cloud data.
        poses_key:             Input channel for per-frame poses — must be ``(4, 4)``
                               float64.  Apply ``PoseTo4x4()`` from ``apairo_transform``
                               if needed.
        robot_radius:          Half-width of the robot footprint in XY (metres).
                               Isotropic circle test, ignored when ``robot_length`` /
                               ``robot_width`` are given.
        robot_length:          Full length of the robot footprint along its heading
                               (metres). Must be given together with ``robot_width``.
                               When set, the footprint is a heading-aligned rectangle
                               (matching the physical robot, e.g. its wheel-contact
                               box) instead of ``robot_radius``'s isotropic circle —
                               use this whenever the robot is not roughly as wide as
                               it is long, since a circle sized to the robot's length
                               over-labels its width (and vice versa). Heading is read
                               from each future pose's own rotation matrix, not
                               finite-differenced from consecutive positions, so it
                               stays accurate even where the robot is nearly
                               stationary (position noise dominates finite differences
                               there).
        robot_width:           Full width of the robot footprint, lateral to its
                               heading (metres). See ``robot_length``.
        height_min:            Lowest point height, relative to the ground under a
                               future pose, to be traversable (metres, ≤ 0).
        height_max:            Highest point height relative to that ground
                               (metres, ≥ 0).
        pose_height:           Height of the pose frame's origin above the ground
                               (metres). ``0`` (default) for poses given at
                               ground level. With lidar odometry (KISS-ICP, GICP)
                               the pose is the sensor's: pass the lidar's
                               mounting height, or every ground point lies below
                               the window and no point is ever traversable.
        forward_window:        Maximum number of future poses to look ahead.
                               ``None`` (default) uses the entire remaining trajectory.
        forward_distance:      Maximum *path length* (metres) of future trajectory
                               to consider.  Caps the look-ahead where the path
                               loops back near already-seen terrain: without it, a
                               point can be labelled traversable because a much
                               later part of the trajectory passes nearby.
                               ``None`` (default) does not cap by distance.
        sequence_gap:          Distance threshold (metres) to detect sequence
                               boundaries and avoid look-ahead across discontinuous
                               sessions.
        near_exclusion_radius: Sensor-frame distance below which a point is never
                               labelled traversable, suppressing dynamic objects
                               (e.g. humans) close to the robot.  The shape of the
                               exclusion zone is controlled by ``near_exclusion_norm``.
                               ``0.0`` disables the exclusion (default).
        near_exclusion_norm:   Norm order for the exclusion distance: ``np.inf``
                               (L∞) for a cube, ``2`` for a sphere. Defaults to
                               ``np.inf``. A callable ``(N, 3) -> (N,)`` is
                               accepted too.
        output_key:            Override the default output channel name ``"trav_traj"``.
        valid_key:             Channel holding one bool per scan: whether its pose
                               can be trusted (e.g. ``reference_poses``' second
                               output). An invalid scan cuts the trajectory as
                               ``sequence_gap`` does -- no look-ahead across it --
                               and none of its points is traversable. Without it,
                               a non-finite pose is an error.
        unknown_label:         Label of every point of an invalid scan, e.g. 255
                               for "unknown". ``None`` (default) labels them 0,
                               which a consumer cannot tell from "not
                               traversable".
    """

    output_key: ClassVar[str] = "trav_traj"
    input_keys: ClassVar[list[str]] = ["lidar", "poses"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["lidar", "poses"]

    def __init__(
        self,
        lidar_key: str = "lidar",
        poses_key: str = "poses",
        robot_radius: float = 0.75,
        robot_length: float | None = None,
        robot_width: float | None = None,
        height_min: float = -0.3,
        height_max: float = 0.5,
        forward_window: int | None = None,
        forward_distance: float | None = None,
        sequence_gap: float = 5.0,
        near_exclusion_radius: float = 0.0,
        near_exclusion_norm: float = np.inf,
        output_key: str | None = None,
        pose_height: float = 0.0,
        valid_key: str | None = None,
        unknown_label: int | None = None,
    ) -> None:
        if (robot_length is None) != (robot_width is None):
            raise ValueError("robot_length and robot_width must be given together")

        self._lidar_key = lidar_key
        self._poses_key = poses_key
        self._robot_radius = robot_radius
        self._robot_length = robot_length
        self._robot_width = robot_width
        self._oriented = robot_length is not None
        # Heights are tested against the pose origin: the window is shifted
        # down by the origin's height above the ground.
        self.pose_height = float(pose_height)
        self._height_min = height_min - self.pose_height
        self._height_max = height_max - self.pose_height
        self._forward_window = forward_window
        self._forward_distance = forward_distance
        self._sequence_gap = sequence_gap
        self._near_filter = (
            RangeFilter(
                min=near_exclusion_radius,
                max=None,
                norm=(
                    near_exclusion_norm
                    if callable(near_exclusion_norm)
                    else _OrderNorm(near_exclusion_norm)
                ),
            )
            if near_exclusion_radius > 0
            else None
        )

        if unknown_label is not None and not 0 <= int(unknown_label) <= 255:
            raise ValueError(f"unknown_label must fit in uint8, got {unknown_label}")
        if unknown_label is not None and int(unknown_label) in (0, 1):
            raise ValueError("unknown_label must differ from the labels 0 and 1")
        self._valid_key = valid_key
        self._unknown = None if unknown_label is None else int(unknown_label)
        keys = [lidar_key, poses_key] + ([valid_key] if valid_key is not None else [])
        self.input_keys = keys
        self.sources = list(keys)
        if output_key is not None:
            self.output_key = output_key

    def __call__(self, frames: Iterator[Sample]) -> np.ndarray:
        all_samples = list(frames)
        n = len(all_samples)
        if n == 0:
            return np.empty(0, dtype=object)

        poses = np.asarray(
            np.stack([s.data[self._poses_key] for s in all_samples]), dtype=np.float64
        )
        if poses.shape[1:] != (4, 4):
            raise ValueError(
                f"poses must be (N, 4, 4) — apply PoseTo4x4() from apairo_transform first. "
                f"Got {poses.shape}"
            )

        pose_finite = np.all(np.isfinite(poses.reshape(n, -1)), axis=1)
        if self._valid_key is None:
            if not pose_finite.all():
                raise ValueError(
                    f"{int((~pose_finite).sum())} pose(s) are not finite. Pass the "
                    f"validity of each pose as valid_key (reference_poses gives "
                    f"it), so that those scans are left out."
                )
            valid = pose_finite
        else:
            valid = np.array(
                [bool(np.asarray(s.data[self._valid_key])) for s in all_samples]
            ) & pose_finite

        # Sequence boundary detection — done here where poses are available.
        # A step is broken by a gap larger than sequence_gap or by an invalid
        # scan at either end; NaN distances (invalid poses) are broken too.
        positions = poses[:, :3, 3]
        dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        broken = ~(dists <= self._sequence_gap) | ~valid[:-1] | ~valid[1:]
        boundaries = np.where(broken)[0]
        ends = np.concatenate([boundaries, [n - 1]])
        seq_end = ends[np.searchsorted(ends, np.arange(n))]

        # Cumulative path length, for the forward_distance cap. A broken step
        # adds nothing: no look-ahead crosses it.
        path = np.concatenate([[0.0], np.cumsum(np.where(broken, 0.0, dists))])

        if self._oriented:
            # Yaw from each pose's own rotation matrix -- robust where the robot is
            # nearly stationary (finite-differenced heading is noise-dominated there).
            headings = np.arctan2(poses[:, 1, 0], poses[:, 0, 0])
            half_length = self._robot_length / 2.0
            half_width = self._robot_width / 2.0
            covering_radius = float(np.hypot(half_length, half_width))

        results = []
        for idx, sample in enumerate(all_samples):
            pc = np.asarray(sample.data[self._lidar_key])
            if not valid[idx]:
                fill = 0 if self._unknown is None else self._unknown
                results.append(np.full(len(pc), fill, dtype=np.uint8))
                continue
            xyz_all = pc[:, :3].astype(np.float64)
            # A return without a position (NaN or inf) is never traversable; it
            # used to abort the whole sequence in the nearest-pose search.
            finite = np.all(np.isfinite(xyz_all), axis=1)
            xyz_sensor = xyz_all[finite]
            n_pts = len(xyz_sensor)

            T = poses[idx]
            xyz_h = np.column_stack([xyz_sensor, np.ones(n_pts)])
            xyz_world = (T @ xyz_h.T).T[:, :3]

            end = int(seq_end[idx]) + 1
            if self._forward_window is not None:
                end = min(end, idx + 1 + self._forward_window)
            if self._forward_distance is not None:
                end = min(
                    end,
                    int(np.searchsorted(path, path[idx] + self._forward_distance, side="right")),
                )
            future_pos = poses[idx + 1 : end, :3, 3]

            if len(future_pos) == 0:
                results.append(np.zeros(len(xyz_all), dtype=np.uint8))
                continue

            # Every future pose whose footprint may hold the point is tested, in
            # both modes: a path that passes over a point at its own height and
            # crosses the same spot again higher up (a ramp, a bridge, vertical
            # drift on a loop) still drove over it.
            reach = covering_radius if self._oriented else float(self._robot_radius)
            tree = KDTree(future_pos[:, :2])
            # workers=-1 deadlocks when called from a ThreadPoolExecutor (e.g. viewer)
            cand_lists = tree.query_ball_point(xyz_world[:, :2], r=reach)
            lengths = np.fromiter((len(c) for c in cand_lists), dtype=np.int64, count=n_pts)
            trav = np.zeros(n_pts, dtype=bool)
            n_pairs = int(lengths.sum())
            if n_pairs > 0:
                point_idx = np.repeat(np.arange(n_pts), lengths)
                cand_idx = np.fromiter(
                    (c for cl in cand_lists for c in cl), dtype=np.int64, count=n_pairs
                )
                ddx = xyz_world[point_idx, 0] - future_pos[cand_idx, 0]
                ddy = xyz_world[point_idx, 1] - future_pos[cand_idx, 1]
                dz_pair = xyz_world[point_idx, 2] - future_pos[cand_idx, 2]
                if self._oriented:
                    th = headings[idx + 1 : end][cand_idx]
                    c, s = np.cos(th), np.sin(th)
                    # world delta -> candidate pose's local frame (rotate by -heading)
                    dx = c * ddx + s * ddy
                    dy = -s * ddx + c * ddy
                    footprint = (np.abs(dx) <= half_length) & (np.abs(dy) <= half_width)
                else:
                    footprint = np.hypot(ddx, ddy) < self._robot_radius
                inside = (
                    footprint
                    & (dz_pair >= self._height_min)
                    & (dz_pair <= self._height_max)
                )
                np.logical_or.at(trav, point_idx[inside], True)
            if self._near_filter is not None:
                trav &= self._near_filter.compute_mask(xyz_sensor)

            labels = np.zeros(len(xyz_all), dtype=np.uint8)
            labels[finite] = trav
            results.append(labels)

        # Uniform scans stack to (N, P) as before; ragged scans (real lidar,
        # variable return count) become an object array of per-frame rows,
        # which the per-frame runner and direct callers iterate identically.
        if len({len(r) for r in results}) <= 1:
            return np.stack(results)
        out = np.empty(n, dtype=object)
        out[:] = results
        return out
