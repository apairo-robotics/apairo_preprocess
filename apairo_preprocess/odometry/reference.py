"""Poses delivered with a dataset, on the lidar clock.

An INS, a GNSS-INS, or the trajectory a dataset ships with is a recorded
measurement: turning it into one ``(4, 4)`` pose per scan is interpolation and
composition, with no parameter that changes from sensor to sensor and no way
to diverge. That is the point of this module, as opposed to
:class:`~apairo_preprocess.KissICPOdometry` and
:class:`~apairo_preprocess.GICPOdometry`, which *estimate* a trajectory and
return a pose even where they have lost track.

The rules are fixed and checkable by hand:

- a scan gets a pose only when two pose rows bracket its time, at most
  ``max_gap`` seconds apart; nothing is ever extrapolated;
- a step between two consecutive scans faster than ``max_speed`` (or turning
  faster than ``max_yaw_rate``) marks both of its scans invalid, since which
  one is wrong cannot be told;
- an invalid scan's pose is NaN, never a pose made up from its neighbours.

The pose rows and the scans are on two clocks, which apairo's runner does not
pair for a preprocessor yet: :func:`reference_poses` is therefore called
directly, and its two outputs are written as channels with
``apairo.ChannelWriter``.
"""

from __future__ import annotations

import numpy as np
from apairo_transform import PoseTo4x4
from apairo_transform.interp import Se3Interp

__all__ = ["ReferencePoses", "pose_validity", "reference_poses"]


def _increasing(rows: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows that repeat the previous timestamp (keeping the first); refuse
    a clock that goes backwards."""
    step = np.diff(times)
    if np.any(step < 0):
        k = int(np.flatnonzero(step < 0)[0])
        raise ValueError(
            f"pose timestamps go backwards at row {k + 1} "
            f"({times[k + 1]!r} after {times[k]!r}); sort them first."
        )
    keep = np.concatenate([[True], step > 0])
    return rows[keep], times[keep]


def _yaw(poses: np.ndarray) -> np.ndarray:
    return np.arctan2(poses[:, 1, 0], poses[:, 0, 0])


def pose_validity(
    poses: np.ndarray,
    times: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    max_speed: float,
    max_yaw_rate: float | None = None,
) -> np.ndarray:
    """``(N,)`` bool: the scans whose pose passes the physical bounds.

    Args:
        poses: ``(N, 4, 4)`` poses, one per scan, NaN where there is none.
        times: ``(N,)`` scan times, seconds.
        valid: Scans already known valid (default: those with a finite pose).
        max_speed: The robot's top speed, m/s. No default: it is a property of
            the robot, not of the data.
        max_yaw_rate: Optional bound on the turn rate, rad/s.

    Consecutive *valid* scans are compared. A step that exceeds a bound marks
    both of its scans invalid. A step with no elapsed time (a scan repeated
    under the same stamp, as some recordings do) has no speed and is not
    checked; a time going backwards is an error.
    """
    poses = np.asarray(poses, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if max_speed <= 0:
        raise ValueError(f"max_speed must be positive, got {max_speed}")
    finite = np.all(np.isfinite(poses.reshape(len(poses), -1)), axis=1)
    out = finite if valid is None else (np.asarray(valid, dtype=bool) & finite)
    idx = np.flatnonzero(out)
    if len(idx) < 2:
        return out
    dt = np.diff(times[idx])
    if np.any(dt < 0):
        raise ValueError("scan times go backwards; sort them first")
    timed = dt > 0
    safe = np.where(timed, dt, 1.0)
    dist = np.linalg.norm(np.diff(poses[idx, :3, 3], axis=0), axis=1)
    bad = timed & (dist / safe > max_speed)
    if max_yaw_rate is not None:
        dyaw = np.abs(np.angle(np.exp(1j * np.diff(_yaw(poses[idx])))))
        bad |= timed & (dyaw / safe > max_yaw_rate)
    out = out.copy()
    out[idx[:-1][bad]] = False
    out[idx[1:][bad]] = False
    return out


def reference_poses(
    rows: np.ndarray,
    row_times: np.ndarray,
    scan_times: np.ndarray,
    *,
    clock_offset: float = 0.0,
    extrinsic: np.ndarray | None = None,
    max_gap: float = 0.1,
    max_speed: float | None = None,
    max_yaw_rate: float | None = None,
    relative: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """One pose per scan from delivered pose rows.

    Args:
        rows: ``(M, >=7)`` rows ``[x, y, z, qx, qy, qz, qw, ...]``: the body's
            pose in the world frame. Extra columns (a twist) are ignored.
        row_times: ``(M,)`` row times, seconds. A row repeating the previous
            time is dropped; a time going backwards is an error.
        scan_times: ``(N,)`` scan times, seconds.
        clock_offset: Added to the scan times before looking the poses up:
            the pose source's clock minus the lidar's, as measured once for
            the rig.
        extrinsic: ``(4, 4)`` pose of the sensor in the body frame, composed on
            the right (``T_world_sensor = T_world_body @ extrinsic``). ``None``
            when the sensor is the body frame.
        max_gap: Largest time between the two rows bracketing a scan, seconds.
        max_speed: If given, :func:`pose_validity` is applied with it.
        max_yaw_rate: Passed to :func:`pose_validity`.
        relative: Express every pose relative to the first valid one, in
            float64. Positions recorded in a global frame (UTM: ~1e6 m) keep
            their millimetres only this way once a consumer casts to float32.

    Returns:
        ``(poses, valid)``: ``(N, 4, 4)`` float64, NaN where ``valid`` is False,
        and ``(N,)`` bool.
    """
    rows = np.asarray(rows, dtype=np.float64)
    row_times = np.asarray(row_times, dtype=np.float64)
    scan_times = np.asarray(scan_times, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 7:
        raise ValueError(
            f"rows must be (M, >=7) [x y z qx qy qz qw ...], got {rows.shape}"
        )
    if len(rows) != len(row_times):
        raise ValueError(f"{len(rows)} rows but {len(row_times)} row times")
    if max_gap <= 0:
        raise ValueError(f"max_gap must be positive, got {max_gap}")
    rows, row_times = _increasing(rows[:, :7], row_times)

    n = len(scan_times)
    poses = np.full((n, 4, 4), np.nan)
    valid = np.zeros(n, dtype=bool)
    if len(rows) == 0 or n == 0:
        return poses, valid

    tq = scan_times + clock_offset
    hi = np.searchsorted(row_times, tq, side="left")
    exact = (hi < len(row_times)) & (
        row_times[np.minimum(hi, len(row_times) - 1)] == tq
    )
    interp = Se3Interp()
    to4 = PoseTo4x4()
    for i in range(n):
        if exact[i]:
            pose7 = rows[hi[i]]
        else:
            j = int(hi[i])
            if j == 0 or j == len(row_times):
                continue  # outside the recorded span: no extrapolation
            t0, t1 = row_times[j - 1], row_times[j]
            if t1 - t0 > max_gap:
                continue
            pose7 = interp(tq[i], t0, rows[j - 1], t1, rows[j])
        poses[i] = to4(pose7)
        valid[i] = True

    if extrinsic is not None:
        extrinsic = np.asarray(extrinsic, dtype=np.float64)
        if extrinsic.shape != (4, 4):
            raise ValueError(f"extrinsic must be (4, 4), got {extrinsic.shape}")
        poses[valid] = poses[valid] @ extrinsic
    if max_speed is not None:
        valid = pose_validity(
            poses, scan_times, valid, max_speed=max_speed, max_yaw_rate=max_yaw_rate
        )
        poses[~valid] = np.nan
    if relative and valid.any():
        first = poses[int(np.flatnonzero(valid)[0])]
        poses[valid] = np.linalg.inv(first) @ poses[valid]
    return poses, valid


class ReferencePoses:
    """:func:`reference_poses` with its parameters fixed once for a rig.

    Example::

        ins = ReferencePoses(clock_offset=-0.05, max_gap=0.1, max_speed=20.0)
        poses, valid = ins(rows, row_times, scan_times)
        # then write both with apairo.ChannelWriter, on the lidar clock
    """

    def __init__(
        self,
        *,
        clock_offset: float = 0.0,
        extrinsic: np.ndarray | None = None,
        max_gap: float = 0.1,
        max_speed: float | None = None,
        max_yaw_rate: float | None = None,
        relative: bool = True,
    ) -> None:
        self.clock_offset = float(clock_offset)
        self.extrinsic = (
            None if extrinsic is None else np.asarray(extrinsic, dtype=np.float64)
        )
        self.max_gap = float(max_gap)
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.relative = relative

    def __call__(
        self, rows: np.ndarray, row_times: np.ndarray, scan_times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return reference_poses(
            rows,
            row_times,
            scan_times,
            clock_offset=self.clock_offset,
            extrinsic=self.extrinsic,
            max_gap=self.max_gap,
            max_speed=self.max_speed,
            max_yaw_rate=self.max_yaw_rate,
            relative=self.relative,
        )

    def get_config(self) -> dict:
        return {
            "clock_offset": self.clock_offset,
            "extrinsic": None if self.extrinsic is None else self.extrinsic.tolist(),
            "max_gap": self.max_gap,
            "max_speed": self.max_speed,
            "max_yaw_rate": self.max_yaw_rate,
            "relative": self.relative,
        }

    def __repr__(self) -> str:
        return (
            f"ReferencePoses(clock_offset={self.clock_offset}, max_gap={self.max_gap}, "
            f"max_speed={self.max_speed})"
        )
