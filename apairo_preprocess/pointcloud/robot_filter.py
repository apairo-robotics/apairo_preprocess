"""Filtering of LiDAR points that originate from the robot's own body.

When the sensor is mounted on the robot, some returns will hit the robot
chassis.  This preprocessor removes those self-hits by masking out points
that fall inside a geometric approximation of the robot shape (axis-aligned
box or upright cylinder), expressed in the sensor frame.

Typical usage::

    dataset.run_preprocess(
        RemoveRobotPoints(
            shape="box",
            x_range=(-0.6, 0.6),
            y_range=(-0.4, 0.4),
            z_range=(-0.5, 0.2),
        ),
        split_dir,
    )
"""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample

_Shape = Literal["box", "cylinder"]


class RemoveRobotPoints(FramePreprocessor):
    """Remove scan points that lie inside the robot's body volume.

    Points whose coordinates (in the sensor frame) fall inside the given
    geometric shape are discarded.  All other points and their extra channels
    (intensity, ring, …) are preserved unchanged.

    Args:
        lidar_key:  Input channel for point cloud data.
        shape:      Approximation geometry for the robot body:

                    * ``"box"``      — axis-aligned bounding box defined by
                      ``x_range``, ``y_range``, and ``z_range``.
                    * ``"cylinder"`` — upright cylinder defined by ``radius``
                      and ``z_range``.

        x_range:    ``(min, max)`` extents along X (metres). Box only.
        y_range:    ``(min, max)`` extents along Y (metres). Box only.
        z_range:    ``(min, max)`` extents along Z (metres). Both shapes.
        radius:     Radius of the cylinder in the XY plane (metres).
                    Cylinder only.
        output_key: Override the default output channel name
                    ``"lidar_no_robot"``.
    """

    output_key: ClassVar[str] = "lidar_no_robot"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["lidar"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["lidar"]

    def __init__(
        self,
        lidar_key: str = "lidar",
        shape: _Shape = "box",
        x_range: tuple[float, float] = (-1.0, 1.0),
        y_range: tuple[float, float] = (-1.0, 1.0),
        z_range: tuple[float, float] = (-1.0, 1.0),
        radius: float = 1.0,
        output_key: str | None = None,
    ) -> None:
        if shape not in ("box", "cylinder"):
            raise ValueError(f"shape must be 'box' or 'cylinder', got {shape!r}")
        if shape == "box":
            if x_range[0] >= x_range[1]:
                raise ValueError(f"x_range must satisfy min < max, got {x_range}")
            if y_range[0] >= y_range[1]:
                raise ValueError(f"y_range must satisfy min < max, got {y_range}")
        if z_range[0] >= z_range[1]:
            raise ValueError(f"z_range must satisfy min < max, got {z_range}")
        if shape == "cylinder" and radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")

        self._lidar_key = lidar_key
        self._shape = shape
        self._x_range = x_range
        self._y_range = y_range
        self._z_range = z_range
        self._radius = float(radius)

        self.input_keys = [lidar_key]
        self.sources = [lidar_key]
        if output_key is not None:
            self.output_key = output_key

    def process(self, sample: Sample) -> np.ndarray:
        pc = np.asarray(sample.data[self._lidar_key], dtype=np.float64)
        if pc.ndim != 2 or pc.shape[1] < 3:
            raise ValueError(f"Point cloud must be (N, D>=3), got {pc.shape}")

        xyz = pc[:, :3]

        if self._shape == "box":
            inside = (
                (xyz[:, 0] >= self._x_range[0])
                & (xyz[:, 0] <= self._x_range[1])
                & (xyz[:, 1] >= self._y_range[0])
                & (xyz[:, 1] <= self._y_range[1])
                & (xyz[:, 2] >= self._z_range[0])
                & (xyz[:, 2] <= self._z_range[1])
            )
        else:
            dist_xy = np.linalg.norm(xyz[:, :2], axis=1)
            inside = (
                (dist_xy <= self._radius)
                & (xyz[:, 2] >= self._z_range[0])
                & (xyz[:, 2] <= self._z_range[1])
            )

        return pc[~inside]
