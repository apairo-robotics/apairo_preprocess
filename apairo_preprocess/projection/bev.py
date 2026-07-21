"""Rasterise per-point values into a top-down (bird's-eye-view) grid.

Unlike :class:`LidarCameraProjection` + :class:`ImageMaskFromPointLabels`
(perspective projection into a camera, nearest-depth-wins occlusion), a BEV
grid needs no camera at all: points are binned by their XY position onto an
orthographic top-down grid, so it works on any lidar scan regardless of
whether a calibrated camera exists for it -- useful for datasets that ship
point clouds without a registered camera/lidar extrinsic.

Row 0 of the output is north (``ymax``), so plotting the grid directly reads
top-down.

Typical usage::

    TartanKittiDataset.run_preprocess(
        BEVRasterisation(bounds=(-20, 20, -20, 20), resolution=0.2,
                          points_key="lidar", values_key="trav_traj"),
        seq_dir,
    )
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample


def _rasterise(points, values, *, bounds, resolution, reduce, background):
    xmin, xmax, ymin, ymax = bounds
    w = int(round((xmax - xmin) / resolution))
    h = int(round((ymax - ymin) / resolution))

    xy = np.asarray(points)[:, :2]
    vals = np.asarray(values, dtype=np.float64)

    col = np.floor((xy[:, 0] - xmin) / resolution).astype(np.int64)
    row = np.floor((ymax - xy[:, 1]) / resolution).astype(np.int64)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    cell = row[inside] * w + col[inside]
    vals = vals[inside]

    grid = np.full(h * w, background, dtype=np.float64)
    if reduce == "max":
        # A plain np.maximum.at into `grid` would use `background` as an
        # implicit floor -- wrong whenever real values can be below it (e.g.
        # negative heights). Accumulate from -inf, then fill only the cells
        # no point touched.
        acc = np.full(h * w, -np.inf)
        np.maximum.at(acc, cell, vals)
        hit = np.isfinite(acc)
        grid[hit] = acc[hit]
    elif reduce == "last":
        grid[cell] = vals
    else:  # "mean"
        count = np.zeros(h * w)
        total = np.zeros(h * w)
        np.add.at(count, cell, 1.0)
        np.add.at(total, cell, vals)
        hit = count > 0
        grid[hit] = total[hit] / count[hit]
    return grid.reshape(h, w)


class BEVRasterisation(FramePreprocessor):
    """Per-point values splatted onto a top-down ``(H, W)`` grid.

    Output is float64, row-major with row 0 at ``ymax`` (north up) and
    column 0 at ``xmin``. Cells no point falls into hold ``background``.

    Args:
        bounds:      ``(xmin, xmax, ymin, ymax)`` in metres, in the point
                     cloud's own frame (typically sensor- or robot-centred).
        resolution:  Metres per pixel.
        points_key:  Input channel for the point cloud, ``(N, >=2)``; only
                     the X/Y columns are used.
        values_key:  Input channel for the per-point scalar to rasterise
                     (e.g. height, a traversability label), ``(N,)``,
                     row-aligned with ``points_key``.
        reduce:      How to combine points landing in the same cell --
                     ``"max"`` (default), ``"mean"`` or ``"last"``.
        background:  Value for cells no point falls into.
        output_key:  Override the default output channel name ``"bev"``.
    """

    output_key: ClassVar[str] = "bev"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["lidar", "values"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["lidar", "values"]

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        resolution: float,
        points_key: str = "lidar",
        values_key: str = "values",
        reduce: str = "max",
        background: float = 0.0,
        output_key: str | None = None,
    ) -> None:
        xmin, xmax, ymin, ymax = bounds
        if not (xmax > xmin and ymax > ymin):
            raise ValueError(f"bounds must have xmax > xmin and ymax > ymin, got {bounds}")
        if resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {resolution}")
        if reduce not in ("max", "mean", "last"):
            raise ValueError(f"reduce must be 'max', 'mean' or 'last', got {reduce!r}")

        self._bounds = (float(xmin), float(xmax), float(ymin), float(ymax))
        self._resolution = float(resolution)
        self._reduce = reduce
        self._background = float(background)
        self._points_key = points_key
        self._values_key = values_key

        self.input_keys = [points_key, values_key]
        self.sources = [points_key, values_key]
        self.timestamps_from = points_key
        if output_key is not None:
            self.output_key = output_key

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` of the output grid, from ``bounds``/``resolution``."""
        xmin, xmax, ymin, ymax = self._bounds
        w = int(round((xmax - xmin) / self._resolution))
        h = int(round((ymax - ymin) / self._resolution))
        return h, w

    def __call__(self, sample: Sample) -> np.ndarray:
        points = np.asarray(sample.data[self._points_key])
        values = np.asarray(sample.data[self._values_key])
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError(f"points must be (N, >=2), got {points.shape}")
        if len(values) != len(points):
            raise ValueError(
                f"points ({len(points)}) and values ({len(values)}) are not "
                f"row-aligned -- they must derive from the same scan."
            )
        return _rasterise(
            points, values,
            bounds=self._bounds, resolution=self._resolution,
            reduce=self._reduce, background=self._background,
        )


def to_uint8_image(
    grid: np.ndarray, *, vmin: float | None = None, vmax: float | None = None
) -> np.ndarray:
    """Normalise a ``(H, W)`` grid to an ``(H, W, 3)`` uint8 image.

    Pass fixed ``vmin``/``vmax`` for anything beyond one-off visualisation:
    left at ``None``, each call normalises to its *own* min/max, so the same
    physical value maps to a different pixel intensity from one frame to the
    next -- fine to eyeball a single grid, wrong for a dataset a model trains
    on (a given value must mean the same thing in every frame).
    """
    grid = np.asarray(grid, dtype=np.float64)
    lo = grid.min() if vmin is None else vmin
    hi = grid.max() if vmax is None else vmax
    norm = (grid - lo) / (hi - lo) if hi > lo else np.zeros_like(grid)
    gray = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)
