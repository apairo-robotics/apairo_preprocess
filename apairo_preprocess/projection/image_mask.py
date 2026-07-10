"""Rasterize per-point labels into an image mask — point cloud → image.

The forward direction of the projection composition: given any per-point
label channel (e.g. ``trav_traj`` from
:class:`~apairo_preprocess.TraversabilityFromTrajectory`) and the per-point
pixel coordinates written by :class:`LidarCameraProjection`, paint each
label at its point's pixel.  The result is a sparse image-space ground-truth
mask — the traversability channel of an image dataset.

Pixels hit by several points keep the **nearest** point's label (depth
ordering), and an optional occlusion filter drops points that are
significantly behind the nearest return in their neighbourhood, so
traversable ground behind an obstacle does not bleed onto the obstacle's
pixels.

This preprocessor reads two channels, both on the lidar clock.  On profiled
synchronous datasets it runs via ``run_preprocess`` directly; on
asynchronous datasets, run it over a ``synchronize()`` view and persist with
``ChannelWriter`` (see ``examples/traversability_image_mask.py``).

Typical usage::

    Rellis3DDataset.run_preprocess(
        ImageMaskFromPointLabels(image_size=(1200, 1920), radius=2),
        root,
    )
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample


class ImageMaskFromPointLabels(FramePreprocessor):
    """Paint per-point labels into an ``(H, W)`` uint8 mask.

    Output is ``(height, width)`` uint8: ``unknown_value`` where no point
    projects, the nearest projecting point's label elsewhere.  With the
    default inputs (``trav_traj``) that is ``255`` = no data, ``1`` =
    traversable, ``0`` = not traversable.

    Args:
        image_size:      ``(height, width)`` of the output mask — must match
                         the ``image_size`` the projection was computed for.
        labels_key:      Input channel with per-point integer labels ``(N,)``,
                         row-aligned with the projection.
        projection_key:  Input channel with ``(N, 3)`` ``[u, v, depth]`` rows
                         from :class:`LidarCameraProjection`.
        radius:          Splat radius in pixels.  ``0`` (default) paints one
                         pixel per point; ``r`` paints a disc of radius ``r``,
                         densifying the sparse lidar coverage.
        unknown_value:   Value for pixels no point projects to (default 255).
                         Labels must not collide with it.
        occlusion_bin_px:      Optional occlusion filtering: bin size in pixels
                               of a coarse depth buffer.  Points deeper than
                               the nearest return in their bin by more than
                               ``occlusion_depth_margin`` are dropped before
                               painting.  ``None`` (default) disables it.
        occlusion_depth_margin: Depth slack in metres for the occlusion filter.
        output_key:      Override the default output channel name
                         ``"trav_mask"``.
    """

    output_key: ClassVar[str] = "trav_mask"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["trav_traj", "lidar_uv"]
    timestamps_from: ClassVar[str] = "trav_traj"
    sources: ClassVar[list[str]] = ["trav_traj", "lidar_uv"]

    def __init__(
        self,
        image_size: tuple[int, int],
        labels_key: str = "trav_traj",
        projection_key: str = "lidar_uv",
        radius: int = 0,
        unknown_value: int = 255,
        occlusion_bin_px: int | None = None,
        occlusion_depth_margin: float = 0.5,
        output_key: str | None = None,
    ) -> None:
        height, width = image_size
        if height <= 0 or width <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if radius < 0:
            raise ValueError(f"radius must be >= 0, got {radius}")
        if not 0 <= unknown_value <= 255:
            raise ValueError(f"unknown_value must fit uint8, got {unknown_value}")
        if occlusion_bin_px is not None and occlusion_bin_px <= 0:
            raise ValueError(f"occlusion_bin_px must be > 0, got {occlusion_bin_px}")

        self._height = int(height)
        self._width = int(width)
        self._labels_key = labels_key
        self._projection_key = projection_key
        self._radius = int(radius)
        self._unknown = int(unknown_value)
        self._occ_bin = occlusion_bin_px
        self._occ_margin = occlusion_depth_margin

        self.input_keys = [labels_key, projection_key]
        self.timestamps_from = labels_key
        self.sources = [labels_key, projection_key]
        if output_key is not None:
            self.output_key = output_key

    def __call__(self, sample: Sample) -> np.ndarray:
        labels = np.asarray(sample.data[self._labels_key]).ravel()
        uv = np.asarray(sample.data[self._projection_key])
        if uv.ndim != 2 or uv.shape[1] < 3:
            raise ValueError(f"projection must be (N, 3) [u, v, depth], got {uv.shape}")
        if len(labels) != len(uv):
            raise ValueError(
                f"labels ({len(labels)}) and projection ({len(uv)}) are not "
                f"row-aligned — they must derive from the same scan."
            )

        height, width = self._height, self._width
        mask = np.full((height, width), self._unknown, dtype=np.uint8)

        valid = np.isfinite(uv[:, 0])
        if not valid.any():
            return mask

        cols = uv[valid, 0].astype(np.intp)
        rows = uv[valid, 1].astype(np.intp)
        depth = uv[valid, 2].astype(np.float64)
        vals = labels[valid]
        if vals.min() < 0 or vals.max() > 255:
            raise ValueError(
                f"labels must fit uint8, got range [{vals.min()}, {vals.max()}]"
            )
        vals = vals.astype(np.uint8)

        if self._occ_bin is not None:
            keep = self._occlusion_keep(rows, cols, depth)
            rows, cols, depth, vals = rows[keep], cols[keep], depth[keep], vals[keep]

        # Splat each point over a disc of offsets, then resolve every pixel
        # to its nearest candidate in one pass.
        r = self._radius
        offsets = [
            (du, dv)
            for du in range(-r, r + 1)
            for dv in range(-r, r + 1)
            if du * du + dv * dv <= r * r
        ]
        pix_parts, depth_parts, val_parts = [], [], []
        for du, dv in offsets:
            rr = rows + dv
            cc = cols + du
            ok = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            pix_parts.append(rr[ok] * width + cc[ok])
            depth_parts.append(depth[ok])
            val_parts.append(vals[ok])
        pix = np.concatenate(pix_parts)
        d = np.concatenate(depth_parts)
        v = np.concatenate(val_parts)

        order = np.lexsort((d, pix))  # by pixel, nearest first within a pixel
        pix, v = pix[order], v[order]
        first = np.ones(len(pix), dtype=bool)
        first[1:] = pix[1:] != pix[:-1]
        mask.flat[pix[first]] = v[first]
        return mask

    def _occlusion_keep(
        self, rows: np.ndarray, cols: np.ndarray, depth: np.ndarray
    ) -> np.ndarray:
        """Keep points within ``occlusion_depth_margin`` of their bin's nearest
        return, on a coarse ``occlusion_bin_px`` grid."""
        b = self._occ_bin
        n_bins_x = (self._width + b - 1) // b
        n_bins_y = (self._height + b - 1) // b
        bin_id = (rows // b) * n_bins_x + (cols // b)
        nearest = np.full(n_bins_x * n_bins_y, np.inf)
        np.minimum.at(nearest, bin_id, depth)
        return depth <= nearest[bin_id] + self._occ_margin
