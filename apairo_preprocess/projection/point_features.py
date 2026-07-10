"""Sample image values at projected lidar points — image → point cloud.

The inverse direction of :class:`ImageMaskFromPointLabels`: given the
per-point pixel coordinates written by :class:`LidarCameraProjection`, read
the image at each valid point and attach the pixel's value (RGB, semantic
class, …) to the point.  Output is row-aligned with the scan, so it loads as
a per-point feature channel next to ``lidar``.

Colouring the cloud with camera RGB and inspecting it in a 3-D viewer
(projector, apairo_rr) is the standard visual check that extrinsics and
intrinsics are right — miscalibration shows up immediately as colour bleeding
across object boundaries.

This preprocessor reads **two** channels on different clocks (the projection
on the lidar clock, the image on the camera clock).  On profiled synchronous
datasets it runs via ``run_preprocess`` directly; on asynchronous datasets,
run it over a ``synchronize()`` view and persist with ``ChannelWriter`` (see
``examples/traversability_image_mask.py``).

Typical usage::

    Rellis3DDataset.run_preprocess(
        PointFeaturesFromImage(image_key="image_left", output_key="lidar_rgb"),
        root,
    )
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample


class PointFeaturesFromImage(FramePreprocessor):
    """Per-point image values sampled at each point's projected pixel.

    Output is ``(N, C)`` in the image's dtype, row-aligned with the scan
    (``C`` = image channels; a single-channel ``(H, W)`` image yields
    ``(N, 1)``).  Points without a valid projection get ``fill_value``.
    Sampling is nearest-pixel (truncation of the continuous coordinates).

    Args:
        projection_key: Input channel with ``(N, 3)`` ``[u, v, depth]`` rows
                        from :class:`LidarCameraProjection`.
        image_key:      Input channel for the image, ``(H, W)`` or ``(H, W, C)``.
        fill_value:     Value written for points with no valid projection.
        output_key:     Override the default output channel name
                        ``"point_features"``.
    """

    output_key: ClassVar[str] = "point_features"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["lidar_uv", "image"]
    timestamps_from: ClassVar[str] = "lidar_uv"
    sources: ClassVar[list[str]] = ["lidar_uv", "image"]

    def __init__(
        self,
        projection_key: str = "lidar_uv",
        image_key: str = "image",
        fill_value: float = 0,
        output_key: str | None = None,
    ) -> None:
        self._projection_key = projection_key
        self._image_key = image_key
        self._fill_value = fill_value

        self.input_keys = [projection_key, image_key]
        self.timestamps_from = projection_key
        self.sources = [projection_key, image_key]
        if output_key is not None:
            self.output_key = output_key

    def __call__(self, sample: Sample) -> np.ndarray:
        uv = np.asarray(sample.data[self._projection_key])
        img = np.asarray(sample.data[self._image_key])
        if uv.ndim != 2 or uv.shape[1] < 2:
            raise ValueError(f"projection must be (N, >=2) [u, v, ...], got {uv.shape}")
        if img.ndim == 2:
            img = img[:, :, None]
        if img.ndim != 3:
            raise ValueError(f"image must be (H, W) or (H, W, C), got {img.shape}")

        n = len(uv)
        height, width, channels = img.shape
        out = np.full((n, channels), self._fill_value, dtype=img.dtype)

        valid = np.isfinite(uv[:, 0])
        cols = uv[valid, 0].astype(np.intp)
        rows = uv[valid, 1].astype(np.intp)
        if len(cols) and (cols.max() >= width or rows.max() >= height):
            raise ValueError(
                f"projection exceeds image bounds ({height}, {width}) — the "
                f"projection was computed for a different image_size."
            )
        out[valid] = img[rows, cols]
        return out
