"""Project lidar points onto a camera image plane.

For each scan, every point is mapped through the lidar→camera extrinsic and
the pinhole intrinsics to continuous pixel coordinates.  The output is
**row-aligned** with the input scan: row *i* is ``[u, v, depth]`` for point
*i*, or ``NaN`` when the point does not land in the image (behind the camera
or out of bounds).  Row alignment is what makes the channel composable — any
per-point channel (traversability labels, priors, …) carries into image space
by indexing with the same row (see :class:`ImageMaskFromPointLabels` and
:class:`PointFeaturesFromImage`).

The camera frame must be the **optical** frame (+Z forward, +X right, +Y
down).  Both halves of the projection come from the dataset's calibration —
the extrinsic from the transform graph, the intrinsics from the ``cameras:``
section::

    cal = ds.calibration
    cam = cal.get_intrinsics("camera_left")            # CameraIntrinsics
    T   = cal.get_tf("velodyne_0", "camera_left")      # (4, 4)

Typical usage::

    TartanKittiDataset.run_preprocess(
        LidarCameraProjection(
            intrinsics=cam,              # CameraIntrinsics, or a (3, 3) K
            extrinsics=T,                # (4, 4) T_camera_from_lidar
            lidar_key="velodyne_0",
            output_key="lidar_uv_left",
        ),
        seq_dir,
    )
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from apairo.core.config import CameraIntrinsics
from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample


def _distort_plumb_bob(x: np.ndarray, y: np.ndarray, d: np.ndarray) -> tuple:
    """Apply plumb-bob (radial-tangential) distortion to normalized coords."""
    k1, k2, p1, p2, k3 = d
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def _monotonic_limit(d: np.ndarray) -> float:
    """The squared radius below which the radial model is one-to-one.

    The distorted radius is ``r * (1 + k1 r^2 + k2 r^4 + k3 r^6)``. Past the
    radius where its derivative first reaches zero the polynomial turns back:
    a point far outside the field of view is folded onto the image, as a
    valid-looking pixel. That limit, as ``r^2``; ``inf`` when the model never
    turns back.
    """
    k1, k2, _, _, k3 = d
    # d/dr [r (1 + k1 s + k2 s^2 + k3 s^3)] with s = r^2
    roots = np.roots([7.0 * k3, 5.0 * k2, 3.0 * k1, 1.0])
    positive = [float(r.real) for r in roots if abs(r.imag) < 1e-12 and r.real > 0]
    return min(positive) if positive else float("inf")


class LidarCameraProjection(FramePreprocessor):
    """Per-point pixel coordinates of a lidar scan in a camera image.

    Output is ``(N, 3)`` float32, row-aligned with the input scan:
    ``[u, v, depth]`` — continuous pixel coordinates (``u`` = column, ``v`` =
    row, origin at the top-left pixel corner) and depth along the camera's
    optical axis (metres).  Points behind the camera or projecting outside
    ``image_size`` are ``NaN`` in all three columns, so validity is
    ``np.isfinite(uv[:, 0])`` and ``int(u)``/``int(v)`` of a valid row always
    index into the image.

    Only the scan itself is streamed (``input_keys`` has a single channel):
    extrinsics, intrinsics and image size are static rig properties, so the
    preprocessor runs on asynchronous datasets as well as profiled ones.

    Args:
        intrinsics:  A :class:`~apairo.core.config.CameraIntrinsics` from
                     ``ds.calibration.get_intrinsics(frame)``, or a bare
                     ``(3, 3)`` pinhole camera matrix K.
        extrinsics:  ``(4, 4)`` rigid transform ``T_camera_from_lidar`` — maps
                     lidar-frame points into the camera *optical* frame
                     (+Z forward).  Typically
                     ``ds.calibration.get_tf(lidar_frame, camera_frame)``.
        image_size:  ``(height, width)`` of the target image, in pixels.
                     Defaults to the ``CameraIntrinsics``' recorded size;
                     required when passing a bare K.
        distortion:  Plumb-bob coefficients ``(k1, k2, p1, p2[, k3])``.
                     Defaults to the ``CameraIntrinsics``' coefficients;
                     ``None`` with a bare K means a rectified image.
        lidar_key:   Input channel for point cloud data.
        output_key:  Override the default output channel name ``"lidar_uv"``.
    """

    output_key: ClassVar[str] = "lidar_uv"
    input_keys: ClassVar[list[str]] = ["lidar"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["lidar"]

    def __init__(
        self,
        intrinsics: np.ndarray | CameraIntrinsics,
        extrinsics: np.ndarray,
        image_size: tuple[int, int] | None = None,
        distortion: np.ndarray | None = None,
        lidar_key: str = "lidar",
        output_key: str | None = None,
    ) -> None:
        if isinstance(intrinsics, CameraIntrinsics):
            if intrinsics.model != "plumb_bob" and len(intrinsics.distortion):
                raise ValueError(
                    f"only the plumb_bob distortion model is supported, "
                    f"got {intrinsics.model!r}"
                )
            if distortion is None and len(intrinsics.distortion):
                distortion = intrinsics.distortion
            if image_size is None:
                if intrinsics.height is None or intrinsics.width is None:
                    raise ValueError(
                        "CameraIntrinsics records no image size — pass image_size="
                    )
                image_size = (intrinsics.height, intrinsics.width)
            intrinsics = intrinsics.K
        elif image_size is None:
            raise ValueError("image_size is required when intrinsics is a bare K")
        K = np.asarray(intrinsics, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"intrinsics must be (3, 3), got {K.shape}")
        T = np.asarray(extrinsics, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"extrinsics must be (4, 4), got {T.shape}")
        height, width = image_size
        if height <= 0 or width <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if distortion is not None:
            d = np.asarray(distortion, dtype=np.float64).ravel()
            if d.size == 4:
                d = np.append(d, 0.0)  # k3 defaults to 0
            if d.size != 5:
                raise ValueError(
                    f"distortion must be (k1, k2, p1, p2[, k3]), got {d.size} values"
                )
            self._dist = d
            self._r2_limit = _monotonic_limit(d)
        else:
            self._dist = None
            self._r2_limit = float("inf")

        self._K = K
        self._T = T
        self._height = int(height)
        self._width = int(width)
        self._lidar_key = lidar_key

        self.input_keys = [lidar_key]
        self.sources = [lidar_key]
        if output_key is not None:
            self.output_key = output_key

    def __call__(self, sample: Sample) -> np.ndarray:
        pc = np.asarray(sample.data[self._lidar_key])
        xyz = pc[:, :3].astype(np.float64)
        n = len(xyz)

        cam = xyz @ self._T[:3, :3].T + self._T[:3, 3]
        z = cam[:, 2]

        out = np.full((n, 3), np.nan, dtype=np.float32)
        front = z > 1e-9
        if self._dist is not None:
            # Outside the radius where the distortion model is one-to-one, a
            # point is not in the image, whatever pixel the polynomial gives:
            # far off-axis lidar returns were folded back into the picture.
            with np.errstate(invalid="ignore", divide="ignore"):
                r2 = (cam[:, 0] ** 2 + cam[:, 1] ** 2) / (z * z)
            front &= r2 < self._r2_limit
        if not front.any():
            return out

        x = cam[front, 0] / z[front]
        y = cam[front, 1] / z[front]
        if self._dist is not None:
            x, y = _distort_plumb_bob(x, y, self._dist)

        # The bound test runs on the float32 values actually returned: a u
        # just below the width in float64 can round up to exactly the width
        # in float32, and must then be outside, not one pixel past the edge.
        u = (self._K[0, 0] * x + self._K[0, 1] * y + self._K[0, 2]).astype(np.float32)
        v = (self._K[1, 1] * y + self._K[1, 2]).astype(np.float32)

        in_image = (u >= 0) & (u < self._width) & (v >= 0) & (v < self._height)
        idx = np.flatnonzero(front)[in_image]
        out[idx, 0] = u[in_image]
        out[idx, 1] = v[in_image]
        out[idx, 2] = z[idx]
        return out
