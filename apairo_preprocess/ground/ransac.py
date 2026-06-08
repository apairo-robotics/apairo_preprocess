"""Ground segmentation via RANSAC plane fitting."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample


def _ransac_plane(
    xyz: np.ndarray,
    n_iter: int = 200,
    inlier_threshold: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Fit a ground plane with RANSAC.

    Returns (normal, d) for the plane equation n · x + d = 0,
    with the normal oriented upward (positive z component).
    """
    rng = np.random.default_rng(seed)
    n = len(xyz)
    best_inliers = -1
    best_normal = np.array([0.0, 0.0, 1.0])
    best_d = float(-np.median(xyz[:, 2]))

    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = xyz[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm
        if normal[2] < 0:
            normal = -normal
        d = -float(np.dot(normal, p0))

        dist = np.abs(xyz @ normal + d)
        n_inl = int((dist < inlier_threshold).sum())
        if n_inl > best_inliers:
            best_inliers = n_inl
            best_normal = normal
            best_d = d

    return best_normal, best_d


class GroundSegmentationRANSAC(FramePreprocessor):
    """Per-point ground/non-ground labels via RANSAC plane fitting.

    Fits a plane to the voxelised point cloud with RANSAC.  Points whose
    distance to the plane is within ``inlier_threshold`` are labelled ground
    (0); all others are labelled non-ground (1).

    Output labels: **0 = ground, 1 = non-ground**.

    Args:
        voxelised_key:     Input channel (voxelised point cloud, shape (N, D≥3)).
        n_iter:            Number of RANSAC iterations.  Default 200.
        inlier_threshold:  Distance threshold (metres) for a point to be ground.
                           Default 0.15 m.
        output_key:        Override the default channel name ``"ground_ransac"``.
    """

    output_key: ClassVar[str] = "ground_ransac"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["voxelised"]
    timestamps_from: ClassVar[str] = "voxelised"
    sources: ClassVar[list[str]] = ["voxelised"]

    def __init__(
        self,
        voxelised_key: str = "voxelised",
        n_iter: int = 200,
        inlier_threshold: float = 0.15,
        output_key: str | None = None,
    ) -> None:
        self._voxelised_key = voxelised_key
        self._n_iter = n_iter
        self._inlier_threshold = inlier_threshold
        self.input_keys = [voxelised_key]
        self.sources = [voxelised_key]
        if output_key is not None:
            self.output_key = output_key

    def process(self, sample: Sample) -> np.ndarray:
        xyz = np.asarray(sample.data[self._voxelised_key], dtype=np.float64)[:, :3]
        normal, d = _ransac_plane(
            xyz, n_iter=self._n_iter, inlier_threshold=self._inlier_threshold
        )
        dist = np.abs(xyz @ normal + d)
        return (dist > self._inlier_threshold).astype(np.uint8)
