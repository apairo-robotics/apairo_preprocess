"""Height above ground surface from binary ground segmentation labels."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.spatial import KDTree

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample


class GroundHeightFromLabels(FramePreprocessor):
    """Per-voxel height above ground from any binary ground segmentation.

    Takes the output of any ground segmentation preprocessor
    (``GroundSegmentationCSF``, ``GroundSegmentationRANSAC``,
    ``TerraSegGroundSegmentation``) and computes the signed height (m) of
    each voxel above the nearest ground point in XY.

    Args:
        ground_key:    Input channel for binary ground labels (0=ground,
                       1=non-ground).  Typically ``"ground_csf"``,
                       ``"ground_ransac"``, or ``"terraseg_ground"``.
        voxelised_key: Input channel for the voxelised point cloud
                       (shape N x D≥3).
        output_key:    Override the default channel name ``"ground_height"``.
    """

    output_key: ClassVar[str] = "ground_height"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["voxelised", "ground_labels"]
    timestamps_from: ClassVar[str] = "voxelised"
    sources: ClassVar[list[str]] = ["voxelised", "ground_labels"]

    def __init__(
        self,
        ground_key: str,
        voxelised_key: str = "voxelised",
        output_key: str | None = None,
    ) -> None:
        self._ground_key = ground_key
        self._voxelised_key = voxelised_key
        self.input_keys = [voxelised_key, ground_key]
        self.sources = [voxelised_key, ground_key]
        if output_key is not None:
            self.output_key = output_key

    def process(self, sample: Sample) -> np.ndarray:
        xyz = np.asarray(sample.data[self._voxelised_key], dtype=np.float64)[:, :3]
        labels = np.asarray(sample.data[self._ground_key])
        ground_pts = xyz[labels == 0]

        if len(ground_pts) == 0:
            z_ground = float(np.percentile(xyz[:, 2], 5))
            return (xyz[:, 2] - z_ground).astype(np.float32)

        tree = KDTree(ground_pts[:, :2])
        _, nn = tree.query(xyz[:, :2])
        return (xyz[:, 2] - ground_pts[nn, 2]).astype(np.float32)
