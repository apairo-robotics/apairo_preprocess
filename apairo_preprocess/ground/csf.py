"""Ground segmentation via Cloth Simulation Filter (CSF)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample


def _run_csf(
    xyz: np.ndarray,
    cloth_resolution: float,
    class_threshold: float,
    rigidness: int,
) -> tuple[list[int], list[int]]:
    """Run CSF and return (ground_indices, non_ground_indices).

    Requires ``pip install CSF``.
    """
    try:
        import CSF  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "GroundSegmentationCSF requires the CSF package: pip install CSF"
        ) from exc

    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = rigidness
    csf.params.time_step = 0.65
    csf.params.class_threshold = class_threshold
    csf.params.interations = 500

    csf.setPointCloud(xyz.tolist())
    ground_idx = CSF.VecInt()
    non_ground_idx = CSF.VecInt()
    csf.do_filtering(ground_idx, non_ground_idx)
    return list(ground_idx), list(non_ground_idx)


class GroundSegmentationCSF(FramePreprocessor):
    """Per-point ground/non-ground labels via Cloth Simulation Filter.

    Runs CSF on the voxelised point cloud.  More accurate than RANSAC on
    uneven terrain; requires ``pip install CSF``.

    Output labels: **0 = ground, 1 = non-ground**.

    Args:
        voxelised_key:    Input channel (voxelised point cloud, shape (N, D≥3)).
        cloth_resolution: CSF cloth resolution (metres).  Default 0.5 m.
        class_threshold:  CSF point-to-cloth distance threshold (metres).
                          Default 0.5 m.
        rigidness:        CSF cloth rigidness (1–3).  3 = flat terrain.
                          Default 3.
        output_key:       Override the default channel name ``"ground_csf"``.
    """

    output_key: ClassVar[str] = "ground_csf"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["voxelised"]
    timestamps_from: ClassVar[str] = "voxelised"
    sources: ClassVar[list[str]] = ["voxelised"]

    def __init__(
        self,
        voxelised_key: str = "voxelised",
        cloth_resolution: float = 0.5,
        class_threshold: float = 0.5,
        rigidness: int = 3,
        output_key: str | None = None,
    ) -> None:
        self._voxelised_key = voxelised_key
        self._cloth_resolution = cloth_resolution
        self._class_threshold = class_threshold
        self._rigidness = rigidness
        self.input_keys = [voxelised_key]
        self.sources = [voxelised_key]
        if output_key is not None:
            self.output_key = output_key

    def process(self, sample: Sample) -> np.ndarray:
        xyz = np.asarray(sample.data[self._voxelised_key], dtype=np.float64)[:, :3]
        ground_idx, _ = _run_csf(
            xyz,
            cloth_resolution=self._cloth_resolution,
            class_threshold=self._class_threshold,
            rigidness=self._rigidness,
        )
        labels = np.ones(len(xyz), dtype=np.uint8)
        labels[ground_idx] = 0
        return labels
