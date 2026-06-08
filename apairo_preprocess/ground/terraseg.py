"""Ground segmentation with TerraSeg (TedLentsch/TerraSeg).

``TerraSegPredictor`` classifies each LiDAR point as ground (0) or non-ground
(1).  The model is self-supervised and domain-agnostic, trained on ~22 M raw
scans from 12 public autonomous-driving benchmarks across 15 distinct sensors.

Weights are downloaded automatically from Hugging Face on first use.

Requires::

    pip install terraseg

Typical usage::

    dataset.run_preprocess(
        TerraSegGroundSegmentation(variant="S"),
        split_dir,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample

_HF_CHECKPOINTS: dict[str, str] = {
    "S": "hf://TedLentsch/TerraSeg/terraseg_s.pth",
    "B": "hf://TedLentsch/TerraSeg/terraseg_b.pth",
}


class TerraSegGroundSegmentation(FramePreprocessor):
    """Per-point ground/non-ground labels from TerraSeg.

    Runs the self-supervised TerraSeg model on the voxelised point cloud.
    Output labels follow TerraSeg's convention: **0 = ground, 1 = non-ground**.

    Weights are fetched automatically from Hugging Face the first time the
    predictor is instantiated (requires network access and ``huggingface_hub``).
    Pass ``checkpoint_path`` to use a local file instead.

    Args:
        variant:         Model size — ``"S"`` (~12 M params, 17-50 Hz on A100)
                         or ``"B"`` (~46 M params, 10-28 Hz on A100).  Default ``"S"``.
        voxelised_key:   Input channel name for the voxelised point cloud
                         (shape N x D≥3, first three columns are XYZ in metres).
        device:          Torch device string.  Defaults to CUDA if available,
                         else CPU.
        grid_size:       Voxel size (metres) used internally by the predictor
                         for feature computation.  Default 0.05 m.
        decision_thres:  Probability threshold for the ground/non-ground split.
                         ``None`` uses TerraSeg's built-in default.
        compile_model:   Enable ``torch.compile`` for higher throughput on
                         supported GPUs.  Default ``False``.
        checkpoint_path: Local path or Hugging Face URI to the checkpoint.
                         Defaults to the official HF weights for the chosen
                         variant.
        output_key:      Override the default output channel name
                         ``"terraseg_ground"``.
    """

    output_key: ClassVar[str] = "terraseg_ground"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["voxelised"]
    timestamps_from: ClassVar[str] = "voxelised"
    sources: ClassVar[list[str]] = ["voxelised"]

    def __init__(
        self,
        variant: str = "S",
        voxelised_key: str = "voxelised",
        device: str | None = None,
        grid_size: float = 0.05,
        decision_thres: float | None = None,
        compile_model: bool = False,
        checkpoint_path: str | Path | None = None,
        output_key: str | None = None,
    ) -> None:
        try:
            import torch
            from terraseg import TerraSegPredictor
        except ImportError as exc:
            raise ImportError(
                "TerraSegGroundSegmentation requires the terraseg package: "
                "pip install terraseg"
            ) from exc

        if variant not in _HF_CHECKPOINTS:
            raise ValueError(
                f"variant must be one of {list(_HF_CHECKPOINTS)}, got {variant!r}"
            )
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")

        resolved_checkpoint = checkpoint_path or _HF_CHECKPOINTS[variant]
        device_obj = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._predictor = TerraSegPredictor(
            variant=variant,
            checkpoint_path=resolved_checkpoint,
            device=device_obj,
            decision_thres=decision_thres,
            compile_model=compile_model,
        )
        self._grid_size = grid_size
        self._voxelised_key = voxelised_key

        self.input_keys = [voxelised_key]
        self.sources = [voxelised_key]
        if output_key is not None:
            self.output_key = output_key

    def process(self, sample: Sample) -> np.ndarray:
        import torch

        pc = np.asarray(sample.data[self._voxelised_key], dtype=np.float32)
        if pc.ndim != 2 or pc.shape[1] < 3:
            raise ValueError(f"Point cloud must be (N, D>=3), got {pc.shape}")

        coords = torch.tensor(
            pc[:, :3], dtype=torch.float32, device=self._predictor._device
        )
        labels = self._predictor.predict(coord=coords, grid_size=self._grid_size)
        return labels.cpu().numpy().astype(np.uint8)
