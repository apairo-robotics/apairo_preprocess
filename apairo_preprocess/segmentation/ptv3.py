"""LiDAR semantic labeling with standalone Point Transformer V3.

Wraps the detached PointTransformerV3 repository as a FramePreprocessor.  The
checkpoint must contain weights for both the PTv3 backbone and a linear
segmentation head.  Input scans are voxelized with a numpy grid sample; one
representative per voxel is forwarded through the model, and voxel predictions
are propagated back to all original points.

Expected checkpoint layout::

    {
        "backbone.<param>": ...,   # PTv3 backbone weights
        "seg_head.<param>": ...,   # nn.Linear head weights
    }

Typical usage::

    dataset.run_preprocess(
        PTv3Labels(
            checkpoint_path="weights/ptv3_outdoor.pt",
            num_classes=19,
            model_root="~/dev/models/PointTransformerV3",
            in_channels=4,
            grid_size=0.05,
        ),
        split_dir,
    )
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

import numpy as np

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample


class PTv3Labels(FramePreprocessor):
    """Per-point semantic labels from Point Transformer V3.

    Builds a PTv3 backbone and a linear segmentation head.  Weights for both
    are loaded from ``checkpoint_path``.  Grid-sampled voxel representatives
    are forwarded; voxel-level predictions are mapped back to the full scan.

    Args:
        checkpoint_path:        Path to the checkpoint file.  State dict keys
            must be prefixed ``"backbone."`` and ``"seg_head."``.
        num_classes:            Number of output semantic classes.
        model_root:             Path to the PointTransformerV3 repository root.
            Added to ``sys.path`` for imports.
        in_channels:            Feature dimensionality fed to the model
            (default 4: XYZ + intensity).  Must match the training setup.
        backbone_out_channels:  Output feature dimension of the PTv3 decoder
            (``dec_channels[0]`` in the training config, default 64).  The
            linear head maps this to ``num_classes``.
        grid_size:              Voxel edge length in metres for grid sampling.
        lidar_key:              Input channel name for the point cloud.
        device:                 Torch device string.  Defaults to CUDA if
            available.
        output_key:             Override the default output channel name
            ``"ptv3_labels"``.
    """

    output_key: ClassVar[str] = "ptv3_labels"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["lidar"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["lidar"]

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int,
        model_root: str,
        in_channels: int = 4,
        backbone_out_channels: int = 64,
        grid_size: float = 0.05,
        lidar_key: str = "lidar",
        device: str | None = None,
        output_key: str | None = None,
    ) -> None:
        import torch
        import torch.nn as nn

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")

        model_root_resolved = str(Path(model_root).expanduser().resolve())
        if model_root_resolved not in sys.path:
            sys.path.insert(0, model_root_resolved)

        try:
            from model import PointTransformerV3
        except ImportError as exc:
            raise ImportError(
                f"Could not import PointTransformerV3 from {model_root_resolved}. "
                "Check that model_root points to the PointTransformerV3 repository."
            ) from exc

        device_obj = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        backbone = PointTransformerV3(in_channels=in_channels)
        seg_head = nn.Linear(backbone_out_channels, num_classes)

        state = torch.load(checkpoint_path, map_location=device_obj)
        state_dict = state.get("model", state.get("state_dict", state))

        backbone_state = {
            k.removeprefix("backbone."): v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
        head_state = {
            k.removeprefix("seg_head."): v
            for k, v in state_dict.items()
            if k.startswith("seg_head.")
        }
        backbone.load_state_dict(backbone_state)
        seg_head.load_state_dict(head_state)

        self._backbone = backbone.to(device_obj).eval()
        self._seg_head = seg_head.to(device_obj).eval()
        self._device = device_obj
        self._in_channels = in_channels
        self._grid_size = float(grid_size)
        self._lidar_key = lidar_key

        self.input_keys = [lidar_key]
        self.sources = [lidar_key]
        if output_key is not None:
            self.output_key = output_key

    def process(self, sample: Sample) -> np.ndarray:
        import torch
        from addict import Dict as ADict

        pc = np.asarray(sample.data[self._lidar_key], dtype=np.float32)
        if pc.ndim != 2 or pc.shape[1] < 3:
            raise ValueError(f"Point cloud must be (N, D>=3), got {pc.shape}")

        xyz = pc[:, :3]
        feats = self._build_features(pc)

        grid_coords = np.floor(xyz / self._grid_size).astype(np.int32)
        _, inds, inverse = np.unique(
            grid_coords, axis=0, return_index=True, return_inverse=True
        )

        input_dict = ADict(
            coord=torch.tensor(xyz[inds], dtype=torch.float32, device=self._device),
            grid_coord=torch.tensor(
                grid_coords[inds], dtype=torch.int, device=self._device
            ),
            feat=torch.tensor(feats[inds], dtype=torch.float32, device=self._device),
            offset=torch.tensor([len(inds)], dtype=torch.long, device=self._device),
        )

        with torch.no_grad():
            point = self._backbone(input_dict)
            logits = self._seg_head(point.feat)  # (N_voxels, num_classes)

        voxel_labels = logits.argmax(dim=1).cpu().numpy()  # (N_voxels,)
        return voxel_labels[inverse].astype(np.int32)       # (N,)

    def _build_features(self, pc: np.ndarray) -> np.ndarray:
        n, available = pc.shape
        if available >= self._in_channels:
            return pc[:, : self._in_channels]
        pad = np.zeros((n, self._in_channels - available), dtype=np.float32)
        return np.column_stack([pc, pad])
