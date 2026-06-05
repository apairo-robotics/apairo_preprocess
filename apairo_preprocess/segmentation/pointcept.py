"""LiDAR semantic labeling via the Pointcept framework.

Wraps Pointcept's model registry as a FramePreprocessor using its native
config system (``pointcept.utils.config.Config``).  Any segmentor registered
in Pointcept's MODELS registry — PTv3, SpUNet, OA-CNN, SPVCNN, … — is
supported via the config file.

Input point cloud channels are used directly as features; the user must
ensure they match the ``in_channels`` value in the training config
(e.g. 6 for ScanNet RGB+normals, 4 for outdoor XYZ+intensity).

Typical usage::

    dataset.run_preprocess(
        PointceptLabels(
            config_path="~/dev/models/Pointcept/configs/scannet/semseg-pt-v3m1-0-base.py",
            checkpoint_path="weights/model_best.pth",
            model_root="~/dev/models/Pointcept",
            grid_size=0.02,
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


class PointceptLabels(FramePreprocessor):
    """Per-point semantic labels via the Pointcept framework.

    Loads any Pointcept segmentor from a config + checkpoint and runs
    per-frame inference.  Grid-sampled voxel representatives are forwarded;
    voxel-level ``seg_logits`` are mapped back to the full scan.

    Args:
        config_path:     Path to a Pointcept Python config file.  Only the
            ``model`` section is used; dataset / training sections are ignored.
        checkpoint_path: Path to a Pointcept checkpoint.  The state dict is
            loaded from key ``"state_dict"`` if present, otherwise directly.
        model_root:      Path to the Pointcept repository root.  Added to
            ``sys.path`` so that ``pointcept.*`` modules can be imported.
        grid_size:       Voxel edge length in metres.  Must match the value
            used during training (e.g. 0.02 m for ScanNet, 0.05 m for KITTI).
        lidar_key:       Input channel name for the point cloud.
        device:          Torch device string.  Defaults to CUDA if available.
        output_key:      Override the default output channel name
            ``"pointcept_labels"``.
    """

    output_key: ClassVar[str] = "pointcept_labels"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["lidar"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["lidar"]

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        model_root: str,
        grid_size: float = 0.02,
        lidar_key: str = "lidar",
        device: str | None = None,
        output_key: str | None = None,
    ) -> None:
        import torch

        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")

        model_root_resolved = str(Path(model_root).expanduser().resolve())
        if model_root_resolved not in sys.path:
            sys.path.insert(0, model_root_resolved)

        try:
            from pointcept.utils.config import Config
            from pointcept.models import build_model
        except ImportError as exc:
            raise ImportError(
                f"Could not import Pointcept from {model_root_resolved}. "
                "Check that model_root points to the Pointcept repository."
            ) from exc

        cfg = Config.fromfile(str(Path(config_path).expanduser()))
        model = build_model(cfg.model)

        device_obj = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        state = torch.load(checkpoint_path, map_location=device_obj)
        model.load_state_dict(state.get("state_dict", state))
        model = model.to(device_obj).eval()

        self._model = model
        self._device = device_obj
        self._grid_size = float(grid_size)
        self._lidar_key = lidar_key

        self.input_keys = [lidar_key]
        self.sources = [lidar_key]
        if output_key is not None:
            self.output_key = output_key

    def process(self, sample: Sample) -> np.ndarray:
        import torch

        pc = np.asarray(sample.data[self._lidar_key], dtype=np.float32)
        if pc.ndim != 2 or pc.shape[1] < 3:
            raise ValueError(f"Point cloud must be (N, D>=3), got {pc.shape}")

        xyz = pc[:, :3]
        grid_coords = np.floor(xyz / self._grid_size).astype(np.int32)
        _, inds, inverse = np.unique(
            grid_coords, axis=0, return_index=True, return_inverse=True
        )

        input_dict = dict(
            coord=torch.tensor(xyz[inds], dtype=torch.float32, device=self._device),
            grid_coord=torch.tensor(
                grid_coords[inds], dtype=torch.int, device=self._device
            ),
            feat=torch.tensor(pc[inds], dtype=torch.float32, device=self._device),
            offset=torch.tensor([len(inds)], dtype=torch.long, device=self._device),
        )

        with torch.no_grad():
            output = self._model(input_dict)

        logits = output["seg_logits"]                          # (N_voxels, num_classes)
        voxel_labels = logits.argmax(dim=1).cpu().numpy()
        return voxel_labels[inverse].astype(np.int32)          # (N,)
