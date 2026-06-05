"""LiDAR semantic labeling with SPVNAS / SPVCNN / MinkUNet.

Wraps the SPVNAS repository as a FramePreprocessor.  The model is loaded once
at construction time; each call to process() voxelizes the input scan with
torchsparse's ``sparse_quantize``, runs inference, and maps voxel predictions
back to the original points via the inverse mapping.

Typical usage::

    dataset.run_preprocess(
        SPVNASLabels(
            checkpoint_path="weights/spvcnn_init",
            config_path="weights/net.config",
            model_root="~/dev/models/spvnas",
        ),
        split_dir,
    )
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample

_ModelType = Literal["spvcnn", "spvnas", "minkunet"]


class SPVNASLabels(FramePreprocessor):
    """Per-point semantic labels from SPVNAS / SPVCNN / MinkUNet.

    Points are voxelized at ``voxel_size`` resolution using torchsparse's
    ``sparse_quantize``; the unique voxels are passed through the model, and
    voxel-level predictions are propagated back to every original point via
    the inverse map.  All extra channels beyond XYZ are preserved to build
    the 4-channel feature vector (XYZ + intensity) expected by these models;
    the intensity column is padded with zeros if absent.

    Args:
        checkpoint_path:   Path to a PyTorch checkpoint.  The state dict must
            be stored under the key ``"model"`` (SPVNAS training convention).
        config_path:       Path to the ``net.config`` JSON describing the
            architecture (``num_classes``, ``pres``, ``vres``, …).
        model_root:        Path to the cloned SPVNAS repository root.  Added
            to ``sys.path`` so that ``core.*`` modules can be imported.
        model_type:        Architecture variant: ``"spvcnn"``, ``"spvnas"``,
            or ``"minkunet"``.
        lidar_key:         Input channel name for the point cloud.
        intensity_channel: Index of the intensity column in the point cloud
            (typically 3).  Pass ``None`` to pad with zeros instead.
        voxel_size:        Quantization step in metres (0.05 m default,
            matching the SemanticKITTI training setup).
        device:            Torch device string.  Defaults to CUDA if available.
        output_key:        Override the default output channel name
            ``"spvnas_labels"``.
    """

    output_key: ClassVar[str] = "spvnas_labels"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["lidar"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["lidar"]

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        model_root: str,
        model_type: _ModelType = "spvcnn",
        lidar_key: str = "lidar",
        intensity_channel: int | None = 3,
        voxel_size: float = 0.05,
        device: str | None = None,
        output_key: str | None = None,
    ) -> None:
        import torch

        if model_type not in ("spvcnn", "spvnas", "minkunet"):
            raise ValueError(
                f"model_type must be 'spvcnn', 'spvnas', or 'minkunet', got {model_type!r}"
            )
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")

        model_root_resolved = str(Path(model_root).expanduser().resolve())
        if model_root_resolved not in sys.path:
            sys.path.insert(0, model_root_resolved)

        try:
            from torchsparse.utils.quantize import sparse_quantize  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "torchsparse is required by SPVNASLabels. "
                "Install it following the instructions in the spvnas README."
            ) from exc

        with open(config_path) as f:
            net_config = json.load(f)

        device_obj = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = _build_spvnas_model(model_type, net_config, device_obj)
        state = torch.load(checkpoint_path, map_location=device_obj)
        model.load_state_dict(state.get("model", state))
        model.eval()

        self._model = model
        self._device = device_obj
        self._lidar_key = lidar_key
        self._intensity_channel = intensity_channel
        self._voxel_size = float(voxel_size)
        self._num_classes = net_config["num_classes"]

        self.input_keys = [lidar_key]
        self.sources = [lidar_key]
        if output_key is not None:
            self.output_key = output_key

    def process(self, sample: Sample) -> np.ndarray:
        import torch
        from torchsparse import SparseTensor
        from torchsparse.utils.collate import sparse_collate
        from torchsparse.utils.quantize import sparse_quantize

        pc = np.asarray(sample.data[self._lidar_key], dtype=np.float32)
        if pc.ndim != 2 or pc.shape[1] < 3:
            raise ValueError(f"Point cloud must be (N, D>=3), got {pc.shape}")

        xyz = pc[:, :3]
        if (
            self._intensity_channel is not None
            and pc.shape[1] > self._intensity_channel
        ):
            feats = np.column_stack([xyz, pc[:, self._intensity_channel]])
        else:
            feats = np.column_stack([xyz, np.zeros(len(xyz), dtype=np.float32)])

        coords = np.round(xyz / self._voxel_size).astype(np.int32)
        coords -= coords.min(0, keepdims=True)
        coords_unique, inds, inverse = sparse_quantize(
            coords, return_index=True, return_inverse=True
        )

        inputs = SparseTensor(
            feats=torch.tensor(feats[inds], dtype=torch.float32),
            coords=torch.tensor(coords_unique, dtype=torch.int),
        )
        inputs = sparse_collate([inputs]).to(self._device)

        with torch.no_grad():
            outputs = self._model(inputs)

        voxel_labels = outputs.F.argmax(dim=1).cpu().numpy()  # (N_voxels,)
        return voxel_labels[inverse].astype(np.int32)          # (N,)


def _build_spvnas_model(
    model_type: str, net_config: dict, device: object
) -> object:
    if model_type == "spvcnn":
        from core.models.semantic_kitti.spvcnn import SPVCNN

        return SPVCNN(
            num_classes=net_config["num_classes"],
            cr=net_config.get("cr", 1.0),
            pres=net_config["pres"],
            vres=net_config["vres"],
        ).to(device)
    if model_type == "spvnas":
        from core.models.semantic_kitti.spvnas import SPVNAS

        model = SPVNAS(
            net_config["num_classes"],
            macro_depth_constraint=1,
            pres=net_config["pres"],
            vres=net_config["vres"],
        ).to(device)
        model.manual_select(net_config)
        return model.determinize()
    from core.models.semantic_kitti.minkunet import MinkUNet

    return MinkUNet(
        num_classes=net_config["num_classes"],
        cr=net_config.get("cr", 1.0),
    ).to(device)
