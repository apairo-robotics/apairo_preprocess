"""Voxel-grid downsampling of LiDAR point clouds and their semantic labels.

Three preprocessors share a common voxel assignment and must be run with the
same ``voxel_size`` and ``max_range`` to stay index-aligned:

* :class:`VoxelisePointCloud` — one representative float point per voxel.
* :class:`VoxeliseLabels`     — one aggregated label per voxel (majority/max).
* :class:`VoxeliseCoords`     — integer voxel coordinates (N', 3) int32,
                                 ready for sparse convolution networks.

After preprocessing, ``voxelised[i]``, ``voxelised_labels[i]``, and
``voxel_coords[i]`` all refer to the same voxel cell.

Typical usage::

    for prep in [
        VoxelisePointCloud(voxel_size=0.1, max_range=50.0),
        VoxeliseLabels(voxel_size=0.1, max_range=50.0),
        VoxeliseCoords(voxel_size=0.1, max_range=50.0),
    ]:
        dataset.run_preprocess(prep, split_dir)
"""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample

_Reduction = Literal["centroid", "first", "random"]
_LabelAggregation = Literal["majority", "max"]


def _filter_and_quantize(
    xyz: np.ndarray,
    voxel_size: float,
    max_range: float | None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, int]:
    """Apply range filter and compute voxel assignment.

    Returns:
        mask:          Boolean keep-mask (or None if no filtering was done).
        unique_coords: Integer voxel coordinates, shape (n_voxels, 3) int32.
        inverse:       Per-point voxel index array, shape (N_kept,).
        n_voxels:      Number of occupied voxels.
    """
    mask: np.ndarray | None = None
    if max_range is not None:
        mask = np.linalg.norm(xyz, axis=1) < max_range
        xyz = xyz[mask]

    if len(xyz) == 0:
        empty_coords = np.empty((0, 3), dtype=np.int32)
        return mask, empty_coords, np.empty(0, dtype=np.int64), 0

    coords = np.floor(xyz / voxel_size).astype(np.int32)
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    return mask, unique_coords, inverse, len(unique_coords)


class VoxelisePointCloud(FramePreprocessor):
    """Downsample a point cloud with a voxel grid.

    Each voxel retains exactly one point according to ``reduction``:

    * ``"centroid"`` — mean of all points falling in the voxel (all channels).
    * ``"first"``    — first point encountered (input order).
    * ``"random"``   — one uniformly random point per voxel.

    Args:
        lidar_key:   Input channel for point cloud data.
        voxel_size:  Edge length of each cubic voxel cell (metres).
        max_range:   Discard points farther than this distance from the sensor
                     origin (metres).  ``None`` keeps all points.
        reduction:   How to pick the representative point per voxel.
        output_key:  Override the default output channel name ``"voxelised"``.
    """

    output_key: ClassVar[str] = "voxelised"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["lidar"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["lidar"]

    def __init__(
        self,
        lidar_key: str = "lidar",
        voxel_size: float = 0.1,
        max_range: float | None = None,
        reduction: _Reduction = "centroid",
        output_key: str | None = None,
    ) -> None:
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")
        if max_range is not None and max_range <= 0:
            raise ValueError(f"max_range must be positive, got {max_range}")
        if reduction not in ("centroid", "first", "random"):
            raise ValueError(
                f"reduction must be 'centroid', 'first', or 'random', got {reduction!r}"
            )

        self._lidar_key = lidar_key
        self._voxel_size = float(voxel_size)
        self._max_range = max_range
        self._reduction = reduction

        self.input_keys = [lidar_key]
        self.sources = [lidar_key]
        if output_key is not None:
            self.output_key = output_key

    def process(self, sample: Sample) -> np.ndarray:
        pc = np.asarray(sample.data[self._lidar_key], dtype=np.float64)
        if pc.ndim != 2 or pc.shape[1] < 3:
            raise ValueError(f"Point cloud must be (N, D>=3), got {pc.shape}")

        mask, _, inverse, n_voxels = _filter_and_quantize(
            pc[:, :3], self._voxel_size, self._max_range
        )
        if mask is not None:
            pc = pc[mask]

        if n_voxels == 0:
            return np.empty((0, pc.shape[1]), dtype=np.float64)

        if self._reduction == "centroid":
            out = np.zeros((n_voxels, pc.shape[1]), dtype=np.float64)
            counts = np.zeros(n_voxels, dtype=np.int64)
            np.add.at(out, inverse, pc)
            np.add.at(counts, inverse, 1)
            out /= counts[:, None]
            return out

        if self._reduction == "first":
            out = np.empty((n_voxels, pc.shape[1]), dtype=np.float64)
            # assign in reverse so the first occurrence wins
            out[inverse] = pc
            return out

        # random: shuffle so a random point ends up last (and thus selected)
        rng = np.random.default_rng()
        chosen = np.empty(n_voxels, dtype=np.int64)
        perm = rng.permutation(len(pc))
        chosen[inverse[perm]] = perm
        return pc[chosen]


class VoxeliseLabels(FramePreprocessor):
    """Aggregate per-point semantic labels onto the same voxel grid.

    Must be run with the **same** ``voxel_size`` and ``max_range`` as
    :class:`VoxelisePointCloud` so the output channels remain index-aligned.

    * ``"majority"`` — most frequent label per voxel (semantically correct).
    * ``"max"``      — highest label ID per voxel (fast, mirrors common
                       sparse-convolution data loaders).

    Args:
        lidar_key:    Input channel for point cloud data (geometry only).
        labels_key:   Input channel for per-point semantic labels.
        voxel_size:   Edge length of each cubic voxel cell (metres).
        max_range:    Same range filter as applied to the point cloud.
        aggregation:  Label merging strategy per voxel.
        output_key:   Override the default output channel name
                      ``"voxelised_labels"``.
    """

    output_key: ClassVar[str] = "voxelised_labels"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["lidar", "labels"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["lidar", "labels"]

    def __init__(
        self,
        lidar_key: str = "lidar",
        labels_key: str = "labels",
        voxel_size: float = 0.1,
        max_range: float | None = None,
        aggregation: _LabelAggregation = "majority",
        output_key: str | None = None,
    ) -> None:
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")
        if max_range is not None and max_range <= 0:
            raise ValueError(f"max_range must be positive, got {max_range}")
        if aggregation not in ("majority", "max"):
            raise ValueError(f"aggregation must be 'majority' or 'max', got {aggregation!r}")

        self._lidar_key = lidar_key
        self._labels_key = labels_key
        self._voxel_size = float(voxel_size)
        self._max_range = max_range
        self._aggregation = aggregation

        self.input_keys = [lidar_key, labels_key]
        self.sources = [lidar_key, labels_key]
        if output_key is not None:
            self.output_key = output_key

    def process(self, sample: Sample) -> np.ndarray:
        xyz = np.asarray(sample.data[self._lidar_key], dtype=np.float64)[:, :3]
        labels = np.asarray(sample.data[self._labels_key])

        mask, _, inverse, n_voxels = _filter_and_quantize(
            xyz, self._voxel_size, self._max_range
        )
        if mask is not None:
            labels = labels[mask]

        if n_voxels == 0:
            return np.empty(0, dtype=labels.dtype)

        if self._aggregation == "max":
            out = np.zeros(n_voxels, dtype=labels.dtype)
            np.maximum.at(out, inverse, labels)
            return out

        # majority: encode (voxel_idx, label) as a single int and bincount
        n_classes = int(labels.max()) + 1
        keys = inverse.astype(np.int64) * n_classes + labels.astype(np.int64)
        pair_counts = np.bincount(keys, minlength=n_voxels * n_classes)
        return pair_counts.reshape(n_voxels, n_classes).argmax(axis=1).astype(labels.dtype)


class VoxeliseCoords(FramePreprocessor):
    """Save integer voxel coordinates for each occupied cell.

    Produces an ``(N', 3)`` int32 array where row ``i`` is the quantised
    coordinate ``floor(xyz / voxel_size)`` of the ``i``-th voxel.  This is the
    format expected by sparse convolution frameworks (torchsparse,
    MinkowskiEngine) and eliminates the ``floor`` recomputation from every
    ``__getitem__``.

    Must be run with the **same** ``voxel_size`` and ``max_range`` as
    :class:`VoxelisePointCloud` and :class:`VoxeliseLabels` so all three
    channels stay index-aligned.

    Args:
        lidar_key:   Input channel for point cloud data.
        voxel_size:  Edge length of each cubic voxel cell (metres).
        max_range:   Same range filter as applied to the point cloud.
        output_key:  Override the default output channel name ``"voxel_coords"``.
    """

    output_key: ClassVar[str] = "voxel_coords"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["lidar"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["lidar"]

    def __init__(
        self,
        lidar_key: str = "lidar",
        voxel_size: float = 0.1,
        max_range: float | None = None,
        output_key: str | None = None,
    ) -> None:
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")
        if max_range is not None and max_range <= 0:
            raise ValueError(f"max_range must be positive, got {max_range}")

        self._lidar_key = lidar_key
        self._voxel_size = float(voxel_size)
        self._max_range = max_range

        self.input_keys = [lidar_key]
        self.sources = [lidar_key]
        if output_key is not None:
            self.output_key = output_key

    def process(self, sample: Sample) -> np.ndarray:
        xyz = np.asarray(sample.data[self._lidar_key], dtype=np.float64)[:, :3]
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"LiDAR channel must be (N, D>=3), got {xyz.shape}")

        _, unique_coords, _, _ = _filter_and_quantize(
            xyz, self._voxel_size, self._max_range
        )
        return unique_coords  # (N', 3) int32
