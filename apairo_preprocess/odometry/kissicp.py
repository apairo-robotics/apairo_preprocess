"""LiDAR odometry via KISS-ICP.

Produces one (4, 4) float64 world pose per scan using KISS-ICP
(https://github.com/PRBonn/kiss-icp).  Much faster than Open3D GICP (~100x)
and handles motion deskewing natively when per-point timestamps are available.

Output channel: ``kissicp_poses``  (npys — one .npy per scan)

Typical usage::

    dataset.run_preprocess(
        KissICPOdometry(voxel_size=1.0),
        split_dir,
    )
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample

try:
    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP
    _KISS_OK = True
except ImportError:
    _KISS_OK = False


class KissICPOdometry(FramePreprocessor):
    """Stateful KISS-ICP odometry → one (4, 4) world pose per frame.

    Each call to :meth:`process` registers the current scan and returns
    the accumulated T_world_lidar pose.  The first scan is the world origin.

    Args:
        lidar_key:  Input channel for point cloud data.
        max_range:  Maximum point range kept before registration (metres).
        min_range:  Minimum point range kept (metres).
        voxel_size: Map voxel size (metres).  Controls both resolution and speed.
        deskew:     Enable motion deskewing.  Requires per-point timestamps in
                    the lidar channel (4th column).
        output_key: Override the default output channel name ``"kissicp_poses"``.
    """

    output_key: ClassVar[str] = "kissicp_poses"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["lidar"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["lidar"]

    def __init__(
        self,
        lidar_key: str = "lidar",
        max_range: float = 50.0,
        min_range: float = 1.0,
        voxel_size: float = 1.0,
        deskew: bool = False,
        output_key: str | None = None,
    ) -> None:
        self._lidar_key = lidar_key
        self.input_keys = [lidar_key]
        self.sources = [lidar_key]
        if output_key is not None:
            self.output_key = output_key
        if not _KISS_OK:
            raise ImportError("kiss-icp is required for KissICPOdometry.  pip install kiss-icp")
        cfg = KISSConfig()
        cfg.data.max_range = max_range
        cfg.data.min_range = min_range
        cfg.data.deskew = deskew
        cfg.mapping.voxel_size = float(voxel_size)
        self._kiss = KissICP(config=cfg)
        self._deskew = deskew

    def process(self, sample: Sample) -> np.ndarray:
        pc = np.asarray(sample.data[self._lidar_key], dtype=np.float64)
        xyz = pc[:, :3]
        timestamps = pc[:, 3] if self._deskew and pc.shape[1] > 3 else np.zeros(len(xyz))
        self._kiss.register_frame(xyz, timestamps)
        return self._kiss.last_pose.copy()  # (4, 4) float64
