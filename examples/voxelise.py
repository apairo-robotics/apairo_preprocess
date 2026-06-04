"""Voxel-grid downsampling of LiDAR scans with consistent label aggregation.

Runs two preprocessors in sequence on the same sequence directory:

  1. ``VoxelisePointCloud`` — reduces each scan to one representative point per
     voxel cell.  Output channel: ``voxelised`` (npys, float64, shape (N', 4)).

  2. ``VoxeliseLabels`` — aggregates per-point semantic labels onto the same
     voxel grid using the chosen strategy (``majority`` or ``max``).
     Output channel: ``voxelised_labels`` (npys, int64, shape (N',)).

Both steps use identical ``voxel_size`` and ``max_range`` values so that
``voxelised[i]`` and ``voxelised_labels[i]`` refer to the same voxel.

Supported datasets: rellis, semantic_kitti.

Usage::

    python examples/voxelise.py /data/RELLIS/00000
    python examples/voxelise.py /data/sequences/00 --dataset semantic_kitti \\
        --voxel-size 0.1 --max-range 50.0 --label-aggregation max
    python examples/voxelise.py /data/RELLIS/00000 --overwrite
"""

import argparse
from pathlib import Path

from apairo.dataset.rellis.dataset import Rellis3DDataset
from apairo.dataset.semantic_kitti.dataset import SemanticKittiDataset

from apairo_preprocess.pointcloud.voxelise import VoxeliseLabels, VoxelisePointCloud

DATASETS = {
    "rellis": Rellis3DDataset,
    "semantic_kitti": SemanticKittiDataset,
}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("seq_dir", help="Sequence directory.")
    p.add_argument("--dataset", choices=DATASETS, default="rellis",
                   help="Dataset format (default: rellis).")
    p.add_argument("--lidar-key", default="lidar",
                   help="Input LiDAR channel (default: lidar).")
    p.add_argument("--labels-key", default="labels",
                   help="Input semantic labels channel (default: labels).")
    p.add_argument("--voxel-size", type=float, default=0.1,
                   help="Voxel edge length in metres (default: 0.1).")
    p.add_argument("--max-range", type=float, default=None,
                   help="Discard points beyond this range in metres (default: no limit).")
    p.add_argument("--reduction", choices=["centroid", "first", "random"],
                   default="centroid",
                   help="Point reduction strategy per voxel (default: centroid).")
    p.add_argument("--label-aggregation", choices=["majority", "max"],
                   default="majority",
                   help="Label aggregation strategy per voxel (default: majority).")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute and overwrite existing output channels.")
    args = p.parse_args()

    seq_dir = Path(args.seq_dir)
    dataset_cls = DATASETS[args.dataset]

    print(f"Dataset          : {dataset_cls.__name__}")
    print(f"Sequence         : {seq_dir}")
    print(f"Voxel size       : {args.voxel_size} m")
    print(f"Max range        : {args.max_range} m" if args.max_range else "Max range        : none")
    print(f"PC reduction     : {args.reduction}")
    print(f"Label aggregation: {args.label_aggregation}")
    print()

    # --- Step 1: voxelise point cloud ---
    print("Step 1/2  Voxelising point cloud …")
    pc_prep = VoxelisePointCloud(
        lidar_key=args.lidar_key,
        voxel_size=args.voxel_size,
        max_range=args.max_range,
        reduction=args.reduction,
    )
    dataset_cls.run_preprocess(pc_prep, seq_dir, overwrite=args.overwrite)
    print(f"          → channel '{pc_prep.output_key}' written.\n")

    # --- Step 2: voxelise labels (same grid) ---
    print("Step 2/2  Aggregating semantic labels …")
    lbl_prep = VoxeliseLabels(
        lidar_key=args.lidar_key,
        labels_key=args.labels_key,
        voxel_size=args.voxel_size,
        max_range=args.max_range,
        aggregation=args.label_aggregation,
    )
    dataset_cls.run_preprocess(lbl_prep, seq_dir, overwrite=args.overwrite)
    print(f"          → channel '{lbl_prep.output_key}' written.\n")

    print("Done.")
    print()
    print("Both channels share the same voxel index — voxelised[i] and")
    print("voxelised_labels[i] correspond to the same voxel cell.")


if __name__ == "__main__":
    main()
