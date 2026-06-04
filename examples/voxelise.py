"""Voxel-grid downsampling of LiDAR scans with labels and integer coordinates.

Runs three preprocessors in sequence on the same sequence directory:

  1. ``VoxelisePointCloud``  — float features (xyz centroid + intensity), shape (N', 4).
  2. ``VoxeliseLabels``      — aggregated semantic label per voxel, shape (N',).
  3. ``VoxeliseCoords``      — integer voxel coordinates floor(xyz/voxel_size),
                               shape (N', 3) int32.

All three use identical ``voxel_size`` and ``max_range`` so that index ``i``
refers to the same voxel cell across all channels — no recomputation needed at
training time.

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

from apairo_preprocess.pointcloud.voxelise import (
    VoxeliseCoords,
    VoxeliseLabels,
    VoxelisePointCloud,
)

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

    shared = dict(voxel_size=args.voxel_size, max_range=args.max_range)

    steps = [
        (
            "Voxelising point cloud",
            VoxelisePointCloud(
                lidar_key=args.lidar_key,
                reduction=args.reduction,
                **shared,
            ),
        ),
        (
            "Aggregating semantic labels",
            VoxeliseLabels(
                lidar_key=args.lidar_key,
                labels_key=args.labels_key,
                aggregation=args.label_aggregation,
                **shared,
            ),
        ),
        (
            "Saving integer voxel coordinates",
            VoxeliseCoords(
                lidar_key=args.lidar_key,
                **shared,
            ),
        ),
    ]

    for i, (desc, prep) in enumerate(steps, 1):
        print(f"Step {i}/{len(steps)}  {desc} …")
        dataset_cls.run_preprocess(prep, seq_dir, overwrite=args.overwrite)
        print(f"          → channel '{prep.output_key}' written.\n")

    print("Done.")
    print()
    print("The three channels share the same voxel index — a data loader can")
    print("load them directly without recomputing floor(xyz / voxel_size):")
    print()
    print("    feats  = np.load('voxelised/000000.npy')        # (N', 4) float64")
    print("    labels = np.load('voxelised_labels/000000.npy') # (N',)   int")
    print("    coords = np.load('voxel_coords/000000.npy')     # (N', 3) int32")


if __name__ == "__main__":
    main()
