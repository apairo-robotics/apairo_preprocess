"""Compute LiDAR odometry with KISS-ICP on any dataset that has a ``lidar`` channel.

Produces a ``kissicp_poses`` channel: one (4, 4) float64 pose per scan,
in the local odometry frame (first scan = identity).

Usage::

    python examples/kissicp_odometry.py /data/Rellis-3D/00000
    python examples/kissicp_odometry.py /data/sequences/00 --dataset semantic_kitti
    python examples/kissicp_odometry.py /data/Rellis-3D/00000 --voxel-size 0.5 --max-range 80

Tip: if you have per-point timestamps in the lidar channel (column 3), pass
``--deskew`` to compensate for motion distortion during each spin.
"""

import argparse

from apairo.dataset.goose.dataset import Goose3DDataset
from apairo.dataset.rellis.dataset import Rellis3DDataset
from apairo.dataset.semantic_kitti.dataset import SemanticKittiDataset

from apairo_preprocess.odometry.kissicp import KissICPOdometry

DATASETS = {
    "goose": Goose3DDataset,
    "rellis": Rellis3DDataset,
    "semantic_kitti": SemanticKittiDataset,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seq_dir", help="Sequence directory.")
    p.add_argument("--dataset", choices=DATASETS, default="rellis")
    p.add_argument("--voxel-size", type=float, default=1.0,
                   help="Map voxel size in metres (default: 1.0).")
    p.add_argument("--max-range", type=float, default=50.0,
                   help="Maximum point range kept before registration (default: 50.0 m).")
    p.add_argument("--min-range", type=float, default=1.0,
                   help="Minimum point range kept (default: 1.0 m).")
    p.add_argument("--deskew", action="store_true",
                   help="Enable motion deskewing (requires per-point timestamps in column 3).")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    preprocessor = KissICPOdometry(
        max_range=args.max_range,
        min_range=args.min_range,
        voxel_size=args.voxel_size,
        deskew=args.deskew,
    )
    dataset_cls = DATASETS[args.dataset]

    print(f"Dataset  : {dataset_cls.__name__}")
    print(f"Sequence : {args.seq_dir}")
    print(f"Config   : voxel_size={args.voxel_size} m, range=[{args.min_range}, {args.max_range}] m, deskew={args.deskew}")

    dataset_cls.run_preprocess(preprocessor, args.seq_dir, overwrite=args.overwrite)
    print(f"Done — channel '{preprocessor.output_key}' written.")


if __name__ == "__main__":
    main()
