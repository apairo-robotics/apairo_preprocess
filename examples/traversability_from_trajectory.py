"""Compute trajectory-based traversability ground truth on a GOOSE-3D sequence.

A point is labelled traversable (1) if it falls within the robot's footprint
along any future pose in the trajectory.  This requires poses to already be
computed — run ``kissicp_odometry.py`` first if ``kissicp_poses`` is not yet
available.

The script runs in two steps:
  1. Odometry  (skipped if the poses channel already exists, unless --overwrite)
  2. Traversability labelling from the resulting trajectory

Usage::

    python examples/traversability_from_trajectory.py /data/goose/seq_001
    python examples/traversability_from_trajectory.py /data/goose/seq_001 \\
        --poses-key kissicp_poses --robot-radius 0.6 --overwrite
"""

import argparse
from pathlib import Path

from apairo.dataset.goose.dataset import Goose3DDataset

from apairo_preprocess.odometry.kissicp import KissICPOdometry
from apairo_preprocess.traversability.from_trajectory import TraversabilityFromTrajectory


def _channel_exists(seq_dir: Path, key: str) -> bool:
    return any((seq_dir / key).glob("*.npy"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seq_dir", help="GOOSE-3D sequence directory.")
    p.add_argument("--lidar-key", default="lidar",
                   help="Input LiDAR channel (default: lidar).")
    p.add_argument("--poses-key", default="kissicp_poses",
                   help="Poses channel produced by the odometry step (default: kissicp_poses).")
    p.add_argument("--robot-radius", type=float, default=0.75,
                   help="Robot half-width in XY (metres, default: 0.75).")
    p.add_argument("--height-min", type=float, default=-0.3,
                   help="Min point height relative to robot position (default: -0.3 m).")
    p.add_argument("--height-max", type=float, default=0.5,
                   help="Max point height relative to robot position (default: 0.5 m).")
    p.add_argument("--forward-window", type=int, default=None,
                   help="Max number of future poses to look ahead. Default: full sequence.")
    p.add_argument("--voxel-size", type=float, default=1.0,
                   help="KISS-ICP voxel size for the odometry step (default: 1.0 m).")
    p.add_argument("--output-key", default=None,
                   help="Output channel name (default: trav_traj).")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute both odometry and traversability even if they exist.")
    args = p.parse_args()

    seq_dir = Path(args.seq_dir)

    # --- Step 1: odometry ---
    if _channel_exists(seq_dir, args.poses_key) and not args.overwrite:
        print(f"Step 1/2  '{args.poses_key}' already present, skipping.")
    else:
        print("Step 1/2  Running KISS-ICP odometry …")
        Goose3DDataset.run_preprocess(
            KissICPOdometry(
                lidar_key=args.lidar_key,
                voxel_size=args.voxel_size,
                output_key=args.poses_key,
            ),
            seq_dir,
            overwrite=args.overwrite,
        )

    # --- Step 2: traversability ---
    print("Step 2/2  Computing traversability ground truth …")
    Goose3DDataset.run_preprocess(
        TraversabilityFromTrajectory(
            lidar_key=args.lidar_key,
            poses_key=args.poses_key,
            robot_radius=args.robot_radius,
            height_min=args.height_min,
            height_max=args.height_max,
            forward_window=args.forward_window,
            output_key=args.output_key,
        ),
        seq_dir,
        overwrite=args.overwrite,
    )
    print("Done.")


if __name__ == "__main__":
    main()
