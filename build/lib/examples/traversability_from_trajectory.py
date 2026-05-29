"""Compute trajectory-based traversability ground truth on a GOOSE-3D sequence.

A point is labelled traversable (1) if it falls within the robot's footprint
along any future pose in the trajectory.  This requires poses to already be
computed — run ``kissicp_odometry.py`` first if ``kissicp_poses`` is not yet
available.

The script runs in two steps:
  1. Odometry  (skipped if ``kissicp_poses`` already exists, unless --overwrite)
  2. Traversability labelling from the resulting trajectory

Usage::

    python examples/traversability_from_trajectory.py /data/goose/seq_001
    python examples/traversability_from_trajectory.py /data/goose/seq_001 \\
        --robot-radius 0.6 --height-min -0.4 --height-max 0.6 --overwrite
"""

import argparse
from pathlib import Path

import numpy as np
from apairo.dataset.goose.dataset import Goose3DDataset

from apairo_preprocess.odometry.kissicp import KissICPOdometry
from apairo_preprocess.traversability.from_trajectory import TraversabilityFromTrajectory


def _poses_exist(seq_dir: Path) -> bool:
    return any((seq_dir / "kissicp_poses").glob("*.npy"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seq_dir", help="GOOSE-3D sequence directory.")
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
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute both odometry and traversability even if they exist.")
    args = p.parse_args()

    seq_dir = Path(args.seq_dir)

    # --- Step 1: odometry ---
    if _poses_exist(seq_dir) and not args.overwrite:
        print("Step 1/2  kissicp_poses already present, skipping.")
    else:
        print("Step 1/2  Running KISS-ICP odometry …")
        Goose3DDataset.run_preprocess(
            KissICPOdometry(voxel_size=args.voxel_size),
            seq_dir,
            overwrite=args.overwrite,
        )

    # --- Step 2: traversability ---
    print("Step 2/2  Loading poses …")
    ds = Goose3DDataset(seq_dir, keys=["kissicp_poses"])
    poses = np.stack([ds[i].data["kissicp_poses"] for i in range(len(ds))])
    print(f"          {len(poses)} poses loaded.")

    print("Step 2/2  Computing traversability ground truth …")
    Goose3DDataset.run_preprocess(
        TraversabilityFromTrajectory(
            poses,
            robot_radius=args.robot_radius,
            height_min=args.height_min,
            height_max=args.height_max,
            forward_window=args.forward_window,
        ),
        seq_dir,
        overwrite=args.overwrite,
    )
    print(f"Done — channel 'trav_gt' written.")


if __name__ == "__main__":
    main()
