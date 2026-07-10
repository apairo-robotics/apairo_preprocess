"""Traversability ground truth in image space — the full projection chain.

Composes four atomic preprocessors on a TartanDrive sequence:

  1. ``KissICPOdometry``              lidar          -> kissicp_poses
  2. ``TraversabilityFromTrajectory`` lidar + poses  -> trav_traj   (per point)
  3. ``LidarCameraProjection``        lidar          -> lidar_uv    (per point)
  4. ``ImageMaskFromPointLabels``     trav_traj + uv -> trav_mask   (per pixel)
  5. ``PointFeaturesFromImage``       uv + image     -> lidar_rgb   (--rgb)

Step 5 colours the cloud with camera RGB — load ``lidar`` + ``lidar_rgb`` in a
3-D viewer (projector, apairo_rr) to visually verify calibration and
projection before trusting ``trav_mask``.

Steps 1 and 3 stream a single channel and run through ``run_preprocess``
directly.  Steps 2, 4 and 5 need several channels per frame, which
``run_preprocess`` cannot assemble on an *asynchronous* dataset (it iterates
the interleaved event timeline — see apairo ``IDEAS.md``, *Multi-channel
preprocess on asynchronous datasets*); they run over a ``synchronize()`` view
and are persisted with ``ChannelWriter``.

Camera parameters come from the dataset's calibration: the intrinsics from
the ``cameras:`` section of ``.apairo/calibration.yaml``
(``cal.get_intrinsics(camera_frame)``), the extrinsic from the transform
graph (``cal.get_tf(lidar_frame, camera_frame)``).

Usage::

    python examples/traversability_image_mask.py /data/tartan/seq \\
        --lidar-frame velodyne --camera-frame multisense_left
    python examples/traversability_image_mask.py /data/tartan/seq \\
        --lidar-frame velodyne --camera-frame multisense_left \\
        --rgb --splat-radius 2 --occlusion-bin 8 --overwrite
"""

import argparse
from pathlib import Path

import numpy as np

from apairo import ChannelWriter
from apairo.dataset.tartan_kitti import TartanKittiDataset

from apairo_preprocess import (
    ImageMaskFromPointLabels,
    KissICPOdometry,
    LidarCameraProjection,
    PointFeaturesFromImage,
    TraversabilityFromTrajectory,
)


def write_channel(seq_dir, key, rows, timestamps, *, timestamps_from, sources, overwrite):
    """Persist per-frame rows as a channel; skip if it already exists."""
    if (Path(seq_dir) / key).exists():
        if not overwrite:
            print(f"  '{key}' exists, skipping (use --overwrite)")
            return
        TartanKittiDataset.remove_channel(seq_dir, key, data=True)
    with ChannelWriter(
        seq_dir, key, loader="npys", timestamps_from=timestamps_from, sources=sources
    ) as w:
        for i, (row, ts) in enumerate(zip(rows, timestamps)):
            w.add(np.asarray(row), stem=f"{i:06d}", timestamp=ts)
    print(f"  '{key}' written ({len(timestamps)} frames)")


def run_preprocess_guarded(preprocessor, seq_dir, overwrite):
    try:
        TartanKittiDataset.run_preprocess(preprocessor, seq_dir, overwrite=overwrite)
        print(f"  '{preprocessor.output_key}' written")
    except FileExistsError:
        print(f"  '{preprocessor.output_key}' exists, skipping (use --overwrite)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seq_dir", help="TartanDrive sequence directory.")
    p.add_argument("--lidar-frame", required=True,
                   help="Lidar tf frame in the calibration (e.g. velodyne).")
    p.add_argument("--camera-frame", required=True,
                   help="Camera tf frame — must have a cameras: entry in calibration.yaml.")
    p.add_argument("--lidar-key", default="velodyne_0",
                   help="Input lidar channel (default: velodyne_0).")
    p.add_argument("--image-key", default="image_left_color",
                   help="Input image channel for --rgb (default: image_left_color).")
    p.add_argument("--poses-key", default="kissicp_poses",
                   help="Poses channel from the odometry step (default: kissicp_poses).")
    p.add_argument("--robot-radius", type=float, default=0.75,
                   help="Robot half-width in XY (metres, default: 0.75).")
    p.add_argument("--splat-radius", type=int, default=2,
                   help="Mask splat radius in pixels (default: 2).")
    p.add_argument("--occlusion-bin", type=int, default=8,
                   help="Occlusion z-buffer bin size in pixels; 0 disables (default: 8).")
    p.add_argument("--tolerance", type=float, default=0.1,
                   help="synchronize() tolerance in seconds (default: 0.1).")
    p.add_argument("--rgb", action="store_true",
                   help="Also write per-point camera RGB (lidar_rgb) for viewer checks.")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute channels that already exist.")
    args = p.parse_args()
    seq = args.seq_dir

    cal = TartanKittiDataset(seq, keys=[args.lidar_key]).calibration
    cam = cal.get_intrinsics(args.camera_frame)
    T = cal.get_tf(args.lidar_frame, args.camera_frame)
    size = (cam.height, cam.width)

    print("Step 1/5  KISS-ICP odometry")
    run_preprocess_guarded(
        KissICPOdometry(lidar_key=args.lidar_key, output_key=args.poses_key),
        seq, args.overwrite,
    )

    print("Step 2/5  Traversability from trajectory")
    sync = TartanKittiDataset(seq, keys=[args.lidar_key, args.poses_key]).synchronize(
        reference=args.lidar_key, tolerance=args.tolerance
    )
    frames = [sync[i] for i in range(len(sync))]
    trav = TraversabilityFromTrajectory(
        lidar_key=args.lidar_key, poses_key=args.poses_key, robot_radius=args.robot_radius
    )(iter(frames))
    write_channel(
        seq, "trav_traj", trav, [s.timestamp for s in frames],
        timestamps_from=args.lidar_key, sources=[args.lidar_key, args.poses_key],
        overwrite=args.overwrite,
    )

    print("Step 3/5  Lidar -> camera projection")
    run_preprocess_guarded(
        LidarCameraProjection(
            intrinsics=cam, extrinsics=T,
            lidar_key=args.lidar_key, output_key="lidar_uv",
        ),
        seq, args.overwrite,
    )

    print("Step 4/5  Traversability mask in image space")
    mask_p = ImageMaskFromPointLabels(
        image_size=size,
        radius=args.splat_radius,
        occlusion_bin_px=args.occlusion_bin or None,
    )
    sync = TartanKittiDataset(seq, keys=["trav_traj", "lidar_uv"]).synchronize(
        reference="trav_traj", tolerance=args.tolerance
    )
    frames = [sync[i] for i in range(len(sync))]
    write_channel(
        seq, "trav_mask", (mask_p(s) for s in frames), [s.timestamp for s in frames],
        timestamps_from=args.lidar_key, sources=["trav_traj", "lidar_uv"],
        overwrite=args.overwrite,
    )

    if args.rgb:
        print("Step 5/5  Per-point camera RGB (viewer verification)")
        rgb_p = PointFeaturesFromImage(image_key=args.image_key, output_key="lidar_rgb")
        sync = TartanKittiDataset(seq, keys=["lidar_uv", args.image_key]).synchronize(
            reference="lidar_uv", tolerance=args.tolerance
        )
        frames = [sync[i] for i in range(len(sync))]
        write_channel(
            seq, "lidar_rgb", (rgb_p(s) for s in frames), [s.timestamp for s in frames],
            timestamps_from=args.lidar_key, sources=["lidar_uv", args.image_key],
            overwrite=args.overwrite,
        )
    else:
        print("Step 5/5  skipped (pass --rgb to write per-point camera RGB)")

    print("Done.")


if __name__ == "__main__":
    main()
