"""Visual demo of the projection pipeline — no dataset required.

Builds a synthetic world (rolling ground, tree trunks, a wall), drives a
robot through it on a sinusoidal trajectory, then runs the real
preprocessors exactly as the dataset chain does:

  TraversabilityFromTrajectory -> trav_traj   (per point)
  LidarCameraProjection        -> lidar_uv    (per point)
  ImageMaskFromPointLabels     -> trav_mask   (per pixel)
  PointFeaturesFromImage       -> per-point colors sampled back from the image

Writes two figures: a 4-panel overview of the pipeline stages, and the
cloud-colored-from-image view — the visual calibration check to reproduce in
projector / apairo_rr on real data.

Requires matplotlib (not a package dependency)::

    pip install matplotlib
    python examples/projection_pipeline_demo.py --out-dir /tmp/projection_demo
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("this demo needs matplotlib: pip install matplotlib") from e

from apairo.core.config import CameraIntrinsics
from apairo.core.sample import Sample

from apairo_preprocess import (
    ImageMaskFromPointLabels,
    LidarCameraProjection,
    PointFeaturesFromImage,
    TraversabilityFromTrajectory,
)

rng = np.random.default_rng(7)


def build_world():
    """Rolling ground, four tree trunks, a wall on the left."""
    gx, gy = np.meshgrid(np.linspace(0, 32, 200), np.linspace(-9, 9, 120))
    ground = np.column_stack(
        [gx.ravel(), gy.ravel(), 0.15 * np.sin(0.4 * gx.ravel()) + 0.05 * gy.ravel()]
    )
    ground += rng.normal(0, 0.02, ground.shape)

    def trunk(x, y, r=0.35, h=3.0, n=700):
        a = rng.uniform(0, 2 * np.pi, n)
        return np.column_stack(
            [x + r * np.cos(a), y + r * np.sin(a), rng.uniform(0.0, h, n)]
        )

    obstacles = np.vstack(
        [
            trunk(9.0, -1.6),
            trunk(13.0, 2.2),
            trunk(17.5, -0.8),
            trunk(22.0, 3.0),
            np.column_stack(
                [
                    rng.uniform(6, 26, 2500),
                    rng.uniform(-6.4, -6.0, 2500),
                    rng.uniform(0, 2.2, 2500),
                ]
            ),
        ]
    )
    return np.vstack([ground, obstacles]).astype(np.float64)


def build_trajectory(n_frames=40):
    """Sinusoidal drive along +X, sensor 1.2 m above ground."""
    tx = np.linspace(0.0, 28.0, n_frames)
    ty = 1.8 * np.sin(0.22 * tx)
    heading = np.arctan2(np.gradient(ty), np.gradient(tx))
    poses = np.tile(np.eye(4), (n_frames, 1, 1))
    for i, (x, y, th) in enumerate(zip(tx, ty, heading)):
        c, s = np.cos(th), np.sin(th)
        poses[i, :3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        poses[i, :3, 3] = [x, y, 1.2]
    return poses


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="projection_demo",
                   help="Directory for the output figures (default: ./projection_demo).")
    p.add_argument("--frame", type=int, default=6,
                   help="Frame index to visualize (default: 6).")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world = build_world()
    poses = build_trajectory()
    n_frames, k = len(poses), args.frame
    tx, ty = poses[:, 0, 3], poses[:, 1, 3]

    def scan_at(i):
        """The world seen from pose i (sensor frame), range-limited like a lidar."""
        Tinv = np.linalg.inv(poses[i])
        pts = world @ Tinv[:3, :3].T + Tinv[:3, 3]
        return pts[np.linalg.norm(pts, axis=1) < 25.0].astype(np.float32)

    frames = [Sample(data={"lidar": scan_at(i), "poses": poses[i]}) for i in range(n_frames)]

    # -- stage 1: traversability from trajectory ---------------------------
    # Poses sit at sensor height (z = 1.2 m): the ground under the robot is
    # at dz ~ -1.2, so the height window is centred there.
    trav_p = TraversabilityFromTrajectory(robot_radius=0.9, height_min=-1.8, height_max=-0.6)
    labels = trav_p(iter(frames))[k]
    scan = frames[k].data["lidar"]

    # -- stage 2: lidar -> camera projection --------------------------------
    # Optical frame: x_cam = -y_lidar, y_cam = -z_lidar, z_cam = x_lidar.
    T_cam_from_lidar = np.array(
        [[0, -1, 0, 0.0], [0, 0, -1, -0.3], [1, 0, 0, 0.1], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    height, width = 360, 640
    cam = CameraIntrinsics(
        K=np.array([[420.0, 0, width / 2], [0, 420.0, height / 2], [0, 0, 1]]),
        width=width, height=height,
    )
    uv = LidarCameraProjection(intrinsics=cam, extrinsics=T_cam_from_lidar)(
        Sample(data={"lidar": scan})
    )
    valid = np.isfinite(uv[:, 0])

    # -- stage 3: mask in image space ----------------------------------------
    mask = ImageMaskFromPointLabels(
        image_size=(height, width), radius=3, occlusion_bin_px=12, occlusion_depth_margin=0.6
    )(Sample(data={"trav_traj": labels, "lidar_uv": uv}))

    # -- synthetic camera image (backdrop for the overlay) -------------------
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (168, 203, 255)  # sky
    cols = uv[valid, 0].astype(int)
    rows = uv[valid, 1].astype(int)
    height_above = np.clip(scan[valid][:, 2] + 1.2, 0, 3) / 3
    for r0, c0, h0 in zip(*(a[np.argsort(-uv[valid, 2])] for a in (rows, cols, height_above))):
        color = (
            np.array([90, 60, 30]) + 80 * h0  # obstacles: brown, lighter with height
            if h0 > 0.25
            else np.array([88, 132, 74])  # ground: grass green
        )
        img[max(r0 - 2, 0):r0 + 3, max(c0 - 2, 0):c0 + 3] = color.astype(np.uint8)

    # -- stage 4: image -> per-point colors ----------------------------------
    point_rgb = PointFeaturesFromImage(image_key="image")(
        Sample(data={"lidar_uv": uv, "image": img})
    )

    # ==================================================== overview figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"Projection pipeline on a synthetic scene — frame {k}/{n_frames}", fontsize=14)

    ax = axes[0, 0]
    w_scan = scan @ poses[k][:3, :3].T + poses[k][:3, 3]
    ax.scatter(w_scan[:, 0], w_scan[:, 1], s=1.2,
               c=np.where(labels == 1, "#2e9e4f", "#b0b0b0"), rasterized=True)
    ax.plot(tx, ty, "-", color="#1f4e9e", lw=1.8, label="trajectory")
    ax.plot(tx[k], ty[k], "o", color="#e33", ms=9, label="robot (frame k)")
    ax.set_title("1. TraversabilityFromTrajectory — green = trav_traj == 1 (top view)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 1]
    sc = ax.scatter(uv[valid, 0], uv[valid, 1], s=2.5, c=uv[valid, 2],
                    cmap="viridis", rasterized=True)
    ax.set_xlim(0, width); ax.set_ylim(height, 0)
    ax.set_title(f"2. LidarCameraProjection — lidar_uv ({valid.sum()} / {len(scan)} points in frustum)")
    ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
    plt.colorbar(sc, ax=ax, label="depth (m)", shrink=0.85)

    ax = axes[1, 0]
    mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    mask_rgb[mask == 255] = (238, 238, 238)
    mask_rgb[mask == 0] = (200, 60, 60)
    mask_rgb[mask == 1] = (46, 158, 79)
    ax.imshow(mask_rgb)
    cov = (mask != 255).mean() * 100
    ax.set_title(f"3. ImageMaskFromPointLabels — trav_mask (radius=3, occlusion on, {cov:.0f}% covered)")
    ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in ("#2e9e4f", "#c83c3c", "#eeeeee")]
    ax.legend(handles, ["traversable (1)", "not traversable (0)", "no data (255)"],
              loc="lower right", fontsize=8)

    ax = axes[1, 1]
    overlay = img.copy().astype(float)
    green = mask == 1
    overlay[green] = 0.45 * overlay[green] + 0.55 * np.array([40, 220, 80])
    ax.imshow(overlay.astype(np.uint8))
    ax.set_title("4. trav_mask overlaid on the camera image — the dataset channel")
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_dir / "pipeline_overview.png", dpi=110)

    # ============================================ calibration-check figure
    fig2, ax = plt.subplots(figsize=(9, 5.5))
    v3 = valid & (point_rgb.sum(axis=1) > 0)
    ax.scatter(w_scan[v3, 0], w_scan[v3, 2], s=3, c=point_rgb[v3] / 255.0, rasterized=True)
    ax.set_title("PointFeaturesFromImage — cloud colored from the image (side view)\n"
                 "the visual calibration check to run in projector / apairo_rr")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    fig2.tight_layout()
    fig2.savefig(out_dir / "point_features_check.png", dpi=110)

    print(f"frame {k}: {len(scan)} pts, {int(labels.sum())} traversable, "
          f"{int(valid.sum())} in frustum, mask coverage {cov:.1f}%")
    print(f"wrote {out_dir}/pipeline_overview.png and {out_dir}/point_features_check.png")


if __name__ == "__main__":
    main()
