"""End-to-end smoke test of the projection composition on a synthetic scene.

No dataset, no disk: a synthetic world (ground plane + obstacles) and a
straight trajectory drive the four preprocessors exactly as the example
chain does, asserting each stage semantically:

  TraversabilityFromTrajectory -> the corridor is traversable, obstacles not
  LidarCameraProjection        -> geometry lands where the pinhole says
  ImageMaskFromPointLabels     -> the mask paints both classes
  PointFeaturesFromImage       -> image values round-trip back to the points
"""

import numpy as np
import pytest
from apairo.core.config import CameraIntrinsics
from apairo.core.sample import Sample

from apairo_preprocess import (
    ImageMaskFromPointLabels,
    LidarCameraProjection,
    PointFeaturesFromImage,
    TraversabilityFromTrajectory,
)

GROUND, OBSTACLE = 0, 1
H, W = 240, 320
SENSOR_Z = 1.0
K_FRAME = 2

# Lidar (x fwd, y left, z up) -> camera optical (z fwd, x right, y down).
T_CAM_FROM_LIDAR = np.array(
    [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)
CAM = CameraIntrinsics(
    K=np.array([[200.0, 0, W / 2], [0, 200.0, H / 2], [0, 0, 1]]), width=W, height=H
)


def _scene():
    """World points, their class ids, and a straight trajectory along +X."""
    gx, gy = np.meshgrid(np.arange(0, 20, 0.25), np.arange(-5, 5, 0.25))
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])

    rng = np.random.default_rng(0)
    a = rng.uniform(0, 2 * np.pi, 400)
    trunk = np.column_stack(
        [8.0 + 0.3 * np.cos(a), 2.0 + 0.3 * np.sin(a), rng.uniform(0, 2.5, 400)]
    )
    wall = np.column_stack(
        [rng.uniform(2, 18, 800), np.full(800, -4.0), rng.uniform(0, 2.0, 800)]
    )

    world = np.vstack([ground, trunk, wall])
    classes = np.concatenate(
        [np.full(len(ground), GROUND), np.full(len(trunk) + len(wall), OBSTACLE)]
    )

    poses = np.tile(np.eye(4), (19, 1, 1))
    poses[:, 0, 3] = np.arange(19, dtype=np.float64)  # x = 0..18, one metre steps
    poses[:, 2, 3] = SENSOR_Z
    return world, classes, poses


@pytest.fixture(scope="module")
def pipeline():
    """Every stage, computed once: (world, classes, poses, labels, uv, mask)."""
    world, classes, poses = _scene()

    # The trajectory is axis-aligned and the world static: sensor-frame scans
    # are the world shifted by the pose, so labels stay row-aligned with world.
    frames = [
        Sample(data={"lidar": (world - p[:3, 3]).astype(np.float32), "poses": p})
        for p in poses
    ]
    labels_all = TraversabilityFromTrajectory(
        robot_radius=0.75, height_min=-1.5, height_max=-0.5
    )(iter(frames))
    labels = labels_all[K_FRAME]

    scan = frames[K_FRAME].data["lidar"]
    uv = LidarCameraProjection(intrinsics=CAM, extrinsics=T_CAM_FROM_LIDAR)(
        Sample(data={"lidar": scan})
    )
    mask = ImageMaskFromPointLabels(image_size=(H, W))(
        Sample(data={"trav_traj": labels, "lidar_uv": uv})
    )
    return world, classes, poses, labels, uv, mask


def test_corridor_is_traversable_and_obstacles_are_not(pipeline):
    world, classes, poses, labels, _, _ = pipeline
    robot_x = poses[K_FRAME, 0, 3]
    corridor = (
        (classes == GROUND)
        & (np.abs(world[:, 1]) < 0.5)
        & (world[:, 0] > robot_x + 1)
        & (world[:, 0] < 17)
    )
    assert labels[corridor].all(), "ground under the future trajectory must be 1"
    assert not labels[classes == OBSTACLE].any(), "obstacle points must be 0"
    off_path = (classes == GROUND) & (np.abs(world[:, 1]) > 2)
    assert not labels[off_path].any(), "ground far from the trajectory must be 0"


def test_projection_geometry(pipeline):
    world, _, poses, _, uv, _ = pipeline
    assert uv.shape == (len(world), 3) and uv.dtype == np.float32

    valid = np.isfinite(uv[:, 0])
    assert valid.any()
    assert (uv[valid, 0] >= 0).all() and (uv[valid, 0] < W).all()
    assert (uv[valid, 1] >= 0).all() and (uv[valid, 1] < H).all()
    assert (uv[valid, 2] > 0).all()

    behind = world[:, 0] < poses[K_FRAME, 0, 3]
    assert not valid[behind].any(), "points behind the camera must be NaN"

    # Ground sits below the sensor: it must project below the principal point,
    # and closer ground lower in the image than farther ground.
    ground_ahead = valid & (world[:, 2] == 0)
    assert (uv[ground_ahead, 1] > H / 2).all()
    v_by_depth = uv[ground_ahead][np.argsort(uv[ground_ahead, 2])][:, 1]
    assert v_by_depth[0] > v_by_depth[-1]


def test_mask_paints_both_classes(pipeline):
    *_, labels, uv, mask = pipeline
    assert mask.shape == (H, W) and mask.dtype == np.uint8
    assert (mask == 1).any() and (mask == 0).any() and (mask == 255).any()

    # The nearest valid point wins its pixel unconditionally.
    valid = np.isfinite(uv[:, 0])
    nearest = np.nanargmin(np.where(valid, uv[:, 2], np.nan))
    col, row = int(uv[nearest, 0]), int(uv[nearest, 1])
    assert mask[row, col] == labels[nearest]

    # Traversable paint stays in the lower image half (ground, below horizon).
    assert not (mask[: H // 2] == 1).any()


def test_image_values_round_trip_to_points(pipeline):
    world, classes, _, _, uv, _ = pipeline
    palette = np.array([[0, 0, 0], [60, 140, 60], [150, 90, 40]], dtype=np.uint8)

    # Paint the classes into an image with the mask preprocessor (1=ground,
    # 2=obstacle, 255=unknown), build an RGB image from it, then sample it
    # back onto the points: each point must recover its own class color,
    # except where a nearer point of the other class won the pixel.
    class_mask = ImageMaskFromPointLabels(image_size=(H, W))(
        Sample(data={"trav_traj": classes + 1, "lidar_uv": uv})
    )
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for value in (1, 2):
        img[class_mask == value] = palette[value]

    feats = PointFeaturesFromImage()(Sample(data={"lidar_uv": uv, "image": img}))
    assert feats.shape == (len(world), 3) and feats.dtype == np.uint8

    valid = np.isfinite(uv[:, 0])
    expected = palette[classes[valid] + 1]
    match = (feats[valid] == expected).all(axis=1).mean()
    assert match > 0.8, f"only {match:.0%} of points recovered their class color"
