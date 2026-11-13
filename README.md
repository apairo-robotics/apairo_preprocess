# apairo-preprocess

Preprocessing pipelines for [apairo](https://github.com/apairo-robotics/apairo) datasets — LiDAR odometry, ground segmentation, camera projection, and traversability ground truth generation.

---

## Installation

```bash
pip install apairo-preprocess
```

It brings `apairo` (≥ 0.9) and `apairo-transform` (≥ 0.3).

### Optional dependencies

| Preprocessor | Dependency | Install |
|---|---|---|
| `KissICPOdometry` | KISS-ICP | `pip install kiss-icp` |
| `GICPOdometry` | Open3D | `pip install open3d` |
| `GroundSegmentationCSF` / `GroundHeightFromLabels` (CSF backend) | CSF | `pip install cloth-simulation-filter` |

Requires Python ≥ 3.11.

---

## Preprocessors

### Poses

Labels computed from a trajectory are only as good as the trajectory. When a
dataset ships its own poses (an INS, a GNSS-INS, the trajectory of its
release), use them: turning them into one pose per scan is interpolation and
composition, with fixed rules and no parameter to tune per sensor.

| Function / class | Output | Source |
|---|---|---|
| `reference_poses` / `ReferencePoses` | `(N, 4, 4)` float64 poses + `(N,)` bool validity, on the scan clock | pose rows delivered with the dataset |
| `pose_validity` | `(N,)` bool | physical bounds: top speed, turn rate |

A scan gets a pose only when two rows bracket its time, at most `max_gap`
apart; nothing is extrapolated; a step faster than `max_speed` invalidates
both of its scans; an invalid scan's pose is NaN. The rows and the scans are on
two clocks, so the function is called directly and its outputs written with
`apairo.ChannelWriter` (see the quickstart).

Lidar odometry estimates a trajectory instead. It returns a pose even where it
has lost track, with no signal of it, and its drift depends on the terrain:

| Class | Output channel | Backend | Output |
|---|---|---|---|
| `KissICPOdometry` | `kissicp_poses` | [KISS-ICP](https://github.com/PRBonn/kiss-icp) | `(4, 4)` float64 pose per scan |
| `GICPOdometry` | `gicp_poses` | Open3D GICP | `(4, 4)` float64 pose per scan |

Measured on two TartanDrive 2.0 recordings against the vehicle's INS and the
release's own SLAM trajectory: the future path seen from a scan agrees within
5-10 cm (median) at 10 m ahead for every source, but sources drift apart with
the look-ahead, to 1.2-1.5 m (p95) at 40 m. In vegetation, KISS-ICP was the one
that drifted. Use lidar odometry for recordings without delivered poses, keep
`forward_distance` short (15 m), and check it against another source when one
exists.

### Ground segmentation

Binary ground/non-ground labels (0 = ground, 1 = non-ground). All three algorithms share the same label convention and can be compared directly.

| Class | Output channel | Method | Extra dep |
|---|---|---|---|
| `GroundSegmentationCSF` | `ground_csf` | Cloth Simulation Filter — accurate on uneven terrain | `CSF` |
| `GroundSegmentationRANSAC` | `ground_ransac` | RANSAC plane fitting — fast, assumes flat ground | — |
| `GroundSegmentationFromLabels` | `ground_labels` | Maps existing semantic class IDs to ground/non-ground | — |

### Priors

Scalar per-point features used as traversability priors. Depend on previously computed channels.

| Class | Output channel | Input channels | Output |
|---|---|---|---|
| `GroundHeightFromLabels` | `ground_height` | any ground segmentation + `voxelised` | float32 height above nearest ground point (m) |
| `TrajectoryDistance` | `trajectory_distance` | `voxelised` + `poses` | float32 distance to nearest trajectory waypoint (m) |

### Traversability

Binary traversability labels (1 = traversable, 0 = not).

| Class | Output channel | Method |
|---|---|---|
| `TraversabilityFromLabels` | `trav_label` | Maps semantic class IDs to binary traversable/non-traversable |
| `TraversabilityFromTrajectory` | `trav_traj` | Labels points inside the robot's forward footprint along the trajectory |

### Camera projection

Atomic bridges between the point cloud and image spaces.  All outputs are
**row-aligned** with the lidar scan, so any per-point channel composes with
the projection by row index.

| Class | Output channel | Input channels | Output |
|---|---|---|---|
| `LidarCameraProjection` | `lidar_uv` | lidar only (extrinsics/intrinsics are static) | `(N, 3)` float32 `[u, v, depth]`, NaN outside the frustum |
| `PointFeaturesFromImage` | `point_features` | `lidar_uv` + image | `(N, C)` image values per point (e.g. RGB) — image → cloud |
| `ImageMaskFromPointLabels` | `trav_mask` | `trav_traj` + `lidar_uv` | `(H, W)` uint8 label mask, 255 = no data — cloud → image |

Pixel conflicts resolve to the nearest point (depth ordering); an optional
occlusion filter (`occlusion_bin_px`) drops points hidden behind nearer
returns so ground behind an obstacle does not bleed onto its pixels.
Both halves of the projection come from the dataset calibration: the
extrinsic via `ds.calibration.get_tf(lidar_frame, camera_frame)`, the
intrinsics via `ds.calibration.get_intrinsics(camera_frame)`.

**Asynchronous datasets:** `LidarCameraProjection` streams a single channel
and runs anywhere.  The two multi-channel preprocessors run via
`run_preprocess` on synchronous (profiled) datasets only; on an async dataset
(TartanDrive, raw rigs) run them over a `synchronize()` view and persist with
`ChannelWriter` — see
[`examples/traversability_image_mask.py`](examples/traversability_image_mask.py).

---

## Quickstart

### Ground segmentation

```python
from apairo.dataset.rellis import Rellis3DDataset
from apairo_preprocess import GroundSegmentationCSF, GroundSegmentationFromLabels, GroundSegmentationRANSAC

dataset_dir = "/data/Rellis-3D/00000"

Rellis3DDataset.run_preprocess(GroundSegmentationRANSAC(), dataset_dir)
Rellis3DDataset.run_preprocess(GroundSegmentationCSF(), dataset_dir)   # requires: pip install CSF

# From existing semantic labels (default ground IDs are for RELLIS-3D)
Rellis3DDataset.run_preprocess(GroundSegmentationFromLabels(), dataset_dir)

# writes ground_ransac/, ground_csf/, ground_labels/  (uint8: 0=ground, 1=non-ground)
```

### Height above ground (prior)

Ground segmentation must be computed first.

```python
from apairo_preprocess import GroundSegmentationCSF, GroundHeightFromLabels

Rellis3DDataset.run_preprocess(GroundSegmentationCSF(), dataset_dir)
Rellis3DDataset.run_preprocess(
    GroundHeightFromLabels(ground_key="ground_csf"),
    dataset_dir,
)
# writes ground_height/  (float32, metres above nearest ground point)
```

`GroundHeightFromLabels` accepts any ground key — swap `"ground_csf"` for `"ground_ransac"` or `"ground_labels"` to change the backend without re-running CSF.

### Traversability from semantic labels

```python
from apairo_preprocess import TraversabilityFromLabels

# Default traversable IDs for RELLIS-3D: {dirt, grass, asphalt, concrete, puddle, mud}
Rellis3DDataset.run_preprocess(TraversabilityFromLabels(), dataset_dir)
# writes trav_label/  (uint8: 1=traversable, 0=not)
```

Custom IDs for SemanticKITTI:

```python
from apairo.dataset.semantic_kitti import SemanticKittiDataset
from apairo_preprocess import TraversabilityFromLabels

SemanticKittiDataset.run_preprocess(
    TraversabilityFromLabels(traversable_ids=frozenset({40, 44, 48, 49, 60, 72})),
    "/data/sequences/00",
)
```

### Poses delivered with a dataset

```python
import numpy as np
from apairo import ChannelWriter
from apairo_preprocess import ReferencePoses

rows = np.load(f"{seq}/gps_odom/odometry.npy")        # [x y z qx qy qz qw, twist]
row_times = np.loadtxt(f"{seq}/gps_odom/timestamps.txt")
scan_times = np.loadtxt(f"{seq}/velodyne_0/timestamps.txt")

ins = ReferencePoses(clock_offset=-0.05, max_gap=0.1, max_speed=20.0)
poses, valid = ins(rows, row_times, scan_times)        # relative to the first scan

with ChannelWriter(seq, "ins_poses", timestamps_from="velodyne_0", sources=["gps_odom"]) as w:
    for i, t in enumerate(scan_times):
        w.add(poses[i], stem=f"{i:06d}", timestamp=t)
with ChannelWriter(seq, "ins_valid", timestamps_from="velodyne_0", sources=["gps_odom"]) as w:
    for i, t in enumerate(scan_times):
        w.add(np.asarray(valid[i]), stem=f"{i:06d}", timestamp=t)
```

`clock_offset` is the pose clock minus the lidar's, measured once per rig;
`max_speed` is the robot's top speed. Pass the validity to the labeller as
`valid_key`: an invalid scan then cuts the look-ahead, and with
`unknown_label=255` its points read "unknown" rather than "not traversable".

### Traversability ground truth from trajectory

Requires poses first: delivered ones (above), or lidar odometry
(`KissICPOdometry`) for a recording without any.

Lidar odometry gives the pose of the *sensor*, not of the ground under the
robot: pass the lidar's mounting height as `pose_height`, or every ground point
lies below the height window and nothing is labelled traversable.

```python
from apairo.dataset.goose import Goose3DDataset
from apairo_preprocess import KissICPOdometry, TraversabilityFromTrajectory

Goose3DDataset.run_preprocess(KissICPOdometry(voxel_size=1.0), "/data/goose/seq_001")
Goose3DDataset.run_preprocess(
    TraversabilityFromTrajectory(
        poses_key="kissicp_poses", robot_radius=0.75, pose_height=1.2
    ),
    "/data/goose/seq_001",
)
# writes trav_traj/  (uint8: 1=traversable, 0=not)
```

With delivered poses and their validity:

```python
TraversabilityFromTrajectory(
    poses_key="ins_poses", valid_key="ins_valid", unknown_label=255,
    robot_radius=0.8, pose_height=1.77, forward_distance=15.0,
)
# uint8: 1=traversable, 0=not, 255=unknown (scan without a trusted pose)
```

### Traversability mask in image space

Projects the trajectory ground truth into the camera: `trav_traj` labels the
points, `lidar_uv` places them in the image, `ImageMaskFromPointLabels`
paints the mask.

```python
from apairo_preprocess import ImageMaskFromPointLabels, LidarCameraProjection

cal = Rellis3DDataset(dataset_dir, keys=["lidar"]).calibration
cam = cal.get_intrinsics("camera")               # K, distortion, image size
T = cal.get_tf("lidar", "camera")                # lidar -> camera optical frame

Rellis3DDataset.run_preprocess(
    LidarCameraProjection(intrinsics=cam, extrinsics=T),
    dataset_dir,
)
Rellis3DDataset.run_preprocess(
    ImageMaskFromPointLabels(image_size=(cam.height, cam.width), radius=2, occlusion_bin_px=8),
    dataset_dir,
)
# writes lidar_uv/  (float32 [u, v, depth] per point)
#    and trav_mask/ (uint8 per pixel: 1=traversable, 0=not, 255=no data)
```

To sanity-check calibration and projection first, colour the cloud with
camera RGB and inspect it in a 3-D viewer (projector, apairo_rr):

```python
from apairo_preprocess import PointFeaturesFromImage

Rellis3DDataset.run_preprocess(
    PointFeaturesFromImage(image_key="image", output_key="lidar_rgb"),
    dataset_dir,
)
```

---

## Examples

Ready-to-run scripts in [`examples/`](examples/):

```bash
python examples/kissicp_odometry.py /data/Rellis-3D/00000 --dataset rellis
python examples/traversability_from_labels.py /data/Rellis-3D/00000
python examples/traversability_from_trajectory.py /data/goose/seq_001
python examples/traversability_image_mask.py /data/tartan/seq --lidar-frame velodyne --camera-frame multisense_left --rgb

# no dataset needed — synthetic scene, renders every pipeline stage (needs matplotlib)
python examples/projection_pipeline_demo.py --out-dir /tmp/projection_demo
```

---

## License

MIT
