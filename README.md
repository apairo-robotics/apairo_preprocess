# apairo-preprocess

Preprocessing pipelines for [apairo](https://github.com/apairo-robotics/apairo) datasets — LiDAR odometry, ground segmentation, and traversability ground truth generation.

---

## Installation

```bash
pip install git+https://github.com/apairo-robotics/apairo_preprocess.git
```

### Optional dependencies

| Preprocessor | Dependency | Install |
|---|---|---|
| `KissICPOdometry` | KISS-ICP | `pip install kiss-icp` |
| `GICPOdometry` | Open3D | `pip install open3d` |
| `GroundSegmentationCSF` / `GroundHeightFromLabels` (CSF backend) | CSF | `pip install CSF` |
| `TerraSegGroundSegmentation` | TerraSeg | `pip install terraseg` |

Requires Python ≥ 3.11.

---

## Preprocessors

### Odometry

| Class | Output channel | Backend | Output |
|---|---|---|---|
| `KissICPOdometry` | `kissicp_poses` | [KISS-ICP](https://github.com/PRBonn/kiss-icp) | `(4, 4)` float64 pose per scan |
| `GICPOdometry` | `gicp_poses` | Open3D GICP | `(4, 4)` float64 pose per scan |

### Ground segmentation

Binary ground/non-ground labels (0 = ground, 1 = non-ground). All three algorithms share the same label convention and can be compared directly.

| Class | Output channel | Method | Extra dep |
|---|---|---|---|
| `GroundSegmentationCSF` | `ground_csf` | Cloth Simulation Filter — accurate on uneven terrain | `CSF` |
| `GroundSegmentationRANSAC` | `ground_ransac` | RANSAC plane fitting — fast, assumes flat ground | — |
| `TerraSegGroundSegmentation` | `terraseg_ground` | Self-supervised ML model (TerraSeg) | `terraseg` |

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

---

## Quickstart

### Ground segmentation

```python
from apairo.dataset.rellis import Rellis3DDataset
from apairo_preprocess import GroundSegmentationCSF, GroundSegmentationRANSAC, TerraSegGroundSegmentation

dataset_dir = "/data/Rellis-3D/00000"

# Classical methods (no GPU required)
Rellis3DDataset.run_preprocess(GroundSegmentationRANSAC(), dataset_dir)
Rellis3DDataset.run_preprocess(GroundSegmentationCSF(), dataset_dir)   # requires: pip install CSF

# ML-based (requires: pip install terraseg)
Rellis3DDataset.run_preprocess(TerraSegGroundSegmentation(variant="S"), dataset_dir)

# writes ground_ransac/, ground_csf/, terraseg_ground/  (uint8: 0=ground, 1=non-ground)
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

`GroundHeightFromLabels` accepts any ground key — swap `"ground_csf"` for `"ground_ransac"` or `"terraseg_ground"` to change the backend without re-running CSF.

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

### Traversability ground truth from trajectory

Requires poses to be computed first (e.g. with `KissICPOdometry`).

```python
from apairo.dataset.goose import Goose3DDataset
from apairo_preprocess import KissICPOdometry, TraversabilityFromTrajectory

Goose3DDataset.run_preprocess(KissICPOdometry(voxel_size=1.0), "/data/goose/seq_001")
Goose3DDataset.run_preprocess(
    TraversabilityFromTrajectory(poses_key="kissicp_poses", robot_radius=0.75),
    "/data/goose/seq_001",
)
# writes trav_traj/  (uint8: 1=traversable, 0=not)
```

---

## Examples

Ready-to-run scripts in [`examples/`](examples/):

```bash
python examples/kissicp_odometry.py /data/Rellis-3D/00000 --dataset rellis
python examples/traversability_from_labels.py /data/Rellis-3D/00000
python examples/traversability_from_trajectory.py /data/goose/seq_001
```

---

## License

MIT
