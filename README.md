# apairo-preprocess

Preprocessing pipelines for [apairo](https://github.com/apairo-robotics/apairo) datasets — LiDAR odometry and traversability ground truth generation.

---

## Installation

```bash
pip install git+https://github.com/apairo-robotics/apairo_preprocess.git
```

Optional dependencies:

```bash
pip install kiss-icp   # for KissICPOdometry
pip install open3d     # for GICPOdometry
```

Requires Python ≥ 3.11.

---

## Preprocessors

### Odometry

| Class | Output channel | Backend | Output |
|---|---|---|---|
| `KissICPOdometry` | `kissicp_poses` | [KISS-ICP](https://github.com/PRBonn/kiss-icp) | `(4, 4)` float64 pose per scan |
| `GICPOdometry` | `gicp_poses` | Open3D GICP | `(4, 4)` float64 pose per scan |

### Traversability

| Class | Output channel | Method |
|---|---|---|
| `TraversabilityFromLabels` | `trav_label` | Maps semantic class IDs to binary traversable/non-traversable |
| `TraversabilityFromTrajectory` | `trav_gt` | Labels points inside the robot's forward footprint along the trajectory |

---

## Quickstart

### KISS-ICP odometry

```python
from apairo.dataset.rellis import Rellis3DDataset
from apairo_preprocess import KissICPOdometry

Rellis3DDataset.run_preprocess(
    KissICPOdometry(voxel_size=1.0),
    "/data/Rellis-3D/00000",
)
# writes kissicp_poses/000000.npy, 000001.npy, ...
```

### Traversability from semantic labels

```python
from apairo.dataset.rellis import Rellis3DDataset
from apairo_preprocess import TraversabilityFromLabels

# Default traversable IDs for RELLIS-3D: {dirt, grass, asphalt, concrete, puddle, mud}
Rellis3DDataset.run_preprocess(
    TraversabilityFromLabels(),
    "/data/Rellis-3D/00000",
)
# writes trav_label/000000.npy, ...  (uint8: 1=traversable, 0=not)
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
import numpy as np
from apairo.dataset.goose import Goose3DDataset
from apairo_preprocess import KissICPOdometry, TraversabilityFromTrajectory

# Step 1 — odometry
Goose3DDataset.run_preprocess(
    KissICPOdometry(voxel_size=1.0),
    "/data/goose/seq_001",
)

# Step 2 — load poses and compute traversability ground truth
ds = Goose3DDataset("/data/goose/seq_001", keys=["kissicp_poses"])
poses = np.stack([ds[i].data["kissicp_poses"] for i in range(len(ds))])

Goose3DDataset.run_preprocess(
    TraversabilityFromTrajectory(poses, robot_radius=0.75, height_min=-0.3, height_max=0.5),
    "/data/goose/seq_001",
)
# writes trav_gt/000000.npy, ...  (uint8: 1=traversable, 0=not)
```

---

## Examples

Ready-to-run scripts in [`examples/`](examples/):

```bash
# KISS-ICP odometry on any supported dataset
python examples/kissicp_odometry.py /data/Rellis-3D/00000 --dataset rellis

# Traversability from semantic labels
python examples/traversability_from_labels.py /data/Rellis-3D/00000

# Traversability ground truth from trajectory (runs odometry first if needed)
python examples/traversability_from_trajectory.py /data/goose/seq_001
```

---

## License

MIT
