# Changelog

## 0.3.0

### Added

- `TraversabilityFromTrajectory`: oriented footprint (`robot_length`,
  `robot_width`), heading read from each pose.
- `TraversabilityFromTrajectory(pose_height=...)`: height of the pose frame
  above the ground. With lidar odometry the pose is the sensor's; without this
  the height window sat around the sensor and no ground point was ever
  labelled traversable.
- `VoxelisePointCloud(seed=...)` makes `reduction="random"` reproducible.
- `KissICPOdometry.start_sequence()` and `GICPOdometry.start_sequence()` reset
  the estimator, so that each sequence of a root starts at its own origin
  (called by apairo's runner from 0.11).

### Fixed

- `VoxelisePointCloud(reduction="first")` kept an arbitrary point of the voxel,
  not the first in input order.
- `VoxeliseLabels`: negative labels (`-1` for "unlabelled") raised or were
  aggregated wrongly; large ids (panoptic) allocated a table of their size.
- Voxelisers: NaN points produced a voxel at an arbitrary integer coordinate;
  they are dropped.
- `TraversabilityFromTrajectory`:
  - the circular footprint tested the height against the XY-nearest future
    pose only, so a later passage above the same spot (bridge, switchback)
    cancelled a real one; every future pose is tested, as the oriented
    footprint did;
  - `near_exclusion_radius > 0` raised with the documented norms;
  - one NaN point aborted the sequence; it is labelled 0;
  - an empty sequence raised.
- `LidarCameraProjection`: with lens distortion, points far outside the field
  of view folded back into the image (the radial polynomial is not monotonic)
  and could overwrite visible points in a mask. They are NaN.
- Projected pixels are bound-tested on the values that are stored.
- `BEVRasterisation`: bounds that are not a multiple of the resolution lost
  their last cells; points outside the bounds could land in the grid;
  `reduce="max"` kept the background for a cell of negative values and a NaN
  value poisoned its cell.
- `ImageMaskFromPointLabels` truncated fractional labels; it refuses them.
- `GroundSegmentationRANSAC`, `GroundSegmentationCSF`, `GroundHeightFromLabels`
  raised (CSF: crashed the interpreter) on an empty or tiny cloud.

## 0.2.0

First release on PyPI.
