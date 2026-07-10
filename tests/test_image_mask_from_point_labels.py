import numpy as np
import pytest
from apairo.core.sample import Sample

from apairo_preprocess.projection.image_mask import ImageMaskFromPointLabels


SIZE = (10, 12)  # (height, width)


def _sample(labels, uv, labels_key="trav_traj", projection_key="lidar_uv"):
    return Sample(
        data={
            labels_key: np.asarray(labels, dtype=np.uint8),
            projection_key: np.asarray(uv, dtype=np.float32),
        }
    )


def _mask(labels, uv, **kwargs):
    proc = ImageMaskFromPointLabels(image_size=SIZE, **kwargs)
    return proc(_sample(labels, uv))


# ------------------------------------------------------------------
# Constructor / API
# ------------------------------------------------------------------

def test_default_keys():
    proc = ImageMaskFromPointLabels(image_size=SIZE)
    assert proc.input_keys == ["trav_traj", "lidar_uv"]
    assert proc.timestamps_from == "trav_traj"
    assert proc.output_key == "trav_mask"


def test_custom_keys():
    proc = ImageMaskFromPointLabels(
        image_size=SIZE,
        labels_key="trav_label",
        projection_key="lidar_uv_left",
        output_key="trav_mask_left",
    )
    assert proc.input_keys == ["trav_label", "lidar_uv_left"]
    assert proc.timestamps_from == "trav_label"
    assert proc.sources == ["trav_label", "lidar_uv_left"]
    assert proc.output_key == "trav_mask_left"


def test_bad_parameters_rejected():
    with pytest.raises(ValueError):
        ImageMaskFromPointLabels(image_size=(0, 5))
    with pytest.raises(ValueError):
        ImageMaskFromPointLabels(image_size=SIZE, radius=-1)
    with pytest.raises(ValueError):
        ImageMaskFromPointLabels(image_size=SIZE, unknown_value=300)
    with pytest.raises(ValueError):
        ImageMaskFromPointLabels(image_size=SIZE, occlusion_bin_px=0)


# ------------------------------------------------------------------
# Painting
# ------------------------------------------------------------------

def test_paints_labels_at_projected_pixels():
    # uv rows are [u (col), v (row), depth]
    mask = _mask([1, 0], [[3.4, 2.8, 5.0], [7.0, 6.0, 5.0]])
    assert mask.shape == SIZE
    assert mask.dtype == np.uint8
    assert mask[2, 3] == 1
    assert mask[6, 7] == 0
    assert (mask == 255).sum() == SIZE[0] * SIZE[1] - 2


def test_invalid_points_leave_unknown():
    mask = _mask([1], [[np.nan, np.nan, np.nan]])
    assert (mask == 255).all()


def test_custom_unknown_value():
    mask = _mask([1], [[np.nan, np.nan, np.nan]], unknown_value=7)
    assert (mask == 7).all()


def test_nearest_point_wins_pixel_conflict():
    # Both points land in pixel (row 4, col 3); the one at depth 2 wins.
    mask = _mask([0, 1], [[3.2, 4.7, 10.0], [3.8, 4.1, 2.0]])
    assert mask[4, 3] == 1


def test_row_misalignment_rejected():
    with pytest.raises(ValueError):
        _mask([1, 0, 1], [[3.0, 2.0, 5.0]])


def test_labels_out_of_uint8_rejected():
    proc = ImageMaskFromPointLabels(image_size=SIZE)
    sample = Sample(
        data={
            "trav_traj": np.array([300], dtype=np.int64),
            "lidar_uv": np.array([[3.0, 2.0, 5.0]], dtype=np.float32),
        }
    )
    with pytest.raises(ValueError):
        proc(sample)


# ------------------------------------------------------------------
# Splat radius
# ------------------------------------------------------------------

def test_radius_paints_disc():
    mask = _mask([1], [[5.0, 5.0, 3.0]], radius=1)
    # r=1 disc is a plus shape around (row 5, col 5).
    for row, col in [(5, 5), (4, 5), (6, 5), (5, 4), (5, 6)]:
        assert mask[row, col] == 1
    assert mask[4, 4] == 255  # diagonal excluded: 1^2 + 1^2 > 1^2


def test_radius_clipped_at_image_border():
    mask = _mask([1], [[0.0, 0.0, 3.0]], radius=1)
    assert mask[0, 0] == 1
    assert mask[0, 1] == 1
    assert mask[1, 0] == 1


def test_radius_conflict_resolved_by_depth():
    # Two overlapping discs; every contested pixel goes to the nearer point.
    mask = _mask(
        [1, 0],
        [[5.0, 5.0, 1.0], [6.0, 5.0, 5.0]],
        radius=1,
    )
    assert mask[5, 5] == 1  # P1 centre, contested by P2's disc
    assert mask[5, 6] == 1  # P2 centre, contested by P1's disc — P1 nearer
    assert mask[5, 7] == 0  # P2 only


# ------------------------------------------------------------------
# Occlusion filtering
# ------------------------------------------------------------------

def test_occluded_point_dropped():
    # Same 8x8 bin: the point 8 m behind the nearest return is not painted.
    mask = _mask(
        [1, 1],
        [[2.0, 2.0, 2.0], [5.0, 5.0, 10.0]],
        occlusion_bin_px=8,
        occlusion_depth_margin=0.5,
    )
    assert mask[2, 2] == 1
    assert mask[5, 5] == 255


def test_point_within_margin_kept():
    mask = _mask(
        [1, 1],
        [[2.0, 2.0, 2.0], [5.0, 5.0, 2.3]],
        occlusion_bin_px=8,
        occlusion_depth_margin=0.5,
    )
    assert mask[2, 2] == 1
    assert mask[5, 5] == 1


def test_points_in_different_bins_not_occluded():
    mask = _mask(
        [1, 1],
        [[2.0, 2.0, 2.0], [10.0, 2.0, 10.0]],  # cols 2 and 10: bins 0 and 1
        occlusion_bin_px=8,
        occlusion_depth_margin=0.5,
    )
    assert mask[2, 2] == 1
    assert mask[2, 10] == 1
