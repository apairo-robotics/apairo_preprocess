import numpy as np
import pytest
from apairo.core.sample import Sample

from apairo_preprocess.projection.point_features import PointFeaturesFromImage


def _image(height=4, width=6, channels=3):
    """Image whose pixel (row, col) holds [row, col, row + col] — position-decodable."""
    img = np.zeros((height, width, channels), dtype=np.uint8)
    rows, cols = np.mgrid[0:height, 0:width]
    img[..., 0] = rows
    img[..., 1] = cols
    img[..., 2] = rows + cols
    return img


def _sample(uv, img, projection_key="lidar_uv", image_key="image"):
    return Sample(
        data={
            projection_key: np.asarray(uv, dtype=np.float32),
            image_key: img,
        }
    )


# ------------------------------------------------------------------
# Constructor / API
# ------------------------------------------------------------------

def test_default_keys():
    proc = PointFeaturesFromImage()
    assert proc.input_keys == ["lidar_uv", "image"]
    assert proc.timestamps_from == "lidar_uv"
    assert proc.output_key == "point_features"


def test_custom_keys():
    proc = PointFeaturesFromImage(
        projection_key="lidar_uv_left",
        image_key="image_left_color",
        output_key="lidar_rgb",
    )
    assert proc.input_keys == ["lidar_uv_left", "image_left_color"]
    assert proc.timestamps_from == "lidar_uv_left"
    assert proc.sources == ["lidar_uv_left", "image_left_color"]
    assert proc.output_key == "lidar_rgb"


# ------------------------------------------------------------------
# Sampling
# ------------------------------------------------------------------

def test_samples_nearest_pixel():
    proc = PointFeaturesFromImage()
    uv = [[2.7, 1.2, 5.0]]  # col 2, row 1
    out = proc(_sample(uv, _image()))
    np.testing.assert_array_equal(out[0], [1, 2, 3])


def test_invalid_rows_get_fill_value():
    proc = PointFeaturesFromImage(fill_value=9)
    uv = [[2.0, 1.0, 5.0], [np.nan, np.nan, np.nan]]
    out = proc(_sample(uv, _image()))
    np.testing.assert_array_equal(out[1], [9, 9, 9])


def test_output_row_aligned_with_image_dtype():
    proc = PointFeaturesFromImage()
    uv = [[0.0, 0.0, 1.0], [np.nan, np.nan, np.nan], [5.0, 3.0, 2.0]]
    out = proc(_sample(uv, _image()))
    assert out.shape == (3, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[2], [3, 5, 8])


def test_single_channel_image_yields_n_by_one():
    proc = PointFeaturesFromImage()
    img = np.arange(24, dtype=np.float32).reshape(4, 6)
    out = proc(_sample([[2.0, 1.0, 5.0]], img))
    assert out.shape == (1, 1)
    assert out[0, 0] == img[1, 2]


def test_projection_beyond_image_bounds_rejected():
    proc = PointFeaturesFromImage()
    uv = [[10.0, 1.0, 5.0]]  # col 10 in a 6-wide image
    with pytest.raises(ValueError):
        proc(_sample(uv, _image()))


def test_bad_projection_shape_rejected():
    proc = PointFeaturesFromImage()
    with pytest.raises(ValueError):
        proc(_sample(np.zeros((3,)), _image()))


def test_empty_projection():
    proc = PointFeaturesFromImage()
    out = proc(_sample(np.zeros((0, 3)), _image()))
    assert out.shape == (0, 3)
