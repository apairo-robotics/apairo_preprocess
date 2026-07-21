import numpy as np
import pytest
from apairo.core.sample import Sample

from apairo_preprocess.projection.bev import BEVRasterisation, to_uint8_image

BOUNDS = (-2.0, 2.0, -2.0, 2.0)  # xmin, xmax, ymin, ymax
RES = 1.0  # -> a 4x4 grid


def _sample(points, values, points_key="lidar", values_key="values"):
    return Sample(
        data={
            points_key: np.asarray(points, dtype=np.float32),
            values_key: np.asarray(values, dtype=np.float64),
        }
    )


def _bev(points, values, **kwargs):
    proc = BEVRasterisation(bounds=BOUNDS, resolution=RES, **kwargs)
    return proc(_sample(points, values))


# ------------------------------------------------------------------
# Constructor / API
# ------------------------------------------------------------------

def test_default_keys():
    proc = BEVRasterisation(bounds=BOUNDS, resolution=RES)
    assert proc.input_keys == ["lidar", "values"]
    assert proc.timestamps_from == "lidar"
    assert proc.output_key == "bev"


def test_custom_keys():
    proc = BEVRasterisation(
        bounds=BOUNDS, resolution=RES,
        points_key="scan", values_key="height", output_key="bev_height",
    )
    assert proc.input_keys == ["scan", "height"]
    assert proc.sources == ["scan", "height"]
    assert proc.output_key == "bev_height"


def test_shape_from_bounds_and_resolution():
    proc = BEVRasterisation(bounds=BOUNDS, resolution=RES)
    assert proc.shape == (4, 4)
    proc = BEVRasterisation(bounds=(0.0, 10.0, 0.0, 5.0), resolution=0.5)
    assert proc.shape == (10, 20)


def test_bad_parameters_rejected():
    with pytest.raises(ValueError):
        BEVRasterisation(bounds=(1.0, 0.0, -2.0, 2.0), resolution=RES)  # xmax < xmin
    with pytest.raises(ValueError):
        BEVRasterisation(bounds=BOUNDS, resolution=0.0)
    with pytest.raises(ValueError):
        BEVRasterisation(bounds=BOUNDS, resolution=-1.0)
    with pytest.raises(ValueError):
        BEVRasterisation(bounds=BOUNDS, resolution=RES, reduce="min")


def test_row_misalignment_rejected():
    with pytest.raises(ValueError):
        _bev([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], [1.0])


def test_bad_point_shape_rejected():
    proc = BEVRasterisation(bounds=BOUNDS, resolution=RES)
    with pytest.raises(ValueError):
        proc(Sample(data={"lidar": np.zeros((3,)), "values": np.zeros((3,))}))


# ------------------------------------------------------------------
# Rasterisation
# ------------------------------------------------------------------

def test_paints_values_at_cell():
    # bounds (-2,2,-2,2), res 1 -> row 0 is y in (1,2], col 0 is x in (-2,-1].
    grid = _bev([[-1.5, 1.5, 0.0]], [7.0])
    assert grid.shape == (4, 4)
    assert grid[0, 0] == 7.0


def test_background_fills_empty_cells():
    grid = _bev([[-1.5, 1.5, 0.0]], [7.0], background=-1.0)
    assert grid[0, 0] == 7.0
    assert (grid[grid != 7.0] == -1.0).all()


def test_out_of_bounds_points_dropped():
    grid = _bev([[-1.5, 1.5, 0.0], [100.0, 100.0, 0.0]], [7.0, 9.0])
    assert grid[0, 0] == 7.0
    assert (grid == 9.0).sum() == 0


def test_max_reduce_picks_highest():
    # Two points in the same cell (row 0, col 0).
    grid = _bev([[-1.9, 1.9, 0.0], [-1.1, 1.1, 0.0]], [3.0, 8.0])
    assert grid[0, 0] == 8.0


def test_max_reduce_handles_negative_values_below_background():
    # Real values below the default background=0.0 must not be floor-clamped.
    grid = _bev([[-1.5, 1.5, 0.0]], [-5.0], background=0.0)
    assert grid[0, 0] == -5.0


def test_mean_reduce_averages_cell():
    grid = _bev(
        [[-1.9, 1.9, 0.0], [-1.1, 1.1, 0.0]], [2.0, 6.0], reduce="mean",
    )
    assert grid[0, 0] == 4.0


def test_last_reduce_keeps_final_point():
    grid = _bev(
        [[-1.9, 1.9, 0.0], [-1.1, 1.1, 0.0]], [2.0, 6.0], reduce="last",
    )
    assert grid[0, 0] == 6.0


def test_north_up_row_ordering():
    # A point near ymax (north) lands in row 0; near ymin in the last row.
    grid = _bev([[0.0, 1.9, 0.0], [0.0, -1.9, 0.0]], [1.0, 2.0])
    assert grid[0, :].max() == 1.0
    assert grid[-1, :].max() == 2.0


# ------------------------------------------------------------------
# to_uint8_image
# ------------------------------------------------------------------

def test_to_uint8_image_normalises_to_own_range():
    grid = np.array([[0.0, 5.0], [10.0, 2.5]])
    img = to_uint8_image(grid)
    assert img.shape == (2, 2, 3)
    assert img.dtype == np.uint8
    assert img[0, 0, 0] == 0
    assert img[1, 0, 0] == 255


def test_to_uint8_image_fixed_range():
    grid = np.array([[0.0, 10.0]])
    img = to_uint8_image(grid, vmin=0.0, vmax=20.0)
    assert img[0, 0, 0] == 0
    assert img[0, 1, 0] == 127  # 10/20 * 255 = 127.5, truncated by the uint8 cast


def test_to_uint8_image_constant_grid_does_not_divide_by_zero():
    grid = np.full((2, 2), 3.0)
    img = to_uint8_image(grid)
    assert (img == 0).all()
