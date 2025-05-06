import os
import numpy as np
from splatsnap import render_splats

def test_render_output():
    input_path = "tests/crime.splat"
    output_path = "tests/output.png"

    render_splats.main(
        input_path=input_path,
        output_path=output_path,
        camera_position=(0, 0, 4),
        look_at=(0, 0, 0),
        up=(0, -1, 0),
        fov_deg=70.0,
        image_width=800,
        image_height=600,
        show_progress=False,
        show_image=False
    )

    assert os.path.exists(output_path)
    os.remove(output_path)

def test_gaussian_weight_peak():
    weight = render_splats.gaussian_weight(0.0, 0.0, 1.0)
    assert np.isclose(weight, 1.0 / (2 * np.pi * (1.0 / 4.0) ** 2))

def test_project_splat_center():
    position = np.array([0.0, 0.0, 0.0])
    view_matrix = np.eye(4)
    projection_matrix = np.eye(4)
    image_width = 800
    image_height = 600
    x, y, depth = render_splats.project_splat(position, view_matrix, projection_matrix, image_width, image_height)
    assert np.isclose(x, image_width / 2)
    assert np.isclose(y, image_height / 2)

def test_get_size_scaling():
    splat_scales = np.array([1.0, 1.0, 1.0])
    image_width = 800
    depth = 2.0
    size = render_splats.get_size(splat_scales, image_width, depth)
    expected_size = max(1, int(max(splat_scales) * image_width / depth + 1e-6) * 0.4)
    assert size == expected_size
