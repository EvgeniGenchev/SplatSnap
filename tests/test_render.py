import os
from splatsnap import render_splats

def test_render_output():
    input_path = "tests/garden.splat"
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
