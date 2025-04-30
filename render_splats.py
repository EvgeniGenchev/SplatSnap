import shutil
import typer
import numpy as np
import warnings
import pyfiglet
import subprocess
from numba import njit
from PIL import Image
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm
from typing import Tuple

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
app = typer.Typer()


@njit(fastmath=True)
def is_in_frustum_numba(position, view_matrix, projection_matrix):
    clip_space = projection_matrix @ view_matrix @ np.append(position, 1)
    w = clip_space[3]
    for i in range(3):
        if not (-w <= clip_space[i] <= w):
            return False
    return True

@njit(fastmath=True)
def project_splat(position, view_matrix, projection_matrix, image_width, image_height):
    position_homogeneous = np.append(position, 1)
    camera_space = view_matrix @ position_homogeneous
    clip_space = projection_matrix @ camera_space
    ndc = clip_space[:3] / clip_space[3]
    screen_x = (ndc[0] + 1) * 0.5 * image_width
    screen_y = (1 - ndc[1]) * 0.5 * image_height
    return screen_x, screen_y, clip_space[2]

@njit(fastmath=True)
def gaussian_weight(dx, dy, size):
    sigma = size / 4.0
    return np.exp(-(dx**2 + dy**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

@njit(fastmath=True)
def update_color_and_depth_buffers(x, y, size, depth, splat_color, depth_buffer, color_buffer):
    x_min = max(0, int(x - size))
    x_max = min(depth_buffer.shape[1] - 1, int(x + size))
    y_min = max(0, int(y - size))
    y_max = min(depth_buffer.shape[0] - 1, int(y + size))

    for i in range(x_min, x_max + 1):
        for j in range(y_min, y_max + 1):
            dx = (i - x) / size
            dy = (j - y) / size
            weight = gaussian_weight(dx, dy, size)
            if depth < depth_buffer[j, i]:
                depth_buffer[j, i] = depth
                alpha = weight
                color_buffer[j, i] = alpha * (splat_color[:3] / 255.0) + (1 - alpha) * color_buffer[j, i]

@njit(fastmath=True)
def get_size(splat_scales, image_width, depth):
    return max(1, int(max(splat_scales) * image_width / depth + 1e-6)* 0.4)

@njit(fastmath=True)
def render_one_splat(pos, scale, color, view_matrix, projection_matrix, image_width, image_height, depth_buffer, color_buffer):
    x, y, depth = project_splat(pos, view_matrix, projection_matrix, image_width, image_height)
    if 0 <= x < image_width and 0 <= y < image_height:
        size = get_size(scale, image_width, depth)
        update_color_and_depth_buffers(x, y, size, depth, color, depth_buffer, color_buffer)

def render_gaussian_splats_with_progress(positions, scales, colors, sorted_indices, view_matrix, projection_matrix, image_width, image_height, show_progress=True):
    depth_buffer = np.full((image_height, image_width), np.inf)
    color_buffer = np.zeros((image_height, image_width, 3), dtype=np.float32)

    iterable = tqdm(sorted_indices, desc="Rendering splats") if show_progress else sorted_indices

    for idx in iterable:
        render_one_splat(
            positions[idx],
            scales[idx],
            colors[idx],
            view_matrix,
            projection_matrix,
            image_width,
            image_height,
            depth_buffer,
            color_buffer
        )

    return color_buffer

def setup_camera(position, look_at, up, fov, image_width, image_height):
    forward = look_at - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    view_matrix = np.eye(4)
    view_matrix[:3, 0] = right
    view_matrix[:3, 1] = up
    view_matrix[:3, 2] = -forward
    view_matrix[:3, 3] = -position

    aspect_ratio = image_width / image_height
    f = 1 / np.tan(fov / 2)
    projection_matrix = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, -1, -1],
        [0, 0, -1, 0]
    ])

    return view_matrix, projection_matrix

def read_splat_file(path):
    data = np.fromfile(path, dtype=np.uint8)
    num_records = len(data) // 32
    data = data.reshape((num_records, 32))

    positions = data[:, :12].view(np.float32).reshape(-1, 3)
    scales = data[:, 12:24].view(np.float32).reshape(-1, 3)
    colors = data[:, 24:28]
    rotations = data[:, 28:] / 255.0

    return positions, scales, colors, rotations

@app.callback(invoke_without_command=True)
def main(
    input_path: str = typer.Option(..., help="Path to input .splat file"),
    output_path: str = typer.Option(..., help="Path to save output image"),
    camera_position: Tuple[float, float, float] = typer.Option((0, 0, 5), help="Camera position"),
    look_at: Tuple[float, float, float] = typer.Option((0, 0, 0), help="Look-at point"),
    up: Tuple[float, float, float] = typer.Option((0, -1, 0), help="Up direction"),
    fov_deg: float = typer.Option(70.0, help="Field of view in degrees"),
    image_width: int = typer.Option(800, help="Image width"),
    image_height: int = typer.Option(600, help="Image height"),
    show_progress: bool = typer.Option(True, help="Show progress bar"),
    show_image: bool = typer.Option(False, help="Show image in terminal if possible")
):

    typer.echo(pyfiglet.figlet_format("SplatSnap"))
    typer.echo(f'\n>>Reading {input_path}')
    positions, scales, colors, _ = read_splat_file(input_path)
    fov = np.radians(fov_deg)

    typer.echo(">>Setting up the camera")
    view_matrix, projection_matrix = setup_camera(
        np.array(camera_position),
        np.array(look_at),
        np.array(up),
        fov,
        image_width,
        image_height
    )

    sorted_indices = np.argsort(np.dot(positions, view_matrix[:3, 2]))

    img = render_gaussian_splats_with_progress(
        positions, scales, colors, sorted_indices,
        view_matrix, projection_matrix,
        image_width, image_height,
        show_progress
    )

    image_array = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    image.save(output_path)
    typer.echo(f">>Saved rendered image to {output_path}\n")


    if show_image:
        typer.echo(">>Printing image:")
        if shutil.which("catimg"):
            subprocess.run(["catimg", '-w 120', output_path])
        elif shutil.which("viu"):
            subprocess.run(["viu", output_path])
        else:
            typer.echo("No supported terminal image viewer found (e.g. 'catimg' or 'viu').")

if __name__ == "__main__":
    app()
