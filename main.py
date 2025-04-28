import numpy as np
from scipy.spatial.transform import rotation
from numba import jit, cuda, njit, types
from PIL import image

def read_splat_file(splat_file_path):
    """ return the splat file from the provided file_path as a list of features """ 
    data = np.fromfile(splat_file_path, dtype=np.uint8)
    num_records = len(data) // 32 
    data = data.reshape((num_records, 32))

    positions = data[:, :12].view(np.float32).reshape(-1, 3)
    scales = data[:, 12:24].view(np.float32).reshape(-1, 3)
    colors = data[:, 24:28]
    rotations = data[:, 28:] / 255.0

    return [{'position': pos, 'scales': scale, 'color': color , 'rot': rot}
                for pos, scale, color, rot in zip(positions, scales, colors, rotations)]

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



def get_visible_splats(extracted_data, view_matrix, projection_matrix):
    result = []

    for splat in extracted_data:
        if is_in_frustum_numba(splat['position'], view_matrix, projection_matrix):
            result.append(splat)

    return result

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
                color_buffer[j, i] = alpha * (splat_color[:3] / 255.0) + \
                                     (1 - alpha) * color_buffer[j, i]

@njit(fastmath=True)
def get_size(splat_scales, image_width, depth):
    return max(1, int(max(splat_scales) * image_width / depth + 1e-6)* 0.4)


def render_gaussian_splats_optimized(extracted_data, view_matrix, projection_matrix, image_width, image_height):
    
    depth_buffer = np.full((image_height, image_width), np.inf)
    color_buffer = np.full((image_height, image_width, 3),1, dtype=np.float32)

    visible_splats = get_visible_splats(extracted_data, view_matrix, projection_matrix)

    sorted_splats = sorted(visible_splats, key=lambda s: np.dot(s['position'], view_matrix[:3, 2]))

    for splat in sorted_splats:
        rotated_pos = splat['position']

        x, y, depth = project_splat(rotated_pos, view_matrix,
                                          projection_matrix,
                                          image_width,
                                          image_height)

        if 0 <= x < image_width and 0 <= y < image_height:
            size = get_size(splat['scales'], image_width, depth)

            update_color_and_depth_buffers(x, y, size, depth, splat['color'], depth_buffer, color_buffer)



    return color_buffer

def get_rendered_image(extracted_data):
    camera_position = np.array([0, 0, 5])
    look_at = np.array([0, 0, 0])
    up = np.array([0, -1, 0])
    fov = np.radians(70)
    image_width, image_height = 800, 600

    view_matrix, projection_matrix = setup_camera(camera_position, look_at, up, fov, image_width, image_height)
    rendered_image = render_gaussian_splats_optimized(extracted_data, view_matrix, projection_matrix, image_width, image_height)

    return np.clip(rendered_image, 0, 1)

if __name__ == "__main__":
    path = "./crime.splat"
    extracted_data = read_splat_file(path)
    img = get_rendered_image(extracted_data)
    image_array = (img * 255).astype(np.uint8)
    image = image.fromarray(image_array)
    image.save("output_image.jpg")

