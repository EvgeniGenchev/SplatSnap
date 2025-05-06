# SplatSnap

**SplatSnap** is a command-line tool and Python library for rendering thumbnails from 3D Gaussian splat data. It is ideal for batch preview generation and visualization of 3D scenes using neural rendering techniques.

---

## 🚀 Features

- 🎨 Generate thumbnails from `.splat`
- 🖼️ Customizable output dimensions and camera parameters
- 🧵 Batch processing for multiple scenes or viewpoints
- ⚙️ Python API and CLI support

---

## 🛠️ Installation

```bash
# Using Poetry (recommended)
poetry install

# Or using pip (if not using Poetry)
pip install -e .
```

---

## 📦 Requirements

- Python 3.10+
- NumPy
- Pillow
- tqdm
- numba
- typer
- scipy
- pyfiglet (optional, for fancy CLI output)

Install dependencies via Poetry:

```bash
poetry install
```

---

## 📂 Usage

### CLI

```bash
poetry run splatsnap --input-path path/to/scene.ply --output-path thumb.png
```

### Options

```bash

  --input-path TEXT                     Path to input .splat file  [required]
  --output-path TEXT                    Path to save output image  [required]
  --camera-position FLOAT FLOAT FLOAT   Camera position (x y z)  [default: 0.0 0.0 5.0]
  --look-at FLOAT FLOAT FLOAT           Look-at point (x y z)  [default: 0.0 0.0 0.0]
  --up FLOAT FLOAT FLOAT                Up direction (x y z)  [default: 0.0 -1.0 0.0]
  --fov-deg FLOAT                       Field of view in degrees  [default: 70.0]
  --image-width INTEGER                 Image width in pixels  [default: 800]
  --image-height INTEGER                Image height in pixels  [default: 600]
  --show-progress BOOLEAN               Show progress bar  [default: show-progress]
  --show-image BOOLEAN                  Show image in terminal if possible  [default: no-show-image]
  --install-completion                  Install completion for the current shell.
  --show-completion                     Show completion for the current shell, to copy it or customize the installation.
  --help                                Show this message and exit.
```


### Python API

```python
from splatsnap import render_splats

render_splats.render(
    input_path="path/to/scene.ply",
    output_path="thumb.png",
    width=512,
    height=512,
    camera_config="default"
)
```

---

## 📸 Example

```bash
poetry run splatsnap     --input scenes/chair.ply     --output thumbnails/chair.png     --width 1024     --height 768
```

---

## 📄 License

This project is licensed under the **GPL-3.0 License**. See the [LICENSE](./LICENSE) file for details.

---

## 🙌 Credits

Developed by [Evgeni Genchev](https://github.com/EvgeniGenchev)

Inspired by research on neural point-based graphics and 3D Gaussian splatting.
