# Requirements and installation notes

This document lists Python dependencies (in `requirements.txt`) and common system
packages needed to run parts of the `accelerated_cctv` project. The repo is
multi-faceted (GStreamer, CUDA/PyCUDA, plugins, web server), so not all
system packages are required for every run. Use the "Try it" section below
for quick local setup.

## Python dependencies
The project's Python dependencies are listed in the repository root
`requirements.txt`. As of this change, the file contains:

- numpy
- Pillow
- Flask
- pycuda (not on Windows)
- requests
- PyYAML
- pytest
- PyGObject

Install them in a virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Notes on specific packages
- PyGObject (GObject Introspection) is the Python binding used to access GStreamer
  and other GNOME/GObject libraries. On Debian/Ubuntu you need system packages
  (see below) before `pip` install will work or — preferably — install via the
  OS package manager.

- pycuda requires CUDA drivers and a matching CUDA toolkit installation. On
  machines without CUDA (or when not using GPU paths) you can skip installing
  `pycuda` or install it in a GPU-capable environment.

## System / OS packages (Debian/Ubuntu example)
Some parts of the repo require additional system libraries. Install the
following on Debian/Ubuntu to cover common development needs:

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    python3-dev \
    python3-venv \
    libgirepository1.0-dev \
    gir1.2-gtk-3.0 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    gobject-introspection \
    pkg-config \
    ffmpeg
```

GPU / NVIDIA notes
- Install NVIDIA driver and CUDA toolkit following NVIDIA's docs for your OS.
- For Docker containers, install `nvidia-container-toolkit` and use the
  `--gpus` flag or the runtime. See `docs/gpu.md` for more notes.
- PyCUDA must match the installed CUDA toolkit. When building PyCUDA from
  source, ensure the `CUDAHOME`/`CUDA_PATH` env vars point to the toolkit.

## Minimal "try it" (no GPU)
1. Create a venv and install Python deps (omit `pycuda` on non-GPU machine):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# If you don't have CUDA, edit requirements.txt to remove pycuda or
# install with the environment marker: pip will skip on Windows.
pip install -r requirements.txt
```

2. Start the sample MJPEG server (used by `cuda_encoder.py`):

```bash
python3 web/mjpeg_server.py
```

3. Run tests (quick):

```bash
pytest -q
```

## Next steps / troubleshooting
- If `pip install` fails for `PyGObject`, install the OS packages listed above
  and retry.
- For GPU/CUDA issues, refer to `docs/gpu.md` and ensure driver/toolkit/headers
  match your installed pycuda build.
- If you want, I can create a `dev/setup.sh` script to automate installation on
  Debian/Ubuntu, or a `docker-compose` dev setup that isolates system deps.

---
Generated: October 2025
