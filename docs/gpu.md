# GPU support and detection

This project supports offloading compute-heavy operations to GPUs. The repository contains a small detection helper at `libs/gpu/detect.py` which can be used by plugins and services to decide whether to use hardware acceleration.

Notes:
- NVIDIA: requires `nvidia-smi` available in PATH and the `nvidia-container-runtime` when running in containers.
- Intel VAAPI: requires `vainfo` and appropriate drivers installed.
- If no GPU is detected, plugins must gracefully fall back to software codecs.

Example usage from a plugin:

```python
from libs.gpu.detect import detect_all

gpu = detect_all()
if gpu:
    # prefer NVIDIA if available
    print("GPU available:", gpu)
else:
    print("No GPU detected; using software fallback")
```

When adding new heavy compute plugins, include `gpu_hints` in the plugin manifest (example in `plugins/samples/*/plugin.yaml`).
