"""Minimal GPU detection helpers.

This module provides lightweight detection of common GPU runtimes (NVIDIA via nvidia-smi,
and Intel via vainfo) and returns a normalized hint structure that plugins can consume.
"""
import shutil
import subprocess
from typing import Dict, Any


def detect_nvidia() -> Dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {}
    try:
        out = subprocess.check_output([nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True)
        # parse the first line
        first = out.strip().splitlines()[0]
        name, mem = [p.strip() for p in first.split(",")]
        return {"vendor": "nvidia", "name": name, "memory": mem}
    except Exception:
        return {}


def detect_vaapi() -> Dict[str, Any]:
    vainfo = shutil.which("vainfo")
    if not vainfo:
        return {}
    try:
        out = subprocess.check_output([vainfo], text=True, stderr=subprocess.STDOUT)
        return {"vendor": "intel_vaapi", "info": out.splitlines()[:5]}
    except Exception:
        return {}


def detect_all() -> Dict[str, Any]:
    # Merge detections; priority: nvidia, vaapi
    det = detect_nvidia()
    if det:
        return det
    det = detect_vaapi()
    return det
