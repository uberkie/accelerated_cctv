Zero-copy pipeline (notes & helper sketch)
=========================================

Goal
----
Provide a high-level recipe and a small native helper sketch for achieving
zero-copy handoff between GStreamer/NVDEC (EGLImage or DMA-BUF backed frames)
and CUDA device processing.

Why
---
Zero-copy avoids host <-> device copies and dramatically reduces latency and
CPU overhead for per-frame processing (analytics, transforms, re-encode).

High-level flow (NVIDIA EGL path)
--------------------------------
1. Decode frames into an EGLImage-backed surface (NVDEC / nvv4l2decoder can
   produce EGLImages / DMABUFs depending on pipeline configuration).
2. Export the EGLImage to a native handle or to a DMA-BUF file descriptor.
3. Use a native helper (C) to register that EGLImage / DMA-BUF with CUDA:
   - If using EGLImage: call cudaGraphicsEGLRegisterImage / cudaGraphicsResourceGetMappedEglFrame
   - If using DMA-BUF fd: use appropriate APIs (platform dependent) to obtain
     a CUDA-accessible resource (often via EGL as well).
4. Map the resource in CUDA (cudaGraphicsMapResources) and obtain a device
   pointer or a cudaArray / CUDA surface you can launch kernels against.
5. Launch CUDA kernels directly on the device pointer / array, unmap, and
   optionally re-export or pass to an encoder that accepts device memory.

Key APIs
--------
- EGL: eglCreateImageKHR, eglExportDMABUFImageMESA (availability varies)
- CUDA: cudaGraphicsEGLRegisterImage, cudaGraphicsMapResources,
  cudaGraphicsSubResourceGetMappedArray, cudaMemcpy2DFromArray / kernels
- Linux: DMA-BUF fds (exported from driver) — you can pass fd to other
  processes or register them in-process via EGL extensions.

Why native helper
------------------
Python bindings (e.g., PyCUDA) do not expose all EGL interop primitives.
A small C helper (shared object) can perform the platform-specific calls and
expose a tiny API to Python via ctypes or cffi. This keeps the critical
interop in native code where the drivers expect it.

Helper API sketch (C)
---------------------
The helper should expose functions like:

 - int zc_register_egl_image_from_dma_fd(int dma_fd, void **out_cuda_ptr, size_t *out_size)
   Registers the dmabuf as an EGLImage and a CUDA graphics resource, maps it,
   and returns a device pointer (or a descriptor you can use from Python).

 - int zc_unregister(void *handle)
   Unregister and free resources.

Important notes
---------------
- Driver versions and plugin availability matter. Test on the target
  hardware and driver combo (NVIDIA Jetson vs datacenter GPUs differ).
- Security: exporting dmabuf fds should be done with care; do not leak fds.
- Performance: zero-copy reduces memcpy cost but kernels should be tuned
  (coalesced memory accesses, workgroup sizing) for throughput.

References & examples
---------------------
- NVIDIA CUDA EGL interop examples (search for 'cudaGraphicsEGLRegisterImage')
- Mesa EGL extensions (eglExportDMABUFImageMESA) for DMA-BUF export
- GStreamer elements: nvv4l2decoder, nvvideoconvert, and ways to export DMABUF

Next steps I can take
---------------------
 - Draft the small C helper skeleton (in services/compute/native_worker) with
   build instructions (CMake) and an example of registering an EGLImage with
   CUDA. (I can add this if you want; it will be a sketch requiring tuning.)
 - Add a Python ctypes wrapper example showing how to call the helper from
   `cuda_encoder.py` and get a `void*` device pointer to pass to PyCUDA.
