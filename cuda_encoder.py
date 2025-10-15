#!/usr/bin/env python3
"""
GStreamer appsink -> optional CUDA processing demo.

- Uses NVDEC + CUDA convert/download in the pipeline so appsink receives CPU-accessible RGBA.
- If PyCUDA is installed and kernels compiled, will run the tiny invert kernel (non-zero-copy demo:
  it copies buffers to device, runs kernel, copies back). Otherwise falls back to NumPy invert.
- Extracts width/height from caps to reshape the flat buffer correctly.
"""

import argparse
import sys
import signal
import time
import numpy as np
import os

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
gi.require_version('GstApp', '1.0')
from gi.repository import GstApp
Gst.init(None)

# Attempt PyCUDA
HAS_PYCUDA = False
cuda = None
invert_kernel = None
try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # creates a context
    from pycuda.compiler import SourceModule
    HAS_PYCUDA = True
except Exception as e:
    HAS_PYCUDA = False
    # print("PyCUDA not available:", e, file=sys.stderr)

# Try to compile kernel if pycuda present
if HAS_PYCUDA:
    try:
        cu_path = os.path.join(os.path.dirname(__file__), "cuda_cctv_kernels.cu")
        if os.path.exists(cu_path):
            src = open(cu_path, "r", encoding="utf-8").read()
            mod = SourceModule(src, no_extern_c=True)
        else:
            mod = SourceModule(r"""
            __global__ void invert(unsigned char *img, int size){
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size) img[idx] = 255 - img[idx];
            }
            """)
        try:
            invert_kernel = mod.get_function("invert_u8")
        except Exception:
            invert_kernel = mod.get_function("invert")
    except Exception as e:
        print("Failed to compile CUDA kernel:", e, file=sys.stderr)
        invert_kernel = None
        # keep HAS_PYCUDA True because driver/context exists, but kernel may be None


def cpu_invert(arr: np.ndarray) -> None:
    """In-place invert on numpy array."""
    arr[:] = 255 - arr


def gpu_invert_hostcopy(arr: np.ndarray) -> None:
    """Simple device-host copy, GPU kernel, host copy back (non-zero-copy demo)."""
    if not HAS_PYCUDA or invert_kernel is None:
        raise RuntimeError("PyCUDA or invert kernel not available")
    # allocate device memory, copy, run kernel, copy back, free
    nbytes = arr.nbytes
    dptr = cuda.mem_alloc(nbytes)
    cuda.memcpy_htod(dptr, arr)
    # number of elements (bytes)
    elements = np.int32(arr.size)
    block = (256, 1, 1)
    grid = ((int(elements) + block[0] - 1) // block[0], 1, 1)
    invert_kernel(dptr, elements, block=block, grid=grid)
    cuda.memcpy_dtoh(arr, dptr)
    dptr.free()


def create_pipeline(rtsp_url: str) -> Gst.Pipeline:
    # Pipeline forces download to CPU memory (cudadownload) and provides RGBA to appsink.
    pipeline_str = (
        f"rtspsrc location=rtsp://admin:Sti%40%40n123@{rtsp_url}:554/cam/realmonitor?channel=1&subtype=0 latency=50 ! "
        "rtph264depay ! h264parse ! nvh264dec ! "
        "cudaconvert ! cudadownload ! "
        "videoconvert ! "  # converts to NV12 in system memory
        "appsink name=appsink emit-signals=true max-buffers=1 drop=true"
    )
    return Gst.parse_launch(pipeline_str)


def map_buffer_to_ndarray(buf: Gst.Buffer, caps: Gst.Caps) -> (np.ndarray, int, int):
    """
    Extracts raw bytes and returns a numpy uint8 array plus width,height.

    Uses extract_dup to be robust across GStreamer builds.
    """
    data = buf.extract_dup(0, buf.get_size())
    arr = np.frombuffer(data, dtype=np.uint8).copy()

    # read width/height from caps
    try:
        s = caps.get_structure(0)
        width = int(s.get_value("width"))
        height = int(s.get_value("height"))
        # expected size sanity check: 4 channels RGBA
        expected = width * height * 4
        if arr.size != expected:
            # not fatal: try to recover by inferring w/h
            pass
    except Exception:
        width = height = 0

    return arr, width, height


# App sink callback
frame_count = 0
times = []


def on_new_sample(sink: GstApp.AppSink) -> Gst.FlowReturn:
    from web.mjpeg_server import set_frame
    from PIL import Image
    import io
    import numpy as np
    import time

    global frame_count, times
    t0 = time.time()

    sample = sink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.OK
    buf = sample.get_buffer()
    caps = sample.get_caps()

    try:
        arr, w, h = map_buffer_to_ndarray(buf, caps)

        # Infer width/height if missing
        if w == 0 or h == 0:
            if arr.size % 4 == 0:
                px = arr.size // 4
                w = int(np.sqrt(px))
                h = (px + w - 1) // w
            else:
                w, h = arr.size, 1

        # Process: GPU or CPU
        if HAS_PYCUDA and invert_kernel is not None:
            try:
                gpu_invert_hostcopy(arr)
            except Exception as e:
                print("GPU failed, fallback CPU:", e, file=sys.stderr)
                cpu_invert(arr)
        else:
            cpu_invert(arr)

        # Convert to JPEG and push to MJPEG server
        if arr.size % 4 == 0:
            frame = arr[:w*h*4].reshape((h, w, 4))[:, :, :3]
        else:
            frame = arr[:w*h].reshape((h, w))
        pil_img = Image.fromarray(frame.astype('uint8'), 'RGB')
        bio = io.BytesIO()
        pil_img.save(bio, format='JPEG')
        set_frame(bio.getvalue())

    except Exception as e:
        print("Error processing sample:", e, file=sys.stderr)

    t1 = time.time()
    frame_count += 1
    times.append(t1 - t0)
    if frame_count % 10 == 0:
        avg_ms = 1000.0 * (sum(times) / len(times))
        print(f"[frames={frame_count}] avg processing latency: {avg_ms:.2f} ms")

    return Gst.FlowReturn.OK


def main():
    parser = argparse.ArgumentParser(description="GStreamer -> (optional) CUDA demo")
    parser.add_argument("rtsp_url", help="RTSP location (e.g. rtsp://ip/stream)")
    args = parser.parse_args()

    pipeline = create_pipeline(args.rtsp_url)
    appsink = pipeline.get_by_name("appsink")
    if appsink is None:
        print("Failed to get appsink from pipeline", file=sys.stderr)
        sys.exit(2)

    # ensure appsink API is used
    appsink.set_property("emit-signals", True)
    appsink.connect("new-sample", on_new_sample)

    loop = GLib.MainLoop()

    def _stop(signum, frame):
        pipeline.set_state(Gst.State.NULL)
        loop.quit()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        if len(times) > 0:
            avg_s = sum(times) / len(times)
            print("\n=== SUMMARY ===")
            print(f"frames: {frame_count}")
            print(f"avg processing latency: {avg_s*1000:.2f} ms")
            print(f"approx fps (processing-limited): {1/avg_s:.2f}")

if __name__ == "__main__":
    main()
