"""GStreamer appsink -> (optional) CUDA processing example.

This script is a small demo: it connects to an RTSP source via GStreamer,
pulls samples from an appsink and applies a simple invert operation to the
frame. If PyCUDA is available, it dispatches a tiny kernel; otherwise it
falls back to a numpy-based CPU invert.

Notes:
- This is a demo; real zero-copy requires EGL/DMABUF/CUDA interop and driver
  support. Here we map the appsink buffer to system memory for simplicity.
"""

import argparse
import sys
import signal
import numpy as np

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib
except Exception:
    print("GStreamer (gi.repository.Gst) not available. Ensure PyGObject is installed.")
    raise

Gst.init(None)

# Optional PyCUDA path
HAS_PYCUDA = False
try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    from pycuda.compiler import SourceModule

    HAS_PYCUDA = True
except Exception:
    HAS_PYCUDA = False


# Optional web server hook
HAS_WEB = False
try:
    from web.mjpeg_server import set_frame, run as run_mjpeg_server
    from PIL import Image
    import io
    HAS_WEB = True
except Exception:
    HAS_WEB = False


# If pycuda is present, compile a small invert kernel
invert_kernel = None
if HAS_PYCUDA:
    import os
    cu_path = os.path.join(os.path.dirname(__file__), "cuda_cctv_kernels.cu")
    try:
        if os.path.exists(cu_path):
            src = open(cu_path, "r", encoding="utf-8").read()
            mod = SourceModule(src, no_extern_c=True)
        else:
            # fallback inline kernel
            mod = SourceModule(
                r"""
            __global__ void invert(unsigned char *img, int size){
                int idx = threadIdx.x + blockDim.x * blockIdx.x;
                if (idx < size) img[idx] = 255 - img[idx];
            }
            """
            )
        # kernel name in file is invert_u8 or fallback invert
        try:
            invert_kernel = mod.get_function("invert_u8")
        except Exception:
            invert_kernel = mod.get_function("invert")
    except Exception as e:
        print("Failed to compile CUDA kernels:", e, file=sys.stderr)
        invert_kernel = None


def cpu_invert(arr: np.ndarray) -> None:
    """In-place invert on numpy array."""
    arr[:] = 255 - arr


def gpu_invert(ptr, size: int) -> None:
    """Call the CUDA invert kernel. ptr should be device pointer.

    Note: This code assumes ptr is a device pointer exposed by proper
    registration/mapping. In this demo we don't implement full zero-copy.
    """
    if invert_kernel is None:
        raise RuntimeError("invert kernel not compiled")
    # 256 threads per block
    block = (256, 1, 1)
    grid = ((size + block[0] - 1) // block[0], 1, 1)
    invert_kernel(ptr, np.int32(size), block=block, grid=grid)


def create_pipeline(rtsp_url: str) -> Gst.Pipeline:
    pipeline_str = (
        f"rtspsrc location={rtsp_url} latency=50 ! "
        "rtph264depay ! h264parse ! nvv4l2decoder ! "
        "nvvideoconvert ! video/x-raw,format=RGBA ! "
        "appsink name=appsink emit-signals=true max-buffers=1 drop=true"
    )
    return Gst.parse_launch(pipeline_str)


def map_buffer_to_ndarray(buf: Gst.Buffer) -> np.ndarray:
    """Map a Gst.Buffer to a numpy uint8 array (flattened).

    This uses Gst.Buffer.extract_dup to get a bytes copy. It's not zero-copy
    but is portable for tests and demos.
    """
    success, info = buf.map(Gst.MapFlags.READ)
    if not success:
        # Fallback to copying via extract_dup
        data = buf.extract_dup(0, buf.get_size())
        arr = np.frombuffer(data, dtype=np.uint8).copy()
    else:
        try:
            mv = memoryview(info.data)
            arr = np.frombuffer(mv, dtype=np.uint8).copy()
        finally:
            buf.unmap(info)
    return arr


def on_new_sample(sink: Gst.AppSink) -> Gst.FlowReturn:
    sample = sink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.OK
    buf = sample.get_buffer()
    arr = map_buffer_to_ndarray(buf)

    # Simple demo: invert the whole buffer
    try:
        if HAS_PYCUDA:
            # copy to device, run kernel, copy back (non-zero-copy demo)
            dptr = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod(dptr, arr)
            gpu_invert(dptr, arr.size)
            cuda.memcpy_dtoh(arr, dptr)
            dptr.free()
        else:
            cpu_invert(arr)
    except Exception as exc:
        print("Processing error:", exc, file=sys.stderr)

    # If web server is enabled, encode to JPEG and publish
    if HAS_WEB:
        try:
            # Treat arr as raw bytes -- attempt to interpret as RGBA if length divisible by 4
            if arr.size % 4 == 0:
                px_count = arr.size // 4
                # assume width unknown; try to create a square-ish image for demo
                w = int(np.sqrt(px_count))
                if w * (w) < px_count:
                    h = (px_count + w - 1) // w
                else:
                    h = w
                img = arr.reshape((h, w, 4))[:, :, :3]  # drop alpha
                pil = Image.fromarray(img.astype('uint8'), 'RGB')
            else:
                # fall back to raw grayscale
                px_count = arr.size
                w = int(np.sqrt(px_count))
                h = (px_count + w - 1) // w
                img = arr.reshape((h, w))
                pil = Image.fromarray(img.astype('uint8'), 'L')
            bio = io.BytesIO()
            pil.save(bio, 'JPEG')
            jpeg = bio.getvalue()
            set_frame(jpeg)
        except Exception as exc:
            print('Failed to publish frame to web server:', exc, file=sys.stderr)

    # In a real pipeline we'd push processed frames downstream or encode/send.
    # For this demo we simply drop after processing.
    return Gst.FlowReturn.OK


def main():
    parser = argparse.ArgumentParser(description="Demo GStreamer -> CUDA encoder")
    parser.add_argument("rtsp_url", help="RTSP URL to connect to")
    args = parser.parse_args()

    pipeline = create_pipeline(args.rtsp_url)
    appsink = pipeline.get_by_name("appsink")
    if appsink is None:
        print("Failed to get appsink from pipeline", file=sys.stderr)
        sys.exit(2)

    # connect handler
    appsink.connect("new-sample", on_new_sample)

    # Graceful shutdown on SIGINT/SIGTERM
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


if __name__ == "__main__":
    main()
