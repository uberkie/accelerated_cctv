# GStreamer Zero-Copy Transcode Plugin (prototype)

This directory contains a scaffold for a zero-copy transcode plugin using GStreamer and a native compute worker.

Goal
- Decode RTSP/video with GPU (NVDEC)
- Hand GPU buffers (NVMM/CUDA) to a native worker for inference (zero-copy via CUDA IPC or DMA-BUF)
- Re-inject the processed GPU buffer and encode with NVENC, then output via WebRTC/HLS

Important: this is a prototype scaffold. Full zero-copy requires platform-specific native code and driver support. The native worker skeleton in `services/compute/native_worker/` demonstrates the IPC contract.

Conceptual GStreamer pipeline (illustrative):

gst-launch-1.0 rtspsrc location=<RTSP> ! rtph264depay ! h264parse ! nvv4l2decoder ! "video/x-raw(memory:NVMM)" ! zero_copy_exporter ! queue ! nvv4l2h264enc ! h264parse ! mpegtsmux ! hlssink location=/tmp/hls/index.m3u8

The `zero_copy_exporter` is a conceptual element that exports a buffer handle or FD to the native worker via a Unix domain socket `worker_socket` and waits for the worker to process the buffer and return control.

IPC contract (conceptual)
- The exporter sends a JSON message with buffer metadata and a CUDA IPC handle or file descriptor over a Unix domain socket.
- The worker receives the handle, maps it into its CUDA context, runs inference, and replies with a status. The exporter then continues the pipeline.

Files in this directory:
- `plugin.yaml` — plugin manifest with `gpu_hints`.
- `README.md` — this file.
