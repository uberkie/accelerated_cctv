#!/usr/bin/env bash
set -euo pipefail

SOCK=/tmp/zero_copy_worker.sock

echo "Build native worker (requires cmake and a header for nlohmann/json)"
mkdir -p services/compute/native_worker/build
cd services/compute/native_worker/build || exit 1
cmake .. || true
make || true
echo "To run the mock native worker (it will listen on ${SOCK}):"
echo "  ./native_worker ${SOCK}"
echo
cd - >/dev/null

echo "Conceptual GStreamer pipeline (illustrative only):"
echo "gst-launch-1.0 rtspsrc location=<RTSP> ! rtph264depay ! h264parse ! nvv4l2decoder ! \"
echo "  video/x-raw(memory:NVMM) ! zero_copy_exporter worker_socket=${SOCK} ! queue ! nvv4l2h264enc ! h264parse ! mpegtsmux ! hlssink location=/tmp/hls/index.m3u8"

echo
echo "Notes: the 'zero_copy_exporter' element is conceptual and must be implemented as a native GStreamer element that can export buffer handles to the worker. This script only demonstrates the pieces and the mock worker." 
