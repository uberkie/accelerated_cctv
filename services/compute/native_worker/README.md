# Native compute worker (skeleton)

This folder contains a minimal C++ skeleton for a native compute worker that communicates via a Unix domain socket.

Purpose:
- Demonstrate the IPC handshake: receive JSON metadata and (conceptual) CUDA handle, print it, and reply.
- This is a scaffold; mapping GstCudaMemory/DMA-BUF to a CUDA device pointer requires system-specific code and driver APIs.

Build & run (example):

```bash
mkdir -p services/compute/native_worker/build
cd services/compute/native_worker/build
cmake ..
make
./native_worker /tmp/zero_copy_worker.sock
```
