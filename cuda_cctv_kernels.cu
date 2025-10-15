// cuda_cctv_kernels.cu
// Device kernels for accelerated_cctv demo.
// These kernels expect tightly-packed image data in host or device memory.
// Use PyCUDA's SourceModule(..., no_extern_c=True) to compile at runtime.

#include <stdint.h>

// Invert an array of bytes (generic u8 buffer)
extern "C" __global__ void invert_u8(unsigned char *img, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) img[idx] = 255 - img[idx];
}

// Invert RGBA image where each pixel has 4 bytes (R,G,B,A) interleaved.
// px_count is the number of pixels (not bytes).
extern "C" __global__ void invert_rgba(unsigned char *img, int px_count) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= px_count) return;
    int off = pid * 4;
    img[off + 0] = 255 - img[off + 0]; // R
    img[off + 1] = 255 - img[off + 1]; // G
    img[off + 2] = 255 - img[off + 2]; // B
    // leave alpha as-is
}

// Convert RGBA -> grayscale (output 1 byte per pixel)
extern "C" __global__ void rgba_to_gray(unsigned char *src_rgba, unsigned char *dst_gray, int px_count) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= px_count) return;
    int off = pid * 4;
    unsigned char r = src_rgba[off + 0];
    unsigned char g = src_rgba[off + 1];
    unsigned char b = src_rgba[off + 2];
    // simple luminance
    unsigned char y = (unsigned char)((0.299f * r) + (0.587f * g) + (0.114f * b));
    dst_gray[pid] = y;
}

// Additional kernels (resize, color conversion) can be added as needed.
