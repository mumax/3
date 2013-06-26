#include "stencil.h"

// Copies src to dst, shifting elements by (u, v, w).
// Clamps at the boundaries.
extern "C" __global__ void
shift(float* __restrict__  dst, float* __restrict__  src,
      int N0,  int N1,  int N2,
      int sh0, int sh1, int sh2) {

    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= N0 || j>=N1 || k>=N2) {
        return;	// out of  bounds
    }

    dst[idx(i, j, k)] = src[idxclamp(i-sh0, j-sh1, k-sh2)];
}

