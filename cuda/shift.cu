#include "stencil.h"

// Copies src to dst, shifting elements by (u, v, w).
// Clamps at the boundaries.
extern "C" __global__ void
shift(float* __restrict__  dst, float* __restrict__  src,
      int N0,  int N1,  int N2,
      int sh0, int sh1, int sh2) {

    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(j>=N1 || k>=N2) {
        return;	// out of  bounds
    }

    // loop over N layers
    for (int i=0; i<N0; i++) {
        dst[idx(i, j, k)] = src[idxclamp(i-sh0, j-sh1, k-sh2)];
    }
}

