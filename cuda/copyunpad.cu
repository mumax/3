#include "stencil.h"

// Copy src (size S, larger) to dst (size D, smaller)
extern "C" __global__ void
copyunpad(float* __restrict__  dst, int Dx, int Dy, int Dz,
          float* __restrict__  src, int Sx, int Sy, int Sz) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix<Dx && iy<Dy && iz<Dz) {
		dst[index(ix, iy, iz, Dx, Dy, Dz)] = src[index(ix, iy, iz, Sx, Sy, Sz)];
	}
}

