#include "stencil.h"

// See crop.go
extern "C" __global__ void
crop(float* __restrict__  dst, int Dx, int Dy, int Dz,
     float* __restrict__  src, int Sx, int Sy, int Sz,
     int Offx, int Offy, int Offz) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix<Dx && iy<Dy && iz<Dz) {
		dst[index(ix, iy, iz, Dx, Dy, Dz)] = src[index(ix+Offx, iy+Offy, iz+Offz, Sx, Sy, Sz)];
	}
}

