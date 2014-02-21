#include "stencil.h"

// shift dst by shx cells (positive or negative) along X-axis.
// new edge value is clampL at left edge or clampR at right edge.
extern "C" __global__ void
shiftx(float* __restrict__  dst, float* __restrict__  src,
       int Nx,  int Ny,  int Nz, int shx, float clampL, float clampR) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if(ix < Nx && iy < Ny && iz < Nz) {
		int ix2 = ix-shx;
		float newval;
		if (ix2 < 0) {
			newval = clampL;
		} else if (ix2 >= Nx) {
			newval = clampR;
		} else {
			newval = src[idx(ix2, iy, iz)];
		}
		dst[idx(ix, iy, iz)] = newval;
	}
}

