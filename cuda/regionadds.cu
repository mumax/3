#include <stdint.h>

// add region-based scalar to dst:
// dst[i] += LUT[region[i]]
extern "C" __global__ void
regionadds(float* __restrict__ dst,
           float* __restrict__ LUT,
           uint8_t* regions, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {

		uint8_t r = regions[i];
		dst[i] += LUT[r];
	}
}

