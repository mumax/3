#include <stdint.h>
#include "float3.h"

// set dst to zero in cells where mask != 0
extern "C" __global__ void
zeromask(float* __restrict__  dstx, float* __restrict__  dsty, float* __restrict__  dstz,
         float* maskLUT, uint8_t* regions, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {
		if (maskLUT[regions[i]] != 0){
			dstx[i] = 0;
			dsty[i] = 0;
			dstz[i] = 0;
		}
	}
}
