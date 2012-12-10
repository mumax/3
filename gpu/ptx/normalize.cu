#include "common_func.h"

extern "C" __global__ void
normalize(float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {

		float3 V = {vx[i], vy[i], vz[i]};
		V = normalized(V);
		vx[i] = V.x;
		vy[i] = V.y;
		vz[i] = V.z;
	}
}

