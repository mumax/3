#include "common_func.h"

extern "C" __global__ void
normalize(float* vx, float* vy, float* vz, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {

		float3 V = {vx[i], vy[i], vz[i]};
		normalize(V);
		vx[i] = V.x;
		vy[i] = V.y;
		vz[i] = V.z;
	}
}

