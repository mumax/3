#include "float3.h"

// normalize vector {vx, vy, vz} to unit length, unless length or vol are zero.
extern "C" __global__ void
normalize(float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz, float* __restrict__ vol, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {

		float v = (vol == NULL? 1.0f: vol[i]);
		float3 V = {v*vx[i], v*vy[i], v*vz[i]};
		V = normalized(V);
		vx[i] = V.x;
		vy[i] = V.y;
		vz[i] = V.z;
	}
}

