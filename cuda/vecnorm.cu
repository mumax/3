#include "float3.h"

// dst = sqrt(ax*ax + ay*ay + az*az)
extern "C" __global__ void
vecnorm(float* __restrict__ dst,
           float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
           int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {
		float3 A = {ax[i], ay[i], az[i]};
		dst[i] = sqrtf(dot(A, A));
	}
}
