#include "float3.h"

// Only the damping term of the Landau-Lifshitz torque, with alpha = 1.
extern "C" __global__ void
dampingtorque(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
              float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz, 
              float* __restrict__  hx, float* __restrict__  hy, float* __restrict__  hz, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {

		float3 m = {mx[i], my[i], mz[i]};
		float3 H = {hx[i], hy[i], hz[i]};
		float3 torque = -cross(m, cross(m, H));

		tx[i] = torque.x;
		ty[i] = torque.y;
		tz[i] = torque.z;
	}
}

