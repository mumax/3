#include <stdint.h>
#include "float3.h"

// Steepest descent energy minimizer
extern "C" __global__ void
minimize(float* __restrict__ mx,  float* __restrict__  my,  float* __restrict__ mz,
         float* __restrict__ m0x, float* __restrict__  m0y, float* __restrict__ m0z,
         float* __restrict__ tx,  float* __restrict__  ty,  float* __restrict__ tz,
         float dt, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {

		float3 m0 = {m0x[i], m0y[i], m0z[i]};
		float3 t = {tx[i], ty[i], tz[i]};

		float t2 = dt*dt*dot(t, t);
		float3 result = (4 - t2) * m0 + 4 * dt * t;
		float divisor = 4 + t2;
		
		mx[i] = result.x / divisor;
		my[i] = result.y / divisor;
		mz[i] = result.z / divisor;
	}
}

