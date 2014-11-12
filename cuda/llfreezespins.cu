#include <stdint.h>
#include "float3.h"

// Landau-Lifshitz torque.
extern "C" __global__ void
llfreezespins(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
         float* frozenSpinsLUT, uint8_t* regions, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {

		float3 torque = {tx[i], ty[i], tz[i]};
		float frozenSpins = (1.0f - frozenSpinsLUT[regions[i]]); // the inversion is done to follow the logic of setting this parameter to 1, i.e. freezing the region

		tx[i] = frozenSpins * torque.x;
		ty[i] = frozenSpins * torque.y;
		tz[i] = frozenSpins * torque.z;
	}
}
