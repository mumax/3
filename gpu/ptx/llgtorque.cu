#include "common_func.h"

extern "C" __global__ void
llgtorque(float* tx, float* ty, float* tz,
          float* mx, float* my, float* mz, 
          float* hx, float* hy, float* hz, 
		  float alpha, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {

		float3 m = {mx[i], my[i], mz[i]};
		float3 H = {hx[i], hy[i], hz[i]};

    	float3 mxH = crossf(m, H);
		float gilb = 1.0f / (1.0f + alpha * alpha);
		float3 torque = -mxH + gilb * crossf(m, mxH);

		tx[i] = torque.x;
		ty[i] = torque.y;
		tz[i] = torque.z;
	}
}

