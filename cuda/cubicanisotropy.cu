#include <stdint.h>
#include "float3.h"

// add cubic anisotropy field to B.
// B:      effective field in T
// m:      reduced magnetization (unit length)
// K1:     Kc1/Msat in T
// K2:     Kc2/Msat in T
// C1, C2: anisotropy axes
//
// based on http://www.southampton.ac.uk/~fangohr/software/oxs_cubic8.html
extern "C" __global__ void
addcubicanisotropy(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
                   float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                   float* __restrict__  K1LUT, float* __restrict__  K2LUT, float* __restrict__  K3LUT,
                   float* __restrict__ C1xLUT, float* __restrict__ C1yLUT, float* __restrict__ C1zLUT,
                   float* __restrict__ C2xLUT, float* __restrict__ C2yLUT, float* __restrict__ C2zLUT,
                   uint8_t* __restrict__ regions, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {

		uint8_t r  = regions[i];
		float  k1 = K1LUT[r];
		float  k2 = K2LUT[r];
		float  k3 = K3LUT[r];
		float3 u1 = normalized(make_float3(C1xLUT[r], C1yLUT[r], C1zLUT[r]));
		float3 u2 = normalized(make_float3(C2xLUT[r], C2yLUT[r], C2zLUT[r]));
		float3 u3 = cross(u1, u2); // 3rd axis perpendicular to u1,u2
		float3 m  = make_float3(mx[i], my[i], mz[i]);

		float u1m = dot(u1, m);
		float u2m = dot(u2, m);
		float u3m = dot(u3, m);

		float3 B = -2.0f*k1*((pow2(u2m) + pow2(u3m)) * (    (u1m) * u1) +
		                     (pow2(u1m) + pow2(u3m)) * (    (u2m) * u2) +
		                     (pow2(u1m) + pow2(u2m)) * (    (u3m) * u3))-
                    2.0f*k2*((pow2(u2m) * pow2(u3m)) * (    (u1m) * u1) + 
                             (pow2(u1m) * pow2(u3m)) * (    (u2m) * u2) + 
                             (pow2(u1m) * pow2(u2m)) * (    (u3m) * u3))-
                    4.0f*k3*((pow4(u2m) + pow4(u3m)) * (pow3(u1m) * u1) + 
                             (pow4(u1m) + pow4(u3m)) * (pow3(u2m) * u2) + 
                             (pow4(u1m) + pow4(u2m)) * (pow3(u3m) * u3));
		Bx[i] += B.x;
		By[i] += B.y;
		Bz[i] += B.z;
	}
}
