#include "float3.h"

// Uniaxial magnetocrystalline anisotropy

extern "C" __global__ void
uniaxialanisotropy(float* __restrict__  Hx, float* __restrict__  Hy, float* __restrict__  Hz,
                   float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz, 
                   float ux, float uy, float uz, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {
	
		float3 m = {mx[i], my[i], mz[i]};
		float3 u = {ux, uy, uz};
		float3 H = u * dotf(m, u);

		Hx[i] = H.x;
		Hy[i] = H.y;
		Hz[i] = H.z;
	}
}

