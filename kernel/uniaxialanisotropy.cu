#include "float3.h"

// Uniaxial magnetocrystalline anisotropy.
extern "C" __global__ void
uniaxialanisotropy(float* __restrict__  Bx, float* __restrict__  By, float* __restrict__  Bz,
                   float* __restrict__  Mx, float* __restrict__  My, float* __restrict__  Mz, 
                   float Ux, float Uy, float Uz, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {
	
		float3 M  = {Mx[i], My[i], Mz[i]};
		float  Bs = len(M);
		float3 m  = normalized(M);
		float3 U  = {Ux, Uy, Uz};
		float3 u  = normalized(U);
		float  K  = len(U);
		float3 Ba = (2 * K / Bs) * dot(m, u) * u;

		Bx[i] = Ba.x;
		By[i] = Ba.y;
		Bz[i] = Ba.z;
	}
}

