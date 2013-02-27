#include "float3.h"

#define PI 3.14159265359
#define MU0 (4*PI*1e-7)

// Uniaxial magnetocrystalline anisotropy.
extern "C" __global__ void
uniaxialanisotropy(float* __restrict__  Hx, float* __restrict__  Hy, float* __restrict__  Hz,
                   float* __restrict__  Mx, float* __restrict__  My, float* __restrict__  Mz, 
                   float Ux, float Uy, float Uz, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {
	
		float3 M   = {Mx[i], My[i], Mz[i]};
		float  Bs2 = dot(M, M);
		float3 U   = {Ux, Uy, Uz};
		float  K   = len(U);
		float3 Ha  = (2 * MU0 / (Bs2 * K)) * dot(M, U) * U;

		Hx[i] = Ha.x;
		Hy[i] = Ha.y;
		Hz[i] = Ha.z;
	}
}

