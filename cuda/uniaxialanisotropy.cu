#include "float3.h"
#include "mask.h"

// Add uniaxial magnetocrystalline anisotropy field to B.
// U vector is in Tesla.
extern "C" __global__ void
adduniaxialanisotropy(float* __restrict__  Bx, float* __restrict__  By, float* __restrict__  Bz,
                      float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz, 
                      float* __restrict__  maskX, float* __restrict__  maskY, float* __restrict__  maskZ, 
                      float Ux, float Uy, float Uz, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {
	
		float  ux = Ux * loadmask(maskX, i);
		float  uy = Uy * loadmask(maskY, i);
		float  uz = Uz * loadmask(maskZ, i);
		float3 U  = {ux, uy, uz};
		float3 u  = normalized(U);
		float  K  = len(U);
		float3 m  = {mx[i], my[i], mz[i]};
		float3 Ba = (2 * K) * dot(m, u) * u;

		Bx[i] += Ba.x;
		By[i] += Ba.y;
		Bz[i] += Ba.z;
	}
}

