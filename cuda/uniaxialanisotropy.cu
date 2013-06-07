#include "float3.h"
#include "mask.h"

// Add uniaxial magnetocrystalline anisotropy field to B.
// U vector is in Tesla.
extern "C" __global__ void
adduniaxialanisotropy(float* __restrict__  Bx, float* __restrict__  By, float* __restrict__  Bz,
                      float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz, 
                      float* __restrict__  kx_red, float* __restrict__  ky_red, float* __restrict__  kz_red, 
                      int8_t* regions, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {
	
		int8_t reg = regions[i];
		float  ux = kx_red[reg];
		float  uy = ky_red[reg];
		float  uz = kz_red[reg];
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

