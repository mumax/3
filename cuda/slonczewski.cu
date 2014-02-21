// Original implementation by Mykola Dvornik for mumax2
// Modified for mumax3 by Arne Vansteenkiste, 2013

#include <stdint.h>
#include "float3.h"
#include "constants.h"

extern "C" __global__ void
addslonczewskitorque(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
                     float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz, float* __restrict__ jz,
                     float* __restrict__ pxLUT, float* __restrict__ pyLUT, float* __restrict__ pzLUT,
                     float* __restrict__ msatLUT, float* __restrict__ alphaLUT, float flt,
                     float* __restrict__ polLUT, float* __restrict__ lambdaLUT, float* __restrict__ epsilonPrimeLUT,
                     uint8_t* __restrict__ regions, int N) {

	int I =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (I < N) {

		float3 m = make_float3(mx[I], my[I], mz[I]);
		float  J = jz[I];

		// read parameters
		uint8_t region       = regions[I];

		float3 p            = normalized(make_float3(pxLUT[region], pyLUT[region], pzLUT[region]));
		float  Ms           = msatLUT[region];
		float  alpha        = alphaLUT[region];
		float  pol          = polLUT[region];
		float  lambda       = lambdaLUT[region];
		float  epsilonPrime = epsilonPrimeLUT[region];

		if (J == 0.0f || Ms == 0.0f) {
			return;
		}

		float beta    = (HBAR / QE) * (J / (flt*Ms) );
		float lambda2 = lambda * lambda;
		float epsilon = pol * lambda2 / ((lambda2 + 1.0f) + (lambda2 - 1.0f) * dot(p, m));

		float A = beta * epsilon;
		float B = beta * epsilonPrime;

		float gilb     = 1.0f / (1.0f + alpha * alpha);
		float mxpxmFac = gilb * (A - alpha * B);
		float pxmFac   = gilb * (B - alpha * A);

		float3 pxm      = cross(p, m);
		float3 mxpxm    = cross(m, pxm);

		tx[I] += mxpxmFac * mxpxm.x + pxmFac * pxm.x;
		ty[I] += mxpxmFac * mxpxm.y + pxmFac * pxm.y;
		tz[I] += mxpxmFac * mxpxm.z + pxmFac * pxm.z;
	}
}

