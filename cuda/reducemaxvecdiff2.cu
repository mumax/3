#include "reduce.h"
#include "atomicf.h"
#include "float3.h"

#define load_vecdiff2(i)  \
	pow2(x1[i] - x2[i]) + \
	pow2(y1[i] - y2[i]) + \
	pow2(z1[i] - z2[i])   \
 
extern "C" __global__ void
reducemaxvecdiff2(float* __restrict__ x1, float* __restrict__ y1, float* __restrict__ z1,
                  float* __restrict__ x2, float* __restrict__ y2, float* __restrict__ z2,
                  float* __restrict__ dst, float initVal, int n) {
	reduce(load_vecdiff2, fmax, atomicFmaxabs)
}

