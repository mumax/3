#include "reduce.h"
#include "atomicf.h"

#define load_fabs(i) fabs(src[i])

extern "C" __global__ void
reducemaxabs(float* __restrict__ src, float* __restrict__ dst, float initVal, int n) {
	reduce(load_fabs, fmax, atomicFmaxabs)
}

