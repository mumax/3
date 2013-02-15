#include "reduce.h"
#include "atomicf.h"

#define load_diff(i) fabs(src1[i] - src2[i])

extern "C" __global__ void
reducemaxdiff(float* __restrict__ src1, float* __restrict__  src2, float* __restrict__ dst, float initVal, int n) {
	reduce(load_diff, fmax, atomicFmaxabs)
}

