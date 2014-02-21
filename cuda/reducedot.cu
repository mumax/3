#include "reduce.h"
#include "sum.h"

#define load_prod(i) (x1[i] * x2[i])

extern "C" __global__ void
reducedot(float* __restrict__ x1, float* __restrict__ x2,
          float*__restrict__  dst, float initVal, int n) {
	reduce(load_prod, sum, atomicAdd)
}

