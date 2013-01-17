#include "reduce.h"
#include "sum.h"

#define load(i) src[i]

extern "C" __global__ void
reducesum(float* __restrict__ src, float*__restrict__  dst, float initVal, int n) {
	reduce(load, sum, atomicAdd)
}

