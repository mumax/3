#include "reduce.h"

inline __device__ float sum(float a, float b){
	return a + b;
}

extern "C" __global__ void
reducesum(float *src, float *dst, int n) {
	reduce(ident, sum, atomicAdd, 0)
}

