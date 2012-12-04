#include "reduce.h"

inline __device__ float sum(float a, float b){
	return a + b;
}

#define load(i) src[i]

extern "C" __global__ void
reducesum(float* __restrict__ src, float*__restrict__  dst, float initVal, int n) {
	reduce(load, sum, atomicAdd)
}

