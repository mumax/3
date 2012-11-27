#include "reduce.h"

inline __device__ void atomicFmin(float* a, float b){
	atomicMin((int*)(a), *((int*)(&b)));
}

extern "C" __global__ void
reducemin(float *src, float *dst, int n) {
	reduce(ident, fmin, atomicFmin, 3.4028234663852886e38)
}

