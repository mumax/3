#include "reduce.h"

inline __device__ void atomicFmax(float* a, float b){
	atomicMax((int*)(a), *((int*)(&b)));
}

extern "C" __global__ void
reducemax(float *src, float *dst, int n) {
	reduce(ident, fmax, atomicFmax, -3.4028234663852886e38)
}

