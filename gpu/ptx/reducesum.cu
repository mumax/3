#include "reduce.h"

extern "C" __global__ void
reducesum(float *src, float *dst, int n) {
	reduce(sum)
}

