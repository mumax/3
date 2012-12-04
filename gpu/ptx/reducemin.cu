#include "reduce.h"
#include "atomicf.h"

#define load(i) src[i]

extern "C" __global__ void
reducemin(float *src, float *dst, float initVal, int n) {
	reduce(load, fmin, atomicFmin)
}

