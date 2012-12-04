#include "reduce.h"
#include "atomicf.h"

#define load(i) src[i]

extern "C" __global__ void
reducemax(float *src, float *dst, float initVal, int n) {
	reduce(load, fmax, atomicFmax)
}

