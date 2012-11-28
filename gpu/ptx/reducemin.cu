#include "reduce.h"
#include "atomicf.h"

#define load(i) src[i]

extern "C" __global__ void
reducemin(float *src, float *dst, int n) {
	reduce(load, fmin, atomicFmin, 3.4028234663852886e38)
}

