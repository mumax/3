#include "reduce.h"
#include "atomicf.h"

#define load(i) src[i]

extern "C" __global__ void
reducemax(float *src, float *dst, int n) {
	reduce(load, fmax, atomicFmax, -3.4028234663852886e38)
}

