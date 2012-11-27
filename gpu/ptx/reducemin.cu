#include "reduce.h"
#include "atomicf.h"

extern "C" __global__ void
reducemin(float *src, float *dst, int n) {
	reduce(ident, fmin, atomicFmin, 3.4028234663852886e38)
}

