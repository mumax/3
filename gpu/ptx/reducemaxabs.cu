#include "reduce.h"
#include "atomicf.h"

extern "C" __global__ void
reducemaxabs(float *src, float *dst, int n) {
	reduce(fabs, fmax, atomicFmax, -3.4028234663852886e38)
}

