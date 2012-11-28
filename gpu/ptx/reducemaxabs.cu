#include "reduce.h"
#include "atomicf.h"

#define load_fabs(i) fabs(src[i])

extern "C" __global__ void
reducemaxabs(float *src, float *dst, int n) {
	reduce(load_fabs, fmax, atomicFmax, -3.4028234663852886e38)
}

