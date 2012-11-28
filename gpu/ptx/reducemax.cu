#include "reduce.h"
#include "atomicf.h"

extern "C" __global__ void
reducemax(float *src, float *dst, int n) {
	reduce(load_ident, fmax, atomicFmax, -3.4028234663852886e38)
}

