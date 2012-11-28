#include "reduce.h"
#include "atomicf.h"

#define load_diff(i) fabs(src1[i] - src2[i])

extern "C" __global__ void
reducemaxdiff(float *src1, float* src2, float *dst, int n) {
	reduce(load_diff, fmax, atomicFmax, -3.4028234663852886e38)
}

