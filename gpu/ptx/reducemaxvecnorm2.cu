#include "reduce.h"
#include "atomicf.h"
#include "common_func.h"

#define load_vecnorm2(i) \
	sqr(x[i]) + sqr(y[i]) +  sqr(z[i])

extern "C" __global__ void
reducemaxvecnorm2(float *x, float *y, float *z, float *dst, int n) {
	reduce(load_vecnorm2, fmax, atomicFmax, -3.4028234663852886e38)
}

