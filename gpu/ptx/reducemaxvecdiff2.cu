#include "reduce.h"
#include "atomicf.h"
#include "common_func.h"

#define load_vecdiff2(i)  \
	sqr(x1[i] - x2[i]) + \
	sqr(y1[i] - y2[i]) + \
	sqr(z1[i] - z2[i])   \

extern "C" __global__ void
reducemaxvecdiff2(float *x1, float *y1, float *z1,
                  float *x2, float *y2, float *z2,
                  float *dst, float initVal, int n) {
	reduce(load_vecdiff2, fmax, atomicFmax)
}

