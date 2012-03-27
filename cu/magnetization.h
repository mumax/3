#ifndef magnetization_h
#define magnetization_h

#include "geom.h"

__device__ extern float* magnetization;

#define load_magnetization(mx, my, mz, i)\
	mx = magnetization[0*N+i]; \
	my = magnetization[1*N+i]; \
	mz = magnetization[2*N+i];

#endif
