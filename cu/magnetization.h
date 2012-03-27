#ifndef magnetization_h
#define magnetization_h

#include "load.h"

__device__ extern float* magnetization_array;

#define load_magnetization(mx, my, mz, i) load_vector(mx, my, mz, magnetization_array, i)

#endif
