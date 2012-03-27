#ifndef alpha_h
#define alpha_h

#include "load.h"

extern __device__ float alpha_value;

#define load_alpha(out, i) load_uniformscalar(out, alpha_value)

#endif
