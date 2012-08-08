#ifndef alpha_h
#define alpha_h

#include "load.h"

extern __device__ masker alpha_mask;

#define load_alpha(out, i) load_masker(out, alpha_mask, i)

#endif
