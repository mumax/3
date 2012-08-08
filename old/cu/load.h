#ifndef load_h
#define load_h

#include "geom.h"

/// masker's value is array[i]*value.
/// If array == NULL, it represents uniform 1's.
typedef struct{
	float* array;
	float value;
}masker;

#define load_masker(out, in, i){\
	if(in.array == NULL){\
		out = in.value;\
	}else{\
		out = in.value * in.array[i];\
	}\
}

#define load_vector(out_x, out_y, out_z, in, i){\
	out_x = in[0*Ncell + i]; \
	out_y = in[1*Ncell + i]; \
	out_z = in[2*Ncell + i]; \
}

#define load_uniformscalar(out, in)\
	out = in;

#endif
