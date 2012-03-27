#ifndef load_h
#define load_h

#include "geom.h"

#define load_vector(out_x, out_y, out_z, in, i)\
	out_x = in[0*Ncell + i]; \
	out_y = in[1*Ncell + i]; \
	out_z = in[2*Ncell + i];

#define load_uniformscalar(out, in)\
	out = in;

#endif
