#ifndef _COMMON_STENCIL_H_
#define _COMMON_STENCIL_H_

#include "float3.h"

// clamps i between 0 and N-1
inline __device__ int clamp(int i, int N){
	return min( max(i, 0) , N-1 );
}

// 3D array indexing
#define idx(i,j,k) ((i)*N1*N2 + (j)*N2 + (k))

//inline __device__ float3

#endif

