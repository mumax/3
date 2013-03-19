#ifndef _COMMON_STENCIL_H_
#define _COMMON_STENCIL_H_

// clamps i between 0 and N-1
#define clamp(i, N) min( max((i), 0), (N)-1 )

// clamps i to positive values
#define lclamp(i) max((i), 0)

// clamps i to < N
#define hclamp(i, N) min((i), (N)-1)

// 3D array indexing
#define idx(i,j,k) ((i)*N1*N2 + (j)*N2 + (k))

// Maximum threads per block for stencil op
#define STENCIL_BLOCKSIZE_X 16
#define STENCIL_MAXTHREADS (STENCIL_BLOCKSIZE_X * STENCIL_BLOCKSIZE_X)

#endif

