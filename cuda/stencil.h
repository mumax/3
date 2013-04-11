#ifndef _STENCIL_H_
#define _STENCIL_H_

// clamps i between 0 and N-1
#define clamp(i, N) min( max((i), 0), (N)-1 )

// clamps i to positive values
#define lclamp(i) max((i), 0)

// clamps i to < N
#define hclamp(i, N) min((i), (N)-1)

// 3D array indexing
#define idx(i,j,k) ((i)*N1*N2 + (j)*N2 + (k))

// clamp index to bounds (0:N0, 0:N1, 0:N2)
#define ix(i, j, k) idx(clamp(i, N0), clamp(j, N1), clamp(k, N2))

// spatial derivative along (u, v, w) direction without dividing by cell size
#define delta(in, u, v, w) (in[ix(i+u, j+v, k+w)] - in[ix(i-u, j-v, k-w)])

// spatial derivative along (u, v, w) direction with given cell size
#define diff(in, u, v, w, c) ((delta(in, u, v, w))/(2*c))

// Maximum threads per block for stencil op
#define STENCIL_BLOCKSIZE_X 16
#define STENCIL_MAXTHREADS (STENCIL_BLOCKSIZE_X * STENCIL_BLOCKSIZE_X)

#endif

