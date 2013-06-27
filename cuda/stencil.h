#ifndef _STENCIL_H_
#define _STENCIL_H_

// clamps i between 0 and N-1
#define clamp(i, N) min( max((i), 0), (N)-1 )

// clamps i to positive values
#define lclamp(i) max((i), 0)

// clamps i to < N
#define hclamp(i, N) min((i), (N)-1)

// 3D array indexing
#define idx(i,j,k) (N2*((i)*N1 + (j)) + (k))

// clamp index to bounds (0:N0, 0:N1, 0:N2)
#define idxclamp(i, j, k) idx(clamp(i, N0), clamp(j, N1), clamp(k, N2))

// spatial derivatives without dividing by cell size
#define deltax(in) (in[idx(hclamp(i+1, N0), j, k)] - in[idx(lclamp(i-1), j, k)])
#define deltay(in) (in[idx(i, hclamp(j+1, N1), k)] - in[idx(i, lclamp(j-1), k)])
#define deltaz(in) (in[idx(i, j, hclamp(k+1, N2))] - in[idx(i, j, lclamp(k-1))])

#endif

