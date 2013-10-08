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


// pbc clamps

#define MOD(n, M) ( (( (n) % (M) ) + (M) ) % (M)  )

#define PBC0 (PBC & 1)
#define PBC1 (PBC & 2)
#define PBC2 (PBC & 4)

#define hclamp0(i) (PBC0? MOD(i, N0) : min((i), N0-1))
#define lclamp0(i) (PBC0? MOD(i, N0) : max((i), 0))

#define hclamp1(i) (PBC1? MOD(i, N1) : min((i), N1-1))
#define lclamp1(i) (PBC1? MOD(i, N1) : max((i), 0))

#define hclamp2(i) (PBC2? MOD(i, N2) : min((i), N2-1))
#define lclamp2(i) (PBC2? MOD(i, N2) : max((i), 0))


// spatial derivatives without dividing by cell size
#define deltax(in) (in[idx(hclamp0(i+1), j, k)] - in[idx(lclamp0(i-1), j, k)])
#define deltay(in) (in[idx(i, hclamp1(j+1), k)] - in[idx(i, lclamp1(j-1), k)])
#define deltaz(in) (in[idx(i, j, hclamp2(k+1))] - in[idx(i, j, lclamp2(k-1))])

#endif

