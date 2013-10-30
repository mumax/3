#ifndef _STENCIL_H_
#define _STENCIL_H_

// clamps i between 0 and N-1
#define clamp(i, N) min( max((i), 0), (N)-1 )

// clamps i to positive values
#define lclamp(i) max((i), 0)

// clamps i to < N
#define hclamp(i, N) min((i), (N)-1)

// 3D array indexing
#define index(ix,iy,iz,Nx,Ny,Nz) ( ( (iz)*(Ny) + (iy) ) * (Nx) + (ix) )
#define idx(ix,iy,iz) ( index((ix),(iy),(iz),(Nx),(Ny),(Nz)) )

// clamp index to bounds (0:Nx, 0:Ny, 0:Nz)
#define idxclamp(ix, iy, iz) idx(clamp(ix, Nx), clamp(iy, Ny), clamp(iz, Nz))


// pbc clamps

#define MOD(n, M) ( (( (n) % (M) ) + (M) ) % (M)  )

#define PBCx (PBC & 4)
#define PBCy (PBC & 2)
#define PBCz (PBC & 1)

#define hclampx(ix) (PBCx? MOD(ix, Nx) : min((ix), Nx-1))
#define lclampx(ix) (PBCx? MOD(ix, Nx) : max((ix), 0))

#define hclampy(iy) (PBCy? MOD(iy, Ny) : min((iy), Ny-1))
#define lclampy(iy) (PBCy? MOD(iy, Ny) : max((iy), 0))

#define hclampz(iz) (PBCz? MOD(iz, Nz) : min((iz), Nz-1))
#define lclampz(iz) (PBCz? MOD(iz, Nz) : max((iz), 0))


// spatial derivatives without dividing by cell size
#define deltax(in) (in[idx(hclampx(ix+1), iy, iz)] - in[idx(lclampx(ix-1), iy, iz)])
#define deltay(in) (in[idx(ix, hclampy(iy+1), iz)] - in[idx(ix, lclampy(iy-1), iz)])
#define deltaz(in) (in[idx(ix, iy, hclampz(iz+1))] - in[idx(ix, iy, lclampz(iz-1))])

#endif

