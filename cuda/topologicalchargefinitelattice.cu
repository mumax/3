#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"

// Set s to the toplogogical charge density for lattices
// Based on the solid angle subtended by triangle associated with three spins: a,b,c
// 	  s = 2 atan[(a . b x c /(1 + a.b + a.c + b.c)] / (dx dy)
// After M Boettcher et al, New J Phys 20, 103014 (2018), adapted from
// B. Berg and M. Luescher, Nucl. Phys. B 190, 412 (1981).
// This version is best for finite-sized lattices, but does not provide a useful local density.
// See topologicalchargefinitelattice.go.
extern "C" __global__ void
settopologicalchargefinitelattice(float* __restrict__ s,
                     float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                     int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

    int I = idx(ix, iy, iz);                      // central cell index

    float3 m0 = make_float3(mx[I], my[I], mz[I]); // +0
    float3 bxc = make_float3(0.0f, 0.0f, 0.0f);   // b x c
    int i_;                                       // neighbour index

    if(is0(m0))
    {
        s[I] = 0.0f;
        return;
    }

    // Assign neigbouring spins with the convention:
    // 0: (i,j)
    // 1: (i+1,j)
    // 2: (i,j+1)
    // 3: (i-1,j)
    // 4: (i,j-1)
    // a: (i+1,j+1)
    // The two main triangles are 012 and 034, with 01a and 0a2 being special edge cases
    // The index order is important because the triangles are **oriented**.
    float trig012, trig034, trig01a, trig0a2;
    float numer, denom;

    { 
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f);      // load neighbour m if inside grid, keep 0 otherwise
        i_ = idx(hclampx(ix+1), iy, iz);
        if (ix+1 < Nx || PBCx)
        {
            m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, hclampy(iy+1), iz);
        if  (iy+1 < Ny || PBCy)
        {
            m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m3 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(lclampx(ix-1), iy, iz);
        if (ix-1 >= 0 || PBCx)
        {
            m3 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m4 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, lclampy(iy-1), iz);
        if (iy-1 >= 0 || PBCy)
        {
            m4 = make_float3(mx[i_], my[i_], mz[i_]);
        }

	float3 ma = make_float3(0.0f, 0.0f, 0.0f);	// only case of next-nearest neighbour
        i_ = idx(hclampx(ix+1), lclampy(iy+1), iz);
        if ( (ix+1 < Nx || PBCx) && (iy+1 < Ny || PBCy) )
        {
            ma = make_float3(mx[i_], my[i_], mz[i_]);
        }

        // We don't care whether the neighbours exist, since the dot and
        // cross products will be zero if they don't
        // Triangle 012
        bxc     = cross(m1, m2);
        numer   = dot(m0, bxc);
        denom   = 1.0f + dot(m0, m1) + dot(m0, m2) + dot(m1, m2);
        trig012 = 2.0f * atan2(numer, denom);

        // Triangle 034
        bxc     = cross(m3, m4);
        numer   = dot(m0, bxc);
        denom   = 1.0f + dot(m0, m3) + dot(m0, m4) + dot(m3, m4);
        trig034 = 2.0f * atan2(numer, denom);

        // Special case 1: Triangle 01a
	trig01a = 0.0f;
	if ( dot(m2, m2)== 0.0f && dot(ma, ma) != 0.0f ) // If 2 doesn't exist, but a does
	{
        	bxc     = cross(m1, ma);
        	numer   = dot(m0, bxc);
        	denom   = 1.0f + dot(m0, m1) + dot(m0, ma) + dot(m1, ma);
        	trig01a = 2.0f * atan2(numer, denom);
	}

	// Special case 2: Triangle 0a2
	trig0a2	= 0.0f;
	if ( dot(m1, m1)== 0.0f && dot(ma, ma) != 0.0f ) // If 1 doesn't exist, but a does
	{
		bxc     = cross(ma, m2);
        	numer   = dot(m0, bxc);
        	denom   = 1.0f + dot(m0, ma) + dot(m0, m2) + dot(ma, m2);
        	trig0a2 = 2.0f * atan2(numer, denom);
	}
    
    }

   // Sum over all triangles
   s[I] = trig012 + trig034 + trig01a + trig0a2;
}
