#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"

// Returns the topological charge contribution on an elementary triangle ijk
// Note: the result is zero if an argument is zero, or when two arguments are the same
__device__ inline float triangleCharge(float3 mi, float3 mj, float3 mk) {
    float numer   = dot(mi, cross(mj, mk));
    float denom   = 1.0f + dot(mi, mj) + dot(mi, mk) + dot(mj, mk);
    return 2.0f * atan2(numer, denom);
}

// Set s to the toplogogical charge density for lattices
// Based on the solid angle subtended by triangle associated with three spins: a,b,c
// 	  s = 2 atan[(a . b x c /(1 + a.b + a.c + b.c)] / (dx dy)
// After M Boettcher et al, New J Phys 20, 103014 (2018), adapted from
// B. Berg and M. Luescher, Nucl. Phys. B 190, 412 (1981).
// A unit cell comprises two triangles, but s is a site-dependent quantity so we
// double-count and average over four triangles.
// This implementation works best for extended systems with periodic boundaries and provides a
// workable definition of the local charge density.
extern "C" __global__ void
settopologicalchargelattice(float* __restrict__ s,
                     float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                     float icxcy, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int i0 = idx(ix, iy, iz); // central cell index

    float3 m0 = make_float3(mx[i0], my[i0], mz[i0]);

    if(is0(m0)) {
        s[i0] = 0.0f;
        return;
    }

    // indices of the 4 neighbors
    int i1 = idx(hclampx(ix+1), iy, iz); // (i+1,j)
    int i2 = idx(ix, hclampy(iy+1), iz); // (i,j+1)
    int i3 = idx(lclampx(ix-1), iy, iz); // (i-1,j)
    int i4 = idx(ix, lclampy(iy-1), iz); // (i,j-1)

    // magnetization of the 4 neighbors
    float3 m1 = make_float3(mx[i1], my[i1], mz[i1]);
    float3 m2 = make_float3(mx[i2], my[i2], mz[i2]);
    float3 m3 = make_float3(mx[i3], my[i3], mz[i3]);
    float3 m4 = make_float3(mx[i4], my[i4], mz[i4]);

    float topcharge = 0.0; // local topological charge (accumulator)
    float weight = 0.5;    // avoid double counting of charge contributions

    // order of arguments is important here to preserve the same measure of chirality!
    topcharge += weight * triangleCharge(m0,m1,m2);  
    topcharge += weight * triangleCharge(m0,m2,m3);
    topcharge += weight * triangleCharge(m0,m3,m4);
    topcharge += weight * triangleCharge(m0,m4,m1);

    s[i0] = icxcy * topcharge;
}
