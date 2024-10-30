#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// The elements of the 3x3x3 DMI tensor are stored in an array with 27 elements
// The following directives for the indices are used to avoid apparent magic numbers
#define XXX 0
#define XXY 1
#define XXZ 2
#define XYX 3
#define XYY 4
#define XYZ 5
#define XZX 6
#define XZY 7
#define XZZ 8
#define YXX 9
#define YXY 10
#define YXZ 11
#define YYX 12
#define YYY 13
#define YYZ 14
#define YZX 15
#define YZY 16
#define YZZ 17
#define ZXX 18
#define ZXY 19
#define ZXZ 20
#define ZYX 21
#define ZYY 22
#define ZYZ 23
#define ZZX 24
#define ZZY 25
#define ZZZ 26

extern "C" __global__ void
adddmitensor(float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
             float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
             float* __restrict__ Ms_, float Ms_mul, float* __restrict__ D,
             uint8_t* __restrict__ regions,
             float cx, float cy, float cz, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);                      // central cell index
    float3 h = make_float3(0.0,0.0,0.0);          // add to H
    float3 m0 = make_float3(mx[I], my[I], mz[I]); // central m

    if(is0(m0)) {
        return;
    }


    // x direction
    {
        // left neighbor
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
        if (ix-1 >= 0 || PBCx) {                      // check if neighbor is in simulation box
            int i_ = idx(lclampx(ix-1), iy, iz);      // index of neighbor
            m1 = make_float3(mx[i_], my[i_], mz[i_]); // magnetization of neighbor
        }
        // right neighbor
        float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
        if (ix+1 < Nx || PBCx) {
            int i_ = idx(hclampx(ix+1), iy, iz);
            m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        // bulk DMI
        h.x += ((D[XXY]-D[XYX])*(m1.y-m2.y)+(D[XXZ]-D[XZX])*(m1.z-m2.z))/(2*cx);
        h.y += ((D[XYZ]-D[XZY])*(m1.z-m2.z)+(D[XYX]-D[XXY])*(m1.x-m2.x))/(2*cx);
        h.z += ((D[XZX]-D[XXZ])*(m1.x-m2.x)+(D[XZY]-D[XYZ])*(m1.y-m2.y))/(2*cx);
        // boundary induced DMI
        int sign = !is0(m1) - !is0(m2);
        h.x += sign*((D[XXX]+D[XXX])*m0.x+(D[XXY]+D[XYX])*m0.y+(D[XXZ]+D[XZX])*m0.z)/(2*cx);
        h.y += sign*((D[XYY]+D[XYY])*m0.y+(D[XYZ]+D[XZY])*m0.z+(D[XYX]+D[XXY])*m0.x)/(2*cx);
        h.z += sign*((D[XZZ]+D[XZZ])*m0.z+(D[XZX]+D[XXZ])*m0.x+(D[XZY]+D[XYZ])*m0.y)/(2*cx);
    }


    // y direction
    {
        // back neighbor
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
        if (iy-1 >= 0 || PBCy) {
            int i_ = idx(ix, lclampy(iy-1), iz);
            m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        // front neighbor
        float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
        if (iy+1 < Ny || PBCy) {
            int i_ = idx(ix, hclampy(iy+1), iz);
            m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        // bulk DMI
        h.x += ((D[YXY]-D[YYX])*(m1.y-m2.y)+(D[YXZ]-D[YZX])*(m1.z-m2.z))/(2*cy);
        h.y += ((D[YYZ]-D[YZY])*(m1.z-m2.z)+(D[YYX]-D[YXY])*(m1.x-m2.x))/(2*cy);
        h.z += ((D[YZX]-D[YXZ])*(m1.x-m2.x)+(D[YZY]-D[YYZ])*(m1.y-m2.y))/(2*cy);
        // boundary induced DMI
        int sign = !is0(m2) - !is0(m1);
        h.x += sign*((D[YXX]+D[YXX])*m0.x+(D[YXY]+D[YYX])*m0.y+(D[YXZ]+D[YZX])*m0.z)/(2*cy);
        h.y += sign*((D[YYY]+D[YYY])*m0.y+(D[YYZ]+D[YZY])*m0.z+(D[YYX]+D[YXY])*m0.x)/(2*cy);
        h.z += sign*((D[YZZ]+D[YZZ])*m0.z+(D[YZX]+D[YXZ])*m0.x+(D[YZY]+D[YYZ])*m0.y)/(2*cy);
    }


    // z direction
    {
        // bottom neighbor
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
        if (iz-1 >= 0 || PBCz) {
            int i_ = idx(ix, iy, lclampz(iz-1));
            m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        // top neighbor
        float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
        if (iz+1 < Nz || PBCz) {
            int i_ = idx(ix, iy, hclampz(iz+1));
            m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        // bulk DMI
        h.x += ((D[ZXY]-D[ZYX])*(m1.y-m2.y)+(D[ZXZ]-D[ZZX])*(m1.z-m2.z))/(2*cz);
        h.y += ((D[ZYZ]-D[ZZY])*(m1.z-m2.z)+(D[ZYX]-D[ZXY])*(m1.x-m2.x))/(2*cz);
        h.z += ((D[ZZX]-D[ZXZ])*(m1.x-m2.x)+(D[ZZY]-D[ZYZ])*(m1.y-m2.y))/(2*cz);
        // boundary induced DMI
        int sign = !is0(m1) - !is0(m2);
        h.x += sign*((D[ZXX]+D[ZXX])*m0.x+(D[ZXY]+D[ZYX])*m0.y+(D[ZXZ]+D[ZZX])*m0.z)/(2*cz);
        h.y += sign*((D[ZYY]+D[ZYY])*m0.y+(D[ZYZ]+D[ZZY])*m0.z+(D[ZYX]+D[ZXY])*m0.x)/(2*cz);
        h.z += sign*((D[ZZZ]+D[ZZZ])*m0.z+(D[ZZX]+D[ZXZ])*m0.x+(D[ZZY]+D[ZYZ])*m0.y)/(2*cz);
    }


    float invMs = inv_Msat(Ms_, Ms_mul, I);
    Hx[I] -= h.x*invMs;
    Hy[I] -= h.y*invMs;
    Hz[I] -= h.z*invMs;
}

