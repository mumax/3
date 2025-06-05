#include "stencil.h"

// Shifts a component `src` of a vector field by `shx` cells along the X-axis.
// Unlike the normal `shiftx()`, the new edge value is the current edge value.
//
// To avoid the situation where the magnetization could be set to (0,0,0) within the geometry, it is
// also required to pass the two other vector components `othercomp` and `anothercomp` to this function.
// In cells where the vector (`src`, `othercomp`, `anothercomp`) is the zero-vector,
// `clampL` or `clampR` is used for the component `src` instead.
extern "C" __global__ void
shiftedgecarryX(float* __restrict__  dst, float* __restrict__  src,
    float* __restrict__ othercomp, float* __restrict__ anothercomp,
    int Nx,  int Ny,  int Nz, int shx, float clampL, float clampR) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        int ix2 = ix-shx; // old X-index
        float newval;
        if (ix2 < 0) { // left edge (shifting right)
            newval = src[idx(0, iy, iz)];
            if (newval == 0 && othercomp[idx(0, iy, iz)] == 0 && anothercomp[idx(0, iy, iz)] == 0) { // If zero-vector
                newval = clampL;
            }
        } else if (ix2 >= Nx) { // right edge (shifting left)
            newval = src[idx(Nx-1, iy, iz)];
            if (newval == 0 && othercomp[idx(Nx-1, iy, iz)] == 0 && anothercomp[idx(Nx-1, iy, iz)] == 0) { // If zero-vector
                newval = clampR;
            }
        } else { // bulk, doesn't matter which way the shift is
            newval = src[idx(ix2, iy, iz)];
        }
        dst[idx(ix, iy, iz)] = newval;
    }
}
