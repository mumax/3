#include "stencil.h"

// Shifts a component `src` of a vector field by `shy` cells along the Y-axis.
// Unlike the normal `shifty()`, the new edge value is the current edge value.
//
// To avoid the situation where the magnetization could be set to (0,0,0) within the geometry, it is
// also required to pass the two other vector components `othercomp` and `anothercomp` to this function.
// In cells where the vector (`src`, `othercomp`, `anothercomp`) is the zero-vector,
// `clampD` or `clampU` is used for the component `src` instead.
extern "C" __global__ void
shiftedgecarryY(float* __restrict__  dst, float* __restrict__  src,
    float* __restrict__ othercomp, float* __restrict__ anothercomp,
    int Nx,  int Ny,  int Nz, int shy, float clampD, float clampU) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        int iy2 = iy-shy; // old Y-index
        float newval;
        if (iy2 < 0) { // bottom edge (shifting up)
            newval = src[idx(ix, 0, iz)];
            if (newval == 0 && othercomp[idx(ix, 0, iz)] == 0 && anothercomp[idx(ix, 0, iz)] == 0) { // If zero-vector
                newval = clampD;
            }
        } else if (iy2 >= Ny) { // top edge (shifting down)
            newval = src[idx(ix, Ny-1, iz)];
            if (newval == 0 && othercomp[idx(ix, Ny-1, iz)] == 0 && anothercomp[idx(ix, Ny-1, iz)] == 0) { // If zero-vector
                newval = clampU;
            }
        } else { // bulk, doesn't matter which way the shift is
            newval = src[idx(ix, iy2, iz)];
        }
        dst[idx(ix, iy, iz)] = newval;
    }
}
