#include "stencil.h"

 // shift dst by shx cells (positive or negative) along X-axis.
 // new edge value is the current edge value.
 extern "C" __global__ void
 shiftfudgex(float* __restrict__  dst, float* __restrict__  src,
        int Nx,  int Ny,  int Nz, int shx) {

     int ix = blockIdx.x * blockDim.x + threadIdx.x;
     int iy = blockIdx.y * blockDim.y + threadIdx.y;
     int iz = blockIdx.z * blockDim.z + threadIdx.z;

     if(ix < Nx && iy < Ny && iz < Nz) {
         int ix2 = ix-shx;
         float newval;
         if (ix2 < 0) {
             newval = src[idx(0, iy, iz)];
         } else if (ix2 >= Nx) {
             newval = src[idx(Nx-1, iy, iz)];
         } else {
             newval = src[idx(ix2, iy, iz)];
         }
         dst[idx(ix, iy, iz)] = newval;
     }
 }
