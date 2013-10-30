#include "stencil.h"

// Copy src (size S, smaller) into dst (size D, larger),
// and multiply by Bsat as defined in regions.
extern "C" __global__ void
copypadmul(float* __restrict__ dst, int Dx, int Dy, int Dz,
           float* __restrict__ src, float* __restrict__ vol, int Sx, int Sy, int Sz,
           float* __restrict__ BsatLUT, int8_t* __restrict__ regions) {

    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    if (iz<Sz && iy<Sy && ix<Sx) {
        int sI = index(ix, iy, iz, Sx, Sy, Sz); //(iz*Sy + iy)*Sx + k; // source index
        float Bsat = BsatLUT[regions[sI]];
        float v = (vol == NULL? 1.0f: vol[sI]);
        dst[index(ix, iy, iz, Dx, Dy, Dz)] = Bsat * v * src[sI];
    }
}

