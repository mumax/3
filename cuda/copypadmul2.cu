#include "amul.h"
#include "constants.h"
#include "stencil.h"
#include <stdint.h>

// Copy src (size S, smaller) into dst (size D, larger),
// and multiply by Bsat * vol
extern "C" __global__ void
copypadmul2(float* __restrict__ dst, int Dx, int Dy, int Dz,
            float* __restrict__ src, int Sx, int Sy, int Sz,
            float* __restrict__ Ms_, float Ms_mul,
            float* __restrict__ vol) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix<Sx && iy<Sy && iz<Sz) {
        int sI = index(ix, iy, iz, Sx, Sy, Sz);  // source index
        float Bsat = MU0 * amul(Ms_, Ms_mul, sI);
        float v = amul(vol, 1.0f, sI);
        dst[index(ix, iy, iz, Dx, Dy, Dz)] = Bsat * v * src[sI];
    }
}

