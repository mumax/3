#include <stdint.h>
#include "float3.h"
#include "stencil.h"

// Calculate the vector potential in the gauge that
// A_x = ∫_-∞^y F_z dy'
// A_y = 0
// A_z = -∫_-∞^y F_x dy'

// We approximate these integrals using cumulative sums from the bottom of the system
// (i.e. minimum value of y) up to the cell at which A is to be calculated
// e.g. A_x = ∫_-∞^y F_z dy' ≈ Σ_{iy_ = 0}^{iy_ = iy-1} F_z(ix, iy_, iz) * cy

extern "C" __global__ void
setvectorpotential(float* __restrict__ Ax, float* __restrict__ Ay, float* __restrict__ Az,
                     float* __restrict__ Fx, float* __restrict__ Fy, float* __restrict__ Fz,
                     float cy, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

    int I = idx(ix, iy, iz);  // Index of cell of interest
    int i_;                   // Index of summand

    float3 a = make_float3(0.0f, 0.0f, 0.0f);

    for (int iy_ = 0; iy_ < iy; iy_++) {  // Cumulative sum along y-axis up to cell of interest within system
        i_ = idx(ix, iy_, iz);
        a.x -= Fz[i_] * cy;
        a.z += Fx[i_] * cy;
    }

    Ax[I] = a.x;
    Ay[I] = a.y;
    Az[I] = a.z;

}
