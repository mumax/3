#include <stdint.h>
#include "stencil.h"
#include "float3.h"

// Exchange + Dzyaloshinskii-Moriya interaction according to
// Bagdanov and Röβler, PRL 87, 3, 2001. eq.8 (out-of-plane symmetry breaking).
// Taking into account proper boundary conditions.
// m: normalized magnetization
// H: effective field in Tesla
// D: dmi strength / Msat, in Tesla*m
// A: Aex/Msat
extern "C" __global__ void
adddmi(float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
       float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
       float Dx, float Dy, float Dz, float A,
       float cx, float cy, float cz, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);                     // central cell index
    float3 h = make_float3(Hx[I], Hy[I], Hz[I]); // add to H
    float3 m0 = make_float3(mx[I], my[I], mz[I]); // central m
    int i_;

    if(is0(m0)) {
        return;
    }

    // x derivatives (along length)
    {
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f); // left neighbor
        float3 m2 = make_float3(0.0f, 0.0f, 0.0f); // right neighbor

        // load neighbor m if inside grid, 0 otherwise
        i_ = idx(lclampx(ix-1), iy, iz);
        if (ix-1 >= 0 || PBCx) {
            m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        i_ = idx(hclampx(ix+1), iy, iz);
        if (ix+1 < Nx || PBCx) {
            m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        // BC's for zero cells (either due to grid or hole boundaries)
        float Dx_2A = (Dx/(2.0f*A));
        if (is0(m1)) {             // left neigbor missing
            m1.x = m0.x - (-cx * Dx_2A * m0.z);  // extrapolate to left (-cx)
            m1.y = m0.y;
            m1.z = m0.z + (-cx * Dx_2A * m0.x);
        }
        if (is0(m2)) {            // right neigbor missing
            m2.x = m0.x - (cx * Dx_2A * m0.z);  // extrapolate to right (+cx)
            m2.y = m0.y;
            m2.z = m0.z + (cx * Dx_2A * m0.x);
        }

        h   += (2.0f*A/(cx*cx)) * ((m1 - m0) + (m2 - m0)); // exchange
        h.x += Dx*(m2.z-m1.z)/cx;
        h.z -= Dx*(m2.x-m1.x)/cx;
        // note: actually 2*D * delta / (2*c)
    }

    // y derivatives (along height)
    {
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
        float3 m2 = make_float3(0.0f, 0.0f, 0.0f);

        i_ = idx(ix, lclampy(iy-1), iz);
        if (iy-1 >= 0 || PBCy) {
            m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        i_ = idx(ix, hclampy(iy+1), iz);
        if  (iy+1 < Ny || PBCy) {
            m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float Dy_2A = (Dy/(2.0f*A));
        if (is0(m1)) {
            m1.x = m0.x;
            m1.y = m0.y - (-cy * Dy_2A * m0.z);
            m1.z = m0.z + (-cy * Dy_2A * m0.y);
        }
        if (is0(m2)) {
            m2.x = m0.x;
            m2.y = m0.y - (cy * Dy_2A * m0.z);
            m2.z = m0.z + (cy * Dy_2A * m0.y);
        }

        h   += (2.0f*A/(cy*cy)) * ((m1 - m0) + (m2 - m0));
        h.y += Dy*(m2.z-m1.z)/cy;
        h.z -= Dy*(m2.y-m1.y)/cy;
    }

    // write back, result is H + Hdmi + Hex
    Hx[I] = h.x;
    Hy[I] = h.y;
    Hz[I] = h.z;
}

// Note on boundary conditions.
//
// We need the derivative and laplacian of m in point A, but e.g. C lies out of the boundaries.
// We use the boundary condition in B (derivative of the magnetization) to extrapolate m to point C:
// 	m_C = m_A + (dm/dx)|_B * cellsize
//
// When point C is inside the boundary, we just use its actual value.
//
// Then we can take the central derivative in A:
// 	(dm/dx)|_A = (m_C - m_D) / (2*cellsize)
// And the laplacian:
// 	lapl(m)|_A = (m_C + m_D - 2*m_A) / (cellsize^2)
//
// All these operations should be second order as they involve only central derivatives.
//
//    ------------------------------------------------------------------ *
//   |                                                   |             C |
//   |                                                   |          **   |
//   |                                                   |        ***    |
//   |                                                   |     ***       |
//   |                                                   |   ***         |
//   |                                                   | ***           |
//   |                                                   B               |
//   |                                               *** |               |
//   |                                            ***    |               |
//   |                                         ****      |               |
//   |                                     ****          |               |
//   |                                  ****             |               |
//   |                              ** A                 |               |
//   |                         *****                     |               |
//   |                   ******                          |               |
//   |          *********                                |               |
//   |D ********                                         |               |
//   |                                                   |               |
//   +----------------+----------------+-----------------+---------------+
//  -1              -0.5               0               0.5               1
//                                 x
