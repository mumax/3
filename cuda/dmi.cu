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
       float cx, float cy, float cz, int N0, int N1, int N2) {

    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N0 || j >= N1 || k >= N2) {
        return;
    }

    int I = idx(i, j, k);                        // central cell index
    float3 h = make_float3(Hx[I], Hy[I], Hz[I]); // add to H
    float3 m = make_float3(mx[I], my[I], mz[I]); // central m
    float3 m1, m2;                               // right, left neighbor

    // z derivatives (along length)
    {
        m1 = make_float3(0.0f, 0.0f, 0.0f);
        m2 = make_float3(0.0f, 0.0f, 0.0f);

        // load neighbor m if inside grid, 0 otherwise
        if (k+1 < N2) {
            int I1 = idx(i, j, k+1);
            m1 = make_float3(mx[I1], my[I1], mz[I1]);
        }
        if (k-1 >= 0) {
            int I2 = idx(i, j, k-1);
            m2 = make_float3(mx[I2], my[I2], mz[I2]);
        }

        // BC's for zero cells (either due to grid or hole boundaries)
        float Dz_2A = (Dz/(2.0f*A));
        if (len(m1) == 0.0f) {
            m1.x = m.x - (cz * Dz_2A * m.z);
            m1.y = m.y;
            m1.z = m.z + (cz * Dz_2A * m.x);
        }
        if (len(m2) == 0.0f) {
            m2.x = m.x + (cz * Dz_2A * m.z);
            m2.y = m.y;
            m2.z = m.z - (cz * Dz_2A * m.x);
        }

        h   += (2.0f*A/(cz*cz)) * ((m1 - m) + (m2 - m)); // exchange
        h.x -= Dz*(m1.z-m2.z)/cz;                        // DMI
        h.z += Dz*(m1.x-m2.x)/cz;
        // note: actually 2*D * delta / (2*c)
    }

    // y derivatives (along height)
    {
        m1 = make_float3(0.0f, 0.0f, 0.0f);
        m2 = make_float3(0.0f, 0.0f, 0.0f);

        if (j+1 < N1) {
            int I1 = idx(i, j+1, k);
            m1 = make_float3(mx[I1], my[I1], mz[I1]);
        }
        if (j-1 >= 0) {
            int I2 = idx(i, j-1, k);
            m2 = make_float3(mx[I2], my[I2], mz[I2]);
        }

        float Dy_2A = (Dy/(2.0f*A));
        if (len(m1) == 0.0f) {
            m1.x = m.x - (cy * Dy_2A * m.y);
            m1.y = m.y + (cy * Dy_2A * m.x);
            m1.z = m.z;
        }
        if (len(m2) == 0.0f) {
            m2.x = m.x + (cy * Dy_2A * m.y);
            m2.y = m.y - (cy * Dy_2A * m.x);
            m2.z = m.z;
        }

        h   += (2.0f*A/(cy*cy)) * ((m1 - m) + (m2 - m));
        h.x -= Dy*(m1.y-m2.y)/cy;
        h.y += Dy*(m1.x-m2.x)/cy;
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
