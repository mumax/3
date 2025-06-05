#include <stdint.h>
#include "float3.h"
#include "stencil.h"

// Sets the emergent magnetic field F_i = (1/8π) ε_{ijk} m · (∂m/∂x_j × ∂m/∂x_k)
// See hopfindex-two-point.go
extern "C" __global__ void
setemergentmagneticfieldtwopoint(float* __restrict__ Fx, float* __restrict__ Fy, float* __restrict__ Fz,
                     float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                     float prefactor, float icycz, float iczcx, float icxcy,
                     int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

    int I = idx(ix, iy, iz);  // central cell index

    float3 m0 = make_float3(mx[I], my[I], mz[I]);     // +0
    float3 dmdx = make_float3(0.0f, 0.0f, 0.0f);      // ∂m/∂x
    float3 dmdy = make_float3(0.0f, 0.0f, 0.0f);      // ∂m/∂y
    float3 dmdz = make_float3(0.0f, 0.0f, 0.0f);      // ∂m/∂y
    float3 dmdy_x_dmdz = make_float3(0.0, 0.0, 0.0);  // ∂m/∂y × ∂m/∂z
    float3 dmdz_x_dmdx = make_float3(0.0, 0.0, 0.0);  // ∂m/∂z × ∂m/∂x
    float3 dmdx_x_dmdy = make_float3(0.0, 0.0, 0.0);  // ∂m/∂x × ∂m/∂y
    int    i_;                                        // neighbor index

    if(is0(m0))
    {
        Fx[I] = 0.0f;
        Fy[I] = 0.0f;
        Fz[I] = 0.0f;
        return;
    }

    // x derivatives (along length)
    {
        float3 m_m1 = make_float3(0.0f, 0.0f, 0.0f);     // -1
        i_ = idx(lclampx(ix-1), iy, iz);                 // load neighbor m if inside grid, keep 0 otherwise
        if (ix-1 >= 0 || PBCx)
        {
            m_m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p1 = make_float3(0.0f, 0.0f, 0.0f);     // +1
        i_ = idx(hclampx(ix+1), iy, iz);
        if (ix+1 < Nx || PBCx)
        {
            m_p1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        if (is0(m_p1) && is0(m_m1))                       //  system is one cell thick
        {
            dmdx = make_float3(0.0f, 0.0f, 0.0f);         // --1-- zero
        }
        else if (is0(m_p1))
        {
            dmdx = m0 - m_m1;                            // backward difference
        } 
        else if (is0(m_m1))
        {
            dmdx = -m0 + m_p1;                            // forward difference
        }
        else
        {
            dmdx = 0.5f * (m_p1 - m_m1);                  // central difference
        }
    }

    // y derivatives (along height)
    {
        float3 m_m1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, lclampy(iy-1), iz);
        if (iy-1 >= 0 || PBCy)
        {
            m_m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, hclampy(iy+1), iz);
        if  (iy+1 < Ny || PBCy)
        {
            m_p1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        if (is0(m_p1) && is0(m_m1))                       //  system is one cell thick
        {
            dmdy = make_float3(0.0f, 0.0f, 0.0f);         // --1-- zero
        }
        else if (is0(m_p1))
        {
            dmdy = m0 - m_m1;                            // backward difference
        } 
        else if (is0(m_m1))
        {
            dmdy = -m0 + m_p1;                            // forward difference
        }
        else
        {
            dmdy = 0.5f * (m_p1 - m_m1);                  // central difference
        }
    }

    // z derivatives (along depth)
    {
        float3 m_m1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, lclampz(iz-1));
        if (iz-1 >= 0 || PBCz)
        {
            m_m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, hclampz(iz+1));
        if  (iz+1 < Nz || PBCz)
        {
            m_p1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        if (is0(m_p1) && is0(m_m1))                       //  system is one cell thick
        {
            dmdz = make_float3(0.0f, 0.0f, 0.0f);         // --1-- zero
        }
        else if (is0(m_p1))
        {
            dmdz = m0 - m_m1;                            // backward difference
        } 
        else if (is0(m_m1))
        {
            dmdz = -m0 + m_p1;                            // forward difference
        }
        else
        {
            dmdz = 0.5f * (m_p1 - m_m1);                  // central difference
        }
    }

    dmdy_x_dmdz = cross(dmdy, dmdz);
    dmdz_x_dmdx = cross(dmdz, dmdx);
    dmdx_x_dmdy = cross(dmdx, dmdy);

    Fx[I] = 2 * prefactor * icycz * dot(m0, dmdy_x_dmdz);
    Fy[I] = 2 * prefactor * iczcx * dot(m0, dmdz_x_dmdx);
    Fz[I] = 2 * prefactor * icxcy * dot(m0, dmdx_x_dmdy);
}
