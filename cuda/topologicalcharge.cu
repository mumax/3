#include <stdint.h>
#include <stdio.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"

// Set s to the topological charge density.
// See topologicalcharge.go.
extern "C" __global__ void
settopologicalcharge(float* __restrict__ s,
                     float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                     float icxcy, int Nx, int Ny, int Nz, uint16_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

    int I = idx(ix, iy, iz);                      // central cell index

    float3 m0 = make_float3(mx[I], my[I], mz[I]); // +0
    float3 dmdx = make_float3(0.0f, 0.0f, 0.0f);  // ∂m/∂x
    float3 dmdy = make_float3(0.0f, 0.0f, 0.0f);  // ∂m/∂y
    float3 dmdx_x_dmdy = make_float3(0.0, 0.0, 0.0); // ∂m/∂x ❌ ∂m/∂y
    int i_;                                       // neighbor index

    if(is0(m0))
    {
        s[I] = 0.0f;
        return;
    }

    // x derivatives (along length)
    {
        float3 m_m2 = make_float3(0.0f, 0.0f, 0.0f);     // -2
        i_ = idx(lclampx(ix-2), iy, iz);                 // load neighbor m if inside grid, keep 0 otherwise
        if (ix-2 >= 0 || PBCx)
        {
            m_m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

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

        float3 m_p2 = make_float3(0.0f, 0.0f, 0.0f);     // +2
        i_ = idx(hclampx(ix+2), iy, iz);
        if (ix+2 < Nx || PBCx)
        {
            m_p2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        if (is0(m_p1) && is0(m_m1))                       //  +0
        {
            dmdx = make_float3(0.0f, 0.0f, 0.0f);         // --1-- zero
        }
        else if ((is0(m_m2) | is0(m_p2)) && !is0(m_p1) && !is0(m_m1))
        {
            dmdx = 0.5f * (m_p1 - m_m1);                  // -111-, 1111-, -1111 central difference,  ε ~ h^2
        }
        else if (is0(m_p1) && is0(m_m2))
        {
            dmdx =  m0 - m_m1;                            // -11-- backward difference, ε ~ h^1
        }
        else if (is0(m_m1) && is0(m_p2))
        {
            dmdx = -m0 + m_p1;                            // --11- forward difference,  ε ~ h^1
        }
        else if (!is0(m_m2) && is0(m_p1))
        {
            dmdx =  0.5f * m_m2 - 2.0f * m_m1 + 1.5f * m0; // 111-- backward difference, ε ~ h^2
        }
        else if (!is0(m_p2) && is0(m_m1))
        {
            dmdx = -0.5f * m_p2 + 2.0f * m_p1 - 1.5f * m0; // --111 forward difference,  ε ~ h^2
        }
        else
        {
            dmdx = (2.0f/3.0f)*(m_p1 - m_m1) + (1.0f/12.0f)*(m_m2 - m_p2); // 11111 central difference,  ε ~ h^4
        }
    }

    // y derivatives (along height)
    {
        float3 m_m2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, lclampy(iy-2), iz);
        if (iy-2 >= 0 || PBCy)
        {
            m_m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

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

        float3 m_p2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, hclampy(iy+2), iz);
        if  (iy+2 < Ny || PBCy)
        {
            m_p2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        if (is0(m_p1) && is0(m_m1))                                        //  +0
        {
            dmdy = make_float3(0.0f, 0.0f, 0.0f);                          // --1-- zero
        }
        else if ((is0(m_m2) | is0(m_p2)) && !is0(m_p1) && !is0(m_m1))
        {
            dmdy = 0.5f * (m_p1 - m_m1);                                   // -111-, 1111-, -1111 central difference,  ε ~ h^2
        }
        else if (is0(m_p1) && is0(m_m2))
        {
            dmdy =  m0 - m_m1;                                             // -11-- backward difference, ε ~ h^1
        }
        else if (is0(m_m1) && is0(m_p2))
        {
            dmdy = -m0 + m_p1;                                             // --11- forward difference,  ε ~ h^1
        }
        else if (!is0(m_m2) && is0(m_p1))
        {
            dmdy =  0.5f * m_m2 - 2.0f * m_m1 + 1.5f * m0;                 // 111-- backward difference, ε ~ h^2
        }
        else if (!is0(m_p2) && is0(m_m1))
        {
            dmdy = -0.5f * m_p2 + 2.0f * m_p1 - 1.5f * m0;                 // --111 forward difference,  ε ~ h^2
        }
        else
        {
            dmdy = (2.0f/3.0f)*(m_p1 - m_m1) + (1.0f/12.0f)*(m_m2 - m_p2); // 11111 central difference,  ε ~ h^4
        }
    }
    dmdx_x_dmdy = cross(dmdx, dmdy);

    s[I] = icxcy * dot(m0, dmdx_x_dmdy);
}

