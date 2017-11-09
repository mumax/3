#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"

extern "C"

 __global__ void
calculatemaskJ2(float* __restrict__ jmaskx, float* __restrict__ jmasky, float* __restrict__ jmaskz,
    float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
    float* __restrict__ v,
	float wx, float wy, float wz, int Nx, int Ny, int Nz
    )
{

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    float acumv=0.0;

    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

    // central cell
    int i = idx(ix, iy, iz);
    jmaskx[i]=0.0;  
    jmasky[i]=0.0;    
    jmaskz[i]=0.0;  
    float3 m0 = make_float3(mx[i], my[i], mz[i]);
    float mm=dot(m0,m0);
    if (mm!=0)
    {
        int i_;    // neighbor index
        float3 m_; // neighbor mag
        float mm_;
        int smooth=2;
        int nvec=0;

        // x derivatives
        nvec=0;
        acumv=0;
        for (int ii=-smooth;ii<=smooth;ii++){
            if ((ix+ii>=0)&&(ix+ii<Nx)){
                i_  = idx(ix+ii, iy, iz);           
                m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
                mm_=dot(m_,m_);
                if (mm_!=0)
                {
                    if (ii!=0) {acumv+= (v[i_]-v[i])/ii;}
                    nvec++;
                }
            }
            jmaskx[i]=acumv/nvec;
        }

        // y derivatives
        nvec=0;
        acumv=0;
        for (int ii=-smooth;ii<=smooth;ii++){
            if ((iy+ii>=0)&&(iy+ii<Ny)){
                i_  = idx(ix, iy+ii, iz);           
                m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
                mm_=dot(m_,m_);
                if (mm_!=0)
                {
                    if (ii!=0) {acumv+= (v[i_]-v[i])/ii;}
                    nvec++;
                }
            }
            jmasky[i]=acumv/nvec;
        }

        // z derivatives
        if (Nz>1) {
            nvec=0;
            acumv=0;
            for (int ii=-smooth;ii<=smooth;ii++){
                if ((iz+ii>=0)&&(iz+ii<Nz)){
                    i_  = idx(ix, iy, iz+ii);           
                    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
                    mm_=dot(m_,m_);
                    if (mm_!=0)
                    {
                        if (ii!=0) {acumv+= (v[i_]-v[i])/ii;}
                        nvec++;
                    }
                }
                jmaskz[i]=acumv/nvec;
            }
        }
        else {jmaskz[i]=0;}
        

    }
}


