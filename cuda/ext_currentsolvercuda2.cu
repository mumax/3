#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"

extern "C"

 __global__ void
calculatemaskJ(float* __restrict__ jmaskx, float* __restrict__ jmasky, float* __restrict__ jmaskz,
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

    //printf("%d %d %d\n",ix,iy,iz);
    //int i = idx(ix, iy, iz);
    //return;

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
        float xx=0.0;
        float yy=0.0;
        float zz=0.0;

        // x derivatives
        // left neighbor
        if (ix-1>=0)
        {
            i_  = idx(ix-1, iy, iz);           // clamps or wraps index according to PBC
            m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
            mm_=dot(m_,m_);
            if (mm_!=0)
            {
                acumv+= -v[i_];
                xx+=-1;
            }
        }

        // right neighbor
        if (ix+1<Nx)
        {
            i_  = idx(ix+1, iy, iz);           // clamps or wraps index according to PBC
            m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
            mm_=dot(m_,m_);
            if (mm_!=0)
            {
                acumv+= v[i_];
                xx+=1;
            }
        }

        if (xx==0) {
            jmaskx[i]=acumv/2.0;
        }
        if (xx==-1) {
            jmaskx[i]=acumv+v[i];
        }
        if (xx==1) {
            jmaskx[i]=acumv-v[i];
        }

        acumv=0;

        // y derivatives
 
        // back neighbor
        if (iy-1>=0)
        {
            i_  = idx(ix, iy-1, iz);          // clamps or wraps index according to PBC
            m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
            mm_=dot(m_,m_);
            if (mm_!=0)
            {
                acumv+= -v[i_];
                yy+=-1;
            }
        }

        // front neighbor

        if (iy+1<Ny)
        {
            i_  = idx(ix, iy+1, iz);          // clamps or wraps index according to PBC
            m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
            mm_=dot(m_,m_);
            if (mm_!=0)
            {
                acumv+= v[i_];
                yy+=1;
            }
        }

        if (yy==0) {
            jmasky[i]=acumv/2.0;
        }
        if (yy==-1) {
            jmasky[i]=acumv+v[i];
        }
        if (yy==1) {
            jmasky[i]=acumv-v[i];
        }

        acumv=0;
        // only take vertical derivative for 3D sim
        if (Nz != 1)
        {
            // bottom neighbor
            if (iz-1>=0)
            {
                i_  = idx(ix, iy, iz-1);
                m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
                mm_=dot(m_,m_);
                if (mm_!=0)
                {
                acumv+= -v[i_];
                zz+=-1; 
                }
            }

            // top neighbor
            if (iz+1<Nz)
            {
                i_  = idx(ix, iy,iz+1);
                m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
                mm_=dot(m_,m_);
                if (mm_!=0)
                {
                acumv+= v[i_];
                zz+=1;
                }
            }
            if (zz==0) {
                jmaskz[i]=acumv/2.0;
            }
            if (zz==-1) {
                jmaskz[i]=acumv+v[i];
            }
            if (zz==1) {
                jmaskz[i]=acumv-v[i];
            }
        } else
        {
            jmaskz[i]=0;
        }
    }  //end of mm!=0
    else{
        jmaskx[i]=0.0;    
        jmasky[i]=0.0;    
        jmaskz[i]=0.0;    
    }
}


