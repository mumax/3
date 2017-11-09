#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"

extern "C"

 __global__ void
evaldvolt(float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
    float* __restrict__ v,
    float* __restrict__ ground_, float ground_mul,
    float* __restrict__ terminal_, float terminal_mul,
	float wx, float wy, float wz, int Nx, int Ny, int Nz
    )
{

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    int neighbours=0;
    float acumv=0;

    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

    // central cell
    int i = idx(ix, iy, iz);
    float3 m0 = make_float3(mx[i], my[i], mz[i]);
    float mm=dot(m0,m0);
    if (mm!=0)
    {
        float ground = amul(ground_, ground_mul, i);
        float terminal = amul(terminal_, terminal_mul, i);
        if (ground==1)
        {
            v[i]=0;
            return;
        }
        if (terminal==1)
        {
            v[i]=1;
            return;
        }

        int i_;    // neighbor index
        float3 m_; // neighbor mag
        float mm_;
        float leftv=0;
        float rightv=0;
        float backv=0;
        float frontv=0;
        float topv=0;
        float bottomv=0;
        float nx=0;
        float ny=0;
        float nz=0;
        int smooth=1;

        // left neighbor
        if (ix-1>=0)
        {
            i_  = idx(ix-1, iy, iz);           // clamps or wraps index according to PBC
            m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
            mm_=dot(m_,m_);
            if (mm_!=0)
            {
                leftv= v[i_];
                neighbours++;
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
                rightv= v[i_];
                neighbours++;
            }
        }

        // back neighbor
        if (iy-1>=0)
        {
            i_  = idx(ix, iy-1, iz);          // clamps or wraps index according to PBC
            m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
            mm_=dot(m_,m_);
            if (mm_!=0)
            {
                backv= v[i_];
                neighbours++;
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
                frontv= v[i_];
                neighbours++;
            }
        }

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
                    bottomv= v[i_];
                    neighbours++;
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
                    topv= v[i_];
                    neighbours++;
                }
            }
        }
    
        if (Nz==1)
        {
	       if(neighbours==4)
	       {
		      v[i]=(rightv+leftv+backv+frontv)/neighbours;
	       } else
	       {
            // Normal vector to the edge calculation
             for(int yy=-smooth;yy<=smooth;yy++){
                    for (int xx=-smooth;xx<=smooth;xx++){
                        if ((ix+xx>0)&&(ix+xx<Nx)&&(iy+yy>0)&&(iy+yy<Ny)){
                            i_  = idx(ix+xx, iy+yy,iz);
                            m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
                            mm_=dot(m_,m_);
                            if (mm_!=0){
                                if (xx<0) nx++;
                                if (xx>0) nx--;
                                if (yy<0) ny++;
                                if (yy>0) ny--;
                            }
                        }
                    }
             }

             acumv=0;
		     if (nx!=0) {
                if (nx>0) {
                    acumv+=nx/(abs(nx)+abs(ny))*leftv;
                }
                if (nx<0) {
                    acumv-=nx/(abs(nx)+abs(ny))*rightv;
                }
             }
             if (ny!=0) {
                if (ny>0) {
                    acumv+=ny/(abs(nx)+abs(ny))*backv;
                }
                if (ny<0) {
                    acumv-=ny/(abs(nx)+abs(ny))*frontv;
                }
             }
	
            v[i]=acumv;
            }
        } else
        {
            if(neighbours==6)
            {
                v[i]=(rightv+leftv+backv+frontv+topv+bottomv)/neighbours;
            } else
           {
            // Normal vector to the edge calculation
             for(int yy=-smooth;yy<=smooth;yy++){
                for (int zz=-smooth;zz<=smooth;zz++){
                    for (int xx=-smooth;xx<=smooth;xx++){
                        if ((ix+xx>0)&&(ix+xx<Nx)&&(iy+yy>0)&&(iy+yy<Ny)&&(iz+zz>0)&&(iz+zz<Nz)){
                            i_  = idx(ix+xx, iy+yy,iz+zz);
                            m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
                            mm_=dot(m_,m_);
                            if (mm_!=0){
                                if (xx<0) nx++;
                                if (xx>0) nx--;
                                if (yy<0) ny++;
                                if (yy>0) ny--;
                                if (zz<0) nz++;
                                if (zz>0) nz--;
                            }
                        }
                    }
                }
             }
             acumv=0;
             if (nx!=0) {
                if (nx>0) {
                    acumv+=nx/sqrt(nx*nx+ny*ny+nz*nz)*rightv;
                }else{
                    acumv-=nx/sqrt(nx*nx+ny*ny+nz*nz)*leftv;
                }

                if (ny>0) {
                    acumv+=ny/sqrt(nx*nx+ny*ny+nz*nz)*frontv;
                }else{
                    acumv-=ny/sqrt(nx*nx+ny*ny+nz*nz)*backv;
                }

                if (nz>0) {
                    acumv+=nz/sqrt(nx*nx+ny*ny+nz*nz)*topv;
                }else{
                    acumv-=nz/sqrt(nx*nx+ny*ny+nz*nz)*bottomv;
                }
             }
    
            v[i]=acumv;
            }
        }
    }  //end of mm!=0

}


