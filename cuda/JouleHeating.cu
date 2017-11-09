#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"


extern "C"

 __global__ void
evaldt0(float* __restrict__  temp_,      float* __restrict__ dt0_,
	float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                float* __restrict__ Kth_, float Kth_mul,
                float* __restrict__ Cth_, float Cth_mul,
                float* __restrict__ Dth_, float Dth_mul,
                float* __restrict__ Tsubsth_, float Tsubsth_mul,
                float* __restrict__ Tausubsth_, float Tausubsth_mul,
                float* __restrict__ res_, float res_mul,
                float* __restrict__ Qext_, float Qext_mul,
                  float* __restrict__ jx_, float jx_mul,
                  float* __restrict__ jy_, float jy_mul,
                  float* __restrict__ jz_, float jz_mul,
		float wx, float wy, float wz, int Nx, int Ny, int Nz
//                int N) {
                ) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    // central cell
    int i = idx(ix, iy, iz);
    float3 m0 = make_float3(mx[i], my[i], mz[i]);
    
    float mm=dot(m0,m0);
    dt0_[i]=0.0;
    if (mm!=0)
    {

    float3 J = vmul(jx_, jy_, jz_, jx_mul, jy_mul, jz_mul, i);
    float Kth = amul(Kth_, Kth_mul, i);
    float Cth = amul(Cth_, Cth_mul, i);
    float Dth = amul(Dth_, Dth_mul, i);
    float Tsubsth = amul(Tsubsth_, Tsubsth_mul, i);
    float Tausubsth = amul(Tausubsth_, Tausubsth_mul, i);
    float res = amul(res_, res_mul, i);
    float Qext = amul(Qext_, Qext_mul, i);

    float temp = temp_[i];

    int i_;    // neighbor index
    float3 m_; // neighbor mag
    float mm_;
    float tempv=0;

    // left neighbor
    if (ix-1>=0){
    i_  = idx(ix-1, iy, iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    mm_=dot(m_,m_);
    if (mm_!=0)
    {
     tempv = temp_[i_];
     dt0_[i] += (Kth*(tempv-temp)/wx/wx);
     //if (tempv>=1e4) {   printf("%d %d %d %e %e %e %e\n",ix,iy,iz,dt0_[i],temp,tempv,mm); dt0_[i]=0.0;}
    }
    }

    // right neighbor
    if (ix+1<Nx){
    i_  = idx(ix+1, iy, iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    mm_=dot(m_,m_);
    if (mm_!=0)
    {
     tempv = temp_[i_];
     dt0_[i] += (Kth*(tempv-temp)/wx/wx);
    }
    }

    // back neighbor
    if (iy-1>=0){
    i_  = idx(ix, iy-1, iz);          // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    mm_=dot(m_,m_);
    if (mm_!=0)
    {
     tempv = temp_[i_];
     dt0_[i] += (Kth*(tempv-temp)/wy/wy);
    }
    }

    // front neighbor

    if (iy+1<Ny){
    i_  = idx(ix, iy+1, iz);          // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    mm_=dot(m_,m_);
    if (mm_!=0)
    {
     tempv = temp_[i_];
     dt0_[i] += (Kth*(tempv-temp)/wy/wy);
    }
    }

    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // bottom neighbor
	if (iz-1>=0){
        i_  = idx(ix, iy, iz-1);
        m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
        mm_=dot(m_,m_);
        if (mm_!=0)
        {
	 tempv = temp_[i_];
         dt0_[i] += (Kth*(tempv-temp)/wz/wz); 
        }
        }

        // top neighbor
        if (iz+1<Nz){
        i_  = idx(ix, iy,iz+1);
        m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
        mm_=dot(m_,m_);
        if (mm_!=0)
        {
	 tempv = temp_[i_];
         dt0_[i] += (Kth*(tempv-temp)/wz/wz); 
        }
        }
    }

//     if (tempv>=1e4) {   printf("%d %d %d %e %e %e %e\n",ix,iy,iz,dt0_[i],temp,tempv,mm); dt0_[i]=0.0;}
    dt0_[i]+=dot(J,J)*res;          //Joule Heating
    dt0_[i]+=Qext;                  //External Heating source in W/m3
    dt0_[i]=dt0_[i]/(Dth*Cth);      // Missing thermal constants

    if (Tausubsth!=0) {dt0_[i]=dt0_[i]-(temp-Tsubsth)/Tausubsth; }  // Substrate effect

//   printf("%d %d %d %e %e %e\n",ix,iy,iz,temp,tempv,dt0_[i]);
//    dt0_[i]=0.0;
    }

}


