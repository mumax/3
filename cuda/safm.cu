#include <stdint.h>
//#include "ext_SAFM.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// See exchange.go for more details.
extern "C" __global__ void
addAFMexchange(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            float AFMex, 
            int AFMR1, 
            int AFMR2,
	    float tsp,
            float* __restrict__ Ms_, float Ms_mul,
             uint16_t* __restrict__ regions,
            float wx, float wy, float wz, int Nx, int Ny, int Nz, uint16_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
 


    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);
    float3 m0 = make_float3(mx[I], my[I], mz[I]);
    float3 B  = make_float3(Bx[I], By[I], Bz[I]);
    float Ms   = amul(Ms_, Ms_mul, I);
//printf("AFM %f %d %d\n",AFMex,AFMR1,AFMR2); 
    if (AFMex!=0) {

     if (iz==AFMR1)
     {
     int i_;    // neighbor index
     float3 m_; // neighbor mag
 
     i_  = idx(ix, iy, AFMR2);           // clamps or wraps index according to PBC
     m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
     m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
     B += AFMex/Ms/tsp *(m_ - m0);

     }

     if (iz==AFMR2)
     {
     int i_;    // neighbor index
     float3 m_; // neighbor mag
 
     i_  = idx(ix, iy, AFMR1);           // clamps or wraps index according to PBC
     m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
     m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
     B += AFMex/Ms/tsp *(m_ - m0);
     }

    }


    Bx[I] = B.x;
    By[I] = B.y;
    Bz[I] = B.z;

}

