#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"

extern "C"

 __global__ void
NormMaskJ(float* __restrict__ jmaskx, float* __restrict__ jmasky, float* __restrict__ jmaskz,
    float normJ,
	int Nx, int Ny, int Nz
    )
{

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

    else{
        int i = idx(ix, iy, iz);
        jmaskx[i]=jmaskx[i]/sqrt(normJ);  
        jmasky[i]=jmasky[i]/sqrt(normJ);    
        jmaskz[i]=jmaskz[i]/sqrt(normJ);
    }

}


