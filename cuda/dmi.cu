#include "stencil.h"

// Dzyaloshinskii-Moriya interaction according to
// Bagdanov and Röβler, PRL 87, 3, 2001. Eq. (6) and (8).
// m: normalized magnetization
// H: effective field in Tesla
extern "C" __global__ void
adddmi(float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
       float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
       float Dx, float Dy, float Dz, // DMI vector / cell size, in Tesla
       int N0, int N1, int N2){

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j >= N1 || k >= N2){
		return;
	}

	for(int i=0; i<N0; i++){

		int I = idx(i, j, k);
		float3 h = make_float3(Hx[I], Hy[I], Hz[I]); // add to H

			//h.x += Dz * delta(mz, 0, 0, 1); 
			//h.x += Dy * delta(my, 0, 1, 0); 
			//h.y -= Dy * delta(mx, 0, 1, 0); 

			h.z -= Dz * delta(mx, 0, 0, 1); 

		// write back, result is H + Hdmi
		Hx[I] = h.x;
		Hy[I] = h.y;
		Hz[I] = h.z;
	}
}

