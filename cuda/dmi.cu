#include "stencil.h"

// Dzyaloshinskii-Moriya interaction.
// m is normalized.

extern "C" __global__ void
adddmi(float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
       float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
       float dx, float dy, float dz, // DMI vector in Tesla / m
       float cx, float cy, float cz, // cell size in m
       int N0, int N1, int N2){

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j >= N1 || k >= N2){
		return;
	}

	for(int i=0; i<N0; i++){

		int I = idx(i, j, k);

		if (dx != 0){
			float dmzdy = diff(mz, 0, 1, 0, cy); // ∂mz / ∂y
			float dmydz = diff(my, 0, 0, 1, cz); // ∂my / ∂z
			Hx[I] += dx * (-dmzdy + dmydz); 
		}

		if (dy != 0){
			float dmzdx = diff(mz, 1, 0, 0, cx);
			float dmxdz = diff(mx, 0, 0, 1, cz);
			Hy[I] += dy * (dmzdx - dmxdz); 
		}

		if (dz != 0){
			float dmydx = diff(my, 1, 0, 0, cx);
			float dmxdy = diff(mx, 0, 1, 0, cy);
			Hz[I] += dz * (-dmydx + dmxdy); 
		}
		// note: left-handed coordinate system.
	}
}

