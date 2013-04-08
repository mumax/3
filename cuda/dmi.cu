#include "common_stencil.h"

// Dzyaloshinskii-Moriya interaction.
// m is normalized.

// clamp index to bounds (0:N0, 0:N1, 0:N2)
#define ix(i, j, k) idx(clamp(i, N0), clamp(j, N1), clamp(k, N2))

// spatial derivative along (u, v, w) direction with given cell size
#define diff(out, in, u, v, w, c) out = ((in[ix(i+u, j+v, k+w)] - in[ix(i-u, j-v, k-v)])/(2*c))

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
			float dmzdy; diff(dmzdy, mz, 0, 1, 0, cy); // ∂mz / ∂y
			float dmydz; diff(dmydz, my, 0, 0, 1, cz); // ∂my / ∂z
			Hx[I] += dx * (-dmzdy + dmydz); 
		}

		if (dy != 0){
			float dmzdx; diff(dmzdx, mz, 1, 0, 0, cx);
			float dmxdz; diff(dmxdz, mx, 0, 0, 1, cz);
			Hy[I] += dy * (dmzdx - dmxdz); 
		}

		if (dz != 0){
			float dmydx; diff(dmydx, my, 1, 0, 0, cx);
			float dmxdy; diff(dmxdy, mx, 0, 1, 0, cy);
			Hz[I] += dz * (-dmydx + dmxdy); 
		}
		// note: left-handed coordinate system.
	}
}

