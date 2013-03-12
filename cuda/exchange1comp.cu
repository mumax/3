#include "exchange.h"

// 1 component of exchange interaction.
// m is normalized (in Tesla).

extern "C" __global__ void
exchange1comp(float* __restrict__ h, float* __restrict__ m, 
              float wx, float wy, float wz, int N0, int N1, int N2){

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j >= N1 || k >= N2){
		return;
	}

	for(int i=0; i<N0; i++){

		int I = idx(i, j, k);
		float m0 = m[I];

		float m1 = m[idx(i, j, clamp(k-1, N2))];
		float m2 = m[idx(i, j, clamp(k+1, N2))];
		float H = wz * ((m1-m0) + (m2-m0));

		m1 = m[idx(i, clamp(j+1,N1), k)];
		m2 = m[idx(i, clamp(j-1,N1), k)];
		H += wy * ((m1-m0) + (m2-m0));

		// only take vertical derivative for 3D sim
		if (N0 != 1){
			m1 = m[idx(clamp(i+1,N0), j, k)];
			m2 = m[idx(clamp(i-1,N0), j, k)];
			H  += wx * ((m1-m0) + (m2-m0));
		}

		h[I] = H;
	}
}

