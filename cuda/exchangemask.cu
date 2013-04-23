#include "stencil.h"
#include "mask.h"

// Add 1 component of exchange interaction to Beff (Tesla).
// m is normalized.
// mask defines a pre-factor for (Aex / Msat) to allow space-dependent exchange.
// the mask is staggered over half a cell with respect to the magnetization grid,
// and otherwise has the same size.
// mask{X,Y,Z} defines the coupling between neighbors in the {X,Y,Z} direction, respectively.
// maskX[i, j, k] defines the coupling between m[i, j, k-1] and m[i, j k]
// maskY[i, j, k] defines the coupling between m[i, j-1, k] and m[i, j k]
// maskZ[i, j, k] defines the coupling between m[i-1, j, k] and m[i, j k]
// Each time, the zeroth element defines the coupling at the leftmost boundary and is thus unused,
// but would be used in case if periodic boundary conditions.
extern "C" __global__ void
addexchangemask(float* __restrict__ Beff, float* __restrict__ m, 
                float* __restrict__ maskX, float* __restrict__ maskY, float* __restrict__ maskZ,
                float wx, float wy, float wz, int N0, int N1, int N2){

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j >= N1 || k >= N2){
		return;
	}

	for(int i=0; i<N0; i++){

		int I = idx(i, j, k);
		float B = Beff[I];
		float m0 = m[I];

		float m1 = m[idx(i, j, lclamp(k-1    ))];
		float m2 = m[idx(i, j, hclamp(k+1, N2))];
		float a1 = loadmask(maskX, idx     (i, j, k  ));
		float a2 = loadmask(maskX, idxclamp(i, j, k+1));
		B += wz * (a1*(m1-m0) + a2*(m2-m0));

		m1 = m[idx(i, lclamp(j-1   ), k)];
		m2 = m[idx(i, hclamp(j+1,N1), k)];
		a1 = loadmask(maskY, idx     (i, j,   k));
		a2 = loadmask(maskY, idxclamp(i, j+1, k));
		B += wy * (a1*(m1-m0) + a2*(m2-m0));

		// only take vertical derivative for 3D sim
		if (N0 != 1){
			m1 = m[idx(hclamp(i+1,N0), j, k)];
			m2 = m[idx(lclamp(i-1   ), j, k)];
			a1 = loadmask(maskZ, idx     (i  , j, k));
			a2 = loadmask(maskZ, idxclamp(i+1, j, k));
			B  += wx * (a1*(m1-m0) + a2*(m2-m0));
		}

		Beff[I] = B;
	}
}

