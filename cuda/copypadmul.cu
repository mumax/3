// Copies src (3D array, size S0 x S1 x S2) to larger dst (3D array, size D0 x D1 x D2),
// also multiplies by vol*bsat (same size as src).
// The remainder of dst is NOT zero-padded.
// E.g.:
//	a    ->  a x
//	         x x
//
extern "C" __global__ void 
copypadmul(float* __restrict__ dst, int D0, int D1, int D2, 
           float* __restrict__ src, int S0, int S1, int S2,
           float* __restrict__ vol, float Bsat){

	int j = blockIdx.y * blockDim.y + threadIdx.y; // index in src slice
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	if(j>=S1 || k>=S2 || j>=D1 || k>=D2){
 		return;	// out of  bounds
	}

	// loop over N layers
	int N = min(S0, D0);
	for (int i=0; i<N; i++){
		int sI = i*S1*S2 + j*S2 + k;
		dst[i*D1*D2 + j*D2 + k] = Bsat * vol[sI] * src[sI];
	}
} 

