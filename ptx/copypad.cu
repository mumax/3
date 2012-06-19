
// Copies src (3D array, size S0 x S1 x S2) to larger dst (3D array, size D0 x D1 x D2).
// The remainder of dst is NOT zero-padded.
// E.g.:
//	a    ->  a x
//	         x x
//
// Launch config:
// 	?
__global__ void copypad(float* dst, int D0, int D1, int D2, 
                        float* src, int S0, int S1, int S2, 
                                    int o0, int o1, int o2){

	int j = blockIdx.y * blockDim.y + threadIdx.y; // index in src slice
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	if(j>=S1 || k>=S2){
 		return;	// out of src bounds
	}

	int J = j + o1;  // index in full src
	int K = k + o2; 
	
	for (int i=0; i<S0; i++){
 		int I = i + o0; // index in full src
	
		dst[I*D1*D2 + J*D2 + K] = src[i*S1*S2 + j*S2 + k];
	}
}



