
// Copies src (3D array, size S0 x S1 x S2) to larger dst (3D array, size D0 x D1 x D2).
// src data is offset by o0,o1,o2.
// The remainder of dst is NOT zero-padded.
// E.g.:
//	a    ->  a x
//	         x x
//
// Launch config:
// 	?
extern "C" __global__ void 
copypad(float* dst, int D0, int D1, int D2, 
        float* src, int S0, int S1, int S2, 
        int o0, int o1, int o2){

	// swap j/k, swap S1/S2, D1/D2
	int j = blockIdx.y * blockDim.y + threadIdx.y; // index in src slice
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	if(j>=S1 || k>=S2){
	//printf("(%dx%dx%d x %dx%dx%d):(%d %d %d:%d %d %d): return1\n",
	//		gridDim.x, gridDim.y, gridDim.z, 
	//		blockDim.x, blockDim.y, blockDim.z,
	//		blockIdx.x, blockIdx.y, blockIdx.z,
	//		threadIdx.x, threadIdx.y, threadIdx.z);
 		return;	// out of src bounds
	}
	// TODO: fuse
	if(j>=D1 || k>=D2){
	//printf("(%dx%dx%d x %dx%dx%d):(%d %d %d:%d %d %d): return2\n",
	//		gridDim.x, gridDim.y, gridDim.z, 
	//		blockDim.x, blockDim.y, blockDim.z,
	//		blockIdx.x, blockIdx.y, blockIdx.z,
	//		threadIdx.x, threadIdx.y, threadIdx.z);
 		return;	// out of dst bounds
	}

	int J = j + o1;  // index in full src
	int K = k + o2; 
	
	for (int i=0; i<S0; i++){
 		int I = i + o0; // index in full src

//	printf("(%dx%dx%d x %dx%dx%d):(%d %d %d:%d %d %d): %d -> %d\n",
//			gridDim.x, gridDim.y, gridDim.z, 
//			blockDim.x, blockDim.y, blockDim.z,
//			blockIdx.x, blockIdx.y, blockIdx.z,
//			threadIdx.x, threadIdx.y, threadIdx.z,
//			i*S1*S2 + j*S2 + k, I*D1*D2 + J*D2 + K);
	
		dst[I*D1*D2 + J*D2 + K] = src[i*S1*S2 + j*S2 + k];
	}
}



