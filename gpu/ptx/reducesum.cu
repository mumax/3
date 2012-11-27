// Based on the slides "Optimizing parallel reduction in CUDA" by Mark Harris.

inline __device__ float sum(float a, float b){
	return a + b;
}

#define BLOCK 512

// launch config should be "too small" for number of elements,
// 1D config! (x only)
extern "C" __global__ void
reducesum(float *src, float *dst, int n) {

	__shared__ float sdata[BLOCK];
	int tid = threadIdx.x;
	int i =  blockIdx.x * blockDim.x + threadIdx.x;

	float mySum = 0;
	int stride = gridDim.x * blockDim.x;
	while (i < n) {
    	mySum += src[i];
    	i += stride;
	}
	sdata[tid] = mySum;
	__syncthreads();

	for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
		if (tid < s){
			sdata[tid] = sum(sdata[tid], sdata[tid + s]);
		}
		__syncthreads();
	}

	// warp-synchronous
	if (tid < 32) {
      	volatile float* smem = sdata;
		smem[tid] = sum(smem[tid], smem[tid + 32]); 
		smem[tid] = sum(smem[tid], smem[tid + 16]); 
		smem[tid] = sum(smem[tid], smem[tid +  8]); 
		smem[tid] = sum(smem[tid], smem[tid +  4]); 
		smem[tid] = sum(smem[tid], smem[tid +  2]); 
		smem[tid] = sum(smem[tid], smem[tid +  1]); 
	}

	if (tid == 0) { atomicAdd(dst, sdata[0]); }
}

