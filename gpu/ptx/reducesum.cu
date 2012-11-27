// Based on the slides "Optimizing parallel reduction in CUDA" by Mark Harris.

inline __device__ float sum(float a, float b){
	return a + b;
}

// smemsize needs to be at least 2*32
extern "C" __global__ void
reducesum(float *src, float *dst) {

	extern __shared__ float sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

	// TODO: bounds
	sdata[tid] = src[i];
	sdata[tid] = sum(sdata[tid], src[i+blockDim.x]);

	__syncthreads();

	for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
		if (tid < s){
			sdata[tid] = sum(sdata[tid], sdata[tid + s]);
		}
		__syncthreads();
	}

	if (tid < 32) {
      	volatile float* smem = sdata;
		smem[tid] = sum(smem[tid], smem[tid + 32]); 
		smem[tid] = sum(smem[tid], smem[tid + 16]); 
		smem[tid] = sum(smem[tid], smem[tid +  8]); 
		smem[tid] = sum(smem[tid], smem[tid +  4]); 
		smem[tid] = sum(smem[tid], smem[tid +  2]); 
		smem[tid] = sum(smem[tid], smem[tid +  1]); 
	}

	// write result for this block to global mem
	if (tid == 0) { dst[blockIdx.x] = sdata[0]; }
}

