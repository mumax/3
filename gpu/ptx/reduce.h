#ifndef _REDUCE_H_
#define _REDUCE_H_

// Block size for reduce kernels.
#define BLOCK 512

// This macro expands to a reduce kernel with arbitrary reduce operation.
#define reduce(op)\
 	__shared__ float sdata[BLOCK];                      \
	int tid = threadIdx.x;                              \
	int i =  blockIdx.x * blockDim.x + threadIdx.x;     \
                                                        \
	float mySum = 0;                                    \
	int stride = gridDim.x * blockDim.x;                \
	while (i < n) {                                     \
    	mySum += src[i];                                \
    	i += stride;                                    \
	}                                                   \
	sdata[tid] = mySum;                                 \
	__syncthreads();                                    \
                                                        \
	for (unsigned int s=blockDim.x/2; s>32; s>>=1) {    \
		if (tid < s){                                   \
			sdata[tid] = op(sdata[tid], sdata[tid + s]);\
		}                                               \
		__syncthreads();                                \
	}                                                   \
                                                        \
	if (tid < 32) {                                     \
      	volatile float* smem = sdata;                   \
		smem[tid] = sum(smem[tid], smem[tid + 32]);     \
		smem[tid] = sum(smem[tid], smem[tid + 16]);     \
		smem[tid] = sum(smem[tid], smem[tid +  8]);     \
		smem[tid] = sum(smem[tid], smem[tid +  4]);     \
		smem[tid] = sum(smem[tid], smem[tid +  2]);     \
		smem[tid] = sum(smem[tid], smem[tid +  1]);     \
	}                                                   \
                                                        \
	if (tid == 0) { atomicAdd(dst, sdata[0]); }         \

#endif

// Based on "Optimizing parallel reduction in CUDA" by Mark Harris.
