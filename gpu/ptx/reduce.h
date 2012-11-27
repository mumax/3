#ifndef _REDUCE_H_
#define _REDUCE_H_

// identity preprocess func.
inline __device__ float ident(float a){
	return a;
}

// Block size for reduce kernels.
#define BLOCK 512

// This macro expands to a reduce kernel with arbitrary reduce operation.
// Ugly, perhaps, but arguably nicer than some 1000+ line C++ template.
#define reduce(preprocess, op, atomicOp, initVal)\
 	__shared__ float sdata[BLOCK];                      \
	int tid = threadIdx.x;                              \
	int i =  blockIdx.x * blockDim.x + threadIdx.x;     \
                                                        \
	float mine = initVal;                               \
	int stride = gridDim.x * blockDim.x;                \
	while (i < n) {                                     \
    	mine = op(mine, preprocess(src[i]));            \
    	i += stride;                                    \
	}                                                   \
	sdata[tid] = mine;                                  \
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
		smem[tid] = op(smem[tid], smem[tid + 32]);      \
		smem[tid] = op(smem[tid], smem[tid + 16]);      \
		smem[tid] = op(smem[tid], smem[tid +  8]);      \
		smem[tid] = op(smem[tid], smem[tid +  4]);      \
		smem[tid] = op(smem[tid], smem[tid +  2]);      \
		smem[tid] = op(smem[tid], smem[tid +  1]);      \
	}                                                   \
                                                        \
	if (tid == 0) { atomicOp(dst, sdata[0]); }          \
// Based on "Optimizing parallel reduction in CUDA" by Mark Harris.
#endif
