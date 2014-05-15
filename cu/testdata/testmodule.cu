/*
 * Module to test CUDA module loading and execution.
 * To be compiled with:
 * nvcc -ptx testmodule.cu
 */


#ifdef __cplusplus
extern "C" {
#endif

#define threadindex ( ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x )

/// Sets the first N elements of array to value.
__global__ void testMemset(float* array, float value, int N){
	int i = threadindex;
	if(i < N){
		array[i] = value;
	}
}


#ifdef __cplusplus
}
#endif
