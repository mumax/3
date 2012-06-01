#ifdef __cplusplus
extern "C" {
#endif

#define threadindex ( ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x )

/// Sets the first N elements of array to value.
__global__ void generateLoad(float* array, int howmuchload, int N){
	int i = threadindex;
	if(i < N){
		for(int j=0; j<howmuchload; j++){
			array[i] /= (1+j);
		}
	}
}


#ifdef __cplusplus
}
#endif
