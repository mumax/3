// dst[i] = fac * src[i]
extern "C" __global__ void 
mul(float* __restrict__  dst, float* __restrict__  src, float fac, int N){

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

	if(i < N){
		dst[i] = fac * src[i];
	}
}

