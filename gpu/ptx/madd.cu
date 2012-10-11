extern "C" __global__ void 
madd(float* dst,  float* src,  float scale, int N){
	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if(i < N){
		dst[i] += scale * src[i];
	}
}

