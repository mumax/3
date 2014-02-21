// dst[i] = a[i] * b[i]
extern "C" __global__ void
mul(float* __restrict__  dst, float* __restrict__  a, float* __restrict__ b, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

	if(i < N) {
		dst[i] = a[i] * b[i];
	}
}

