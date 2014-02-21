
// dst[i] = fac1*src1[i] + fac2*src2[i];
extern "C" __global__ void
madd2(float* __restrict__  dst,
      float* __restrict__  src1, float fac1,
      float* __restrict__  src2, float fac2, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

	if(i < N) {
		dst[i] = fac1*src1[i] + fac2*src2[i];
	}
}

