
// dst[i] = src1[i] * fac1 + src2[i] * fac2 + src3[i] * fac3 + src4[i] * fac4
extern "C" __global__ void
madd4(float* __restrict__ dst,
      float* __restrict__ src1, float fac1,
      float* __restrict__ src2, float fac2,
      float* __restrict__ src3, float fac3,
      float* __restrict__ src4, float fac4, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        dst[i] = (fac1*src1[i]) + (fac2*src2[i]) + (fac3*src3[i]) + (fac4*src4[i]);
    }
}

