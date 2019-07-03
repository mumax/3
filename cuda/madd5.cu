
// dst[i] = fac1 * src1[i] + fac2 * src2[i] + fac3 * src3[i]
extern "C" __global__ void
madd5(float* __restrict__ dst,
      float* __restrict__ src1, float fac1,
      float* __restrict__ src2, float fac2,
      float* __restrict__ src3, float fac3,
      float* __restrict__ src4, float fac4,
      float* __restrict__ src5, float fac5, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        dst[i] = (fac1*src1[i]) + (fac2*src2[i]) + (fac3*src3[i]) + (fac4*src4[i]) + (fac5*src5[i]);
    }
}

