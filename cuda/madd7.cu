
// dst[i] = src1[i] * fac1 + src2[i] * fac2 + src3[i] * fac3 + src4[i] * fac4 + src5[i] * fac5 + src6[i] * fac6 + src7[i] * fac7
extern "C" __global__ void
madd7(float* __restrict__ dst,
      float* __restrict__ src1, float fac1,
      float* __restrict__ src2, float fac2,
      float* __restrict__ src3, float fac3,
      float* __restrict__ src4, float fac4,
      float* __restrict__ src5, float fac5,
      float* __restrict__ src6, float fac6,
      float* __restrict__ src7, float fac7, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        dst[i] = (fac1*src1[i]) + (fac2*src2[i]) + (fac3*src3[i]) + (fac4*src4[i]) + (fac5*src5[i]) + (fac6*src6[i]) + (fac7*src7[i]);
    }
}

