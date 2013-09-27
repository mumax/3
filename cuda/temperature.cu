
extern "C" __global__ void
addtemperature(float* __restrict__  B,      float* __restrict__ noise, float kB2_VgammaDt,
               float* __restrict__ TempLUT, float* __restrict__ alphaLUT, float* __restrict__ BsLUT,
               int8_t* __restrict__ regions, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        int8_t reg  = regions[i];
        float T     = TempLUT[reg];
        float alpha = alphaLUT[reg];
        float Bs    = BsLUT[reg];

        B[i] += noise[i] * sqrtf( kB2_VgammaDt * alpha * T / Bs );
    }
}

