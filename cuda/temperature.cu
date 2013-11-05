
extern "C" __global__ void
addtemperature(float* __restrict__  B,      float* __restrict__ noise, float kB2_VgammaDt,
               float* __restrict__ tempRedLUT, int8_t* __restrict__ regions, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        int8_t reg  = regions[i];
        float alphaT_Bs  = tempRedLUT[reg];
        B[i] += noise[i] * sqrtf( kB2_VgammaDt * alphaT_Bs );
    }
}

