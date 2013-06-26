
// Copy src (size S, smaller) into dst (size D, larger),
// and multiply by Bsat as defined in regions.
extern "C" __global__ void
copypadmul(float* __restrict__ dst, int D0, int D1, int D2,
           float* __restrict__ src, int S0, int S1, int S2,
           float* __restrict__ BsatLUT, int8_t* __restrict__ regions) {

    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<S0 && j<S1 && k<S2) {
        int sI = S2*(i*S1 + j) + k; // source index
        float Bsat = BsatLUT[regions[sI]];
        dst[D2*(i*D1 + j) + k] = Bsat * src[sI];
    }
}

