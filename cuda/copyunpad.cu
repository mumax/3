
// Copy src (size S, larger) to dst (size D, smaller)
extern "C" __global__ void
copyunpad(float* __restrict__  dst, int D0, int D1, int D2,
          float* __restrict__  src, int S0, int S1, int S2) {

    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<D0 && j<D1 && k<D2) {
        dst[D2*(i*D1 + j) + k] = src[S2*(i*S1 + j) + k];
    }
}

