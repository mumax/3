
// Copy src (size S, larger) to dst (size D, smaller)
extern "C" __global__ void
copyunpad(float* __restrict__  dst, int D0, int D1, int D2,
          float* __restrict__  src, int S0, int S1, int S2) {

    int j = blockIdx.y * blockDim.y + threadIdx.y; // index in src slice
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(j>=D1 || k>=D2) {
        return;	// out of  bounds
    }

    for (int i=0; i<D0; i++) {
        dst[D2*(i*D1 + j) + k] = src[S2*(i*S1 + j) + k];
    }
}

