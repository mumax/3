// dst[i] = log(a[i]), returns 0 for non-positive input
extern "C" __global__ void
unary_log(float* __restrict__ dst, float* __restrict__ a, int N) {
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        if (a[i] > 0.0f) {
            dst[i] = logf(a[i]);
        } else {
            dst[i] = 0.0f;
        }
    }
}