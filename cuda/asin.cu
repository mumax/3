// dst[i] = asin(a[i]), returns 0 for a[i] outside of domain [-1, 1]

extern "C" __global__ void
unary_asin(float* __restrict__ dst, float* __restrict__ a, int N) {
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        float x = a[i];
        if (x >= -1.0f && x <= 1.0f) {
            dst[i] = asinf(x);
        } else {
            dst[i] = 0.0f;
        }
    }
}