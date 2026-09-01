// dst[i] = gamma(a[i]), returns 0 for x <= 0, returns 0 for a[i] outside of domain (0, inf) with poles at non-positive integers
extern "C" __global__ void
unary_gamma(float* __restrict__ dst, float* __restrict__ a, int N) {

    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        float x = a[i];
        if (x > 0.0f) {
            dst[i] = tgammaf(x);
        } else {
            dst[i] = 0.0f;
        }
    }
}