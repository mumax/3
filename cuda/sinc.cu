// dst[i] = sinc(a[i])
extern "C" __global__ void
unary_sinc(float *__restrict__ dst, float *__restrict__ a, int N) {
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    
    if (i < N)
    {
        float x = a[i];
        dst[i] = (x == 0.0f) ? 1.0f : sinf(x) / x;
    }
}