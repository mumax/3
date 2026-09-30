//dst[i] = sin(a[i])
extern "C" __global__ void
unary_sin(float *__restrict__ dst, float *__restrict__ a, int N) {
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    
    if (i < N)
    {
        dst[i] = sinf(a[i]);
    }
}
