//dst[i] = exp(a[i])
extern "C" __global__ void
unary_exp(float *__restrict__ dst, float *__restrict__ a, int N){
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N){
        dst[i] = expf(a[i]);
    }
}