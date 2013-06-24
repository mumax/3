// dst[i] = fac * src[i] + y
extern "C" __global__ void
saxpb(float* __restrict__  dst, float* __restrict__  src, float fac, float y, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        dst[i] = fac * src[i] + y;
    }
}

