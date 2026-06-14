// Set dst(x|y|z) to the x-, y- and z-index of the corresponding cell
// Adapted from Jake Love's kernel in PR#289
extern "C" __global__ void
cellindices(float* __restrict__  dstx,
            float* __restrict__  dsty,
            float* __restrict__  dstz, 
            float nx, float ny, float nz, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {
        float idx_i = fmodf(i, nx);
        float idx_j = floorf(fmodf(i / nx, ny));
        float idx_k = floorf(i / (nx*ny));

        dstx[i] = idx_i;
        dsty[i] = idx_j;
        dstz[i] = idx_k;
    }
}
