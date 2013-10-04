
// Select and resize one layer for interactive output
extern "C" __global__ void
resize(float* __restrict__  dst, int D0, int D1, int D2,
       float* __restrict__  src, int S0, int S1, int S2,
       int layer, int scale1, int scale2) {

    //int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (j<D1 && k<D2) {
        float sum = 0.0f;
        for(int J=0; J<scale1; J++) {
            for(int K=0; K<scale2; K++) {
                sum += src[S2*(layer*S1 + j+J) + k+K];
            }
        }
        dst[D2*(layer*D1 + j) + k] = sum / (float)(scale1 * scale2);
    }
}

