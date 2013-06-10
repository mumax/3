// 2D X (out-of-plane only) micromagnetic kernel multiplication:
// Mx = Kxx * Mx
//
// ~kernel has mirror symmetry along Y-axis,
// apart form first row,
// and is only stored (roughly) half:
//
// K00:
// xxxxx
// aaaaa
// bbbbb
// ....
// bbbbb
// aaaaa
//
extern "C" __global__ void
kernmulRSymm2Dx(float* __restrict__  fftMx, float* __restrict__  fftKxx, int N1, int N2) {

    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(j>= N1 || k>=N2) {
        return;
    }

    int I = j*N2 + k;       // linear index for upper half of kernel
    int I2 = (N1-j)*N2 + k; // linear index for re-use of lower half

    int e = 2 * I;

    float reMx = fftMx[e  ];
    float imMx = fftMx[e+1];

    float Kxx;
    if (j < N1/2 + 1) {
        Kxx = fftKxx[I];
    } else {
        Kxx = fftKxx[I2];
    }

    fftMx[e  ] = reMx * Kxx;
    fftMx[e+1] = imMx * Kxx;
}

