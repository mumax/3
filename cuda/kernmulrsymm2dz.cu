// 2D Z (out-of-plane only) micromagnetic kernel multiplication:
// Mz = Kzz * Mz
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
kernmulRSymm2Dz(float* __restrict__  fftMz, float* __restrict__  fftKzz, int Nx, int Ny) {

    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(j>= Ny || k>=Nx) {
        return;
    }

    int I = j*Ny + k;       // linear index for upper half of kernel
    int I2 = (Ny-j)*Nx + k; // linear index for re-use of lower half

    int e = 2 * I;

    float reMz = fftMz[e  ];
    float imMz = fftMz[e+1];

    float Kzz;
    if (j < Ny/2 + 1) {
        Kzz = fftKzz[I];
    } else {
        Kzz = fftKzz[I2];
    }

    fftMz[e  ] = reMz * Kzz;
    fftMz[e+1] = imMz * Kzz;
}

