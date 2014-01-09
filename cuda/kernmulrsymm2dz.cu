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

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix>= Nx || iy>=Ny) {
        return;
    }

    int I = iy*Nx + ix; // linear index for upper half of kernel
    int e = 2 * I;

    float reMz = fftMz[e  ];
    float imMz = fftMz[e+1];

    float Kzz;
    if (iy < Ny/2 + 1) {
        Kzz = fftKzz[I];
    } else {
        int I2 = (Ny-iy)*Nx + ix; // linear index for re-use of lower half
        Kzz = fftKzz[I2];
    }

    fftMz[e  ] = reMz * Kzz;
    fftMz[e+1] = imMz * Kzz;
}

