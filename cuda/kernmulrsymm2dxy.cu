// 2D XY (in-plane) micromagnetic kernel multiplication:
// |Mx| = |Kxx Kxy| * |Mx|
// |My|   |Kyx Kyy|   |My|
//
// ~kernel has mirror symmetry along Y-axis,
// apart form first row,
// and is only stored (roughly) half:
//
// K11, K22:
// xxxxx
// aaaaa
// bbbbb
// ....
// bbbbb
// aaaaa
//
// K12:
// xxxxx
// aaaaa
// bbbbb
// ...
// -aaaa
// -bbbb
extern "C" __global__ void
kernmulRSymm2Dxy(float* __restrict__  fftMx,  float* __restrict__  fftMy,
                 float* __restrict__  fftKxx, float* __restrict__  fftKyy, float* __restrict__  fftKxy,
                 int Nx, int Ny) {

    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(j>= Ny || k>=Nx) {
        return;
    }

    int I = j*Ny + k;       // linear index for upper half of kernel
    int I2 = (Ny-j)*Nx + k; // linear index for re-use of lower half

    int e = 2 * I;

    float reMy = fftMy[e  ];
    float imMy = fftMy[e+1];
    float reMx = fftMx[e  ];
    float imMx = fftMx[e+1];

    float Kyy, Kxx, Kxy;
    if (j < Ny/2 + 1) {
        Kyy = fftKyy[I];
        Kxx = fftKxx[I];
        Kxy = fftKxy[I];
    } else {
        Kyy =  fftKyy[I2];
        Kxx =  fftKxx[I2];
        Kxy = -fftKxy[I2];
    }

    fftMx[e  ] = reMx * Kxx + reMy * Kxy ;
    fftMx[e+1] = imMx * Kxx + imMy * Kxy ;
    fftMy[e  ] = reMx * Kxy + reMy * Kyy ;
    fftMy[e+1] = imMx * Kxy + imMy * Kyy ;
}

