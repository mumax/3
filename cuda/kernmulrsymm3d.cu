// 3D micromagnetic kernel multiplication:
//
// |Mx|   |Kxx Kxy Kxz|   |Mx|
// |My| = |Kxy Kyy Kyz| * |My|
// |Mz|   |Kxz Kyz Kzz|   |Mz|
//
// ~kernel has mirror symmetry along Y and X-axis,
// apart form first row,
// and is only stored (roughly) half:
//
// K11, K22, K02:
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

// 3D array indexing

extern "C" __global__ void
kernmulRSymm3D(float* __restrict__  fftMx,  float* __restrict__  fftMy,  float* __restrict__  fftMz,
               float* __restrict__  fftKxx, float* __restrict__  fftKyy, float* __restrict__  fftKzz,
               float* __restrict__  fftKyz, float* __restrict__  fftKxz, float* __restrict__  fftKxy,
               int N0, int N1, int N2) {

    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(i>= N0 || j>= N1 || k>=N2) {
        return;
    }

    float Kxx, Kyy, Kzz, Kxy, Kxz, Kyz;

    int I = N2*(i*N1 + j) + k;
    int e = 2 * I;
    float reMx = fftMx[e  ];
    float imMx = fftMx[e+1];
    float reMy = fftMy[e  ];
    float imMy = fftMy[e+1];
    float reMz = fftMz[e  ];
    float imMz = fftMz[e+1];

    if (j < N1/2 + 1) {
        Kxx = fftKxx[I];
        Kyy = fftKyy[I];
        Kzz = fftKzz[I];
        Kyz = fftKyz[I];
        Kxz = fftKxz[I];
        Kxy = fftKxy[I];
    } else {
        int I2 = N2*(i*N1 + (N1-j)) + k;
        Kxx =  fftKxx[I2];
        Kyy =  fftKyy[I2];
        Kzz =  fftKzz[I2];
        Kyz = -fftKyz[I2]; // !
        Kxz =  fftKxz[I2];
        Kxy = -fftKxy[I2]; // !
    }

    fftMx[e  ] = reMx * Kxx + reMy * Kxy + reMz * Kxz;
    fftMx[e+1] = imMx * Kxx + imMy * Kxy + imMz * Kxz;
    fftMy[e  ] = reMx * Kxy + reMy * Kyy + reMz * Kyz;
    fftMy[e+1] = imMx * Kxy + imMy * Kyy + imMz * Kyz;
    fftMz[e  ] = reMx * Kxz + reMy * Kyz + reMz * Kzz;
    fftMz[e+1] = imMx * Kxz + imMy * Kyz + imMz * Kzz;
}

