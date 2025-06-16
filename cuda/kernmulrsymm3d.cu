// 3D micromagnetic kernel multiplication:
//
// |Mx|   |Kxx Kxy Kxz|   |Mx|
// |My| = |Kxy Kyy Kyz| * |My|
// |Mz|   |Kxz Kyz Kzz|   |Mz|
//
// ~kernel has mirror symmetry along Y and Z-axis,
// apart from first row,
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
// -bbbb
// -aaaa

extern "C" __global__ void
kernmulRSymm3D(float* __restrict__  fftMx,  float* __restrict__  fftMy,  float* __restrict__  fftMz,
               float* __restrict__  fftKxx, float* __restrict__  fftKyy, float* __restrict__  fftKzz,
               float* __restrict__  fftKyz, float* __restrict__  fftKxz, float* __restrict__  fftKxy,
               int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix>= Nx || iy>= Ny || iz>=Nz) {
        return;
    }

    // fetch (complex) FFT'ed magnetization
    int I = (iz*Ny + iy)*Nx + ix;
    int e = 2 * I;
    float reMx = fftMx[e  ];
    float imMx = fftMx[e+1];
    float reMy = fftMy[e  ];
    float imMy = fftMy[e+1];
    float reMz = fftMz[e  ];
    float imMz = fftMz[e+1];

    // fetch kernel

    // minus signs are added to some elements if
    // reconstructed from symmetry.
    float signYZ = 1.0f;
    float signXZ = 1.0f;
    float signXY = 1.0f;

    // use symmetry to fetch from redundant parts:
    // mirror index into first quadrant and set signs.
    if (iy > Ny/2) {
        iy = Ny-iy;
        signYZ = -signYZ;
        signXY = -signXY;
    }
    if (iz > Nz/2) {
        iz = Nz-iz;
        signYZ = -signYZ;
        signXZ = -signXZ;
    }

    // fetch kernel element from non-redundant part
    // and apply minus signs for mirrored parts.
    I = (iz*(Ny/2+1) + iy)*Nx + ix; // Ny/2+1: only half is stored
    float Kxx = fftKxx[I];
    float Kyy = fftKyy[I];
    float Kzz = fftKzz[I];
    float Kyz = fftKyz[I] * signYZ;
    float Kxz = fftKxz[I] * signXZ;
    float Kxy = fftKxy[I] * signXY;

    // m * K matrix multiplication, overwrite m with result.
    fftMx[e  ] = reMx * Kxx + reMy * Kxy + reMz * Kxz;
    fftMx[e+1] = imMx * Kxx + imMy * Kxy + imMz * Kxz;
    fftMy[e  ] = reMx * Kxy + reMy * Kyy + reMz * Kyz;
    fftMy[e+1] = imMx * Kxy + imMy * Kyy + imMz * Kyz;
    fftMz[e  ] = reMx * Kxz + reMy * Kyz + reMz * Kzz;
    fftMz[e+1] = imMx * Kxz + imMy * Kyz + imMz * Kzz;
}

