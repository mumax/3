// 3D micromagnetic kernel multiplication:
//
// |Mx|   |Kxx Kxy Kxz|   |Mx|
// |My| = |Kxy Kyy Kyz| * |My|
// |Mz|   |Kxz Kyz Kzz|   |Mz|
//
// without assuming any spatial symmetry of the kernel.

extern "C" __global__ void
kernmulC3D(float* __restrict__  fftMx,  float* __restrict__  fftMy,  float* __restrict__  fftMz,
           float* __restrict__  fftKxx, float* __restrict__  fftKyy, float* __restrict__  fftKzz,
           float* __restrict__  fftKyz, float* __restrict__  fftKxz, float* __restrict__  fftKxy,
           int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    // fetch (complex) FFT'ed magnetization and kernel elements
    int I = (iz*Ny + iy)*Nx + ix;
    int e = 2 * I;
    float reMx = fftMx[e]; float imMx = fftMx[e+1];
    float reMy = fftMy[e]; float imMy = fftMy[e+1];
    float reMz = fftMz[e]; float imMz = fftMz[e+1];
    float reKxx = fftKxx[e]; float imKxx = fftKxx[e+1];
    float reKyy = fftKyy[e]; float imKyy = fftKyy[e+1];
    float reKzz = fftKzz[e]; float imKzz = fftKzz[e+1];
    float reKyz = fftKyz[e]; float imKyz = fftKyz[e+1];
    float reKxz = fftKxz[e]; float imKxz = fftKxz[e+1];
    float reKxy = fftKxy[e]; float imKxy = fftKxy[e+1];

    // m * K complex matrix multiplication, overwrite m with result.
    fftMx[e  ] = (reMx * reKxx - imKxx * imMx) + (reMy * reKxy - imKxy * imMy) + (reMz * reKxz - imKxz * imMz);
    fftMx[e+1] = (imMx * reKxx + reMx * imKxx) + (imMy * reKxy + reMy * imKxy) + (imMz * reKxz + reMz * imKxz);
    fftMy[e  ] = (reMx * reKxy - imKxy * imMx) + (reMy * reKyy - imKyy * imMy) + (reMz * reKyz - imKyz * imMz);
    fftMy[e+1] = (imMx * reKxy + reMx * imKxy) + (imMy * reKyy + reMy * imKyy) + (imMz * reKyz + reMz * imKyz);
    fftMz[e  ] = (reMx * reKxz - imKxz * imMx) + (reMy * reKyz - imKyz * imMy) + (reMz * reKzz - imKzz * imMz);
    fftMz[e+1] = (imMx * reKxz + reMx * imKxz) + (imMy * reKyz + reMy * imKyz) + (imMz * reKzz + reMz * imKzz);
}
