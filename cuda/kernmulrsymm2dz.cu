// 2D Z (out-of-plane only) micromagnetic kernel multiplication:
// Mz = Kzz * Mz
//
// fftKzz only stores the non-redundant rows iy in [0, Ny/2].
// Launch grid covers only that half range; each thread handles row iy
// and, if it has a distinct mirror row Ny-iy (same Kzz value), handles
// that row too, avoiding launching threads for rows that would just
// re-read the same kernel value.
extern "C" __global__ void
kernmulRSymm2Dz(float* __restrict__  fftMz, float* __restrict__  fftKzz, int Nx, int Ny) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix>=Nx || iy>Ny/2) {
        return;
    }

    int I = iy*Nx + ix;
    float Kzz = fftKzz[I];

    // row iy
    {
        int e = 2 * (iy*Nx + ix);
        float reMz = fftMz[e  ];
        float imMz = fftMz[e+1];
        fftMz[e  ] = reMz * Kzz;
        fftMz[e+1] = imMz * Kzz;
    }

    // mirror row Ny-iy, if distinct from iy
    if (iy != 0 && 2*iy != Ny) {
        int my = Ny - iy;
        int e = 2 * (my*Nx + ix);
        float reMz = fftMz[e  ];
        float imMz = fftMz[e+1];
        fftMz[e  ] = reMz * Kzz;
        fftMz[e+1] = imMz * Kzz;
    }
}

