// 2D XY (in-plane) micromagnetic kernel multiplication:
// |Mx| = |Kxx Kxy| * |Mx|
// |My|   |Kyx Kyy|   |My|
//
// fftKxx/fftKyy/fftKxy only store the non-redundant rows iy in [0, Ny/2].
// Launch grid covers only that half range; each thread handles row iy
// (using Kxy as-is) and, if it has a distinct mirror row Ny-iy, also
// handles that row (using -Kxy), avoiding launching threads for rows
// that would just re-read the same kernel value.
extern "C" __global__ void
kernmulRSymm2Dxy(float* __restrict__  fftMx,  float* __restrict__  fftMy,
                 float* __restrict__  fftKxx, float* __restrict__  fftKyy, float* __restrict__  fftKxy,
                 int Nx, int Ny) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix>=Nx || iy>Ny/2) {
        return;
    }

    int I = iy*Nx + ix;
    float Kxx = fftKxx[I];
    float Kyy = fftKyy[I];
    float Kxy = fftKxy[I];

    // row iy
    {
        int e = 2 * (iy*Nx + ix);
        float reMx = fftMx[e  ];
        float imMx = fftMx[e+1];
        float reMy = fftMy[e  ];
        float imMy = fftMy[e+1];

        fftMx[e  ] = reMx * Kxx + reMy * Kxy;
        fftMx[e+1] = imMx * Kxx + imMy * Kxy;
        fftMy[e  ] = reMx * Kxy + reMy * Kyy;
        fftMy[e+1] = imMx * Kxy + imMy * Kyy;
    }

    // mirror row Ny-iy, if distinct from iy (sign of Kxy flips)
    if (iy != 0 && 2*iy != Ny) {
        int my = Ny - iy;
        int e = 2 * (my*Nx + ix);
        float reMx = fftMx[e  ];
        float imMx = fftMx[e+1];
        float reMy = fftMy[e  ];
        float imMy = fftMy[e+1];
        float Kxym = -Kxy;

        fftMx[e  ] = reMx * Kxx + reMy * Kxym;
        fftMx[e+1] = imMx * Kxx + imMy * Kxym;
        fftMy[e  ] = reMx * Kxym + reMy * Kyy;
        fftMy[e+1] = imMx * Kxym + imMy * Kyy;
    }
}

