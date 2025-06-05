// Reconstructs the full Fourier transformed field array for negative wavenumbers using Hermitian symmetry F(-k_x, -k_y, -k_z) = F(k_x, k_y, k_z)^*
extern "C" __global__ void
solidanglefourierfield(float* __restrict__  fftFx_partial,  float* __restrict__  fftFy_partial,  float* __restrict__  fftFz_partial,
                    float* __restrict__ fftFx, float* __restrict__ fftFy, float* __restrict__ fftFz,
                    int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix>= Nx || iy>= Ny || iz>=Nz) {
        return;
    }

    int I = (iz*Ny + iy)*Nx + ix;
    int e = 2 * I;

    int I_partial = (iz*Ny + iy)*(Nx/2+1) + ix;
    int e_partial = 2 * I_partial;

    if (ix <= Nx/2) {

        fftFx[e  ] = fftFx_partial[e_partial  ];
        fftFx[e+1] = fftFx_partial[e_partial+1];
        fftFy[e  ] = fftFy_partial[e_partial  ];
        fftFy[e+1] = fftFy_partial[e_partial+1];
        fftFz[e  ] = fftFz_partial[e_partial  ];
        fftFz[e+1] = fftFz_partial[e_partial+1];

    } else {

        int ix_neg = (Nx - ix) % Nx;
        int iy_neg = (Ny - iy) % Ny;
        int iz_neg = (Nz - iz) % Nz;

        int I_neg = (iz_neg*Ny + iy_neg)*(Nx/2+1) + ix_neg;
        int e_neg = 2 * I_neg;

        // Fill in the rest of the values using Hermitian symmetry: F(-k_x, -k_y, -k_z) = F(k_x, k_y, k_z)^*
        fftFx[e  ] =  fftFx_partial[e_neg  ];
        fftFx[e+1] = -fftFx_partial[e_neg+1];
        fftFy[e  ] =  fftFy_partial[e_neg  ];
        fftFy[e+1] = -fftFy_partial[e_neg+1];
        fftFz[e  ] =  fftFz_partial[e_neg  ];
        fftFz[e+1] = -fftFz_partial[e_neg+1];

    }

}
