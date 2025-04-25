#include <cuComplex.h>

// Calculates the summand F(-k) · [k × F(k)] / k^2
extern "C" __global__ void
solidanglefouriersummand(float* __restrict__  summand_array,
                    float* __restrict__ FkX_array, float* __restrict__ FkY_array, float* __restrict__ FkZ_array,
                    int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix>= Nx || iy>= Ny || iz>=Nz) {
        return;
    }

    float kx = static_cast<float>(ix) / Nx;
    float ky = static_cast<float>(iy) / Ny;
    float kz = static_cast<float>(iz) / Nz;

    // Account for positive and negative frequencies (k-space values are in the range [-1/2, 1/2])
    if (ix >= Nx/2)
        kx -= 1.0f;
    if (iy >= Ny/2)
        ky -= 1.0f;
    if (iz >= Nz/2)
        kz -= 1.0f;

    float k2 = kx*kx + ky*ky + kz*kz;

    int I = (iz*Ny + iy)*Nx + ix;
    int e = 2 * I;

    // Avoid division by zero at kx = ky = kz = 0
    if (k2 == 0.0f) {
        summand_array[I] = 0.0f;

    } else {

        float reFkX  =  FkX_array[e  ];
        float reFkY  =  FkY_array[e  ];
        float reFkZ  =  FkZ_array[e  ];
        float imFkX  =  FkX_array[e+1];
        float imFkY  =  FkY_array[e+1];
        float imFkZ  =  FkZ_array[e+1];
        float imFmkX = -FkX_array[e+1];
        float imFmkY = -FkY_array[e+1];
        float imFmkZ = -FkZ_array[e+1];

        cuDoubleComplex FkX  = make_cuDoubleComplex(reFkX, imFkX);
        cuDoubleComplex FkY  = make_cuDoubleComplex(reFkY, imFkY);
        cuDoubleComplex FkZ  = make_cuDoubleComplex(reFkZ, imFkZ);
        cuDoubleComplex FmkX = make_cuDoubleComplex(reFkX, imFmkX);
        cuDoubleComplex FmkY = make_cuDoubleComplex(reFkY, imFmkY);
        cuDoubleComplex FmkZ = make_cuDoubleComplex(reFkZ, imFmkZ);

        cuDoubleComplex kx_comp = make_cuDoubleComplex(kx, 0.0f);
        cuDoubleComplex ky_comp = make_cuDoubleComplex(ky, 0.0f);
        cuDoubleComplex kz_comp = make_cuDoubleComplex(kz, 0.0f);

        // Calculate F(-k) x (k · F(k)) / k^2
        float summand = cuCimag(
                            cuCadd(
                                cuCadd(
                                    cuCmul(FmkX, cuCsub(cuCmul(ky_comp, FkZ), cuCmul(kz_comp, FkY))),
                                    cuCmul(FmkY, cuCsub(cuCmul(kz_comp, FkX), cuCmul(kx_comp, FkZ)))
                                ),
                                cuCmul(FmkZ, cuCsub(cuCmul(kx_comp, FkY), cuCmul(ky_comp, FkX)))
                            )
                        );
        
        summand /= k2;
        summand_array[I] = summand;
    }
}
