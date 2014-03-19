// 2D XY (in-plane) micromagnetic kernel multiplication:
// |Mx| = |Kxx Kxy| * |Mx|
// |My|   |Kyx Kyy|   |My|
// Using the same symmetries as kernmulrsymm3d.cu
extern "C" __global__ void
kernmulRSymm2Dxy(float* __restrict__  fftMx,  float* __restrict__  fftMy,
                 float* __restrict__  fftKxx, float* __restrict__  fftKyy, float* __restrict__  fftKxy,
                 int Nx, int Ny) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if(ix>= Nx || iy>=Ny) {
		return;
	}

	int I = iy*Nx + ix;
	int e = 2 * I;

	float reMx = fftMx[e  ];
	float imMx = fftMx[e+1];
	float reMy = fftMy[e  ];
	float imMy = fftMy[e+1];

	// symmetry factor
	float fxy = 1.0f;
	if (iy > Ny/2) {
		iy = Ny-iy;
		fxy = -fxy;
	}
	I = iy*Nx + ix;

	float Kxx = fftKxx[I];
	float Kyy = fftKyy[I];
	float Kxy = fxy * fftKxy[I];

	fftMx[e  ] = reMx * Kxx + reMy * Kxy;
	fftMx[e+1] = imMx * Kxx + imMy * Kxy;
	fftMy[e  ] = reMx * Kxy + reMy * Kyy;
	fftMy[e+1] = imMx * Kxy + imMy * Kyy;
}

