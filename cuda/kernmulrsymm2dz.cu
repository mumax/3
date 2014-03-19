// 2D Z (out-of-plane only) micromagnetic kernel multiplication:
// Mz = Kzz * Mz
// Using the same symmetries as kernmulrsymm3d.cu
extern "C" __global__ void
kernmulRSymm2Dz(float* __restrict__  fftMz, float* __restrict__  fftKzz, int Nx, int Ny) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if(ix>= Nx || iy>=Ny) {
		return;
	}

	int I = iy*Nx + ix;
	int e = 2 * I;

	float reMz = fftMz[e  ];
	float imMz = fftMz[e+1];

	if (iy > Ny/2) {
		iy = Ny-iy;
	}
	I = iy*Nx + ix;

	float Kzz = fftKzz[I];

	fftMz[e  ] = reMz * Kzz;
	fftMz[e+1] = imMz * Kzz;
}

