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
               int Nx, int Ny, int Nz) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if(ix>= Nx || iy>= Ny || iz>=Nz) {
		return;
	}

	float Kxx, Kyy, Kzz, Kxy, Kxz, Kyz;

	int I = (iz*Ny + iy)*Nx + ix;
	int e = 2 * I;
	float reMx = fftMx[e  ];
	float imMx = fftMx[e+1];
	float reMy = fftMy[e  ];
	float imMy = fftMy[e+1];
	float reMz = fftMz[e  ];
	float imMz = fftMz[e+1];

	// symmetry factors
	float fyz = 1.0f;
	float fxz = 1.0f;
	float fxy = 1.0f;

	if (iy > Ny/2) {
		iy = Ny-iy;
		fyz = -fyz;
		fxy = -fxy;
	}
	if (iz > Nz/2) {
		iz = Nz-iz;
		fyz = -fyz;
		fxz = -fxz;
	}

	I = (iz*Ny + iy)*Nx + ix;

	Kxx = fftKxx[I];
	Kyy = fftKyy[I];
	Kzz = fftKzz[I];
	Kyz = fyz * fftKyz[I];
	Kxz = fxz * fftKxz[I];
	Kxy = fxy * fftKxy[I];

	fftMx[e  ] = reMx * Kxx + reMy * Kxy + reMz * Kxz;
	fftMx[e+1] = imMx * Kxx + imMy * Kxy + imMz * Kxz;
	fftMy[e  ] = reMx * Kxy + reMy * Kyy + reMz * Kyz;
	fftMy[e+1] = imMx * Kxy + imMy * Kyy + imMz * Kyz;
	fftMz[e  ] = reMx * Kxz + reMy * Kyz + reMz * Kzz;
	fftMz[e+1] = imMx * Kxz + imMy * Kyz + imMz * Kzz;
}

