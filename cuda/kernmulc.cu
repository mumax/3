extern "C" __global__ void
kernmulC(float* __restrict__  fftM, float* __restrict__  fftK, int Nx, int Ny) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if(ix>= Nx || iy>=Ny) {
		return;
	}

	int I = iy*Nx + ix;
	int e = 2 * I;

	float reM = fftM[e  ];
	float imM = fftM[e+1];
	float reK = fftK[e  ];
	float imK = fftK[e+1];

	fftM[e  ] = reM * reK - imM * imK;
	fftM[e+1] = reM * imK + imM * reK;
}

