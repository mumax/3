// 2D YZ (in-plane) micromagnetic kernel multiplication:
// |My| = |Kyy Kyz| * |My|
// |Mz|   |Kyz Kzz|   |Mz|
//
// ~kernel has mirror symmetry along Y-axis,
// apart form first row,
// and is only stored (roughly) half:
//
// K11, K22:
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
extern "C" __global__ void 
kernmulRSymm2Dyz(float* __restrict__  fftMy,  float* __restrict__  fftMz,
                 float* __restrict__  fftKyy, float* __restrict__  fftKzz, float* __restrict__  fftKyz, 
                 int N1, int N2){

	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	if(j>= N1 || k>=N2){
 		return;	
	}

	int I = j*N2 + k;       // linear index for upper half of kernel
	int I2 = (N1-j)*N2 + k; // linear index for re-use of lower half

    float Kyy;
    float Kzz;
    float Kyz;

	if (j < N1/2 + 1){
		Kyy = fftKyy[I];
		Kzz = fftKzz[I];
		Kyz = fftKyz[I];
	}else{
		Kyy = fftKyy[I2];
		Kzz = fftKzz[I2];
		Kyz = -fftKyz[I2];
	}

  	int e = 2 * I;

    float reMy = fftMy[e  ];
    float imMy = fftMy[e+1];
    float reMz = fftMz[e  ];
    float imMz = fftMz[e+1];

    fftMy[e  ] = reMy * Kyy + reMz * Kyz;
    fftMy[e+1] = imMy * Kyy + imMz * Kyz;
    fftMz[e  ] = reMy * Kyz + reMz * Kzz;
    fftMz[e+1] = imMy * Kyz + imMz * Kzz;
}

