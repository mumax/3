
// 2D micromagnetic kernel multiplication:
// |Mx|   |Kxx 0   0  |   |Mx|
// |My| = |0   Kyy Kyz| * |My|
// |Mz|   |0   Kyz Kzz|   |Mz|
//
// ~kernel has mirror symmetry along Y-axis,
// apart form first row,
// and is only stored (roughly) half:
//
// K00, K11, K22:
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
kernmulRSymm2D(float* fftMx,  float* fftMy,  float* fftMz,
        float* fftKxx, float* fftKyy, float* fftKzz, float* fftKyz, 
		int N1, int N2){

	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	if(j>= N1 || k>=N2){
 		return;	
	}

--- TODO: loop over entire array, fetch half Kernel

	int I = j*N2 + k; // linear index

    float Kxx = fftKxx[I];  --- branch here?
    float Kyy = fftKyy[I];
    float Kzz = fftKzz[I];
    float Kyz = fftKyz[I];

  	int e = 2 * I;

    float reMx = fftMx[e  ];
    float imMx = fftMx[e+1];
    float reMy = fftMy[e  ];
    float imMy = fftMy[e+1];
    float reMz = fftMz[e  ];
    float imMz = fftMz[e+1];

    fftMx[e  ] = reMx * Kxx;
    fftMx[e+1] = imMx * Kxx;
    fftMy[e  ] =            reMy * Kyy + reMz * Kyz;
    fftMy[e+1] =            imMy * Kyy + imMz * Kyz;
    fftMz[e  ] =            reMy * Kyz + reMz * Kzz;
    fftMz[e+1] =            imMy * Kyz + imMz * Kzz;

}

