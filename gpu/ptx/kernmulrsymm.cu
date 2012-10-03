
// Micromagnetic kernel multiplication
// with purely real, symmetric kernel.
// |Mx|   |Kxx Kxy Kxz|   |Mx|
// |My| = |Kxy Kyy Kyz| * |My|
// |Mz|   |Kxz Kyz Kzz|   |Mz|
// Elements may be NULL, interpreted as all zeros.
extern "C" __global__ void 
kernmulRSymm(float* fftMx,  float* fftMy,  float* fftMz,
        float* fftKxx, float* fftKyy, float* fftKzz,
        float* fftKyz, float* fftKxz, float* fftKxy, int N){

  int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
  int e = 2 * i;

  if(i < N){
    float reMx = fftMx[e  ];
    float imMx = fftMx[e+1];

    float reMy = fftMy[e  ];
    float imMy = fftMy[e+1];

    float reMz = fftMz[e  ];
    float imMz = fftMz[e+1];

    float Kxx = (fftKxx == NULL? 0.f: fftKxx[i]);
    float Kyy = (fftKyy == NULL? 0.f: fftKyy[i]);
    float Kzz = (fftKzz == NULL? 0.f: fftKzz[i]);

    float Kyz = (fftKyz == NULL? 0.f: fftKyz[i]);
    float Kxz = (fftKxz == NULL? 0.f: fftKxz[i]);
    float Kxy = (fftKxy == NULL? 0.f: fftKxy[i]);

    fftMx[e  ] = reMx * Kxx + reMy * Kxy + reMz * Kxz;
    fftMx[e+1] = imMx * Kxx + imMy * Kxy + imMz * Kxz;

    fftMy[e  ] = reMx * Kxy + reMy * Kyy + reMz * Kyz;
    fftMy[e+1] = imMx * Kxy + imMy * Kyy + imMz * Kyz;

    fftMz[e  ] = reMx * Kxz + reMy * Kyz + reMz * Kzz;
    fftMz[e+1] = imMx * Kxz + imMy * Kyz + imMz * Kzz;
  }
}

