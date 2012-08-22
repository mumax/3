#define remul(a, b, c, d) ((a)*(c)-(b)*(d))
#define immul(a, b, c, d) ((a)*(d)+(b)*(c))

// General kernel multiplication
// with complex, not necessarily symmetric kernel.
// |Mx|   |Kxx Kxy Kxz|   |Mx|
// |My| = |Kyx Kyy Kyz| * |My|
// |Mz|   |Kzx Kzy Kzz|   |Mz|
extern "C" __global__ void 
kernmulRSymm(float* Mx,  float* My,  float* Mz,
             float* Kxx, float* Kyy, float* Kzz,
             float* Kyz, float* Kxz, float* Kxy, 
             float* Kzy, float* Kzx, float* Kyx, 
             int N){

  int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
  int e = 2 * i;

  if(i < N){
    float reMx = Mx[e  ];
    float imMx = Mx[e+1];

    float reMy = My[e  ];
    float imMy = My[e+1];

    float reMz = Mz[e  ];
    float imMz = Mz[e+1];

    float reKxx = Kxx[e  ];
    float reKyy = Kyy[e  ];
    float reKzz = Kzz[e  ];
    float imKxx = Kxx[e+1];
    float imKyy = Kyy[e+1];
    float imKzz = Kzz[e+1];

    float reKyz = Kyz[e  ];
    float reKxz = Kxz[e  ];
    float reKxy = Kxy[e  ];
    float imKyz = Kyz[e+1];
    float imKxz = Kxz[e+1];
    float imKxy = Kxy[e+1];

    float reKzy = Kzy[e  ];
    float reKzx = Kzx[e  ];
    float reKyx = Kyx[e  ];
    float imKzy = Kzy[e+1];
    float imKzx = Kzx[e+1];
    float imKyx = Kyx[e+1];

	Mx[e  ] = remul(reMx, imMx, reKxx, imKxx) + remul(reMy, imMy, reKxy, imKxy) + remul(reMx, imMx, reKxz, imKxz);
//	Mx[e+1] =
//	         
//	My[e  ] =
//	My[e+1] =
//	         
//	Mz[e  ] =
//	Mz[e+1] =



    //fftMx[e  ] = reMx * Kxx + reMy * Kxy + reMz * Kxz;
    //fftMx[e+1] = imMx * Kxx + imMy * Kxy + imMz * Kxz;

    //fftMy[e  ] = reMx * Kxy + reMy * Kyy + reMz * Kyz;
    //fftMy[e+1] = imMx * Kxy + imMy * Kyy + imMz * Kyz;

    //fftMz[e  ] = reMx * Kxz + reMy * Kyz + reMz * Kzz;
    //fftMz[e+1] = imMx * Kxz + imMy * Kyz + imMz * Kzz;
  }
}

