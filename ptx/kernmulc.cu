#define remul(a, b, c, d) ((a)*(c)-(b)*(d))
#define immul(a, b, c, d) ((a)*(d)+(b)*(c))

// General kernel multiplication
// with complex, not necessarily symmetric kernel.
// |Mx|   |Kxx Kxy Kxz|   |Mx|
// |My| = |Kyx Kyy Kyz| * |My|
// |Mz|   |Kzx Kzy Kzz|   |Mz|
extern "C" __global__ void 
kernmulC(float* __restrict__  Mx,  float* __restrict__  My,  float* __restrict__  Mz,
         float* __restrict__  Kxx, float* __restrict__  Kyy, float* __restrict__  Kzz,
         float* __restrict__  Kyz, float* __restrict__  Kxz, float* __restrict__  Kxy, 
         float* __restrict__  Kzy, float* __restrict__  Kzx, float* __restrict__  Kyx, 
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

	Mx[e  ] = remul(reMx, imMx, reKxx, imKxx) + remul(reMy, imMy, reKxy, imKxy) + remul(reMz, imMz, reKxz, imKxz);
	Mx[e+1] = immul(reMx, imMx, reKxx, imKxx) + immul(reMy, imMy, reKxy, imKxy) + immul(reMz, imMz, reKxz, imKxz);

	My[e  ] = remul(reMx, imMx, reKyx, imKyx) + remul(reMy, imMy, reKyy, imKyy) + remul(reMz, imMz, reKyz, imKyz);
	My[e+1] = immul(reMx, imMx, reKyx, imKyx) + immul(reMy, imMy, reKyy, imKyy) + immul(reMz, imMz, reKyz, imKyz);

	Mz[e  ] = remul(reMx, imMx, reKzx, imKzx) + remul(reMy, imMy, reKzy, imKzy) + remul(reMz, imMz, reKzz, imKzz);
	Mz[e+1] = immul(reMx, imMx, reKzx, imKzx) + immul(reMy, imMy, reKzy, imKzy) + immul(reMz, imMz, reKzz, imKzz);
  }
}

