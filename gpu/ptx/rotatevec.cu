extern "C" __global__ void
rotatevec(float* vx, float* vy, float* vz, 
          float* dx, float* dy, float* dz,
          float factor, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {

    	float Vx = vx[i] + factor * dx[i];
    	float Vy = vy[i] + factor * dy[i];
    	float Vz = vz[i] + factor * dz[i];
    	
		float norm = 1.0f;
//		float norm = sqrtf(Vx*Vx + Vy*Vy + Vz*Vz);
//		if (norm == 0){ norm = 1; }

		vx[i] = Vx / norm;
		vy[i] = Vy / norm;
		vz[i] = Vz / norm;
	}
}

