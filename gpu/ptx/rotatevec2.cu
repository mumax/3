extern "C" __global__ void
rotatevec2(float* vx, float* vy, float* vz, 
           float* dx1, float* dy1, float* dz1, float factor1, 
           float* dx2, float* dy2, float* dz2, float factor2, 
           int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {

    	float Vx = vx[i] + factor1 * dx1[i] + factor2 * dx2[i];
    	float Vy = vy[i] + factor1 * dy1[i] + factor2 * dy2[i];
    	float Vz = vz[i] + factor1 * dz1[i] + factor2 * dz2[i];
    	
		float norm = sqrtf(Vx*Vx + Vy*Vy + Vz*Vz);
		if (norm == 0){ norm = 1; }

		vx[i] = Vx / norm;
		vy[i] = Vy / norm;
		vz[i] = Vz / norm;
	}
}

