// Rotates unit vector v in the direction pointed by d.
extern "C" __global__ void
llgtorque(float* vx, float* vy, float* vz, 
          float* dx, float* dy, float* dz,
		  int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {

    	float Vx = vx[i] + dx[i];
    	float Vy = vy[i] + dy[i];
    	float Vz = vz[i] + dz[i];
    	
		float norm = sqrtf(Vx*Vx + Vy*Vy + Vz*Vz);
		if (norm == 0){ norm = 1; }

		vx[i] = Vx / norm;
		vy[i] = Vy / norm;
		vz[i] = Vz / norm;
	}
}

