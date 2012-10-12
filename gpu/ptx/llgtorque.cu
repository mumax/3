__global__ void llgtorque(float* tx, float* ty, float* tz,
                          float* mx, float* my, float* mz, 
                          float* hx, float* hy, float* hz, 
						  float alpha, int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {

    	float Mx = mx[i];
    	float My = my[i];
    	float Mz = mz[i];
    	
    	float Hx = hx[i];
    	float Hy = hy[i];
    	float Hz = hz[i];
    	
    	//  m cross H
    	float _mxHx =  My * Hz - Hy * Mz;
    	float _mxHy = -Mx * Hz + Hx * Mz;
    	float _mxHz =  Mx * Hy - Hx * My;

    	// - m cross (m cross H)
    	float _mxmxHx = -My * _mxHz + _mxHy * Mz;
    	float _mxmxHy = +Mx * _mxHz - _mxHx * Mz;
    	float _mxmxHz = -Mx * _mxHy + _mxHx * My;

		float gilb = 1.0f / (1.0f + alpha * alpha);
    	tx[i] = gilb * (_mxmxHx * alpha);
    	ty[i] = gilb * (_mxmxHy * alpha);
    	tz[i] = gilb * (_mxmxHz * alpha);
	}
}

