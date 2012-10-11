__global__ void exchange6(float* h, float* m, 
                          float fac0, float fac1, float fac2,
                          int wrap0, int wrap1, int wrap2,
                          int N0, int N1, int N2){

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j >= N1 || k >= N2){
		return;
	}

	for(int i=0; i<N0; i++){
  		int I = i*N1*N2 + j*N2 + k; // linear array index
  
		float m0 = m[I]; // mag component of central cell 
		float m1, m2 ;   // mag component of neighbors in 2 directions
	
    	// neighbors in I direction
		int idx;
    	if (i-1 >= 0){                         // neighbor in bounds...
      		idx = (i-1)*N1*N2 + j*N2 + k;      // ... no worries
    	} else {                               // neighbor out of bounds...
			if(wrap0){                         // ... PBC?
				idx = (N0-1)*N1*N2 + j*N2 + k; // yes: wrap around!
			}else{                                    
      			idx = I;                       // no: use central m (Neumann BC) 
			}
    	}
		m1 = m[idx];

	 	if (i+1 < N0){
			idx = (i+1)*N1*N2 + j*N2 + k;
	    } else {
			if(wrap0){
				idx = (0)*N1*N2 + j*N2 + k;
			}else{
	      		idx = I;
			}
	    } 
		m2 = m[idx]; 

    	float H = fac2 * ((m1-m0) + (m2-m0));

		// neighbors in J direction
		if (j-1 >= 0){
			idx = i*N1*N2 + (j-1)*N2 + k;
		} else {
			if(wrap1){
				idx = i*N1*N2 + (N1-1)*N2 + k;
			}else{
		  		idx = I;
			}
		}
		m1 = m[idx];
		
		if (j+1 < N1){
		  idx =  i*N1*N2 + (j+1)*N2 + k;
		} else {
			if(wrap1){
				idx = i*N1*N2 + (0)*N2 + k;
			}else{
		  		idx = I;
			}
		} 
		m2 = m[idx];
		
		H += fac1 * ((m1-m0) + (m2-m0));

		// neighbors in K direction
		if (k-1 >= 0){
			idx = i*N1*N2 + j*N2 + (k-1);
		} else {
			if(wrap2){
				idx = i*N1*N2 + j*N2 + (N2-1);
			}else{
		  		idx = I;
			}
		}
		m1 = m[idx];
		
		if (k+1 < N2){
		  idx =  i*N1*N2 + j*N2 + (k+1);
		} else {
			if(wrap2){
				idx = i*N1*N2 + j*N2 + (0);
			}else{
		  		idx = I;
			}
		} 
		m2 = m[idx];
		
		H += fac2 * ((m1-m0) + (m2-m0));

		// Write back to global memory
		h[I] = H;
	}
} 

