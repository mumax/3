// Accumulating 3rd oder 3D stencil operation.
// 	dst[i,j,k] += w0*src[i,j,k] 
// 	            + wl*src[i,j,k-1] + wr*src[i,j,k+1]
// 	            + wu*src[i,j-1,k] + wd*src[i,j+1,k]
// 	            + wt*src[i-1,j,k] + wb*src[i-1,j,k]
// (l,r,u,d,t,b means left, right, up, down, top, bottom)
// Clamping boundary conditions.
// This is a "naive" implementation perfect for verifying
// and benchmarking implementations with shared memory.

// clamps i between 0 and N-1
inline __device__ int clamp(int i, int N){
	return min( max(i, 0) , N-1 );
}

extern "C" __global__ void
stencil3(float* dst, float* src, 
          float w0, float wl, float wr, float wu, float wd, float wt, float wb,
          int wrap0, int wrap1, int wrap2, int N0, int N1, int N2){

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j >= N1 || k >= N2){
		return;
	}

	for(int i=0; i<N0; i++){

  		int I = i*N1*N2 + j*N2 + k; // linear array index
		float H = w0 * src[I];

    	// neighbors in I direction
      	int idx = clamp(i-1, N0)*N1*N2 + j*N2 + k;
		H += wt*src[idx];

      	idx = clamp(i+1, N0)*N1*N2 + j*N2 + k;
		H += wb*src[idx]; 

		//// neighbors in J direction
		//if (j-1 >= 0){
		//	idx = i*N1*N2 + (j-1)*N2 + k;
		//} else {
		//	if(wrap1){
		//		idx = i*N1*N2 + (N1-1)*N2 + k;
		//	}else{
		//  		idx = I;
		//	}
		//}
		//m1 = m[idx];
		//
		//if (j+1 < N1){
		//  idx =  i*N1*N2 + (j+1)*N2 + k;
		//} else {
		//	if(wrap1){
		//		idx = i*N1*N2 + (0)*N2 + k;
		//	}else{
		//  		idx = I;
		//	}
		//} 
		//m2 = m[idx];
		//
		//H += fac1 * ((m1-m0) + (m2-m0));

		//// neighbors in K direction
		//if (k-1 >= 0){
		//	idx = i*N1*N2 + j*N2 + (k-1);
		//} else {
		//	if(wrap2){
		//		idx = i*N1*N2 + j*N2 + (N2-1);
		//	}else{
		//  		idx = I;
		//	}
		//}
		//m1 = m[idx];
		//
		//if (k+1 < N2){
		//  idx =  i*N1*N2 + j*N2 + (k+1);
		//} else {
		//	if(wrap2){
		//		idx = i*N1*N2 + j*N2 + (0);
		//	}else{
		//  		idx = I;
		//	}
		//} 
		//m2 = m[idx];
		//
		//H += fac2 * ((m1-m0) + (m2-m0));

		// Write back to global memory
		dst[I] += H;
	}
} 

