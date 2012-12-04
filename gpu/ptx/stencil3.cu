// Accumulating 3rd oder 3D stencil operation.
// 
// 	dst[i,j,k] += w0*src[i,j,k] 
// 	            + wr*src[i,j,k+1] + wl*src[i,j,k-1]
// 	            + wu*src[i,j+1,k] + wd*src[i,j-1,k]
// 	            + wt*src[i+1,j,k] + wb*src[i-1,j,k]
// 
// (r,l,u,d,t,b mean right, left, up, down, top, bottom)
// Clamping boundary conditions.
// This is a "naive" implementation perfect for verifying
// and benchmarking implementations with shared memory.
// TODO: wrap

// clamps i between 0 and N-1
inline __device__ int clamp(int i, int N){
	return min( max(i, 0) , N-1 );
}

// 3D array indexing
#define idx(i,j,k) ((i)*N1*N2 + (j)*N2 + (k))

extern "C" __global__ void
stencil3(float* __restrict__ dst, float* __restrict__ src, 
          float w0, float wt, float wb, float wu, float wd, float wl, float wr, 
          int wrap0, int wrap1, int wrap2, int N0, int N1, int N2){

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j >= N1 || k >= N2){
		return;
	}

	for(int i=0; i<N0; i++){

		float H = 0;
		if(w0 != 0.f){ H += w0 * src[idx(i,j,k)]; }
		if(wt != 0.f){ H += wt*src[idx(clamp(i+1,N0), j, k)] + wb*src[idx(clamp(i-1,N0), j, k)]; }
		if(wu != 0.f){ H += wu*src[idx(i, clamp(j+1,N1), k)] + wd*src[idx(i, clamp(j-1,N1), k)]; }
		if(wr != 0.f){ H += wr*src[idx(i, j, clamp(k+1,N2))] + wl*src[idx(i, j, clamp(k-1,N2))]; }

		dst[idx(i,j,k)] += H;
	}
}

