
// Copies src (3D array, size S0 x S1 x S2) to dst (3D array, size D0 x D1 x D2).
// If src is larger than dst, only the src part that fits dst is copied.
// If src is smaller than dst, the remainder of dst is zero-padded.
// E.g.:
// 	a b  ->  a
//	c d
//	
//	a    ->  a 0
//	         0 0
//
// Launch config:
//  dim3 gridSize(divUp(D2, BLOCKSIZE), divUp(D1, BLOCKSIZE), 1); // range over destination size
//  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
__global__ void copypad(float* dst, int D0, int D1, int D2, float* src, int S0, int S1, int S2){

   int j = blockIdx.y * blockDim.y + threadIdx.y;
   int k = blockIdx.x * blockDim.x + threadIdx.x;

  // this check makes it work for padding as well as for unpadding.
  for (int i=0; i<D0; i++){
    if (j<D1 && k<D2){ // if we are in the destination array we should write something
      if(i<S0 && j<S1 && k<S2){ // we are in the source array: copy the source
        dst[i*D1*D2 + j*D2 + k] = src[i*S1*S2 + j*S2 + k];
      }else{ // we are out of the source array: write zero
        dst[i*D1*D2 + j*D2 + k] = 0.0f; 
      }
    }
  }
}



