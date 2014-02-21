
// Select and resize one layer for interactive output
extern "C" __global__ void
resize(float* __restrict__  dst, int Dx, int Dy, int Dz,
       float* __restrict__  src, int Sx, int Sy, int Sz,
       int layer, int scalex, int scaley) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix<Dx && iy<Dy) {

		float sum = 0.0f;
		float n = 0.0f;

		for(int J=0; J<scaley; J++) {
			int j2 = iy*scaley+J;

			for(int K=0; K<scalex; K++) {
				int k2 = ix*scalex+K;

				if (j2 < Sy && k2 < Sx) {
					sum += src[(layer*Sy + j2)*Sx + k2];
					n += 1.0f;
				}
			}
		}
		dst[iy*Dx + ix] = sum / n;
	}
}

