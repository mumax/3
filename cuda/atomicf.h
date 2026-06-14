#ifndef _ATOMICF_H_
#define _ATOMICF_H_

// Atomic max of abs value.
inline __device__ void atomicFmaxabs(float* a, float b){
	b = fabs(b);
	atomicMax((int*)(a), *((int*)(&b)));
}

#endif
