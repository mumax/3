#ifndef _ATOMICF_H_
#define _ATOMICF_H_

// Atomic min.
inline __device__ void atomicFmin(float* a, float b){
	atomicMin((int*)(a), *((int*)(&b)));
not correct
}

// Atomic max.
inline __device__ void atomicFmax(float* a, float b){
	atomicMax((int*)(a), *((int*)(&b)));
}

#endif

