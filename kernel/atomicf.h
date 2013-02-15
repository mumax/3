#ifndef _ATOMICF_H_
#define _ATOMICF_H_

// Atomic min.
inline __device__ void atomicFmin(float* address, float val){
    unsigned int* address_as_i = (unsigned int*)address;
    unsigned int old = *address_as_i
	unsigned int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Atomic max.
inline __device__ void atomicFmax(float* a, float b){
	atomicMax(a, b);
}



#endif

