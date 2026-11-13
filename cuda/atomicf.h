#ifndef _ATOMICF_H_
#define _ATOMICF_H_

// Atomic max of abs value.
// The accumulator and all candidates are non-negative (b = fabs(b) and the
// destination is seeded non-negative), so the IEEE-754 bit pattern is monotonic
// with the float value and an integer max over the bits is correct.
//
// On NVIDIA (the #else branch, upstream code) this is a single integer
// atomicMax on the float reinterpreted as int. On AMD CDNA an integer atomicMax
// can be silently dropped on coarse-grained memory, so the HIP branch uses an
// atomicCAS loop instead, which is honored on every ROCm coherence mode and is
// equivalent for non-negative inputs.
#ifdef __HIP_PLATFORM_AMD__
inline __device__ void atomicFmaxabs(float* a, float b){
	b = fabs(b);
	int  bbits = __float_as_int(b);
	int* aint  = (int*)(a);
	int  old   = *aint;
	int  assumed;
	do {
		assumed = old;
		if (__int_as_float(assumed) >= b) break;
		old = atomicCAS(aint, assumed, bbits);
	} while (assumed != old);
}
#else
inline __device__ void atomicFmaxabs(float* a, float b){
	b = fabs(b);
	atomicMax((int*)(a), *((int*)(&b)));
}
#endif

#endif
