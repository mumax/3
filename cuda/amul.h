#ifndef _AMUL_H_
#define _AMUL_H_

// Returns mul * arr[i], or mul when arr == NULL;
inline __device__ float amul(float *arr, float mul, int i){
		return (arr == NULL)? (mul): (mul * arr[i]);
}

// Returns 1/Msat, or 0 when Msat == 0.
inline __device__ float inv_Msat(float *Ms_, float Ms_mul, int i){
	float ms = amul(Ms_, Ms_mul, i);
	if (ms == 0.0f){
		return 0.0f;	
	}else{
		return 1.0f / ms;	
	}
}

#endif
