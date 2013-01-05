#ifndef _COMMON_FUNC_H_
#define _COMMON_FUNC_H_

// This file implements common functions on float3 (vector).
// Author: Mykola Dvornik, Arne Vansteenkiste

inline __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ void operator+=(float3 &a, float3 b) {
    a.x += b.x; 
	a.y += b.y; 
	a.z += b.z;
}

inline __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

inline __device__ void operator-=(float3 &a, float3 b) {
    a.x -= b.x; 
	a.y -= b.y; 
	a.z -= b.z;
}

inline __device__ float3 operator*(float s, float3 a) {
    return make_float3(s*a.x, s*a.y, s*a.z);
}

inline __device__ float3 operator*(float3 a, float s) {
    return make_float3(s*a.x, s*a.y, s*a.z);
}

inline __device__ void operator*=(float3 &a, float s) {
    a.x *= s; 
	a.y *= s; 
	a.z *= s;
}

// dot product
inline __device__ float dotf(float3 a, float3 b) { 
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

// cross product in LHR system 
inline __device__ float3 crossf(float3 a, float3 b) { 
	return make_float3( - a.y*b.z + a.z*b.y,  - a.z*b.x + a.x*b.z, - a.x*b.y + a.y*b.x); 
}

// lenght of the 3-components vector
inline __device__ float len(float3 a) {
	return sqrtf(dotf(a,a));
}

// returns a normalized copy of the 3-components vector
inline __device__ float3 normalized(float3 a){
    float veclen = (len(a) != 0.0f) ? 1.0f / len(a) : 0.0f;
	return veclen * a;
}

// square
inline __device__ float sqr(float x){
	return x * x;
}

#endif
