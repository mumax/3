#ifndef _FLOAT3_H_
#define _FLOAT3_H_

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
inline __device__ float dot(float3 a, float3 b) { 
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

// cross product
inline __device__ float3 cross(float3 a, float3 b) { 
	return make_float3( a.y*b.z - a.z*b.y,  a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

// length of the 3-components vector
inline __device__ float len(float3 a) {
	return sqrtf(dot(a,a));
}

// returns a normalized copy of the 3-components vector
inline __device__ float3 normalized(float3 a){
    float veclen = (len(a) != 0.0f) ? 1.0f / len(a) : 0.0f;
	return veclen * a;
}

// square
inline __device__ float pow2(float x){
	return x * x;
}


// pow(x, 3)
inline __device__ float pow3(float x){
	return x * x * x;
}


// pow(x, 4)
inline __device__ float pow4(float x){
	float s = x*x;
	return s*s;
}

#define is0(m) ( dot(m, m) == 0.0f )

#endif
