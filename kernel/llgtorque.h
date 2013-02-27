#ifndef _LLGTORQUE_H_
#define _LLGTORQUE_H_

#include "float3.h"

inline __device__ float3 _llgtorque(float3 M, float3 H, float alpha) {
	float3 MxH = cross(M, H);
	float gilb = -1.0f / (1.0f + alpha * alpha);
	return gilb * (MxH + (alpha/len(M)) * cross(M, MxH));
}

#endif

