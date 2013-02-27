#ifndef _LLGTORQUE_H_
#define _LLGTORQUE_H_

#include "float3.h"

inline __device__ float3 _llgtorque(float3 m, float3 H, float alpha) {
	float3 mxH = cross(m, H);
	float gilb = -1.0f / (1.0f + alpha * alpha);
	return gilb * (mxH + alpha/len(m) * cross(m, mxH));
}

#endif

