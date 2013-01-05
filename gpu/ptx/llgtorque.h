#include "common_func.h"

inline __device__ float3 _llgtorque(float3 m, float3 H, float alpha) {
	float3 mxH = crossf(m, H);
	float gilb = -1.0f / (1.0f + alpha * alpha);
	return gilb * (mxH + alpha * crossf(m, mxH));
}

