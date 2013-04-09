#include "float3.h"
#include "stencil.h"
#include "mask.h"

// (ux, uy, uz) is 0.5 * U_spintorque / cellsize(x, y, z)
extern "C" __global__ void
addzhanglitorque(float* __restrict__    tx, float* __restrict__    ty, float* __restrict__    tz,
                 float* __restrict__    mx, float* __restrict__    my, float* __restrict__    mz,
                 float                  ux, float                  uy, float                  uz,
                 float* __restrict__ jmapx, float* __restrict__ jmapy, float* __restrict__ jmapz,
                 float alpha, float epsillon,
                 int N0, int N1, int N2){

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j >= N1 || k >= N2){
		return;
	}

	for(int i=0; i<N0; i++){
		int I = idx(i, j, k);

		ux *= loadmask(jmapx, I);
		uy *= loadmask(jmapy, I);
		uz *= loadmask(jmapz, I);

		float3 dm_dx = make_float3(delta(mx, 1,0,0), delta(my, 1,0,0), delta(mz, 1,0,0)); // ∂m/∂x
		float3 dm_dy = make_float3(delta(mx, 0,1,0), delta(my, 0,1,0), delta(mz, 0,1,0)); // ∂m/∂y
		float3 dm_dz = make_float3(delta(mx, 0,0,1), delta(my, 0,0,1), delta(mz, 0,0,1)); // ∂m/∂z

		float3 hspin  = ux*dm_dx + uy*dm_dy + uz*dm_dz; // (u·∇)m
   		float3 m      = make_float3(mx[I], my[I], mz[I]); 
		float  gilb   = 1./(1. + alpha*alpha);
		float3 torque = -gilb*( (1+alpha*epsillon) * cross(m, cross(m, hspin)) 
		                      + (epsillon-alpha)   * cross(m, hspin)          );
	
		// write back, adding to torque
		tx[I] += torque.x;
		ty[I] += torque.y;
		tz[I] += torque.z;
	}
}
  
