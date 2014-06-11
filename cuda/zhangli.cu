#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "constants.h"

#define PREFACTOR ((MUB * MU0) / (2 * QE * GAMMA0))

// spatial derivatives without dividing by cell size
#define deltax(in) (in[idx(hclampx(ix+1), iy, iz)] - in[idx(lclampx(ix-1), iy, iz)])
#define deltay(in) (in[idx(ix, hclampy(iy+1), iz)] - in[idx(ix, lclampy(iy-1), iz)])
#define deltaz(in) (in[idx(ix, iy, hclampz(iz+1))] - in[idx(ix, iy, lclampz(iz-1))])

extern "C" __global__ void
addzhanglitorque(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
                 float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                 float* __restrict__ jx, float* __restrict__ jy, float* __restrict__ jz,
                 float cx, float cy, float cz,
                 float* __restrict__ bsatLUT, float* __restrict__ alphaLUT, float* __restrict__ xiLUT, float* __restrict__ polLUT,
                 uint8_t* __restrict__ regions, int Nx, int Ny, int Nz, uint8_t PBC) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= Nx || iy >= Ny || iz >= Nz) {
		return;
	}

	int I = idx(ix, iy, iz);

	uint8_t r = regions[I];
	float alpha = alphaLUT[r];
	float xi    = xiLUT[r];
	float bsat  = bsatLUT[r];
	float pol   = polLUT[r];
	float b = PREFACTOR / (bsat * (1.0f + xi*xi));
	if(bsat == 0.0f) {
		b = 0.0f;
	}
	float Jx = pol*jx[I];
	float Jy = pol*jy[I];
	float Jz = pol*jz[I];

	float3 hspin = make_float3(0.0f, 0.0f, 0.0f); // (u·∇)m
	if (Jx != 0.0f) {
		hspin += (b/cx)*Jx * make_float3(deltax(mx), deltax(my), deltax(mz));
	}
	if (Jy != 0.0f) {
		hspin += (b/cy)*Jy * make_float3(deltay(mx), deltay(my), deltay(mz));
	}
	if (Jz != 0.0f) {
		hspin += (b/cz)*Jz * make_float3(deltaz(mx), deltaz(my), deltaz(mz));
	}

	float3 m      = make_float3(mx[I], my[I], mz[I]);
	float3 torque = (-1.0f/(1.0f + alpha*alpha)) * (
	                    (1.0f+xi*alpha) * cross(m, cross(m, hspin))
	                    +(  xi-alpha) * cross(m, hspin)           );

	// write back, adding to torque
	tx[I] += torque.x;
	ty[I] += torque.y;
	tz[I] += torque.z;
}

