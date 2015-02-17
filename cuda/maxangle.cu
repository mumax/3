#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"

// See maxangle.go for more details.
extern "C" __global__ void
setmaxangle(float* __restrict__ dst,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            float* __restrict__ aLUT2d, uint8_t* __restrict__ regions,
            int Nx, int Ny, int Nz, uint8_t PBC) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= Nx || iy >= Ny || iz >= Nz) {
		return;
	}

	// central cell
	int I = idx(ix, iy, iz);
	float3 m0 = make_float3(mx[I], my[I], mz[I]);

	if (is0(m0)) {
		return;
	}

	uint8_t r0 = regions[I];
	float angle  = 0.0f;

	int i_;    // neighbor index
	float3 m_; // neighbor mag
	float a__; // inter-cell exchange stiffness

	// left neighbor
	i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
	m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
	m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
	a__ = aLUT2d[symidx(r0, regions[i_])];
	if (a__ != 0) {angle = max(angle, acosf(dot(m_,m0)));}

	// right neighbor
	i_  = idx(hclampx(ix+1), iy, iz);
	m_  = make_float3(mx[i_], my[i_], mz[i_]);
	m_  = ( is0(m_)? m0: m_ );
	a__ = aLUT2d[symidx(r0, regions[i_])];
	if (a__ != 0) {angle = max(angle, acosf(dot(m_,m0)));}

	// back neighbor
	i_  = idx(ix, lclampy(iy-1), iz);
	m_  = make_float3(mx[i_], my[i_], mz[i_]);
	m_  = ( is0(m_)? m0: m_ );
	a__ = aLUT2d[symidx(r0, regions[i_])];
	if (a__ != 0) {angle = max(angle, acosf(dot(m_,m0)));}

	// front neighbor
	i_  = idx(ix, hclampy(iy+1), iz);
	m_  = make_float3(mx[i_], my[i_], mz[i_]);
	m_  = ( is0(m_)? m0: m_ );
	a__ = aLUT2d[symidx(r0, regions[i_])];
	if (a__ != 0) {angle = max(angle, acosf(dot(m_,m0)));}

	// only take vertical derivative for 3D sim
	if (Nz != 1) {
		// bottom neighbor
		i_  = idx(ix, iy, lclampz(iz-1));
		m_  = make_float3(mx[i_], my[i_], mz[i_]);
		m_  = ( is0(m_)? m0: m_ );
		a__ = aLUT2d[symidx(r0, regions[i_])];
		if (a__ != 0) {angle = max(angle, acosf(dot(m_,m0)));}

		// top neighbor
		i_  = idx(ix, iy, hclampz(iz+1));
		m_  = make_float3(mx[i_], my[i_], mz[i_]);
		m_  = ( is0(m_)? m0: m_ );
		a__ = aLUT2d[symidx(r0, regions[i_])];
		if (a__ != 0) {angle = max(angle, acosf(dot(m_,m0)));}
	}

	dst[I] = angle;
}

