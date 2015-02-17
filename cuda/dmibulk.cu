#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"

// Exchange + Dzyaloshinskii-Moriya interaction for bulk material.
// Energy:
//
// 	E  = D M . rot(M)
//
// Effective field:
//
// 	Hx = 2A/Bs nabla²Mx + 2D/Bs dzMy - 2D/Bs dyMz
// 	Hy = 2A/Bs nabla²My + 2D/Bs dxMz - 2D/Bs dzMx
// 	Hz = 2A/Bs nabla²Mz + 2D/Bs dyMx - 2D/Bs dxMy
//
// Boundary conditions:
//
// 	        2A dxMx = 0
// 	 D Mz + 2A dxMy = 0
// 	-D My + 2A dxMz = 0
//
// 	-D Mz + 2A dyMx = 0
// 	        2A dyMy = 0
// 	 D Mx + 2A dyMz = 0
//
// 	 D My + 2A dzMx = 0
// 	-D Mx + 2A dzMy = 0
// 	        2A dzMz = 0
//
extern "C" __global__ void
adddmibulk(float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
           float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
           float* __restrict__ aLUT2d, float* __restrict__ DLUT2d,
           uint8_t* __restrict__ regions,
           float cx, float cy, float cz, int Nx, int Ny, int Nz, uint8_t PBC) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= Nx || iy >= Ny || iz >= Nz) {
		return;
	}

	int I = idx(ix, iy, iz);                      // central cell index
	float3 h = make_float3(Hx[I], Hy[I], Hz[I]);  // add to H
	float3 m0 = make_float3(mx[I], my[I], mz[I]); // central m
	uint8_t r0 = regions[I];
	float A = aLUT2d[symidx(r0, r0)];
	float D = DLUT2d[symidx(r0, r0)]; 
	float D_2A = D/(2.0f*A);   
	int i_;                                       // neighbor index

	if(is0(m0)) {
		return;
	}

	// x derivatives (along length)
	{
		float3 m1 = make_float3(0.0f, 0.0f, 0.0f);     // left neighbor
		i_ = idx(lclampx(ix-1), iy, iz);               // load neighbor m if inside grid, keep 0 otherwise
		if (ix-1 >= 0 || PBCx) {
			m1 = make_float3(mx[i_], my[i_], mz[i_]);
		}
		if (is0(m1)) {                                 // neighbor missing
			m1.x = m0.x;
			m1.y = m0.y - (-cx * D_2A * m0.z);
			m1.z = m0.z + (-cx * D_2A * m0.y);
		}
		h   += (2.0f*A/(cx*cx)) * (m1 - m0);          // exchange
		h.y += (D/cx)*(-m1.z);                  // actually (2*D)/(2*cx) * delta m
		h.z -= (D/cx)*(-m1.y);
	}


	{
		float3 m2 = make_float3(0.0f, 0.0f, 0.0f);     // right neighbor
		i_ = idx(hclampx(ix+1), iy, iz);
		if (ix+1 < Nx || PBCx) {
			m2 = make_float3(mx[i_], my[i_], mz[i_]);
		}
		if (is0(m2)) {
			m2.x = m0.x;
			m2.y = m0.y - (+cx * D_2A * m0.z);
			m2.z = m0.z + (+cx * D_2A * m0.y);
		}
		h   += (2.0f*A/(cx*cx)) * (m2 - m0);
		h.y += (D/cx)*(m2.z);
		h.z -= (D/cx)*(m2.y);
	}

	// y derivatives (along height)
	{
		float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
		i_ = idx(ix, lclampy(iy-1), iz);
		if (iy-1 >= 0 || PBCy) {
			m1 = make_float3(mx[i_], my[i_], mz[i_]);
		}
		if (is0(m1)) {
			m1.x = m0.x + (-cy * D_2A * m0.z);
			m1.y = m0.y;
			m1.z = m0.z - (-cy * D_2A * m0.x);
		}
		h   += (2.0f*A/(cy*cy)) * (m1 - m0);
		h.x -= (D/cy)*(-m1.z);
		h.z += (D/cy)*(-m1.x);
	}

	{
		float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
		i_ = idx(ix, hclampy(iy+1), iz);
		if  (iy+1 < Ny || PBCy) {
			m2 = make_float3(mx[i_], my[i_], mz[i_]);
		}
		if (is0(m2)) {
			m2.x = m0.x + (+cy * D_2A * m0.z);
			m2.y = m0.y;
			m2.z = m0.z - (+cy * D_2A * m0.x);
		}
		h   += (2.0f*A/(cy*cy)) * (m2 - m0);
		h.x -= (D/cy)*(m2.z);
		h.z += (D/cy)*(m2.x);
	}

	// only take vertical derivative for 3D sim
	if (Nz != 1) {
		// bottom neighbor
		{
			float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
			i_ = idx(ix, iy, lclampz(iz-1));
			if (iz-1 >= 0 || PBCz) {
				m1 = make_float3(mx[i_], my[i_], mz[i_]);
			}
			if (is0(m1)) {
				m1.x = m0.x - (-cz * D_2A * m0.y);
				m1.y = m0.y + (-cz * D_2A * m0.x);
				m1.z = m0.z;
			}
			h   += (2.0f*A/(cz*cz)) * (m1 - m0);
			h.x += (D/cz)*(- m1.y);
			h.y -= (D/cz)*(- m1.x);
		}

		// top neighbor
		{
			float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
			i_ = idx(ix, iy, hclampz(iz+1));
			if (iz+1 < Nz || PBCz) {
				m2 = make_float3(mx[i_], my[i_], mz[i_]);
			}
			if (is0(m2)) {
				m2.x = m0.x - (+cz * D_2A * m0.y);
				m2.y = m0.y + (+cz * D_2A * m0.x);
				m2.z = m0.z;
			}
			h   += (2.0f*A/(cz*cz)) * (m2 - m0);
			h.x += (D/cz)*(m2.y );
			h.y -= (D/cz)*(m2.x );
		}
	}

	// write back, result is H + Hdmi + Hex
	Hx[I] = h.x;
	Hy[I] = h.y;
	Hz[I] = h.z;
}

// Note on boundary conditions.
//
// We need the derivative and laplacian of m in point A, but e.g. C lies out of the boundaries.
// We use the boundary condition in B (derivative of the magnetization) to extrapolate m to point C:
// 	m_C = m_A + (dm/dx)|_B * cellsize
//
// When point C is inside the boundary, we just use its actual value.
//
// Then we can take the central derivative in A:
// 	(dm/dx)|_A = (m_C - m_D) / (2*cellsize)
// And the laplacian:
// 	lapl(m)|_A = (m_C + m_D - 2*m_A) / (cellsize^2)
//
// All these operations should be second order as they involve only central derivatives.
//
//    ------------------------------------------------------------------ *
//   |                                                   |             C |
//   |                                                   |          **   |
//   |                                                   |        ***    |
//   |                                                   |     ***       |
//   |                                                   |   ***         |
//   |                                                   | ***           |
//   |                                                   B               |
//   |                                               *** |               |
//   |                                            ***    |               |
//   |                                         ****      |               |
//   |                                     ****          |               |
//   |                                  ****             |               |
//   |                              ** A                 |               |
//   |                         *****                     |               |
//   |                   ******                          |               |
//   |          *********                                |               |
//   |D ********                                         |               |
//   |                                                   |               |
//   +----------------+----------------+-----------------+---------------+
//  -1              -0.5               0               0.5               1
//                                 x
