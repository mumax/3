#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include <stdio.h>

// Add interlayer exchange field to B.
extern "C" __global__ void
addinterlayerexchange(float* __restrict__  Bx, float* __restrict__  By, float* __restrict__  Bz,
                      float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
                      float* __restrict__ J1LUT, float* __restrict__ J2LUT,  float* __restrict__ toplayer,  float* __restrict__ bottomlayer,
                      float* __restrict__ uxLUT, float* __restrict__ uyLUT, float* __restrict__ uzLUT,
		      float cx, float cy, float cz, int Nx, int Ny, int Nz, uint8_t* __restrict__ regions) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	int i = idx(ix, iy, iz);
	//int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (ix >= Nx || iy >= Ny || iz >= Nz) {
		return;
	}
	//printf("%d \n", i);
	
	//printf("%d \n", i_temp);
	uint8_t reg = regions[i];
	float toptmp = toplayer[reg];
	float bottomtmp = bottomlayer[reg];
	float  ux  = uxLUT[reg];
	float  uy  = uyLUT[reg];
	float  uz  = uzLUT[reg];
	float3 u   = normalized(make_float3(ux, uy, uz));
	float J_linear  = J1LUT[reg];
	float J_quadratic  = J2LUT[reg];

	int toppos= __double2int_rn(toptmp);
	int bottompos=__double2int_rn(bottomtmp);	
	//printf("%d \n", __double2int_rn(u.z));
	if (__double2int_rn(u.x) == 0 && __double2int_rn(u.y) == 0 && __double2int_rn(u.z) == 1) {
		//if (ix > Nx || iy > Ny || iz > Nz) return;
		//printf("%d \t %d \t %d \t %d \n", i, ix, iy ,iz);
		//
		float  cellsize_z = cz;
	
		int start_pos_bot = Nx*Ny*(bottompos);
		int end_pos_bot = Nx*Ny*(bottompos) + Nx*Ny-1;

		int start_pos_top = Nx*Ny*(toppos);
		int end_pos_top = Nx*Ny*(toppos) + Nx*Ny-1; 
		if ( i >= start_pos_top && i <= end_pos_top) {
			int i_below = i - (toppos-bottompos)*Nx*Ny;
			float3 m   = {mx[i], my[i], mz[i]};
			float3 m_prime = { mx[i_below], my[i_below], mz[i_below]};
			if ( is0(m) || is0(m_prime)) {
				return;
			}
			float3 Biec_top  = (J_linear * m_prime + 2.0f * J_quadratic * m_prime * dot(m, m_prime)); // calc. IEC in toplayer
		
			Bx[i] += Biec_top.x/cellsize_z;
			By[i] += Biec_top.y/cellsize_z;
			Bz[i] += Biec_top.z/cellsize_z;	
			//printf("%e \n", cellsize_z);
			//printf("%e \t %e \t %e\n", Biec_top.x,Biec_top.y,Biec_top.z);
		}

		if ( i >= start_pos_bot && i <= end_pos_bot) {
			int i_above = i + (toppos-bottompos)*Nx*Ny;
			float3 m   = {mx[i_above], my[i_above], mz[i_above]};
			float3 m_prime = { mx[i], my[i], mz[i]};		
			if ( is0(m) || is0(m_prime)) {
				return;
			}
			float3 Biec_bottom  = (J_linear * m + 2.0f * J_quadratic * m * dot(m_prime, m)); // calc. IEC in bottomlayer		
			//float3 Biec_bottom  = (J_linear * m + 2.0f * J_quadratic * m * dot(m, m_prime)); // calc. IEC in bottomlayer		

			Bx[i] += Biec_bottom.x/cellsize_z;
			By[i] += Biec_bottom.y/cellsize_z;
			Bz[i] += Biec_bottom.z/cellsize_z;
			//printf("%e \t %e \t %e\n", Bx[i],By[i],Bz[i]);
		}
	}
	if (__double2int_rn(u.x) == 0 && __double2int_rn(u.y) == 1 && __double2int_rn(u.z) == 0) {
		//int i_y =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
		//if (ix > Nx || iy > Ny || iz > Nz) return;
		//printf("%d \t %d \t %d \t %d \n", i, ix, iy ,iz);
		float  cellsize_y = cy;
	
		int start_pos_bot = (bottompos+1)*Nx-Nx;
		int end_pos_bot = start_pos_bot + Nx-1;

		int start_pos_top = (toppos+1)*Nx-Nx;
		int end_pos_top = start_pos_top + Nx-1;
		//printf("%d \t %d \n", start_pos_bot, end_pos_bot);
		//printf("%d \t %d \n",start_pos_top, end_pos_top); 
		if ( i >= start_pos_top +iz*Nx*Ny && i <= end_pos_top +iz*Nx*Ny) {
		//if ( i >= start_pos_top && i <= end_pos_top) {
			int i_below = i - (toppos-bottompos)*Nx;
			float3 m   = {mx[i], my[i], mz[i]};
			float3 m_prime = { mx[i_below], my[i_below], mz[i_below]};
			if ( is0(m) || is0(m_prime)) {
				return;
			}
			float3 Biec_top  = (J_linear * m_prime + 2.0f * J_quadratic * m_prime * dot(m, m_prime)); // calc. IEC in toplayer
			//printf("%e \n", cellsize_y);
			//printf("%e \t %e \t %e\n", Biec_top.x,Biec_top.y,Biec_top.z);
			Bx[i] += Biec_top.x/cellsize_y;
			By[i] += Biec_top.y/cellsize_y;
			Bz[i] += Biec_top.z/cellsize_y;
		}

		if ( i >= start_pos_bot+iz*Nx*Ny && i <= end_pos_bot+iz*Nx*Ny) {
		//if ( i >= start_pos_bot && i <= end_pos_bot)	{
			int i_above = i + (toppos-bottompos)*Nx;
			//printf("%d \t %d \n", i, i_above);
			float3 m   = {mx[i_above], my[i_above], mz[i_above]};
			float3 m_prime = { mx[i], my[i], mz[i]};		
			if ( is0(m) || is0(m_prime)) {
				return;
			}
			float3 Biec_bottom  = (J_linear * m + 2.0f * J_quadratic * m * dot(m_prime, m)); // calc. IEC in bottomlayer		
			//float3 Biec_bottom  = (J_linear * m + 2.0f * J_quadratic * m * dot(m, m_prime)); // calc. IEC in bottomlayer		
			Bx[i] += Biec_bottom.x/cellsize_y;
			By[i] += Biec_bottom.y/cellsize_y;
			Bz[i] += Biec_bottom.z/cellsize_y;
			//printf("%e \t %e \t %e\n", Bx[i],By[i],Bz[i]);
		}
	}
	if (__double2int_rn(u.x) == 1 && __double2int_rn(u.y) == 0 && __double2int_rn(u.z) == 0) {
		float  cellsize_x = cx;
	
		if ( i == bottompos + iy*Nx + iz*Nx*Ny   && i < Nx*Ny*Nz) {
			int i_above = i + (toppos-bottompos);
			//printf("%d \t %d \n", i, i_above);
			float3 m_prime   = {mx[i], my[i], mz[i]};
			float3 m = { mx[i_above], my[i_above], mz[i_above]};
			if ( is0(m) || is0(m_prime)) {
				return;
			}
			float3 Biec_bottom  = (J_linear * m + 2.0f * J_quadratic * m * dot(m_prime, m)); // calc. IEC in bottomlayer
		
			Bx[i] += Biec_bottom.x/cellsize_x;
			By[i] += Biec_bottom.y/cellsize_x;
			Bz[i] += Biec_bottom.z/cellsize_x;
		}

		if ( i == toppos + iy*Nx + iz*Nx*Ny   && i < Nx*Ny*Nz) {
			int i_below = i - (toppos-bottompos);
			float3 m   = {mx[i], my[i], mz[i]};
			float3 m_prime = { mx[i_below], my[i_below], mz[i_below]};		
			if ( is0(m) || is0(m_prime)) {
				return;
			}
			float3 Biec_top  = (J_linear * m_prime + 2.0f * J_quadratic * m_prime * dot(m, m_prime)); // calc. IEC in toplayer		
			//float3 Biec_bottom  = (J_linear * m + 2.0f * J_quadratic * m * dot(m, m_prime)); // calc. IEC in bottomlayer		
	
			Bx[i] += Biec_top.x/cellsize_x;
			By[i] += Biec_top.y/cellsize_x;
			Bz[i] += Biec_top.z/cellsize_x;
		}
	}
}

