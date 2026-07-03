#include <stdint.h>
#include "stencil.h"
#include "amul.h"

// Reduced Planck constant and elementary charge (SI units).
#define HBAR 1.054571817e-34f
#define QE   1.602176634e-19f

// Spin-orbit (spin-Hall) torque, delivered as a compensated effective field.
//
// The spin-Hall effect drives damping-like (DL) and field-like (FL) torques
//   H_DL = thetaSH * (hbar * Jc) / (2 e Msat t)
//   H_FL = thetaFL * (hbar * Jc) / (2 e Msat t)
// with charge current density Jc along +x and spin polarization sigma = +y.
//
// mumax applies B_eff through the LLG torque
//   tau/gamma0 = -1/(1+a^2) [ m x B + a m x (m x B) ].
// Writing the desired DL+FL torque as B = A*sigma + Bc*(m x sigma) and matching
// gives A = a*H_DL - H_FL and Bc = H_DL + a*H_FL (a = damping). With sigma = +y,
// m x sigma = (-mz, 0, mx).
//
// Convention: Jc in A/m^2, t in m, field in Tesla. Uses the 1/Msat convention
// (no explicit mu0), consistent with the rest of mumax. See sot.go.
extern "C" __global__ void
addsot(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
       float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
       float* __restrict__ Ms_, float Ms_mul,
       float* __restrict__ Jc_, float Jc_mul,
       float* __restrict__ alpha_, float alpha_mul,
       float thetaSH, float thetaFL, float thickness,
       int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    int i = idx(ix, iy, iz);

    float ms = amul(Ms_, Ms_mul, i);
    float jc = amul(Jc_, Jc_mul, i);
    if (ms == 0.0f || jc == 0.0f || thickness <= 0.0f) {
        return;
    }
    float alpha = amul(alpha_, alpha_mul, i);

    float pref = HBAR / (2.0f * QE * ms * thickness);
    float H_DL = thetaSH * jc * pref;
    float H_FL = thetaFL * jc * pref;

    // compensated field so the LLG yields the DL+FL torque
    float A     = alpha * H_DL - H_FL;
    float Bcomp = H_DL + alpha * H_FL;

    // sigma = +y  ->  m x sigma = (-mz, 0, mx)
    float mxi = mx[i];
    float mzi = mz[i];
    Bx[i] += Bcomp * (-mzi);
    By[i] += A;
    Bz[i] += Bcomp * mxi;
}
