#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// See exchange_fourth_order.go for more details.
// The fourth-order derivative of a function f(x) discretised with cell size dx is given approximately by
// (1/dx^4) * (f(x + 2dx) - 4f(x + dx) + 6f(x) - 4f(x - dx) + f(x - 2dx))
extern "C" __global__ void
addexchangefourthorder(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            float* __restrict__ Ms_, float Ms_mul,
            float* __restrict__ aSecondOrderLUT2d, float* __restrict__ aFourthOrderLUT2d,
            uint8_t* __restrict__ regions,
            float cx, float cy, float cz, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    float wxSecondOrder = 2 / (cx * cx);
    float wySecondOrder = 2 / (cy * cy);
    float wzSecondOrder = 2 / (cz * cz);

    float wxFourthOrder = 2 / (cx * cx * cx * cx);
    float wyFourthOrder = 2 / (cy * cy * cy * cy);
    float wzFourthOrder = 2 / (cz * cz * cz * cz);

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
    float3 B  = make_float3(0.0,0.0,0.0);

    int i_;    // neighbor index
    float3 m_; // neighbor mag
    float a__; // inter-cell exchange stiffness

    // neighbor 2x to left
    i_  = idx(lclampx(ix-2), iy, iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    a__ = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B += wxFourthOrder * a__ * m_;

    // neighbor directly to left
    i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    a__ = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B -= 4 * wxFourthOrder * a__ * m_;
    a__ = aSecondOrderLUT2d[symidx(r0, regions[i_])];
    B += wxSecondOrder * a__ * m_;  // second-order exchange

    // the cell itself
    i_  = idx(ix, iy, iz);             // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    a__ = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B += 6 * wxFourthOrder * a__ * m_;
    a__ = aSecondOrderLUT2d[symidx(r0, regions[i_])];
    B -= 2 * wxSecondOrder * a__ * m_;  // second-order exchange

    // neighbor directly to right
    i_  = idx(hclampx(ix+1), iy, iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    a__ = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B -= 4 * wxFourthOrder * a__ * m_;
    a__ = aSecondOrderLUT2d[symidx(r0, regions[i_])];
    B += wxSecondOrder * a__ * m_;  // second-order exchange

    // neighbor 2x to right
    i_  = idx(hclampx(ix+2), iy, iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    a__ = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B += wxFourthOrder * a__ * m_;

    // neighbor 2x back
    i_  = idx(ix, lclampy(iy-2), iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    a__ = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B += wyFourthOrder * a__ * m_;

    // neighbor directly behind
    i_  = idx(ix, lclampy(iy-1), iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    a__ = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B -= 4 * wyFourthOrder * a__ * m_;
    a__ = aSecondOrderLUT2d[symidx(r0, regions[i_])];
    B += wySecondOrder * a__ * m_;  // second-order exchange

    // the cell itself
    i_  = idx(ix, iy, iz);             // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    a__ = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B += 6 * wyFourthOrder * a__ * m_;
    a__ = aSecondOrderLUT2d[symidx(r0, regions[i_])];
    B -= 2 * wySecondOrder * a__ * m_;  // second-order exchange

    // neighbor directly in front
    i_  = idx(ix, lclampy(iy+1), iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    a__ = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B -= 4 * wyFourthOrder * a__ * m_;
    a__ = aSecondOrderLUT2d[symidx(r0, regions[i_])];
    B += wySecondOrder * a__ * m_;  // second-order exchange

    // neighbor 2x in front
    i_  = idx(ix, hclampy(iy+2), iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    a__ = aFourthOrderLUT2d[symidx(r0, regions[i_])];
    B += wyFourthOrder * a__ * m_;

    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // neighbor 2x below
        i_  = idx(ix, iy, lclampz(iz-2));           // clamps or wraps index according to PBC
        m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
        m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
        a__ = aFourthOrderLUT2d[symidx(r0, regions[i_])];
        B += wzFourthOrder * a__ * m_;

        // neighbor directly below
        i_  = idx(ix, iy, lclampx(iz-1));           // clamps or wraps index according to PBC
        m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
        m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
        a__ = aFourthOrderLUT2d[symidx(r0, regions[i_])];
        B -= 4 * wzFourthOrder * a__ * m_;
        a__ = aSecondOrderLUT2d[symidx(r0, regions[i_])];
        B += wzSecondOrder * a__ * m_;  // second-order exchange

        // the cell itself
        i_  = idx(ix, iy, iz);                      // clamps or wraps index according to PBC
        m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
        m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
        a__ = aFourthOrderLUT2d[symidx(r0, regions[i_])];
        B += 6 * wzFourthOrder * a__ * m_;
        a__ = aSecondOrderLUT2d[symidx(r0, regions[i_])];
        B -= 2 * wzSecondOrder * a__ * m_;  // second-order exchange

        // neighbor directly in above
        i_  = idx(ix, iy, hclampz(iz+1));           // clamps or wraps index according to PBC
        m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
        m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
        a__ = aFourthOrderLUT2d[symidx(r0, regions[i_])];
        B -= 4 * wzFourthOrder * a__ * m_;
        a__ = aSecondOrderLUT2d[symidx(r0, regions[i_])];
        B += wzSecondOrder * a__ * m_;  // second-order exchange

        // neighbor 2x in above
        i_  = idx(ix, iy, hclampz(iz+2));           // clamps or wraps index according to PBC
        m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
        m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
        a__ = aFourthOrderLUT2d[symidx(r0, regions[i_])];
        B += wzFourthOrder * a__ * m_;
    }

    float invMs = inv_Msat(Ms_, Ms_mul, I);
    // we subtract the field here as the NNN exchange field has the opposite sign to NN
    Bx[I] -= B.x*invMs;
    By[I] -= B.y*invMs;
    Bz[I] -= B.z*invMs;
}

