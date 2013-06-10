#include "float3.h"
#include "stencil.h"
#include "mask.h"

// (ux, uy, uz) is 0.5 * U_spintorque / cellsize(x, y, z)
extern "C" __global__ void
addzhanglitorque(float* __restrict__    tx, float* __restrict__    ty, float* __restrict__    tz,
                 float* __restrict__    mx, float* __restrict__    my, float* __restrict__    mz,
                 float                  ux, float                  uy, float                  uz,
                 float* __restrict__ jmapx, float* __restrict__ jmapy, float* __restrict__ jmapz,
                 float alpha, float xi,
                 int N0, int N1, int N2) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= N1 || k >= N2) {
        return;
    }

    for(int i=0; i<N0; i++) {
        int I = idx(i, j, k);

        float3 hspin = make_float3(0, 0, 0); // (u·∇)m
        if (ux != 0.) {
            //ux *= loadmask(jmapx, I);
            hspin += ux * make_float3(delta(mx, 1,0,0), delta(my, 1,0,0), delta(mz, 1,0,0));
        }
        if (uy != 0.) {
            //uy *= loadmask(jmapy, I);
            hspin += uy * make_float3(delta(mx, 0,1,0), delta(my, 0,1,0), delta(mz, 0,1,0));
        }
        if (uz != 0.) {
            //uz *= loadmask(jmapz, I);
            hspin += uz * make_float3(delta(mx, 0,0,1), delta(my, 0,0,1), delta(mz, 0,0,1));
        }

        float3 m      = make_float3(mx[I], my[I], mz[I]);
        float3 torque = (-1./(1. + alpha*alpha)) * (
                            (1+xi*alpha) * cross(m, cross(m, hspin))
                            +(  xi-alpha) * cross(m, hspin)           );

        // write back, adding to torque
        tx[I] += torque.x;
        ty[I] += torque.y;
        tz[I] += torque.z;
    }
}

