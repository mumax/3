// Original implementation by Mykola Dvornik for mumax2
// Modified for mumax3 by Arne Vansteenkiste, 2013

#include "float3.h"

extern "C" __global__ void
addslonczewskitorque(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
                     float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                     float* __restrict__ jx, float* __restrict__ jy, float* __restrict__ jz,
                     float* __restrict__ pxLUT, float* __restrict__ pyLUT, float* __restrict__ pzLUT,
                     float* __restrict__ msatLUT,
                     float* __restrict__ alphaLUT,
                     float* __restrict__ t_flLUT,
                     float* __restrict__ polLUT,
                     float* __restrict__ lambdaLUT,
                     float* __restrict__ epsilonPrimeLUT,
                     float* __restrict__ regions,
                     float preX, float preY,
                     float meshSizeX, float meshSizeY, float meshSizeZ,
                     int N) {

    int I =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (I < N) {

        int8_t region = regions[I];

        float3 J   = make_float3(jx[I], jy[I], jz[I]);
        float  Ms  = msatLUT[region];
        float  nJn = len(J);
        float  free_layer_thickness = t_flLUT[region];

        if (nJn == 0.0f || Ms == 0.0f || free_layer_thickness == 0.0f)
        {
            tx[I] = 0.0f;
            ty[I] = 0.0f;
            tz[I] = 0.0f;
            return;
        }

        Ms = 1.0f / Ms;
        preX *= Ms;
        preY *= Ms;

        float3 m = make_float3(mx[I], my[I], mz[I]);
        float3 p = make_float3(pxLUT[region], pyLUT[region], pzLUT[region]);
        p = normalized(p);

        float3 pxm   = cross(p, m);   // plus
        float3 mxpxm = cross(m, pxm); // plus
        float  pdotm = dot(p, m);

        J = normalized(J);
        float Jdir = dot(make_float3(1.0f, 1.0f, 1.0f), J);
        float Jsign = Jdir / fabsf(Jdir);
        nJn *= Jsign;
        preX *= nJn;
        preY *= nJn;

        free_layer_thickness = 1.0f / free_layer_thickness;
        preX *= free_layer_thickness;
        preY *= free_layer_thickness;

        // take into account spatial profile of scattering control parameter
        float lambda = lambdaLUT[region];
        float lambda2 = lambda * lambda;
        float epsilon = lambda2 / ((lambda2 + 1.0f) + (lambda2 - 1.0f) * pdotm);
        preX *= epsilon;
        // preY ???

        // take into account spatial profile of polarization efficiency
        float pol = polLUT[region];
        preX *= pol;

        // take into account spatial profile of the secondary spin transfer term
        float epsilonPrime = epsilonPrimeLUT[region];
        preY *= epsilonPrime;

        float alpha = alphaLUT[region];
        float alphaFac = 1.0f / (1.0f + alpha * alpha);

        float mxpxmFac = preX - alpha * preY;
        float pxmFac = preY - alpha * preX;

        mxpxmFac *= alphaFac;
        pxmFac *= alphaFac;

        tx[I] += mxpxmFac * mxpxm.x + pxmFac * pxm.x;
        ty[I] += mxpxmFac * mxpxm.y + pxmFac * pxm.y;
        tz[I] += mxpxmFac * mxpxm.z + pxmFac * pxm.z;
    }
}

