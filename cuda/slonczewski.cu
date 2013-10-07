// Original implementation by Mykola Dvornik for mumax2
// Modified for mumax3 by Arne Vansteenkiste, 2013

#include "float3.h"

extern "C" __global__ void
addslonczewskitorque(float* __restrict__ sttx, float* __restrict__ stty, float* __restrict__ sttz,
                     float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                     float* __restrict__ msat,
                     float* __restrict__ jx, float* __restrict__ jy, float* __restrict__ jz,
                     float* __restrict__ px, float* __restrict__ py, float* __restrict__ pz,
                     float* __restrict__ alphaMsk,
                     float* __restrict__ t_flMsk,
                     float* __restrict__ polMsk,
                     float* __restrict__ lambdaMsk,
                     float* __restrict__ epsilonPrimeMsk,
                     float* __restrict__ regions,
                     float preX, float preY,
                     float meshSizeX, float meshSizeY, float meshSizeZ,
                     int N) {

    int I =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (I < N) {

        int8_t region = regions[I];
        float Ms = msat[region];

        float j_x = jx[I];
        float j_y = jy[I];
        float j_z = jz[I];

        float3 J = make_float3(j_x, j_y, j_z);
        float nJn = len(J);

        float free_layer_thickness =  t_flMsk[region];

        if (nJn == 0.0f || Ms == 0.0f || free_layer_thickness == 0.0f)
        {
            sttx[I] = 0.0f;
            stty[I] = 0.0f;
            sttz[I] = 0.0f;
            return;
        }


        free_layer_thickness = 1.0f / free_layer_thickness;
        Ms = 1.0f / Ms;

        preX *= Ms;
        preY *= Ms;

        float3 m = make_float3(mx[I], my[I], mz[I]);

        float p_x = px[I];
        float p_y = py[I];
        float p_z = pz[I];

        float3 p = make_float3(p_x, p_y, p_z);
        p = normalized(p);

        float3 pxm = cross(p, m); // plus
        float3 mxpxm = cross(m, pxm); // plus
        float  pdotm = dot(p, m);

        J = normalized(J);
        float Jdir = dot(make_float3(1.0f, 1.0f, 1.0f), J);
        float Jsign = Jdir / fabsf(Jdir);
        nJn *= Jsign;
        preX *= nJn;
        preY *= nJn;

        preX *= free_layer_thickness;
        preY *= free_layer_thickness;

        // take into account spatial profile of scattering control parameter
        float lambda = lambdaMsk[region];
        float lambda2 = lambda * lambda;
        float epsilon = lambda2 / ((lambda2 + 1.0f) + (lambda2 - 1.0f) * pdotm);
        preX *= epsilon;
        // preY ???

        float alpha = alphaMsk[region];
        float alphaFac = 1.0f / (1.0f + alpha * alpha);

        // take into account spatial profile of polarization efficiency
        float pol = polMsk[region];
        preX *= pol;

        // take into account spatial profile of the secondary spin transfer term
        float epsilonPrime = epsilonPrimeMsk[region];
        preY *= epsilonPrime;

        float mxpxmFac = preX - alpha * preY;
        float pxmFac = preY - alpha * preX;

        mxpxmFac *= alphaFac;
        pxmFac *= alphaFac;

        sttx[I] = mxpxmFac * mxpxm.x + pxmFac * pxm.x;
        stty[I] = mxpxmFac * mxpxm.y + pxmFac * pxm.y;
        sttz[I] = mxpxmFac * mxpxm.z + pxmFac * pxm.z;

    }
}

