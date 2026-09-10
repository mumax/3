// Compatibility prelude for cuda/metal/cmd/cuda2metal.
// Source of truth for kernel bodies: the audited CUDA dialect in cuda/*.cu.
//
// ABI: every original CUDA argument occupies the same numbered Metal buffer
// binding. Pointer arguments use setBuffer; scalar arguments use setBytes. A
// final uint32 binding records pointer presence because nil Metal buffers are
// represented by a safe dummy resource at dispatch.

#include <metal_stdlib>
using namespace metal;

#define REDUCE_BLOCKSIZE 512
#define PI     3.1415926535897932384626433
#define MU0    (4*PI*1e-7)
#define QE     1.60217646E-19
#define MUB    9.2740091523E-24
#define GAMMA0 1.7595e11
#define HBAR   1.05457173E-34

#define index(ix,iy,iz,Nx,Ny,Nz) ((((iz)*(Ny) + (iy)) * (Nx)) + (ix))
#define idx(ix,iy,iz) (index((ix),(iy),(iz),(Nx),(Ny),(Nz)))
#define MOD(n,M) (((n) % (M) + (M)) % (M))
#define PBCx (PBC & 1)
#define PBCy (PBC & 2)
#define PBCz (PBC & 4)
#define hclampx(ix) (PBCx ? MOD((ix), Nx) : min((ix), Nx-1))
#define lclampx(ix) (PBCx ? MOD((ix), Nx) : max((ix), 0))
#define hclampy(iy) (PBCy ? MOD((iy), Ny) : min((iy), Ny-1))
#define lclampy(iy) (PBCy ? MOD((iy), Ny) : max((iy), 0))
#define hclampz(iz) (PBCz ? MOD((iz), Nz) : min((iz), Nz-1))
#define lclampz(iz) (PBCz ? MOD((iz), Nz) : max((iz), 0))
#define symidx(i,j) ((j)<=(i) ? ((((i)*((i)+1))/2)+(j)) : ((((j)*((j)+1))/2)+(i)))
#define is0(m) (dot((m), (m)) == 0.0f)

inline float3 make_float3(float x, float y, float z) {
    return float3(x, y, z);
}

inline float len(float3 value) {
    return length(value);
}

inline float3 normalized(float3 value) {
    float valueLength = length(value);
    return valueLength != 0.0f ? value / valueLength : float3(0.0f);
}

inline float pow2(float value) {
    return value * value;
}

inline float pow3(float value) {
    return value * value * value;
}

inline float pow4(float value) {
    float square = value * value;
    return square * square;
}

// One Hopf-index kernel uses CUDA's double-complex helper even though all
// source arrays and outputs are float32. Apple GPUs do not expose fp64;
// preserve the complex algebra in the engine's native float32 precision.
inline float2 mumaxMakeComplex(float realPart, float imaginaryPart) {
    return float2(realPart, imaginaryPart);
}

inline float2 mumaxComplexAdd(float2 a, float2 b) {
    return a + b;
}

inline float2 mumaxComplexSub(float2 a, float2 b) {
    return a - b;
}

inline float2 mumaxComplexMul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

inline float mumaxComplexImag(float2 value) {
    return value.y;
}

inline bool mumaxPointerPresent(uint mask, uint argumentIndex) {
    return (mask & (1u << argumentIndex)) != 0u;
}

inline float amul(device const float* array, bool present,
                  float multiplier, int indexValue) {
    return present ? multiplier * array[indexValue] : multiplier;
}

inline float3 vmul(device const float* x, device const float* y, device const float* z,
                   bool xPresent, bool yPresent, bool zPresent,
                   float mx, float my, float mz, int indexValue) {
    return make_float3(amul(x, xPresent, mx, indexValue),
                       amul(y, yPresent, my, indexValue),
                       amul(z, zPresent, mz, indexValue));
}

inline float inv_Msat(device const float* ms, bool present,
                      float multiplier, int indexValue) {
    float value = amul(ms, present, multiplier, indexValue);
    return value == 0.0f ? 0.0f : 1.0f / value;
}

inline float sum(float a, float b) {
    return a + b;
}

// Float atomics use a uint compare/exchange loop, avoiding a hard requirement
// on Metal 3's atomic_float extension.
inline void mumaxAtomicAdd(device float* address, float value) {
    device atomic_uint* target = reinterpret_cast<device atomic_uint*>(address);
    uint expected = atomic_load_explicit(target, memory_order_relaxed);
    while (true) {
        uint desired = as_type<uint>(as_type<float>(expected) + value);
        if (atomic_compare_exchange_weak_explicit(
                target, &expected, desired,
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
}

inline void atomicFmaxabs(device float* address, float value) {
    device atomic_uint* target = reinterpret_cast<device atomic_uint*>(address);
    uint desired = as_type<uint>(fabs(value));
    uint expected = atomic_load_explicit(target, memory_order_relaxed);
    // CUDA's source performs signed atomicMax over the non-negative float bit
    // pattern. Retaining that comparison also preserves its NaN/Inf behavior.
    while (as_type<int>(expected) < as_type<int>(desired)) {
        if (atomic_compare_exchange_weak_explicit(
                target, &expected, desired,
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
}

#define atomicAdd mumaxAtomicAdd

// CUDA's reduction uses implicit warp lockstep for its last 32 lanes. Metal
// makes no equivalent guarantee, so every tree level has an explicit barrier.
// Current MuMax3 callers use a power-of-two group no larger than 512.
#define reduce(load, op, atomicOp)                                           \
    threadgroup float sdata[REDUCE_BLOCKSIZE];                              \
    uint tid = threadIdx.x;                                                 \
    int i = int(blockIdx.x * blockDim.x + threadIdx.x);                     \
    float mine = initVal;                                                   \
    int stride = int(gridDim.x * blockDim.x);                               \
    while (i < n) {                                                         \
        mine = op(mine, load(i));                                           \
        i += stride;                                                        \
    }                                                                       \
    sdata[tid] = mine;                                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                        \
    for (uint s = blockDim.x >> 1; s > 0; s >>= 1) {                        \
        if (tid < s) {                                                      \
            sdata[tid] = op(sdata[tid], sdata[tid + s]);                    \
        }                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
    }                                                                       \
    if (tid == 0) {                                                         \
        atomicOp(dst, sdata[0]);                                            \
    }

// -----------------------------------------------------------------------------
// Source: cuda/cellindices.cu
// -----------------------------------------------------------------------------
#line 1 "cellindices.cu"
// Set dst(x|y|z) to the x-, y- and z-index of the corresponding cell
// Adapted from Jake Love's kernel in PR#289
kernel void cellindices(
    device float* dstx [[buffer(0)]],
    device float* dsty [[buffer(1)]],
    device float* dstz [[buffer(2)]],
    constant float& nx [[buffer(3)]],
    constant float& ny [[buffer(4)]],
    constant float& nz [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant uint& mumaxPointerMask [[buffer(7)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {
        float idx_i = fmod(i, nx);
        float idx_j = floor(fmod(i / nx, ny));
        float idx_k = floor(i / (nx*ny));

        dstx[i] = idx_i;
        dsty[i] = idx_j;
        dstz[i] = idx_k;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/copypadmul2.cu
// -----------------------------------------------------------------------------
#line 1 "copypadmul2.cu"

// Copy src (size S, smaller) into dst (size D, larger),
// and multiply by Bsat * vol
kernel void copypadmul2(
    device float* dst [[buffer(0)]],
    constant int& Dx [[buffer(1)]],
    constant int& Dy [[buffer(2)]],
    constant int& Dz [[buffer(3)]],
    device float* src [[buffer(4)]],
    constant int& Sx [[buffer(5)]],
    constant int& Sy [[buffer(6)]],
    constant int& Sz [[buffer(7)]],
    device float* Ms_ [[buffer(8)]],
    constant float& Ms_mul [[buffer(9)]],
    device float* vol [[buffer(10)]],
    constant uint& mumaxPointerMask [[buffer(11)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix<Sx && iy<Sy && iz<Sz) {
        int sI = index(ix, iy, iz, Sx, Sy, Sz);  // source index
        float Bsat = MU0 * amul(Ms_, mumaxPointerPresent(mumaxPointerMask, 8u), Ms_mul, sI);
        float v = amul(vol, mumaxPointerPresent(mumaxPointerMask, 10u), 1.0f, sI);
        dst[index(ix, iy, iz, Dx, Dy, Dz)] = Bsat * v * src[sI];
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/copyunpad.cu
// -----------------------------------------------------------------------------
#line 1 "copyunpad.cu"

// Copy src (size S, larger) to dst (size D, smaller)
kernel void copyunpad(
    device float* dst [[buffer(0)]],
    constant int& Dx [[buffer(1)]],
    constant int& Dy [[buffer(2)]],
    constant int& Dz [[buffer(3)]],
    device float* src [[buffer(4)]],
    constant int& Sx [[buffer(5)]],
    constant int& Sy [[buffer(6)]],
    constant int& Sz [[buffer(7)]],
    constant uint& mumaxPointerMask [[buffer(8)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix<Dx && iy<Dy && iz<Dz) {
        dst[index(ix, iy, iz, Dx, Dy, Dz)] = src[index(ix, iy, iz, Sx, Sy, Sz)];
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/crop.cu
// -----------------------------------------------------------------------------
#line 1 "crop.cu"

// See crop.go
kernel void crop(
    device float* dst [[buffer(0)]],
    constant int& Dx [[buffer(1)]],
    constant int& Dy [[buffer(2)]],
    constant int& Dz [[buffer(3)]],
    device float* src [[buffer(4)]],
    constant int& Sx [[buffer(5)]],
    constant int& Sy [[buffer(6)]],
    constant int& Sz [[buffer(7)]],
    constant int& Offx [[buffer(8)]],
    constant int& Offy [[buffer(9)]],
    constant int& Offz [[buffer(10)]],
    constant uint& mumaxPointerMask [[buffer(11)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix<Dx && iy<Dy && iz<Dz) {
        dst[index(ix, iy, iz, Dx, Dy, Dz)] = src[index(ix+Offx, iy+Offy, iz+Offz, Sx, Sy, Sz)];
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/crossproduct.cu
// -----------------------------------------------------------------------------
#line 1 "crossproduct.cu"

kernel void crossproduct(
    device float* dstx [[buffer(0)]],
    device float* dsty [[buffer(1)]],
    device float* dstz [[buffer(2)]],
    device float* ax [[buffer(3)]],
    device float* ay [[buffer(4)]],
    device float* az [[buffer(5)]],
    device float* bx [[buffer(6)]],
    device float* by [[buffer(7)]],
    device float* bz [[buffer(8)]],
    constant int& N [[buffer(9)]],
    constant uint& mumaxPointerMask [[buffer(10)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        float3 A = {ax[i], ay[i], az[i]};
        float3 B = {bx[i], by[i], bz[i]};
        float3 AxB = cross(A, B);
        dstx[i] = AxB.x;
        dsty[i] = AxB.y;
        dstz[i] = AxB.z;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/cubicanisotropy2.cu
// -----------------------------------------------------------------------------
#line 1 "cubicanisotropy2.cu"

// add cubic anisotropy field to B.
// B:      effective field in T
// m:      reduced magnetization (unit length)
// Ms:     saturation magnetization in A/m.
// K1:     Kc1 in J/m3
// K2:     Kc2 in T/m3
// C1, C2: anisotropy axes
//
// based on http://www.southampton.ac.uk/~fangohr/software/oxs_cubic8.html
kernel void addcubicanisotropy2(
    device float* Bx [[buffer(0)]],
    device float* By [[buffer(1)]],
    device float* Bz [[buffer(2)]],
    device float* mx [[buffer(3)]],
    device float* my [[buffer(4)]],
    device float* mz [[buffer(5)]],
    device float* Ms_ [[buffer(6)]],
    constant float& Ms_mul [[buffer(7)]],
    device float* k1_ [[buffer(8)]],
    constant float& k1_mul [[buffer(9)]],
    device float* k2_ [[buffer(10)]],
    constant float& k2_mul [[buffer(11)]],
    device float* k3_ [[buffer(12)]],
    constant float& k3_mul [[buffer(13)]],
    device float* c1x_ [[buffer(14)]],
    constant float& c1x_mul [[buffer(15)]],
    device float* c1y_ [[buffer(16)]],
    constant float& c1y_mul [[buffer(17)]],
    device float* c1z_ [[buffer(18)]],
    constant float& c1z_mul [[buffer(19)]],
    device float* c2x_ [[buffer(20)]],
    constant float& c2x_mul [[buffer(21)]],
    device float* c2y_ [[buffer(22)]],
    constant float& c2y_mul [[buffer(23)]],
    device float* c2z_ [[buffer(24)]],
    constant float& c2z_mul [[buffer(25)]],
    constant int& N [[buffer(26)]],
    constant uint& mumaxPointerMask [[buffer(27)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float invMs = inv_Msat(Ms_, mumaxPointerPresent(mumaxPointerMask, 6u), Ms_mul, i);
        float  k1 = amul(k1_, mumaxPointerPresent(mumaxPointerMask, 8u), k1_mul, i) * invMs;
        float  k2 = amul(k2_, mumaxPointerPresent(mumaxPointerMask, 10u), k2_mul, i) * invMs;
        float  k3 = amul(k3_, mumaxPointerPresent(mumaxPointerMask, 12u), k3_mul, i) * invMs;
        float3 u1 = normalized(vmul(c1x_, c1y_, c1z_, mumaxPointerPresent(mumaxPointerMask, 14u), mumaxPointerPresent(mumaxPointerMask, 16u), mumaxPointerPresent(mumaxPointerMask, 18u), c1x_mul, c1y_mul, c1z_mul, i));
        float3 u2 = normalized(vmul(c2x_, c2y_, c2z_, mumaxPointerPresent(mumaxPointerMask, 20u), mumaxPointerPresent(mumaxPointerMask, 22u), mumaxPointerPresent(mumaxPointerMask, 24u), c2x_mul, c2y_mul, c2z_mul, i));
        float3 u3 = cross(u1, u2); // 3rd axis perpendicular to u1,u2
        float3 m  = make_float3(mx[i], my[i], mz[i]);

        float u1m = dot(u1, m);
        float u2m = dot(u2, m);
        float u3m = dot(u3, m);

        float3 B = -2.0f*k1*((pow2(u2m) + pow2(u3m)) * (    (u1m) * u1) +
                             (pow2(u1m) + pow2(u3m)) * (    (u2m) * u2) +
                             (pow2(u1m) + pow2(u2m)) * (    (u3m) * u3))-
                   2.0f*k2*((pow2(u2m) * pow2(u3m)) * (    (u1m) * u1) +
                            (pow2(u1m) * pow2(u3m)) * (    (u2m) * u2) +
                            (pow2(u1m) * pow2(u2m)) * (    (u3m) * u3))-
                   4.0f*k3*((pow4(u2m) + pow4(u3m)) * (pow3(u1m) * u1) +
                            (pow4(u1m) + pow4(u3m)) * (pow3(u2m) * u2) +
                            (pow4(u1m) + pow4(u2m)) * (pow3(u3m) * u3));
        Bx[i] += B.x;
        By[i] += B.y;
        Bz[i] += B.z;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/div.cu
// -----------------------------------------------------------------------------
#line 1 "div.cu"
// dst[i] = a[i] / b[i]
kernel void pointwise_div(
    device float* dst [[buffer(0)]],
    device float* a [[buffer(1)]],
    device float* b [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant uint& mumaxPointerMask [[buffer(4)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        if (b[i] != 0.0f) {
            dst[i] = a[i] / b[i];
        } else {
            dst[i] = 0.0f;
        }
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/dmi.cu
// -----------------------------------------------------------------------------
#line 1 "dmi.cu"

// Exchange + Dzyaloshinskii-Moriya interaction according to
// Bagdanov and Röβler, PRL 87, 3, 2001. eq.8 (out-of-plane symmetry breaking).
// Taking into account proper boundary conditions.
// m: normalized magnetization
// H: effective field in Tesla
// D: dmi strength, in Tesla*m
// A: Aex
kernel void adddmi(
    device float* Hx [[buffer(0)]],
    device float* Hy [[buffer(1)]],
    device float* Hz [[buffer(2)]],
    device float* mx [[buffer(3)]],
    device float* my [[buffer(4)]],
    device float* mz [[buffer(5)]],
    device float* Ms_ [[buffer(6)]],
    constant float& Ms_mul [[buffer(7)]],
    device float* aLUT2d [[buffer(8)]],
    device float* dLUT2d [[buffer(9)]],
    device uchar* regions [[buffer(10)]],
    constant float& cx [[buffer(11)]],
    constant float& cy [[buffer(12)]],
    constant float& cz [[buffer(13)]],
    constant int& Nx [[buffer(14)]],
    constant int& Ny [[buffer(15)]],
    constant int& Nz [[buffer(16)]],
    constant uchar& PBC [[buffer(17)]],
    constant uchar& OpenBC [[buffer(18)]],
    constant uint& mumaxPointerMask [[buffer(19)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);                      // central cell index
    float3 h = make_float3(0.0,0.0,0.0);          // add to H
    float3 m0 = make_float3(mx[I], my[I], mz[I]); // central m
    uchar r0 = regions[I];
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
        int r1 = is0(m1)? r0 : regions[i_];                // don't use inter region params if m1=0
        float A1 = aLUT2d[symidx(r0, r1)];                 // inter-region Aex
        float D1 = dLUT2d[symidx(r0, r1)];                 // inter-region Dex
        if (!is0(m1) || !OpenBC){                          // do nothing at an open boundary
            if (is0(m1)) {                                 // neighbor missing
                m1.x = m0.x - (-cx * (0.5f*D1/A1) * m0.z); // extrapolate missing m from Neumann BC's
                m1.y = m0.y;
                m1.z = m0.z + (-cx * (0.5f*D1/A1) * m0.x);
            }
            h   += (2.0f*A1/(cx*cx)) * (m1 - m0);          // exchange
            h.x += (D1/cx)*(- m1.z);
            h.z -= (D1/cx)*(- m1.x);
        }
    }

    {
        float3 m2 = make_float3(0.0f, 0.0f, 0.0f);     // right neighbor
        i_ = idx(hclampx(ix+1), iy, iz);
        if (ix+1 < Nx || PBCx) {
            m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        int r2 = is0(m2)? r0 : regions[i_];
        float A2 = aLUT2d[symidx(r0, r2)];
        float D2 = dLUT2d[symidx(r0, r2)];
        if (!is0(m2) || !OpenBC){
            if (is0(m2)) {
                m2.x = m0.x - (cx * (0.5f*D2/A2) * m0.z);
                m2.y = m0.y;
                m2.z = m0.z + (cx * (0.5f*D2/A2) * m0.x);
            }
            h   += (2.0f*A2/(cx*cx)) * (m2 - m0);
            h.x += (D2/cx)*(m2.z);
            h.z -= (D2/cx)*(m2.x);
        }
    }

    // y derivatives (along height)
    {
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, lclampy(iy-1), iz);
        if (iy-1 >= 0 || PBCy) {
            m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        int r1 = is0(m1)? r0 : regions[i_];
        float A1 = aLUT2d[symidx(r0, r1)];
        float D1 = dLUT2d[symidx(r0, r1)];
        if (!is0(m1) || !OpenBC){
            if (is0(m1)) {
                m1.x = m0.x;
                m1.y = m0.y - (-cy * (0.5f*D1/A1) * m0.z);
                m1.z = m0.z + (-cy * (0.5f*D1/A1) * m0.y);
            }
            h   += (2.0f*A1/(cy*cy)) * (m1 - m0);
            h.y += (D1/cy)*(- m1.z);
            h.z -= (D1/cy)*(- m1.y);
        }
    }

    {
        float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, hclampy(iy+1), iz);
        if  (iy+1 < Ny || PBCy) {
            m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        int r2 = is0(m2)? r0 : regions[i_];
        float A2 = aLUT2d[symidx(r0, r2)];
        float D2 = dLUT2d[symidx(r0, r2)];
        if (!is0(m2) || !OpenBC){
            if (is0(m2)) {
                m2.x = m0.x;
                m2.y = m0.y - (cy * (0.5f*D2/A2) * m0.z);
                m2.z = m0.z + (cy * (0.5f*D2/A2) * m0.y);
            }
            h   += (2.0f*A2/(cy*cy)) * (m2 - m0);
            h.y += (D2/cy)*(m2.z);
            h.z -= (D2/cy)*(m2.y);
        }
    }

    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // bottom neighbor
        {
            i_  = idx(ix, iy, lclampz(iz-1));
            float3 m1  = make_float3(mx[i_], my[i_], mz[i_]);
            m1  = ( is0(m1)? m0: m1 );                         // Neumann BC
            float A1 = aLUT2d[symidx(r0, regions[i_])];
            h += (2.0f*A1/(cz*cz)) * (m1 - m0);                // Exchange only
        }

        // top neighbor
        {
            i_  = idx(ix, iy, hclampz(iz+1));
            float3 m2  = make_float3(mx[i_], my[i_], mz[i_]);
            m2  = ( is0(m2)? m0: m2 );
            float A2 = aLUT2d[symidx(r0, regions[i_])];
            h += (2.0f*A2/(cz*cz)) * (m2 - m0);
        }
    }

    // write back, result is H + Hdmi + Hex
    float invMs = inv_Msat(Ms_, mumaxPointerPresent(mumaxPointerMask, 6u), Ms_mul, I);
    Hx[I] += h.x*invMs;
    Hy[I] += h.y*invMs;
    Hz[I] += h.z*invMs;
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

// -----------------------------------------------------------------------------
// Source: cuda/dmibulk.cu
// -----------------------------------------------------------------------------
#line 1 "dmibulk.cu"

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
kernel void adddmibulk(
    device float* Hx [[buffer(0)]],
    device float* Hy [[buffer(1)]],
    device float* Hz [[buffer(2)]],
    device float* mx [[buffer(3)]],
    device float* my [[buffer(4)]],
    device float* mz [[buffer(5)]],
    device float* Ms_ [[buffer(6)]],
    constant float& Ms_mul [[buffer(7)]],
    device float* aLUT2d [[buffer(8)]],
    device float* DLUT2d [[buffer(9)]],
    device uchar* regions [[buffer(10)]],
    constant float& cx [[buffer(11)]],
    constant float& cy [[buffer(12)]],
    constant float& cz [[buffer(13)]],
    constant int& Nx [[buffer(14)]],
    constant int& Ny [[buffer(15)]],
    constant int& Nz [[buffer(16)]],
    constant uchar& PBC [[buffer(17)]],
    constant uchar& OpenBC [[buffer(18)]],
    constant uint& mumaxPointerMask [[buffer(19)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);                      // central cell index
    float3 h = make_float3(0.0,0.0,0.0);          // add to H
    float3 m0 = make_float3(mx[I], my[I], mz[I]); // central m
    uchar r0 = regions[I];
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
        int r1 = is0(m1)? r0 : regions[i_];
        float A = aLUT2d[symidx(r0, r1)];
        float D = DLUT2d[symidx(r0, r1)];
        float D_2A = D/(2.0f*A);
        if (!is0(m1) || !OpenBC){                      // do nothing at an open boundary
            if (is0(m1)) {                             // neighbor missing
                m1.x = m0.x;
                m1.y = m0.y - (-cx * D_2A * m0.z);
                m1.z = m0.z + (-cx * D_2A * m0.y);
            }
            h   += (2.0f*A/(cx*cx)) * (m1 - m0);       // exchange
            h.y += (D/cx)*(-m1.z);
            h.z -= (D/cx)*(-m1.y);
        }
    }


    {
        float3 m2 = make_float3(0.0f, 0.0f, 0.0f);     // right neighbor
        i_ = idx(hclampx(ix+1), iy, iz);
        if (ix+1 < Nx || PBCx) {
            m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        int r1 = is0(m2)? r0 : regions[i_];
        float A = aLUT2d[symidx(r0, r1)];
        float D = DLUT2d[symidx(r0, r1)];
        float D_2A = D/(2.0f*A);
        if (!is0(m2) || !OpenBC){
            if (is0(m2)) {
                m2.x = m0.x;
                m2.y = m0.y - (+cx * D_2A * m0.z);
                m2.z = m0.z + (+cx * D_2A * m0.y);
            }
            h   += (2.0f*A/(cx*cx)) * (m2 - m0);
            h.y += (D/cx)*(m2.z);
            h.z -= (D/cx)*(m2.y);
        }
    }

    // y derivatives (along height)
    {
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, lclampy(iy-1), iz);
        if (iy-1 >= 0 || PBCy) {
            m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        int r1 = is0(m1)? r0 : regions[i_];
        float A = aLUT2d[symidx(r0, r1)];
        float D = DLUT2d[symidx(r0, r1)];
        float D_2A = D/(2.0f*A);
        if (!is0(m1) || !OpenBC){
            if (is0(m1)) {
                m1.x = m0.x + (-cy * D_2A * m0.z);
                m1.y = m0.y;
                m1.z = m0.z - (-cy * D_2A * m0.x);
            }
            h   += (2.0f*A/(cy*cy)) * (m1 - m0);
            h.x -= (D/cy)*(-m1.z);
            h.z += (D/cy)*(-m1.x);
        }
    }

    {
        float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, hclampy(iy+1), iz);
        if  (iy+1 < Ny || PBCy) {
            m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        int r1 = is0(m2)? r0 : regions[i_];
        float A = aLUT2d[symidx(r0, r1)];
        float D = DLUT2d[symidx(r0, r1)];
        float D_2A = D/(2.0f*A);
        if (!is0(m2) || !OpenBC){
            if (is0(m2)) {
                m2.x = m0.x + (+cy * D_2A * m0.z);
                m2.y = m0.y;
                m2.z = m0.z - (+cy * D_2A * m0.x);
            }
            h   += (2.0f*A/(cy*cy)) * (m2 - m0);
            h.x -= (D/cy)*(m2.z);
            h.z += (D/cy)*(m2.x);
        }
    }

    // only take vertical derivative for 3D sim or for Neumann BC
    if ((Nz != 1) || (!OpenBC)) {
        // bottom neighbor
        {
            float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
            i_ = idx(ix, iy, lclampz(iz-1));
            if (iz-1 >= 0 || PBCz) {
                m1 = make_float3(mx[i_], my[i_], mz[i_]);
            }
            int r1 = is0(m1)? r0 : regions[i_];
            float A = aLUT2d[symidx(r0, r1)];
            float D = DLUT2d[symidx(r0, r1)];
            float D_2A = D/(2.0f*A);
            if (!is0(m1) || !OpenBC){
                if (is0(m1)) {
                    m1.x = m0.x - (-cz * D_2A * m0.y);
                    m1.y = m0.y + (-cz * D_2A * m0.x);
                    m1.z = m0.z;
                }
                h   += (2.0f*A/(cz*cz)) * (m1 - m0);
                h.x += (D/cz)*(- m1.y);
                h.y -= (D/cz)*(- m1.x);
            }
        }

        // top neighbor
        {
            float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
            i_ = idx(ix, iy, hclampz(iz+1));
            if (iz+1 < Nz || PBCz) {
                m2 = make_float3(mx[i_], my[i_], mz[i_]);
            }
            int r1 = is0(m2)? r0 : regions[i_];
            float A = aLUT2d[symidx(r0, r1)];
            float D = DLUT2d[symidx(r0, r1)];
            float D_2A = D/(2.0f*A);
            if (!is0(m2) || !OpenBC){
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
    }

    // write back, result is H + Hdmi + Hex
    float invMs = inv_Msat(Ms_, mumaxPointerPresent(mumaxPointerMask, 6u), Ms_mul, I);
    Hx[I] += h.x*invMs;
    Hy[I] += h.y*invMs;
    Hz[I] += h.z*invMs;
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

// -----------------------------------------------------------------------------
// Source: cuda/dotproduct.cu
// -----------------------------------------------------------------------------
#line 1 "dotproduct.cu"


// dst += prefactor * dot(a,b)
kernel void dotproduct(
    device float* dst [[buffer(0)]],
    constant float& prefactor [[buffer(1)]],
    device float* ax [[buffer(2)]],
    device float* ay [[buffer(3)]],
    device float* az [[buffer(4)]],
    device float* bx [[buffer(5)]],
    device float* by [[buffer(6)]],
    device float* bz [[buffer(7)]],
    constant int& N [[buffer(8)]],
    constant uint& mumaxPointerMask [[buffer(9)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        float3 A = {ax[i], ay[i], az[i]};
        float3 B = {bx[i], by[i], bz[i]};
        dst[i] += prefactor * dot(A, B);
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/exchange.cu
// -----------------------------------------------------------------------------
#line 1 "exchange.cu"

// See exchange.go for more details.
kernel void addexchange(
    device float* Bx [[buffer(0)]],
    device float* By [[buffer(1)]],
    device float* Bz [[buffer(2)]],
    device float* mx [[buffer(3)]],
    device float* my [[buffer(4)]],
    device float* mz [[buffer(5)]],
    device float* Ms_ [[buffer(6)]],
    constant float& Ms_mul [[buffer(7)]],
    device float* aLUT2d [[buffer(8)]],
    device uchar* regions [[buffer(9)]],
    constant float& wx [[buffer(10)]],
    constant float& wy [[buffer(11)]],
    constant float& wz [[buffer(12)]],
    constant int& Nx [[buffer(13)]],
    constant int& Ny [[buffer(14)]],
    constant int& Nz [[buffer(15)]],
    constant uchar& PBC [[buffer(16)]],
    constant uint& mumaxPointerMask [[buffer(17)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

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

    uchar r0 = regions[I];
    float3 B  = make_float3(0.0,0.0,0.0);

    int i_;    // neighbor index
    float3 m_; // neighbor mag
    float a__; // inter-cell exchange stiffness

    // left neighbor
    i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wx * a__ *(m_ - m0);

    // right neighbor
    i_  = idx(hclampx(ix+1), iy, iz);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    m_  = ( is0(m_)? m0: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wx * a__ *(m_ - m0);

    // back neighbor
    i_  = idx(ix, lclampy(iy-1), iz);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    m_  = ( is0(m_)? m0: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ *(m_ - m0);

    // front neighbor
    i_  = idx(ix, hclampy(iy+1), iz);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    m_  = ( is0(m_)? m0: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ *(m_ - m0);

    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // bottom neighbor
        i_  = idx(ix, iy, lclampz(iz-1));
        m_  = make_float3(mx[i_], my[i_], mz[i_]);
        m_  = ( is0(m_)? m0: m_ );
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wz * a__ *(m_ - m0);

        // top neighbor
        i_  = idx(ix, iy, hclampz(iz+1));
        m_  = make_float3(mx[i_], my[i_], mz[i_]);
        m_  = ( is0(m_)? m0: m_ );
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wz * a__ *(m_ - m0);
    }

    float invMs = inv_Msat(Ms_, mumaxPointerPresent(mumaxPointerMask, 6u), Ms_mul, I);
    Bx[I] += B.x*invMs;
    By[I] += B.y*invMs;
    Bz[I] += B.z*invMs;
}

// -----------------------------------------------------------------------------
// Source: cuda/exchangedecode.cu
// -----------------------------------------------------------------------------
#line 1 "exchangedecode.cu"

// see exchange.go
kernel void exchangedecode(
    device float* dst [[buffer(0)]],
    device float* aLUT2d [[buffer(1)]],
    device uchar* regions [[buffer(2)]],
    constant float& wx [[buffer(3)]],
    constant float& wy [[buffer(4)]],
    constant float& wz [[buffer(5)]],
    constant int& Nx [[buffer(6)]],
    constant int& Ny [[buffer(7)]],
    constant int& Nz [[buffer(8)]],
    constant uchar& PBC [[buffer(9)]],
    constant uint& mumaxPointerMask [[buffer(10)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    // central cell
    int I = idx(ix, iy, iz);
    uchar r0 = regions[I];

    int i_;    // neighbor index
    float avg = 0.0f;

    // left neighbor
    i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
    avg += aLUT2d[symidx(r0, regions[i_])];

    // right neighbor
    i_  = idx(hclampx(ix+1), iy, iz);
    avg += aLUT2d[symidx(r0, regions[i_])];

    // back neighbor
    i_  = idx(ix, lclampy(iy-1), iz);
    avg += aLUT2d[symidx(r0, regions[i_])];

    // front neighbor
    i_  = idx(ix, hclampy(iy+1), iz);
    avg += aLUT2d[symidx(r0, regions[i_])];

    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // bottom neighbor
        i_  = idx(ix, iy, lclampz(iz-1));
        avg += aLUT2d[symidx(r0, regions[i_])];

        // top neighbor
        i_  = idx(ix, iy, hclampz(iz+1));
        avg += aLUT2d[symidx(r0, regions[i_])];

        avg /= 6;
    } else {
        avg /= 4;
    }

    dst[I] = avg;
}

// -----------------------------------------------------------------------------
// Source: cuda/hopf-emergentmagneticfield-solidangle.cu
// -----------------------------------------------------------------------------
#line 1 "hopf-emergentmagneticfield-solidangle.cu"

// Returns the topological charge contribution on an elementary triangle ijk
// Order of arguments is important here to preserve the same measure of chirality
// Note: the result is zero if an argument is zero, or when two arguments are the same
 inline float triangleCharge__hopf_emergentmagneticfield_solidangle(float3 mi, float3 mj, float3 mk) {
    float numer   = dot(mi, cross(mj, mk));
    float denom   = 1.0f + dot(mi, mj) + dot(mi, mk) + dot(mj, mk);
    return 2.0f * atan2(numer, denom);
}

// Set the emergent magnetic field F_i = (1/8π) ε_{ijk} m · (∂m/∂x_j × ∂m/∂x_k) based on the solid angle
// subtended by triangle associated with three spins: a,b,c
//
// 	  q_{a,b,c} = 2 atan[(a . b x c /(1 + a.b + a.c + b.c)]
//
//    F_i = (1/16) (q_{0,1,2} + q_{0,2,3} + q_{0,3,4} + q_{0,4,1})
//
// analogous to the method for calculating the topological charge density in topologicalchargelattice.cu
kernel void setemergentmagneticfieldsolidangle(
    device float* Fx [[buffer(0)]],
    device float* Fy [[buffer(1)]],
    device float* Fz [[buffer(2)]],
    device float* mx [[buffer(3)]],
    device float* my [[buffer(4)]],
    device float* mz [[buffer(5)]],
    constant float& prefactor [[buffer(6)]],
    constant float& icycz [[buffer(7)]],
    constant float& iczcx [[buffer(8)]],
    constant float& icxcy [[buffer(9)]],
    constant int& Nx [[buffer(10)]],
    constant int& Ny [[buffer(11)]],
    constant int& Nz [[buffer(12)]],
    constant uchar& PBC [[buffer(13)]],
    constant uint& mumaxPointerMask [[buffer(14)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int i0 = idx(ix, iy, iz);                        // central cell index
    float3 m0 = make_float3(mx[i0], my[i0], mz[i0]); // central cell magnetization

    if(is0(m0)) {
        Fx[i0] = 0.0f;
        Fy[i0] = 0.0f;
        Fz[i0] = 0.0f;
        return;
    }


    ////////
    // Fx //
    ////////

    // accumulator for Fx
    float fx = 0.0;

    // indices of the 4 neighbors (counter clockwise)
    int i1 = idx(ix, hclampy(iy+1), iz); // (i+1,j)
    int i2 = idx(ix, iy, hclampz(iz+1)); // (i,j+1)
    int i3 = idx(ix, lclampy(iy-1), iz); // (i-1,j)
    int i4 = idx(ix, iy, lclampz(iz-1)); // (i,j-1)

    // magnetization of the 4 neighbors
    float3 m1 = make_float3(mx[i1], my[i1], mz[i1]);
    float3 m2 = make_float3(mx[i2], my[i2], mz[i2]);
    float3 m3 = make_float3(mx[i3], my[i3], mz[i3]);
    float3 m4 = make_float3(mx[i4], my[i4], mz[i4]);

    // contribution from the upper right triangle
    // if diagonally opposite neighbor is not zero, use a weight of 1/2 to avoid counting charges twice
    if ((iy+1<Ny || PBCy) && (iz+1<Nz || PBCz)) {
        int i_ = idx(ix, hclampy(iy+1), hclampz(iz+1)); // diagonal opposite neighbor in upper right quadrant
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fx += weight * triangleCharge__hopf_emergentmagneticfield_solidangle(m0, m1, m2);
    }

    // upper left
    if ((iy-1>=0 || PBCy) && (iz+1<Nz || PBCz)) {
        int i_ = idx(ix, lclampy(iy-1), hclampz(iz+1));
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fx += weight * triangleCharge__hopf_emergentmagneticfield_solidangle(m0, m2, m3);
    }

    // bottom left
    if ((iy-1>=0 || PBCy) && (iz-1>=0 || PBCz)) {
        int i_ = idx(ix, lclampy(iy-1), lclampz(iz-1));
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fx += weight * triangleCharge__hopf_emergentmagneticfield_solidangle(m0, m3, m4);
    }

    // bottom right
    if ((iy+1<Ny || PBCy) && (iz-1>=0 || PBCz)) {
        int i_ = idx(ix, hclampy(iy+1), lclampz(iz-1));
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fx += weight * triangleCharge__hopf_emergentmagneticfield_solidangle(m0, m4, m1);
    }


    ////////
    // Fy //
    ////////

    // accumulator for Fy
    float fy = 0.0;

    // indices of the 4 neighbors (counter clockwise)
    i1 = idx(ix, iy, hclampz(iz+1)); // (i+1,j)
    i2 = idx(hclampx(ix+1), iy, iz); // (i,j+1)
    i3 = idx(ix, iy, lclampz(iz-1)); // (i-1,j)
    i4 = idx(lclampx(ix-1), iy, iz); // (i,j-1)

    // magnetization of the 4 neighbors
    m1 = make_float3(mx[i1], my[i1], mz[i1]);
    m2 = make_float3(mx[i2], my[i2], mz[i2]);
    m3 = make_float3(mx[i3], my[i3], mz[i3]);
    m4 = make_float3(mx[i4], my[i4], mz[i4]);

    // contribution from the upper right triangle
    // if diagonally opposite neighbor is not zero, use a weight of 1/2 to avoid counting charges twice
    if ((iz+1<Nz || PBCz) && (ix+1<Nx || PBCx)) {
        int i_ = idx(hclampx(ix+1), iy, hclampz(iz+1)); // diagonal opposite neighbor in upper right quadrant
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fy += weight * triangleCharge__hopf_emergentmagneticfield_solidangle(m0, m1, m2);
    }

    // upper left
    if ((iz-1>=0 || PBCz) && (ix+1<Nx || PBCx)) {
        int i_ = idx(hclampx(ix+1), iy, lclampz(iz-1));
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fy += weight * triangleCharge__hopf_emergentmagneticfield_solidangle(m0, m2, m3);
    }

    // bottom left
    if ((ix-1>=0 || PBCx) && (iy-1>=0 || PBCy)) {
        int i_ = idx(lclampx(ix-1), iy, lclampz(iz-1));
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fy += weight * triangleCharge__hopf_emergentmagneticfield_solidangle(m0, m3, m4);
    }

    // bottom right
    if ((ix+1<Nx || PBCx) && (iy-1>=0 || PBCy)) {
        int i_ = idx(lclampx(ix-1), iy, hclampz(iz+1));
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fy += weight * triangleCharge__hopf_emergentmagneticfield_solidangle(m0, m4, m1);
    }


    ////////
    // Fz //
    ////////

    // accumulator for Fz
    float fz = 0.0;

    // indices of the 4 neighbors (counter clockwise)
    i1 = idx(hclampx(ix+1), iy, iz); // (i+1,j)
    i2 = idx(ix, hclampy(iy+1), iz); // (i,j+1)
    i3 = idx(lclampx(ix-1), iy, iz); // (i-1,j)
    i4 = idx(ix, lclampy(iy-1), iz); // (i,j-1)

    // magnetization of the 4 neighbors
    m1 = make_float3(mx[i1], my[i1], mz[i1]);
    m2 = make_float3(mx[i2], my[i2], mz[i2]);
    m3 = make_float3(mx[i3], my[i3], mz[i3]);
    m4 = make_float3(mx[i4], my[i4], mz[i4]);

    // contribution from the upper right triangle
    // if diagonally opposite neighbor is not zero, use a weight of 1/2 to avoid counting charges twice
    if ((ix+1<Nx || PBCx) && (iy+1<Ny || PBCy)) {
        int i_ = idx(hclampx(ix+1), hclampy(iy+1), iz); // diagonal opposite neighbor in upper right quadrant
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fz += weight * triangleCharge__hopf_emergentmagneticfield_solidangle(m0, m1, m2);
    }

    // upper left
    if ((ix-1>=0 || PBCx) && (iy+1<Ny || PBCy)) {
        int i_ = idx(lclampx(ix-1), hclampy(iy+1), iz);
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fz += weight * triangleCharge__hopf_emergentmagneticfield_solidangle(m0, m2, m3);
    }

    // bottom left
    if ((ix-1>=0 || PBCx) && (iy-1>=0 || PBCy)) {
        int i_ = idx(lclampx(ix-1), lclampy(iy-1), iz);
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fz += weight * triangleCharge__hopf_emergentmagneticfield_solidangle(m0, m3, m4);
    }

    // bottom right
    if ((ix+1<Nx || PBCx) && (iy-1>=0 || PBCy)) {
        int i_ = idx(hclampx(ix+1), lclampy(iy-1), iz);
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        fz += weight * triangleCharge__hopf_emergentmagneticfield_solidangle(m0, m4, m1);
    }

    Fx[i0] = 2 * prefactor * icycz * fx;
    Fy[i0] = 2 * prefactor * iczcx * fy;
    Fz[i0] = 2 * prefactor * icxcy * fz;
}

// -----------------------------------------------------------------------------
// Source: cuda/hopf-emergentmagneticfieldfivepoint.cu
// -----------------------------------------------------------------------------
#line 1 "hopf-emergentmagneticfieldfivepoint.cu"

// Sets the emergent magnetic field F_i = (1/8π) ε_{ijk} m · (∂m/∂x_j × ∂m/∂x_k)
// See hopfindex-five-point.go
kernel void setemergentmagneticfieldfivepoint(
    device float* Fx [[buffer(0)]],
    device float* Fy [[buffer(1)]],
    device float* Fz [[buffer(2)]],
    device float* mx [[buffer(3)]],
    device float* my [[buffer(4)]],
    device float* mz [[buffer(5)]],
    constant float& prefactor [[buffer(6)]],
    constant float& icycz [[buffer(7)]],
    constant float& iczcx [[buffer(8)]],
    constant float& icxcy [[buffer(9)]],
    constant int& Nx [[buffer(10)]],
    constant int& Ny [[buffer(11)]],
    constant int& Nz [[buffer(12)]],
    constant uchar& PBC [[buffer(13)]],
    constant uint& mumaxPointerMask [[buffer(14)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

    int I = idx(ix, iy, iz);  // central cell index

    float3 m0 = make_float3(mx[I], my[I], mz[I]);     // +0
    float3 dmdx = make_float3(0.0f, 0.0f, 0.0f);      // ∂m/∂x
    float3 dmdy = make_float3(0.0f, 0.0f, 0.0f);      // ∂m/∂y
    float3 dmdz = make_float3(0.0f, 0.0f, 0.0f);      // ∂m/∂y
    float3 dmdy_x_dmdz = make_float3(0.0, 0.0, 0.0);  // ∂m/∂y × ∂m/∂z
    float3 dmdz_x_dmdx = make_float3(0.0, 0.0, 0.0);  // ∂m/∂z × ∂m/∂x
    float3 dmdx_x_dmdy = make_float3(0.0, 0.0, 0.0);  // ∂m/∂x × ∂m/∂y
    int    i_;                                        // neighbor index

    if(is0(m0))
    {
        Fx[I] = 0.0f;
        Fy[I] = 0.0f;
        Fz[I] = 0.0f;
        return;
    }

    // x derivatives (along length)
    {
        float3 m_m2 = make_float3(0.0f, 0.0f, 0.0f);     // -2
        i_ = idx(lclampx(ix-2), iy, iz);                 // load neighbor m if inside grid, keep 0 otherwise
        if (ix-2 >= 0 || PBCx)
        {
            m_m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_m1 = make_float3(0.0f, 0.0f, 0.0f);     // -1
        i_ = idx(lclampx(ix-1), iy, iz);                 // load neighbor m if inside grid, keep 0 otherwise
        if (ix-1 >= 0 || PBCx)
        {
            m_m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p1 = make_float3(0.0f, 0.0f, 0.0f);     // +1
        i_ = idx(hclampx(ix+1), iy, iz);
        if (ix+1 < Nx || PBCx)
        {
            m_p1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p2 = make_float3(0.0f, 0.0f, 0.0f);     // +2
        i_ = idx(hclampx(ix+2), iy, iz);
        if (ix+2 < Nx || PBCx)
        {
            m_p2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        if (is0(m_p1) && is0(m_m1))                       //  +0
        {
            dmdx = make_float3(0.0f, 0.0f, 0.0f);         // --1-- zero
        }
        else if ((is0(m_m2) | is0(m_p2)) && !is0(m_p1) && !is0(m_m1))
        {
            dmdx = 0.5f * (m_p1 - m_m1);                  // -111-, 1111-, -1111 central difference,  ε ~ h^2
        }
        else if (is0(m_p1) && is0(m_m2))
        {
            dmdx =  m0 - m_m1;                            // -11-- backward difference, ε ~ h^1
        }
        else if (is0(m_m1) && is0(m_p2))
        {
            dmdx = -m0 + m_p1;                            // --11- forward difference,  ε ~ h^1
        }
        else if (!is0(m_m2) && is0(m_p1))
        {
            dmdx =  0.5f * m_m2 - 2.0f * m_m1 + 1.5f * m0; // 111-- backward difference, ε ~ h^2
        }
        else if (!is0(m_p2) && is0(m_m1))
        {
            dmdx = -0.5f * m_p2 + 2.0f * m_p1 - 1.5f * m0; // --111 forward difference,  ε ~ h^2
        }
        else
        {
            dmdx = (2.0f/3.0f)*(m_p1 - m_m1) + (1.0f/12.0f)*(m_m2 - m_p2); // 11111 central difference,  ε ~ h^4
        }
    }

    // y derivatives (along height)
    {
        float3 m_m2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, lclampy(iy-2), iz);
        if (iy-2 >= 0 || PBCy)
        {
            m_m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_m1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, lclampy(iy-1), iz);
        if (iy-1 >= 0 || PBCy)
        {
            m_m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, hclampy(iy+1), iz);
        if  (iy+1 < Ny || PBCy)
        {
            m_p1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, hclampy(iy+2), iz);
        if  (iy+2 < Ny || PBCy)
        {
            m_p2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        if (is0(m_p1) && is0(m_m1))                                        //  +0
        {
            dmdy = make_float3(0.0f, 0.0f, 0.0f);                          // --1-- zero
        }
        else if ((is0(m_m2) | is0(m_p2)) && !is0(m_p1) && !is0(m_m1))
        {
            dmdy = 0.5f * (m_p1 - m_m1);                                   // -111-, 1111-, -1111 central difference,  ε ~ h^2
        }
        else if (is0(m_p1) && is0(m_m2))
        {
            dmdy =  m0 - m_m1;                                             // -11-- backward difference, ε ~ h^1
        }
        else if (is0(m_m1) && is0(m_p2))
        {
            dmdy = -m0 + m_p1;                                             // --11- forward difference,  ε ~ h^1
        }
        else if (!is0(m_m2) && is0(m_p1))
        {
            dmdy =  0.5f * m_m2 - 2.0f * m_m1 + 1.5f * m0;                 // 111-- backward difference, ε ~ h^2
        }
        else if (!is0(m_p2) && is0(m_m1))
        {
            dmdy = -0.5f * m_p2 + 2.0f * m_p1 - 1.5f * m0;                 // --111 forward difference,  ε ~ h^2
        }
        else
        {
            dmdy = (2.0f/3.0f)*(m_p1 - m_m1) + (1.0f/12.0f)*(m_m2 - m_p2); // 11111 central difference,  ε ~ h^4
        }
    }

    // z derivatives (along depth)
    {
        float3 m_m2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, lclampz(iz-2));
        if (iz-2 >= 0 || PBCz)
        {
            m_m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_m1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, lclampz(iz-1));
        if (iz-1 >= 0 || PBCz)
        {
            m_m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, hclampz(iz+1));
        if  (iz+1 < Nz || PBCz)
        {
            m_p1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, hclampz(iz+2));
        if  (iz+2 < Nz || PBCz)
        {
            m_p2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        if (is0(m_p1) && is0(m_m1))                                        //  +0
        {
            dmdz = make_float3(0.0f, 0.0f, 0.0f);                          // --1-- zero
        }
        else if ((is0(m_m2) | is0(m_p2)) && !is0(m_p1) && !is0(m_m1))
        {
            dmdz = 0.5f * (m_p1 - m_m1);                                   // -111-, 1111-, -1111 central difference,  ε ~ h^2
        }
        else if (is0(m_p1) && is0(m_m2))
        {
            dmdz =  m0 - m_m1;                                             // -11-- backward difference, ε ~ h^1
        }
        else if (is0(m_m1) && is0(m_p2))
        {
            dmdz = -m0 + m_p1;                                             // --11- forward difference,  ε ~ h^1
        }
        else if (!is0(m_m2) && is0(m_p1))
        {
            dmdz =  0.5f * m_m2 - 2.0f * m_m1 + 1.5f * m0;                 // 111-- backward difference, ε ~ h^2
        }
        else if (!is0(m_p2) && is0(m_m1))
        {
            dmdz = -0.5f * m_p2 + 2.0f * m_p1 - 1.5f * m0;                 // --111 forward difference,  ε ~ h^2
        }
        else
        {
            dmdz = (2.0f/3.0f)*(m_p1 - m_m1) + (1.0f/12.0f)*(m_m2 - m_p2); // 11111 central difference,  ε ~ h^4
        }
    }


    dmdy_x_dmdz = cross(dmdy, dmdz);
    dmdz_x_dmdx = cross(dmdz, dmdx);
    dmdx_x_dmdy = cross(dmdx, dmdy);

    Fx[I] = 2 * prefactor * icycz * dot(m0, dmdy_x_dmdz);
    Fy[I] = 2 * prefactor * iczcx * dot(m0, dmdz_x_dmdx);
    Fz[I] = 2 * prefactor * icxcy * dot(m0, dmdx_x_dmdy);
}

// -----------------------------------------------------------------------------
// Source: cuda/hopf-emergentmagneticfieldtwopoint.cu
// -----------------------------------------------------------------------------
#line 1 "hopf-emergentmagneticfieldtwopoint.cu"

// Sets the emergent magnetic field F_i = (1/8π) ε_{ijk} m · (∂m/∂x_j × ∂m/∂x_k)
// See hopfindex-two-point.go
kernel void setemergentmagneticfieldtwopoint(
    device float* Fx [[buffer(0)]],
    device float* Fy [[buffer(1)]],
    device float* Fz [[buffer(2)]],
    device float* mx [[buffer(3)]],
    device float* my [[buffer(4)]],
    device float* mz [[buffer(5)]],
    constant float& prefactor [[buffer(6)]],
    constant float& icycz [[buffer(7)]],
    constant float& iczcx [[buffer(8)]],
    constant float& icxcy [[buffer(9)]],
    constant int& Nx [[buffer(10)]],
    constant int& Ny [[buffer(11)]],
    constant int& Nz [[buffer(12)]],
    constant uchar& PBC [[buffer(13)]],
    constant uint& mumaxPointerMask [[buffer(14)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

    int I = idx(ix, iy, iz);  // central cell index

    float3 m0 = make_float3(mx[I], my[I], mz[I]);     // +0
    float3 dmdx = make_float3(0.0f, 0.0f, 0.0f);      // ∂m/∂x
    float3 dmdy = make_float3(0.0f, 0.0f, 0.0f);      // ∂m/∂y
    float3 dmdz = make_float3(0.0f, 0.0f, 0.0f);      // ∂m/∂y
    float3 dmdy_x_dmdz = make_float3(0.0, 0.0, 0.0);  // ∂m/∂y × ∂m/∂z
    float3 dmdz_x_dmdx = make_float3(0.0, 0.0, 0.0);  // ∂m/∂z × ∂m/∂x
    float3 dmdx_x_dmdy = make_float3(0.0, 0.0, 0.0);  // ∂m/∂x × ∂m/∂y
    int    i_;                                        // neighbor index

    if(is0(m0))
    {
        Fx[I] = 0.0f;
        Fy[I] = 0.0f;
        Fz[I] = 0.0f;
        return;
    }

    // x derivatives (along length)
    {
        float3 m_m1 = make_float3(0.0f, 0.0f, 0.0f);     // -1
        i_ = idx(lclampx(ix-1), iy, iz);                 // load neighbor m if inside grid, keep 0 otherwise
        if (ix-1 >= 0 || PBCx)
        {
            m_m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p1 = make_float3(0.0f, 0.0f, 0.0f);     // +1
        i_ = idx(hclampx(ix+1), iy, iz);
        if (ix+1 < Nx || PBCx)
        {
            m_p1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        if (is0(m_p1) && is0(m_m1))                       //  system is one cell thick
        {
            dmdx = make_float3(0.0f, 0.0f, 0.0f);         // --1-- zero
        }
        else if (is0(m_p1))
        {
            dmdx = m0 - m_m1;                            // backward difference
        }
        else if (is0(m_m1))
        {
            dmdx = -m0 + m_p1;                            // forward difference
        }
        else
        {
            dmdx = 0.5f * (m_p1 - m_m1);                  // central difference
        }
    }

    // y derivatives (along height)
    {
        float3 m_m1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, lclampy(iy-1), iz);
        if (iy-1 >= 0 || PBCy)
        {
            m_m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, hclampy(iy+1), iz);
        if  (iy+1 < Ny || PBCy)
        {
            m_p1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        if (is0(m_p1) && is0(m_m1))                       //  system is one cell thick
        {
            dmdy = make_float3(0.0f, 0.0f, 0.0f);         // --1-- zero
        }
        else if (is0(m_p1))
        {
            dmdy = m0 - m_m1;                            // backward difference
        }
        else if (is0(m_m1))
        {
            dmdy = -m0 + m_p1;                            // forward difference
        }
        else
        {
            dmdy = 0.5f * (m_p1 - m_m1);                  // central difference
        }
    }

    // z derivatives (along depth)
    {
        float3 m_m1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, lclampz(iz-1));
        if (iz-1 >= 0 || PBCz)
        {
            m_m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, iy, hclampz(iz+1));
        if  (iz+1 < Nz || PBCz)
        {
            m_p1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        if (is0(m_p1) && is0(m_m1))                       //  system is one cell thick
        {
            dmdz = make_float3(0.0f, 0.0f, 0.0f);         // --1-- zero
        }
        else if (is0(m_p1))
        {
            dmdz = m0 - m_m1;                            // backward difference
        }
        else if (is0(m_m1))
        {
            dmdz = -m0 + m_p1;                            // forward difference
        }
        else
        {
            dmdz = 0.5f * (m_p1 - m_m1);                  // central difference
        }
    }

    dmdy_x_dmdz = cross(dmdy, dmdz);
    dmdz_x_dmdx = cross(dmdz, dmdx);
    dmdx_x_dmdy = cross(dmdx, dmdy);

    Fx[I] = 2 * prefactor * icycz * dot(m0, dmdy_x_dmdz);
    Fy[I] = 2 * prefactor * iczcx * dot(m0, dmdz_x_dmdx);
    Fz[I] = 2 * prefactor * icxcy * dot(m0, dmdx_x_dmdy);
}

// -----------------------------------------------------------------------------
// Source: cuda/hopf-vectorpotential.cu
// -----------------------------------------------------------------------------
#line 1 "hopf-vectorpotential.cu"

// Calculate the vector potential in the gauge that
// A_x = ∫_-∞^y F_z dy'
// A_y = 0
// A_z = -∫_-∞^y F_x dy'

// We approximate these integrals using cumulative sums from the bottom of the system
// (i.e. minimum value of y) up to the cell at which A is to be calculated
// e.g. A_x = ∫_-∞^y F_z dy' ≈ Σ_{iy_ = 0}^{iy_ = iy-1} F_z(ix, iy_, iz) * cy

kernel void setvectorpotential(
    device float* Ax [[buffer(0)]],
    device float* Ay [[buffer(1)]],
    device float* Az [[buffer(2)]],
    device float* Fx [[buffer(3)]],
    device float* Fy [[buffer(4)]],
    device float* Fz [[buffer(5)]],
    constant float& cy [[buffer(6)]],
    constant int& Nx [[buffer(7)]],
    constant int& Ny [[buffer(8)]],
    constant int& Nz [[buffer(9)]],
    constant uchar& PBC [[buffer(10)]],
    constant uint& mumaxPointerMask [[buffer(11)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

    int I = idx(ix, iy, iz);  // Index of cell of interest
    int i_;                   // Index of summand

    float3 a = make_float3(0.0f, 0.0f, 0.0f);

    for (int iy_ = 0; iy_ < iy; iy_++) {  // Cumulative sum along y-axis up to cell of interest within system
        i_ = idx(ix, iy_, iz);
        a.x -= Fz[i_] * cy;
        a.z += Fx[i_] * cy;
    }

    Ax[I] = a.x;
    Ay[I] = a.y;
    Az[I] = a.z;

}

// -----------------------------------------------------------------------------
// Source: cuda/hopfindex-solidangle-fourier-field.cu
// -----------------------------------------------------------------------------
#line 1 "hopfindex-solidangle-fourier-field.cu"
// Reconstructs the full Fourier transformed field array for negative wavenumbers using Hermitian symmetry F(-k_x, -k_y, -k_z) = F(k_x, k_y, k_z)^*
kernel void solidanglefourierfield(
    device float* fftFx_partial [[buffer(0)]],
    device float* fftFy_partial [[buffer(1)]],
    device float* fftFz_partial [[buffer(2)]],
    device float* fftFx [[buffer(3)]],
    device float* fftFy [[buffer(4)]],
    device float* fftFz [[buffer(5)]],
    constant int& Nx [[buffer(6)]],
    constant int& Ny [[buffer(7)]],
    constant int& Nz [[buffer(8)]],
    constant uint& mumaxPointerMask [[buffer(9)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix>= Nx || iy>= Ny || iz>=Nz) {
        return;
    }

    int I = (iz*Ny + iy)*Nx + ix;
    int e = 2 * I;

    int I_partial = (iz*Ny + iy)*(Nx/2+1) + ix;
    int e_partial = 2 * I_partial;

    if (ix <= Nx/2) {

        fftFx[e  ] = fftFx_partial[e_partial  ];
        fftFx[e+1] = fftFx_partial[e_partial+1];
        fftFy[e  ] = fftFy_partial[e_partial  ];
        fftFy[e+1] = fftFy_partial[e_partial+1];
        fftFz[e  ] = fftFz_partial[e_partial  ];
        fftFz[e+1] = fftFz_partial[e_partial+1];

    } else {

        int ix_neg = (Nx - ix) % Nx;
        int iy_neg = (Ny - iy) % Ny;
        int iz_neg = (Nz - iz) % Nz;

        int I_neg = (iz_neg*Ny + iy_neg)*(Nx/2+1) + ix_neg;
        int e_neg = 2 * I_neg;

        // Fill in the rest of the values using Hermitian symmetry: F(-k_x, -k_y, -k_z) = F(k_x, k_y, k_z)^*
        fftFx[e  ] =  fftFx_partial[e_neg  ];
        fftFx[e+1] = -fftFx_partial[e_neg+1];
        fftFy[e  ] =  fftFy_partial[e_neg  ];
        fftFy[e+1] = -fftFy_partial[e_neg+1];
        fftFz[e  ] =  fftFz_partial[e_neg  ];
        fftFz[e+1] = -fftFz_partial[e_neg+1];

    }

}

// -----------------------------------------------------------------------------
// Source: cuda/hopfindex-solidangle-fourier-scale.cu
// -----------------------------------------------------------------------------
#line 1 "hopfindex-solidangle-fourier-scale.cu"
// Rescale the effective field to a unit system where the cell spacing = 1
kernel void scaleemergentfield(
    device float* Fx_scale [[buffer(0)]],
    device float* Fy_scale [[buffer(1)]],
    device float* Fz_scale [[buffer(2)]],
    device float* Fx [[buffer(3)]],
    device float* Fy [[buffer(4)]],
    device float* Fz [[buffer(5)]],
    constant float& cx [[buffer(6)]],
    constant float& cy [[buffer(7)]],
    constant float& cz [[buffer(8)]],
    constant int& Nx [[buffer(9)]],
    constant int& Ny [[buffer(10)]],
    constant int& Nz [[buffer(11)]],
    constant uint& mumaxPointerMask [[buffer(12)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix>= Nx || iy>= Ny || iz>=Nz) {
        return;
    }

    int I = (iz*Ny + iy)*Nx + ix;

    Fx_scale[I] = Fx[I] * cy * cz;
    Fy_scale[I] = Fy[I] * cx * cz;
    Fz_scale[I] = Fz[I] * cx * cy;
}

// -----------------------------------------------------------------------------
// Source: cuda/hopfindex-solidangle-fourier-summand.cu
// -----------------------------------------------------------------------------
#line 1 "hopfindex-solidangle-fourier-summand.cu"

// Calculates the summand F(-k) · [k × F(k)] / k^2
kernel void solidanglefouriersummand(
    device float* summand_array [[buffer(0)]],
    device float* FkX_array [[buffer(1)]],
    device float* FkY_array [[buffer(2)]],
    device float* FkZ_array [[buffer(3)]],
    constant int& Nx [[buffer(4)]],
    constant int& Ny [[buffer(5)]],
    constant int& Nz [[buffer(6)]],
    constant uint& mumaxPointerMask [[buffer(7)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix>= Nx || iy>= Ny || iz>=Nz) {
        return;
    }

    float kx = static_cast<float>(ix) / Nx;
    float ky = static_cast<float>(iy) / Ny;
    float kz = static_cast<float>(iz) / Nz;

    // Account for positive and negative frequencies (k-space values are in the range [-1/2, 1/2])
    if (ix >= Nx/2)
        kx -= 1.0f;
    if (iy >= Ny/2)
        ky -= 1.0f;
    if (iz >= Nz/2)
        kz -= 1.0f;

    float k2 = kx*kx + ky*ky + kz*kz;

    int I = (iz*Ny + iy)*Nx + ix;
    int e = 2 * I;

    // Avoid division by zero at kx = ky = kz = 0
    if (k2 == 0.0f) {
        summand_array[I] = 0.0f;

    } else {

        float reFkX  =  FkX_array[e  ];
        float reFkY  =  FkY_array[e  ];
        float reFkZ  =  FkZ_array[e  ];
        float imFkX  =  FkX_array[e+1];
        float imFkY  =  FkY_array[e+1];
        float imFkZ  =  FkZ_array[e+1];
        float imFmkX = -FkX_array[e+1];
        float imFmkY = -FkY_array[e+1];
        float imFmkZ = -FkZ_array[e+1];

        float2 FkX  = mumaxMakeComplex(reFkX, imFkX);
        float2 FkY  = mumaxMakeComplex(reFkY, imFkY);
        float2 FkZ  = mumaxMakeComplex(reFkZ, imFkZ);
        float2 FmkX = mumaxMakeComplex(reFkX, imFmkX);
        float2 FmkY = mumaxMakeComplex(reFkY, imFmkY);
        float2 FmkZ = mumaxMakeComplex(reFkZ, imFmkZ);

        float2 kx_comp = mumaxMakeComplex(kx, 0.0f);
        float2 ky_comp = mumaxMakeComplex(ky, 0.0f);
        float2 kz_comp = mumaxMakeComplex(kz, 0.0f);

        // Calculate F(-k) x (k · F(k)) / k^2
        float summand = mumaxComplexImag(
                            mumaxComplexAdd(
                                mumaxComplexAdd(
                                    mumaxComplexMul(FmkX, mumaxComplexSub(mumaxComplexMul(ky_comp, FkZ), mumaxComplexMul(kz_comp, FkY))),
                                    mumaxComplexMul(FmkY, mumaxComplexSub(mumaxComplexMul(kz_comp, FkX), mumaxComplexMul(kx_comp, FkZ)))
                                ),
                                mumaxComplexMul(FmkZ, mumaxComplexSub(mumaxComplexMul(kx_comp, FkY), mumaxComplexMul(ky_comp, FkX)))
                            )
                        );

        summand /= k2;
        summand_array[I] = summand;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/kernmulc.cu
// -----------------------------------------------------------------------------
#line 1 "kernmulc.cu"
kernel void kernmulC(
    device float* fftM [[buffer(0)]],
    device float* fftK [[buffer(1)]],
    constant int& Nx [[buffer(2)]],
    constant int& Ny [[buffer(3)]],
    constant uint& mumaxPointerMask [[buffer(4)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix>= Nx || iy>=Ny) {
        return;
    }

    int I = iy*Nx + ix;
    int e = 2 * I;

    float reM = fftM[e  ];
    float imM = fftM[e+1];
    float reK = fftK[e  ];
    float imK = fftK[e+1];

    fftM[e  ] = reM * reK - imM * imK;
    fftM[e+1] = reM * imK + imM * reK;
}

// -----------------------------------------------------------------------------
// Source: cuda/kernmulrsymm2dxy.cu
// -----------------------------------------------------------------------------
#line 1 "kernmulrsymm2dxy.cu"
// 2D XY (in-plane) micromagnetic kernel multiplication:
// |Mx| = |Kxx Kxy| * |Mx|
// |My|   |Kyx Kyy|   |My|
// Using the same symmetries as kernmulrsymm3d.cu
kernel void kernmulRSymm2Dxy(
    device float* fftMx [[buffer(0)]],
    device float* fftMy [[buffer(1)]],
    device float* fftKxx [[buffer(2)]],
    device float* fftKyy [[buffer(3)]],
    device float* fftKxy [[buffer(4)]],
    constant int& Nx [[buffer(5)]],
    constant int& Ny [[buffer(6)]],
    constant uint& mumaxPointerMask [[buffer(7)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix>= Nx || iy>=Ny) {
        return;
    }

    int I = iy*Nx + ix;
    int e = 2 * I;

    float reMx = fftMx[e  ];
    float imMx = fftMx[e+1];
    float reMy = fftMy[e  ];
    float imMy = fftMy[e+1];

    // symmetry factor
    float fxy = 1.0f;
    if (iy > Ny/2) {
        iy = Ny-iy;
        fxy = -fxy;
    }
    I = iy*Nx + ix;

    float Kxx = fftKxx[I];
    float Kyy = fftKyy[I];
    float Kxy = fxy * fftKxy[I];

    fftMx[e  ] = reMx * Kxx + reMy * Kxy;
    fftMx[e+1] = imMx * Kxx + imMy * Kxy;
    fftMy[e  ] = reMx * Kxy + reMy * Kyy;
    fftMy[e+1] = imMx * Kxy + imMy * Kyy;
}

// -----------------------------------------------------------------------------
// Source: cuda/kernmulrsymm2dz.cu
// -----------------------------------------------------------------------------
#line 1 "kernmulrsymm2dz.cu"
// 2D Z (out-of-plane only) micromagnetic kernel multiplication:
// Mz = Kzz * Mz
// Using the same symmetries as kernmulrsymm3d.cu
kernel void kernmulRSymm2Dz(
    device float* fftMz [[buffer(0)]],
    device float* fftKzz [[buffer(1)]],
    constant int& Nx [[buffer(2)]],
    constant int& Ny [[buffer(3)]],
    constant uint& mumaxPointerMask [[buffer(4)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix>= Nx || iy>=Ny) {
        return;
    }

    int I = iy*Nx + ix;
    int e = 2 * I;

    float reMz = fftMz[e  ];
    float imMz = fftMz[e+1];

    if (iy > Ny/2) {
        iy = Ny-iy;
    }
    I = iy*Nx + ix;

    float Kzz = fftKzz[I];

    fftMz[e  ] = reMz * Kzz;
    fftMz[e+1] = imMz * Kzz;
}

// -----------------------------------------------------------------------------
// Source: cuda/kernmulrsymm3d.cu
// -----------------------------------------------------------------------------
#line 1 "kernmulrsymm3d.cu"
// 3D micromagnetic kernel multiplication:
//
// |Mx|   |Kxx Kxy Kxz|   |Mx|
// |My| = |Kxy Kyy Kyz| * |My|
// |Mz|   |Kxz Kyz Kzz|   |Mz|
//
// ~kernel has mirror symmetry along Y and Z-axis,
// apart from first row,
// and is only stored (roughly) half:
//
// K11, K22, K02:
// xxxxx
// aaaaa
// bbbbb
// ....
// bbbbb
// aaaaa
//
// K12:
// xxxxx
// aaaaa
// bbbbb
// ...
// -bbbb
// -aaaa

kernel void kernmulRSymm3D(
    device float* fftMx [[buffer(0)]],
    device float* fftMy [[buffer(1)]],
    device float* fftMz [[buffer(2)]],
    device float* fftKxx [[buffer(3)]],
    device float* fftKyy [[buffer(4)]],
    device float* fftKzz [[buffer(5)]],
    device float* fftKyz [[buffer(6)]],
    device float* fftKxz [[buffer(7)]],
    device float* fftKxy [[buffer(8)]],
    constant int& Nx [[buffer(9)]],
    constant int& Ny [[buffer(10)]],
    constant int& Nz [[buffer(11)]],
    constant uint& mumaxPointerMask [[buffer(12)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix>= Nx || iy>= Ny || iz>=Nz) {
        return;
    }

    // fetch (complex) FFT'ed magnetization
    int I = (iz*Ny + iy)*Nx + ix;
    int e = 2 * I;
    float reMx = fftMx[e  ];
    float imMx = fftMx[e+1];
    float reMy = fftMy[e  ];
    float imMy = fftMy[e+1];
    float reMz = fftMz[e  ];
    float imMz = fftMz[e+1];

    // fetch kernel

    // minus signs are added to some elements if
    // reconstructed from symmetry.
    float signYZ = 1.0f;
    float signXZ = 1.0f;
    float signXY = 1.0f;

    // use symmetry to fetch from redundant parts:
    // mirror index into first quadrant and set signs.
    if (iy > Ny/2) {
        iy = Ny-iy;
        signYZ = -signYZ;
        signXY = -signXY;
    }
    if (iz > Nz/2) {
        iz = Nz-iz;
        signYZ = -signYZ;
        signXZ = -signXZ;
    }

    // fetch kernel element from non-redundant part
    // and apply minus signs for mirrored parts.
    I = (iz*(Ny/2+1) + iy)*Nx + ix; // Ny/2+1: only half is stored
    float Kxx = fftKxx[I];
    float Kyy = fftKyy[I];
    float Kzz = fftKzz[I];
    float Kyz = fftKyz[I] * signYZ;
    float Kxz = fftKxz[I] * signXZ;
    float Kxy = fftKxy[I] * signXY;

    // m * K matrix multiplication, overwrite m with result.
    fftMx[e  ] = reMx * Kxx + reMy * Kxy + reMz * Kxz;
    fftMx[e+1] = imMx * Kxx + imMy * Kxy + imMz * Kxz;
    fftMy[e  ] = reMx * Kxy + reMy * Kyy + reMz * Kyz;
    fftMy[e+1] = imMx * Kxy + imMy * Kyy + imMz * Kyz;
    fftMz[e  ] = reMx * Kxz + reMy * Kyz + reMz * Kzz;
    fftMz[e+1] = imMx * Kxz + imMy * Kyz + imMz * Kzz;
}

// -----------------------------------------------------------------------------
// Source: cuda/llnoprecess.cu
// -----------------------------------------------------------------------------
#line 1 "llnoprecess.cu"

// Landau-Lifshitz torque without precession
kernel void llnoprecess(
    device float* tx [[buffer(0)]],
    device float* ty [[buffer(1)]],
    device float* tz [[buffer(2)]],
    device float* mx [[buffer(3)]],
    device float* my [[buffer(4)]],
    device float* mz [[buffer(5)]],
    device float* hx [[buffer(6)]],
    device float* hy [[buffer(7)]],
    device float* hz [[buffer(8)]],
    constant int& N [[buffer(9)]],
    constant uint& mumaxPointerMask [[buffer(10)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 m = {mx[i], my[i], mz[i]};
        float3 H = {hx[i], hy[i], hz[i]};

        float3 mxH = cross(m, H);
        float3 torque = -cross(m, mxH);

        tx[i] = torque.x;
        ty[i] = torque.y;
        tz[i] = torque.z;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/lltorque2.cu
// -----------------------------------------------------------------------------
#line 1 "lltorque2.cu"

// Landau-Lifshitz torque.
kernel void lltorque2(
    device float* tx [[buffer(0)]],
    device float* ty [[buffer(1)]],
    device float* tz [[buffer(2)]],
    device float* mx [[buffer(3)]],
    device float* my [[buffer(4)]],
    device float* mz [[buffer(5)]],
    device float* hx [[buffer(6)]],
    device float* hy [[buffer(7)]],
    device float* hz [[buffer(8)]],
    device float* alpha_ [[buffer(9)]],
    constant float& alpha_mul [[buffer(10)]],
    constant int& N [[buffer(11)]],
    constant uint& mumaxPointerMask [[buffer(12)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 m = {mx[i], my[i], mz[i]};
        float3 H = {hx[i], hy[i], hz[i]};
        float alpha = amul(alpha_, mumaxPointerPresent(mumaxPointerMask, 9u), alpha_mul, i);

        float3 mxH = cross(m, H);
        float gilb = -1.0f / (1.0f + alpha * alpha);
        float3 torque = gilb * (mxH + alpha * cross(m, mxH));

        tx[i] = torque.x;
        ty[i] = torque.y;
        tz[i] = torque.z;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/madd2.cu
// -----------------------------------------------------------------------------
#line 1 "madd2.cu"

// dst[i] = fac1*src1[i] + fac2*src2[i];
kernel void madd2(
    device float* dst [[buffer(0)]],
    device float* src1 [[buffer(1)]],
    constant float& fac1 [[buffer(2)]],
    device float* src2 [[buffer(3)]],
    constant float& fac2 [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant uint& mumaxPointerMask [[buffer(6)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        dst[i] = fac1*src1[i] + fac2*src2[i];
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/madd3.cu
// -----------------------------------------------------------------------------
#line 1 "madd3.cu"

// dst[i] = fac1 * src1[i] + fac2 * src2[i] + fac3 * src3[i]
kernel void madd3(
    device float* dst [[buffer(0)]],
    device float* src1 [[buffer(1)]],
    constant float& fac1 [[buffer(2)]],
    device float* src2 [[buffer(3)]],
    constant float& fac2 [[buffer(4)]],
    device float* src3 [[buffer(5)]],
    constant float& fac3 [[buffer(6)]],
    constant int& N [[buffer(7)]],
    constant uint& mumaxPointerMask [[buffer(8)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        dst[i] = (fac1 * src1[i]) + (fac2 * src2[i] + fac3 * src3[i]);
        // parens for better accuracy heun solver.
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/madd4.cu
// -----------------------------------------------------------------------------
#line 1 "madd4.cu"

// dst[i] = src1[i] * fac1 + src2[i] * fac2 + src3[i] * fac3 + src4[i] * fac4
kernel void madd4(
    device float* dst [[buffer(0)]],
    device float* src1 [[buffer(1)]],
    constant float& fac1 [[buffer(2)]],
    device float* src2 [[buffer(3)]],
    constant float& fac2 [[buffer(4)]],
    device float* src3 [[buffer(5)]],
    constant float& fac3 [[buffer(6)]],
    device float* src4 [[buffer(7)]],
    constant float& fac4 [[buffer(8)]],
    constant int& N [[buffer(9)]],
    constant uint& mumaxPointerMask [[buffer(10)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        dst[i] = (fac1*src1[i]) + (fac2*src2[i]) + (fac3*src3[i]) + (fac4*src4[i]);
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/madd5.cu
// -----------------------------------------------------------------------------
#line 1 "madd5.cu"

// dst[i] = src1[i] * fac1 + src2[i] * fac2 + src3[i] * fac3 + src4[i] * fac4 + src5[i] * fac5
kernel void madd5(
    device float* dst [[buffer(0)]],
    device float* src1 [[buffer(1)]],
    constant float& fac1 [[buffer(2)]],
    device float* src2 [[buffer(3)]],
    constant float& fac2 [[buffer(4)]],
    device float* src3 [[buffer(5)]],
    constant float& fac3 [[buffer(6)]],
    device float* src4 [[buffer(7)]],
    constant float& fac4 [[buffer(8)]],
    device float* src5 [[buffer(9)]],
    constant float& fac5 [[buffer(10)]],
    constant int& N [[buffer(11)]],
    constant uint& mumaxPointerMask [[buffer(12)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        dst[i] = (fac1*src1[i]) + (fac2*src2[i]) + (fac3*src3[i]) + (fac4*src4[i]) + (fac5*src5[i]);
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/madd6.cu
// -----------------------------------------------------------------------------
#line 1 "madd6.cu"

// dst[i] = src1[i] * fac1 + src2[i] * fac2 + src3[i] * fac3 + src4[i] * fac4 + src5[i] * fac5 + src6[i] * fac6
kernel void madd6(
    device float* dst [[buffer(0)]],
    device float* src1 [[buffer(1)]],
    constant float& fac1 [[buffer(2)]],
    device float* src2 [[buffer(3)]],
    constant float& fac2 [[buffer(4)]],
    device float* src3 [[buffer(5)]],
    constant float& fac3 [[buffer(6)]],
    device float* src4 [[buffer(7)]],
    constant float& fac4 [[buffer(8)]],
    device float* src5 [[buffer(9)]],
    constant float& fac5 [[buffer(10)]],
    device float* src6 [[buffer(11)]],
    constant float& fac6 [[buffer(12)]],
    constant int& N [[buffer(13)]],
    constant uint& mumaxPointerMask [[buffer(14)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        dst[i] = (fac1*src1[i]) + (fac2*src2[i]) + (fac3*src3[i]) + (fac4*src4[i]) + (fac5*src5[i]) + (fac6*src6[i]);
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/madd7.cu
// -----------------------------------------------------------------------------
#line 1 "madd7.cu"

// dst[i] = src1[i] * fac1 + src2[i] * fac2 + src3[i] * fac3 + src4[i] * fac4 + src5[i] * fac5 + src6[i] * fac6 + src7[i] * fac7
kernel void madd7(
    device float* dst [[buffer(0)]],
    device float* src1 [[buffer(1)]],
    constant float& fac1 [[buffer(2)]],
    device float* src2 [[buffer(3)]],
    constant float& fac2 [[buffer(4)]],
    device float* src3 [[buffer(5)]],
    constant float& fac3 [[buffer(6)]],
    device float* src4 [[buffer(7)]],
    constant float& fac4 [[buffer(8)]],
    device float* src5 [[buffer(9)]],
    constant float& fac5 [[buffer(10)]],
    device float* src6 [[buffer(11)]],
    constant float& fac6 [[buffer(12)]],
    device float* src7 [[buffer(13)]],
    constant float& fac7 [[buffer(14)]],
    constant int& N [[buffer(15)]],
    constant uint& mumaxPointerMask [[buffer(16)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        dst[i] = (fac1*src1[i]) + (fac2*src2[i]) + (fac3*src3[i]) + (fac4*src4[i]) + (fac5*src5[i]) + (fac6*src6[i]) + (fac7*src7[i]);
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/magnetoelasticfield.cu
// -----------------------------------------------------------------------------
#line 1 "magnetoelasticfield.cu"

// Add magneto-elastic coupling field to B.
// H = - δUmel / δM,
// where Umel is magneto-elastic energy density given by the eq. (12.18) of Gurevich&Melkov "Magnetization Oscillations and Waves", CRC Press, 1996
kernel void addmagnetoelasticfield(
    device float* Bx [[buffer(0)]],
    device float* By [[buffer(1)]],
    device float* Bz [[buffer(2)]],
    device float* mx [[buffer(3)]],
    device float* my [[buffer(4)]],
    device float* mz [[buffer(5)]],
    device float* exx_ [[buffer(6)]],
    constant float& exx_mul [[buffer(7)]],
    device float* eyy_ [[buffer(8)]],
    constant float& eyy_mul [[buffer(9)]],
    device float* ezz_ [[buffer(10)]],
    constant float& ezz_mul [[buffer(11)]],
    device float* exy_ [[buffer(12)]],
    constant float& exy_mul [[buffer(13)]],
    device float* exz_ [[buffer(14)]],
    constant float& exz_mul [[buffer(15)]],
    device float* eyz_ [[buffer(16)]],
    constant float& eyz_mul [[buffer(17)]],
    device float* B1_ [[buffer(18)]],
    constant float& B1_mul [[buffer(19)]],
    device float* B2_ [[buffer(20)]],
    constant float& B2_mul [[buffer(21)]],
    device float* Ms_ [[buffer(22)]],
    constant float& Ms_mul [[buffer(23)]],
    constant int& N [[buffer(24)]],
    constant uint& mumaxPointerMask [[buffer(25)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

	int I =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

	if (I < N) {

	    float Exx = amul(exx_, mumaxPointerPresent(mumaxPointerMask, 6u), exx_mul, I);
	    float Eyy = amul(eyy_, mumaxPointerPresent(mumaxPointerMask, 8u), eyy_mul, I);
	    float Ezz = amul(ezz_, mumaxPointerPresent(mumaxPointerMask, 10u), ezz_mul, I);

	    float Exy = amul(exy_, mumaxPointerPresent(mumaxPointerMask, 12u), exy_mul, I);
	    float Eyx = Exy;

	    float Exz = amul(exz_, mumaxPointerPresent(mumaxPointerMask, 14u), exz_mul, I);
	    float Ezx = Exz;

	    float Eyz = amul(eyz_, mumaxPointerPresent(mumaxPointerMask, 16u), eyz_mul, I);
	    float Ezy = Eyz;

		float invMs = inv_Msat(Ms_, mumaxPointerPresent(mumaxPointerMask, 22u), Ms_mul, I);

		float B1 = amul(B1_, mumaxPointerPresent(mumaxPointerMask, 18u), B1_mul, I) * invMs;
	    float B2 = amul(B2_, mumaxPointerPresent(mumaxPointerMask, 20u), B2_mul, I) * invMs;

	    float3 m = {mx[I], my[I], mz[I]};

	    Bx[I] += -2.0f*(B1*m.x*Exx + B2*(m.y*Exy + m.z*Exz));
	    By[I] += -2.0f*(B1*m.y*Eyy + B2*(m.x*Eyx + m.z*Eyz));
	    Bz[I] += -2.0f*(B1*m.z*Ezz + B2*(m.x*Ezx + m.y*Ezy));
	}
}

// -----------------------------------------------------------------------------
// Source: cuda/magnetoelasticforce.cu
// -----------------------------------------------------------------------------
#line 1 "magnetoelasticforce.cu"

// Calculate magneto-elastic force density
// fmelp = Σ ∂σpq / ∂xq (q = x, y, z) , σpq = ∂Umel / ∂epq,
// where epq is the strain tensor and
// Umel is the magneto-elastic energy density given by the eq. (12.18) of Gurevich&Melkov "Magnetization Oscillations and Waves", CRC Press, 1996
kernel void getmagnetoelasticforce(
    device float* fx [[buffer(0)]],
    device float* fy [[buffer(1)]],
    device float* fz [[buffer(2)]],
    device float* mx [[buffer(3)]],
    device float* my [[buffer(4)]],
    device float* mz [[buffer(5)]],
    device float* B1_ [[buffer(6)]],
    constant float& B1_mul [[buffer(7)]],
    device float* B2_ [[buffer(8)]],
    constant float& B2_mul [[buffer(9)]],
    constant float& rcsx [[buffer(10)]],
    constant float& rcsy [[buffer(11)]],
    constant float& rcsz [[buffer(12)]],
    constant int& Nx [[buffer(13)]],
    constant int& Ny [[buffer(14)]],
    constant int& Nz [[buffer(15)]],
    constant uchar& PBC [[buffer(16)]],
    constant uint& mumaxPointerMask [[buffer(17)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;


    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

    int I = idx(ix, iy, iz);                      // central cell index
    float3 m0 = make_float3(mx[I], my[I], mz[I]); // +0
    float3 dmdx = make_float3(0.0f, 0.0f, 0.0f);  // ∂m/∂x
    float3 dmdy = make_float3(0.0f, 0.0f, 0.0f);  // ∂m/∂y
    float3 dmdz = make_float3(0.0f, 0.0f, 0.0f);  // ∂m/∂z
    int i_;                                       // neighbor index

    // ∂m/∂x
	{
		float3 m_m2 = make_float3(0.0f, 0.0f, 0.0f);     // -2
        i_ = idx(lclampx(ix-2), iy, iz);                 // load neighbor m if inside grid, keep 0 otherwise
        if (ix-2 >= 0 || PBCx)
        {
            m_m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_m1 = make_float3(0.0f, 0.0f, 0.0f);     // -1
        i_ = idx(lclampx(ix-1), iy, iz);                 // load neighbor m if inside grid, keep 0 otherwise
        if (ix-1 >= 0 || PBCx)
        {
            m_m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p1 = make_float3(0.0f, 0.0f, 0.0f);     // +1
        i_ = idx(hclampx(ix+1), iy, iz);
        if (ix+1 < Nx || PBCx)
        {
            m_p1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p2 = make_float3(0.0f, 0.0f, 0.0f);     // +2
        i_ = idx(hclampx(ix+2), iy, iz);
        if (ix+2 < Nx || PBCx)
        {
            m_p2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

	    if (is0(m_p1) && is0(m_m1))                                        //  +0
	    {
	        dmdx = make_float3(0.0f, 0.0f, 0.0f);                          // --1-- zero
	    }
	    else if ((is0(m_m2) | is0(m_p2)) && !is0(m_p1) && !is0(m_m1))
	    {
	        dmdx = 0.5f * (m_p1 - m_m1);                                   // -111-, 1111-, -1111 central difference,  ε ~ h^2
	    }
	    else if (is0(m_p1) && is0(m_m2))
	    {
	        dmdx =  m0 - m_m1;                                             // -11-- backward difference, ε ~ h^1
	    }
	    else if (is0(m_m1) && is0(m_p2))
	    {
	        dmdx = -m0 + m_p1;                                             // --11- forward difference,  ε ~ h^1
	    }
	    else if (!is0(m_m2) && is0(m_p1))
	    {
	        dmdx =  0.5f * m_m2 - 2.0f * m_m1 + 1.5f * m0;                 // 111-- backward difference, ε ~ h^2
	    }
	    else if (!is0(m_p2) && is0(m_m1))
	    {
	        dmdx = -0.5f * m_p2 + 2.0f * m_p1 - 1.5f * m0;                 // --111 forward difference,  ε ~ h^2
	    }
	    else
	    {
	        dmdx = (2.0f/3.0f)*(m_p1 - m_m1) + (1.0f/12.0f)*(m_m2 - m_p2); // 11111 central difference,  ε ~ h^4
	    }
	}

	// ∂m/∂y
    {
	    float3 m_m2 = make_float3(0.0f, 0.0f, 0.0f);
	    i_ = idx(ix, lclampy(iy-2), iz);
	    if (iy-2 >= 0 || PBCy)
	    {
	        m_m2 = make_float3(mx[i_], my[i_], mz[i_]);
	    }

	    float3 m_m1 = make_float3(0.0f, 0.0f, 0.0f);
	    i_ = idx(ix, lclampy(iy-1), iz);
	    if (iy-1 >= 0 || PBCy)
	    {
	        m_m1 = make_float3(mx[i_], my[i_], mz[i_]);
	    }

	    float3 m_p1 = make_float3(0.0f, 0.0f, 0.0f);
	    i_ = idx(ix, hclampy(iy+1), iz);
	    if  (iy+1 < Ny || PBCy)
	    {
	        m_p1 = make_float3(mx[i_], my[i_], mz[i_]);
	    }

	    float3 m_p2 = make_float3(0.0f, 0.0f, 0.0f);
	    i_ = idx(ix, hclampy(iy+2), iz);
	    if  (iy+2 < Ny || PBCy)
	    {
	        m_p2 = make_float3(mx[i_], my[i_], mz[i_]);
	    }

	    if (is0(m_p1) && is0(m_m1))                                        //  +0
	    {
	        dmdy = make_float3(0.0f, 0.0f, 0.0f);                          // --1-- zero
	    }
	    else if ((is0(m_m2) | is0(m_p2)) && !is0(m_p1) && !is0(m_m1))
	    {
	        dmdy = 0.5f * (m_p1 - m_m1);                                   // -111-, 1111-, -1111 central difference,  ε ~ h^2
	    }
	    else if (is0(m_p1) && is0(m_m2))
	    {
	        dmdy =  m0 - m_m1;                                             // -11-- backward difference, ε ~ h^1
	    }
	    else if (is0(m_m1) && is0(m_p2))
	    {
	        dmdy = -m0 + m_p1;                                             // --11- forward difference,  ε ~ h^1
	    }
	    else if (!is0(m_m2) && is0(m_p1))
	    {
	        dmdy =  0.5f * m_m2 - 2.0f * m_m1 + 1.5f * m0;                 // 111-- backward difference, ε ~ h^2
	    }
	    else if (!is0(m_p2) && is0(m_m1))
	    {
	        dmdy = -0.5f * m_p2 + 2.0f * m_p1 - 1.5f * m0;                 // --111 forward difference,  ε ~ h^2
	    }
	    else
	    {
	        dmdy = (2.0f/3.0f)*(m_p1 - m_m1) + (1.0f/12.0f)*(m_m2 - m_p2); // 11111 central difference,  ε ~ h^4
	    }
    }


	// ∂u/∂z
    {
	    float3 m_m2 = make_float3(0.0f, 0.0f, 0.0f);
	    i_ = idx(ix, iy, lclampz(iz-2));
	    if (iz-2 >= 0 || PBCz)
	    {
	        m_m2 = make_float3(mx[i_], my[i_], mz[i_]);
	    }

	    float3 m_m1 = make_float3(0.0f, 0.0f, 0.0f);
	    i_ = idx(ix, iy, lclampz(iz-1));
	    if (iz-1 >= 0 || PBCz)
	    {
	        m_m1 = make_float3(mx[i_], my[i_], mz[i_]);
	    }

	    float3 m_p1 = make_float3(0.0f, 0.0f, 0.0f);
	    i_ = idx(ix, iy, hclampz(iz+1));
	    if  (iz+1 < Nz || PBCz)
	    {
	        m_p1 = make_float3(mx[i_], my[i_], mz[i_]);
	    }

	    float3 m_p2 = make_float3(0.0f, 0.0f, 0.0f);
	    i_ = idx(ix, iy, hclampz(iz+2));
	    if  (iz+2 < Nz || PBCz)
	    {
	        m_p2 = make_float3(mx[i_], my[i_], mz[i_]);
	    }

	    if (is0(m_p1) && is0(m_m1))                                        //  +0
	    {
	        dmdz = make_float3(0.0f, 0.0f, 0.0f);                          // --1-- zero
	    }
	    else if ((is0(m_m2) | is0(m_p2)) && !is0(m_p1) && !is0(m_m1))
	    {
	        dmdz = 0.5f * (m_p1 - m_m1);                                   // -111-, 1111-, -1111 central difference,  ε ~ h^2
	    }
	    else if (is0(m_p1) && is0(m_m2))
	    {
	        dmdz =  m0 - m_m1;                                             // -11-- backward difference, ε ~ h^1
	    }
	    else if (is0(m_m1) && is0(m_p2))
	    {
	        dmdz = -m0 + m_p1;                                             // --11- forward difference,  ε ~ h^1
	    }
	    else if (!is0(m_m2) && is0(m_p1))
	    {
	        dmdz =  0.5f * m_m2 - 2.0f * m_m1 + 1.5f * m0;                 // 111-- backward difference, ε ~ h^2
	    }
	    else if (!is0(m_p2) && is0(m_m1))
	    {
	        dmdz = -0.5f * m_p2 + 2.0f * m_p1 - 1.5f * m0;                 // --111 forward difference,  ε ~ h^2
	    }
	    else
	    {
	        dmdz = (2.0f/3.0f)*(m_p1 - m_m1) + (1.0f/12.0f)*(m_m2 - m_p2); // 11111 central difference,  ε ~ h^4
	    }
    }

    dmdx *= rcsx;
    dmdy *= rcsy;
    dmdz *= rcsz;

	float B1 = amul(B1_, mumaxPointerPresent(mumaxPointerMask, 6u), B1_mul, I);
	float B2 = amul(B2_, mumaxPointerPresent(mumaxPointerMask, 8u), B2_mul, I);

    fx[I] = 2.0f*B1*m0.x*dmdx.x + B2*(m0.x*(dmdy.y + dmdz.z) + m0.y*dmdy.x + m0.z*dmdz.x);
    fy[I] = 2.0f*B1*m0.y*dmdy.y + B2*(m0.x*dmdx.y + m0.y*(dmdx.x + dmdz.z) + m0.z*dmdz.y);
    fz[I] = 2.0f*B1*m0.z*dmdz.z + B2*(m0.x*dmdx.z + m0.y*dmdy.z + m0.z*(dmdx.x + dmdy.y));
}

// -----------------------------------------------------------------------------
// Source: cuda/maxangle.cu
// -----------------------------------------------------------------------------
#line 1 "maxangle.cu"

// See maxangle.go for more details.
kernel void setmaxangle(
    device float* dst [[buffer(0)]],
    device float* mx [[buffer(1)]],
    device float* my [[buffer(2)]],
    device float* mz [[buffer(3)]],
    device float* aLUT2d [[buffer(4)]],
    device uchar* regions [[buffer(5)]],
    constant int& Nx [[buffer(6)]],
    constant int& Ny [[buffer(7)]],
    constant int& Nz [[buffer(8)]],
    constant uchar& PBC [[buffer(9)]],
    constant uint& mumaxPointerMask [[buffer(10)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

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

    uchar r0 = regions[I];
    float angle  = 0.0f;

    int i_;    // neighbor index
    float3 m_; // neighbor mag
    float a__; // inter-cell exchange stiffness

    // left neighbor
    i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    a__ = aLUT2d[symidx(r0, regions[i_])];
    if (a__ != 0) {
        angle = max(angle, acos(dot(m_,m0)));
    }

    // right neighbor
    i_  = idx(hclampx(ix+1), iy, iz);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    m_  = ( is0(m_)? m0: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    if (a__ != 0) {
        angle = max(angle, acos(dot(m_,m0)));
    }

    // back neighbor
    i_  = idx(ix, lclampy(iy-1), iz);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    m_  = ( is0(m_)? m0: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    if (a__ != 0) {
        angle = max(angle, acos(dot(m_,m0)));
    }

    // front neighbor
    i_  = idx(ix, hclampy(iy+1), iz);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    m_  = ( is0(m_)? m0: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    if (a__ != 0) {
        angle = max(angle, acos(dot(m_,m0)));
    }

    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // bottom neighbor
        i_  = idx(ix, iy, lclampz(iz-1));
        m_  = make_float3(mx[i_], my[i_], mz[i_]);
        m_  = ( is0(m_)? m0: m_ );
        a__ = aLUT2d[symidx(r0, regions[i_])];
        if (a__ != 0) {
            angle = max(angle, acos(dot(m_,m0)));
        }

        // top neighbor
        i_  = idx(ix, iy, hclampz(iz+1));
        m_  = make_float3(mx[i_], my[i_], mz[i_]);
        m_  = ( is0(m_)? m0: m_ );
        a__ = aLUT2d[symidx(r0, regions[i_])];
        if (a__ != 0) {
            angle = max(angle, acos(dot(m_,m0)));
        }
    }

    dst[I] = angle;
}

// -----------------------------------------------------------------------------
// Source: cuda/minimize.cu
// -----------------------------------------------------------------------------
#line 1 "minimize.cu"

// Steepest descent energy minimizer
kernel void minimize(
    device float* mx [[buffer(0)]],
    device float* my [[buffer(1)]],
    device float* mz [[buffer(2)]],
    device float* m0x [[buffer(3)]],
    device float* m0y [[buffer(4)]],
    device float* m0z [[buffer(5)]],
    device float* tx [[buffer(6)]],
    device float* ty [[buffer(7)]],
    device float* tz [[buffer(8)]],
    constant float& dt [[buffer(9)]],
    constant int& N [[buffer(10)]],
    constant uint& mumaxPointerMask [[buffer(11)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 m0 = {m0x[i], m0y[i], m0z[i]};
        float3 t = {tx[i], ty[i], tz[i]};

        float t2 = dt*dt*dot(t, t);
        float3 result = (4 - t2) * m0 + 4 * dt * t;
        float divisor = 4 + t2;

        mx[i] = result.x / divisor;
        my[i] = result.y / divisor;
        mz[i] = result.z / divisor;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/mul.cu
// -----------------------------------------------------------------------------
#line 1 "mul.cu"
// dst[i] = a[i] * b[i]
kernel void mul(
    device float* dst [[buffer(0)]],
    device float* a [[buffer(1)]],
    device float* b [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant uint& mumaxPointerMask [[buffer(4)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        dst[i] = a[i] * b[i];
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/normalize.cu
// -----------------------------------------------------------------------------
#line 1 "normalize.cu"

// normalize vector {vx, vy, vz} to unit length, unless length or vol are zero.
kernel void normalize(
    device float* vx [[buffer(0)]],
    device float* vy [[buffer(1)]],
    device float* vz [[buffer(2)]],
    device float* vol [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint& mumaxPointerMask [[buffer(5)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float v = (!mumaxPointerPresent(mumaxPointerMask, 3u)? 1.0f: vol[i]);
        float3 V = {v*vx[i], v*vy[i], v*vz[i]};
        V = normalized(V);
        vx[i] = V.x;
        vy[i] = V.y;
        vz[i] = V.z;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/phi.cu
// -----------------------------------------------------------------------------
#line 1 "phi.cu"

kernel void setPhi(
    device float* phi [[buffer(0)]],
    device float* mx [[buffer(1)]],
    device float* my [[buffer(2)]],
    constant int& Nx [[buffer(3)]],
    constant int& Ny [[buffer(4)]],
    constant int& Nz [[buffer(5)]],
    constant uint& mumaxPointerMask [[buffer(6)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

    int I = idx(ix, iy, iz);                      // central cell index
    phi[I] = atan2(my[I], mx[I]);
}

// -----------------------------------------------------------------------------
// Source: cuda/reducedot.cu
// -----------------------------------------------------------------------------
#line 1 "reducedot.cu"

#define load_prod(i) (x1[i] * x2[i])

kernel void reducedot(
    device float* x1 [[buffer(0)]],
    device float* x2 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant float& initVal [[buffer(3)]],
    constant int& n [[buffer(4)]],
    constant uint& mumaxPointerMask [[buffer(5)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {
    reduce(load_prod, sum, atomicAdd)
}
#undef load_prod

// -----------------------------------------------------------------------------
// Source: cuda/reducemaxabs.cu
// -----------------------------------------------------------------------------
#line 1 "reducemaxabs.cu"

#define load_fabs(i) fabs(src[i])

kernel void reducemaxabs(
    device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant float& initVal [[buffer(2)]],
    constant int& n [[buffer(3)]],
    constant uint& mumaxPointerMask [[buffer(4)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {
    reduce(load_fabs, fmax, atomicFmaxabs)
}
#undef load_fabs

// -----------------------------------------------------------------------------
// Source: cuda/reducemaxdiff.cu
// -----------------------------------------------------------------------------
#line 1 "reducemaxdiff.cu"

#define load_diff(i) fabs(src1[i] - src2[i])

kernel void reducemaxdiff(
    device float* src1 [[buffer(0)]],
    device float* src2 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant float& initVal [[buffer(3)]],
    constant int& n [[buffer(4)]],
    constant uint& mumaxPointerMask [[buffer(5)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {
    reduce(load_diff, fmax, atomicFmaxabs)
}
#undef load_diff

// -----------------------------------------------------------------------------
// Source: cuda/reducemaxvecdiff2.cu
// -----------------------------------------------------------------------------
#line 1 "reducemaxvecdiff2.cu"

#define load_vecdiff2(i)  \
	pow2(x1[i] - x2[i]) + \
	pow2(y1[i] - y2[i]) + \
	pow2(z1[i] - z2[i])   \

kernel void reducemaxvecdiff2(
    device float* x1 [[buffer(0)]],
    device float* y1 [[buffer(1)]],
    device float* z1 [[buffer(2)]],
    device float* x2 [[buffer(3)]],
    device float* y2 [[buffer(4)]],
    device float* z2 [[buffer(5)]],
    device float* dst [[buffer(6)]],
    constant float& initVal [[buffer(7)]],
    constant int& n [[buffer(8)]],
    constant uint& mumaxPointerMask [[buffer(9)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {
    reduce(load_vecdiff2, fmax, atomicFmaxabs)
}
#undef load_vecdiff2

// -----------------------------------------------------------------------------
// Source: cuda/reducemaxvecnorm2.cu
// -----------------------------------------------------------------------------
#line 1 "reducemaxvecnorm2.cu"

#define load_vecnorm2(i) \
	pow2(x[i]) + pow2(y[i]) +  pow2(z[i])

kernel void reducemaxvecnorm2(
    device float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    device float* z [[buffer(2)]],
    device float* dst [[buffer(3)]],
    constant float& initVal [[buffer(4)]],
    constant int& n [[buffer(5)]],
    constant uint& mumaxPointerMask [[buffer(6)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {
    reduce(load_vecnorm2, fmax, atomicFmaxabs)
}
#undef load_vecnorm2

// -----------------------------------------------------------------------------
// Source: cuda/reducesum.cu
// -----------------------------------------------------------------------------
#line 1 "reducesum.cu"

#define load(i) src[i]

kernel void reducesum(
    device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant float& initVal [[buffer(2)]],
    constant int& n [[buffer(3)]],
    constant uint& mumaxPointerMask [[buffer(4)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {
    reduce(load, sum, atomicAdd)
}
#undef load

// -----------------------------------------------------------------------------
// Source: cuda/regionadds.cu
// -----------------------------------------------------------------------------
#line 1 "regionadds.cu"

// add region-based scalar to dst:
// dst[i] += LUT[region[i]]
kernel void regionadds(
    device float* dst [[buffer(0)]],
    device float* LUT [[buffer(1)]],
    device uchar* regions [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant uint& mumaxPointerMask [[buffer(4)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {

		uchar r = regions[i];
		dst[i] += LUT[r];
	}
}

// -----------------------------------------------------------------------------
// Source: cuda/regionaddv.cu
// -----------------------------------------------------------------------------
#line 1 "regionaddv.cu"

// add region-based vector to dst:
// dst[i] += LUT[region[i]]
kernel void regionaddv(
    device float* dstx [[buffer(0)]],
    device float* dsty [[buffer(1)]],
    device float* dstz [[buffer(2)]],
    device float* LUTx [[buffer(3)]],
    device float* LUTy [[buffer(4)]],
    device float* LUTz [[buffer(5)]],
    device uchar* regions [[buffer(6)]],
    constant int& N [[buffer(7)]],
    constant uint& mumaxPointerMask [[buffer(8)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        uchar r = regions[i];
        dstx[i] += LUTx[r];
        dsty[i] += LUTy[r];
        dstz[i] += LUTz[r];
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/regiondecode.cu
// -----------------------------------------------------------------------------
#line 1 "regiondecode.cu"

// decode the regions+LUT pair into an uncompressed array
kernel void regiondecode(
    device float* dst [[buffer(0)]],
    device float* LUT [[buffer(1)]],
    device uchar* regions [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant uint& mumaxPointerMask [[buffer(4)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        dst[i] = LUT[regions[i]];

    }
}

// -----------------------------------------------------------------------------
// Source: cuda/regionselect.cu
// -----------------------------------------------------------------------------
#line 1 "regionselect.cu"

kernel void regionselect(
    device float* dst [[buffer(0)]],
    device float* src [[buffer(1)]],
    device uchar* regions [[buffer(2)]],
    constant uchar& region [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint& mumaxPointerMask [[buffer(5)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i = ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        dst[i] = (regions[i] == region? src[i]: 0.0f);
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/resize.cu
// -----------------------------------------------------------------------------
#line 1 "resize.cu"

// Select and resize one layer for interactive output
kernel void resize(
    device float* dst [[buffer(0)]],
    constant int& Dx [[buffer(1)]],
    constant int& Dy [[buffer(2)]],
    constant int& Dz [[buffer(3)]],
    device float* src [[buffer(4)]],
    constant int& Sx [[buffer(5)]],
    constant int& Sy [[buffer(6)]],
    constant int& Sz [[buffer(7)]],
    constant int& layer [[buffer(8)]],
    constant int& scalex [[buffer(9)]],
    constant int& scaley [[buffer(10)]],
    constant uint& mumaxPointerMask [[buffer(11)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix<Dx && iy<Dy) {

        float sum = 0.0f;
        float n = 0.0f;

        for(int J=0; J<scaley; J++) {
            int j2 = iy*scaley+J;

            for(int K=0; K<scalex; K++) {
                int k2 = ix*scalex+K;

                if (j2 < Sy && k2 < Sx) {
                    sum += src[(layer*Sy + j2)*Sx + k2];
                    n += 1.0f;
                }
            }
        }
        dst[iy*Dx + ix] = sum / n;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/shiftbytes.cu
// -----------------------------------------------------------------------------
#line 1 "shiftbytes.cu"

// shift dst by shx cells (positive or negative) along X-axis.
// new edge value is clamp.
kernel void shiftbytes(
    device uchar* dst [[buffer(0)]],
    device uchar* src [[buffer(1)]],
    constant int& Nx [[buffer(2)]],
    constant int& Ny [[buffer(3)]],
    constant int& Nz [[buffer(4)]],
    constant int& shx [[buffer(5)]],
    constant uchar& clamp [[buffer(6)]],
    constant uint& mumaxPointerMask [[buffer(7)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        int ix2 = ix-shx;
        uchar newval;
        if (ix2 < 0 || ix2 >= Nx) {
            newval = clamp;
        } else {
            newval = src[idx(ix2, iy, iz)];
        }
        dst[idx(ix, iy, iz)] = newval;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/shiftbytesy.cu
// -----------------------------------------------------------------------------
#line 1 "shiftbytesy.cu"

// shift dst by shy cells (positive or negative) along Y-axis.
// new edge value is clamp.
kernel void shiftbytesy(
    device uchar* dst [[buffer(0)]],
    device uchar* src [[buffer(1)]],
    constant int& Nx [[buffer(2)]],
    constant int& Ny [[buffer(3)]],
    constant int& Nz [[buffer(4)]],
    constant int& shy [[buffer(5)]],
    constant uchar& clamp [[buffer(6)]],
    constant uint& mumaxPointerMask [[buffer(7)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        int iy2 = iy-shy;
        uchar newval;
        if (iy2 < 0 || iy2 >= Ny) {
            newval = clamp;
        } else {
            newval = src[idx(ix, iy2, iz)];
        }
        dst[idx(ix, iy, iz)] = newval;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/shiftedgecarryx.cu
// -----------------------------------------------------------------------------
#line 1 "shiftedgecarryx.cu"

// Shifts a component `src` of a vector field by `shx` cells along the X-axis.
// Unlike the normal `shiftx()`, the new edge value is the current edge value.
//
// To avoid the situation where the magnetization could be set to (0,0,0) within the geometry, it is
// also required to pass the two other vector components `othercomp` and `anothercomp` to this function.
// In cells where the vector (`src`, `othercomp`, `anothercomp`) is the zero-vector,
// `clampL` or `clampR` is used for the component `src` instead.
kernel void shiftedgecarryX(
    device float* dst [[buffer(0)]],
    device float* src [[buffer(1)]],
    device float* othercomp [[buffer(2)]],
    device float* anothercomp [[buffer(3)]],
    constant int& Nx [[buffer(4)]],
    constant int& Ny [[buffer(5)]],
    constant int& Nz [[buffer(6)]],
    constant int& shx [[buffer(7)]],
    constant float& clampL [[buffer(8)]],
    constant float& clampR [[buffer(9)]],
    constant uint& mumaxPointerMask [[buffer(10)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        int ix2 = ix-shx; // old X-index
        float newval;
        if (ix2 < 0) { // left edge (shifting right)
            newval = src[idx(0, iy, iz)];
            if (newval == 0 && othercomp[idx(0, iy, iz)] == 0 && anothercomp[idx(0, iy, iz)] == 0) { // If zero-vector
                newval = clampL;
            }
        } else if (ix2 >= Nx) { // right edge (shifting left)
            newval = src[idx(Nx-1, iy, iz)];
            if (newval == 0 && othercomp[idx(Nx-1, iy, iz)] == 0 && anothercomp[idx(Nx-1, iy, iz)] == 0) { // If zero-vector
                newval = clampR;
            }
        } else { // bulk, doesn't matter which way the shift is
            newval = src[idx(ix2, iy, iz)];
        }
        dst[idx(ix, iy, iz)] = newval;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/shiftedgecarryy.cu
// -----------------------------------------------------------------------------
#line 1 "shiftedgecarryy.cu"

// Shifts a component `src` of a vector field by `shy` cells along the Y-axis.
// Unlike the normal `shifty()`, the new edge value is the current edge value.
//
// To avoid the situation where the magnetization could be set to (0,0,0) within the geometry, it is
// also required to pass the two other vector components `othercomp` and `anothercomp` to this function.
// In cells where the vector (`src`, `othercomp`, `anothercomp`) is the zero-vector,
// `clampD` or `clampU` is used for the component `src` instead.
kernel void shiftedgecarryY(
    device float* dst [[buffer(0)]],
    device float* src [[buffer(1)]],
    device float* othercomp [[buffer(2)]],
    device float* anothercomp [[buffer(3)]],
    constant int& Nx [[buffer(4)]],
    constant int& Ny [[buffer(5)]],
    constant int& Nz [[buffer(6)]],
    constant int& shy [[buffer(7)]],
    constant float& clampD [[buffer(8)]],
    constant float& clampU [[buffer(9)]],
    constant uint& mumaxPointerMask [[buffer(10)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        int iy2 = iy-shy; // old Y-index
        float newval;
        if (iy2 < 0) { // bottom edge (shifting up)
            newval = src[idx(ix, 0, iz)];
            if (newval == 0 && othercomp[idx(ix, 0, iz)] == 0 && anothercomp[idx(ix, 0, iz)] == 0) { // If zero-vector
                newval = clampD;
            }
        } else if (iy2 >= Ny) { // top edge (shifting down)
            newval = src[idx(ix, Ny-1, iz)];
            if (newval == 0 && othercomp[idx(ix, Ny-1, iz)] == 0 && anothercomp[idx(ix, Ny-1, iz)] == 0) { // If zero-vector
                newval = clampU;
            }
        } else { // bulk, doesn't matter which way the shift is
            newval = src[idx(ix, iy2, iz)];
        }
        dst[idx(ix, iy, iz)] = newval;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/shiftx.cu
// -----------------------------------------------------------------------------
#line 1 "shiftx.cu"

// shift dst by shx cells (positive or negative) along X-axis.
// new edge value is clampL at left edge (-X) or clampR at right edge (+X).
kernel void shiftx(
    device float* dst [[buffer(0)]],
    device float* src [[buffer(1)]],
    constant int& Nx [[buffer(2)]],
    constant int& Ny [[buffer(3)]],
    constant int& Nz [[buffer(4)]],
    constant int& shx [[buffer(5)]],
    constant float& clampL [[buffer(6)]],
    constant float& clampR [[buffer(7)]],
    constant uint& mumaxPointerMask [[buffer(8)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        int ix2 = ix-shx;
        float newval;
        if (ix2 < 0) {
            newval = clampL;
        } else if (ix2 >= Nx) {
            newval = clampR;
        } else {
            newval = src[idx(ix2, iy, iz)];
        }
        dst[idx(ix, iy, iz)] = newval;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/shifty.cu
// -----------------------------------------------------------------------------
#line 1 "shifty.cu"

// shift dst by shy cells (positive or negative) along Y-axis.
// new edge value is clampD at bottom edge (-Y) or clampU at top edge (+Y).
kernel void shifty(
    device float* dst [[buffer(0)]],
    device float* src [[buffer(1)]],
    constant int& Nx [[buffer(2)]],
    constant int& Ny [[buffer(3)]],
    constant int& Nz [[buffer(4)]],
    constant int& shy [[buffer(5)]],
    constant float& clampD [[buffer(6)]],
    constant float& clampU [[buffer(7)]],
    constant uint& mumaxPointerMask [[buffer(8)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        int iy2 = iy-shy;
        float newval;
        if (iy2 < 0) {
            newval = clampD;
        } else if (iy2 >= Ny) {
            newval = clampU;
        } else {
            newval = src[idx(ix, iy2, iz)];
        }
        dst[idx(ix, iy, iz)] = newval;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/shiftz.cu
// -----------------------------------------------------------------------------
#line 1 "shiftz.cu"

// shift dst by shz cells (positive or negative) along Z-axis.
// new edge value is clampB at back edge (-Z) or clampF at front edge (+Z).
kernel void shiftz(
    device float* dst [[buffer(0)]],
    device float* src [[buffer(1)]],
    constant int& Nx [[buffer(2)]],
    constant int& Ny [[buffer(3)]],
    constant int& Nz [[buffer(4)]],
    constant int& shz [[buffer(5)]],
    constant float& clampB [[buffer(6)]],
    constant float& clampF [[buffer(7)]],
    constant uint& mumaxPointerMask [[buffer(8)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz) {
        int iz2 = iz-shz;
        float newval;
        if (iz2 < 0) {
            newval = clampB;
        } else if (iz2 >= Nz) {
            newval = clampF;
        } else {
            newval = src[idx(ix, iy, iz2)];
        }
        dst[idx(ix, iy, iz)] = newval;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/slonczewski2.cu
// -----------------------------------------------------------------------------
#line 1 "slonczewski2.cu"
// Original implementation by Mykola Dvornik for mumax2
// Modified for mumax3 by Arne Vansteenkiste, 2013, 2016


kernel void addslonczewskitorque2(
    device float* tx [[buffer(0)]],
    device float* ty [[buffer(1)]],
    device float* tz [[buffer(2)]],
    device float* mx [[buffer(3)]],
    device float* my [[buffer(4)]],
    device float* mz [[buffer(5)]],
    device float* Ms_ [[buffer(6)]],
    constant float& Ms_mul [[buffer(7)]],
    device float* jz_ [[buffer(8)]],
    constant float& jz_mul [[buffer(9)]],
    device float* px_ [[buffer(10)]],
    constant float& px_mul [[buffer(11)]],
    device float* py_ [[buffer(12)]],
    constant float& py_mul [[buffer(13)]],
    device float* pz_ [[buffer(14)]],
    constant float& pz_mul [[buffer(15)]],
    device float* alpha_ [[buffer(16)]],
    constant float& alpha_mul [[buffer(17)]],
    device float* pol_ [[buffer(18)]],
    constant float& pol_mul [[buffer(19)]],
    device float* lambda_ [[buffer(20)]],
    constant float& lambda_mul [[buffer(21)]],
    device float* epsPrime_ [[buffer(22)]],
    constant float& epsPrime_mul [[buffer(23)]],
    device float* thickness_ [[buffer(24)]],
    constant float& thickness_mul [[buffer(25)]],
    constant float& meshThickness [[buffer(26)]],
    constant float& freeLayerPosition [[buffer(27)]],
    constant int& N [[buffer(28)]],
    constant uint& mumaxPointerMask [[buffer(29)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 m = make_float3(mx[i], my[i], mz[i]);
        float  J = amul(jz_, mumaxPointerPresent(mumaxPointerMask, 8u), jz_mul, i);
        float3 p = normalized(vmul(px_, py_, pz_, mumaxPointerPresent(mumaxPointerMask, 10u), mumaxPointerPresent(mumaxPointerMask, 12u), mumaxPointerPresent(mumaxPointerMask, 14u), px_mul, py_mul, pz_mul, i));
        float  Ms           = amul(Ms_, mumaxPointerPresent(mumaxPointerMask, 6u), Ms_mul, i);
        float  alpha        = amul(alpha_, mumaxPointerPresent(mumaxPointerMask, 16u), alpha_mul, i);
        float  pol          = amul(pol_, mumaxPointerPresent(mumaxPointerMask, 18u), pol_mul, i);
        float  lambda       = amul(lambda_, mumaxPointerPresent(mumaxPointerMask, 20u), lambda_mul, i);
        float  epsilonPrime = amul(epsPrime_, mumaxPointerPresent(mumaxPointerMask, 22u), epsPrime_mul, i);

        float thickness = amul(thickness_, mumaxPointerPresent(mumaxPointerMask, 24u), thickness_mul, i);
        if (thickness == 0.0) { // if thickness is not set, use the thickness of the mesh instead
            thickness = meshThickness;
        }
        thickness *= freeLayerPosition; // switch sign if fixedlayer is at the bottom

        if (J == 0.0f || Ms == 0.0f) {
            return;
        }

        float beta    = (HBAR / QE) * (J / (thickness*Ms) );
        float lambda2 = lambda * lambda;
        float epsilon = pol * lambda2 / ((lambda2 + 1.0f) + (lambda2 - 1.0f) * dot(p, m));

        float A = beta * epsilon;
        float B = beta * epsilonPrime;

        float gilb     = 1.0f / (1.0f + alpha * alpha);
        float mxpxmFac = gilb * (A + alpha * B);
        float pxmFac   = gilb * (B - alpha * A);

        float3 pxm      = cross(p, m);
        float3 mxpxm    = cross(m, pxm);

        tx[i] += mxpxmFac * mxpxm.x + pxmFac * pxm.x;
        ty[i] += mxpxmFac * mxpxm.y + pxmFac * pxm.y;
        tz[i] += mxpxmFac * mxpxm.z + pxmFac * pxm.z;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/temperature2.cu
// -----------------------------------------------------------------------------
#line 1 "temperature2.cu"

// TODO: this could act on x,y,z, so that we need to call it only once.
kernel void settemperature2(
    device float* B [[buffer(0)]],
    device float* noise [[buffer(1)]],
    constant float& kB2_VgammaDt [[buffer(2)]],
    device float* Ms_ [[buffer(3)]],
    constant float& Ms_mul [[buffer(4)]],
    device float* temp_ [[buffer(5)]],
    constant float& temp_mul [[buffer(6)]],
    device float* alpha_ [[buffer(7)]],
    constant float& alpha_mul [[buffer(8)]],
    constant int& N [[buffer(9)]],
    constant uint& mumaxPointerMask [[buffer(10)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        float invMs = inv_Msat(Ms_, mumaxPointerPresent(mumaxPointerMask, 3u), Ms_mul, i);
        float temp = amul(temp_, mumaxPointerPresent(mumaxPointerMask, 5u), temp_mul, i);
        float alpha = amul(alpha_, mumaxPointerPresent(mumaxPointerMask, 7u), alpha_mul, i);
        B[i] = noise[i] * sqrt((kB2_VgammaDt * alpha * temp * invMs ));
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/theta.cu
// -----------------------------------------------------------------------------
#line 1 "theta.cu"

kernel void setTheta(
    device float* theta [[buffer(0)]],
    device float* mz [[buffer(1)]],
    constant int& Nx [[buffer(2)]],
    constant int& Ny [[buffer(3)]],
    constant int& Nz [[buffer(4)]],
    constant uint& mumaxPointerMask [[buffer(5)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

    int I = idx(ix, iy, iz);                      // central cell index
    theta[I] = acos(mz[I]);
}

// -----------------------------------------------------------------------------
// Source: cuda/topologicalcharge-lattice.cu
// -----------------------------------------------------------------------------
#line 1 "topologicalcharge-lattice.cu"

// Returns the topological charge contribution on an elementary triangle ijk
// Order of arguments is important here to preserve the same measure of chirality
// Note: the result is zero if an argument is zero, or when two arguments are the same
 inline float triangleCharge__topologicalcharge_lattice(float3 mi, float3 mj, float3 mk) {
    float numer   = dot(mi, cross(mj, mk));
    float denom   = 1.0f + dot(mi, mj) + dot(mi, mk) + dot(mj, mk);
    return 2.0f * atan2(numer, denom);
}

// Set s to the topological charge density for lattices based on the solid angle
// subtended by triangle associated with three spins: a,b,c
//
// 	  s = 2 atan[(a . b x c /(1 + a.b + a.c + b.c)] / (dx dy)
//
// After M Boettcher et al, New J Phys 20, 103014 (2018), adapted from
// B. Berg and M. Luescher, Nucl. Phys. B 190, 412 (1981), and implemented by
// Joo-Von Kim.
//
// A unit cell comprises two triangles, but s is a site-dependent quantity so we
// double-count and average over four triangles.
kernel void settopologicalchargelattice(
    device float* s [[buffer(0)]],
    device float* mx [[buffer(1)]],
    device float* my [[buffer(2)]],
    device float* mz [[buffer(3)]],
    constant float& icxcy [[buffer(4)]],
    constant int& Nx [[buffer(5)]],
    constant int& Ny [[buffer(6)]],
    constant int& Nz [[buffer(7)]],
    constant uchar& PBC [[buffer(8)]],
    constant uint& mumaxPointerMask [[buffer(9)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int i0 = idx(ix, iy, iz);                        // central cell index
    float3 m0 = make_float3(mx[i0], my[i0], mz[i0]); // central cell magnetization

    if(is0(m0)) {
        s[i0] = 0.0f;
        return;
    }

    // indices of the 4 neighbors (counter clockwise)
    int i1 = idx(hclampx(ix+1), iy, iz); // (i+1,j)
    int i2 = idx(ix, hclampy(iy+1), iz); // (i,j+1)
    int i3 = idx(lclampx(ix-1), iy, iz); // (i-1,j)
    int i4 = idx(ix, lclampy(iy-1), iz); // (i,j-1)

    // magnetization of the 4 neighbors
    float3 m1 = make_float3(mx[i1], my[i1], mz[i1]);
    float3 m2 = make_float3(mx[i2], my[i2], mz[i2]);
    float3 m3 = make_float3(mx[i3], my[i3], mz[i3]);
    float3 m4 = make_float3(mx[i4], my[i4], mz[i4]);

    // local topological charge (accumulator)
    float topcharge = 0.0;

    // charge contribution from the upper right triangle
    // if diagonally opposite neighbor is not zero, use a weight of 1/2 to avoid counting charges twice
    if ((ix+1<Nx || PBCx) && (iy+1<Ny || PBCy)) {
        int i_ = idx(hclampx(ix+1), hclampy(iy+1), iz); // diagonal opposite neighbor in upper right quadrant
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        topcharge += weight * triangleCharge__topologicalcharge_lattice(m0, m1, m2);
    }

    // upper left
    if ((ix-1>=0 || PBCx) && (iy+1<Ny || PBCy)) {
        int i_ = idx(lclampx(ix-1), hclampy(iy+1), iz);
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        topcharge += weight * triangleCharge__topologicalcharge_lattice(m0, m2, m3);
    }

    // bottom left
    if ((ix-1>=0 || PBCx) && (iy-1>=0 || PBCy)) {
        int i_ = idx(lclampx(ix-1), lclampy(iy-1), iz);
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        topcharge += weight * triangleCharge__topologicalcharge_lattice(m0, m3, m4);
    }

    // bottom right
    if ((ix+1<Nx || PBCx) && (iy-1>=0 || PBCy)) {
        int i_ = idx(hclampx(ix+1), lclampy(iy-1), iz);
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);
        float weight = is0(m_) ? 1 : 0.5;
        topcharge += weight * triangleCharge__topologicalcharge_lattice(m0, m4, m1);
    }

    s[i0] = icxcy * topcharge;
}

// -----------------------------------------------------------------------------
// Source: cuda/topologicalcharge.cu
// -----------------------------------------------------------------------------
#line 1 "topologicalcharge.cu"

// Set s to the topological charge density.
// See topologicalcharge.go.
kernel void settopologicalcharge(
    device float* s [[buffer(0)]],
    device float* mx [[buffer(1)]],
    device float* my [[buffer(2)]],
    device float* mz [[buffer(3)]],
    constant float& icxcy [[buffer(4)]],
    constant int& Nx [[buffer(5)]],
    constant int& Ny [[buffer(6)]],
    constant int& Nz [[buffer(7)]],
    constant uchar& PBC [[buffer(8)]],
    constant uint& mumaxPointerMask [[buffer(9)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

    int I = idx(ix, iy, iz);                      // central cell index

    float3 m0 = make_float3(mx[I], my[I], mz[I]); // +0
    float3 dmdx = make_float3(0.0f, 0.0f, 0.0f);  // ∂m/∂x
    float3 dmdy = make_float3(0.0f, 0.0f, 0.0f);  // ∂m/∂y
    float3 dmdx_x_dmdy = make_float3(0.0, 0.0, 0.0); // ∂m/∂x ❌ ∂m/∂y
    int i_;                                       // neighbor index

    if(is0(m0))
    {
        s[I] = 0.0f;
        return;
    }

    // x derivatives (along length)
    {
        float3 m_m2 = make_float3(0.0f, 0.0f, 0.0f);     // -2
        i_ = idx(lclampx(ix-2), iy, iz);                 // load neighbor m if inside grid, keep 0 otherwise
        if (ix-2 >= 0 || PBCx)
        {
            m_m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_m1 = make_float3(0.0f, 0.0f, 0.0f);     // -1
        i_ = idx(lclampx(ix-1), iy, iz);                 // load neighbor m if inside grid, keep 0 otherwise
        if (ix-1 >= 0 || PBCx)
        {
            m_m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p1 = make_float3(0.0f, 0.0f, 0.0f);     // +1
        i_ = idx(hclampx(ix+1), iy, iz);
        if (ix+1 < Nx || PBCx)
        {
            m_p1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p2 = make_float3(0.0f, 0.0f, 0.0f);     // +2
        i_ = idx(hclampx(ix+2), iy, iz);
        if (ix+2 < Nx || PBCx)
        {
            m_p2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        if (is0(m_p1) && is0(m_m1))                       //  +0
        {
            dmdx = make_float3(0.0f, 0.0f, 0.0f);         // --1-- zero
        }
        else if ((is0(m_m2) | is0(m_p2)) && !is0(m_p1) && !is0(m_m1))
        {
            dmdx = 0.5f * (m_p1 - m_m1);                  // -111-, 1111-, -1111 central difference,  ε ~ h^2
        }
        else if (is0(m_p1) && is0(m_m2))
        {
            dmdx =  m0 - m_m1;                            // -11-- backward difference, ε ~ h^1
        }
        else if (is0(m_m1) && is0(m_p2))
        {
            dmdx = -m0 + m_p1;                            // --11- forward difference,  ε ~ h^1
        }
        else if (!is0(m_m2) && is0(m_p1))
        {
            dmdx =  0.5f * m_m2 - 2.0f * m_m1 + 1.5f * m0; // 111-- backward difference, ε ~ h^2
        }
        else if (!is0(m_p2) && is0(m_m1))
        {
            dmdx = -0.5f * m_p2 + 2.0f * m_p1 - 1.5f * m0; // --111 forward difference,  ε ~ h^2
        }
        else
        {
            dmdx = (2.0f/3.0f)*(m_p1 - m_m1) + (1.0f/12.0f)*(m_m2 - m_p2); // 11111 central difference,  ε ~ h^4
        }
    }

    // y derivatives (along height)
    {
        float3 m_m2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, lclampy(iy-2), iz);
        if (iy-2 >= 0 || PBCy)
        {
            m_m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_m1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, lclampy(iy-1), iz);
        if (iy-1 >= 0 || PBCy)
        {
            m_m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p1 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, hclampy(iy+1), iz);
        if  (iy+1 < Ny || PBCy)
        {
            m_p1 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        float3 m_p2 = make_float3(0.0f, 0.0f, 0.0f);
        i_ = idx(ix, hclampy(iy+2), iz);
        if  (iy+2 < Ny || PBCy)
        {
            m_p2 = make_float3(mx[i_], my[i_], mz[i_]);
        }

        if (is0(m_p1) && is0(m_m1))                                        //  +0
        {
            dmdy = make_float3(0.0f, 0.0f, 0.0f);                          // --1-- zero
        }
        else if ((is0(m_m2) | is0(m_p2)) && !is0(m_p1) && !is0(m_m1))
        {
            dmdy = 0.5f * (m_p1 - m_m1);                                   // -111-, 1111-, -1111 central difference,  ε ~ h^2
        }
        else if (is0(m_p1) && is0(m_m2))
        {
            dmdy =  m0 - m_m1;                                             // -11-- backward difference, ε ~ h^1
        }
        else if (is0(m_m1) && is0(m_p2))
        {
            dmdy = -m0 + m_p1;                                             // --11- forward difference,  ε ~ h^1
        }
        else if (!is0(m_m2) && is0(m_p1))
        {
            dmdy =  0.5f * m_m2 - 2.0f * m_m1 + 1.5f * m0;                 // 111-- backward difference, ε ~ h^2
        }
        else if (!is0(m_p2) && is0(m_m1))
        {
            dmdy = -0.5f * m_p2 + 2.0f * m_p1 - 1.5f * m0;                 // --111 forward difference,  ε ~ h^2
        }
        else
        {
            dmdy = (2.0f/3.0f)*(m_p1 - m_m1) + (1.0f/12.0f)*(m_m2 - m_p2); // 11111 central difference,  ε ~ h^4
        }
    }
    dmdx_x_dmdy = cross(dmdx, dmdy);

    s[I] = icxcy * dot(m0, dmdx_x_dmdy);
}

// -----------------------------------------------------------------------------
// Source: cuda/uniaxialanisotropy2.cu
// -----------------------------------------------------------------------------
#line 1 "uniaxialanisotropy2.cu"

// Add uniaxial magnetocrystalline anisotropy field to B.
// http://www.southampton.ac.uk/~fangohr/software/oxs_uniaxial4.html
kernel void adduniaxialanisotropy2(
    device float* Bx [[buffer(0)]],
    device float* By [[buffer(1)]],
    device float* Bz [[buffer(2)]],
    device float* mx [[buffer(3)]],
    device float* my [[buffer(4)]],
    device float* mz [[buffer(5)]],
    device float* Ms_ [[buffer(6)]],
    constant float& Ms_mul [[buffer(7)]],
    device float* K1_ [[buffer(8)]],
    constant float& K1_mul [[buffer(9)]],
    device float* K2_ [[buffer(10)]],
    constant float& K2_mul [[buffer(11)]],
    device float* ux_ [[buffer(12)]],
    constant float& ux_mul [[buffer(13)]],
    device float* uy_ [[buffer(14)]],
    constant float& uy_mul [[buffer(15)]],
    device float* uz_ [[buffer(16)]],
    constant float& uz_mul [[buffer(17)]],
    constant int& N [[buffer(18)]],
    constant uint& mumaxPointerMask [[buffer(19)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 u   = normalized(vmul(ux_, uy_, uz_, mumaxPointerPresent(mumaxPointerMask, 12u), mumaxPointerPresent(mumaxPointerMask, 14u), mumaxPointerPresent(mumaxPointerMask, 16u), ux_mul, uy_mul, uz_mul, i));
        float invMs = inv_Msat(Ms_, mumaxPointerPresent(mumaxPointerMask, 6u), Ms_mul, i);
        float  K1  = amul(K1_, mumaxPointerPresent(mumaxPointerMask, 8u), K1_mul, i) * invMs;
        float  K2  = amul(K2_, mumaxPointerPresent(mumaxPointerMask, 10u), K2_mul, i) * invMs;
        float3 m   = {mx[i], my[i], mz[i]};
        float  mu  = dot(m, u);
        float3 Ba  = 2.0f*K1*    (mu)*u+
                     4.0f*K2*pow3(mu)*u;

        Bx[i] += Ba.x;
        By[i] += Ba.y;
        Bz[i] += Ba.z;
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/zeromask.cu
// -----------------------------------------------------------------------------
#line 1 "zeromask.cu"

// set dst to zero in cells where mask != 0
kernel void zeromask(
    device float* dst [[buffer(0)]],
    device float* maskLUT [[buffer(1)]],
    device uchar* regions [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant uint& mumaxPointerMask [[buffer(4)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        if (maskLUT[regions[i]] != 0) {
            dst[i] = 0;
        }
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/zeromaskinv.cu
// -----------------------------------------------------------------------------
#line 1 "zeromaskinv.cu"

// set dst to zero in cells where mask != 0
kernel void zeromaskinv(
    device float* dst [[buffer(0)]],
    device float* maskLUT [[buffer(1)]],
    device uchar* regions [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant uint& mumaxPointerMask [[buffer(4)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        if (maskLUT[regions[i]] == 0) {
            dst[i] = 0;
        }
    }
}

// -----------------------------------------------------------------------------
// Source: cuda/zhangli2.cu
// -----------------------------------------------------------------------------
#line 1 "zhangli2.cu"

#define PREFACTOR ((MUB) / (2 * QE * GAMMA0))

// spatial derivatives without dividing by cell size
#define deltax(in) (in[idx(hclampx(ix+1), iy, iz)] - in[idx(lclampx(ix-1), iy, iz)])
#define deltay(in) (in[idx(ix, hclampy(iy+1), iz)] - in[idx(ix, lclampy(iy-1), iz)])
#define deltaz(in) (in[idx(ix, iy, hclampz(iz+1))] - in[idx(ix, iy, lclampz(iz-1))])

kernel void addzhanglitorque2(
    device float* tx [[buffer(0)]],
    device float* ty [[buffer(1)]],
    device float* tz [[buffer(2)]],
    device float* mx [[buffer(3)]],
    device float* my [[buffer(4)]],
    device float* mz [[buffer(5)]],
    device float* Ms_ [[buffer(6)]],
    constant float& Ms_mul [[buffer(7)]],
    device float* jx_ [[buffer(8)]],
    constant float& jx_mul [[buffer(9)]],
    device float* jy_ [[buffer(10)]],
    constant float& jy_mul [[buffer(11)]],
    device float* jz_ [[buffer(12)]],
    constant float& jz_mul [[buffer(13)]],
    device float* alpha_ [[buffer(14)]],
    constant float& alpha_mul [[buffer(15)]],
    device float* xi_ [[buffer(16)]],
    constant float& xi_mul [[buffer(17)]],
    device float* pol_ [[buffer(18)]],
    constant float& pol_mul [[buffer(19)]],
    constant float& cx [[buffer(20)]],
    constant float& cy [[buffer(21)]],
    constant float& cz [[buffer(22)]],
    constant int& Nx [[buffer(23)]],
    constant int& Ny [[buffer(24)]],
    constant int& Nz [[buffer(25)]],
    constant uchar& PBC [[buffer(26)]],
    constant uint& mumaxPointerMask [[buffer(27)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int i = idx(ix, iy, iz);

    float alpha = amul(alpha_, mumaxPointerPresent(mumaxPointerMask, 14u), alpha_mul, i);
    float xi    = amul(xi_, mumaxPointerPresent(mumaxPointerMask, 16u), xi_mul, i);
    float pol   = amul(pol_, mumaxPointerPresent(mumaxPointerMask, 18u), pol_mul, i);
    float invMs = inv_Msat(Ms_, mumaxPointerPresent(mumaxPointerMask, 6u), Ms_mul, i);
    float b = invMs * PREFACTOR / (1.0f + xi*xi);
    float3 J = pol*vmul(jx_, jy_, jz_, mumaxPointerPresent(mumaxPointerMask, 8u), mumaxPointerPresent(mumaxPointerMask, 10u), mumaxPointerPresent(mumaxPointerMask, 12u), jx_mul, jy_mul, jz_mul, i);

    float3 hspin = make_float3(0.0f, 0.0f, 0.0f); // (u·∇)m
    if (J.x != 0.0f) {
        hspin += (b/cx)*J.x * make_float3(deltax(mx), deltax(my), deltax(mz));
    }
    if (J.y != 0.0f) {
        hspin += (b/cy)*J.y * make_float3(deltay(mx), deltay(my), deltay(mz));
    }
    if (J.z != 0.0f) {
        hspin += (b/cz)*J.z * make_float3(deltaz(mx), deltaz(my), deltaz(mz));
    }

    float3 m      = make_float3(mx[i], my[i], mz[i]);
    float3 torque = (-1.0f/(1.0f + alpha*alpha)) * (
                        (1.0f+xi*alpha) * cross(m, cross(m, hspin))
                        +(  xi-alpha) * cross(m, hspin)           );

    // write back, adding to torque
    tx[i] += torque.x;
    ty[i] += torque.y;
    tz[i] += torque.z;
}
#undef PREFACTOR
#undef deltax
#undef deltay
#undef deltaz
