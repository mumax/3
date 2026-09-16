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
