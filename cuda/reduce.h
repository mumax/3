#ifndef _REDUCE_H_
#define _REDUCE_H_

// Block size for reduce kernels.
#define REDUCE_BLOCKSIZE 512

// This macro expands to a reduce kernel with arbitrary reduce operation.
// Ugly, perhaps, but arguably nicer than some 1000+ line C++ template.
// load(i): loads element i, possibly pre-processing the data
// op(a, b): reduce operation. e.g. sum
// atomicOp(a, b): atomic reduce operation in global mem.
//
// Two variants of the in-block tree reduction follow. The NVIDIA variant (the
// upstream code, in the #else branch) drops to a warp-synchronous tail once the
// tree reaches 32 elements, relying on implicit 32-lane lockstep. That
// assumption holds on a 32-lane CUDA warp but is invalid on an AMD wave64
// (gfx90a): the low 32 lanes of a 64-lane wavefront are not guaranteed lockstep
// across the unsynchronized "volatile" steps, so the reduction would be wrong
// and non-deterministic. The HIP variant lets the __syncthreads()-synchronized
// tree run all the way down to s>0, which keeps a block-wide barrier between
// every step and is correct on any wavefront width (wave32 and wave64).
#ifdef __HIP_PLATFORM_AMD__
#define reduce(load, op, atomicOp)                      \
    __shared__ float sdata[REDUCE_BLOCKSIZE];           \
    int tid = threadIdx.x;                              \
    int i =  blockIdx.x * blockDim.x + threadIdx.x;     \
                                                        \
    float mine = initVal;                               \
    int stride = gridDim.x * blockDim.x;                \
    while (i < n) {                                     \
        mine = op(mine, load(i));                       \
        i += stride;                                    \
    }                                                   \
    sdata[tid] = mine;                                  \
    __syncthreads();                                    \
                                                        \
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {     \
        if (tid < s){                                   \
            sdata[tid] = op(sdata[tid], sdata[tid + s]);\
        }                                               \
        __syncthreads();                                \
    }                                                   \
                                                        \
    if (tid == 0) { atomicOp(dst, sdata[0]); }          \
// Based on "Optimizing parallel reduction in CUDA" by Mark Harris.
#else
#define reduce(load, op, atomicOp)                      \
    __shared__ float sdata[REDUCE_BLOCKSIZE];           \
    int tid = threadIdx.x;                              \
    int i =  blockIdx.x * blockDim.x + threadIdx.x;     \
                                                        \
    float mine = initVal;                               \
    int stride = gridDim.x * blockDim.x;                \
    while (i < n) {                                     \
        mine = op(mine, load(i));                       \
        i += stride;                                    \
    }                                                   \
    sdata[tid] = mine;                                  \
    __syncthreads();                                    \
                                                        \
    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {    \
        if (tid < s){                                   \
            sdata[tid] = op(sdata[tid], sdata[tid + s]);\
        }                                               \
        __syncthreads();                                \
    }                                                   \
                                                        \
    if (tid < 32) {                                     \
        volatile float* smem = sdata;                   \
        smem[tid] = op(smem[tid], smem[tid + 32]);      \
        smem[tid] = op(smem[tid], smem[tid + 16]);      \
        smem[tid] = op(smem[tid], smem[tid +  8]);      \
        smem[tid] = op(smem[tid], smem[tid +  4]);      \
        smem[tid] = op(smem[tid], smem[tid +  2]);      \
        smem[tid] = op(smem[tid], smem[tid +  1]);      \
    }                                                   \
                                                        \
    if (tid == 0) { atomicOp(dst, sdata[0]); }          \
// Based on "Optimizing parallel reduction in CUDA" by Mark Harris.
#endif
#endif
