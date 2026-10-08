#ifndef _CU_COMPLEX_COMPAT_H_
#define _CU_COMPLEX_COMPAT_H_

// CUDA-to-HIP compat shim for cuComplex.h. The kernels stay CUDA-spelled and
// #include <cuComplex.h>; on ROCm this header (found ahead of any CUDA
// install via the kernel build's include path) maps the cuComplex names to
// their hipComplex equivalents (a 1:1 correspondence). On NVIDIA the real
// cuComplex.h is used and this file is never seen.

#include <hip/hip_complex.h>

typedef hipDoubleComplex cuDoubleComplex;
typedef hipFloatComplex  cuComplex;
typedef hipFloatComplex  cuFloatComplex;

#define make_cuDoubleComplex make_hipDoubleComplex
#define make_cuFloatComplex  make_hipFloatComplex

#define cuCreal  hipCreal
#define cuCimag  hipCimag
#define cuCadd   hipCadd
#define cuCsub   hipCsub
#define cuCmul   hipCmul
#define cuCdiv   hipCdiv
#define cuConj   hipConj

#define cuCrealf hipCrealf
#define cuCimagf hipCimagf
#define cuCaddf  hipCaddf
#define cuCsubf  hipCsubf
#define cuCmulf  hipCmulf
#define cuCdivf  hipCdivf
#define cuConjf  hipConjf

#endif
