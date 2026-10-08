/*
 * Minimal C-compatible shim for the hiprand/rocrand functions used by mumax3.
 * Avoids including hiprand.h directly in cgo preambles, which pulls in
 * hip_runtime.h and rocrand.h together in C mode -- rocrand.h's C fallback
 * block redefines uint4 that hip_vector_types.h already defined, causing a
 * "conflicting types" error on newer SDKs (TheRock 7.14+).
 *
 * Only the declarations actually used by generator.go / status.go are listed.
 */

#ifndef MUMAX3_CURAND_SHIM_H
#define MUMAX3_CURAND_SHIM_H

#include <stddef.h>    /* size_t */
#include <stdint.h>    /* uint32_t, unsigned long long */

/* Opaque generator handle */
typedef struct hiprandGenerator_st* hiprandGenerator_t;

/* RNG type enum -- only values used by mumax3 are listed */
typedef enum {
    HIPRAND_RNG_PSEUDO_DEFAULT          = 400,
    HIPRAND_RNG_PSEUDO_XORWOW           = 401,
    HIPRAND_RNG_QUASI_DEFAULT           = 500,
    HIPRAND_RNG_QUASI_SOBOL32           = 501,
    HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 502,
    HIPRAND_RNG_QUASI_SOBOL64           = 503,
    HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 504
} hiprandRngType_t;

/* Status enum -- all values used by mumax3 */
typedef enum {
    HIPRAND_STATUS_SUCCESS               = 0,
    HIPRAND_STATUS_VERSION_MISMATCH      = 100,
    HIPRAND_STATUS_NOT_INITIALIZED       = 101,
    HIPRAND_STATUS_ALLOCATION_FAILED     = 102,
    HIPRAND_STATUS_TYPE_ERROR            = 103,
    HIPRAND_STATUS_OUT_OF_RANGE          = 104,
    HIPRAND_STATUS_LENGTH_NOT_MULTIPLE   = 105,
    HIPRAND_STATUS_LAUNCH_FAILURE        = 201,
    HIPRAND_STATUS_PREEXISTING_FAILURE   = 202,
    HIPRAND_STATUS_INITIALIZATION_FAILED = 203,
    HIPRAND_STATUS_ARCH_MISMATCH         = 204,
    HIPRAND_STATUS_INTERNAL_ERROR        = 999
} hiprandStatus_t;

/* Function declarations used by mumax3 */
#ifdef __cplusplus
extern "C" {
#endif

hiprandStatus_t hiprandCreateGenerator(hiprandGenerator_t* generator,
                                       hiprandRngType_t rng_type);

hiprandStatus_t hiprandGenerateNormal(hiprandGenerator_t generator,
                                      float* output_data,
                                      size_t n,
                                      float mean,
                                      float stddev);

hiprandStatus_t hiprandSetPseudoRandomGeneratorSeed(hiprandGenerator_t generator,
                                                    unsigned long long seed);

#ifdef __cplusplus
}
#endif

#endif /* MUMAX3_CURAND_SHIM_H */
