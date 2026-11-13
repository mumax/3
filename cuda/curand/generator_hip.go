//go:build hip

package curand

// hiprand.h is C++-only: in C mode it includes both hip_runtime.h (which
// defines uint4 via hip_vector_types.h) and rocrand.h (which in its C fallback
// redefines uint4 as a plain struct), causing conflicting-types errors on
// newer SDK builds. Use a minimal shim that declares only what mumax3 needs.

//#include "curand_shim.h"
import "C"

import (
	"unsafe"
)

type Generator uintptr

type RngType int

const (
	PSEUDO_DEFAULT          RngType = C.HIPRAND_RNG_PSEUDO_DEFAULT          // Default pseudorandom generator
	PSEUDO_XORWOW           RngType = C.HIPRAND_RNG_PSEUDO_XORWOW           // XORWOW pseudorandom generator
	QUASI_DEFAULT           RngType = C.HIPRAND_RNG_QUASI_DEFAULT           // Default quasirandom generator
	QUASI_SOBOL32           RngType = C.HIPRAND_RNG_QUASI_SOBOL32           // Sobol32 quasirandom generator
	QUASI_SCRAMBLED_SOBOL32 RngType = C.HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32 // Scrambled Sobol32 quasirandom generator
	QUASI_SOBOL64           RngType = C.HIPRAND_RNG_QUASI_SOBOL64           // Sobol64 quasirandom generator
	QUASI_SCRAMBLED_SOBOL64 RngType = C.HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64 // Scrambled Sobol64 quasirandom generator
)

func CreateGenerator(rngType RngType) Generator {
	var rng C.hiprandGenerator_t
	err := Status(C.hiprandCreateGenerator(&rng, C.hiprandRngType_t(rngType)))
	if err != SUCCESS {
		panic(err)
	}
	return Generator(uintptr(unsafe.Pointer(rng))) // cgo
}

func (g Generator) GenerateNormal(output uintptr, n int64, mean, stddev float32) {
	err := Status(C.hiprandGenerateNormal(
		C.hiprandGenerator_t(unsafe.Pointer(uintptr(g))),
		(*C.float)(unsafe.Pointer(output)),
		C.size_t(n),
		C.float(mean),
		C.float(stddev)))
	if err != SUCCESS {
		panic(err)
	}
}

func (g Generator) SetSeed(seed int64) {
	err := Status(C.hiprandSetPseudoRandomGeneratorSeed(C.hiprandGenerator_t(unsafe.Pointer(uintptr(g))), C.ulonglong(seed)))
	if err != SUCCESS {
		panic(err)
	}
}

// Documentation was taken from the hipRAND headers.
