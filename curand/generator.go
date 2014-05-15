package curand

//#include <curand.h>
import "C"

import (
	"unsafe"
)

type Generator uintptr

type RngType int

const (
	PSEUDO_DEFAULT          RngType = C.CURAND_RNG_PSEUDO_DEFAULT          // Default pseudorandom generator
	PSEUDO_XORWOW           RngType = C.CURAND_RNG_PSEUDO_XORWOW           // XORWOW pseudorandom generator
	QUASI_DEFAULT           RngType = C.CURAND_RNG_QUASI_DEFAULT           // Default quasirandom generator
	QUASI_SOBOL32           RngType = C.CURAND_RNG_QUASI_SOBOL32           // Sobol32 quasirandom generator
	QUASI_SCRAMBLED_SOBOL32 RngType = C.CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 // Scrambled Sobol32 quasirandom generator
	QUASI_SOBOL64           RngType = C.CURAND_RNG_QUASI_SOBOL64           // Sobol64 quasirandom generator
	QUASI_SCRAMBLED_SOBOL64 RngType = C.CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 // Scrambled Sobol64 quasirandom generator
)

func CreateGenerator(rngType RngType) Generator {
	var rng C.curandGenerator_t
	err := Status(C.curandCreateGenerator(&rng, C.curandRngType_t(rngType)))
	if err != SUCCESS {
		panic(err)
	}
	return Generator(uintptr(unsafe.Pointer(rng))) // cgo
}

func (g Generator) GenerateNormal(output uintptr, n int64, mean, stddev float32) {
	err := Status(C.curandGenerateNormal(
		C.curandGenerator_t(unsafe.Pointer(uintptr(g))),
		(*C.float)(unsafe.Pointer(output)),
		C.size_t(n),
		C.float(mean),
		C.float(stddev)))
	if err != SUCCESS {
		panic(err)
	}
}

func (g Generator) SetSeed(seed int64) {
	err := Status(C.curandSetPseudoRandomGeneratorSeed(C.curandGenerator_t(unsafe.Pointer(uintptr(g))), _Ctype_ulonglong(seed)))
	if err != SUCCESS {
		panic(err)
	}
}

// Documentation was taken from the curand headers.
