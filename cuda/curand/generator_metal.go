//go:build darwin && arm64 && cgo
// +build darwin,arm64,cgo

package curand

import metalrng "github.com/mumax/3/cuda/metal/rng"

func CreateGenerator(rngType RngType) Generator {
	generator, err := metalrng.CreateGenerator(int(rngType))
	if err != nil {
		panic(err)
	}
	return Generator(generator)
}

func (g Generator) GenerateNormal(output uintptr, n int64, mean, stddev float32) {
	if err := metalrng.GenerateNormal(uintptr(g), output, n, mean, stddev); err != nil {
		panic(err)
	}
}

func (g Generator) SetSeed(seed int64) {
	if err := metalrng.SetSeed(uintptr(g), uint64(seed)); err != nil {
		panic(err)
	}
}
