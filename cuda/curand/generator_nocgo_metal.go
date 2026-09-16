//go:build darwin && arm64 && !cgo
// +build darwin,arm64,!cgo

package curand

const metalCgoRequired = "metal curand: Apple Silicon RNG requires CGO_ENABLED=1"

func CreateGenerator(rngType RngType) Generator {
	panic(metalCgoRequired)
}

func (g Generator) GenerateNormal(output uintptr, n int64, mean, stddev float32) {
	panic(metalCgoRequired)
}

func (g Generator) SetSeed(seed int64) {
	panic(metalCgoRequired)
}
