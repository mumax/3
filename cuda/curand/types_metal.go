//go:build darwin && arm64
// +build darwin,arm64

package curand

type Generator uintptr

type RngType int

const (
	PSEUDO_DEFAULT          RngType = 100
	PSEUDO_XORWOW           RngType = 101
	QUASI_DEFAULT           RngType = 200
	QUASI_SOBOL32           RngType = 201
	QUASI_SCRAMBLED_SOBOL32 RngType = 202
	QUASI_SOBOL64           RngType = 203
	QUASI_SCRAMBLED_SOBOL64 RngType = 204
)
