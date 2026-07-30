// Package kernels exposes the generated MuMax3 Metal shader source.
package kernels

import _ "embed"

// Source is the complete, deterministic Metal Shading Language translation
// of the audited production CUDA kernels.
//
//go:embed mumax3_kernels.metal
var Source string
