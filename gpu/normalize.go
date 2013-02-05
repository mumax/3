package gpu

import (
	"code.google.com/p/mx3/gpu/ptx"
	"code.google.com/p/mx3/nimble"
)

// Normalize the vector field to unit length.
// 0-length vectors are unaffected.
func Normalize(vec nimble.Slice) {
	N := vec.Len()
	gridDim, blockDim := Make1DConf(N)
	ptx.K_normalize(vec.DevPtr(0), vec.DevPtr(1), vec.DevPtr(2), N, gridDim, blockDim)
}
