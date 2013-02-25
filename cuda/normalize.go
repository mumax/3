package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/kernel"
)

// Normalize the vector field to unit length.
// 0-length vectors are unaffected.
func Normalize(vec *data.Slice, norm float32) {
	N := vec.Len()
	gridDim, blockDim := Make1DConf(N)
	kernel.K_normalize(vec.DevPtr(0), vec.DevPtr(1), vec.DevPtr(2), norm, N, gridDim, blockDim)
}
