package gpu

import (
	"code.google.com/p/mx3/gpu/ptx"
	"github.com/barnex/cuda5/safe"
)

// Normalize the vector field to unit length.
// 0-length vectors are unaffected.
func Normalize(vec [3]safe.Float32s) {
	N := vec[0].Len()
	gridDim, blockDim := Make1DConf(N)
	ptx.K_normalize(vec[0].Pointer(), vec[1].Pointer(), vec[2].Pointer(), N, gridDim, blockDim)
}
