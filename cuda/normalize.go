package cuda

import (
	"code.google.com/p/mx3/data"
)

// Normalize the vector field to length mask * norm.
// nil mask interpreted as 1s.
// 0-length vectors are unaffected.
func Normalize(vec *data.Slice, mask *data.Slice, norm float32) {
	N := vec.Len()
	cfg := Make1DConf(N)
	k_normalize(vec.DevPtr(0), vec.DevPtr(1), vec.DevPtr(2), mask.DevPtr(0), norm, N, cfg)
}
