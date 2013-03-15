package cuda

import (
	"code.google.com/p/mx3/data"
)

// Normalize the vector field to length mask * norm.
// nil mask interpreted as 1s.
// 0-length vectors are unaffected.
func Normalize(vec *data.Slice) {
	N := vec.Len()
	cfg := make1DConf(N)
	k_normalize(vec.DevPtr(0), vec.DevPtr(1), vec.DevPtr(2), N, cfg)
}
