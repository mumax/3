package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

// multiply: dst[i] = src[i] * factor
func Mul(dst, src *data.Slice, factor float32) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src.Len() == N && src.NComp() == nComp)
	cfg := make1DConf(N)
	str := stream()
	for c := 0; c < nComp; c++ {
		k_mul_async(dst.DevPtr(c), src.DevPtr(c), factor, N, cfg, str)
	}
	syncAndRecycle(str)
}
