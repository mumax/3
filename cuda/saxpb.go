package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

// dst[i] = src[i] * factor + cnst.
// TODO: remove? Used only by FFTM
func Saxpb(dst, src *data.Slice, factor, cnst float32) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src.Len() == N && src.NComp() == nComp)
	cfg := make1DConf(N)
	str := stream()
	for c := 0; c < nComp; c++ {
		k_saxpb_async(dst.DevPtr(c), src.DevPtr(c), factor, cnst, N, cfg, str)
	}
	syncAndRecycle(str)
}
