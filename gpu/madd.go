package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/gpu/ptx"
	"code.google.com/p/mx3/nimble"
)

// multiply-add: dst[i] = src1[i] * factor1 + src2[i] * factor2
func Madd2(dst, src1, src2 nimble.Slice, factor1, factor2 float32) {
	N := dst.Len()
	nComp := dst.NComp()
	core.Assert(src1.Len() == N && src2.Len() == N)
	core.Assert(src1.NComp() == nComp && src2.NComp() == nComp)
	gridDim, blockDim := Make1DConf(N)
	for c := 0; c < nComp; c++ {
		ptx.K_madd2(dst.DevPtr(c), src1.DevPtr(c), factor1,
			src2.DevPtr(c), factor2, N, gridDim, blockDim)
	}
}

// multiply-add: dst[i] = src1[i] * factor1 + src2[i] * factor2 + src3 * factor3
func Madd3(dst, src1, src2, src3 nimble.Slice, factor1, factor2, factor3 float32) {
	N := dst.Len()
	nComp := dst.NComp()
	core.Assert(src1.Len() == N && src2.Len() == N && src3.Len() == N)
	gridDim, blockDim := Make1DConf(N)
	for c := 0; c < nComp; c++ {
		ptx.K_madd3(dst.DevPtr(c), src1.DevPtr(c), factor1,
			src2.DevPtr(c), factor2, src3.DevPtr(c), factor3,
			N, gridDim, blockDim)
	}
}
