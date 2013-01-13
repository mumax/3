package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/gpu/ptx"
	"github.com/barnex/cuda5/safe"
)

// multiply-add: dst[i] = src1[i] * factor1 + src2[i] * factor2
func Madd2(dst, src1, src2 safe.Float32s, factor1, factor2 float32) {
	core.Assert(dst.Len() == src1.Len() && dst.Len() == src2.Len())
	N := dst.Len()
	gridDim, blockDim := Make1DConf(N)
	ptx.K_madd2(dst.Pointer(), src1.Pointer(), factor1,
		src2.Pointer(), factor2, N, gridDim, blockDim)
}

// multiply-add: dst[i] = src1[i] * factor1 + src2[i] * factor2 + src3 * factor3
func Madd3(dst, src1, src2, src3 safe.Float32s, factor1, factor2, factor3 float32) {
	core.Assert(dst.Len() == src1.Len() && dst.Len() == src2.Len())
	N := dst.Len()
	gridDim, blockDim := Make1DConf(N)
	ptx.K_madd3(dst.Pointer(), src1.Pointer(), factor1,
		src2.Pointer(), factor2, src3.Pointer(), factor3,
		N, gridDim, blockDim)
}
