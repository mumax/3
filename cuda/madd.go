package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

// multiply-add: dst[i] = src1[i] * factor1 + src2[i] * factor2
func Madd2(dst, src1, src2 *data.Slice, factor1, factor2 float32) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src2.Len() == N)
	util.Assert(src1.NComp() == nComp && src2.NComp() == nComp)
	cfg := make1DConf(N)
	str := stream()
	for c := 0; c < nComp; c++ {
		k_madd2_async(dst.DevPtr(c), src1.DevPtr(c), factor1,
			src2.DevPtr(c), factor2, N, cfg, str)
	}
	syncAndRecycle(str)
}

// multiply-add: dst[i] = src1[i] * factor1 + src2[i] * factor2 + src3 * factor3
func Madd3(dst, src1, src2, src3 *data.Slice, factor1, factor2, factor3 float32) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src2.Len() == N && src3.Len() == N)
	util.Assert(src1.NComp() == nComp && src2.NComp() == nComp && src3.NComp() == nComp)
	cfg := make1DConf(N)
	str := stream()
	for c := 0; c < nComp; c++ {
		k_madd3_async(dst.DevPtr(c), src1.DevPtr(c), factor1,
			src2.DevPtr(c), factor2, src3.DevPtr(c), factor3, N, cfg, str)
	}
	syncAndRecycle(str)
}

//// Adds a constant to each element of the slice.
//// 	dst[comp][index] += cnst[comp]
//func AddConst(dst *data.Slice, cnst ...float32) {
//	util.Argument(len(cnst) == dst.NComp())
//	N := dst.Len()
//	cfg := make1DConf(N)
//	str := stream()
//	for c := 0; c < dst.NComp(); c++ {
//		if cnst[c] != 0 {
//			k_madd2_async(dst.DevPtr(c), dst.DevPtr(c), 1, nil, cnst[c], N, cfg, str)
//		}
//	}
//	syncAndRecycle(str)
//}
