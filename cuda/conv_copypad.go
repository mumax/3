package cuda

import (
	"github.com/barnex/cuda5/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"unsafe"
)

// Copies src (larger) into dst (smaller).
// Uses to extract demag field after convolution on padded m.
func copyUnPad(dst, src *data.Slice, dstsize, srcsize [3]int, str cu.Stream) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Argument(dst.Len() == prod(dstsize) && src.Len() == prod(srcsize))

	cfg := make3DConf(dstsize)

	k_copyunpad_async(dst.DevPtr(0), dstsize[0], dstsize[1], dstsize[2],
		src.DevPtr(0), srcsize[0], srcsize[1], srcsize[2], cfg, str)
}

// Copies src into dst, which is larger, and multiplies by vol*Bsat.
// The remainder of dst is not filled with zeros.
func copyPadMul(dst, src, vol *data.Slice, dstsize, srcsize [3]int, Bsat LUTPtr, regions *Bytes, str cu.Stream) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == prod(dstsize) && src.Len() == prod(srcsize))

	cfg := make3DConf(srcsize)

	k_copypadmul_async(dst.DevPtr(0), dstsize[0], dstsize[1], dstsize[2],
		src.DevPtr(0), vol.DevPtr(0), srcsize[0], srcsize[1], srcsize[2],
		unsafe.Pointer(Bsat), regions.Ptr, cfg, str)
}
