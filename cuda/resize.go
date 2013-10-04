package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Select and resize one layer for interactive output
func Resize(dst, src *data.Slice, layer int) {

	dstsize := dst.Mesh().Size()
	srcsize := src.Mesh().Size()
	util.Assert(dstsize[0] == 1)
	util.Assert(dst.NComp() == 1 && src.NComp() == 1)

	scale1 := srcsize[1] / dstsize[1]
	scale2 := srcsize[2] / dstsize[2]
	util.Assert(srcsize[1]%dstsize[1] == 0)
	util.Assert(srcsize[2]%dstsize[2] == 0)

	cfg := make3DConf(dstsize)

	k_resize(dst.DevPtr(0), dstsize[0], dstsize[1], dstsize[2],
		src.DevPtr(0), srcsize[0], srcsize[1], srcsize[2], layer, scale1, scale2, cfg)
}
