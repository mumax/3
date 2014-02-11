package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func crop(dst, src *data.Slice, dstsize, offX, offY, offZ int) {
	D := dst.Size()
	S := src.Size()
	util.Argument(dst.NComp() == src.NComp())
	util.Argument(D[X]+offX <= S[X] && D[Y]+offY <= S[Y] && D[Z]+offZ <= S[Z])

	cfg := make3DConf(D)

	for c := 0; c < dst.NComp(); c++ {
		k_crop_async(dst.DevPtr(c), D[X], D[Y], D[Z],
			src.DevPtr(c), S[X], S[Y], S[Z],
			offX, offY, offZ, cfg)
	}
}
