package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

// dst += LUT[region], for vectors
func RegionAddV(dst *data.Slice, lut LUTPtrs, regions *Bytes) {
	util.Argument(dst.NComp() == 3)
	N := dst.Len()
	cfg := make1DConf(N)
	k_regionaddv(dst.DevPtr(0), dst.DevPtr(1), dst.DevPtr(2),
		lut[0], lut[1], lut[2], regions.Ptr, N, cfg)
}
