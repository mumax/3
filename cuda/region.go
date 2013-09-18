package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"unsafe"
)

// dst += LUT[region], for vectors. Used for complex excitation.
func RegionAddV(dst *data.Slice, lut LUTPtrs, regions *Bytes) {
	util.Argument(dst.NComp() == 3)
	N := dst.Len()
	cfg := make1DConf(N)
	k_regionaddv(dst.DevPtr(0), dst.DevPtr(1), dst.DevPtr(2),
		lut[0], lut[1], lut[2], regions.Ptr, N, cfg)
}

// decode the regions+LUT pair into an uncompressed array
func RegionDecode(dst *data.Slice, lut LUTPtr, regions *Bytes) {
	N := dst.Len()
	cfg := make1DConf(N)
	k_regiondecode(dst.DevPtr(0), unsafe.Pointer(lut), regions.Ptr, N, cfg)
}

// select the part of src within the specified region, set 0's everywhere else.
func RegionSelect(dst, src *data.Slice, regions *Bytes, region byte) {
	util.Argument(dst.NComp() == src.NComp())
	N := dst.Len()
	cfg := make1DConf(N)

	str := stream()
	for c := 0; c < dst.NComp(); c++ {
		k_regionselect_async(dst.DevPtr(c), src.DevPtr(c), regions.Ptr, region, N, cfg, str)
	}
	syncAndRecycle(str)
}
