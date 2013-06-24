package cuda

import (
	"code.google.com/p/mx3/data"
	"unsafe"
)

// decode the regions+LUT pair into an uncompressed array
func RegionDecode(dst *data.Slice, lut LUTPtr, regions *Bytes) {
	N := dst.Len()
	cfg := make1DConf(N)
	k_regiondecode(dst.DevPtr(0), unsafe.Pointer(lut), regions.Ptr, N, cfg)
}
