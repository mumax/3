package cuda

import (
	"github.com/mumax/3/data"
	"unsafe"
)

// Sets vector dst to zero where mask != 0.
func ZeroMask(dst *data.Slice, mask LUTPtr, regions *Bytes) {
	N := dst.Len()
	cfg := make1DConf(N)

	k_zeromask_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		unsafe.Pointer(mask), regions.Ptr, N, cfg)
}
