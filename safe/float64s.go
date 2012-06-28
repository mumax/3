package safe

import (
	"fmt"
	"github.com/barnex/cuda4/cu"
	"unsafe"
)

// Slice of float64's on the GPU.
type Float64s struct{ slice }

// Make a slice of float64's on the GPU.
// Initialized to zero.
func MakeFloat64s(len_ int) Float64s {
	return Float64s{makeslice(len_, cu.SIZEOF_FLOAT64)}
}

// Return a slice from start (inclusive) to stop (exclusive),
// sharing the underlying storage with the original slice.
// Slices obtained in this way should not be Free()'d
func (s Float64s) Slice(start, stop int) Float64s {
	return Float64s{s.slice.slice(start, stop, cu.SIZEOF_FLOAT64)}
}

// Copy src from host to dst on the device.
func (dst Float64s) CopyHtoD(src []float64) {
	dst.copyHtoD(unsafe.Pointer(&src[0]), len(src), cu.SIZEOF_FLOAT64)
}

// Copy src form device to dst on host.
func (src Float64s) CopyDtoH(dst []float64) {
	src.copyDtoH(unsafe.Pointer(&dst[0]), len(dst), cu.SIZEOF_FLOAT64)
}

// Copy src on host to dst on host.
func (dst Float64s) CopyDtoD(src Float64s) {
	dst.copyDtoD(&src.slice, cu.SIZEOF_FLOAT64)
}

// Copy src from host to dst on the device, asynchronously.
func (dst Float64s) CopyHtoDAsync(src []float64, stream cu.Stream) {
	dst.copyHtoDAsync(unsafe.Pointer(&src[0]), len(src), cu.SIZEOF_FLOAT64, stream)
}

// Copy src form device to dst on host, asynchronously.
func (src Float64s) CopyDtoHAsync(dst []float64, stream cu.Stream) {
	src.copyDtoHAsync(unsafe.Pointer(&dst[0]), len(dst), cu.SIZEOF_FLOAT64, stream)
}

// Copy src on host to dst on host, asynchronously.
func (dst Float64s) CopyDtoDAsync(src Float64s, stream cu.Stream) {
	dst.copyDtoDAsync(&src.slice, cu.SIZEOF_FLOAT64, stream)
}

// Returns a fresh copy on host.
func (src Float64s) Host() []float64 {
	cpy := make([]float64, src.Len())
	src.CopyDtoH(cpy)
	return cpy
}

// Re-interpret the array as complex numbers,
// in interleaved format. Underlying storage
// is shared.
func (s Float64s) Complex() Complex128s {
	if s.Len()%2 != 0 {
		panic(fmt.Errorf("complex: need even number of elements, have:%v", s.Len()))
	}
	return Complex128s{slice{s.ptr_, s.len_ / 2, s.cap_ / 2}}
}
