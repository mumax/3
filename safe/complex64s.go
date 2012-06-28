package safe

import (
	"github.com/barnex/cuda4/cu"
	"unsafe"
)

// Slice of complex64's on the GPU.
type Complex64s struct{ slice }

// Make a slice of complex64's on the GPU.
// Initialized to zero.
func MakeComplex64s(len_ int) Complex64s {
	return Complex64s{makeslice(len_, cu.SIZEOF_COMPLEX64)}
}

// Return a slice from start (inclusive) to stop (exclusive),
// sharing the underlying storage with the original slice.
// Slices obtained in this way should not be Free()'d
func (s Complex64s) Slice(start, stop int) Complex64s {
	return Complex64s{s.slice.slice(start, stop, cu.SIZEOF_COMPLEX64)}
}

// Copy src from host to dst on the device.
func (dst Complex64s) CopyHtoD(src []complex64) {
	dst.copyHtoD(unsafe.Pointer(&src[0]), len(src), cu.SIZEOF_COMPLEX64)
}

// Copy src form device to dst on host.
func (src Complex64s) CopyDtoH(dst []complex64) {
	src.copyDtoH(unsafe.Pointer(&dst[0]), len(dst), cu.SIZEOF_COMPLEX64)
}

// Copy src on host to dst on host.
func (dst Complex64s) CopyDtoD(src Complex64s) {
	dst.copyDtoD(&src.slice, cu.SIZEOF_COMPLEX64)
}

// Copy src from host to dst on the device, asynchronously.
func (dst Complex64s) CopyHtoDAsync(src []complex64, stream cu.Stream) {
	dst.copyHtoDAsync(unsafe.Pointer(&src[0]), len(src), cu.SIZEOF_COMPLEX64, stream)
}

// Copy src form device to dst on host, asynchronously.
func (src Complex64s) CopyDtoHAsync(dst []complex64, stream cu.Stream) {
	src.copyDtoHAsync(unsafe.Pointer(&dst[0]), len(dst), cu.SIZEOF_COMPLEX64, stream)
}

// Copy src on host to dst on host, asynchronously.
func (dst Complex64s) CopyDtoDAsync(src Complex64s, stream cu.Stream) {
	dst.copyDtoDAsync(&src.slice, cu.SIZEOF_COMPLEX64, stream)
}

// Returns a fresh copy on host.
func (src Complex64s) Host() []complex64 {
	cpy := make([]complex64, src.Len())
	src.CopyDtoH(cpy)
	return cpy
}

// Re-interpret the array as float numbers,
// in interleaved format. Underlying storage
// is shared.
func (s Complex64s) Float() Float32s {
	return Float32s{slice{s.ptr_, s.len_ * 2, s.cap_ * 2}}
}
