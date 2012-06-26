package safe

import (
	"github.com/barnex/cuda4/cu"
	"unsafe"
)

// Slice of complex128's on the GPU.
type Complex128s struct{ slice }

// Make a slice of complex128's on the GPU.
// Initialized to zero.
func MakeComplex128s(len_ int) Complex128s {
	return Complex128s{makeslice(len_, cu.SIZEOF_COMPLEX128)}
}

// Return a slice from start (inclusive) to stop (exclusive),
// sharing the underlying storage with the original slice.
// Slices obtained in this way should not be Free()'d
func (s Complex128s) Slice(start, stop int) Complex128s {
	return Complex128s{s.slice.slice(start, stop, cu.SIZEOF_COMPLEX128)}
}

// Copy src from host to dst on the device.
func (dst Complex128s) CopyHtoD(src []complex128) {
	dst.copyHtoD(unsafe.Pointer(&src[0]), len(src), cu.SIZEOF_COMPLEX128)
}

// Copy src form device to dst on host.
func (src Complex128s) CopyDtoH(dst []complex128) {
	src.copyDtoH(unsafe.Pointer(&dst[0]), len(dst), cu.SIZEOF_COMPLEX128)
}

// Copy src on host to dst on host.
func (dst Complex128s) CopyDtoD(src Complex128s) {
	dst.copyDtoD(&src.slice, cu.SIZEOF_COMPLEX128)
}

// Copy src from host to dst on the device, asynchronously.
func (dst Complex128s) CopyHtoDAsync(src []complex128, stream cu.Stream) {
	dst.copyHtoDAsync(unsafe.Pointer(&src[0]), len(src), cu.SIZEOF_COMPLEX128, stream)
}

// Copy src form device to dst on host, asynchronously.
func (src Complex128s) CopyDtoHAsync(dst []complex128, stream cu.Stream) {
	src.copyDtoHAsync(unsafe.Pointer(&dst[0]), len(dst), cu.SIZEOF_COMPLEX128, stream)
}

// Copy src on host to dst on host, asynchronously.
func (dst Complex128s) CopyDtoDAsync(src Complex128s, stream cu.Stream) {
	dst.copyDtoDAsync(&src.slice, cu.SIZEOF_COMPLEX128, stream)
}

// Returns a fresh copy on host.
func (src Complex128s) Host() []complex128 {
	cpy := make([]complex128, src.Len())
	src.CopyDtoH(cpy)
	return cpy
}
