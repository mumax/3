package safe

import "unsafe"

// Slice of complex64's on the GPU.
type Complex64s struct{ slice }

// Make a slice of complex64's on the GPU.
// Initialized to zero.
func MakeComplex64s(len_ int) Complex64s {
	return Complex64s{makeslice(len_, sizeofComplex64)}
}

func (s Complex64s) Slice(start, stop int) Complex64s {
	return Complex64s{s.slice.slice(start, stop, sizeofComplex64)}
}

func (dst Complex64s) CopyHtoD(src []complex64) {
	dst.copyHtoD(unsafe.Pointer(&src[0]), len(src), sizeofComplex64)
}

func (src Complex64s) CopyDtoH(dst []complex64) {
	src.copyDtoH(unsafe.Pointer(&dst[0]), len(dst), sizeofComplex64)
}

func (dst Complex64s) CopyDtoD(src Complex64s) {
	dst.copyDtoD(&src.slice, sizeofComplex64)
}

// Returns a fresh copy on host.
func (src Complex64s) Host() []complex64 {
	cpy := make([]complex64, src.Len())
	src.CopyDtoH(cpy)
	return cpy
}

const sizeofComplex64 = 8
