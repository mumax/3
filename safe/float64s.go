package safe

import "unsafe"

type Float64s struct{ slice }

func MakeFloat64s(len_ int) Float64s {
	return Float64s{makeslice(len_, sizeofFloat64)}
}

func (s Float64s) Slice(start, stop int) Float64s {
	return Float64s{s.slice.slice(start, stop, sizeofFloat64)}
}

func (dst Float64s) CopyHtoD(src []float64) {
	dst.copyHtoD(unsafe.Pointer(&src[0]), len(src), sizeofFloat64)
}

func (src Float64s) CopyDtoH(dst []float64) {
	src.copyDtoH(unsafe.Pointer(&dst[0]), len(dst), sizeofFloat64)
}

func (dst Float64s) CopyDtoD(src Float64s) {
	dst.copyDtoD(&src.slice, sizeofFloat64)
}

// Returns a fresh copy on host.
func (src Float64s) Host() []float64 {
	cpy := make([]float64, src.Len())
	src.CopyDtoH(cpy)
	return cpy
}

const sizeofFloat64 = 8
