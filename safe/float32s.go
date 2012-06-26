package safe

import "unsafe"

type Float32s struct{ slice }

func MakeFloat32s(len_ int) Float32s {
	return Float32s{makeslice(len_, sizeofFloat32)}
}

func (s Float32s) Slice(start, stop int) Float32s {
	return Float32s{s.slice.slice(start, stop, sizeofFloat32)}
}

func (dst Float32s) CopyHtoD(src []float32) {
	dst.copyHtoD(unsafe.Pointer(&src[0]), len(src), sizeofFloat32)
}

func (src Float32s) CopyDtoH(dst []float32) {
	src.copyDtoH(unsafe.Pointer(&dst[0]), len(dst), sizeofFloat32)
}

func (dst Float32s) CopyDtoD(src Float32s) {
	dst.copyDtoD(&src.slice, sizeofFloat32)
}

const sizeofFloat32 = 4
