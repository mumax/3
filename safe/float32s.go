package safe

import (
	"github.com/barnex/cuda4/cu"
	"unsafe"
)

// Slice of float32's on the GPU.
type Float32s struct{ slice }

// Make a slice of float32's on the GPU.
// Initialized to zero.
func MakeFloat32s(len_ int) Float32s {
	return Float32s{makeslice(len_, sizeofFloat32)}
}

// Return a slice from start (inclusive) to stop (exclusive),
// sharing the underlying storage with the original slice.
// Slices obtained in this way should not be Free()'d
func (s Float32s) Slice(start, stop int) Float32s {
	return Float32s{s.slice.slice(start, stop, sizeofFloat32)}
}

// Copy src from host to dst on the device.
func (dst Float32s) CopyHtoD(src []float32) {
	dst.copyHtoD(unsafe.Pointer(&src[0]), len(src), sizeofFloat32)
}

// Copy src form device to dst on host.
func (src Float32s) CopyDtoH(dst []float32) {
	src.copyDtoH(unsafe.Pointer(&dst[0]), len(dst), sizeofFloat32)
}

// Copy src on host to dst on host.
func (dst Float32s) CopyDtoD(src Float32s) {
	dst.copyDtoD(&src.slice, sizeofFloat32)
}

// Copy src from host to dst on the device, asynchronously.
func (dst Float32s) CopyHtoDAsync(src []float32, stream cu.Stream) {
	dst.copyHtoDAsync(unsafe.Pointer(&src[0]), len(src), sizeofFloat32, stream)
}

// Copy src form device to dst on host, asynchronously.
func (src Float32s) CopyDtoHAsync(dst []float32, stream cu.Stream) {
	src.copyDtoHAsync(unsafe.Pointer(&dst[0]), len(dst), sizeofFloat32, stream)
}

// Copy src on host to dst on host, asynchronously.
func (dst Float32s) CopyDtoDAsync(src Float32s, stream cu.Stream) {
	dst.copyDtoDAsync(&src.slice, sizeofFloat32, stream)
}

// Returns a fresh copy on host.
func (src Float32s) Host() []float32 {
	cpy := make([]float32, src.Len())
	src.CopyDtoH(cpy)
	return cpy
}

const sizeofFloat32 = 4
