package safe

import (
	"fmt"
	"github.com/barnex/cuda5/cu"
	"math"
	"unsafe"
)

// Slice of float32's on the GPU.
type Float32s struct{ slice }

// Make a slice of float32's on the GPU.
// Initialized to zero.
func MakeFloat32s(len_ int) Float32s {
	return Float32s{makeslice(len_, cu.SIZEOF_FLOAT32)}
}

// Return a slice from start (inclusive) to stop (exclusive),
// sharing the underlying storage with the original slice.
// Slices obtained in this way should not be Free()'d
func (s Float32s) Slice(start, stop int) Float32s {
	return Float32s{s.slice.slice(start, stop, cu.SIZEOF_FLOAT32)}
}

// Copy src from host to dst on the device.
func (dst Float32s) CopyHtoD(src []float32) {
	dst.copyHtoD(unsafe.Pointer(&src[0]), len(src), cu.SIZEOF_FLOAT32)
}

// Copy src form device to dst on host.
func (src Float32s) CopyDtoH(dst []float32) {
	src.copyDtoH(unsafe.Pointer(&dst[0]), len(dst), cu.SIZEOF_FLOAT32)
}

// Copy src on host to dst on host.
func (dst Float32s) CopyDtoD(src Float32s) {
	dst.copyDtoD(&src.slice, cu.SIZEOF_FLOAT32)
}

// Copy src from host to dst on the device, asynchronously.
func (dst Float32s) CopyHtoDAsync(src []float32, stream cu.Stream) {
	dst.copyHtoDAsync(unsafe.Pointer(&src[0]), len(src), cu.SIZEOF_FLOAT32, stream)
}

// Copy src form device to dst on host, asynchronously.
func (src Float32s) CopyDtoHAsync(dst []float32, stream cu.Stream) {
	src.copyDtoHAsync(unsafe.Pointer(&dst[0]), len(dst), cu.SIZEOF_FLOAT32, stream)
}

// Copy src on host to dst on host, asynchronously.
func (dst Float32s) CopyDtoDAsync(src Float32s, stream cu.Stream) {
	dst.copyDtoDAsync(&src.slice, cu.SIZEOF_FLOAT32, stream)
}

// Returns a fresh copy on host.
func (src Float32s) Host() []float32 {
	cpy := make([]float32, src.Len())
	src.CopyDtoH(cpy)
	return cpy
}

// Set the entire slice to this value.
func (s Float32s) Memset(value float32) {
	cu.MemsetD32(s.Pointer(), math.Float32bits(value), int64(s.Len()))
	cu.CtxSynchronize()
}

// Set the entire slice to this value, asynchronously.
func (s Float32s) MemsetAsync(value float32, stream cu.Stream) {
	cu.MemsetD32Async(s.Pointer(), math.Float32bits(value), int64(s.Len()), stream)
}

// Re-interpret the array as complex numbers,
// in interleaved format. Underlying storage
// is shared.
func (s Float32s) Complex() Complex64s {
	if s.Len()%2 != 0 {
		panic(fmt.Errorf("complex: need even number of elements, have:%v", s.Len()))
	}
	return Complex64s{slice{s.ptr_, s.len_ / 2, s.cap_ / 2}}
}
