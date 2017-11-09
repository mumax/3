package cuda

// This file provides GPU byte slices, used to store regions.

import (
	"log"
	"unsafe"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/util"
)

// 3D byte slice, used for region lookup.
type Bytes struct {
	Ptr unsafe.Pointer
	Len int
}

// Construct new byte slice with given length,
// initialised to zeros.
func NewBytes(Len int) *Bytes {
	ptr := cu.MemAlloc(int64(2*Len))
	cu.MemsetD16(cu.DevicePtr(ptr), 0, int64(Len))
	return &Bytes{unsafe.Pointer(uintptr(ptr)), Len}
}

// Upload src (host) to dst (gpu).
func (dst *Bytes) Upload(src []uint16) {
	util.Argument(dst.Len == len(src))
	MemCpyHtoD(dst.Ptr, unsafe.Pointer(&src[0]), int64(2*dst.Len))
}

// Copy on device: dst = src.
func (dst *Bytes) Copy(src *Bytes) {
	util.Argument(dst.Len == src.Len)
	MemCpy(dst.Ptr, src.Ptr, int64(2*dst.Len))
}

// Copy to host: dst = src.
func (src *Bytes) Download(dst []uint16) {
	util.Argument(src.Len == len(dst))
	MemCpyDtoH(unsafe.Pointer(&dst[0]), src.Ptr, int64(2*src.Len))
}

// Set one element to value.
// data.Index can be used to find the index for x,y,z.
func (dst *Bytes) Set(index int, value uint16) {
	if index < 0 || index >= dst.Len {
		log.Panic("Bytes.Set: index out of range:", index)
	}
	src := value
	MemCpyHtoD(unsafe.Pointer(uintptr(dst.Ptr)+uintptr(2*index)), unsafe.Pointer(&src), 2)
}

// Get one element.
// data.Index can be used to find the index for x,y,z.
func (src *Bytes) Get(index int) uint16 {
	if index < 0 || index >= src.Len {
		log.Panic("Bytes.Set: index out of range:", index)
	}
	var dst uint16
	MemCpyDtoH(unsafe.Pointer(&dst), unsafe.Pointer(uintptr(src.Ptr)+uintptr(2*index)), 2)
	return dst
}

// Frees the GPU memory and disables the slice.
func (b *Bytes) Free() {
	if b.Ptr != nil {
		cu.MemFree(cu.DevicePtr(uintptr(b.Ptr)))
	}
	b.Ptr = nil
	b.Len = 0
}
