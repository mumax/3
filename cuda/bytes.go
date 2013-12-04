package cuda

import (
	"github.com/barnex/cuda5/cu"
	"github.com/mumax/3/util"
	"unsafe"
)

// 3D byte slice, used for region lookup.
type Bytes struct {
	Ptr unsafe.Pointer
	Len int
}

// Construct new 3D byte slice for given mesh.
func NewBytes(Len int) *Bytes {
	ptr := cu.MemAlloc(int64(Len))
	cu.MemsetD8(cu.DevicePtr(ptr), 0, int64(Len))
	return &Bytes{unsafe.Pointer(uintptr(ptr)), Len}
}

// Upload src (host) to dst (gpu)
func (dst *Bytes) Upload(src []byte) {
	util.Argument(dst.Len == len(src))
	cu.MemcpyHtoD(cu.DevicePtr(uintptr(dst.Ptr)), unsafe.Pointer(&src[0]), int64(dst.Len))
}

func (dst *Bytes) Copy(src *Bytes) {
	util.Argument(dst.Len == src.Len)
	cu.MemcpyDtoD(cu.DevicePtr(uintptr(dst.Ptr)), cu.DevicePtr(uintptr(src.Ptr)), int64(dst.Len))
}

func (src *Bytes) Download(dst []byte) {
	util.Argument(src.Len == len(dst))
	cu.MemcpyDtoH(unsafe.Pointer(&dst[0]), cu.DevicePtr(uintptr(src.Ptr)), int64(src.Len))
}

func (dst *Bytes) Set(index int, value byte) {
	src := value
	cu.MemcpyHtoD(cu.DevicePtr(uintptr(dst.Ptr)+uintptr(index)), unsafe.Pointer(&src), 1)
}

// Frees the GPU memory and disables the slice.
func (b *Bytes) Free() {
	if b.Ptr != nil {
		cu.MemFree(cu.DevicePtr(uintptr(b.Ptr)))
	}
	b.Ptr = nil
	b.Len = 0
}
