package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

// 3D byte slice, used for region lookup
type Bytes struct {
	Ptr  unsafe.Pointer
	mesh *data.Mesh
	Len  int32
}

func NewBytes(m *data.Mesh) *Bytes {
	Len := int64(m.NCell())
	ptr := cu.MemAlloc(Len)
	cu.MemsetD8(cu.DevicePtr(ptr), 0, Len)
	return &Bytes{unsafe.Pointer(ptr), m, int32(Len)}
}

func (dst *Bytes) Upload(src []byte) {
	util.Argument(int(dst.Len) == len(src))
	cu.MemcpyHtoD(cu.DevicePtr(dst.Ptr), unsafe.Pointer(&src[0]), int64(dst.Len))
}

func (b *Bytes) Mesh() *data.Mesh { return b.mesh }

//func (SetCell(s *data.Slice, comp int, i, j, k int, value float32) {
//	SetElem(s, comp, index(i, j, k, s.Mesh().Size()), value)
//}
//
//func SetElem(s *data.Slice, comp int, index int, value float32) {
//	f := value
//	dst := unsafe.Pointer(uintptr(s.DevPtr(comp)) + uintptr(index)*cu.SIZEOF_FLOAT32)
//	memCpyHtoD(dst, unsafe.Pointer(&f), cu.SIZEOF_FLOAT32)
//}
//
//func GetElem(s *data.Slice, comp int, index int) float32 {
//	var f float32
//	src := unsafe.Pointer(uintptr(s.DevPtr(comp)) + uintptr(index)*cu.SIZEOF_FLOAT32)
//	memCpyDtoH(unsafe.Pointer(&f), src, cu.SIZEOF_FLOAT32)
//	return f
//}
//
//func GetCell(s *data.Slice, comp, i, j, k int) float32 {
//	return GetElem(s, comp, index(i, j, k, s.Mesh().Size()))
//}
