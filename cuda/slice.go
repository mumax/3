package cuda

import (
	"github.com/barnex/cuda5/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
	"unsafe"
)

// Make a GPU Slice with nComp components each of size length.
func NewSlice(nComp int, m *data.Mesh) *data.Slice {
	return newSlice(nComp, m, MemAlloc, data.GPUMemory)
}

// Make a GPU Slice with nComp components each of size length.
func NewUnifiedSlice(nComp int, m *data.Mesh) *data.Slice {
	return newSlice(nComp, m, cu.MemAllocHost, data.UnifiedMemory)
}

func newSlice(nComp int, m *data.Mesh, alloc func(int64) unsafe.Pointer, memType int8) *data.Slice {
	data.EnableGPU(memFree, cu.MemFreeHost, memCpy, memCpyDtoH, memCpyHtoD)
	length := m.NCell()
	bytes := int64(length) * cu.SIZEOF_FLOAT32
	ptrs := make([]unsafe.Pointer, nComp)
	for c := range ptrs {
		ptrs[c] = unsafe.Pointer(alloc(bytes))
		cu.MemsetD32(cu.DevicePtr(uintptr(ptrs[c])), 0, int64(length))
	}
	return data.SliceFromPtrs(m, memType, ptrs)
}

// wrappers for data.EnableGPU arguments

func memFree(ptr unsafe.Pointer) { cu.MemFree(cu.DevicePtr(uintptr(ptr))) }

func memCpyDtoH(dst, src unsafe.Pointer, bytes int64) {
	cu.MemcpyDtoH(dst, cu.DevicePtr(uintptr(src)), bytes)
}

func memCpyHtoD(dst, src unsafe.Pointer, bytes int64) {
	cu.MemcpyHtoD(cu.DevicePtr(uintptr(dst)), src, bytes)
}

func memCpy(dst, src unsafe.Pointer, bytes int64) {
	str := stream()
	cu.MemcpyAsync(cu.DevicePtr(uintptr(dst)), cu.DevicePtr(uintptr(src)), bytes, str)
	syncAndRecycle(str)
}

// Memset sets the Slice's components to the specified values.
func Memset(s *data.Slice, val ...float32) {
	util.Argument(len(val) == s.NComp())
	str := stream()
	for c, v := range val {
		cu.MemsetD32Async(cu.DevicePtr(uintptr(s.DevPtr(c))), math.Float32bits(v), int64(s.Len()), str)
	}
	syncAndRecycle(str)
}

// Set all elements of all components to zero.
func Zero(s *data.Slice) {
	Memset(s, make([]float32, s.NComp())...)
}

func index(i, j, k int, size [3]int) int {
	util.Argument(i >= 0 && j >= 0 && k >= 0 &&
		i < size[0] && j < size[1] && k < size[2])
	return ((i)*size[1]*size[2] + (j)*size[2] + (k))
}

func SetCell(s *data.Slice, comp int, i, j, k int, value float32) {
	SetElem(s, comp, index(i, j, k, s.Mesh().Size()), value)
}

func SetElem(s *data.Slice, comp int, index int, value float32) {
	f := value
	dst := unsafe.Pointer(uintptr(s.DevPtr(comp)) + uintptr(index)*cu.SIZEOF_FLOAT32)
	memCpyHtoD(dst, unsafe.Pointer(&f), cu.SIZEOF_FLOAT32)
}

func GetElem(s *data.Slice, comp int, index int) float32 {
	var f float32
	src := unsafe.Pointer(uintptr(s.DevPtr(comp)) + uintptr(index)*cu.SIZEOF_FLOAT32)
	memCpyDtoH(unsafe.Pointer(&f), src, cu.SIZEOF_FLOAT32)
	return f
}

func GetCell(s *data.Slice, comp, i, j, k int) float32 {
	return GetElem(s, comp, index(i, j, k, s.Mesh().Size()))
}
