package cuda

import (
	"math"
	"unsafe"

	"github.com/mumax/3/v3/cuda/cu"
	"github.com/mumax/3/v3/data"
	"github.com/mumax/3/v3/timer"
	"github.com/mumax/3/v3/util"
)

// Make a GPU Slice with nComp components each of size length.
func NewSlice(nComp int, size [3]int) *data.Slice {
	return newSlice(nComp, size, MemAlloc, data.GPUMemory)
}

// Make a GPU Slice with nComp components each of size length.
//func NewUnifiedSlice(nComp int, m *data.Mesh) *data.Slice {
//	return newSlice(nComp, m, cu.MemAllocHost, data.UnifiedMemory)
//}

func newSlice(nComp int, size [3]int, alloc func(int64) unsafe.Pointer, memType int8) *data.Slice {
	data.EnableGPU(memFree, cu.MemFreeHost, MemCpy, MemCpyDtoH, MemCpyHtoD)
	length := prod(size)
	bytes := int64(length) * cu.SIZEOF_FLOAT32
	ptrs := make([]unsafe.Pointer, nComp)
	for c := range ptrs {
		ptrs[c] = unsafe.Pointer(alloc(bytes))
		cu.MemsetD32(cu.DevicePtr(uintptr(ptrs[c])), 0, int64(length))
	}
	return data.SliceFromPtrs(size, memType, ptrs)
}

// wrappers for data.EnableGPU arguments

func memFree(ptr unsafe.Pointer) { cu.MemFree(cu.DevicePtr(uintptr(ptr))) }

func MemCpyDtoH(dst, src unsafe.Pointer, bytes int64) {
	Sync() // sync previous kernels
	timer.Start("memcpyDtoH")
	cu.MemcpyDtoH(dst, cu.DevicePtr(uintptr(src)), bytes)
	Sync() // sync copy
	timer.Stop("memcpyDtoH")
}

func MemCpyHtoD(dst, src unsafe.Pointer, bytes int64) {
	Sync() // sync previous kernels
	timer.Start("memcpyHtoD")
	cu.MemcpyHtoD(cu.DevicePtr(uintptr(dst)), src, bytes)
	Sync() // sync copy
	timer.Stop("memcpyHtoD")
}

func MemCpy(dst, src unsafe.Pointer, bytes int64) {
	Sync()
	timer.Start("memcpy")
	cu.MemcpyAsync(cu.DevicePtr(uintptr(dst)), cu.DevicePtr(uintptr(src)), bytes, stream0)
	Sync()
	timer.Stop("memcpy")
}

// Memset sets the Slice's components to the specified values.
// To be carefully used on unified slice (need sync)
func Memset(s *data.Slice, val ...float32) {
	if Synchronous { // debug
		Sync()
		timer.Start("memset")
	}
	util.Argument(len(val) == s.NComp())
	for c, v := range val {
		cu.MemsetD32Async(cu.DevicePtr(uintptr(s.DevPtr(c))), math.Float32bits(v), int64(s.Len()), stream0)
	}
	if Synchronous { //debug
		Sync()
		timer.Stop("memset")
	}
}

// Set all elements of all components to zero.
func Zero(s *data.Slice) {
	Memset(s, make([]float32, s.NComp())...)
}

func SetCell(s *data.Slice, comp int, ix, iy, iz int, value float32) {
	SetElem(s, comp, s.Index(ix, iy, iz), value)
}

func SetElem(s *data.Slice, comp int, index int, value float32) {
	f := value
	dst := unsafe.Pointer(uintptr(s.DevPtr(comp)) + uintptr(index)*cu.SIZEOF_FLOAT32)
	MemCpyHtoD(dst, unsafe.Pointer(&f), cu.SIZEOF_FLOAT32)
}

func GetElem(s *data.Slice, comp int, index int) float32 {
	var f float32
	src := unsafe.Pointer(uintptr(s.DevPtr(comp)) + uintptr(index)*cu.SIZEOF_FLOAT32)
	MemCpyDtoH(unsafe.Pointer(&f), src, cu.SIZEOF_FLOAT32)
	return f
}

func GetCell(s *data.Slice, comp, ix, iy, iz int) float32 {
	return GetElem(s, comp, s.Index(ix, iy, iz))
}
