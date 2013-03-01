package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"math"
	"unsafe"
)

// Make a GPU Slice with nComp components each of size length.
func NewSlice(nComp int, m *data.Mesh) *data.Slice {
	return newSlice(nComp, m, memAlloc, data.GPUMemory)
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
		cu.MemsetD32(cu.DevicePtr(ptrs[c]), 0, int64(length))
	}
	return data.SliceFromPtrs(m, memType, ptrs)
}

// wrappers for data.EnableGPU arguments

func memFree(ptr unsafe.Pointer) { cu.MemFree(cu.DevicePtr(ptr)) }

func memCpyDtoH(dst, src unsafe.Pointer, bytes int64) { cu.MemcpyDtoH(dst, cu.DevicePtr(src), bytes) }

func memCpyHtoD(dst, src unsafe.Pointer, bytes int64) { cu.MemcpyHtoD(cu.DevicePtr(dst), src, bytes) }

func memCpy(dst, src unsafe.Pointer, bytes int64) {
	str := Stream()
	cu.MemcpyAsync(cu.DevicePtr(dst), cu.DevicePtr(src), bytes, str)
	SyncAndRecycle(str)
}

// Memset sets the Slice's components to the specified values.
func Memset(s *data.Slice, val ...float32) {
	util.Argument(len(val) == s.NComp())
	str := Stream()
	for c, v := range val {
		cu.MemsetD32Async(cu.DevicePtr(s.DevPtr(c)), math.Float32bits(v), int64(s.Len()), str)
	}
	SyncAndRecycle(str)
}
