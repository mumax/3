package mx

// File: Slice stores N-component GPU or host data.
// Author: Arne Vansteenkiste

import (
	"fmt"
	"github.com/barnex/cuda5/cu"
	"math"
	"unsafe"
)

const MAX_COMP = 3 // Maximum supported number of Slice components

// Number of components
const (
	SCALAR = 1
	VECTOR = 3
)

// Slice is like a [][]float32, but may be stored in GPU or host memory.
type Slice struct {
	ptr     [MAX_COMP]unsafe.Pointer
	len_    int32
	nComp   int8
	memType int8
}

// Make a GPU Slice with nComp components each of size length.
func MakeSlice(nComp, length int) Slice {
	return gpuSlice(nComp, length)
}

// alloc slice on gpu
func gpuSlice(nComp, length int) Slice {
	Argument(nComp > 0 && length > 0)
	bytes := int64(length) * cu.SIZEOF_FLOAT32
	var ptrs [MAX_COMP]unsafe.Pointer
	for c := 0; c < nComp; c++ {
		ptrs[c] = unsafe.Pointer(cu.MemAlloc(bytes))
	}
	s := Slice{ptrs, int32(length), int8(nComp), gpuMemory}
	s.Memset(make([]float32, nComp)...)
	return s
}

// Frees the underlying storage and zeros the Slice header to avoid accidental use.
// Slices sharing storage will be invalid after Free.
func (s *Slice) Free() {
	if s.memType == gpuMemory {
		for c := 0; c < s.NComp(); c++ {
			cu.MemFree(cu.DevicePtr(s.ptr[c]))
			s.ptr[c] = unsafe.Pointer(uintptr(0))
		}
	} else {
		panic("todo")
	}

	s.len_ = 0
	s.nComp = 0
	s.memType = 0
}

// value for Slice.memType
const (
	cpuMemory     = 1 << 0
	gpuMemory     = 1 << 1
	unifiedMemory = cpuMemory | gpuMemory
)

// GPUAccess returns whether the Slice is accessible by the GPU.
// true means it is either stored on GPU or in unified host memory.
func (s *Slice) GPUAccess() bool {
	return s.memType&gpuMemory != 0
}

// CPUAccess returns whether the Slice is accessible by the CPU.
// true means it is stored in host memory.
func (s *Slice) CPUAccess() bool {
	return s.memType&cpuMemory != 0
}

// NComp returns the number of components.
func (s *Slice) NComp() int {
	return int(s.nComp)
}

// Len returns the number of elements per component.
func (s *Slice) Len() int {
	return int(s.len_)
}

// Bytes returns the number of storage bytes per component.
func (s *Slice) bytes() int64 {
	return int64(s.len_) * cu.SIZEOF_FLOAT32
}

// Comp returns a single component of the Slice.
func (s *Slice) Comp(i int) Slice {
	return Slice{[MAX_COMP]unsafe.Pointer{s.ptr[i]}, s.len_, 1, s.memType}
}

// DevPtr returns a CUDA device pointer to a component.
// Slice must have GPUAccess.
func (s *Slice) DevPtr(component int) cu.DevicePtr {
	if !s.GPUAccess() {
		panic("slice not accessible by GPU")
	}
	return cu.DevicePtr(s.ptr[:s.nComp][component])
}

// Slice returns a slice sharing memory with the original.
func (s *Slice) Slice(a, b int) Slice {
	len_ := int(s.len_)
	if a >= len_ || b > len_ || a > b || a < 0 || b < 0 {
		panic(fmt.Errorf("slice range out of bounds: [%v:%v] (len=%v)", a, b, len_))
	}
	var slice Slice
	for i := range s.ptr {
		s.ptr[i] = unsafe.Pointer(uintptr(s.ptr[i]) + cu.SIZEOF_FLOAT32*uintptr(a))
	}
	slice.len_ = int32(b - a)
	slice.nComp = s.nComp
	return slice
}

// Set the entire slice to this value.
func (s *Slice) Memset(val ...float32) {
	Argument(len(val) == s.NComp())
	str := Stream()
	for c, v := range val {
		cu.MemsetD32Async(s.DevPtr(c), math.Float32bits(v), int64(s.Len()), str)
	}
	SyncAndRecycle(str)
}

//func unifiedSlice(N int) Slice {
//	bytes := int64(N) * SizeofFloat32
//	ptr := unsafe.Pointer(cu.MemAllocHost(bytes))
//	ptrs := [MAX_COMP]unsafe.Pointer{ptr}
//	return Slice{ptrs, N, 1, UnifiedMemory}
//}
//
//func ToSlice(list []float32) Slice {
//	ptr := unsafe.Pointer(&list[0])
//	ptrs := [MAX_COMP]unsafe.Pointer{ptr}
//	return Slice{ptrs, len(list), 1, CPUMemory}
//}
//
//func UnsafeSlice(ptr unsafe.Pointer, len_ int, flag MemType) Slice {
//	if len_ < 0 {
//		panic("negative slice length")
//	}
//	ptrs := [MAX_COMP]unsafe.Pointer{ptr}
//	return Slice{ptrs, len_, 1, flag}
//}

// TODO: rm
//func (s *Slice) Safe(component int) safe.Float32s {
//	var f safe.Float32s
//	f.UnsafeSet(unsafe.Pointer(s.DevPtr(component)), s.Len(), s.Len())
//	return f
//}

//func (s *Slice) Host() []float32 {
//	if s.nComp != 1 {
//		panic("need to implement for components")
//	}
//
//	if s.MemType&CPUMemory == 0 {
//		core.Panicf("slice not accessible by CPU (memory type %v)", s.MemType)
//	}
//	var list []float32
//	hdr := (*reflect.SliceHeader)(unsafe.Pointer(&list))
//	hdr.Data = uintptr(s.ptr[0])
//	hdr.Len = int(s.len_)
//	hdr.Cap = hdr.Len
//	return list
//}
//
//func (s *Slice) Device() safe.Float32s {
//	if s.nComp != 1 {
//		panic(fmt.Errorf("slice.device: need 1 component, have %v", s.nComp))
//	}
//	if s.MemType&GPUMemory == 0 {
//		panic(fmt.Errorf("slice not accessible by GPU (memory type %v)", s.MemType))
//	}
//	var floats safe.Float32s
//	floats.UnsafeSet(s.ptr[0], s.Len(), s.Len())
//	return floats
//}
