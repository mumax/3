package mx

// File: Slice stores N-component GPU or host data.
// Author: Arne Vansteenkiste

import (
	"code.google.com/p/mx3/streams"
	"github.com/barnex/cuda5/cu"
	"math"
	"unsafe"
)

// Slice is like a [][]float32, but may be stored in GPU or host memory.
type Slice struct {
	ptr_    [MAX_COMP]unsafe.Pointer // keeps data local
	ptrs    []unsafe.Pointer         // points into ptr_
	len_    int32
	memType int8
}

// Make a GPU Slice with nComp components each of size length.
func NewSlice(nComp, length int) *Slice {
	s := newSlice(nComp, length)
	bytes := int64(length) * cu.SIZEOF_FLOAT32
	for c := range s.ptrs {
		s.ptrs[c] = unsafe.Pointer(MemAlloc(bytes))
	}
	s.memType = gpuMemory
	s.Memset(make([]float32, nComp)...)
	return s
}

// Make a GPU Slice with nComp components each of size length.
func NewUnifiedSlice(nComp, length int) *Slice {
	s := newSlice(nComp, length)
	bytes := int64(length) * cu.SIZEOF_FLOAT32
	for c := range s.ptrs {
		s.ptrs[c] = cu.MemAllocHost(bytes)
	}
	s.memType = unifiedMemory
	s.Memset(make([]float32, nComp)...)
	return s
}

func newSlice(nComp, length int) *Slice {
	Argument(nComp > 0 && length > 0)
	s := new(Slice)
	s.ptrs = s.ptr_[:nComp]
	s.len_ = int32(length)
	return s
}

//func NewUnifiedSlice(nComp, length int)*Slice{
//	Argument(nComp > 0 && length > 0)
//	bytes := int64(length) * cu.SIZEOF_FLOAT32
//}
////	ptr := unsafe.Pointer(cu.MemAllocHost(bytes))
////	ptrs := [MAX_COMP]unsafe.Pointer{ptr}
////	return Slice{ptrs, N, 1, UnifiedMemory}
////}

const MAX_COMP = 3 // Maximum supported number of Slice components

// Number of components
const (
	SCALAR = 1
	VECTOR = 3
)

// Frees the underlying storage and zeros the Slice header to avoid accidental use.
// Slices sharing storage will be invalid after Free. Double free is OK.
func (s *Slice) Free() {
	// free storage
	switch s.memType {
	case 0:
		return // already freed
	case gpuMemory:
		for _, ptr := range s.ptrs {
			cu.MemFree(cu.DevicePtr(ptr))
		}
	default:
		panic("todo")
	}
	// zero the struct
	for c := range s.ptr_ {
		s.ptr_[c] = unsafe.Pointer(uintptr(0))
	}
	s.ptrs = s.ptrs[:0]
	s.len_ = 0
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
	return len(s.ptrs)
}

// Len returns the number of elements per component.
func (s *Slice) Len() int {
	return int(s.len_)
}

//// Bytes returns the number of storage bytes per component.
//func (s *Slice) bytes() int64 {
//	return int64(s.len_) * cu.SIZEOF_FLOAT32
//}

// Comp returns a single component of the Slice.
func (s *Slice) Comp(i int) *Slice {
	sl := new(Slice)
	sl.ptr_[0] = s.ptrs[i]
	sl.ptrs = sl.ptr_[:1]
	sl.len_ = s.len_
	sl.memType = s.memType
	return sl
}

// DevPtr returns a CUDA device pointer to a component.
// Slice must have GPUAccess.
func (s *Slice) DevPtr(component int) cu.DevicePtr {
	if !s.GPUAccess() {
		panic("slice not accessible by GPU")
	}
	return cu.DevicePtr(s.ptrs[component])
}

func MemAlloc(bytes int64) cu.DevicePtr {
	defer func() {
		err := recover()
		if err == cu.ERROR_OUT_OF_MEMORY {
			FatalErr(err)
		}
		if err != nil {
			panic(err)
		}
	}()
	return cu.MemAlloc(bytes)
}

//// Slice returns a slice sharing memory with the original.
//func (s *Slice) Slice(a, b int) Slice {
//	len_ := int(s.len_)
//	if a >= len_ || b > len_ || a > b || a < 0 || b < 0 {
//		panic(fmt.Errorf("slice range out of bounds: [%v:%v] (len=%v)", a, b, len_))
//	}
//	var slice Slice
//	for i := range s.ptr {
//		s.ptr[i] = unsafe.Pointer(uintptr(s.ptr[i]) + cu.SIZEOF_FLOAT32*uintptr(a))
//	}
//	slice.len_ = int32(b - a)
//	slice.nComp = s.nComp
//	return slice
//}

// Set the entire slice to this value.
func (s *Slice) Memset(val ...float32) {
	Argument(len(val) == s.NComp())
	str := streams.Get()
	for c, v := range val {
		cu.MemsetD32Async(s.DevPtr(c), math.Float32bits(v), int64(s.Len()), str)
	}
	streams.SyncAndRecycle(str)
}

////func unifiedSlice(N int) Slice {
////	bytes := int64(N) * SizeofFloat32
////	ptr := unsafe.Pointer(cu.MemAllocHost(bytes))
////	ptrs := [MAX_COMP]unsafe.Pointer{ptr}
////	return Slice{ptrs, N, 1, UnifiedMemory}
////}
////
////func ToSlice(list []float32) Slice {
////	ptr := unsafe.Pointer(&list[0])
////	ptrs := [MAX_COMP]unsafe.Pointer{ptr}
////	return Slice{ptrs, len(list), 1, CPUMemory}
////}
////
////func UnsafeSlice(ptr unsafe.Pointer, len_ int, flag MemType) Slice {
////	if len_ < 0 {
////		panic("negative slice length")
////	}
////	ptrs := [MAX_COMP]unsafe.Pointer{ptr}
////	return Slice{ptrs, len_, 1, flag}
////}
//
//// TODO: rm
////func (s *Slice) Safe(component int) safe.Float32s {
////	var f safe.Float32s
////	f.UnsafeSet(unsafe.Pointer(s.DevPtr(component)), s.Len(), s.Len())
////	return f
////}
//
////func (s *Slice) Host() []float32 {
////	if s.nComp != 1 {
////		panic("need to implement for components")
////	}
////
////	if s.MemType&CPUMemory == 0 {
////		core.Panicf("slice not accessible by CPU (memory type %v)", s.MemType)
////	}
////	var list []float32
////	hdr := (*reflect.SliceHeader)(unsafe.Pointer(&list))
////	hdr.Data = uintptr(s.ptr[0])
////	hdr.Len = int(s.len_)
////	hdr.Cap = hdr.Len
////	return list
////}
////
////func (s *Slice) Device() safe.Float32s {
////	if s.nComp != 1 {
////		panic(fmt.Errorf("slice.device: need 1 component, have %v", s.nComp))
////	}
////	if s.MemType&GPUMemory == 0 {
////		panic(fmt.Errorf("slice not accessible by GPU (memory type %v)", s.MemType))
////	}
////	var floats safe.Float32s
////	floats.UnsafeSet(s.ptr[0], s.Len(), s.Len())
////	return floats
////}
