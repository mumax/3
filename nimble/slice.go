package nimble

import (
	"fmt"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"unsafe"
)

const MAX_COMP = 3

// Slice is like a [][]float32, but may be stored in GPU or host memory.
type Slice struct {
	ptr   [MAX_COMP]unsafe.Pointer
	len_  int32
	nComp int8
	MemType
}

// mv to makeGPUFrame etc.
func makeSlice(nComp, length int, mem MemType) Slice {
	if mem != GPUMemory {
		panic("todo")
	}
	bytes := int64(length) * cu.SIZEOF_FLOAT32
	var ptrs [MAX_COMP]unsafe.Pointer
	for c := 0; c < nComp; c++ {
		ptrs[c] = unsafe.Pointer(cu.MemAlloc(bytes))
	}
	return Slice{ptrs, int32(length), int8(nComp), mem}
}

// Len returns the number of elements.
func (s *Slice) Len() int {
	return int(s.len_)
}

// Slice returns a slice sharing memory with the original.
func (s *Slice) Slice(a, b int) Slice {
	len_ := int(s.len_)
	if a >= len_ || b > len_ || a > b || a < 0 || b < 0 {
		panic(fmt.Errorf("slice range out of bounds: [%v:%v] (len=%v)", a, b, len_))
	}
	var slice Slice
	for i := range s.ptr {
		s.ptr[i] = unsafe.Pointer(uintptr(s.ptr[i]) + SizeofFloat32*uintptr(a))
	}
	slice.len_ = int32(b - a)
	slice.nComp = s.nComp
	return slice
}

func (s *Slice) Comp(i int) Slice {
	return Slice{[MAX_COMP]unsafe.Pointer{s.ptr[i]}, s.len_, 1, s.MemType}
}

const SizeofFloat32 = 4

//func MakeSliceN(length int, memtype MemType) Slice {
//	return MakeSlices(1, length, memtype)
//	//	switch memtype {
//	//	default:
//	//		core.Panic("makeslice: illegal memtype:", memtype)
//	//	case CPUMemory:
//	//		return cpuSlice(length)
//	//	case GPUMemory:
//	//		return gpuSlice(length)
//	//	case UnifiedMemory:
//	//		return unifiedSlice(length)
//	//	}
//	//	panic("bug")
//	//	var s Slice
//	//	return s
//}

//func MakeSlices(nComp int, length int, memType MemType) []Slice {
//	s := make([]Slice, nComp)
//	for i := range s {
//		s[i] = MakeSlice(length, memType)
//	}
//	return s
//}

//func cpuSlice(N int) Slice {
//	return ToSlice(make([]float32, N))
//}
//
//func gpuSlice(N int) Slice {
//	s := safe.MakeFloat32s(N)
//	ptrs := [MAX_COMP]unsafe.Pointer{unsafe.Pointer(s.Pointer())}
//	return Slice{ptrs, N, 1, GPUMemory}
//}
//
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

//func (s Slice) Host() []float32 {
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
//	hdr.Len = s.len_
//	hdr.Cap = s.len_
//	return list
//}

func (s Slice) Device() safe.Float32s {
	if s.nComp != 1 {
		panic(fmt.Errorf("slice.device: need 1 component, have %v", s.nComp))
	}
	if s.MemType&GPUMemory == 0 {
		panic(fmt.Errorf("slice not accessible by GPU (memory type %v)", s.MemType))
	}
	var floats safe.Float32s
	floats.UnsafeSet(s.ptr[0], s.Len(), s.Len())
	return floats
}
