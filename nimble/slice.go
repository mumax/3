package nimble

import (
	"code.google.com/p/mx3/core"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"reflect"
	"unsafe"
)

const MAX_COMP = 3

// Slice of float32, accessible by CPU, GPU or both.
type Slice struct {
	ptr   [MAX_COMP]unsafe.Pointer
	mesh  *Mesh
	len_  int // TODO int32?
	nComp int
	MemType
	tag, unit string
}

func makeSliceN(nComp, length int, mem MemType) Slice {
	if mem != GPUMemory {
		panic("todo")
	}
	bytes := int64(length) * cu.SIZEOF_FLOAT32
	var ptrs [MAX_COMP]unsafe.Pointer
	for c := 0; c < nComp; c++ {
		ptrs[c] = cu.MemAlloc(bytes)
	}
	return Slice{ptrs, length, nComp, mem}
}

func MakeSlice(length int, memtype MemType) Slice {
	switch memtype {
	default:
		core.Panic("makeslice: illegal memtype:", memtype)
	case CPUMemory:
		return cpuSlice(length)
	case GPUMemory:
		return gpuSlice(length)
	case UnifiedMemory:
		return unifiedSlice(length)
	}
	panic("bug")
	var s Slice
	return s
}

func MakeSlices(nComp int, length int, memType MemType) []Slice {
	s := make([]Slice, nComp)
	for i := range s {
		s[i] = MakeSlice(length, memType)
	}
	return s
}

func cpuSlice(N int) Slice {
	return ToSlice(make([]float32, N))
}

func gpuSlice(N int) Slice {
	s := safe.MakeFloat32s(N)
	ptrs := [MAX_COMP]unsafe.Pointer{unsafe.Pointer(s.Pointer())}
	return Slice{ptrs, N, 1, GPUMemory}
}

func unifiedSlice(N int) Slice {
	bytes := int64(N) * SizeofFloat32
	ptr := unsafe.Pointer(cu.MemAllocHost(bytes))
	ptrs := [MAX_COMP]unsafe.Pointer{ptr}
	return Slice{ptrs, N, 1, UnifiedMemory}
}

func ToSlice(list []float32) Slice {
	ptr := unsafe.Pointer(&list[0])
	ptrs := [MAX_COMP]unsafe.Pointer{ptr}
	return Slice{ptrs, len(list), 1, CPUMemory}
}

func UnsafeSlice(ptr unsafe.Pointer, len_ int, flag MemType) Slice {
	if len_ < 0 {
		panic("negative slice length")
	}
	ptrs := [MAX_COMP]unsafe.Pointer{ptr}
	return Slice{ptrs, len_, 1, flag}
}

func (s *Slice) Slice(a, b int) Slice {
	if s.nComp != 1 {
		panic("need to implement for components")
	}
	ncomp := s.nComp

	ptr := unsafe.Pointer(uintptr(s.ptr[0]) + SizeofFloat32*uintptr(a))
	len_ := b - a
	if a >= s.len_ || b > s.len_ || a > b || a < 0 || b < 0 {
		core.Panicf("slice range out of bounds: [%v:%v] (len=%v)", a, b, s.len_)
	}
	ptrs := [MAX_COMP]unsafe.Pointer{ptr}
	return Slice{ptrs, len_, ncomp, s.MemType}
}

func (s Slice) Host() []float32 {
	if s.nComp != 1 {
		panic("need to implement for components")
	}

	if s.MemType&CPUMemory == 0 {
		core.Panicf("slice not accessible by CPU (memory type %v)", s.MemType)
	}
	var list []float32
	hdr := (*reflect.SliceHeader)(unsafe.Pointer(&list))
	hdr.Data = uintptr(s.ptr[0])
	hdr.Len = s.len_
	hdr.Cap = s.len_
	return list
}

func (s Slice) Device() safe.Float32s {
	if s.nComp != 1 {
		panic("need to implement for components")
	}
	if s.MemType&GPUMemory == 0 {
		core.Panicf("slice not accessible by GPU (memory type %v)", s.MemType)
	}
	var floats safe.Float32s
	floats.UnsafeSet(s.ptr[0], s.len_, s.len_)
	return floats
}

func (s Slice) Len() int { return s.len_ }

const SizeofFloat32 = 4

func (s *Slice) Unit() string { return s.unit }
func (s *Slice) Tag() string  { return s.tag }
