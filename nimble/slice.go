package nimble

import (
	"code.google.com/p/nimble-cube/core"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"reflect"
	"unsafe"
)

// Slice of float32, accessible by CPU, GPU or both.
type Slice struct {
	ptr  unsafe.Pointer
	len_ int
	flag MemType
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

func MakeSlices(nComp int, length int, memType MemType)[]Slice{
	s:=make([]Slice, nComp)
	for i:=range s{
		s[i] = MakeSlice(length, memType)
	}
	return s
}

func cpuSlice(N int) Slice {
	return ToSlice(make([]float32, N))
}

func gpuSlice(N int) Slice {
	bytes := int64(N) * SizeofFloat32
	ptr := unsafe.Pointer(uintptr(cu.MemAlloc(bytes)))
	return Slice{ptr, N, GPUMemory}
}

func unifiedSlice(N int) Slice {
	bytes := int64(N) * SizeofFloat32
	ptr := unsafe.Pointer(cu.MemAllocHost(bytes))
	return Slice{ptr, N, UnifiedMemory}
}

func ToSlice(list []float32) Slice {
	return Slice{unsafe.Pointer(&list[0]), len(list), CPUMemory}
}

const (
	CPUMemory     MemType = 1 << 0
	GPUMemory     MemType = 1 << 1
	UnifiedMemory MemType = CPUMemory | GPUMemory
)

type MemType byte

func (s Slice) GPUAccess() bool {
	return s.flag&GPUMemory != 0
}

func (s Slice) CPUAccess() bool {
	return s.flag&CPUMemory != 0
}

func UnsafeSlice(ptr unsafe.Pointer, len_ int, flag MemType) Slice {
	if len_ < 0 {
		panic("negative slice length")
	}
	return Slice{ptr, len_, flag}
}

func (s *Slice) Slice(a, b int) Slice {
	ptr := unsafe.Pointer(uintptr(s.ptr) + SizeofFloat32*uintptr(a))
	len_ := b - a
	if a >= s.len_ || b > s.len_ || a > b || a < 0 || b < 0 {
		core.Panicf("slice range out of bounds: [%v:%v] (len=%v)", a, b, s.len_)
	}
	return Slice{ptr, len_, s.flag}
}

func (s Slice) Host() []float32 {
	if s.flag&CPUMemory == 0 {
		core.Panicf("slice not accessible by CPU (memory type %v)", s.flag)
	}
	var list []float32
	hdr := (*reflect.SliceHeader)(unsafe.Pointer(&list))
	hdr.Data = uintptr(s.ptr)
	hdr.Len = s.len_
	hdr.Cap = s.len_
	return list
}

func (s Slice) Device() safe.Float32s {
	if s.flag&GPUMemory == 0 {
		core.Panicf("slice not accessible by GPU (memory type %v)", s.flag)
	}
	var floats safe.Float32s
	floats.UnsafeSet(s.ptr, s.len_, s.len_)
	return floats
}

func (s Slice) Len() int { return s.len_ }

const SizeofFloat32 = 4
