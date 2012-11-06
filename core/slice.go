package core

import (
	"github.com/barnex/cuda5/safe"
	"reflect"
	"unsafe"
)

type Slice struct {
	ptr  unsafe.Pointer
	len_ int
	flag MemType
}

const (
	CPUMemory MemType = 1 << iota
	GPUMemory
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

func Float32ToSlice(list []float32) Slice {
	return Slice{unsafe.Pointer(&list[0]), len(list), CPUMemory}
}

const SizeofFloat32 = 4

func (s *Slice) Slice(a, b int) Slice {
	ptr := unsafe.Pointer(uintptr(s.ptr) + SizeofFloat32*uintptr(a))
	len_ := b - a
	if a >= s.len_ || b > s.len_ || a > b || a< 0||b< 0 {
		Panicf("slice range out of bounds: [%v:%v] (len=%v)", a, b, s.len_)
	}
	return Slice{ptr, len_, s.flag}
}

func (s Slice) Host() []float32 {
	if s.flag&CPUMemory == 0 {
		Panicf("slice not accessible by CPU (memory type %v)", s.flag)
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
		Panicf("slice not accessible by GPU (memory type %v)", s.flag)
	}
	var floats safe.Float32s
	floats.UnsafeSet(s.ptr, s.len_, s.len_)
	return floats
}

func(s Slice)Len()int{return s.len_}
