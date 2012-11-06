package core

import (
	"github.com/barnex/cuda5/safe"
	"unsafe"
	"reflect"
)

type Slice struct {
	ptr unsafe.Pointer
	len_ int
	MemType 
}

const (
	CPUMemory MemType = 1 << iota
	GPUMemory
)

type MemType byte

func(m MemType) GPUAccess() bool{
	return m & GPUMemory != 0
}

func(m MemType) CPUAccess() bool{
	return m & CPUMemory != 0
}

func MakeSlice(ptr unsafe.Pointer, len_ int, flag MemType) Slice{
	return Slice{ptr, len_, flag}
}

func (s *Slice) Slice(a, b int) Slice {

}

func (s Slice) Host() []float32 {
	if s.MemType & CPUMemory == 0{
		Panicf("slice not accessible by CPU (memory type %v)", s.MemType)
	}
	var list []float32
	hdr := (*reflect.SliceHeader)(unsafe.Pointer(&list))
	hdr.Data = uintptr(s.ptr)
	hdr.Len = s.len_
	hdr.Cap = s.len_
	return list
}

func (s Slice) Gpu() safe.Float32s {
	if s.MemType & GPUMemory == 0{
		Panicf("slice not accessible by GPU (memory type %v)", s.MemType)
	}
	var floats safe.Float32s
	floats.UnsafeSet(s.ptr, s.len_, s.len_)
	return floats
}
