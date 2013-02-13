package mx

// File: Slice stores N-component GPU or host data.
// Author: Arne Vansteenkiste

import (
	"code.google.com/p/mx3/streams"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"math"
	"reflect"
	"unsafe"
)

// Slice is like a [][]float32, but may be stored in GPU or host memory.
type Slice struct {
	ptr_ [MAX_COMP]unsafe.Pointer // keeps data local
	ptrs []unsafe.Pointer         // points into ptr_
	*info
	len_    int32
	memType int8
}

// Make a GPU Slice with nComp components each of size length.
func NewGPUSlice(nComp int, m *Mesh) *Slice {
	s := newSlice(nComp, m)
	length := m.NCell()
	bytes := int64(length) * cu.SIZEOF_FLOAT32
	for c := range s.ptrs {
		s.ptrs[c] = unsafe.Pointer(MemAlloc(bytes))
	}
	s.memType = GPUMemory
	s.Memset(make([]float32, nComp)...)
	return s
}

// Make a GPU Slice with nComp components each of size length.
func NewUnifiedSlice(nComp int, m *Mesh) *Slice {
	s := newSlice(nComp, m)
	length := m.NCell()
	bytes := int64(length) * cu.SIZEOF_FLOAT32
	for c := range s.ptrs {
		s.ptrs[c] = cu.MemAllocHost(bytes)
	}
	s.memType = UnifiedMemory
	s.Memset(make([]float32, nComp)...)
	return s
}

// Make a CPU Slice with nComp components of size length.
func NewCPUSlice(nComp int, m *Mesh) *Slice {
	s := newSlice(nComp, m)
	length := m.NCell()
	for c := range s.ptrs {
		s.ptrs[c] = unsafe.Pointer(&(make([]float32, length)[0]))
	}
	s.memType = CPUMemory
	return s
}

func NewSliceMemtype(nComp int, m *Mesh, memType int) *Slice {
	switch memType {
	default:
		Panicf("illegal memory type: %v", memType)
	case GPUMemory:
		return NewGPUSlice(nComp, m)
	case CPUMemory:
		return NewCPUSlice(nComp, m)
	case UnifiedMemory:
		return NewUnifiedSlice(nComp, m)
	}
	panic("unreachable")
	return nil
}

func newSlice(nComp int, m *Mesh) *Slice {
	length := m.NCell()
	Argument(nComp > 0 && length > 0)
	s := new(Slice)
	s.ptrs = s.ptr_[:nComp]
	s.len_ = int32(length)
	s.info = new(info)
	s.info.mesh = *m
	return s
}

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
	case GPUMemory:
		for _, ptr := range s.ptrs {
			cu.MemFree(cu.DevicePtr(ptr))
		}
	case UnifiedMemory:
		for _, ptr := range s.ptrs {
			cu.MemFreeHost(ptr)
		}
	case CPUMemory:
		// nothing to do
	default:
		panic("invalid memory type")
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
	CPUMemory     = 1 << 0
	GPUMemory     = 1 << 1
	UnifiedMemory = CPUMemory | GPUMemory
)

// MemType returns the memory type of the underlying storage:
// CPUMemory, GPUMemory or UnifiedMemory
func (s *Slice) MemType() int {
	return int(s.memType)
}

// GPUAccess returns whether the Slice is accessible by the GPU.
// true means it is either stored on GPU or in unified host memory.
func (s *Slice) GPUAccess() bool {
	return s.memType&GPUMemory != 0
}

// CPUAccess returns whether the Slice is accessible by the CPU.
// true means it is stored in host memory.
func (s *Slice) CPUAccess() bool {
	return s.memType&CPUMemory != 0
}

// NComp returns the number of components.
func (s *Slice) NComp() int {
	return len(s.ptrs)
}

// Len returns the number of elements per component.
func (s *Slice) Len() int {
	return int(s.len_)
}

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

// DevPtr returns a pointer to a component.
// Slice must have CPUAccess.
func (s *Slice) HostPtr(component int) unsafe.Pointer {
	if !s.CPUAccess() {
		panic("slice not accessible by CPU")
	}
	return s.ptrs[component]
}

// Wrapper for cu.MemAlloc, fatal exit on out of memory.
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

// Slice returns a slice sharing memory with the original.
func (s *Slice) Slice(a, b int) *Slice {
	len_ := int(s.len_)
	if a >= len_ || b > len_ || a > b || a < 0 || b < 0 {
		Panicf("slice range out of bounds: [%v:%v] (len=%v)", a, b, len_)
	}

	slice := new(Slice)
	slice.ptrs = s.ptr_[:s.NComp()]
	for i := range s.ptrs {
		slice.ptrs[i] = unsafe.Pointer(uintptr(s.ptrs[i]) + cu.SIZEOF_FLOAT32*uintptr(a))
	}
	slice.len_ = int32(b - a)
	slice.memType = s.memType
	return slice
}

// Set the entire slice to this value, component by component.
func (s *Slice) Memset(val ...float32) {
	Argument(len(val) == s.NComp())
	str := streams.Get()
	for c, v := range val {
		cu.MemsetD32Async(s.DevPtr(c), math.Float32bits(v), int64(s.Len()), str)
	}
	streams.SyncAndRecycle(str)
}

// Host returns the Slice as a [][]float32,
// indexed by component, cell number.
// It should have CPUAccess() == true.
func (s *Slice) Host() [][]float32 {
	if !s.CPUAccess() {
		Panic("slice not accessible by CPU")
	}
	list := make([][]float32, s.NComp())
	for c := range list {
		hdr := (*reflect.SliceHeader)(unsafe.Pointer(&list[c]))
		hdr.Data = uintptr(s.ptrs[c])
		hdr.Len = int(s.len_)
		hdr.Cap = hdr.Len
	}
	return list
}

// Floats returns the data as 3D array,
// indexed by cell position. Data should be
// scalar (1 component) and have CPUAccess() == true.
func (s *Slice) Floats() [][][]float32 {
	x := s.Tensors()
	if len(x) != 1 {
		Panicf("expecting 1 component, got %v", s.NComp())
	}
	return x[0]
}

// Vectors returns the data as 4D array,
// indexed by component, cell position. Data should have
// 3 components and have CPUAccess() == true.
func (s *Slice) Vectors() [3][][][]float32 {
	x := s.Tensors()
	if len(x) != 3 {
		Panicf("expecting 3 components, got %v", s.NComp())
	}
	return [3][][][]float32{x[0], x[1], x[2]}
}

// Tensors returns the data as 4D array,
// indexed by component, cell position.
// Requires CPUAccess() == true.
func (s *Slice) Tensors() [][][][]float32 {
	tensors := make([][][][]float32, s.NComp())
	host := s.Host()
	for i := range tensors {
		tensors[i] = util.Reshape(host[i], s.Mesh().Size())
	}
	return tensors
}
