package data

// File: Slice stores N-component GPU or host data.
// Author: Arne Vansteenkiste

import (
	"log"
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

// this package must not depend on CUDA. If CUDA is
// loaded, these functions are set to cu.MemFree, ...
var (
	memFree, memFreeHost           func(unsafe.Pointer)
	memCpy, memCpyDtoH, memCpyHtoD func(dst, src unsafe.Pointer, bytes int64)
)

// Internal: enables slices on GPU. Called upon cuda init.
func EnableGPU(free, freeHost func(unsafe.Pointer),
	cpy, cpyDtoH, cpyHtoD func(dst, src unsafe.Pointer, bytes int64)) {
	memFree = free
	memFreeHost = freeHost
	memCpy = cpy
	memCpyDtoH = cpyDtoH
	memCpyHtoD = cpyHtoD
}

// Make a CPU Slice with nComp components of size length.
func NewSlice(nComp int, m *Mesh) *Slice {
	length := m.NCell()
	ptrs := make([]unsafe.Pointer, nComp)
	for i := range ptrs {
		ptrs[i] = unsafe.Pointer(&(make([]float32, length)[0]))
	}
	return SliceFromPtrs(m, CPUMemory, ptrs)
}

// Internal: construct a Slice using bare memory pointers. Used by cuda.
func SliceFromPtrs(m *Mesh, memType int8, ptrs []unsafe.Pointer) *Slice {
	length := m.NCell()
	nComp := len(ptrs)
	argument(nComp > 0 && length > 0 && nComp <= MAX_COMP)
	s := new(Slice)
	s.ptrs = s.ptr_[:nComp]
	s.len_ = int32(length)
	s.info = new(info)
	s.info.mesh = *m
	for c := range ptrs {
		s.ptrs[c] = ptrs[c]
	}
	s.memType = memType
	return s
}

func SliceFromList(data [][]float32, mesh *Mesh) *Slice {
	ptrs := make([]unsafe.Pointer, len(data))
	for i := range ptrs {
		argument(len(data[i]) == mesh.NCell())
		ptrs[i] = unsafe.Pointer(&data[i][0])
	}
	return SliceFromPtrs(mesh, CPUMemory, ptrs)
}

const MAX_COMP = 3 // Maximum supported number of Slice components

// Frees the underlying storage and zeros the Slice header to avoid accidental use.
// Slices sharing storage will be invalid after Free. Double free is OK.
func (s *Slice) Free() {
	// free storage
	switch s.memType {
	case 0:
		return // already freed
	case GPUMemory:
		for _, ptr := range s.ptrs {
			memFree(ptr)
		}
	case UnifiedMemory:
		for _, ptr := range s.ptrs {
			memFreeHost(ptr)
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
	sl.info = s.info
	sl.len_ = s.len_
	sl.memType = s.memType
	return sl
}

// DevPtr returns a CUDA device pointer to a component.
// Slice must have GPUAccess.
func (s *Slice) DevPtr(component int) unsafe.Pointer {
	if !s.GPUAccess() {
		panic("slice not accessible by GPU")
	}
	return s.ptrs[component]
}

//// HostPtr returns a pointer to a component.
//// Slice must have CPUAccess.
//func (s *Slice) HostPtr(component int) unsafe.Pointer {
//	if !s.CPUAccess() {
//		panic("slice not accessible by CPU")
//	}
//	return s.ptrs[component]
//}

// Slice returns a slice sharing memory with the original.
func (s *Slice) Slice(a, b int) *Slice {
	len_ := int(s.len_)
	if a >= len_ || b > len_ || a > b || a < 0 || b < 0 {
		log.Panicf("slice range out of bounds: [%v:%v] (len=%v)", a, b, len_)
	}

	slice := new(Slice)
	slice.ptrs = slice.ptr_[:s.NComp()]
	for i := range s.ptrs {
		slice.ptrs[i] = unsafe.Pointer(uintptr(s.ptrs[i]) + SIZEOF_FLOAT32*uintptr(a))
	}
	slice.len_ = int32(b - a)
	slice.memType = s.memType
	slice.info = s.info
	return slice
}

const SIZEOF_FLOAT32 = 4

// Set the entire slice to this value, component by component.
//func (s *Slice) Memset(val ...float32) {
//}

// Host returns the Slice as a [][]float32,
// indexed by component, cell number.
// It should have CPUAccess() == true.
func (s *Slice) Host() [][]float32 {
	if !s.CPUAccess() {
		log.Panic("slice not accessible by CPU")
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

// Returns a copy of the Slice, allocated on CPU.
func (s *Slice) HostCopy() *Slice {
	cpy := NewSlice(s.NComp(), s.Mesh())
	Copy(cpy, s)
	return cpy
}

func Copy(dst, src *Slice) {
	if dst.NComp() != src.NComp() || dst.Len() != src.Len() {
		log.Panicf("slice copy: illegal sizes: dst: %vx%v, src: %vx%v", dst.NComp(), dst.Len(), src.NComp(), src.Len())
	}
	d, s := dst.GPUAccess(), src.GPUAccess()
	bytes := SIZEOF_FLOAT32 * int64(dst.Len())
	switch {
	default:
		panic("bug")
	case d && s:
		for c := 0; c < dst.NComp(); c++ {
			memCpy(dst.DevPtr(c), src.DevPtr(c), bytes)
		}
	case s && !d:
		for c := 0; c < dst.NComp(); c++ {
			memCpyDtoH(dst.ptr_[c], src.DevPtr(c), bytes)
		}
	case !s && d:
		for c := 0; c < dst.NComp(); c++ {
			memCpyHtoD(dst.DevPtr(c), src.ptr_[c], bytes)
		}
	case !d && !s:
		dst, src := dst.Host(), src.Host()
		for c := range dst {
			copy(dst[c], src[c])
		}
	}
}

// Floats returns the data as 3D array,
// indexed by cell position. Data should be
// scalar (1 component) and have CPUAccess() == true.
func (f *Slice) Scalars() [][][]float32 {
	x := f.Tensors()
	if len(x) != 1 {
		log.Panicf("expecting 1 component, got %v", f.NComp())
	}
	return x[0]
}

// Vectors returns the data as 4D array,
// indexed by component, cell position. Data should have
// 3 components and have CPUAccess() == true.
func (f *Slice) Vectors() [3][][][]float32 {
	x := f.Tensors()
	if len(x) != 3 {
		log.Panicf("expecting 3 components, got %v", f.NComp())
	}
	return [3][][][]float32{x[0], x[1], x[2]}
}

// Tensors returns the data as 4D array,
// indexed by component, cell position.
// Requires CPUAccess() == true.
func (f *Slice) Tensors() [][][][]float32 {
	tensors := make([][][][]float32, f.NComp())
	host := f.Host()
	for i := range tensors {
		tensors[i] = Reshape(host[i], f.Mesh().Size())
	}
	return tensors
}

//package data
//
//// File: Slice stores N-component GPU or host data.
//// Author: Arne Vansteenkiste
//
//import (
//	"log"
//	"reflect"
//	"unsafe"
//)
//
//// Slice is like a [][]float32, but may be stored in GPU or host memory.
//type Slice struct {
//	ptr_ [MAX_COMP]unsafe.Pointer // keeps data local
//	ptrs []unsafe.Pointer         // points into ptr_
//	*info
//	len_ int32
//	dev  Device // has free, memset, ...
//}
//
//// Make a CPU Slice with nComp components of size length.
//func NewSlice(nComp int, m *Mesh) *Slice {
//	s := newSlice(nComp, m)
//	length := m.NCell()
//	for c := range s.ptrs {
//		s.ptrs[c] = unsafe.Pointer(&(make([]float32, length)[0]))
//	}
//	s.dev = cpu{}
//	return s
//}
//
//func newSlice(nComp int, m *Mesh) *Slice {
//	length := m.NCell()
//	argument(nComp > 0 && length > 0)
//	s := new(Slice)
//	s.ptrs = s.ptr_[:nComp]
//	s.len_ = int32(length)
//	s.info = new(info)
//	s.info.mesh = *m
//	return s
//}
//
//const MAX_COMP = 3 // Maximum supported number of Slice components
//
//// Number of components
//const (
//	SCALAR = 1
//	VECTOR = 3
//)
//
//// Frees the underlying storage and zeros the Slice header to avoid accidental use.
//// Slices sharing storage will be invalid after Free. Double free is OK.
//func (s *Slice) Free() {
//	for _, ptr := range s.ptrs {
//		s.dev.MemFree(ptr)
//	}
//	// zero the struct
//	for c := range s.ptr_ {
//		s.ptr_[c] = unsafe.Pointer(uintptr(0))
//	}
//	s.ptrs = s.ptrs[:0]
//	s.len_ = 0
//}
//
//// GPUAccess returns whether the Slice is accessible by the GPU.
//// true means it is either stored on GPU or in unified host memory.
//func (s *Slice) GPUAccess() bool {
//	return s.dev.GPUAccess()
//}
//
//// CPUAccess returns whether the Slice is accessible by the CPU.
//// true means it is stored in host memory.
//func (s *Slice) CPUAccess() bool {
//	return s.dev.CPUAccess()
//}
//
//// NComp returns the number of components.
//func (s *Slice) NComp() int {
//	return len(s.ptrs)
//}
//
//// Len returns the number of elements per component.
//func (s *Slice) Len() int {
//	return int(s.len_)
//}
//
//// Comp returns a single component of the Slice.
//func (s *Slice) Comp(i int) *Slice {
//	sl := new(Slice)
//	sl.ptr_[0] = s.ptrs[i]
//	sl.ptrs = sl.ptr_[:1]
//	sl.len_ = s.len_
//	sl.dev = s.dev
//	return sl
//}
//
//// DevPtr returns a CUDA device pointer to a component.
//// Slice must have GPUAccess.
//func (s *Slice) DevPtr(component int) uintptr {
//	if !s.GPUAccess() {
//		panic("slice not accessible by GPU")
//	}
//	return uintptr(s.ptrs[component])
//}
//
//// DevPtr returns a pointer to a component.
//// Slice must have CPUAccess.
//func (s *Slice) HostPtr(component int) unsafe.Pointer {
//	if !s.CPUAccess() {
//		panic("slice not accessible by CPU")
//	}
//	return s.ptrs[component]
//}
//
//// Slice returns a slice sharing memory with the original.
//func (s *Slice) Slice(a, b int) *Slice {
//	len_ := int(s.len_)
//	if a >= len_ || b > len_ || a > b || a < 0 || b < 0 {
//		log.Panicf("slice range out of bounds: [%v:%v] (len=%v)", a, b, len_)
//	}
//
//	slice := new(Slice)
//	slice.ptrs = s.ptr_[:s.NComp()]
//	for i := range s.ptrs {
//		slice.ptrs[i] = unsafe.Pointer(uintptr(s.ptrs[i]) + SIZEOF_FLOAT32*uintptr(a))
//	}
//	slice.len_ = int32(b - a)
//	slice.dev = s.dev
//	return slice
//}
//
//const SIZEOF_FLOAT32 = 4
//
//// Set the entire slice to this value, component by component.
//func (s *Slice) Memset(val ...float32) {
//	argument(len(val) == s.NComp())
//	for c, v := range val {
//		s.dev.Memset(s.ptrs[c], v, s.Len())
//	}
//}
//
//// Host returns the Slice as a [][]float32,
//// indexed by component, cell number.
//// It should have CPUAccess() == true.
//func (s *Slice) Host() [][]float32 {
//	if !s.CPUAccess() {
//		log.Panicf("slice not accessible by CPU")
//	}
//	list := make([][]float32, s.NComp())
//	for c := range list {
//		list[c] = assembleSlice(s.ptrs[c], int(s.len_))
//	}
//	return list
//}
//
//func assembleSlice(ptr unsafe.Pointer, len_ int) []float32 {
//	var list []float32
//	hdr := (*reflect.SliceHeader)(unsafe.Pointer(&list))
//	hdr.Data = uintptr(ptr)
//	hdr.Len = int(len_)
//	hdr.Cap = hdr.Len
//	return list
//}
//
//// Floats returns the data as 3D array,
//// indexed by cell position. Data should be
//// scalar (1 component) and have CPUAccess() == true.
//func (s *Slice) Floats() [][][]float32 {
//	x := s.Tensors()
//	if len(x) != 1 {
//		log.Panicf("expecting 1 component, got %v", s.NComp())
//	}
//	return x[0]
//}
//
//// Vectors returns the data as 4D array,
//// indexed by component, cell position. Data should have
//// 3 components and have CPUAccess() == true.
//func (s *Slice) Vectors() [3][][][]float32 {
//	x := s.Tensors()
//	if len(x) != 3 {
//		log.Panicf("expecting 3 components, got %v", s.NComp())
//	}
//	return [3][][][]float32{x[0], x[1], x[2]}
//}
//
//// Tensors returns the data as 4D array,
//// indexed by component, cell position.
//// Requires CPUAccess() == true.
//func (s *Slice) Tensors() [][][][]float32 {
//	tensors := make([][][][]float32, s.NComp())
//	host := s.Host()
//	for i := range tensors {
//		tensors[i] = Reshape(host[i], s.Mesh().Size())
//	}
//	return tensors
//}
