package cuda

import (
	"code.google.com/p/mx3/data"
)

// Make a GPU Slice with nComp components each of size length.
func NewSlice(nComp int, m *data.Mesh) *data.Slice {
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
func NewUnifiedSlice(nComp int, m *Mesh) *data.Slice {
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
