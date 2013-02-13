package data

import (
	"unsafe"
)

type Device interface {
	MemFree(unsafe.Pointer)
	GPUAccess() bool
	CPUAccess() bool
	Memset(ptr unsafe.Pointer, val float32, N int)
}

type cpu struct{}

func (c cpu) MemFree(unsafe.Pointer) {}

func (c cpu) CPUAccess() bool { return true }

func (c cpu) GPUAccess() bool { return false }

func (c cpu) Memset(ptr unsafe.Pointer, val float32, N int) {
	s := assembleSlice(ptr, N)
	for i := range s {
		s[i] = val
	}
}
