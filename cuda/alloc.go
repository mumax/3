package cuda

import (
	"log"
	"unsafe"

	"github.com/mumax/3/v3/cuda/cu"
	"github.com/mumax/3/v3/data"
)

// Wrapper for cu.MemAlloc, fatal exit on out of memory.
func MemAlloc(bytes int64) unsafe.Pointer {
	defer func() {
		err := recover()
		if err == cu.ERROR_OUT_OF_MEMORY {
			log.Fatal(err)
		}
		if err != nil {
			panic(err)
		}
	}()
	return unsafe.Pointer(uintptr(cu.MemAlloc(bytes)))
}

// Returns a copy of in, allocated on GPU.
func GPUCopy(in *data.Slice) *data.Slice {
	s := NewSlice(in.NComp(), in.Size())
	data.Copy(s, in)
	return s
}
