package cuda

import (
	"github.com/barnex/cuda5/cu"
	"github.com/mumax/3/data"
	"log"
	"unsafe"
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

// make slice of given size, but with dummy mesh. // TODO: rm
func makeFloats(size [3]int) *data.Slice {
	return NewSlice(1, size)
}

// Returns a copy of in, allocated on GPU.
func GPUCopy(in *data.Slice) *data.Slice {
	s := NewSlice(in.NComp(), in.Size())
	data.Copy(s, in)
	return s
}
