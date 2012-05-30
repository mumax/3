package cu

// This file implements CUDA memset functions.

//#include <cuda.h>
import "C"

import (
	"unsafe"
)

// Sets the first N 32-bit values of dst array to value.
func MemsetD32(deviceptr uintptr, value uint32, N int64) {
	err := Result(C.cuMemsetD32(C.CUdeviceptr(deviceptr), C.uint(value), C.size_t(N)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously sets the first N 32-bit values of dst array to value.
func MemsetD32Async(deviceptr uintptr, value uint32, N int64, stream Stream) {
	err := Result(C.cuMemsetD32Async(C.CUdeviceptr(deviceptr), C.uint(value), C.size_t(N), C.CUstream(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}
