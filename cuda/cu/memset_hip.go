//go:build hip

package cu

// This file implements HIP memset functions.

//#include <hip/hip_runtime.h>
import "C"

import (
	"unsafe"
)

// Sets the first N 32-bit values of dst array to value.
func MemsetD32(deviceptr DevicePtr, value uint32, N int64) {
	err := Result(C.hipMemsetD32(deviceptr.hip(), C.int(value), C.size_t(N)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously sets the first N 32-bit values of dst array to value.
func MemsetD32Async(deviceptr DevicePtr, value uint32, N int64, stream Stream) {
	err := Result(C.hipMemsetD32Async(deviceptr.hip(), C.int(value), C.size_t(N), C.hipStream_t(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Sets the first N 8-bit values of dst array to value.
func MemsetD8(deviceptr DevicePtr, value uint8, N int64) {
	err := Result(C.hipMemsetD8(deviceptr.hip(), C.uchar(value), C.size_t(N)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously sets the first N 8-bit values of dst array to value.
func MemsetD8Async(deviceptr DevicePtr, value uint8, N int64, stream Stream) {
	err := Result(C.hipMemsetD8Async(deviceptr.hip(), C.uchar(value), C.size_t(N), C.hipStream_t(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}
