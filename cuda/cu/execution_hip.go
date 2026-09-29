//go:build hip

package cu

// This file implements execution of HIP kernels via the driver API

//#include <hip/hip_runtime.h>
//#include <stdlib.h>
import "C"

import (
	"unsafe"
)

const pointerSize = 8 // sorry, 64 bits only.

func LaunchKernel(f Function, gridDimX, gridDimY, gridDimZ int, blockDimX, blockDimY, blockDimZ int, sharedMemBytes int, stream Stream, kernelParams []unsafe.Pointer) {

	// Since Go 1.6, a cgo argument cannot have a Go pointer to Go pointer,
	// so we copy the argument values to C memory first.
	argv := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	argp := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	defer C.free(argv)
	defer C.free(argp)
	for i := range kernelParams {
		*((*unsafe.Pointer)(offset(argp, i))) = offset(argv, i)       // argp[i] = &argv[i]
		*((*uint64)(offset(argv, i))) = *((*uint64)(kernelParams[i])) // argv[i] = *kernelParams[i]
	}

	err := Result(C.hipModuleLaunchKernel(
		C.hipFunction_t(unsafe.Pointer(uintptr(f))),
		C.uint(gridDimX),
		C.uint(gridDimY),
		C.uint(gridDimZ),
		C.uint(blockDimX),
		C.uint(blockDimY),
		C.uint(blockDimZ),
		C.uint(sharedMemBytes),
		C.hipStream_t(unsafe.Pointer(uintptr(stream))),
		(*unsafe.Pointer)(argp),
		(*unsafe.Pointer)(unsafe.Pointer(uintptr(0)))))
	if err != SUCCESS {
		panic(err)
	}
}

func offset(ptr unsafe.Pointer, i int) unsafe.Pointer {
	return unsafe.Pointer(uintptr(ptr) + pointerSize*uintptr(i))
}
