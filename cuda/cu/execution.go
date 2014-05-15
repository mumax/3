package cu

// This file implements execution of CUDA kernels

//#include <cuda.h>
import "C"

import (
	"unsafe"
)

func LaunchKernel(f Function, gridDimX, gridDimY, gridDimZ int, blockDimX, blockDimY, blockDimZ int, sharedMemBytes int, stream Stream, kernelParams []unsafe.Pointer) {

	err := Result(C.cuLaunchKernel(
		C.CUfunction(unsafe.Pointer(uintptr(f))),
		C.uint(gridDimX),
		C.uint(gridDimY),
		C.uint(gridDimZ),
		C.uint(blockDimX),
		C.uint(blockDimY),
		C.uint(blockDimZ),
		C.uint(sharedMemBytes),
		C.CUstream(unsafe.Pointer(uintptr(stream))),
		(*unsafe.Pointer)(&kernelParams[0]),
		(*unsafe.Pointer)(unsafe.Pointer(uintptr(0)))))
	if err != SUCCESS {
		panic(err)
	}
}

//func (f Function) Launch(gridDimX, gridDimY, gridDimZ int, blockDimX, blockDimY, blockDimZ int, sharedMemBytes int, stream Stream, kernelParams ...interface{}) {
//	var argPtr [32]unsafe.Pointer
//
//	for i, arg := range kernelParams {
//		argPtr[i] = unsafe.Pointer(reflect.ValueOf(arg).UnsafeAddr())
//	}
//
//	LaunchKernel(f,
//		gridDimX, gridDimY, gridDimZ,
//		blockDimX, blockDimY, blockDimZ,
//		sharedMemBytes, stream,
//		argPtr[:len(kernelParams)])
//}
