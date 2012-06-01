package cu

// This file implements execution of CUDA kernels

//#include <cuda.h>
import "C"

import (
	"unsafe"
)

// Launches a CUDA kernel on the device.
// Example:
//		mod := ModuleLoad("file.ptx")
//		f := mod.GetFunction("test")
//	
//		var arg1 uintptr
//		arg1 = uintptr(someArray)
//	
//		var arg2 float32
//		arg2 = 42
//	
//		var arg3 int
//		arg3 = 1024
//	
//		args := []uintptr{(uintptr)(unsafe.Pointer(&array)), (uintptr)(unsafe.Pointer(&value)), (uintptr)(unsafe.Pointer(&n))}
//		
//		block := 128
//		grid := DivUp(N, block)
//		shmem := 0
//		stream := STREAM0
//		LaunchKernel(f, grid, 1, 1, block, 1, 1, shmem, stream, args)
//
// A more easy-to-use wrapper is implemented in closure.go
//
func LaunchKernel(f Function, gridDimX, gridDimY, gridDimZ int, blockDimX, blockDimY, blockDimZ int, sharedMemBytes int, stream Stream, kernelParams []unsafe.Pointer) {

	//debug: print all arguments
	//argvals := make([]int64, len(kernelParams))
	//for i := range kernelParams {
	//	argvals[i] = *(*int64)(unsafe.Pointer(kernelParams[i]))
	//}
	//fmt.Println("LaunchKernel: ", "func: ", f, "gridDim: ", gridDimX, gridDimY, gridDimZ, "blockDim: ", blockDimX, blockDimY, blockDimZ, "shmem: ", sharedMemBytes, "stream: ", stream, "argptrs: ", kernelParams, "argvals:", argvals)

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
