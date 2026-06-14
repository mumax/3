package cu

// This file implements CUDA driver initialization

//#include <cuda.h>
import "C"

// Initialize the CUDA driver API.
// Currently, flags must be 0.
// If Init() has not been called, any function from the driver API will panic with ERROR_NOT_INITIALIZED.
func Init(flags int) {
	err := Result(C.cuInit(C.uint(flags)))
	if err != SUCCESS {
		panic(err)
	}
}
