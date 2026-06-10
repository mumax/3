//go:build hip

package cu

// This file implements HIP driver initialization

//#include <hip/hip_runtime.h>
import "C"

// Initialize the HIP driver API.
// Currently, flags must be 0.
// If Init() has not been called, any function from the driver API will panic with ERROR_NOT_INITIALIZED.
func Init(flags int) {
	err := Result(C.hipInit(C.uint(flags)))
	if err != SUCCESS {
		panic(err)
	}
}
