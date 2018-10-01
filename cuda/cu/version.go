package cu

// This file implements CUDA driver version management

//#include <cuda_runtime_api.h>
import "C"

// Returns the CUDA driver version.
func Version() int {
	return int(C.CUDART_VERSION)
}
