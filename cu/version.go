package cu

// This file implements CUDA driver version management

//#include <cuda.h>
import "C"

// Returns the CUDA driver version.
func Version() int {
	var version C.int
	err := Result(C.cuDriverGetVersion(&version))
	if err != SUCCESS {
		panic(err)
	}
	return int(version)
}
