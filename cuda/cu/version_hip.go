//go:build hip

package cu

// This file implements HIP driver version management

//#include <hip/hip_runtime.h>
import "C"

const HIP_VERSION = C.HIP_VERSION

// Returns the HIP driver version.
func Version() int {
	var version C.int
	err := Result(C.hipDriverGetVersion(&version))
	if err != SUCCESS {
		panic(err)
	}
	return int(version)
}
