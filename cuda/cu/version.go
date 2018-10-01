package cu

// This file implements CUDA driver version management

//#include <cuda.h>
//#include <cuda_runtime_api.h>
import "C"

// Returns the CUDA driver version.
func Version() int {
        var version C.int
        err := Result(C.cuDriverGetVersion(&version))
        if err != SUCCESS {
                panic(err)
        }
        cu_version := Result(C.CUDART_VERSION)
        return int(cu_version)
}

