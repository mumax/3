package cuda

//#include <cuda_runtime.h>
//#include <cuda.h>
import "C"

import (
	"github.com/barnex/cuda5/cu"
)

// Reset the current GPU device.
func DeviceReset() {
	err := cu.Result(C.cudaDeviceReset())
	if err != cu.SUCCESS {
		panic(err)
	}
}

// Set preference for more cache or shared memory.
func DeviceSetCacheConfig(cacheConfig FuncCache) {
	err := cu.Result(C.cudaDeviceSetCacheConfig(uint32(cacheConfig)))
	if err != cu.SUCCESS {
		panic(err)
	}
}

// Cache preference option.
type FuncCache int

const (
	FUNC_CACHE_PREFER_NONE   FuncCache = C.CU_FUNC_CACHE_PREFER_NONE
	FUNC_CACHE_PREFER_SHARED FuncCache = C.CU_FUNC_CACHE_PREFER_SHARED
	FUNC_CACHE_PREFER_L1     FuncCache = C.CU_FUNC_CACHE_PREFER_L1
	FUNC_CACHE_PREFER_EQUAL  FuncCache = C.CU_FUNC_CACHE_PREFER_EQUAL
)
