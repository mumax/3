package cuda

import (
	"code.google.com/p/mx3/data"
	"github.com/barnex/cuda5/cu"
	"log"
	"unsafe"
)

// Wrapper for cu.MemAlloc, fatal exit on out of memory.
func memAlloc(bytes int64) unsafe.Pointer {
	defer func() {
		err := recover()
		if err == cu.ERROR_OUT_OF_MEMORY {
			log.Fatal(err)
		}
		if err != nil {
			panic(err)
		}
	}()
	return unsafe.Pointer(cu.MemAlloc(bytes))
}

// Try to allocate on GPU, spill to unified host if out of GPU memory.
func TryMakeFloats(N int) cu.DevicePtr {
	return cu.DevicePtr(memAlloc(data.SIZEOF_FLOAT32 * int64(N)))
	//var s safe.Float32s
	//ptr := tryMalloc(cu.SIZEOF_FLOAT32 * int64(N))
	//s.UnsafeSet(ptr, N, N)
	//return s
}

// Try to allocate on GPU, spill to unified host if out of GPU memory.
func TryMakeComplexs(N int) cu.DevicePtr {
	return TryMakeFloats(2 * N)
}

//func HostFloats(N int) safe.Float32s {
//	var s safe.Float32s
//	ptr := cu.MemAllocHost(cu.SIZEOF_FLOAT32 * int64(N))
//	s.UnsafeSet(ptr, N, N)
//	return s
//}
//
//func tryMalloc(bytes int64) (ptr unsafe.Pointer) {
//	defer func() {
//		err := recover()
//		if err == cu.ERROR_OUT_OF_MEMORY {
//			MB := bytes / (1024 * 1024)
//			core.Log("out of GPU memory, allocating", MB, "MB on host")
//			ptr = cu.MemAllocHost(bytes)
//			return
//		} else if err != nil {
//			panic(err)
//		}
//	}()
//	ptr = unsafe.Pointer(uintptr(cu.MemAlloc(bytes)))
//	return
//}
