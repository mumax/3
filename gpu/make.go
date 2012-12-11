package gpu

import (
	"code.google.com/p/mx3/core"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"unsafe"
)

// Try to allocate on GPU, spill to unified host if out of GPU memory.
func TryMakeVectors(n int) [3]safe.Float32s {
	return [3]safe.Float32s{TryMakeFloats(n), TryMakeFloats(n), TryMakeFloats(n)}
}

// Try to allocate on GPU, spill to unified host if out of GPU memory.
func TryMakeFloats(N int) safe.Float32s {
	var s safe.Float32s
	ptr := tryMalloc(cu.SIZEOF_FLOAT32 * int64(N))
	s.UnsafeSet(ptr, N, N)
	return s
}

// Try to allocate on GPU, spill to unified host if out of GPU memory.
func TryMakeComplexs(N int) safe.Complex64s {
	return TryMakeFloats(2 * N).Complex()
}

func HostFloats(N int) safe.Float32s {
	var s safe.Float32s
	ptr := cu.MemAllocHost(cu.SIZEOF_FLOAT32 * int64(N))
	s.UnsafeSet(ptr, N, N)
	return s
}

func tryMalloc(bytes int64) (ptr unsafe.Pointer) {
	defer func() {
		err := recover()
		if err == cu.ERROR_OUT_OF_MEMORY {
			MB := bytes / (1024 * 1024)
			core.Log("out of GPU memory, allocating", MB, "MB on host")
			ptr = cu.MemAllocHost(bytes)
			return
		} else if err != nil {
			panic(err)
		}
	}()
	ptr = unsafe.Pointer(uintptr(cu.MemAlloc(bytes)))
	return
}
