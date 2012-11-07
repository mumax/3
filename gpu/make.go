package gpu

import (
	"code.google.com/p/nimble-cube/nimble"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"unsafe"
)

func MakeVectors(n int) [3]safe.Float32s {
	return [3]safe.Float32s{MakeFloats(n), MakeFloats(n), MakeFloats(n)}
}

func MakeFloats(N int) safe.Float32s {
	var s safe.Float32s
	ptr := tryMalloc(cu.SIZEOF_FLOAT32 * int64(N))
	s.UnsafeSet(ptr, N, N)
	return s
}

func MakeComplexs(N int) safe.Complex64s {
	return MakeFloats(2 * N).Complex()
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
			nimble.Log("out of GPU memory, allocating", MB, "MB on host")
			ptr = cu.MemAllocHost(bytes)
			return
		} else if err != nil {
			panic(err)
		}
	}()
	ptr = unsafe.Pointer(uintptr(cu.MemAlloc(bytes)))
	return
}
