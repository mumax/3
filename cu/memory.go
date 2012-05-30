package cu

// This file implements CUDA memory management on the driver level

//#include <cuda.h>
import "C"

import (
	"unsafe"
)

// Allocates a number of bytes of device memory.
func MemAlloc(bytes int64) uintptr {
	var devptr C.CUdeviceptr
	err := Result(C.cuMemAlloc(&devptr, C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
	return uintptr(devptr)
}

// Frees device memory allocated by MemAlloc().
// Overwrites the pointer with NULL.
// It is safe to double-free.
func MemFree(ptr *uintptr) {
	p := *ptr
	if p == 0 {
		return // Allready freed
	}
	*ptr = 0
	err := Result(C.cuMemFree(C.CUdeviceptr(p)))
	if err != SUCCESS {
		panic(err)
	}
}

// Copies a number of bytes on the current device.
// Requires unified addressing to be supported.
// See also: MemcpyDtoD().
// TODO(a): is actually an auto copy for device and/or host memory
func Memcpy(dst, src uintptr, bytes int64) {
	err := Result(C.cuMemcpy(C.CUdeviceptr(dst), C.CUdeviceptr(src), C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously copies a number of bytes on the current device.
func MemcpyAsync(dst, src uintptr, bytes int64, stream Stream) {
	err := Result(C.cuMemcpyAsync(C.CUdeviceptr(dst), C.CUdeviceptr(src), C.size_t(bytes), C.CUstream(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Copies a number of bytes from host to device.
func MemcpyDtoD(dst, src uintptr, bytes int64) {
	err := Result(C.cuMemcpyDtoD(C.CUdeviceptr(dst), C.CUdeviceptr(src), C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously copies a number of bytes from host to device.
func MemcpyDtoDAsync(dst, src uintptr, bytes int64, stream Stream) {
	err := Result(C.cuMemcpyDtoDAsync(C.CUdeviceptr(dst), C.CUdeviceptr(src), C.size_t(bytes), C.CUstream(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Copies a number of bytes from host to device.
func MemcpyHtoD(dst uintptr, src unsafe.Pointer, bytes int64) {
	err := Result(C.cuMemcpyHtoD(C.CUdeviceptr(dst), unsafe.Pointer(src), C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously copies a number of bytes from host to device.
// The host memory must be page-locked (see MemRegister)
func MemcpyHtoDAsync(dst uintptr, src unsafe.Pointer, bytes int64, stream Stream) {
	err := Result(C.cuMemcpyHtoDAsync(C.CUdeviceptr(dst), unsafe.Pointer(src), C.size_t(bytes), C.CUstream(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Copies a number of bytes from device to host.
func MemcpyDtoH(dst unsafe.Pointer, src uintptr, bytes int64) {
	err := Result(C.cuMemcpyDtoH(unsafe.Pointer(dst), C.CUdeviceptr(src), C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously copies a number of bytes device host to host.
// The host memory must be page-locked (see MemRegister)
func MemcpyDtoHAsync(dst unsafe.Pointer, src uintptr, bytes int64, stream Stream) {
	err := Result(C.cuMemcpyDtoHAsync(unsafe.Pointer(dst), C.CUdeviceptr(src), C.size_t(bytes), C.CUstream(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Copies from device memory in one context (device) to another.
func MemcpyPeer(dst uintptr, dstCtx Context, src uintptr, srcCtx Context, bytes int64) {
	err := Result(C.cuMemcpyPeer(C.CUdeviceptr(dst), C.CUcontext(unsafe.Pointer(uintptr(dstCtx))), C.CUdeviceptr(src), C.CUcontext(unsafe.Pointer(uintptr(srcCtx))), C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously copies from device memory in one context (device) to another.
func MemcpyPeerAsync(dst uintptr, dstCtx Context, src uintptr, srcCtx Context, bytes int64, stream Stream) {
	err := Result(C.cuMemcpyPeerAsync(C.CUdeviceptr(dst), C.CUcontext(unsafe.Pointer(uintptr(dstCtx))), C.CUdeviceptr(uintptr(src)), C.CUcontext(unsafe.Pointer(uintptr(srcCtx))), C.size_t(bytes), C.CUstream(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Returns the base address and size of the allocation (by MemAlloc) that contains the input pointer ptr.
func MemGetAddressRange(ptr uintptr) (bytes int64, base uintptr) {
	var cbytes C.size_t
	var cptr C.CUdeviceptr
	err := Result(C.cuMemGetAddressRange(&cptr, &cbytes, C.CUdeviceptr(ptr)))
	if err != SUCCESS {
		panic(err)
	}
	bytes = int64(cbytes)
	base = uintptr(cptr)
	return
}

// Returns the free and total amount of memroy in the current Context (in bytes).
func MemGetInfo() (free, total int64) {
	var cfree, ctotal C.size_t
	err := Result(C.cuMemGetInfo(&cfree, &ctotal))
	if err != SUCCESS {
		panic(err)
	}
	free = int64(cfree)
	total = int64(ctotal)
	return
}

// Page-locks memory specified by the pointer and bytes.
// The pointer and byte size must be aligned to the host page size (4KB)
// See also: MemHostUnregister()
func MemHostRegister(hostptr unsafe.Pointer, bytes int64, flags MemHostRegisterFlag) {
	err := Result(C.cuMemHostRegister(unsafe.Pointer(hostptr), C.size_t(bytes), C.uint(flags)))
	if err != SUCCESS {
		panic(err)
	}
}

// Unmaps memory locked by MemHostRegister().
func MemHostUnregister(hostptr unsafe.Pointer) {
	err := Result(C.cuMemHostUnregister(unsafe.Pointer(hostptr)))
	if err != SUCCESS {
		panic(err)
	}
}

type MemHostRegisterFlag int

// Flag for MemHostRegister
const (
	// Memory is pinned in all CUDA contexts.
	MEMHOSTREGISTER_PORTABLE MemHostRegisterFlag = C.CU_MEMHOSTREGISTER_PORTABLE
	// Maps the allocation in CUDA address space. TODO(a): cuMemHostGetDevicePointer()
	MEMHOSTREGISTER_DEVICEMAP MemHostRegisterFlag = C.CU_MEMHOSTREGISTER_DEVICEMAP
)

const(
	SIZEOF_FLOAT32 = 4
	SIZEOF_FLOAT64 = 8
)
